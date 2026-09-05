"""Language-server client (LSP over stdio, Content-Length framing).

One :class:`LspClient` instance per repository, for one sync run:

- the workspace is the repository working tree (checked out at the target
  commit by :mod:`corvidex_mcp.git_manager`);
- when the server reads a workspace config file (vhdl_ls's
  ``vhdl_ls.toml``, Veridian's ``veridian.yaml``), the file must exist
  before the session starts: an existing one is respected and left in
  place, then the repository's config hook (a shell command run at the
  root) is tried and its output is owned by the hook, and only as a
  fallback the client generates a default config itself (removing it on
  shutdown);
- after opening the changed files the client waits for the server to go
  quiet (servers push ``publishDiagnostics`` for the files they analyze
  but may send nothing for clean files, so a per-file wait is not
  possible);
- ``documentSymbol`` results are parsed into a plain
  :class:`SymbolInfo` tree (hierarchical, as advertised during
  ``initialize``).

All failures are contained: a hung or broken language server surfaces as
:class:`LspError` and the chunker falls back to structural parsing.

Server-specific behavior is supplied by subclasses: the extra
command-line arguments (:meth:`LspClient.build_args`), the LSP
``languageId`` of opened documents (:attr:`LspClient.language_id`), the
workspace config handling (:attr:`LspClient.config_name` +
:meth:`LspClient.default_config_text`), and which diagnostics mark a
file as syntactically broken (:meth:`LspClient.is_syntax_error`).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: How long to wait for the server to stop emitting diagnostics.
QUIET_TIMEOUT = 20.0
#: Diagnostics silence that counts as "the analysis is done".
QUIET_WINDOW = 1.5
#: Per-request timeout for ordinary LSP requests.
REQUEST_TIMEOUT = 30.0
#: Timeout for the initialize handshake.
INITIALIZE_TIMEOUT = 60.0
#: Timeout for a repository's workspace-config generation hook.
HOOK_TIMEOUT = 120.0
#: Timeout for a language server's version probe.
VERSION_TIMEOUT = 5.0


class LspError(RuntimeError):
    """The language server failed a request or closed the connection."""


class LspTimeout(LspError):
    """A language-server request did not answer in time."""


@dataclass(frozen=True)
class SymbolInfo:
    """One LSP document symbol with its (0-based, inclusive) line range."""

    name: str
    kind: int
    start_line: int
    end_line: int
    children: tuple[SymbolInfo, ...] = ()


@dataclass(frozen=True)
class DiagnosticInfo:
    """One LSP diagnostic (code such as ``syntax_error``, ``unresolved``)."""

    code: str
    message: str
    severity: int
    start_line: int
    end_line: int


def default_libraries_dir(binary: str) -> Path | None:
    """Locate the ``vhdl_libraries`` directory shipped next to the binary.

    The official distribution layout is ``<root>/bin/vhdl_ls`` plus
    ``<root>/vhdl_libraries``. Returns None when it cannot be found (the
    server then runs without ``-l``).
    """
    bin_path = Path(binary).expanduser()
    for candidate in (
        bin_path.parent / "vhdl_libraries",
        bin_path.parent.parent / "vhdl_libraries",
    ):
        if candidate.is_dir():
            return candidate
    return None


def server_version(binary: str, timeout: float = VERSION_TIMEOUT) -> str | None:
    """The language server's self-reported version, or None.

    Tries ``--version`` and then ``-V``; the first non-empty line of the
    output is returned. Never raises: a missing or broken binary simply
    yields None (the caller records the error instead).
    """
    for flag in ("--version", "-V"):
        try:
            proc = subprocess.run(
                [str(binary), flag],
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        output = (proc.stdout or proc.stderr).decode("utf-8", "replace").strip()
        if output:
            return output.splitlines()[0].strip()
    return None


def _parse_symbols(items: Any) -> tuple[SymbolInfo, ...]:
    if not isinstance(items, list):
        return ()
    out: list[SymbolInfo] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rng = item.get("range")
        if not isinstance(rng, dict):
            # Some servers report "location" instead of "range".
            loc = item.get("location")
            if isinstance(loc, dict) and isinstance(loc.get("range"), dict):
                rng = loc["range"]
        if not isinstance(rng, dict):
            continue
        try:
            out.append(
                SymbolInfo(
                    name=str(item.get("name", "")),
                    kind=int(item.get("kind", 0)),
                    start_line=int(rng["start"]["line"]),
                    end_line=int(rng["end"]["line"]),
                    children=_parse_symbols(item.get("children")),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(out)


def path_to_uri(path: Path) -> str:
    """The ``file://`` URI for ``path`` (one stable URI per path).

    Absolute paths convert directly; paths ``as_uri`` cannot express
    (relative, or rooted without a drive on Windows) are resolved
    first. Both the open requests and the diagnostics lookups go
    through this helper, so a lookup always reproduces exactly the
    URI the server echoed.
    """
    try:
        return path.as_uri()
    except ValueError:
        return path.resolve().as_uri()


class LspClient:
    """Async client for one language-server process rooted at one workspace."""

    #: LSP ``languageId`` for ``textDocument/didOpen``.
    language_id: str = "vhdl"
    #: Workspace config file the server reads (None: no config handling).
    config_name: str | None = None

    def __init__(
        self,
        binary: str,
        workspace: Path,
        *,
        config_hook: str | None = None,
    ) -> None:
        self._binary = binary
        self._workspace = workspace
        self._hook = config_hook
        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self._stderr_reader: asyncio.Task[None] | None = None
        self._write_lock = asyncio.Lock()
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._next_id = 0
        self._diagnostics: dict[str, list[DiagnosticInfo]] = {}
        self._quiet_event = asyncio.Event()
        self._supports_document_symbol = False
        self._owns_workspace_config = False
        self._stream_closed = False

    # -- server-specific surface ------------------------------------------

    def build_args(self) -> list[str]:
        """Extra command-line arguments (after the binary)."""
        return []

    def default_config_text(self) -> str | None:
        """Built-in workspace config written when none exists (None: skip)."""
        return None

    def is_syntax_error(self, diagnostic: DiagnosticInfo) -> bool:
        """Whether one diagnostic marks its file as syntactically broken."""
        return diagnostic.severity == 1

    def has_syntax_error(self, path: Path) -> bool:
        return any(self.is_syntax_error(d) for d in self.diagnostics_for(path))

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Spawn the server, perform the LSP handshake, and open the workspace."""
        await self._ensure_workspace_config()
        args = [str(self._binary), *self.build_args()]
        self._proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(self._workspace),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._reader = asyncio.create_task(self._read_loop())
        self._stderr_reader = asyncio.create_task(self._drain_stderr())
        try:
            root_uri = path_to_uri(self._workspace)
            result = await self._request(
                "initialize",
                {
                    "processId": None,
                    "rootUri": root_uri,
                    "capabilities": {
                        "textDocument": {
                            "documentSymbol": {
                                "hierarchicalDocumentSymbolSupport": True
                            }
                        }
                    },
                    "workspaceFolders": [
                        {
                            "uri": root_uri,
                            "name": self._workspace.name,
                        }
                    ],
                },
                timeout=INITIALIZE_TIMEOUT,
            )
            caps = result.get("capabilities") if isinstance(result, dict) else None
            self._supports_document_symbol = bool(
                (caps or {}).get("documentSymbolProvider")
            )
            await self._notify("initialized", {})
            logger.info(
                "language server %s started for %s", self._binary, self._workspace
            )
        except BaseException:
            await self.shutdown()
            raise

    async def shutdown(self) -> None:
        """Best-effort graceful shutdown; never raises."""
        if self._proc is not None:
            with contextlib.suppress(Exception):
                await self._request("shutdown", None, timeout=10.0)
                await self._notify("exit", {})
            # Give the server a moment to exit cleanly after "exit".
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._proc.wait(), 5.0)
            if self._proc.returncode is None:
                self._proc.terminate()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(self._proc.wait(), 5.0)
            if self._proc.returncode is None:
                self._proc.kill()
                with contextlib.suppress(Exception):
                    await self._proc.wait()
        if self._reader is not None:
            self._reader.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await self._reader
            self._reader = None
        if self._stderr_reader is not None:
            self._stderr_reader.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await self._stderr_reader
            self._stderr_reader = None
        self._fail_pending(LspError("language server shut down"))
        if self._owns_workspace_config and self.config_name is not None:
            self._workspace.joinpath(self.config_name).unlink(missing_ok=True)

    async def _ensure_workspace_config(self) -> None:
        if self.config_name is None:
            return
        config_path = self._workspace / self.config_name
        if config_path.exists():
            logger.info(
                "using repository-provided %s in %s",
                self.config_name,
                self._workspace,
            )
            return
        if self._hook is not None:
            if await self._run_hook():
                logger.info("config hook generated %s", config_path)
                return
            logger.warning(
                "config hook completed but %s is still missing; "
                "generating the default config",
                config_path.name,
            )
        text = self.default_config_text()
        if text is None:
            return
        self._owns_workspace_config = True
        config_path.write_text(text, encoding="utf-8")

    async def _run_hook(self) -> bool:
        """Run the repository's config hook at the workspace root.

        The hook is a shell command that must leave the workspace config
        file at the repository root. Returns True when it exits
        successfully and the file now exists. Its output is owned by the
        hook, so the server never removes it.
        """
        assert self._hook is not None and self.config_name is not None
        if sys.platform == "win32":
            shell_args = ["cmd", "/c", self._hook]
        else:
            shell_args = ["sh", "-c", self._hook]
        proc = await asyncio.create_subprocess_exec(
            *shell_args,
            cwd=str(self._workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _out, err = await asyncio.wait_for(proc.communicate(), HOOK_TIMEOUT)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            logger.warning(
                "config hook timed out after %.0fs in %s",
                HOOK_TIMEOUT,
                self._workspace,
            )
            return False
        if proc.returncode != 0:
            logger.warning(
                "config hook failed (exit %s) in %s: %s",
                proc.returncode,
                self._workspace,
                err.decode(errors="replace").strip()[-500:],
            )
            return False
        return self._workspace.joinpath(self.config_name).exists()

    # -- document flow --------------------------------------------------------

    async def open_document(self, path: Path, text: str | None = None) -> None:
        """didOpen one file (content is read from the working tree, or
        ``text`` is used verbatim when the caller already has it).

        Never raises: a server that has died mid-session (some servers
        crash on malformed files) simply loses this file — its symbols
        come back empty and the chunker falls back to structural
        parsing.
        """
        uri = path_to_uri(path)
        if text is None:
            text = path.read_text(encoding="utf-8", errors="replace")
        try:
            await self._notify(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": uri,
                        "languageId": self.language_id,
                        "version": 1,
                        "text": text,
                    }
                },
            )
        except LspError as exc:
            logger.warning("could not open %s (server unavailable): %s", path, exc)

    async def wait_until_quiet(self, timeout: float = QUIET_TIMEOUT) -> None:
        """Wait until the server stops emitting diagnostics.

        The server pushes ``publishDiagnostics`` for the files it
        analyzes (including files we did not open) and emits nothing for
        clean files, so the analysis is "done" when no new diagnostics
        have arrived for ``QUIET_WINDOW`` seconds (bounded by
        ``timeout``).
        """
        if self._stream_closed:
            logger.warning(
                "language server gone before analysis in %s", self._workspace
            )
            return
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                logger.warning(
                    "language server diagnostics not quiet after %.0fs in %s",
                    timeout,
                    self._workspace,
                )
                return
            self._quiet_event.clear()
            try:
                await asyncio.wait_for(
                    self._quiet_event.wait(), min(QUIET_WINDOW, remaining)
                )
                # New diagnostics arrived: wait for the silence again.
            except TimeoutError:
                return

    def diagnostics_for(self, path: Path) -> tuple[DiagnosticInfo, ...]:
        """Diagnostics collected so far for one file (empty when clean)."""
        return tuple(self._diagnostics.get(path_to_uri(path), ()))

    @property
    def supports_document_symbol(self) -> bool:
        return self._supports_document_symbol

    @property
    def server_alive(self) -> bool:
        """False once the server's stdout stream has closed."""
        return not self._stream_closed

    async def document_symbols(self, path: Path) -> tuple[SymbolInfo, ...]:
        """Hierarchical document symbols for one file; () when unavailable."""
        if not self._supports_document_symbol:
            return ()
        try:
            result = await self._request(
                "textDocument/documentSymbol",
                {"textDocument": {"uri": path_to_uri(path)}},
            )
        except LspError as exc:
            logger.warning("documentSymbol failed for %s: %s", path, exc)
            return ()
        return _parse_symbols(result)

    # -- transport --------------------------------------------------------------

    async def _request(
        self, method: str, params: Any, timeout: float = REQUEST_TIMEOUT
    ) -> Any:
        assert self._proc is not None and self._proc.stdin is not None
        # Id allocation is atomic (single-threaded asyncio, no ``await``
        # between the increment and the ``_pending`` registration) and
        # ``_send`` serializes the actual frame writes through
        # ``_write_lock``, so several requests may be in flight at once;
        # responses are matched back to their future by id in
        # ``_dispatch`` regardless of the order they arrive in.
        self._next_id += 1
        request_id = self._next_id
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params,
                }
            )
        except LspError as exc:
            self._pending.pop(request_id, None)
            future.cancel()
            raise exc
        try:
            return await asyncio.wait_for(future, timeout)
        except TimeoutError:
            self._pending.pop(request_id, None)
            future.cancel()
            raise LspTimeout(f"{method} timed out after {timeout:.0f}s") from None

    async def _notify(self, method: str, params: Any) -> None:
        assert self._proc is not None
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def _send(self, obj: dict[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        body = json.dumps(obj).encode("utf-8")
        frame = (
            b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n\r\n" + body
        )
        try:
            async with self._write_lock:
                self._proc.stdin.write(frame)
                await self._proc.stdin.drain()
        except Exception as exc:
            raise LspError(f"failed to send to language server: {exc}") from exc

    async def _drain_stderr(self) -> None:
        """Log the server's stderr at DEBUG so its pipe never backs up.

        Some servers are chatty on stderr; nothing else reads that
        pipe, so an unread server would eventually block on a full
        64 KiB buffer and every request would then look like a hung
        server (``wait_until_quiet``/request timeouts) with no clue as
        to why.
        """
        assert self._proc is not None and self._proc.stderr is not None
        stream = self._proc.stderr
        name = Path(self._binary).name
        while True:
            line = await stream.readline()
            if not line:
                return
            logger.debug("%s: %s", name, line.decode(errors="replace").rstrip())

    async def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        stream = self._proc.stdout
        try:
            while True:
                try:
                    header = await stream.readuntil(b"\r\n\r\n")
                except (asyncio.IncompleteReadError, asyncio.LimitOverrunError):
                    break
                length = _parse_content_length(header)
                if length is None:
                    continue
                try:
                    body = await stream.readexactly(length)
                except (asyncio.IncompleteReadError, asyncio.LimitOverrunError):
                    break
                try:
                    message = json.loads(body)
                except json.JSONDecodeError:
                    logger.warning("ignoring malformed LSP message")
                    continue
                self._dispatch(message)
        finally:
            self._stream_closed = True
            self._fail_pending(LspError("language server connection closed"))

    def _dispatch(self, message: Any) -> None:
        if not isinstance(message, dict):
            return
        if "id" in message and ("result" in message or "error" in message):
            request_id = message["id"]
            future = self._pending.pop(request_id, None)
            if future is None or future.done():
                return
            if "error" in message:
                error = message["error"]
                detail = error.get("message") if isinstance(error, dict) else str(error)
                future.set_exception(LspError(f"LSP error: {detail}"))
            else:
                future.set_result(message["result"])
        elif "method" in message:
            self._handle_notification(message["method"], message.get("params"))

    def _handle_notification(self, method: str, params: Any) -> None:
        if method != "textDocument/publishDiagnostics" or not isinstance(params, dict):
            return
        uri = str(params.get("uri", ""))
        diagnostics: list[DiagnosticInfo] = []
        for item in params.get("diagnostics") or []:
            if not isinstance(item, dict):
                continue
            rng = item.get("range")
            if not isinstance(rng, dict):
                continue
            try:
                diagnostics.append(
                    DiagnosticInfo(
                        code=str(item.get("code") or ""),
                        message=str(item.get("message") or ""),
                        severity=int(item.get("severity", 1)),
                        start_line=int(rng["start"]["line"]),
                        end_line=int(rng["end"]["line"]),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        self._diagnostics[uri] = diagnostics
        self._quiet_event.set()

    def _fail_pending(self, error: LspError) -> None:
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()


def _parse_content_length(header: bytes) -> int | None:
    for line in header.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            with contextlib.suppress(ValueError):
                return int(line.split(b":", 1)[1].strip())
    return None


class VhdlLsp(LspClient):
    """vhdl_ls client: ``-l <libraries> --silent`` + a ``vhdl_ls.toml``.

    vhdl_ls reads its workspace/library configuration from a
    ``vhdl_ls.toml`` in the repository root; the official distribution
    ships an ``vhdl_libraries`` directory next to the binary.
    """

    language_id = "vhdl"
    config_name = "vhdl_ls.toml"

    _DEFAULTLIB_GLOBS = ("**/*.vhd", "**/*.vhdl")

    def __init__(
        self,
        binary: str,
        workspace: Path,
        libraries_dir: Path | None = None,
        vhdl_ls_hook: str | None = None,
        files: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(binary, workspace, config_hook=vhdl_ls_hook)
        self._libraries = libraries_dir
        #: Relative paths (as returned by the indexing pipeline, already
        #: filtered through the repository's ``exclude`` patterns) used to
        #: build the generated ``defaultlib`` when no ``vhdl_ls.toml``
        #: exists. ``None`` falls back to the old ``**/*.vhd`` workspace
        #: glob (used by callers that just want a generic single-file or
        #: ad-hoc workspace, e.g. tests).
        #:
        #: A blanket glob is unsafe against a real repository checkout: it
        #: also matches gitignored build-output directories whose names
        #: happen to end in ``.vhd`` (e.g. GHDL's per-run library cache
        #: under ``vunit_out/``, which are actually directories, not
        #: files), and it dumps unrelated vendored/submodule trees into a
        #: single ``defaultlib`` library, causing duplicate-declaration
        #: errors. Restricting the default config to exactly the files
        #: the pipeline already resolved for this repository avoids both.
        self._files = files

    def build_args(self) -> list[str]:
        args: list[str] = []
        if self._libraries is not None and self._libraries.is_dir():
            args += ["-l", str(self._libraries)]
        args.append("--silent")
        return args

    def is_syntax_error(self, diagnostic: DiagnosticInfo) -> bool:
        return diagnostic.code == "syntax_error"

    def default_config_text(self) -> str | None:
        entries = self._files if self._files is not None else self._DEFAULTLIB_GLOBS
        files_list = ", ".join(f"'{entry}'" for entry in entries)
        lines = ["[libraries.defaultlib]", f"files = [{files_list}]", ""]
        if self._libraries is not None and self._libraries.is_dir():
            lib = str(self._libraries)
            ieee_files = ", ".join(
                f"'{lib}/{name}/*.vhdl'"
                for name in ("ieee2008", "synopsys", "vital2000")
            )
            lines += [
                "[libraries.std]",
                f"files = ['{lib}/std/*.vhd']",
                "is_third_party = true",
                "",
                "[libraries.ieee]",
                f"files = [{ieee_files}]",
                "is_third_party = true",
            ]
        return "\n".join(lines)
