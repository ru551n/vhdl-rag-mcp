"""Tests for the vhdl_ls LSP client.

Most tests run fully offline against a fake LSP server (a small Python
script speaking the same Content-Length framing), so the transport,
handshake, diagnostics handling, and symbol parsing are verified without
downloading anything. One test runs against a real vhdl_ls binary when
one is available (``VHDL_LS_TEST_BIN`` or ``vhdl_ls`` on PATH) and is
skipped otherwise.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest
from fake_lsp_util import executable_lsp_script

from corvidex_mcp.lsp import (
    DiagnosticInfo,
    LspError,
    SymbolInfo,
    VhdlLsp,
    default_libraries_dir,
    path_to_uri,
)

FAKE_SERVER = r"""#!/usr/bin/env python3
import json
import sys


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            break
        key, _, value = line.partition(b":")
        headers[key.strip().lower()] = value.strip()
    length = int(headers.get(b"content-length", b"0"))
    return json.loads(sys.stdin.buffer.read(length))


def send(obj):
    body = json.dumps(obj).encode()
    frame = b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n"
    sys.stdout.buffer.write(frame + body)
    sys.stdout.buffer.flush()


def symbol_range(start_line, end_line):
    return {
        "start": {"line": start_line, "character": 0},
        "end": {"line": end_line, "character": 3},
    }


def symbols():
    return [
        {
            "name": "fifo",
            "kind": 2,
            "range": symbol_range(0, 9),
            "children": [
                {
                    "name": "rtl",
                    "kind": 2,
                    "range": symbol_range(2, 8),
                    "children": [
                        {
                            "name": "p_write",
                            "kind": 3,
                            "range": symbol_range(3, 6),
                            "children": [],
                        }
                    ],
                }
            ],
        }
    ]


def diagnostic(code, message):
    return {
        "code": code,
        "message": message,
        "severity": 1,
        "range": symbol_range(0, 0),
    }


read_message()  # the initialize request
send(
    {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"capabilities": {"documentSymbolProvider": True}},
    }
)
msg = read_message()  # initialized
assert msg is not None and msg.get("method") == "initialized", msg
while True:
    msg = read_message()
    if msg is None:
        break
    method = msg.get("method")
    if method == "textDocument/didOpen":
        uri = msg["params"]["textDocument"]["uri"]
        diags = [diagnostic("syntax_error", "syntax error")] if "badfile" in uri else []
        send(
            {
                "jsonrpc": "2.0",
                "method": "textDocument/publishDiagnostics",
                "params": {"uri": uri, "diagnostics": diags},
            }
        )
    elif method == "textDocument/documentSymbol":
        uri = msg["params"]["textDocument"]["uri"]
        send(
            {
                "jsonrpc": "2.0",
                "id": msg["id"],
                "result": symbols() if "badfile" not in uri else None,
            }
        )
    elif method == "shutdown":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": None})
    elif method == "exit":
        sys.exit(0)
"""


@pytest.fixture
def fake_server(tmp_path: Path) -> Path:
    return executable_lsp_script(tmp_path, "fake_lsp.py", FAKE_SERVER)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "good.vhd").write_text("entity good is end;\n")
    (ws / "badfile.vhd").write_text("entity broken is end broken;\n")
    return ws


# -- pure parsing ----------------------------------------------------------


def test_parse_symbols_nested():
    from corvidex_mcp.lsp.client import _parse_symbols

    out = _parse_symbols(
        [
            {
                "name": "ent",
                "kind": 2,
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 9, "character": 3},
                },
                "children": [
                    {
                        "name": "arch",
                        "kind": 2,
                        "range": {
                            "start": {"line": 1, "character": 0},
                            "end": {"line": 8, "character": 3},
                        },
                        "children": [],
                    }
                ],
            },
            "garbage",
            {"name": "no-range", "kind": 2},
            {
                "name": "loc-based",
                "kind": 3,
                "location": {
                    "range": {
                        "start": {"line": 5, "character": 0},
                        "end": {"line": 7, "character": 0},
                    }
                },
            },
        ]
    )
    # "garbage" and the range-less entry are skipped.
    assert len(out) == 2
    assert out[0].name == "ent"
    assert out[0].kind == 2
    assert (out[0].start_line, out[0].end_line) == (0, 9)
    assert len(out[0].children) == 1
    assert out[0].children[0].name == "arch"
    assert out[1].name == "loc-based"
    assert (out[1].start_line, out[1].end_line) == (5, 7)


def test_parse_symbols_empty():
    from corvidex_mcp.lsp.client import _parse_symbols

    assert _parse_symbols(None) == ()
    assert _parse_symbols([]) == ()


def test_parse_content_length():
    from corvidex_mcp.lsp.client import _parse_content_length

    assert _parse_content_length(b"Content-Length: 42\r\n") == 42
    assert _parse_content_length(b"CONTENT-LENGTH: 7\r\nOther: x\r\n") == 7
    assert _parse_content_length(b"X: 1\r\n") is None
    assert _parse_content_length(b"Content-Length: abc\r\n") is None


def test_default_config_text_uses_glob_when_no_files_given():
    lsp = VhdlLsp("vhdl_ls", Path("/tmp"))
    text = lsp.default_config_text()
    assert text is not None
    assert "'**/*.vhd'" in text
    assert "'**/*.vhdl'" in text


def test_default_config_text_uses_explicit_files_list_when_given():
    """The pipeline passes the exact set of files it already resolved
    (post exclude-filtering) instead of letting vhdl_ls glob the whole
    workspace. A blanket '**/*.vhd' glob matches gitignored build-output
    directories whose names happen to end in '.vhd' (e.g. GHDL's
    per-run library cache under vunit_out/, which are directories, not
    files) and dumps unrelated vendored/submodule trees into a single
    defaultlib, causing duplicate-declaration errors."""
    lsp = VhdlLsp(
        "vhdl_ls", Path("/tmp"), files=("modules/foo/src/foo.vhd", "src/bar.vhd")
    )
    text = lsp.default_config_text()
    assert text is not None
    assert "'modules/foo/src/foo.vhd'" in text
    assert "'src/bar.vhd'" in text
    assert "**/*.vhd" not in text


def test_default_config_text_explicit_empty_files_list():
    lsp = VhdlLsp("vhdl_ls", Path("/tmp"), files=())
    text = lsp.default_config_text()
    assert text is not None
    assert "[libraries.defaultlib]" in text
    assert "files = []" in text


def test_default_libraries_dir(tmp_path: Path):
    root = tmp_path / "dist"
    (root / "bin").mkdir(parents=True)
    (root / "bin" / "vhdl_ls").write_text("x")
    (root / "vhdl_libraries").mkdir()
    assert (
        default_libraries_dir(str(root / "bin" / "vhdl_ls")) == root / "vhdl_libraries"
    )
    assert default_libraries_dir("/nonexistent/vhdl_ls") is None


# -- transport against the fake server ---------------------------------------


async def test_handshake_open_symbols_and_shutdown(fake_server: Path, workspace: Path):
    lsp = VhdlLsp(str(fake_server), workspace)
    try:
        await lsp.start()
        assert lsp.supports_document_symbol
        # The client generates a workspace config for the repository.
        assert (workspace / "vhdl_ls.toml").exists()
        toml = (workspace / "vhdl_ls.toml").read_text()
        assert "[libraries.defaultlib]" in toml

        good = workspace / "good.vhd"
        bad = workspace / "badfile.vhd"
        await lsp.open_document(good)
        await lsp.open_document(bad)
        await lsp.wait_until_quiet(timeout=10.0)

        assert not lsp.has_syntax_error(good)
        assert lsp.has_syntax_error(bad)
        diags = lsp.diagnostics_for(bad)
        assert len(diags) == 1
        assert diags[0].code == "syntax_error"
        assert diags[0].severity == 1

        syms = await lsp.document_symbols(good)
        assert len(syms) == 1
        assert syms[0].name == "fifo"
        assert syms[0].kind == 2
        assert syms[0].children[0].name == "rtl"
        assert syms[0].children[0].children[0].name == "p_write"
        assert isinstance(syms[0], SymbolInfo)

        # A file with no symbols yields an empty tuple.
        assert await lsp.document_symbols(bad) == ()
    finally:
        await lsp.shutdown()
    assert lsp._proc.returncode == 0
    # The generated config is removed on shutdown (the client owned it).
    assert not (workspace / "vhdl_ls.toml").exists()


async def test_open_document_uses_given_text_without_disk_read(tmp_path: Path):
    """``text=`` is used verbatim, skipping the disk read entirely."""
    lsp = VhdlLsp("unused-binary", tmp_path)
    captured: dict[str, object] = {}

    async def fake_notify(method: str, params: dict[str, object]) -> None:
        captured["method"] = method
        captured["params"] = params

    lsp._notify = fake_notify  # type: ignore[method-assign]
    missing = tmp_path / "does-not-exist.vhd"  # never written to disk
    await lsp.open_document(missing, text="entity given is end;\n")
    assert captured["method"] == "textDocument/didOpen"
    params = captured["params"]
    assert isinstance(params, dict)
    assert params["textDocument"]["text"] == "entity given is end;\n"  # type: ignore[index]


async def test_repository_config_respected(fake_server: Path, workspace: Path):
    (workspace / "vhdl_ls.toml").write_text("[libraries.defaultlib]\nfiles = ['x']\n")
    lsp = VhdlLsp(str(fake_server), workspace)
    try:
        await lsp.start()
        assert (
            workspace / "vhdl_ls.toml"
        ).read_text() == "[libraries.defaultlib]\nfiles = ['x']\n"
    finally:
        await lsp.shutdown()
    # A repository-provided config must be left in place.
    assert (workspace / "vhdl_ls.toml").exists()


# -- vhdl_ls_hook ------------------------------------------------------------


HOOK_GENERATOR = (
    "import pathlib\n"
    'pathlib.Path("vhdl_ls.toml").write_text(\n'
    "    \"[libraries.defaultlib]\\nfiles = ['hook']\\n\"\n"
    ")\n"
)


def hook_command(tmp_path: Path) -> str:
    script = tmp_path / "hook.py"
    script.write_text(HOOK_GENERATOR, encoding="utf-8")
    return f"{sys.executable} {script}"


async def test_hook_generates_config_when_missing(
    fake_server: Path, workspace: Path, tmp_path: Path
) -> None:
    lsp = VhdlLsp(str(fake_server), workspace, vhdl_ls_hook=hook_command(tmp_path))
    try:
        await lsp.start()
        assert (workspace / "vhdl_ls.toml").read_text() == (
            "[libraries.defaultlib]\nfiles = ['hook']\n"
        )
    finally:
        await lsp.shutdown()
    # Hook output is owned by the hook: the server never removes it.
    assert (workspace / "vhdl_ls.toml").exists()


async def test_hook_failure_falls_back_to_default(
    fake_server: Path, workspace: Path
) -> None:
    lsp = VhdlLsp(str(fake_server), workspace, vhdl_ls_hook="exit 1")
    try:
        await lsp.start()
        assert "[libraries.defaultlib]" in (workspace / "vhdl_ls.toml").read_text()
    finally:
        await lsp.shutdown()
    # The fallback config is server-generated and cleaned up on shutdown.
    assert not (workspace / "vhdl_ls.toml").exists()


async def test_hook_not_called_when_config_present(
    fake_server: Path, workspace: Path, tmp_path: Path
) -> None:
    marker = tmp_path / "hook-ran"
    script = tmp_path / "marker_hook.py"
    script.write_text(
        f"import pathlib\npathlib.Path({str(marker)!r}).touch()\n",
        encoding="utf-8",
    )
    (workspace / "vhdl_ls.toml").write_text("[libraries.defaultlib]\n")
    lsp = VhdlLsp(
        str(fake_server), workspace, vhdl_ls_hook=f"{sys.executable} {script}"
    )
    try:
        await lsp.start()
    finally:
        await lsp.shutdown()
    assert not marker.exists()
    assert (workspace / "vhdl_ls.toml").read_text() == "[libraries.defaultlib]\n"


async def test_wait_until_quiet_is_bounded(fake_server: Path, workspace: Path):
    lsp = VhdlLsp(str(fake_server), workspace)
    try:
        await lsp.start()
        # No documents opened: nothing to wait for, but the call must
        # return within roughly the timeout.
        loop = asyncio.get_running_loop()
        start = loop.time()
        await lsp.wait_until_quiet(timeout=0.3)
        assert loop.time() - start < 2.0
    finally:
        await lsp.shutdown()


async def test_shutdown_without_start(fake_server: Path, workspace: Path):
    lsp = VhdlLsp(str(fake_server), workspace)
    await lsp.shutdown()  # must not raise


async def test_request_to_dead_server_raises(fake_server: Path, workspace: Path):
    lsp = VhdlLsp(str(fake_server), workspace)
    await lsp.start()
    await lsp.shutdown()
    # Low-level requests surface the failure...
    with pytest.raises(LspError):
        await lsp._request("textDocument/documentSymbol", {})
    # ...while the high-level API degrades gracefully to an empty result.
    assert await lsp.document_symbols(workspace / "good.vhd") == ()


# -- concurrent requests ------------------------------------------------------

# Buffers two ``test/echo`` requests and replies to the second one first,
# proving that responses are matched back to their request by id rather
# than by the order in which they were sent or answered.
FAKE_SERVER_OUT_OF_ORDER = r"""#!/usr/bin/env python3
import json
import sys


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            break
        key, _, value = line.partition(b":")
        headers[key.strip().lower()] = value.strip()
    length = int(headers.get(b"content-length", b"0"))
    return json.loads(sys.stdin.buffer.read(length))


def send(obj):
    body = json.dumps(obj).encode()
    frame = b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n"
    sys.stdout.buffer.write(frame + body)
    sys.stdout.buffer.flush()


read_message()  # the initialize request
send({"jsonrpc": "2.0", "id": 1, "result": {"capabilities": {}}})
msg = read_message()  # initialized
assert msg is not None and msg.get("method") == "initialized", msg

pending = []
while True:
    msg = read_message()
    if msg is None:
        break
    method = msg.get("method")
    if method == "test/echo":
        pending.append(msg)
        if len(pending) == 2:
            for request in reversed(pending):
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": request["id"],
                        "result": request["params"],
                    }
                )
            pending = []
    elif method == "shutdown":
        send({"jsonrpc": "2.0", "id": msg["id"], "result": None})
    elif method == "exit":
        sys.exit(0)
"""


async def test_concurrent_requests_are_in_flight_and_matched_by_id(
    tmp_path: Path, workspace: Path
) -> None:
    server = executable_lsp_script(
        tmp_path, "fake_lsp_out_of_order.py", FAKE_SERVER_OUT_OF_ORDER
    )
    lsp = VhdlLsp(str(server), workspace)
    try:
        await lsp.start()
        # Both requests must be sent before either response arrives: the
        # fake server only replies once it has received both, buffering
        # them and replying to the second one first. With a lock around
        # the whole request/response cycle this would deadlock (the
        # second request would never be sent until the first's future
        # resolves, which never happens).
        first, second = await asyncio.wait_for(
            asyncio.gather(
                lsp._request("test/echo", {"tag": "first"}),
                lsp._request("test/echo", {"tag": "second"}),
            ),
            timeout=5.0,
        )
        assert first == {"tag": "first"}
        assert second == {"tag": "second"}
    finally:
        await lsp.shutdown()


# -- dispatch unit tests -----------------------------------------------------


def test_dispatch_response_completes_pending():
    lsp = VhdlLsp("vhdl_ls", Path("/tmp"))
    loop = asyncio.new_event_loop()
    try:
        future = loop.create_future()
        lsp._pending[7] = future
        lsp._dispatch({"jsonrpc": "2.0", "id": 7, "result": {"ok": True}})
        assert future.result() == {"ok": True}
        # Error responses raise through the future.
        future2 = loop.create_future()
        lsp._pending[8] = future2
        lsp._dispatch(
            {"jsonrpc": "2.0", "id": 8, "error": {"code": -32601, "message": "boom"}}
        )
        with pytest.raises(LspError, match="boom"):
            future2.result()
    finally:
        loop.close()


def test_dispatch_diagnostics_notification(tmp_path: Path):
    lsp = VhdlLsp("vhdl_ls", tmp_path)
    target = tmp_path / "x.vhd"
    lsp._dispatch(
        {
            "jsonrpc": "2.0",
            "method": "textDocument/publishDiagnostics",
            "params": {
                "uri": path_to_uri(target),
                "diagnostics": [
                    {
                        "code": "unresolved",
                        "message": "could not resolve",
                        "severity": 2,
                        "range": {
                            "start": {"line": 3, "character": 1},
                            "end": {"line": 3, "character": 9},
                        },
                    },
                    "not-a-dict",
                ],
            },
        }
    )
    diags = lsp.diagnostics_for(target)
    assert len(diags) == 1
    assert diags[0] == DiagnosticInfo(
        code="unresolved",
        message="could not resolve",
        severity=2,
        start_line=3,
        end_line=3,
    )
    # Other notifications are ignored.
    lsp._dispatch({"jsonrpc": "2.0", "method": "window/logMessage", "params": {}})


# -- real binary (skipped when unavailable) ----------------------------------


REAL_BIN = os.environ.get("VHDL_LS_TEST_BIN")
REAL_LIBS = os.environ.get("VHDL_LS_TEST_LIBRARIES_DIR")


def _find_real_binary() -> str | None:
    if REAL_BIN and Path(REAL_BIN).exists():
        return REAL_BIN
    import shutil

    return shutil.which("vhdl_ls")


def _find_real_libraries_dir(binary: str) -> Path | None:
    """The 'vhdl_libraries' dir for the real-binary tests.

    ``default_libraries_dir`` only finds anything for the official
    release layout (``<root>/bin/vhdl_ls`` plus ``<root>/vhdl_libraries``).
    A binary installed with ``cargo install --path <rust_hdl checkout>``
    has no libraries bundled next to it at all, so vhdl_ls panics on
    every invocation without an explicit ``-l`` (this is exactly the
    production bug: 'language server connection closed' from every
    sync/reindex). ``VHDL_LS_TEST_LIBRARIES_DIR`` lets a dev environment
    point at the checkout's own ``vhdl_libraries`` directory instead.
    """
    if REAL_LIBS and Path(REAL_LIBS).is_dir():
        return Path(REAL_LIBS)
    return default_libraries_dir(binary)


real_binary = pytest.mark.skipif(
    _find_real_binary() is None,
    reason="no vhdl_ls binary available (set VHDL_LS_TEST_BIN)",
)


@real_binary
async def test_real_vhdl_ls_roundtrip(tmp_path: Path):
    binary = _find_real_binary()
    assert binary is not None
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "rtl").mkdir()
    (ws / "rtl" / "fifo.vhd").write_text(
        "entity fifo is\n  generic (WIDTH : integer := 8);\nend entity;\n\n"
        "architecture rtl of fifo is\n"
        "begin\n  p : process is\n  begin\n    wait;\n  end process;\n"
        "end architecture;\n"
    )
    lsp = VhdlLsp(binary, ws, libraries_dir=_find_real_libraries_dir(binary))
    try:
        await lsp.start()
        await lsp.open_document(ws / "rtl" / "fifo.vhd")
        await lsp.wait_until_quiet()
        syms = await lsp.document_symbols(ws / "rtl" / "fifo.vhd")
        assert syms, "expected at least one symbol"
        # vhdl_ls prefixes names with the kind (e.g. "entity 'fifo'") and
        # reports entity and architecture as top-level siblings.
        names = [s.name for s in syms]
        assert any("fifo" in n for n in names)
        arch = next(s for s in syms if "rtl" in s.name)
        assert any("p" in c.name for c in arch.children)
    finally:
        await lsp.shutdown()


@pytest.mark.skipif(
    _find_real_binary() is None
    or default_libraries_dir(_find_real_binary() or "") is not None,
    reason=(
        "no vhdl_ls binary available, or this environment's binary already "
        "has a sibling vhdl_libraries dir (the bug this test documents does "
        "not reproduce there)"
    ),
)
async def test_real_vhdl_ls_panics_without_libraries_dir(tmp_path: Path) -> None:
    """Documents the actual production bug: a vhdl_ls built with 'cargo
    install --path' (no bundled vhdl_libraries next to the installed
    binary) panics on *every* invocation unless an explicit libraries_dir
    is supplied, surfacing to callers as 'language server connection
    closed' regardless of workspace content."""
    binary = _find_real_binary()
    assert binary is not None
    ws = tmp_path / "ws"
    ws.mkdir()
    lsp = VhdlLsp(binary, ws, libraries_dir=None, files=())
    try:
        with pytest.raises(LspError, match="connection closed"):
            await lsp.start()
    finally:
        await lsp.shutdown()
