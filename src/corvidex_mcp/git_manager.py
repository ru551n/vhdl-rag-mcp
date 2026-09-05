"""Git repository management: clone, fetch, incremental change detection.

For remote repositories the manager owns the local clones (under
``<data_dir>/repos/<name>``) and computes what changed between the last
indexed commit and the commit the configured ``ref`` resolves to:
added/modified files, renamed files, and deleted files. For local
working repositories (configured with ``path``) the user's own checkout
is indexed in place — HEAD plus uncommitted changes and untracked files
— without ever cloning, fetching, or mutating the tree. Nothing here
touches the vector store or embeddings; the indexing pipeline consumes the
:class:`SyncPlan`.

Submodules
----------
Gitlinks (submodule pointers) are descended into, so a repository's
submodule contents are indexed under their prefixed paths (e.g.
``hdl-modules/modules/fifo/src/fifo.vhd``). Remote repositories have
their submodules initialized at the recorded SHAs (``git submodule
update --init --force --recursive``); a gitlink whose submodule SHA
changed re-chunks every file in the submodule (files added at the new
SHA are indexed, files gone at the new SHA are dropped via the old
SHA's file list — or a prefix deletion when the old SHA is not
available). Local working repositories re-chunk a submodule when its
pointer or working tree moved, and otherwise diff inside it (tracked
changes plus untracked files, honoring the repository's
``index_untracked`` flag). The last indexed SHA per gitlink is persisted
in the repository state (``submodules``).

Refs
----
``ref`` is any resolvable Git ref. A branch is tracked: every sync fetches
and moves with the remote branch. A tag or commit SHA pins the repository
to a fixed version; a full-SHA pin is resolved purely locally and never
touches the network.

All commands run with ``GIT_TERMINAL_PROMPT=0`` so an unattended server
can never hang on a credential prompt.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .config import RepositoryConfig

logger = logging.getLogger(__name__)

CLONE_TIMEOUT = 600.0
FETCH_TIMEOUT = 300.0
GIT_TIMEOUT = 120.0


class GitError(RuntimeError):
    """A git operation failed."""


def _hint(stderr: str) -> str:
    """An actionable hint for a common git failure ("" when unknown).

    The raw git stderr is terse ("fatal: ... not found"); the hint tells
    the user what to check, so a misconfigured url or ref produces a
    log line that can be acted on directly.
    """
    e = stderr.lower()
    if "does not appear to be a git repository" in e or (
        "repository" in e and "does not exist" in e
    ):
        return (
            "hint: the url does not point at a reachable git repository - "
            "check the spelling and protocol (https://, git@host:..., "
            "file://); for a local directory, 'url' must point at an "
            "existing git checkout"
        )
    if "repository" in e and "not found" in e:
        return (
            "hint: the remote is reachable but the repository was not "
            "found - check the repository name/spelling and that this "
            "machine has access to it"
        )
    if "could not resolve hostname" in e:
        return (
            "hint: the host name could not be resolved - check the network "
            "connection and the host spelling"
        )
    if "host key verification failed" in e:
        return (
            "hint: the ssh host key is not trusted - run "
            "'ssh-keyscan <host>' and append the output to ~/.ssh/known_hosts"
        )
    if "permission denied" in e:
        return (
            "hint: authentication failed - for git@ urls check that an ssh "
            "key for the host is loaded (ssh-add -l); for https:// urls "
            "check the credentials (credential helper / token)"
        )
    if "authentication failed" in e:
        return (
            "hint: the remote rejected the credentials - check the username "
            "and access token for this host"
        )
    if "connection timed out" in e:
        return (
            "hint: the connection timed out - check the network connection, "
            "firewall, and proxy settings"
        )
    if "unable to access" in e:
        return (
            "hint: git could not reach the remote - check the network "
            "connection, proxy settings, and the url"
        )
    return ""


@dataclass(frozen=True)
class SyncPlan:
    """What the indexing pipeline must do for one repository.

    For a full plan, ``added_or_modified`` lists every file at the target
    commit (including submodule files, prefixed with the gitlink path)
    and the pipeline deletes the repository from the store first.
    For an incremental plan it lists only the changed files (renames
    appear under their new path, with the old path in ``deleted``).
    """

    name: str
    ref: str
    commit: str
    full: bool
    added_or_modified: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()
    #: Local working repositories only: the current untracked files
    #: (honoring ``.gitignore``; includes submodule-prefixed paths for
    #: initialized submodules). The pipeline fingerprints their content
    #: before merging them into ``added_or_modified``/``deleted``.
    untracked: tuple[str, ...] = ()
    #: Local working repositories only: working-tree fingerprint at plan
    #: time (see :meth:`GitManager.local_fingerprint`).
    fingerprint: str | None = None
    #: Gitlink (submodule) path -> submodule SHA indexed by this plan.
    #: Persisted in the repository state so the next sync can diff each
    #: submodule against its previously indexed SHA.
    submodules: dict[str, str] = field(default_factory=dict)
    #: Submodule prefixes to purge wholesale (gitlink removed, or a
    #: submodule SHA change whose old SHA is unavailable locally). The
    #: pipeline deletes every chunk whose file starts with ``prefix/``.
    deleted_submodule_prefixes: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        """True when a sync found nothing to do (ref did not move)."""
        return (
            not self.full
            and not self.added_or_modified
            and not self.deleted
            and not self.deleted_submodule_prefixes
        )


def parse_name_status_z(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Parse ``git diff -z --name-status`` output.

    The ``-z`` form is a flat NUL-separated token stream: every entry is
    its status code followed by one path (``A``, ``M``, ``T``, ``D``) or
    two paths (``R100``, ``C100``: old then new).

    Returns ``(added_or_modified, deleted)``; a rename contributes its
    old path to ``deleted`` and its new path to ``added_or_modified``.
    """
    added: list[str] = []
    deleted: list[str] = []
    tokens = text.split("\0")
    i = 0
    while i < len(tokens):
        status = tokens[i]
        if not status:
            i += 1
            continue
        code = status[0]
        if code in ("R", "C"):
            if i + 2 >= len(tokens):
                break
            deleted.append(tokens[i + 1])
            added.append(tokens[i + 2])
            i += 3
        else:
            if i + 1 >= len(tokens):
                break
            path = tokens[i + 1]
            if code in ("A", "M", "T"):
                added.append(path)
            elif code == "D":
                deleted.append(path)
            i += 2
    return tuple(added), tuple(deleted)


def parse_porcelain_untracked_z(text: str) -> tuple[str, ...]:
    """Untracked (``??``) paths from ``git status --porcelain -z
    --untracked-files=all`` output.

    Tokenized the same way as :func:`parse_name_status_z` (NUL-separated;
    a rename/copy entry is followed by an extra token carrying the old
    path), but only ``??`` entries are kept. Equivalent to ``git ls-files
    -z --others --exclude-standard`` (both honor ``.gitignore`` and list
    one entry per file, since ``--untracked-files=all`` disables status's
    directory-summary shorthand), so a status text already fetched for
    the working-tree fingerprint can double as the untracked-file list.
    """
    paths: list[str] = []
    tokens = text.split("\0")
    i = 0
    while i < len(tokens):
        entry = tokens[i]
        if not entry:
            i += 1
            continue
        code = entry[:2]
        if code[0] in ("R", "C"):
            i += 2  # skip the paired "orig path" token
            continue
        if code == "??":
            paths.append(entry[3:])
        i += 1
    return tuple(sorted(paths))


class GitManager:
    """Clones and synchronizes repositories; reports changes as SyncPlans."""

    def __init__(self, repos_dir: Path) -> None:
        self._repos_dir = repos_dir

    def repo_dir(self, cfg: RepositoryConfig) -> Path:
        """Working tree for ``cfg``: the user's checkout for local
        repositories, the managed clone for remote ones."""
        if cfg.path is not None:
            return cfg.path
        return self._repos_dir / cfg.name

    # -- low-level git ----------------------------------------------------

    async def local_fingerprint(self, cfg: RepositoryConfig) -> str:
        """Cheap, read-only fingerprint of a local repository.

        For filesystem repositories it is the walk fingerprint (see
        :meth:`_filesystem_fingerprint`): paths plus mtimes and sizes, so
        the fast poller notices file-set changes and content edits.

        For working Git repositories it is
        ``sha256(HEAD + porcelain status)`` with untracked files expanded
        (``--untracked-files=all``), so it changes when a commit lands, a
        tracked file is staged/unstaged-edited, or an untracked file is
        added, removed, or renamed. A content edit to an *existing*
        untracked file does not change it (the path and its ``??`` line
        are unchanged); those edits are picked up by the periodic
        sync's plan-time content fingerprinting. The command is
        non-mutating, so it tolerates the user's in-progress work.
        """
        assert cfg.path is not None
        if cfg.filesystem:
            fingerprint, _ = self._filesystem_fingerprint(cfg.path)
            return fingerprint
        head, status = await self._local_head_and_status(cfg.path)
        return self._fingerprint_hash(head, status)

    async def _local_head_and_status(self, repo_dir: Path) -> tuple[str, str]:
        """``(HEAD sha, porcelain status text)`` in one ``rev-parse`` and
        one ``status`` call.

        Shared by :meth:`local_fingerprint` and :meth:`_sync_local` so a
        sync does not re-run ``rev-parse HEAD`` a second time right after
        computing the same fingerprint itself needs.
        """
        head = (await self._run(repo_dir, "rev-parse", "HEAD")).strip()
        status = await self._try(
            repo_dir, "status", "--porcelain", "-z", "--untracked-files=all"
        )
        return head, status or ""

    @staticmethod
    def _fingerprint_hash(head: str, status: str) -> str:
        """``sha256(head + status)``: the local working-tree fingerprint."""
        h = hashlib.sha256()
        h.update(head.encode("utf-8"))
        h.update(b"\x00")
        h.update(status.encode("utf-8"))
        return h.hexdigest()

    def _make_plan(
        self,
        cfg: RepositoryConfig,
        ref: str,
        commit: str,
        *,
        full: bool,
        added_or_modified: tuple[str, ...] = (),
        deleted: tuple[str, ...] = (),
        untracked: tuple[str, ...] = (),
        fingerprint: str | None = None,
        submodules: dict[str, str] | None = None,
        deleted_submodule_prefixes: tuple[str, ...] = (),
    ) -> SyncPlan:
        """Build a :class:`SyncPlan` for ``cfg`` (fills in ``cfg.name``,
        common to every plan returned by :meth:`sync`/:meth:`_sync_local`)."""
        return SyncPlan(
            cfg.name,
            ref,
            commit,
            full=full,
            added_or_modified=added_or_modified,
            deleted=deleted,
            untracked=untracked,
            fingerprint=fingerprint,
            submodules=submodules if submodules is not None else {},
            deleted_submodule_prefixes=deleted_submodule_prefixes,
        )

    async def _git(
        self, cwd: Path, *args: str, timeout: float = GIT_TIMEOUT
    ) -> tuple[int, str, str]:
        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GIT_LFS_SKIP_SMUDGE"] = "1"
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=str(cwd),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise GitError(f"git {args[0]} timed out after {timeout:.0f}s") from None
        return (
            proc.returncode if proc.returncode is not None else -1,
            out.decode(errors="replace"),
            err.decode(errors="replace"),
        )

    async def _run(self, cwd: Path, *args: str, timeout: float = GIT_TIMEOUT) -> str:
        code, out, err = await self._git(cwd, *args, timeout=timeout)
        if code != 0:
            message = err.strip()[-500:] or out.strip()[-500:]
            hint = _hint(err)
            if hint:
                message = f"{message}\n{hint}"
            raise GitError(f"git {args[0]} failed (exit {code}): {message}")
        return out

    async def _try(
        self, cwd: Path, *args: str, timeout: float = GIT_TIMEOUT
    ) -> str | None:
        code, out, _ = await self._git(cwd, *args, timeout=timeout)
        return out if code == 0 else None

    # -- repository lifecycle ---------------------------------------------

    async def ensure_clone(self, cfg: RepositoryConfig) -> None:
        """Clone the repository on first use (idempotent)."""
        assert cfg.url is not None  # remote repositories only
        repo_dir = self.repo_dir(cfg)
        if (repo_dir / ".git").exists():
            return
        if repo_dir.exists():
            # Leftover of an interrupted clone: unusable without .git.
            logger.warning("removing incomplete clone directory %s", repo_dir)
            shutil.rmtree(repo_dir, ignore_errors=True)
        self._repos_dir.mkdir(parents=True, exist_ok=True)
        logger.info("cloning %s -> %s", cfg.url, repo_dir)
        await self._run(
            self._repos_dir,
            "clone",
            "--",
            cfg.url,
            str(repo_dir),
            timeout=CLONE_TIMEOUT,
        )

    async def _resolve(self, cfg: RepositoryConfig, repo_dir: Path) -> str:
        """Resolve the configured ref to a commit SHA (fetching if allowed)."""
        ref = cfg.ref
        if cfg.is_pinned_sha:
            commit = await self._try(
                repo_dir, "rev-parse", "--verify", f"{ref}^{{commit}}"
            )
            if commit is None:
                raise GitError(
                    f"pinned SHA {ref[:12]} not present in the local clone of "
                    f"{cfg.name!r}; full-SHA pins are never fetched, so the "
                    "commit must be reachable from a cloned branch"
                )
            return commit.strip()
        await self._run(
            repo_dir,
            "fetch",
            "--prune",
            "--tags",
            "origin",
            timeout=FETCH_TIMEOUT,
        )
        # Remote-tracking ref first: we check out detached HEADs, so a local
        # branch (if any) may still point at the previous commit. Tags and
        # plain SHAs only resolve as the ref itself.
        for candidate in (f"origin/{ref}", ref):
            commit = await self._try(
                repo_dir, "rev-parse", "--verify", f"{candidate}^{{commit}}"
            )
            if commit is not None:
                return commit.strip()
        available = await self._try(
            repo_dir,
            "for-each-ref",
            "--count=10",
            "--format=%(refname:short)",
            "refs/remotes/origin",
        )
        names = [
            n.strip()
            for n in (available or "").splitlines()
            if n.strip() and "/" in n.strip()
        ]
        hint = f"; available remote refs: {', '.join(names)}" if names else ""
        raise GitError(
            f"ref {ref!r} does not resolve to a commit in {cfg.name!r}{hint}"
        )

    async def _list_files_at(
        self, cfg: RepositoryConfig, commit: str
    ) -> tuple[tuple[str, ...], dict[str, str]]:
        """Every file at ``commit``, descending into gitlinks (submodules).

        Submodule files carry their gitlink path as prefix (``ip/rtl/a.vhd``).
        Submodules whose working tree is absent (not initialized) are
        skipped with a warning — the rest of the repository still syncs.

        Returns ``(files, gitlinks)``: the caller needs both to build a
        full :class:`SyncPlan` (``gitlinks`` becomes ``plan.submodules``),
        and this way it does not have to call :meth:`_gitlinks_at_commit`
        a second time for the same commit.
        """
        repo_dir = self.repo_dir(cfg)
        files = set(await self._top_level_files_at(repo_dir, commit))
        gitlinks = await self._gitlinks_at_commit(repo_dir, commit)
        for path, sha in gitlinks.items():
            try:
                files.update(
                    await self._submodule_tree_files(
                        repo_dir / path, sha, path, depth=1
                    )
                )
            except GitError as exc:
                logger.warning(
                    "%s: submodule %s not expanded: %s",
                    cfg.name,
                    path,
                    exc,
                )
        return tuple(sorted(files)), gitlinks

    async def _top_level_files_at(self, repo_dir: Path, commit: str) -> list[str]:
        out = await self._run(repo_dir, "ls-tree", "-r", "-z", "--name-only", commit)
        return [path for path in out.split("\0") if path]

    # -- submodule helpers ----------------------------------------------------

    _SUBMODULE_DEPTH = 3

    async def _gitlinks_at_commit(self, repo_dir: Path, commit: str) -> dict[str, str]:
        """Top-level gitlinks at ``commit``: submodule path -> SHA."""
        out = await self._run(repo_dir, "ls-tree", "-z", commit)
        links: dict[str, str] = {}
        for entry in out.split("\0"):
            if not entry or "\t" not in entry:
                continue
            meta, path = entry.split("\t", 1)
            parts = meta.split(" ")
            if len(parts) == 3 and parts[0] == "160000":
                links[path] = parts[2]
        return links

    async def _gitlinks_in_index(self, repo_dir: Path) -> dict[str, str]:
        """Top-level gitlinks in the index/working tree of a local
        repository: submodule path -> recorded SHA."""
        out = await self._run(repo_dir, "ls-files", "--stage", "-z")
        links: dict[str, str] = {}
        for entry in out.split("\0"):
            if not entry or "\t" not in entry:
                continue
            meta, path = entry.split("\t", 1)
            parts = meta.split(" ")
            # ls-files --stage format: "<mode> <sha> <stage>\t<path>".
            if len(parts) == 3 and parts[0] == "160000":
                links[path] = parts[1]
        return links

    async def _submodule_tree_files(
        self, sub_dir: Path, sha: str, prefix: str, depth: int
    ) -> set[str]:
        """All files of a submodule at ``sha``, prefixed with ``prefix``.

        Recursive: nested gitlinks are descended into (bounded by
        ``_SUBMODULE_DEPTH``). Raises :class:`GitError` when the submodule
        is not present on disk or ``sha`` is not a commit in it.
        """
        if not (sub_dir / ".git").exists():
            raise GitError(f"submodule {prefix!r} is not initialized")
        files: set[str] = set()
        out = await self._run(sub_dir, "ls-tree", "-r", "-z", "--name-only", sha)
        for path in (p for p in out.split("\0") if p):
            files.add(f"{prefix}/{path}")
        if depth < self._SUBMODULE_DEPTH:
            for nested, nested_sha in (
                await self._gitlinks_at_commit(sub_dir, sha)
            ).items():
                files.update(
                    await self._submodule_tree_files(
                        sub_dir / nested, nested_sha, f"{prefix}/{nested}", depth + 1
                    )
                )
        return files

    async def _submodule_worktree_files(
        self,
        sub_dir: Path,
        prefix: str,
        index_untracked: bool,
        depth: int,
    ) -> set[str]:
        """All files of an initialized submodule working tree (tracked,
        plus untracked honoring ``.gitignore`` when enabled), prefixed.
        Nested submodules are descended into when initialized."""
        if not (sub_dir / ".git").exists():
            raise GitError(f"submodule {prefix!r} is not initialized")
        tracked = await self._run(sub_dir, "ls-files", "-z")
        files = {p for p in tracked.split("\0") if p}
        if index_untracked:
            others = await self._run(
                sub_dir, "ls-files", "-z", "--others", "--exclude-standard"
            )
            files.update(p for p in others.split("\0") if p)
        # Nested gitlinks come from the index directly (one cheap call),
        # instead of stat'ing every tracked file's `<path>/.git`.
        gitlinks = (
            await self._gitlinks_in_index(sub_dir)
            if depth < self._SUBMODULE_DEPTH
            else {}
        )
        result: set[str] = set()
        for path in files:
            if path in gitlinks and (sub_dir / path / ".git").exists():
                # A nested submodule: the parent lists it as a gitlink
                # (a bare path entry), so enumerate its tree instead. The
                # recursive call already returns paths fully prefixed
                # (from its own `prefix`, which is `{prefix}/{path}` here)
                # — add them as-is, don't prefix them again below.
                result.update(
                    await self._submodule_worktree_files(
                        sub_dir / path,
                        f"{prefix}/{path}",
                        index_untracked,
                        depth + 1,
                    )
                )
            else:
                result.add(f"{prefix}/{path}")
        return result

    async def _update_submodules(self, repo_dir: Path) -> None:
        """Initialize submodules at the recorded SHAs (best effort).

        Failures (e.g. a private submodule whose credentials are not in
        the ambient Git/SSH setup) are logged, not raised: the rest of
        the repository still syncs, and the failed submodule is skipped
        by the file expansion.
        """
        out = await self._try(
            repo_dir,
            "submodule",
            "update",
            "--init",
            "--force",
            "--recursive",
            timeout=CLONE_TIMEOUT,
        )
        if out is None:
            logger.warning(
                "%s: git submodule update failed; submodule files are "
                "skipped until it succeeds",
                repo_dir.name,
            )

    async def sync(
        self,
        cfg: RepositoryConfig,
        last_commit: str | None,
        last_submodules: dict[str, str] | None = None,
    ) -> SyncPlan:
        """Synchronize the repository and report the changes vs
        ``last_commit`` (and, per submodule, vs ``last_submodules``).

        ``last_commit`` is the last fully indexed commit (``None`` before
        the first index run). Remote repositories have their working
        tree left at the target commit (submodules initialized at the
        recorded SHAs); local working repositories (``path``) are indexed
        in place — HEAD plus uncommitted changes and untracked files —
        and are never modified. Filesystem repositories (``path`` +
        ``filesystem``) are indexed in place as plain files with no Git
        at all.
        """
        if cfg.filesystem:
            return await self._sync_filesystem(cfg, last_commit)
        if cfg.is_local:
            return await self._sync_local(cfg, last_commit, last_submodules)
        await self.ensure_clone(cfg)
        repo_dir = self.repo_dir(cfg)
        commit = await self._resolve(cfg, repo_dir)
        head = await self._try(repo_dir, "rev-parse", "--verify", "HEAD")
        if head is None or head.strip() != commit:
            await self._run(repo_dir, "checkout", "--detach", commit)
        await self._update_submodules(repo_dir)
        if last_commit is None:
            files, gitlinks = await self._list_files_at(cfg, commit)
            logger.info(
                "%s: first sync, full index of %d files at %s",
                cfg.name,
                len(files),
                commit[:12],
            )
            return self._make_plan(
                cfg,
                cfg.ref,
                commit,
                full=True,
                added_or_modified=files,
                submodules=gitlinks,
            )
        if last_commit == commit:
            return self._make_plan(
                cfg,
                cfg.ref,
                commit,
                full=False,
                submodules=await self._gitlinks_at_commit(repo_dir, commit),
            )
        diff = await self._try(
            repo_dir, "diff", "-z", "--name-status", last_commit, commit
        )
        if diff is None:
            # Previous commit is gone (history rewrite / force push):
            # the safe fallback is a full reindex.
            logger.warning(
                "%s: cannot diff %s..%s; falling back to full reindex",
                cfg.name,
                last_commit[:12],
                commit[:12],
            )
            files, gitlinks = await self._list_files_at(cfg, commit)
            return self._make_plan(
                cfg,
                cfg.ref,
                commit,
                full=True,
                added_or_modified=files,
                submodules=gitlinks,
            )
        added, deleted = parse_name_status_z(diff)
        new_links = await self._gitlinks_at_commit(repo_dir, commit)
        old_links = last_submodules or {}
        # Gitlink entries in the diff are pointer changes; they are
        # resolved to file-level changes below (or a prefix purge).
        link_paths = set(new_links) | set(old_links)
        plan_added: set[str] = {p for p in added if p not in link_paths}
        plan_deleted: set[str] = {p for p in deleted if p not in link_paths}
        plan_deleted_prefixes: list[str] = []
        for path, new_sha in sorted(new_links.items()):
            old_sha = old_links.get(path)
            try:
                new_files = await self._submodule_tree_files(
                    repo_dir / path, new_sha, path, 1
                )
            except GitError as exc:
                # Submodule not initialized (update failed, e.g. missing
                # credentials): skip it; the rest of the repo still syncs.
                logger.warning("%s: submodule %s skipped: %s", cfg.name, path, exc)
                continue
            if old_sha is None:
                # New submodule: index everything it contains.
                plan_added.update(new_files)
                continue
            if old_sha == new_sha:
                continue
            # SHA changed: re-chunk the whole submodule; drop files gone
            # at the new SHA via the old SHA's file list.
            plan_added.update(new_files)
            try:
                old_files = await self._submodule_tree_files(
                    repo_dir / path, old_sha, path, 1
                )
                plan_deleted.update(old_files - new_files)
            except GitError:
                logger.warning(
                    "%s: submodule %s: old SHA %s unavailable; purging "
                    "its prefix instead",
                    cfg.name,
                    path,
                    old_sha[:12],
                )
                plan_deleted_prefixes.append(path)
        for path in sorted(set(old_links) - set(new_links)):
            # Submodule removed from the repository: purge its prefix.
            plan_deleted_prefixes.append(path)
        logger.info(
            "%s: %s..%s: %d added/modified, %d deleted, %d submodule prefix(es) purged",
            cfg.name,
            last_commit[:12],
            commit[:12],
            len(plan_added),
            len(plan_deleted),
            len(plan_deleted_prefixes),
        )
        return self._make_plan(
            cfg,
            cfg.ref,
            commit,
            full=False,
            added_or_modified=tuple(sorted(plan_added)),
            deleted=tuple(sorted(plan_deleted)),
            submodules=new_links,
            deleted_submodule_prefixes=tuple(sorted(plan_deleted_prefixes)),
        )

    # -- local working repositories -----------------------------------------

    async def _sync_local(
        self,
        cfg: RepositoryConfig,
        last_commit: str | None,
        last_submodules: dict[str, str] | None = None,
    ) -> SyncPlan:
        """Index the user's working repository in place (no clone/fetch).

        The index covers HEAD plus uncommitted changes (staged and
        unstaged) and untracked files (honoring ``.gitignore``; skipped
        when ``index_untracked`` is false);
        attribution is the current HEAD commit. Untracked files are
        reported via ``plan.untracked`` together with the working-tree
        ``plan.fingerprint`` so the pipeline can fingerprint their
        content, skip re-chunking unchanged ones, and detect deleted
        untracked files.

        Submodules are descended into: a submodule whose pointer or
        working tree moved is re-chunked wholesale (files gone at the
        new content are dropped via the old pointer's file list);
        otherwise its tracked changes and untracked files are diffed
        inside it and reported with the gitlink path as prefix.
        """
        assert cfg.path is not None
        repo_dir = cfg.path
        if not repo_dir.is_dir():
            raise GitError(
                f"repository {cfg.name!r}: path {repo_dir} does not exist "
                "or is not a directory"
            )
        if await self._try(repo_dir, "rev-parse", "--git-dir") is None:
            raise GitError(
                f"repository {cfg.name!r}: path {repo_dir} is not a Git repository"
            )
        commit, status = await self._local_head_and_status(repo_dir)
        branch = (
            await self._run(repo_dir, "rev-parse", "--abbrev-ref", "HEAD")
        ).strip()
        fingerprint = self._fingerprint_hash(commit, status)
        # The top-level untracked set is derived from the status text
        # already fetched above for the fingerprint (see
        # `parse_porcelain_untracked_z`), instead of a further
        # `ls-files --others` call.
        top_untracked = parse_porcelain_untracked_z(status)
        new_links = await self._gitlinks_in_index(repo_dir)
        old_links = last_submodules or {}
        if last_commit is None:
            files = await self._list_local_files(
                cfg, repo_dir, new_links, top_untracked
            )
            logger.info(
                "%s: first sync, full index of %d files at %s (%s)",
                cfg.name,
                len(files),
                commit[:12],
                branch,
            )
            return self._make_plan(
                cfg,
                branch,
                commit,
                full=True,
                added_or_modified=files,
                untracked=await self._local_untracked_all(
                    cfg, repo_dir, new_links, top_untracked
                ),
                fingerprint=fingerprint,
                submodules=new_links,
            )
        added: set[str] = set()
        deleted: set[str] = set()
        deleted_prefixes: list[str] = []
        handled_links: set[str] = set()
        if last_commit != commit:
            diff = await self._try(
                repo_dir, "diff", "-z", "--name-status", last_commit, commit
            )
            if diff is None:
                # The last indexed commit is gone (history rewrite):
                # a full reindex is the safe fallback.
                logger.warning(
                    "%s: cannot diff %s..%s; falling back to full reindex",
                    cfg.name,
                    last_commit[:12],
                    commit[:12],
                )
                files = await self._list_local_files(
                    cfg, repo_dir, new_links, top_untracked
                )
                return self._make_plan(
                    cfg,
                    branch,
                    commit,
                    full=True,
                    added_or_modified=files,
                    untracked=await self._local_untracked_all(
                        cfg, repo_dir, new_links, top_untracked
                    ),
                    fingerprint=fingerprint,
                    submodules=new_links,
                )
            a, d = parse_name_status_z(diff)
            for p in a:
                if p in new_links:
                    # Committed pointer move (or new submodule): re-chunk
                    # the submodule's working tree; old content via the
                    # previously recorded SHA.
                    await self._rechunk_local_submodule(
                        cfg,
                        repo_dir,
                        p,
                        old_links.get(p),
                        added,
                        deleted,
                        deleted_prefixes,
                    )
                    handled_links.add(p)
                else:
                    added.add(p)
            for p in d:
                if p in old_links:
                    # Submodule removed: purge its prefix.
                    deleted_prefixes.append(p)
                else:
                    deleted.add(p)
        # Uncommitted work: index/worktree-vs-HEAD (staged + unstaged).
        worktree = await self._try(repo_dir, "diff", "-z", "--name-status", "HEAD")
        if worktree is not None:
            a, d = parse_name_status_z(worktree)
            for p in a:
                if p in new_links and p not in handled_links:
                    sub_dir = repo_dir / p
                    current = await self._try(sub_dir, "rev-parse", "HEAD")
                    recorded = old_links.get(p)
                    if current and current.strip() != recorded:
                        # Commit drift (submodule HEAD moved away from the
                        # recorded pointer, or the pointer was just
                        # staged): re-chunk the submodule against the
                        # previously indexed content.
                        await self._rechunk_local_submodule(
                            cfg,
                            repo_dir,
                            p,
                            old_links.get(p) or new_links[p],
                            added,
                            deleted,
                            deleted_prefixes,
                        )
                        handled_links.add(p)
                    # Content-level changes (tracked or untracked files)
                    # are resolved per file by the submodule loop below.
                else:
                    added.add(p)
            for p in d:
                if p in new_links:
                    deleted_prefixes.append(p)
                else:
                    deleted.add(p)
        # Submodules: stable pointers get an inside diff (tracked changes
        # + untracked files); a missing working tree (deinit/rm) purges.
        untracked: list[str] = list(top_untracked) if cfg.index_untracked else []
        for path in sorted(new_links):
            sub_dir = repo_dir / path
            if not (sub_dir / ".git").exists():
                if path in old_links:
                    deleted_prefixes.append(path)
                continue
            if path in handled_links:
                continue
            sub_diff = await self._try(sub_dir, "diff", "-z", "--name-status", "HEAD")
            if sub_diff:
                a, d = parse_name_status_z(sub_diff)
                added.update(f"{path}/{p}" for p in a)
                deleted.update(f"{path}/{p}" for p in d)
            if cfg.index_untracked:
                others = await self._run(
                    sub_dir, "ls-files", "-z", "--others", "--exclude-standard"
                )
                untracked.extend(f"{path}/{p}" for p in sorted(others.split("\0")) if p)
        added.difference_update(deleted)
        plan_added = tuple(sorted(added))
        plan_deleted = tuple(sorted(deleted))
        plan_untracked = tuple(sorted(set(untracked)))
        if (
            not plan_added
            and not plan_deleted
            and not plan_untracked
            and not deleted_prefixes
        ):
            logger.info("%s: working tree unchanged at %s", cfg.name, commit[:12])
            return self._make_plan(
                cfg,
                branch,
                commit,
                full=False,
                fingerprint=fingerprint,
                submodules=new_links,
            )
        logger.info(
            "%s: %d added/modified, %d deleted (working tree at %s)",
            cfg.name,
            len(plan_added),
            len(plan_deleted),
            commit[:12],
        )
        return self._make_plan(
            cfg,
            branch,
            commit,
            full=False,
            added_or_modified=plan_added,
            deleted=plan_deleted,
            untracked=plan_untracked,
            fingerprint=fingerprint,
            submodules=new_links,
            deleted_submodule_prefixes=tuple(sorted(set(deleted_prefixes))),
        )

    async def _list_local_files(
        self,
        cfg: RepositoryConfig,
        repo_dir: Path,
        new_links: dict[str, str],
        top_untracked: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Everything to index in a working repository: top-level tracked
        plus git-respected untracked files (unless the repository
        disabled untracked indexing), descending into initialized
        submodules (their files carry the gitlink path as prefix).

        ``top_untracked`` is the ``??`` set already parsed from the
        working-tree status text (see :func:`parse_porcelain_untracked_z`),
        so this does not re-run ``ls-files --others`` at the top level.
        """
        tracked = await self._run(repo_dir, "ls-files", "-z")
        files = {p for p in tracked.split("\0") if p}
        files.difference_update(new_links)  # gitlinks expand below
        if cfg.index_untracked:
            files.update(top_untracked)
            files.difference_update(new_links)
        for path in sorted(new_links):
            sub_dir = repo_dir / path
            try:
                files.update(
                    await self._submodule_worktree_files(
                        sub_dir, path, cfg.index_untracked, depth=1
                    )
                )
            except GitError as exc:
                logger.warning("%s: submodule %s not expanded: %s", cfg.name, path, exc)
        return tuple(sorted(files))

    async def _local_untracked_all(
        self,
        cfg: RepositoryConfig,
        repo_dir: Path,
        new_links: dict[str, str],
        top_untracked: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Untracked files of a working repository and its initialized
        submodules (honoring ``.gitignore``), submodule paths prefixed,
        sorted. Empty when untracked indexing is disabled.

        ``top_untracked`` is the ``??`` set already parsed from the
        working-tree status text (see :func:`parse_porcelain_untracked_z`),
        reused here instead of a further top-level ``ls-files --others``
        call.
        """
        if not cfg.index_untracked:
            return ()
        files = set(top_untracked)
        for path in sorted(new_links):
            sub_dir = repo_dir / path
            try:
                tracked = set((await self._run(sub_dir, "ls-files", "-z")).split("\0"))
                others = await self._run(
                    sub_dir, "ls-files", "-z", "--others", "--exclude-standard"
                )
                for p in others.split("\0"):
                    if p and p not in tracked:
                        files.add(f"{path}/{p}")
            except GitError:
                continue  # uninitialized submodule: nothing to report
        return tuple(sorted(files))

    async def _rechunk_local_submodule(
        self,
        cfg: RepositoryConfig,
        repo_dir: Path,
        path: str,
        old_sha: str | None,
        added: set[str],
        deleted: set[str],
        deleted_prefixes: list[str],
    ) -> None:
        """Re-chunk a local submodule whose pointer or working tree moved.

        Every file in the working tree goes to ``added`` (the pipeline
        drops the file's previous chunks before re-upserting); files
        gone versus ``old_sha``'s content go to ``deleted``. A missing
        working tree (deinit/rm) or an unavailable old SHA degrades to a
        prefix purge."""
        sub_dir = repo_dir / path
        try:
            new_files = await self._submodule_worktree_files(
                sub_dir, path, cfg.index_untracked, depth=1
            )
        except GitError as exc:
            logger.warning("%s: submodule %s: %s", cfg.name, path, exc)
            if old_sha is not None:
                deleted_prefixes.append(path)
            return
        added.update(new_files)
        if old_sha is None:
            return
        try:
            old_files = await self._submodule_tree_files(sub_dir, old_sha, path, 1)
            deleted.update(old_files - new_files)
        except GitError:
            logger.warning(
                "%s: submodule %s: old SHA %s unavailable; purging its prefix instead",
                cfg.name,
                path,
                old_sha[:12],
            )
            deleted_prefixes.append(path)

    # -- filesystem repositories ----------------------------------------------

    async def _sync_filesystem(
        self, cfg: RepositoryConfig, last_commit: str | None
    ) -> SyncPlan:
        """Index a plain directory with no Git involved.

        The walked file set is carried in ``untracked`` together with the
        walk ``fingerprint`` (paths + mtimes + sizes), so the pipeline's
        untracked-fingerprint refinement performs the new/changed/deleted
        detection and the fast local poller notices file-set and content
        changes between syncs. ``commit`` is the fingerprint itself, so
        chunk attribution is stable for an unchanged tree. A first sync
        (no last commit) is a full plan listing every file. Every file
        is listed in ``untracked`` on every sync — that part is cheap,
        just the walk above — but when the walk fingerprint matches the
        one stored at the last successful sync, the pipeline's plan
        refinement (:meth:`IndexPipeline._refine_local_plan`) skips
        reading and hashing every one of those files: an unchanged
        fingerprint already proves no file's content changed.
        """
        assert cfg.path is not None
        fingerprint, files = self._filesystem_fingerprint(cfg.path)
        first = last_commit is None
        if first:
            logger.info(
                "%s: first filesystem sync, full index of %d files",
                cfg.name,
                len(files),
            )
        else:
            logger.debug(
                "%s: filesystem walk, %d files at %s",
                cfg.name,
                len(files),
                fingerprint[:12],
            )
        return SyncPlan(
            cfg.name,
            "-",
            fingerprint,
            full=first,
            added_or_modified=files if first else (),
            untracked=files,
            fingerprint=fingerprint,
        )

    @staticmethod
    def _filesystem_fingerprint(root: Path) -> tuple[str, tuple[str, ...]]:
        """Walk a plain directory: return (sha256 of the walk, sorted
        repository-relative file paths).

        Hidden entries (dot files and dot directories) are skipped, which
        keeps an embedded ``.git`` directory out of the index; so are
        symlinks (no cycles, no dangling targets) and other non-regular
        files. Each entry contributes ``path + mtime + size``, so the
        fingerprint changes when the file set changes or a file is
        touched/edited (the poller uses it); content-level change
        detection remains the pipeline's sha256 content fingerprints.

        This runs from the fast local poller every ``local_sync_interval``
        (as well as at plan time), so it is written for one ``os.scandir``
        pass per directory rather than ``os.walk`` + ``Path.is_symlink``
        / ``Path.is_file`` / ``Path.stat``: each ``os.scandir`` entry
        already carries the directory-entry type (``d_type`` on
        platforms that support it), so ``is_symlink()`` and
        ``is_dir()``/``is_file(follow_symlinks=False)`` are typically
        free, and the one ``stat(follow_symlinks=False)`` call needed for
        mtime/size reuses that same cached result — one syscall per file
        instead of three. The final entry list is sorted before hashing,
        so the (unspecified) order ``os.scandir`` yields entries in does
        not affect the result; the output is byte-identical to an
        ``os.walk``-based walk of the same tree.
        """
        if not root.is_dir():
            raise GitError(
                f"filesystem path {root} does not exist or is not a directory"
            )
        entries: list[str] = []
        GitManager._scan_filesystem_dir(str(root), "", entries)
        entries.sort()
        digest = hashlib.sha256()
        for line in entries:
            digest.update(line.encode("utf-8"))
            digest.update(b"\n")
        paths = tuple(entry.split("\t", 1)[0] for entry in entries)
        return digest.hexdigest(), paths

    @staticmethod
    def _scan_filesystem_dir(dirpath: str, rel_prefix: str, entries: list[str]) -> None:
        """Recursively collect ``"{rel}\\t{mtime_ns}\\t{size}"`` lines for
        one directory into ``entries`` (appended, unordered).

        ``rel_prefix`` is ``""`` at the root and ``"<dir>/"`` for a
        descendant, so relative paths are built with plain string
        concatenation instead of ``Path.relative_to`` /
        ``PurePath.as_posix``. Hidden entries (dot files/dirs) and
        symlinks (files or directories) are skipped, matching
        :meth:`_filesystem_fingerprint`'s docstring; a symlinked
        directory is neither descended into nor recorded, exactly like
        ``os.walk(..., followlinks=False)``.
        """
        with os.scandir(dirpath) as it:
            for entry in it:
                if entry.name.startswith("."):
                    continue
                if entry.is_symlink():
                    continue
                rel = rel_prefix + entry.name
                if entry.is_dir(follow_symlinks=False):
                    GitManager._scan_filesystem_dir(entry.path, rel + "/", entries)
                elif entry.is_file(follow_symlinks=False):
                    st = entry.stat(follow_symlinks=False)
                    entries.append(f"{rel}\t{st.st_mtime_ns}\t{st.st_size}")

    # -- file access --------------------------------------------------------

    def read_file(self, cfg: RepositoryConfig, relpath: str) -> str:
        """Read a file from the working tree (remote repositories sit
        at the last synced commit; local working repositories at the
        user's current working tree)."""
        if ".." in Path(relpath).parts or relpath.startswith("/"):
            raise GitError(f"invalid repository file path: {relpath!r}")
        path = self.repo_dir(cfg) / relpath
        if not path.is_file():
            raise GitError(
                f"file {relpath!r} not found in {cfg.name!r}; use "
                "repository_files to list valid paths in this repository"
            )
        return path.read_text(encoding="utf-8", errors="replace")
