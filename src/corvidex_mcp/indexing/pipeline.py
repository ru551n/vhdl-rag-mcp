"""The indexing pipeline: incremental repository synchronization.

One :meth:`IndexPipeline.sync_repository` call moves one configured
repository from its last indexed commit to the commit its ``ref``
currently resolves to:

1. git sync (clone/fetch/diff) via :mod:`corvidex_mcp.git_manager`;
2. deletion of stale chunks (whole repository for a full plan, per-file
   for deleted/renamed-away files);
3. per-domain chunking of the changed files (VHDL via vhdl_ls, docs,
   general code) respecting the repository's ``domains`` and ``exclude``;
4. embedding (per-collection dense) and upsert into the vector store;
5. state update — only after the index update fully succeeded, so a
   failed run leaves the previous commit as the last indexed one and the
   next sync retries the same diff.

All failures are contained per repository: one broken repository records
its error in the state store and does not affect the others.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
from collections.abc import Callable
from pathlib import Path

from ..config import AppConfig, RepositoryConfig
from ..embeddings.providers import EmbeddingProviders
from ..git_manager import GitManager, SyncPlan
from ..lsp import (
    LspClient,
    SymbolInfo,
    VeridianLsp,
    VhdlLsp,
    default_libraries_dir,
    resolve_binary,
)
from ..models import Chunk, CollectionName, ContentType
from ..routing import FileKind, classify_file
from ..state import StateStore
from ..vector_store import VectorStore
from .code import chunk_code_file
from .docs import chunk_doc_file
from .verilog import chunk_verilog_file
from .vhdl import chunk_vhdl_file

logger = logging.getLogger(__name__)

VHDL_KIND = FileKind(ContentType.SOURCE, CollectionName.HDL, "vhdl")
VERILOG_KIND = FileKind(ContentType.SOURCE, CollectionName.HDL, "verilog")
SYSTEMVERILOG_KIND = FileKind(ContentType.SOURCE, CollectionName.HDL, "systemverilog")


class IndexPipeline:
    """Coordinates git, chunkers, embeddings, and the vector store."""

    def __init__(
        self,
        config: AppConfig,
        git: GitManager,
        store: VectorStore,
        providers: EmbeddingProviders,
        states: StateStore,
    ) -> None:
        self._config = config
        self._git = git
        self._store = store
        self._providers = providers
        self._states = states
        # One sync per repository at a time: concurrent syncs would race
        # on the same git working tree (checkout/read interleaving).
        self._locks: dict[str, asyncio.Lock] = {}
        # Resolved HDL analyzer binaries (None when unavailable): the
        # pipeline degrades to structural/generic parsing per analyzer.
        self._vhdl_ls_bin, _ = resolve_binary(self._config.vhdl_ls_path, "vhdl_ls")
        self._veridian_bin, _ = resolve_binary(self._config.veridian_path, "veridian")
        # Explicit config wins over the sibling-directory auto-detect,
        # which only finds anything for the official release layout
        # (<root>/bin/vhdl_ls plus <root>/vhdl_libraries) and misses e.g. a
        # 'cargo install --path' build, where vhdl_ls has no bundled
        # libraries next to the installed binary at all and panics on
        # every invocation without an explicit -l.
        self._vhdl_ls_libraries_dir = self._config.vhdl_ls_libraries_dir or (
            default_libraries_dir(self._vhdl_ls_bin)
            if self._vhdl_ls_bin is not None
            else None
        )

    def _lock_for(self, name: str) -> asyncio.Lock:
        lock = self._locks.get(name)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[name] = lock
        return lock

    async def sync_repository(self, cfg: RepositoryConfig) -> None:
        """Synchronize one repository from its last indexed commit."""
        async with self._lock_for(cfg.name):
            await self._sync(cfg, self._states.get(cfg.name).indexed_commit)

    async def reindex_repository(self, cfg: RepositoryConfig) -> None:
        """Force a full reindex of one repository (ignores last commit)."""
        async with self._lock_for(cfg.name):
            await self._sync(cfg, None)

    async def _sync(self, cfg: RepositoryConfig, last_commit: str | None) -> None:
        last_state = self._states.get(cfg.name)
        try:
            plan = await self._git.sync(cfg, last_commit, last_state.submodules or None)
        except Exception as exc:
            logger.exception("%s: git sync failed: %s", cfg.name, exc)
            self._states.record_sync(cfg.name, str(exc))
            raise
        plan, untracked_fps = self._refine_local_plan(cfg, plan)
        # None for filesystem repositories (no gitlinks); otherwise the
        # plan carries the current per-gitlink SHA map (possibly empty).
        submodules = None if cfg.filesystem else plan.submodules
        if plan.empty:
            if last_commit is not None and plan.commit != last_commit:
                # HEAD/ref moved without a content change (an amend or
                # a force-push of an identical tree): the chunks are
                # still current, only the attribution commit changed.
                # Advance the indexed commit — otherwise the next sync
                # cannot diff against the rewritten-away commit and
                # falls back to a full reindex.
                logger.info(
                    "%s: commit moved %s -> %s without content change",
                    cfg.name,
                    last_commit[:12],
                    plan.commit[:12],
                )
                self._states.set_indexed(
                    cfg.name,
                    plan.commit,
                    file_count=self._states.get(cfg.name).last_indexed_file_count,
                    submodules=submodules,
                )
            else:
                logger.info(
                    "%s: ref %r did not move (still at %s)",
                    cfg.name,
                    plan.ref,
                    plan.commit[:12],
                )
            self._apply_local_state(cfg, plan, untracked_fps)
            self._states.record_sync(cfg.name, None)
            return
        logger.info(
            "%s: %s plan, %d files at %s%s",
            cfg.name,
            "FULL" if plan.full else "incremental",
            len(plan.added_or_modified),
            plan.commit[:12],
            f" (from {last_commit[:12]})" if last_commit else "",
        )
        try:
            await self._apply_plan(cfg, plan)
        except Exception as exc:
            # Keep the previous indexed commit so the next sync retries
            # the same diff.
            logger.exception(
                "%s: index update failed (%s); previous state kept",
                cfg.name,
                exc,
            )
            self._states.record_sync(cfg.name, str(exc))
            raise
        self._states.set_indexed(
            cfg.name,
            plan.commit,
            file_count=len(plan.added_or_modified),
            submodules=submodules,
        )
        self._apply_local_state(cfg, plan, untracked_fps)
        self._states.record_sync(cfg.name, None)

    # -- local working repositories -------------------------------------------

    def _refine_local_plan(
        self, cfg: RepositoryConfig, plan: SyncPlan
    ) -> tuple[SyncPlan, dict[str, str]]:
        """Merge a local repo's untracked files into the plan using the
        persisted content fingerprints.

        Returns ``(refined_plan, untracked_fingerprints)``. New or
        content-changed untracked files go into ``added_or_modified``;
        untracked files indexed previously but no longer present go into
        ``deleted``; unchanged untracked files are dropped (no
        re-chunking). ``untracked_fingerprints`` is what the caller
        persists after a successful sync. For remote repositories
        (``plan.fingerprint is None``) the plan is returned unchanged.

        Filesystem repositories get a further shortcut: their walk
        fingerprint already covers every file's content (path + mtime +
        size — see :meth:`GitManager._filesystem_fingerprint`), so once
        the repository has been indexed at least once, an unchanged
        fingerprint proves no file's content changed and the whole
        content-hashing loop below is skipped — no file is read. Working
        Git repositories cannot take this shortcut: their fingerprint is
        ``HEAD`` + porcelain status, which does not move when an
        existing untracked file is edited in place (see
        :meth:`GitManager.local_fingerprint`), so their untracked files
        must still be re-hashed every sync to catch that case.

        Only files :func:`classify_file` considers indexable are hashed;
        binaries and other unindexable files are never read.
        """
        if plan.fingerprint is None:
            return plan, {}
        state = self._states.get(cfg.name)
        stored = dict(state.untracked_indexed)
        if (
            cfg.filesystem
            and state.indexed_commit is not None
            and plan.fingerprint == state.local_fingerprint
        ):
            return plan, stored
        repo_dir = self._git.repo_dir(cfg)
        fingerprints: dict[str, str] = {}
        new_or_changed: list[str] = []
        for path in plan.untracked:
            if classify_file(path, cfg.enabled_collections, cfg.exclude) is None:
                continue
            digest = self._content_fingerprint(repo_dir / path)
            if digest is None:
                # Unreadable or gone mid-sync: not indexed this round (if it
                # was indexed before it is dropped via the stored diff).
                continue
            fingerprints[path] = digest
            if stored.get(path) != digest:
                new_or_changed.append(path)
        deleted_untracked = [p for p in stored if p not in fingerprints]
        if not new_or_changed and not deleted_untracked:
            return plan, fingerprints
        refined = dataclasses.replace(
            plan,
            added_or_modified=tuple(
                sorted(set(plan.added_or_modified) | set(new_or_changed))
            ),
            deleted=tuple(sorted(set(plan.deleted) | set(deleted_untracked))),
        )
        return refined, fingerprints

    @staticmethod
    def _content_fingerprint(path: Path) -> str | None:
        """sha256 of a file's content, or ``None`` if unreadable."""
        try:
            data = path.read_bytes()
        except OSError:
            return None
        return hashlib.sha256(data).hexdigest()

    def _apply_local_state(
        self,
        cfg: RepositoryConfig,
        plan: SyncPlan,
        untracked_fps: dict[str, str],
    ) -> None:
        """Record a local repo's working-tree fingerprint and untracked
        content fingerprints after a successful sync. The caller's
        subsequent state save makes it durable. No-op for remote repos."""
        if plan.fingerprint is None:
            return
        state = self._states.get(cfg.name)
        state.local_fingerprint = plan.fingerprint
        state.untracked_indexed = untracked_fps

    # -- plan application ----------------------------------------------------

    async def _apply_plan(self, cfg: RepositoryConfig, plan: SyncPlan) -> None:
        if plan.full:
            self._store.delete_repository(cfg.name)
        for prefix in plan.deleted_submodule_prefixes:
            self._store.delete_file_prefix(cfg.name, prefix)

        files_by_kind: dict[FileKind, list[str]] = {}
        for f in plan.added_or_modified:
            kind = classify_file(f, cfg.enabled_collections, cfg.exclude)
            if kind is None:
                continue
            files_by_kind.setdefault(kind, []).append(f)

        # Chunk IDs embed the line range, so a modified file's new chunks
        # get new IDs: remove deleted files' and modified files' previous
        # chunks in one batched call before upserting (full plans already
        # dropped the whole repository above).
        stale_files = list(plan.deleted)
        if not plan.full:
            for files in files_by_kind.values():
                stale_files.extend(files)
        if stale_files:
            self._store.delete_files(cfg.name, stale_files)

        chunks: list[Chunk] = []
        vhdl_files = files_by_kind.pop(VHDL_KIND, [])
        if vhdl_files:
            chunks.extend(await self._chunk_vhdl_files(cfg, plan, vhdl_files))
        verilog_files = files_by_kind.pop(VERILOG_KIND, [])
        systemverilog_files = files_by_kind.pop(SYSTEMVERILOG_KIND, [])
        if verilog_files or systemverilog_files:
            chunks.extend(
                await self._chunk_verilog_sv_files(
                    cfg, plan, verilog_files, systemverilog_files
                )
            )
        for kind, files in files_by_kind.items():
            for f in files:
                content = self._git.read_file(cfg, f)
                if kind.collection is CollectionName.DOCS:
                    chunks.extend(
                        chunk_doc_file(
                            cfg, f, content, plan.commit, kind.language, branch=plan.ref
                        )
                    )
                else:
                    chunks.extend(
                        chunk_code_file(
                            cfg, f, content, plan.commit, kind.language, branch=plan.ref
                        )
                    )

        if chunks:
            self._upsert(cfg, chunks)
            logger.info("%s: indexed %d chunks", cfg.name, len(chunks))

    async def _chunk_with_lsp(
        self,
        cfg: RepositoryConfig,
        files: list[str],
        lsp: LspClient,
        chunk_fn: Callable[[str, str, tuple[SymbolInfo, ...] | None], list[Chunk]],
    ) -> list[Chunk]:
        """Chunk ``files`` with one LSP session shared across all of them.

        Each file's content is read once (reused for both the LSP
        ``didOpen`` and the chunker) and all files are opened before the
        server is waited on once, so the quiet window is not paid per
        file. A file with a syntax error, or whose ``document_symbols``
        call fails/times out, falls back to ``chunk_fn``'s own structural
        parsing (``lsp_symbols=None``) rather than aborting the whole
        repository sync.
        """
        repo_dir = self._git.repo_dir(cfg)
        contents = {f: self._git.read_file(cfg, f) for f in files}
        chunks: list[Chunk] = []
        try:
            await lsp.start()
            for f in files:
                await lsp.open_document(repo_dir / f, text=contents[f])
            await lsp.wait_until_quiet(timeout=max(20.0, 2.0 * len(files)))
            for f in files:
                path: Path = repo_dir / f
                if lsp.has_syntax_error(path):
                    logger.info(
                        "%s: %s has syntax errors; using structural fallback",
                        cfg.name,
                        f,
                    )
                    symbols = None
                else:
                    try:
                        symbols = await lsp.document_symbols(path)
                    except Exception as exc:  # LspError/timeout, or any other
                        # per-file LSP crash: contain it to this file so one
                        # bad file cannot abort the whole repository sync.
                        logger.warning(
                            "%s: %s: document_symbols failed (%s); using "
                            "structural fallback",
                            cfg.name,
                            f,
                            exc,
                        )
                        symbols = None
                chunks.extend(chunk_fn(f, contents[f], symbols))
        finally:
            await lsp.shutdown()
        return chunks

    async def _chunk_vhdl_files(
        self, cfg: RepositoryConfig, plan: SyncPlan, files: list[str]
    ) -> list[Chunk]:
        """Chunk VHDL files with one LSP session for the whole plan.

        All files are opened first and the server is waited on once, so
        the quiet window is not paid per file. Files with syntax errors
        get the structural fallback (the LSP tree is partial there). When
        vhdl_ls is unavailable every file uses the structural fallback.
        """
        if self._vhdl_ls_bin is None:
            logger.info("%s: vhdl_ls unavailable; structural VHDL fallback", cfg.name)
            chunks: list[Chunk] = []
            for f in files:
                content = self._git.read_file(cfg, f)
                chunks.extend(
                    chunk_vhdl_file(
                        cfg,
                        f,
                        content,
                        plan.commit,
                        lsp_symbols=None,
                        branch=plan.ref,
                    )
                )
            return chunks
        lsp = VhdlLsp(
            self._vhdl_ls_bin,
            self._git.repo_dir(cfg),
            libraries_dir=self._vhdl_ls_libraries_dir,
            vhdl_ls_hook=cfg.vhdl_ls_hook,
            files=tuple(files),
        )

        def chunk_fn(
            f: str, content: str, symbols: tuple[SymbolInfo, ...] | None
        ) -> list[Chunk]:
            return chunk_vhdl_file(
                cfg, f, content, plan.commit, lsp_symbols=symbols, branch=plan.ref
            )

        return await self._chunk_with_lsp(cfg, files, lsp, chunk_fn)

    async def _chunk_verilog_sv_files(
        self,
        cfg: RepositoryConfig,
        plan: SyncPlan,
        verilog_files: list[str],
        systemverilog_files: list[str],
    ) -> list[Chunk]:
        """Chunk Verilog/SV files (one Veridian session for the whole plan).

        When Veridian is unavailable the generic tree-sitter parser is
        used per file (graceful fallback): the whole file becomes one
        chunk so no Verilog/SV is lost from the index.
        """
        if self._veridian_bin is None:
            logger.info(
                "%s: Veridian unavailable; generic parser fallback for "
                "%d Verilog/SV file(s)",
                cfg.name,
                len(verilog_files) + len(systemverilog_files),
            )
            chunks: list[Chunk] = []
            for f, language in [(f, "verilog") for f in verilog_files] + [
                (f, "systemverilog") for f in systemverilog_files
            ]:
                content = self._git.read_file(cfg, f)
                chunks.extend(
                    chunk_code_file(
                        cfg,
                        f,
                        content,
                        plan.commit,
                        language,
                        content_type=ContentType.SOURCE,
                        collection=CollectionName.HDL,
                        branch=plan.ref,
                    )
                )
            return chunks
        all_files = verilog_files + systemverilog_files
        language_by_file = {
            **dict.fromkeys(verilog_files, "verilog"),
            **dict.fromkeys(systemverilog_files, "systemverilog"),
        }
        lsp = VeridianLsp(
            self._veridian_bin, self._git.repo_dir(cfg), config_hook=cfg.veridian_hook
        )

        def chunk_fn(
            f: str, content: str, symbols: tuple[SymbolInfo, ...] | None
        ) -> list[Chunk]:
            return chunk_verilog_file(
                cfg,
                f,
                content,
                plan.commit,
                language_by_file[f],
                lsp_symbols=symbols,
                branch=plan.ref,
            )

        return await self._chunk_with_lsp(cfg, all_files, lsp, chunk_fn)

    #: Embed/upsert stream size: passages are embedded and upserted in
    #: groups this large, so the resident passage/vector buffers (and
    #: each durable store commit) stay bounded no matter how large the
    #: repository index is.
    _STREAM_CHUNK = 256

    def _upsert(self, cfg: RepositoryConfig, chunks: list[Chunk]) -> None:
        """Embed (dense per collection) and upsert, in bounded streams."""
        indexes_by_collection: dict[CollectionName, list[int]] = {}
        for i, chunk in enumerate(chunks):
            indexes_by_collection.setdefault(chunk.collection, []).append(i)
        for collection, indexes in indexes_by_collection.items():
            items = [chunks[i] for i in indexes]
            for start in range(0, len(items), self._STREAM_CHUNK):
                group = items[start : start + self._STREAM_CHUNK]
                dense = self._providers.embed_passages(
                    collection, [c.content for c in group]
                )
                self._store.upsert_chunks(group, dense)

    # -- bulk maintenance ------------------------------------------------------

    def delete_repository(self, name: str) -> None:
        """Remove a repository's chunks and state (config removal)."""
        deleted = self._store.delete_repository(name)
        self._states.remove(name)
        logger.info("%s: removed %d chunks from all collections", name, deleted)
