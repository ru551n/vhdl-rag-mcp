"""Persistent per-repository indexing state.

Repository sync/index state lives in the ``repositories`` table of the
same SQLite database as the vector index (``index.sqlite``), so all
persistent RAG state has one home and one crash boundary. State rows
are committed as they change; only commit state that reflects a fully
successful index update is persisted, and a failed index run leaves
the previous state untouched.

The table is created in the current layout at construction time; an
older database missing a column introduced since (e.g. ``submodules``)
is upgraded in place with an ``ALTER TABLE ... ADD COLUMN`` (see
:meth:`StateStore._add_missing_columns`) — existing rows keep their
data and simply get the new column's default value. Separately, there
is a one-time import of the legacy ``state/repositories.json`` document
from earlier deployments (see :meth:`StateStore.migrate`):

* a current (v2) document is imported as-is — its indexed commits
  remain valid;
* a legacy v1 document (a flat repository map, predating the hdl
  collection) tracks indexed commits that are invalid under the
  current index layout and are forgotten at import — the next sync
  rebuilds the index deterministically from git;
* a corrupt document is quarantined (``.corrupt``) and the store
  starts empty.

After the import the legacy document is renamed to
``repositories.json.migrated`` so the import runs exactly once;
rows already present in the table are never overwritten by an import.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .models import INDEX_SCHEMA_VERSION

logger = logging.getLogger(__name__)

#: Layout version of the ``repositories`` table (stamped in ``meta``).
STATE_SCHEMA_VERSION = 1

_STATE_VERSION_KEY = "state_schema_version"

_UPSERT_SQL = (
    "INSERT INTO repositories ("
    "name, indexed_commit, indexed_at, last_sync_at, last_sync_error, "
    "last_indexed_file_count, local_fingerprint, untracked_indexed, submodules"
    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
    "ON CONFLICT(name) DO UPDATE SET "
    "indexed_commit = excluded.indexed_commit, "
    "indexed_at = excluded.indexed_at, "
    "last_sync_at = excluded.last_sync_at, "
    "last_sync_error = excluded.last_sync_error, "
    "last_indexed_file_count = excluded.last_indexed_file_count, "
    "local_fingerprint = excluded.local_fingerprint, "
    "untracked_indexed = excluded.untracked_indexed, "
    "submodules = excluded.submodules"
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class RepositoryState(BaseModel):
    """Indexing/sync state for one repository."""

    name: str
    #: Last commit fully indexed (chunks embedded + upserted). None until
    #: the first successful index run.
    indexed_commit: str | None = None
    indexed_at: datetime | None = None
    last_sync_at: datetime | None = None
    last_sync_error: str | None = None
    #: Last full index run (clone or reindex).
    last_indexed_file_count: int = 0
    #: Local working and filesystem repositories only: cheap fingerprint
    #: (HEAD + porcelain status, or the filesystem walk) of the working
    #: tree at the last successful sync. The fast local poller compares
    #: its freshly computed fingerprint to this value to decide whether
    #: a sync is needed without running the full plan.
    local_fingerprint: str | None = None
    #: Local working and filesystem repositories only: content
    #: fingerprints (sha256) of untracked / walked files at the last
    #: successful sync, keyed by repository-relative path. Lets the sync
    #: plan skip re-chunking unchanged files and detect deleted ones.
    untracked_indexed: dict[str, str] = {}
    #: Gitlink (submodule) path -> submodule SHA last fully indexed.
    #: Lets an incremental sync diff each submodule against its
    #: previously indexed content.
    submodules: dict[str, str] = {}


class StateStore:
    """Repository state in the index database's ``repositories`` table."""

    def __init__(self, index_path: Path, legacy_json: Path | None = None) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(index_path))
        conn.row_factory = sqlite3.Row
        # WAL: the periodic sync writer never blocks concurrent queries
        # (same settings as the vector store's connection to the file).
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        self._conn = conn
        self._legacy_json = legacy_json
        # A pending legacy document (version, states), awaiting migrate().
        self._legacy: tuple[int, dict[str, RepositoryState]] | None = None
        self._schema_version = INDEX_SCHEMA_VERSION
        self._states: dict[str, RepositoryState] = {}
        self._ensure_table()
        self._load()
        self._detect_legacy()

    def close(self) -> None:
        self._conn.close()

    # -- table layout -------------------------------------------------------

    def _ensure_table(self) -> None:
        conn = self._conn
        conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS repositories ("
            " name TEXT PRIMARY KEY,"
            " indexed_commit TEXT,"
            " indexed_at TEXT,"
            " last_sync_at TEXT,"
            " last_sync_error TEXT,"
            " last_indexed_file_count INTEGER NOT NULL DEFAULT 0,"
            " local_fingerprint TEXT,"
            " untracked_indexed TEXT NOT NULL DEFAULT '{}',"
            " submodules TEXT NOT NULL DEFAULT '{}'"
            ")"
        )
        self._add_missing_columns()
        conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (_STATE_VERSION_KEY, str(STATE_SCHEMA_VERSION)),
        )
        conn.commit()

    def _add_missing_columns(self) -> None:
        """Add columns introduced after a database's table was created.

        ``CREATE TABLE IF NOT EXISTS`` above is a no-op against a
        database from an older deployment, so a newly introduced column
        (e.g. ``submodules``) is missing from it and must be added
        explicitly. Existing rows get the column's default value.
        """
        conn = self._conn
        existing = {
            row["name"] for row in conn.execute("PRAGMA table_info(repositories)")
        }
        if "submodules" not in existing:
            conn.execute(
                "ALTER TABLE repositories "
                "ADD COLUMN submodules TEXT NOT NULL DEFAULT '{}'"
            )

    # -- load / persist ------------------------------------------------------

    def _load(self) -> None:
        rows = self._conn.execute("SELECT * FROM repositories ORDER BY name").fetchall()
        for row in rows:
            state = self._row_to_state(row)
            if state is not None:
                self._states[state.name] = state

    @staticmethod
    def _row_to_state(row: sqlite3.Row) -> RepositoryState | None:
        try:
            untracked = json.loads(row["untracked_indexed"])
            if not isinstance(untracked, dict):
                raise ValueError("untracked_indexed is not an object")
            submodules = json.loads(row["submodules"])
            if not isinstance(submodules, dict):
                raise ValueError("submodules is not an object")
            return RepositoryState(
                name=row["name"],
                indexed_commit=row["indexed_commit"],
                indexed_at=_parse_dt(row["indexed_at"]),
                last_sync_at=_parse_dt(row["last_sync_at"]),
                last_sync_error=row["last_sync_error"],
                last_indexed_file_count=row["last_indexed_file_count"],
                local_fingerprint=row["local_fingerprint"],
                untracked_indexed=untracked,
                submodules=submodules,
            )
        except (ValueError, TypeError) as exc:
            logger.error("skipping corrupt state row %r: %s", row["name"], exc)
            return None

    @staticmethod
    def _state_params(state: RepositoryState) -> tuple[Any, ...]:
        return (
            state.name,
            state.indexed_commit,
            state.indexed_at.isoformat() if state.indexed_at else None,
            state.last_sync_at.isoformat() if state.last_sync_at else None,
            state.last_sync_error,
            state.last_indexed_file_count,
            state.local_fingerprint,
            json.dumps(state.untracked_indexed),
            json.dumps(state.submodules),
        )

    def _upsert(self, state: RepositoryState) -> None:
        self._conn.execute(_UPSERT_SQL, self._state_params(state))
        self._conn.commit()

    def save(self) -> None:
        """Persist every cached state row (one commit)."""
        for state in self._states.values():
            self._conn.execute(_UPSERT_SQL, self._state_params(state))
        self._conn.commit()

    # -- accessors ------------------------------------------------------------

    def get(self, name: str) -> RepositoryState:
        """Return (creating on first access) the in-memory state for
        ``name``.

        Side effect: an unknown ``name`` is inserted as a fresh default
        ``RepositoryState`` — not persisted to disk by ``get()`` itself,
        but from then on it shows up in ``all()`` (e.g. in
        ``repository_status()`` / ``drop_unconfigured_repositories()``).
        This is relied on by ``set_indexed``/``record_sync``, which need
        ``get()`` to create-and-return the same object they then mutate.
        Safe as long as every caller only ever passes an already
        validated name — a configured ``[[repositories]]`` entry, the
        coding-standards pseudo-repository when configured, or a name
        just resolved from one of those — which holds for every call
        site in this codebase (each either iterates configured
        repositories/state directly, or validates the name first, e.g.
        ``RetrievalService._repository``, before it can reach here).
        """
        if name not in self._states:
            self._states[name] = RepositoryState(name=name)
        return self._states[name]

    def all(self) -> list[RepositoryState]:
        return [self._states[name] for name in sorted(self._states)]

    def remove(self, name: str) -> None:
        """Forget one repository's state (config removal)."""
        if name in self._states:
            del self._states[name]
        self._conn.execute("DELETE FROM repositories WHERE name = ?", (name,))
        self._conn.commit()

    @property
    def schema_version(self) -> int:
        """Index layout version the persisted state targets."""
        return self._schema_version

    @property
    def needs_migration(self) -> bool:
        """True while a legacy state document awaits import."""
        return self._legacy is not None

    def reset_all_indexed(self) -> None:
        """Forget every repository's indexed commit: the next sync
        reindexes each one fully and deterministically."""
        for state in self._states.values():
            state.indexed_commit = None
            state.indexed_at = None
        self.save()

    # -- legacy JSON import ---------------------------------------------------

    def _detect_legacy(self) -> None:
        if self._legacy_json is None or not self._legacy_json.exists():
            return
        try:
            raw: dict[str, Any] = json.loads(
                self._legacy_json.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(
                "legacy state file %s is unreadable (%s); quarantining",
                self._legacy_json,
                exc,
            )
            self._quarantine(self._legacy_json)
            return
        if "schema_version" in raw:
            version = raw.get("schema_version")
            repositories: Any = raw.get("repositories") or {}
        else:
            # v1 layout: a flat name -> state document (pre-hdl).
            version = 1
            repositories = raw
        if not isinstance(version, int):
            version = 1
        if not isinstance(repositories, dict):
            repositories = {}
        if version > INDEX_SCHEMA_VERSION:
            logger.warning(
                "legacy state %s was written by a newer schema (v%d); "
                "importing its rows as-is",
                self._legacy_json,
                version,
            )
        states: dict[str, RepositoryState] = {}
        for name, data in repositories.items():
            try:
                states[name] = RepositoryState.model_validate(data)
            except Exception as exc:
                logger.error("skipping corrupt legacy state %r: %s", name, exc)
        self._legacy = (version, states)

    def migrate(self) -> bool:
        """Import a legacy ``repositories.json`` document into the table.

        A v1 document predates the hdl collection layout, so the
        indexed commits it tracks are invalid and are forgotten; the
        next sync rebuilds the index deterministically from git. A
        current (v2) document is imported as-is. Rows already present
        in the table win over the imported document. Returns True when
        an import ran; idempotent on a current deployment.
        """
        if self._legacy is None:
            return False
        version, states = self._legacy
        self._legacy = None
        if version < 2:
            for state in states.values():
                state.indexed_commit = None
                state.indexed_at = None
        imported = 0
        for state in states.values():
            if state.name in self._states:
                logger.warning(
                    "skipping state import for %r: a row already exists",
                    state.name,
                )
                continue
            self._upsert(state)
            self._states[state.name] = state
            imported += 1
        self._mark_imported()
        logger.info(
            "imported %d repository states from legacy document (v%d)",
            imported,
            version,
        )
        return True

    @staticmethod
    def _quarantine(path: Path) -> None:
        with contextlib.suppress(OSError):
            path.rename(path.with_suffix(".corrupt"))

    def _mark_imported(self) -> None:
        if self._legacy_json is not None:
            target = self._legacy_json.with_name(self._legacy_json.name + ".migrated")
            with contextlib.suppress(OSError):
                self._legacy_json.rename(target)

    # -- mutators --------------------------------------------------------------

    def set_indexed(
        self,
        name: str,
        commit: str,
        file_count: int = 0,
        submodules: dict[str, str] | None = None,
    ) -> None:
        """Mark a commit as fully indexed. Call only after the index update
        (embeddings, upserts, deletions) has succeeded. ``submodules``
        replaces the persisted per-gitlink SHA map when given; ``None``
        leaves the stored map untouched."""
        state = self.get(name)
        state.indexed_commit = commit
        state.indexed_at = _utcnow()
        state.last_indexed_file_count = file_count
        if submodules is not None:
            state.submodules = dict(submodules)
        self._upsert(state)

    def record_sync(self, name: str, error: str | None) -> None:
        state = self.get(name)
        state.last_sync_at = _utcnow()
        state.last_sync_error = error
        self._upsert(state)

    def set_local_fingerprint(self, name: str, fingerprint: str) -> None:
        """Persist a cheap local fingerprint (mtime/size or working-tree
        status) without touching the other sync/index fields. Used by
        pseudo-repositories (e.g. the coding-standards file) that have no
        git plan of their own to carry it alongside."""
        state = self.get(name)
        state.local_fingerprint = fingerprint
        self._upsert(state)
