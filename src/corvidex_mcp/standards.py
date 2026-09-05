"""Coding standards: one golden document, indexed as its own repository.

``AppConfig.coding_standards`` points at a single file (txt, md, rst,
pdf, or docx) holding the organization's coding standards. It is
indexed as the pseudo-repository ``coding-standards`` into the docs
collection, so every retrieval feature — repository filters,
cross-referencing, search modes, source access — applies to it, and
``coding_standards_priority`` (a high default) makes it the
authoritative source within its relevance tier.

The file is re-checked on every sync cycle: its content hash is the
"commit" it is indexed at, so an edited file is re-chunked and
re-embedded on the next sync, and removing the option (or the file)
drops the chunks again.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from .config import (
    CODING_STANDARDS_REPO,
    STANDARDS_EXTENSIONS,
    AppConfig,
    RepositoryConfig,
)
from .embeddings.providers import EmbeddingProviders
from .indexing.docs import chunk_doc_file
from .models import Chunk, CollectionName
from .state import StateStore
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

#: Extracted-text language per extension (md/rst keep their structure;
#: everything else is plain text — paragraph chunking).
STANDARDS_LANGUAGES: dict[str, str] = {
    ".txt": "text",
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "restructuredtext",
    ".pdf": "text",
    ".docx": "text",
}

#: WordprocessingML namespace (Office Open XML document part).
_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class StandardsError(RuntimeError):
    """The standards file cannot be read or extracted."""


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover (dependency)
        raise StandardsError(
            "pypdf is required to index a PDF standards file (pip install pypdf)"
        ) from exc
    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise StandardsError(f"cannot extract PDF text from {path}: {exc}") from exc
    return "\n\n".join(pages)


def _extract_docx(path: Path) -> str:
    import xml.etree.ElementTree as ET
    import zipfile

    try:
        with zipfile.ZipFile(path) as zf:
            xml_bytes = zf.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile, OSError) as exc:
        raise StandardsError(f"cannot read DOCX document from {path}: {exc}") from exc
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise StandardsError(f"cannot parse DOCX document.xml: {exc}") from exc
    paragraphs: list[str] = []
    for para in root.iter(f"{_W_NS}p"):
        text = "".join(node.text or "" for node in para.iter(f"{_W_NS}t"))
        if text.strip():
            paragraphs.append(text)
    return "\n".join(paragraphs)


def extract_standards_text(path: Path) -> str:
    """Extract plain text from a standards file (any supported type).

    Text formats are read as-is (UTF-8, replacement on invalid bytes);
    PDF and DOCX are parsed to their text content. Raises
    :class:`StandardsError` when the file is missing or unreadable.
    """
    path = Path(path).expanduser()
    suffix = path.suffix.lower()
    if suffix not in STANDARDS_EXTENSIONS:
        raise StandardsError(
            "standards file must be one of "
            f"{', '.join(sorted(STANDARDS_EXTENSIONS))}; got "
            f"{path.suffix or 'no extension'!r}"
        )
    if not path.is_file():
        raise StandardsError(f"standards file not found: {path}")
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".docx":
        return _extract_docx(path)
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise StandardsError(f"cannot read standards file {path}: {exc}") from exc


def standards_hash(text: str) -> str:
    """Content hash of the extracted text (the pseudo-repository's
    "commit": an edit changes it and triggers a reindex)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_standards_file(path: Path, text: str, digest: str) -> list[Chunk]:
    """Chunk the extracted standards text (docs chunking, one
    pseudo-repository)."""
    suffix = path.suffix.lower()
    language = STANDARDS_LANGUAGES[suffix]
    cfg = RepositoryConfig(name=CODING_STANDARDS_REPO, path=path)
    return chunk_doc_file(cfg, path.name, text, digest, language, branch=digest)


def _remove(store: VectorStore, states: StateStore) -> None:
    deleted = store.delete_repository(CODING_STANDARDS_REPO)
    states.remove(CODING_STANDARDS_REPO)
    if deleted:
        logger.info("%s: removed %d chunks", CODING_STANDARDS_REPO, deleted)


def _local_fingerprint(path: Path) -> str | None:
    """Cheap mtime+size fingerprint of the standards file, no read/parse.

    Same idea as :meth:`GitManager.local_fingerprint`'s filesystem walk
    fingerprint: a fast, read-only check for "did anything on disk
    change" that a full extract-and-hash can be skipped for. ``None``
    when the file cannot be stat'ed (e.g. it was deleted since the last
    successful sync); the caller falls through to the normal
    extract-and-fail path in that case rather than treating a vanished
    file as "unchanged".
    """
    try:
        st = path.stat()
    except OSError:
        return None
    return hashlib.sha256(f"{st.st_mtime_ns}\t{st.st_size}".encode()).hexdigest()


def sync_coding_standards(
    config: AppConfig,
    providers: EmbeddingProviders,
    store: VectorStore,
    states: StateStore,
) -> dict[str, str] | None:
    """Index the configured standards file (no-op when unchanged).

    Returns a report like the per-repository sync reports (``status``
    ``ok`` / ``up-to-date`` / ``error``), or ``None`` when no standards
    file is configured (stale chunks/state from a removed option are
    dropped silently). Failures are contained: an error is recorded in
    the state store and reported, never raised.
    """
    path = config.coding_standards
    if path is None:
        _remove(store, states)
        return None
    name = CODING_STANDARDS_REPO
    cheap_fingerprint = _local_fingerprint(path)
    state = states.get(name)
    if (
        cheap_fingerprint is not None
        and cheap_fingerprint == state.local_fingerprint
        and state.indexed_commit is not None
    ):
        # mtime/size unchanged since the last successful sync: the
        # extracted text (a possibly expensive PDF/DOCX re-parse) cannot
        # have changed either, so skip straight to "up-to-date".
        states.record_sync(name, None)
        return {
            "repository": name,
            "status": "up-to-date",
            "commit": state.indexed_commit[:12],
        }
    try:
        text = extract_standards_text(path)
    except StandardsError as exc:
        logger.error("%s: %s", name, exc)
        states.record_sync(name, str(exc))
        return {"repository": name, "status": "error", "error": str(exc)}
    digest = standards_hash(text)
    if states.get(name).indexed_commit == digest:
        if cheap_fingerprint is not None:
            states.set_local_fingerprint(name, cheap_fingerprint)
        states.record_sync(name, None)
        return {
            "repository": name,
            "status": "up-to-date",
            "commit": digest[:12],
        }
    chunks = chunk_standards_file(path, text, digest)
    try:
        _remove(store, states)  # previous version (replace, not merge)
        if chunks:
            dense = providers.embed_passages(
                CollectionName.DOCS, [c.content for c in chunks]
            )
            store.upsert_chunks(chunks, dense)
    except Exception as exc:
        logger.exception("%s: index update failed (%s); previous state kept", name, exc)
        states.record_sync(name, str(exc))
        return {"repository": name, "status": "error", "error": str(exc)}
    states.set_indexed(name, digest, file_count=len(chunks))
    if cheap_fingerprint is not None:
        states.set_local_fingerprint(name, cheap_fingerprint)
    states.record_sync(name, None)
    logger.info("%s: indexed %d chunks at %s", name, len(chunks), digest[:12])
    return {
        "repository": name,
        "status": "ok",
        "commit": digest[:12],
        "chunks": str(len(chunks)),
    }
