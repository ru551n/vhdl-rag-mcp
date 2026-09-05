"""Tests for the coding-standards file (pseudo-repository indexing).

Runs fully offline: fake embedding providers, a real SQLite store and
state, and standards files (md, docx, pdf) built in the test.
"""

from __future__ import annotations

import os
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest
from capability import sqlite_extensions_supported
from pydantic import ValidationError

from corvidex_mcp.config import (
    CODING_STANDARDS_REPO,
    AppConfig,
    RepositoryConfig,
)
from corvidex_mcp.embeddings.provider import FastEmbedProvider
from corvidex_mcp.embeddings.providers import EmbeddingProviders
from corvidex_mcp.git_manager import GitManager
from corvidex_mcp.models import CollectionName
from corvidex_mcp.retrieval import RetrievalError, RetrievalService
from corvidex_mcp.selfcheck import check_coding_standards
from corvidex_mcp.server import VhdlRagApp
from corvidex_mcp.standards import (
    STANDARDS_LANGUAGES,
    StandardsError,
    chunk_standards_file,
    extract_standards_text,
    standards_hash,
    sync_coding_standards,
)
from corvidex_mcp.state import StateStore
from corvidex_mcp.vector_store import VectorStore

pytestmark = pytest.mark.skipif(
    not sqlite_extensions_supported(),
    reason=(
        "stdlib SQLite lacks loadable-extension support (the sqlite-vec "
        "extension cannot load; use CPython 3.14 or a system/homebrew "
        "Python)"
    ),
)


ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
}

STANDARD_MD = (
    "# Coding standards\n\n"
    "## Reset conventions\n\n"
    "Asynchronous resets are active-low and named `rst_n`.\n\n"
    "```\n"
    "dout <= '0' when rst_n = '0';\n"
    "```\n\n"
    "## Naming\n\n"
    "Signals are lowercase with underscores.\n"
)
STANDARD_MD_V2 = (
    "# Coding standards\n\n"
    "## Reset conventions\n\n"
    "Asynchronous resets are active-low and named `rst_n`.\n"
)

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def make_docx(path: Path, paragraphs: list[str]) -> None:
    uri = _W.strip("{}")
    body = "".join(
        f'<w:p><w:t xml:space="preserve">{p}</w:t></w:p>' for p in paragraphs
    )
    doc = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{uri}"><w:body>{body}</w:body></w:document>'
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/'
            '2006/content-types">'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" ContentType="'
            "application/vnd.openxmlformats-officedocument.wordprocessingml"
            '.document.main+xml"/></Types>',
        )
        zf.writestr("word/document.xml", doc)


def make_pdf(path: Path, lines: list[str]) -> None:
    """A minimal single-page PDF with the given text lines (exact xref
    offsets, so it parses cleanly)."""
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    text_ops = b"".join(
        f"BT /F1 12 Tf 72 {720 - i * 16} Td ({line}) Tj ET\n".encode()
        for i, line in enumerate(lines)
    )
    objects[3] = (
        b"<< /Length "
        + str(len(text_ops)).encode()
        + b" >>\nstream\n"
        + text_ops
        + b"endstream"
    )
    out = b"%PDF-1.4\n"
    offsets: list[int] = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n"
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode()
        + b" /Root 1 0 R >>\nstartxref\n"
        + str(xref_pos).encode()
        + b"\n%%EOF"
    )
    path.write_bytes(out)


# -- config -------------------------------------------------------------------


def test_coding_standards_config_defaults() -> None:
    cfg = AppConfig()
    assert cfg.coding_standards is None
    assert cfg.coding_standards_priority == 10


def test_coding_standards_config_valid_extensions(tmp_path: Path) -> None:
    for ext in (".txt", ".md", ".rst", ".pdf", ".docx"):
        path = tmp_path / f"standards{ext}"
        cfg = AppConfig(coding_standards=path)
        assert cfg.coding_standards == path


def test_coding_standards_config_bad_extension(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="txt, md, rst, pdf, or docx"):
        AppConfig(coding_standards=tmp_path / "standards.py")
    with pytest.raises(ValidationError, match="txt, md, rst, pdf, or docx"):
        AppConfig(coding_standards=tmp_path / "standards")


def test_coding_standards_config_priority_bounds(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        AppConfig(coding_standards=tmp_path / "s.md", coding_standards_priority=0)
    cfg = AppConfig(coding_standards=tmp_path / "s.md", coding_standards_priority=3)
    assert cfg.coding_standards_priority == 3


def test_coding_standards_repository_name_reserved() -> None:
    with pytest.raises(ValidationError, match="reserved"):
        AppConfig(
            repositories=[RepositoryConfig(name=CODING_STANDARDS_REPO, path="~/x")]
        )


# -- extraction ---------------------------------------------------------------


def test_extract_text_formats(tmp_path: Path) -> None:
    for ext in (".txt", ".md", ".rst"):
        path = tmp_path / f"standards{ext}"
        path.write_text(STANDARD_MD, encoding="utf-8")
        assert extract_standards_text(path) == STANDARD_MD


def test_extract_docx(tmp_path: Path) -> None:
    path = tmp_path / "standards.docx"
    make_docx(path, ["Reset conventions", "Resets are named rst_n."])
    text = extract_standards_text(path)
    assert "Reset conventions" in text
    assert "Resets are named rst_n." in text


def test_extract_pdf(tmp_path: Path) -> None:
    path = tmp_path / "standards.pdf"
    make_pdf(path, ["Coding standards", "Resets are named rst_n."])
    text = extract_standards_text(path)
    assert "Coding standards" in text
    assert "Resets are named rst_n." in text


def test_extract_missing_file(tmp_path: Path) -> None:
    with pytest.raises(StandardsError, match="not found"):
        extract_standards_text(tmp_path / "nope.md")


def test_extract_bad_extension(tmp_path: Path) -> None:
    path = tmp_path / "standards.py"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(StandardsError, match="must be one of"):
        extract_standards_text(path)


def test_extract_corrupt_docx(tmp_path: Path) -> None:
    path = tmp_path / "standards.docx"
    path.write_bytes(b"not a zip at all")
    with pytest.raises(StandardsError, match="cannot read DOCX"):
        extract_standards_text(path)


def test_standards_hash_changes_with_content() -> None:
    assert standards_hash("a") != standards_hash("b")
    assert standards_hash("a") == standards_hash("a")


# -- chunking -----------------------------------------------------------------


def test_chunk_standards_md() -> None:
    chunks = chunk_standards_file(Path("/std/standards.md"), STANDARD_MD, "digest123")
    assert chunks
    for chunk in chunks:
        assert chunk.repository == CODING_STANDARDS_REPO
        assert chunk.commit == "digest123"
        assert chunk.branch == "digest123"
        assert chunk.collection is CollectionName.DOCS
        assert chunk.file == "standards.md"
    headings = [c.heading for c in chunks]
    assert any("Reset conventions" in h for h in headings)
    reset = next(c for c in chunks if "Reset conventions" in c.heading)
    assert "rst_n" in reset.symbols


def test_chunk_standards_docx_language_paragraph() -> None:
    assert STANDARDS_LANGUAGES[".docx"] == "text"
    text = (
        "Paragraph one is long enough to stand on its own in the index.\n\n"
        "Paragraph two is also long enough and references rst_n.\n"
    )
    chunks = chunk_standards_file(Path("/std/s.docx"), text, "digest123")
    assert len(chunks) == 2
    assert all(c.language == "text" for c in chunks)


# -- fixtures -----------------------------------------------------------------


class FakeDense:
    embedding_size = 4

    def passage_embed(self, texts, batch_size=32):
        for i, text in enumerate(texts):
            yield np.array([float(len(text)), float(i), 0.0, 0.0], dtype=np.float32)

    def query_embed(self, query, batch_size=32):
        yield np.array([float(len(query)), 0.0, 0.0, 0.0], dtype=np.float32)


def make_config(tmp_path: Path, standards: Path | None) -> AppConfig:
    return AppConfig(
        data_dir=tmp_path / "data",
        vhdl_ls_path="/nonexistent/vhdl_ls",
        veridian_path="/nonexistent/veridian",
        log_level="WARNING",
        coding_standards=standards,
    )


def make_app(config: AppConfig) -> VhdlRagApp:
    providers = EmbeddingProviders(config)
    dense = FastEmbedProvider("fake/dense", dense=FakeDense())
    providers._dense_provider = lambda _collection: dense  # type: ignore[method-assign]
    app = VhdlRagApp(config, providers=providers)
    app.ensure_collections()
    return app


def make_git_repo(path: Path) -> None:
    path.mkdir()
    (path / "docs").mkdir()
    (path / "docs" / "d.md").write_text("# D\n\nBody.\n", encoding="utf-8")
    for args in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "add", "-A"],
        ["git", "commit", "-qm", "one"],
    ):
        subprocess.run(args, cwd=path, env=ENV, capture_output=True, check=True)


# -- sync ---------------------------------------------------------------------


def test_sync_standards_full_lifecycle(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    std = tmp_path / "standards.md"
    std.write_text(STANDARD_MD, encoding="utf-8")
    config = make_config(tmp_path, std)
    app = make_app(config)
    try:
        # First sync: indexed.
        report = sync_coding_standards(config, app.providers, app.store, app.states)
        assert report is not None
        assert report["status"] == "ok", report
        assert report["repository"] == CODING_STANDARDS_REPO
        assert app.store.count_repository(CODING_STANDARDS_REPO) > 0

        # Unchanged sync: up-to-date.
        report = sync_coding_standards(config, app.providers, app.store, app.states)
        assert report is not None
        assert report["status"] == "up-to-date", report

        # Edited file: reindexed at a new digest.
        first_commit = app.states.get(CODING_STANDARDS_REPO).indexed_commit
        std.write_text(STANDARD_MD_V2, encoding="utf-8")
        report = sync_coding_standards(config, app.providers, app.store, app.states)
        assert report is not None
        assert report["status"] == "ok", report
        new_commit = app.states.get(CODING_STANDARDS_REPO).indexed_commit
        assert new_commit and new_commit != first_commit
        # The removed section is gone from the index.
        results = app.retrieval.search(
            CollectionName.DOCS,
            "naming",
            limit=10,
            repository=CODING_STANDARDS_REPO,
        )
        assert all("Naming" not in r.heading for r in results), [
            r.heading for r in results
        ]

        # Option removed: chunks and state are dropped.
        config_none = make_config(tmp_path, None)
        assert (
            sync_coding_standards(config_none, app.providers, app.store, app.states)
            is None
        )
        assert app.store.count_repository(CODING_STANDARDS_REPO) == 0
        assert CODING_STANDARDS_REPO not in {s.name for s in app.states.all()}
    finally:
        app.close()


def test_sync_standards_cheap_fingerprint_skips_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mtime+size fingerprint short-circuits re-extraction/re-hashing
    of an unchanged file, but still detects a real content change, and a
    same-content mtime bump still lands on "up-to-date" (via the
    pre-existing digest dedup, one re-extraction later)."""
    import corvidex_mcp.standards as standards_mod

    data = tmp_path / "data"
    data.mkdir()
    std = tmp_path / "standards.md"
    std.write_text(STANDARD_MD, encoding="utf-8")
    config = make_config(tmp_path, std)
    app = make_app(config)
    calls = 0
    real_extract = standards_mod.extract_standards_text

    def spy_extract(path: Path) -> str:
        nonlocal calls
        calls += 1
        return real_extract(path)

    monkeypatch.setattr(standards_mod, "extract_standards_text", spy_extract)
    try:
        report = sync_coding_standards(config, app.providers, app.store, app.states)
        assert report is not None and report["status"] == "ok", report
        assert calls == 1
        fp_after_first = app.states.get(CODING_STANDARDS_REPO).local_fingerprint
        assert fp_after_first is not None

        # (a) Unchanged file: the cheap fingerprint alone short-circuits
        # before extract_standards_text is even called again.
        report = sync_coding_standards(config, app.providers, app.store, app.states)
        assert report is not None and report["status"] == "up-to-date", report
        assert calls == 1

        # (b) Touch mtime without changing content: the cheap fingerprint
        # changes, triggering one re-extraction, but the content-hash
        # dedup still reports up-to-date (no reindex).
        first_commit = app.states.get(CODING_STANDARDS_REPO).indexed_commit
        os.utime(std, ns=(std.stat().st_atime_ns + 5_000_000_000,) * 2)
        report = sync_coding_standards(config, app.providers, app.store, app.states)
        assert report is not None and report["status"] == "up-to-date", report
        assert calls == 2
        assert app.states.get(CODING_STANDARDS_REPO).indexed_commit == first_commit
        assert app.states.get(CODING_STANDARDS_REPO).local_fingerprint != fp_after_first

        # (c) An actual content change is still detected and reindexed.
        std.write_text(STANDARD_MD_V2, encoding="utf-8")
        report = sync_coding_standards(config, app.providers, app.store, app.states)
        assert report is not None and report["status"] == "ok", report
        assert calls == 3
        new_commit = app.states.get(CODING_STANDARDS_REPO).indexed_commit
        assert new_commit and new_commit != first_commit
    finally:
        app.close()


def test_sync_standards_missing_file_reports_error(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    config = make_config(tmp_path, tmp_path / "missing.md")
    app = make_app(config)
    try:
        report = sync_coding_standards(config, app.providers, app.store, app.states)
        assert report is not None
        assert report["status"] == "error", report
        assert "not found" in report["error"]
        assert app.states.get(CODING_STANDARDS_REPO).last_sync_error is not None
    finally:
        app.close()


async def test_sync_all_includes_standards_report(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    make_git_repo(tmp_path / "repo")
    std = tmp_path / "standards.md"
    std.write_text(STANDARD_MD, encoding="utf-8")
    config = AppConfig(
        data_dir=data,
        vhdl_ls_path="/nonexistent/vhdl_ls",
        veridian_path="/nonexistent/veridian",
        log_level="WARNING",
        coding_standards=std,
        repositories=[RepositoryConfig(name="repo", path=tmp_path / "repo")],
    )
    providers = EmbeddingProviders(config)
    dense = FastEmbedProvider("fake/dense", dense=FakeDense())
    providers._dense_provider = lambda _collection: dense  # type: ignore[method-assign]
    app = VhdlRagApp(config, providers=providers)
    app.ensure_collections()
    try:
        reports = await app.sync_all()
        by_name = {r["repository"]: r for r in reports}
        assert by_name["repo"]["status"] == "ok", reports
        assert by_name[CODING_STANDARDS_REPO]["status"] == "ok", reports
        # Selecting repositories explicitly includes the pseudo-repo.
        reports = await app.sync_all([CODING_STANDARDS_REPO])
        assert [r["repository"] for r in reports] == [CODING_STANDARDS_REPO]
    finally:
        app.close()


# -- retrieval ----------------------------------------------------------------


def test_repository_filter_accepts_standards_when_configured(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    std = tmp_path / "standards.md"
    std.write_text(STANDARD_MD, encoding="utf-8")

    def service_for(standards: Path | None) -> RetrievalService:
        config = make_config(tmp_path, standards)
        return RetrievalService(
            config,
            GitManager(config.repos_dir),
            VectorStore(config),
            EmbeddingProviders(config),
            StateStore(config.sqlite_index_path),
        )

    # Valid when configured...
    service_for(std)._repository(CODING_STANDARDS_REPO)
    # ...and rejected when not.
    with pytest.raises(RetrievalError, match="unknown repository"):
        service_for(None)._repository(CODING_STANDARDS_REPO)


def test_priority_bonus_standards_vs_repository(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    std = tmp_path / "standards.md"
    std.write_text(STANDARD_MD, encoding="utf-8")
    make_git_repo(tmp_path / "repo")
    config = AppConfig(
        data_dir=data,
        vhdl_ls_path="/nonexistent/vhdl_ls",
        veridian_path="/nonexistent/veridian",
        log_level="WARNING",
        coding_standards=std,
        repositories=[
            RepositoryConfig(name="repo", path=tmp_path / "repo", priority=5),
            RepositoryConfig(name="repo2", path=tmp_path / "repo", priority=1),
        ],
    )
    service = RetrievalService(
        config,
        GitManager(config.repos_dir),
        VectorStore(config),
        EmbeddingProviders(config),
        StateStore(config.sqlite_index_path),
    )
    # Standards (default priority 10) outrank a boosted repository
    # (priority 5); a priority-1 repository gets no bonus at all.
    standards_bonus = service._priority_bonus(CODING_STANDARDS_REPO)
    repo_bonus = service._priority_bonus("repo")
    neutral_bonus = service._priority_bonus("repo2")
    assert standards_bonus > repo_bonus > 0.0
    assert neutral_bonus == 0.0
    # Unconfigured standards: no bonus.
    config_none = make_config(tmp_path, None)
    service_none = RetrievalService(
        config_none,
        GitManager(config_none.repos_dir),
        VectorStore(config_none),
        EmbeddingProviders(config_none),
        StateStore(config_none.sqlite_index_path),
    )
    assert service_none._priority_bonus(CODING_STANDARDS_REPO) == 0.0


# -- self-check -----------------------------------------------------------------


def test_check_coding_standards(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    std = tmp_path / "standards.md"
    std.write_text(STANDARD_MD, encoding="utf-8")
    config = make_config(tmp_path, std)
    app = make_app(config)
    try:
        status = check_coding_standards(app)
        assert status is not None
        assert status.ok
        assert status.optional
    finally:
        app.close()

    config_missing = make_config(tmp_path, tmp_path / "gone.md")
    app2 = make_app(config_missing)
    try:
        status = check_coding_standards(app2)
        assert status is not None
        assert not status.ok
        assert status.optional
    finally:
        app2.close()

    config_none = make_config(tmp_path, None)
    app3 = make_app(config_none)
    try:
        assert check_coding_standards(app3) is None
    finally:
        app3.close()


# -- integration fixes ------------------------------------------------------------


def test_drop_unconfigured_keeps_standards(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    std = tmp_path / "standards.md"
    std.write_text(STANDARD_MD, encoding="utf-8")
    config = make_config(tmp_path, std)
    app = make_app(config)
    try:
        sync_coding_standards(config, app.providers, app.store, app.states)
        assert app.store.count_repository(CODING_STANDARDS_REPO) > 0
        # A startup drop must not treat the pseudo-repository as
        # unconfigured while the option is set.
        dropped = app.drop_unconfigured_repositories()
        assert dropped == []
        assert app.store.count_repository(CODING_STANDARDS_REPO) > 0
        assert CODING_STANDARDS_REPO in {st.name for st in app.states.all()}
    finally:
        app.close()


def test_drop_unconfigured_drops_stale_standards(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    std = tmp_path / "standards.md"
    std.write_text(STANDARD_MD, encoding="utf-8")
    config = make_config(tmp_path, std)
    app = make_app(config)
    sync_coding_standards(config, app.providers, app.store, app.states)
    app.close()
    # Option removed from the config: the next startup drops the chunks.
    config_none = make_config(tmp_path, None)
    app2 = make_app(config_none)
    try:
        dropped = app2.drop_unconfigured_repositories()
        assert dropped == [CODING_STANDARDS_REPO]
        assert app2.store.count_repository(CODING_STANDARDS_REPO) == 0
        assert CODING_STANDARDS_REPO not in {st.name for st in app2.states.all()}
    finally:
        app2.close()


def test_get_source_standards(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    std = tmp_path / "standards.md"
    std.write_text(STANDARD_MD, encoding="utf-8")
    config = make_config(tmp_path, std)
    app = make_app(config)
    try:
        sync_coding_standards(config, app.providers, app.store, app.states)
        out = app.retrieval.get_source(CODING_STANDARDS_REPO, std.name)
        assert "coding-standards:standards.md @" in out
        assert "rst_n" in out
        # Line slicing works on the indexed extraction.
        sliced = app.retrieval.get_source(
            CODING_STANDARDS_REPO, std.name, start_line=2, end_line=2
        )
        assert "(lines 2-2" in sliced
        # A different file name is rejected...
        with pytest.raises(RetrievalError, match="single file"):
            app.retrieval.get_source(CODING_STANDARDS_REPO, "other.md")
        # ...as is an invalid range.
        with pytest.raises(RetrievalError, match="invalid line range"):
            app.retrieval.get_source(
                CODING_STANDARDS_REPO, std.name, start_line=9, end_line=2
            )
        # Unconfigured: rejected as unknown.
        config_none = make_config(tmp_path, None)
        service_none = RetrievalService(
            config_none,
            GitManager(config_none.repos_dir),
            VectorStore(config_none),
            EmbeddingProviders(config_none),
            StateStore(config_none.sqlite_index_path),
        )
        with pytest.raises(RetrievalError, match="unknown repository"):
            service_none.get_source(CODING_STANDARDS_REPO, std.name)
    finally:
        app.close()


def test_render_report_up_to_date() -> None:
    from corvidex_mcp.server import _render_report

    out = _render_report(
        [
            {
                "repository": CODING_STANDARDS_REPO,
                "status": "up-to-date",
                "commit": "abcdef1234567890",
            }
        ]
    )
    assert "ok (unchanged)" in out
    assert "commit abcdef1234" in out
    assert "ERROR" not in out


async def test_reindex_standards(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    std = tmp_path / "standards.md"
    std.write_text(STANDARD_MD, encoding="utf-8")
    config = make_config(tmp_path, std)
    app = make_app(config)
    try:
        report = await app.reindex(CODING_STANDARDS_REPO)
        assert report["status"] == "ok", report
        # Unchanged reindex is an up-to-date no-op.
        report = await app.reindex(CODING_STANDARDS_REPO)
        assert report["status"] == "up-to-date", report
        # Edited file: reindex picks up the new content.
        std.write_text(STANDARD_MD_V2, encoding="utf-8")
        report = await app.reindex(CODING_STANDARDS_REPO)
        assert report["status"] == "ok", report
        # Unconfigured: a clear error.
        config_none = make_config(tmp_path, None)
        app2 = make_app(config_none)
        try:
            with pytest.raises(RetrievalError, match="no coding_standards"):
                await app2.reindex(CODING_STANDARDS_REPO)
        finally:
            app2.close()
    finally:
        app.close()
