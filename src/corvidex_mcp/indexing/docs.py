"""Documentation chunking: one chunk per heading section (and paragraphs).

Supports Markdown (``#`` headings, code fences), reStructuredText
(``Heading\\n====`` and ``Heading\\n----`` underlines, ``.. code-block``
directives, literal blocks) and plain text (paragraph splitting).

Code fences inside a section are kept in the chunk content and their
identifiers are added to the chunk's ``symbols`` — this is what lets a
documentation section like "Reset conventions" be cross-referenced to
the VHDL processes and C functions that mention ``rst_n``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from ..config import RepositoryConfig
from ..models import Chunk, CollectionName, ContentType
from .common import MAX_CONTENT_CHARS, extract_code_identifiers

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^(\s*)(```|~~~)\s*(\S*)\s*$")
_RST_DIRECTIVE_RE = re.compile(
    r"^\.\.\s+(?:code-block|code::|literalinclude::|sourcecode::)\b[^\n]*"
)
_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_HEADING2_RE = re.compile(r"^ {0,3}(#{1,6})\s+(.+?)\s*$")
_UNDERLINE_RE = re.compile(r"^ {0,3}(={3,}|-{3,}|~{3,}|\+{3,}|#{3,}|\*{3,})\s*$")
_RST_HEADS = frozenset("=-~#+*^")
_PARAGRAPH_MIN = 40  # chars: shorter paragraphs are merged with the next
_PARAGRAPH_MAX = 2000  # chars: longer paragraphs are hard-split


@dataclass(frozen=True)
class DocSection:
    """A documentation section: heading, line range, content, code symbols."""

    heading: str
    level: int
    start_line: int  # 1-based
    end_line: int  # 1-based
    content: str
    code_symbols: tuple[str, ...]


def _code_symbols_from(content: str) -> tuple[str, ...]:
    """Identifiers from the fenced/directive code blocks in a section."""
    in_fence = False
    fence_marker = ""
    blocks: list[str] = []
    current: list[str] = []
    for line in content.splitlines():
        fence = _FENCE_RE.match(line)
        if fence:
            if not in_fence:
                in_fence = True
                fence_marker = fence.group(2)
                current = []
            elif fence.group(2) == fence_marker:
                in_fence = False
                blocks.append("\n".join(current))
                current = []
            continue
        if in_fence:
            if fence_marker == "rst":
                # RST directive blocks end at the first blank line after
                # some content (or at EOF).
                if line.strip():
                    current.append(line)
                elif current:
                    blocks.append("\n".join(current))
                    current = []
                    in_fence = False
            else:
                current.append(line)
            continue
        if _RST_DIRECTIVE_RE.match(line):
            current = []
            fence_marker = "rst"
            in_fence = True
    if current:
        blocks.append("\n".join(current))
    symbols: tuple[str, ...] = ()
    for block in blocks:
        symbols = extract_code_identifiers(block)
        if symbols:
            break
    return symbols


def _markdown_sections(lines: list[str]) -> list[DocSection]:
    sections: list[tuple[str, int, int, int]] = []  # heading, level, start, end
    current: tuple[str, int, int] | None = None  # heading, level, start
    for i, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if not match:
            continue
        if current is not None:
            sections.append((current[0], current[1], current[2], i))
        current = (match.group(2).strip(), len(match.group(1)), i + 1)
    if current is not None:
        sections.append((current[0], current[1], current[2], len(lines)))
    out: list[DocSection] = []
    for j, (heading, level, start, _end) in enumerate(sections):
        stop = sections[j + 1][2] - 1 if j + 1 < len(sections) else len(lines)
        content = "\n".join(lines[start - 1 : stop])
        out.append(
            DocSection(
                heading,
                level,
                start,
                stop,
                content.strip(),
                _code_symbols_from(content),
            )
        )
    return out


def _rst_sections(lines: list[str]) -> list[DocSection]:
    sections: list[tuple[str, int, int]] = []
    i = 0
    n = len(lines)
    # Underline-style headings: a text line followed by its underline.
    while i < n - 1:
        text_line = lines[i]
        underline = lines[i + 1]
        if (
            text_line.strip()
            and _UNDERLINE_RE.match(underline) is not None
            and not _HEADING2_RE.match(text_line)
        ):
            u_match = _UNDERLINE_RE.match(underline)
            t_match = _UNDERLINE_RE.match(text_line)
            if (
                u_match
                and not t_match
                and len(underline.strip()) >= len(text_line.strip())
                and u_match.group(1)[0] in _RST_HEADS
            ):
                sections.append((text_line.strip(), 1, i + 1))
                i += 2
                continue
        i += 1
    # ATX headings (increasingly common in RST files too).
    for i, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if match:
            sections.append((match.group(2).strip(), len(match.group(1)), i + 1))
    if not sections:
        return []
    sections.sort(key=lambda s: s[2])
    out: list[DocSection] = []
    for j, (heading, level, start) in enumerate(sections):
        stop = sections[j + 1][2] - 1 if j + 1 < len(sections) else n
        content = "\n".join(lines[start - 1 : stop])
        out.append(
            DocSection(
                heading,
                level,
                start,
                stop,
                content.strip(),
                _code_symbols_from(content),
            )
        )
    return out


def _paragraph_sections(lines: list[str]) -> list[DocSection]:
    """Plain text (and heading-less docs): paragraph-based chunks.

    A paragraph is a blank-line-separated run of non-empty lines that is
    at least _PARAGRAPH_MIN chars; shorter runs are merged into the
    following paragraph. Paragraphs longer than _PARAGRAPH_MAX are split
    at the last line boundary before the limit.
    """
    paragraphs: list[tuple[int, list[str]]] = []  # (start line idx, lines)
    current: list[str] = []
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip():
            if not current:
                start_idx = i
            current.append(line)
        elif current:
            paragraphs.append((start_idx, current))
            current = []
    if current:
        paragraphs.append((start_idx, current))
    # Merge short runs with the following paragraph (or keep alone at EOF).
    merged: list[tuple[int, list[str]]] = []
    for start_idx, para in paragraphs:
        if merged and len("\n".join(para)) < _PARAGRAPH_MIN:
            merged[-1] = (merged[-1][0], merged[-1][1] + [""] + para)
        else:
            merged.append((start_idx, para))
    if len(merged) == 1 and len("\n".join(merged[0][1])) < _PARAGRAPH_MIN:
        return []
    out: list[DocSection] = []
    for start_idx, para in merged:
        pieces: list[tuple[str, int, int]] = []  # content, start, end (1-based)
        chunk_lines: list[str] = []
        size = 0
        piece_start = start_idx + 1
        for k, ln in enumerate(para):
            if size + len(ln) + 1 > _PARAGRAPH_MAX and chunk_lines:
                pieces.append(
                    (
                        "\n".join(chunk_lines),
                        piece_start,
                        piece_start + len(chunk_lines) - 1,
                    )
                )
                chunk_lines = []
                size = 0
                piece_start = start_idx + k + 1
            chunk_lines.append(ln)
            size += len(ln) + 1
        if chunk_lines:
            pieces.append(
                (
                    "\n".join(chunk_lines),
                    piece_start,
                    piece_start + len(chunk_lines) - 1,
                )
            )
        for k, (content, s, e) in enumerate(pieces):
            suffix = "" if k == 0 else f" (cont. {k + 1})"
            heading = para[0].strip()[:60]
            out.append(
                DocSection(
                    heading + suffix,
                    1,
                    s,
                    e,
                    content.strip(),
                    _code_symbols_from(content),
                )
            )
    return out


def chunk_doc_file(
    cfg: RepositoryConfig,
    file: str,
    content: str,
    commit: str,
    language: str,
    branch: str | None = None,
) -> list[Chunk]:
    """Chunk one documentation file into :class:`Chunk` objects.

    Markdown/ReST files yield one chunk per heading section (sections
    are bounded by the next heading at any level); files with no
    recognized structure fall back to paragraph chunks. A file with no
    content yields no chunks.
    """
    if not content.strip():
        return []
    lines = content.splitlines()
    sections: list[DocSection] = []
    if language == "markdown":
        sections = _markdown_sections(lines)
    elif language == "restructuredtext":
        sections = _rst_sections(lines)
    paragraph_fallback = not sections
    if paragraph_fallback:
        sections = _paragraph_sections(lines)
    stem = Path(file).stem
    chunks: list[Chunk] = []
    # Section = the most recent heading at a strictly higher level before
    # this chunk (its enclosing section); top-level sections have none.
    # ``sections`` is already in file order (ascending start_line), so a
    # stack of ancestor (level, heading) pairs finds it in one pass
    # instead of rescanning the whole list per section.
    ancestors: list[tuple[int, str]] = []
    for i, section in enumerate(sections):
        heading = (
            section.heading
            if section.heading
            else (stem if i == 0 else f"{stem} (cont.)")
        )
        while ancestors and ancestors[-1][0] >= section.level:
            ancestors.pop()
        section_ctx = ancestors[-1][1] if ancestors else None
        ancestors.append((section.level, heading))
        chunks.append(
            Chunk(
                repository=cfg.name,
                branch=branch if branch is not None else cfg.ref,
                commit=commit,
                file=file,
                content_type=ContentType.DOCUMENTATION,
                language=language,
                collection=CollectionName.DOCS,
                symbol=heading[:200],
                symbol_kind=(
                    "paragraph"
                    if paragraph_fallback
                    else "section"
                    if language in ("markdown", "restructuredtext")
                    else "paragraph"
                ),
                start_line=section.start_line,
                end_line=section.end_line,
                content=section.content[:MAX_CONTENT_CHARS],
                heading=heading[:200],
                section=section_ctx,
                symbols=section.code_symbols,
            )
        )
    return chunks
