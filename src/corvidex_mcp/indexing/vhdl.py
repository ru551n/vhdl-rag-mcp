"""VHDL semantic chunking: one chunk per meaningful construct.

The primary source is the vhdl_ls ``documentSymbol`` result (exact line
ranges, hierarchy, kind-prefixed construct names). When the language
server is unavailable or produced no usable top-level symbols (syntax
errors yield partial or empty trees), a structural line scanner finds
top-level constructs; when even that finds nothing, the whole file
becomes one chunk so no VHDL is ever lost from the index.

Chunk content is bounded (MAX_CONTENT_CHARS, the shared character
bound on chunk size — a token-size proxy for the embedding models): a
construct that exceeds the bound is not truncated but split along its
own structure (declaration part / statement part for architectures and
package bodies, statement by statement), so every line still reaches
the index in a bounded, coherent unit.

Every chunk carries the identifiers it defines or references in its
``symbols`` field — the cross-referencing key used by retrieval. The
function is synchronous: the (async) indexing pipeline owns the LSP
interaction and passes the parsed symbol tree in.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from ..config import RepositoryConfig
from ..lsp.client import SymbolInfo
from ..models import Chunk, CollectionName, ContentType
from .common import MAX_CONTENT_CHARS, MAX_SYMBOLS, MIN_INNER_SPAN

logger = logging.getLogger(__name__)

VHDL_KEYWORDS = frozenset(
    """
    abort access after alias all architecture array assert assign
    attribute begin block body buffer bus case component configuration
    constant context disconnect downto else elsif end entity exit for
    function generate generic group guarded if impure in inertial inout
    is label library literal loop map new next nor not null of on open
    or others out package parameter port postponed procedure process
    pure range record register reject report return role select shared
    signal subtype then to transport type unconditional units until use
    variable wait when while with work
    """.split()  # noqa: SIM905
)

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
#: vhdl_ls prefixes construct names with the kind: "entity 'fifo'".
_PREFIX_RE = re.compile(r"^(\w+(?:\s+\w+)?)\s+'(.+)'$")
_PREFIX_KINDS = frozenset(
    (
        "entity",
        "architecture",
        "package",
        "package body",
        "configuration",
        "process",
        "function",
        "procedure",
        "component",
        "generic",
        "signal",
        "constant",
    )
)
#: LSP SymbolKind ints that can stand at the top level of a VHDL file.
_TOP_LEVEL_KINDS = frozenset({2, 4, 5})
#: LSP SymbolKind ints for inner executable constructs.
_INNER_KINDS = frozenset({3, 12})

_LIBRARY_RE = re.compile(r"^\s*library\s+([A-Za-z_]\w*)\s*;")


@dataclass(frozen=True)
class ChunkSpec:
    """A chunk to emit: construct identity, line range, and parent context.

    Line numbers are 1-based and inclusive.
    """

    symbol: str
    symbol_kind: str
    start_line: int
    end_line: int
    library: str | None = None
    entity: str | None = None
    architecture: str | None = None


def extract_identifiers(content: str) -> tuple[str, ...]:
    """Significant identifiers in VHDL content (keywords removed).

    First-occurrence order, deduplicated, capped at MAX_SYMBOLS. Line
    comments (``--``) are ignored.
    """
    seen: dict[str, None] = {}
    for line in content.splitlines():
        if len(seen) >= MAX_SYMBOLS:
            break
        code = line.split("--", 1)[0]
        for match in _IDENT_RE.finditer(code):
            ident = match.group(0)
            if ident.lower() in VHDL_KEYWORDS:
                continue
            if ident not in seen:
                seen[ident] = None
                if len(seen) >= MAX_SYMBOLS:
                    break
    return tuple(seen)


_KIND_WORD_RE = re.compile(
    r"^\s*(entity|architecture|package|configuration|process|function|procedure)\b",
    re.I,
)
_PACKAGE_BODY_RE = re.compile(r"^\s*package\s+body\b", re.I)


def _refine_kind(kind: str, first_line: str) -> str:
    """Disambiguate a construct kind using its declaration line."""
    match = _KIND_WORD_RE.match(first_line)
    if not match:
        return kind
    word = match.group(1).lower()
    if word == "package":
        return "package_body" if _PACKAGE_BODY_RE.match(first_line) else "package"
    return word


def _resolve_name(symbol: SymbolInfo) -> tuple[str, str]:
    """(kind, name) for a documentSymbol entry.

    vhdl_ls prefixes names with the kind (``entity 'fifo'``); other
    servers may not, in which case the LSP kind int decides.
    """
    match = _PREFIX_RE.match(symbol.name)
    if match and match.group(1) in _PREFIX_KINDS:
        kind = match.group(1).replace("package body", "package_body")
        return kind, match.group(2)
    kind = {
        2: "construct",
        3: "process",
        4: "package",
        5: "component",
        12: "subprogram",
    }.get(symbol.kind, "construct")
    return kind, symbol.name


_ARCH_OF_RE = re.compile(r"^\s*architecture\s+[A-Za-z_]\w*\s+of\s+([A-Za-z_]\w*)", re.I)


def _context_for(
    kind: str, name: str, first_line: str
) -> tuple[str | None, str | None]:
    """(entity, architecture) parent context for one construct."""
    if kind == "entity":
        return name, None
    if kind == "architecture":
        match = _ARCH_OF_RE.match(first_line)
        return (match.group(1) if match else None), name
    return None, None


def _specs_from_symbols(
    symbols: tuple[SymbolInfo, ...], lines: list[str]
) -> list[ChunkSpec]:
    """Chunk specs from a documentSymbol tree (0-based LSP line ranges).

    vhdl_ls reports entity and architecture as top-level siblings, so
    inner contexts are taken from the sibling entities and from each
    construct's own declaration line.
    """
    specs: list[ChunkSpec] = []
    entity_names: list[str] = []
    for symbol in symbols:
        if symbol.kind not in _TOP_LEVEL_KINDS:
            continue
        kind, name = _resolve_name(symbol)
        if kind in ("construct", "component"):
            continue  # ambiguous or declaration-only; covered by the parent
        if kind == "entity":
            entity_names.append(name)
        first_line = lines[symbol.start_line] if symbol.start_line < len(lines) else ""
        entity, arch = _context_for(kind, name, first_line)
        specs.append(
            ChunkSpec(
                name,
                kind,
                symbol.start_line + 1,
                symbol.end_line + 1,
                entity=entity,
                architecture=arch,
            )
        )
        if kind not in ("architecture", "package", "package_body"):
            continue
        for child in symbol.children:
            if child.kind not in _INNER_KINDS:
                continue
            if child.end_line - child.start_line + 1 < MIN_INNER_SPAN:
                continue
            ckind, cname = _resolve_name(child)
            cline = lines[child.start_line] if child.start_line < len(lines) else ""
            specs.append(
                ChunkSpec(
                    cname,
                    _refine_kind(ckind, cline),
                    child.start_line + 1,
                    child.end_line + 1,
                    entity=entity
                    or (entity_names[0] if len(entity_names) == 1 else None),
                    architecture=arch,
                )
            )
    # Back-fill the entity context on architectures whose declaration line
    # did not name it (the sibling entity is the only reasonable answer).
    if len(entity_names) == 1:
        specs = [
            spec
            if spec.entity is not None or spec.symbol_kind != "architecture"
            else ChunkSpec(
                spec.symbol,
                spec.symbol_kind,
                spec.start_line,
                spec.end_line,
                library=spec.library,
                entity=entity_names[0],
                architecture=spec.architecture,
            )
            for spec in specs
        ]
    return specs


# -- structural fallback (line scanner) -------------------------------------

_START_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "architecture",
        re.compile(
            r"^\s*architecture\s+([A-Za-z_]\w*)\s+of\s+([A-Za-z_]\w*)\s+is\b", re.I
        ),
    ),
    (
        "package_body",
        re.compile(r"^\s*package\s+body\s+([A-Za-z_]\w*)\s+is\b", re.I),
    ),
    ("package", re.compile(r"^\s*package\s+([A-Za-z_]\w*)\s+is\b", re.I)),
    ("configuration", re.compile(r"^\s*configuration\s+([A-Za-z_]\w*)\s+is\b", re.I)),
    ("entity", re.compile(r"^\s*entity\s+([A-Za-z_]\w*)\s+is\b", re.I)),
)
_PROCESS_START_RE = re.compile(r"^\s*(?:[A-Za-z_]\w*\s*:\s*)?process\b", re.I)
_SUBPROGRAM_START_RE = re.compile(r"^\s*(function|procedure)\s+([A-Za-z_]\w*)")
_PROCESS_LABEL_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*:\s*process\b", re.I)


def _end_pattern(kind: str, name: str) -> re.Pattern[str]:
    esc = re.escape(name)
    table = {
        "entity": rf"^\s*end\s+(?:entity\s+{esc}|{esc})\s*;",
        "architecture": rf"^\s*end\s+(?:architecture\s+{esc}|{esc})\s*;",
        "package": rf"^\s*end\s+(?:package\s+{esc}|{esc})\s*;",
        "package_body": rf"^\s*end\s+(?:package\s+(?:body\s+)?{esc}|{esc})\s*;",
        "configuration": rf"^\s*end\s+(?:configuration\s+{esc}|{esc})\s*;",
        "process": rf"^\s*end\s+process\s*(?:{esc})?\s*;",
        "function": rf"^\s*end\s+function\s*(?:{esc})?\s*;",
        "procedure": rf"^\s*end\s+procedure\s*(?:{esc})?\s*;",
    }
    return re.compile(table.get(kind, r"^\s*end\s+;"), re.I)


def _match_top_start(line: str) -> tuple[str, str, str | None, int] | None:
    """(kind, name, entity-context, match end) for a top-level construct start."""
    for kind, pattern in _START_PATTERNS:
        match = pattern.match(line)
        if match:
            if kind == "architecture":
                return kind, match.group(1), match.group(2), match.end()
            return kind, match.group(1), None, match.end()
    return None


def _specs_from_scan(lines: list[str]) -> list[ChunkSpec]:
    """Chunk specs from a structural scan (LSP unavailable or broken).

    Top-level constructs are closed by their named ``end ...;``; inside
    architectures and packages, inner process/function/procedure bodies
    (declaration lines ending in ``is``) get their own chunks. The
    scanner is deliberately conservative: when in doubt it extends a
    construct to the end of the file rather than dropping it.
    """
    specs: list[ChunkSpec] = []
    n = len(lines)
    outer: tuple[str, str, int] | None = None  # kind, name, start(0-based)
    outer_entity: str | None = None
    outer_arch: str | None = None
    inner: tuple[str, str, int] | None = None  # kind, name, start(0-based)
    inner_end: re.Pattern[str] | None = None
    outer_end: re.Pattern[str] | None = None

    def close(kind: str, name: str, start: int, end: int) -> None:
        if kind in ("process", "function", "procedure") and (
            end - start + 1 < MIN_INNER_SPAN
        ):
            return  # stays covered by the parent chunk
        specs.append(
            ChunkSpec(
                name,
                kind,
                start + 1,
                end + 1,
                entity=outer_entity if kind != "entity" else name,
                architecture=outer_arch
                if kind not in ("entity", "architecture")
                else (name if kind == "architecture" else None),
            )
        )

    for i, line in enumerate(lines):
        stripped = line.split("--", 1)[0]
        if outer is None:
            match = _match_top_start(stripped)
            if match:
                kind, name, entity, end_pos = match
                outer = (kind, name, i)
                outer_entity = entity
                outer_arch = name if kind == "architecture" else None
                outer_end = _end_pattern(kind, name)
                if outer_end.search(stripped[end_pos:]):
                    # "entity t is end t;" — the whole construct on one line.
                    close(kind, name, i, i)
                    outer = None
                    outer_end = None
                    outer_arch = None
            continue
        if inner is not None and inner_end is not None and inner_end.match(stripped):
            close(inner[0], inner[1], inner[2], i)
            inner = None
            inner_end = None
            continue
        subprog = _SUBPROGRAM_START_RE.match(stripped)
        if outer[0] in ("architecture", "package", "package_body") and subprog:
            kind = subprog.group(1).lower()
            name = subprog.group(2)
            if stripped.rstrip().lower().endswith("is"):
                inner = (kind, name, i)
                inner_end = _end_pattern(kind, name)
                if inner_end.search(stripped[subprog.end() :]):
                    close(kind, name, i, i)
                    inner = None
                    inner_end = None
            continue
        if outer[0] == "architecture" and (
            proc_match := _PROCESS_START_RE.match(stripped)
        ):
            label_match = _PROCESS_LABEL_RE.match(stripped)
            name = label_match.group(1) if label_match else "process"
            inner = ("process", name, i)
            inner_end = _end_pattern("process", name)
            if inner_end.search(stripped[proc_match.end() :]):
                close("process", name, i, i)
                inner = None
                inner_end = None
            continue
        if outer_end is not None and outer_end.match(stripped):
            close(outer[0], outer[1], outer[2], i)
            outer = None
            outer_end = None
            outer_arch = None
    if outer is not None:
        close(outer[0], outer[1], outer[2], n - 1)
        if inner is not None:
            close(inner[0], inner[1], inner[2], n - 1)
    return sorted(specs, key=lambda s: (s.start_line, s.end_line))


# -- size-bounded assembly ----------------------------------------------------


def _windows(lines: list[str], max_chars: int) -> list[list[str]]:
    """Split lines into ordered windows whose joined text is <= max_chars.

    Breaks prefer blank lines; a hard break is the last resort. Every
    line appears in exactly one window, in order. A single line longer
    than max_chars becomes a window of its own — it cannot be split
    without corrupting the line.
    """
    windows: list[list[str]] = []
    cur: list[str] = []
    size = 0
    for line in lines:
        cost = len(line) + 1
        if cur and size + cost > max_chars:
            cut = 0
            for i in range(1, len(cur)):
                if not cur[i - 1].strip():
                    cut = i
            if cut == 0:
                cut = len(cur)
            windows.append(cur[:cut])
            rest = cur[cut:]
            cur = []
            size = 0
            for prev in rest:
                cur.append(prev)
                size += len(prev) + 1
        cur.append(line)
        size += cost
    if cur:
        windows.append(cur)
    return windows


_END_LINE_RE = re.compile(r"^\s*end\b")


def _split_statements(
    lines: list[str], base_indent: int, max_chars: int
) -> list[list[str]]:
    """Windows for a statement part (the lines from ``begin`` to the
    construct's ``end``).

    A new concurrent statement (process, block, generate, assign, ...)
    starts at a line indented no more than the ``begin`` line; the
    construct's closing ``end ...;`` attaches to the last statement.
    Statements are packed into windows of at most max_chars; a statement
    that alone exceeds the bound is windowed on its own.
    """
    blocks: list[list[str]] = []
    cur: list[str] = []
    for line in lines:
        code = line.split("--", 1)[0]
        if code.strip():
            indent = len(line) - len(line.lstrip())
            if cur and indent <= base_indent and not _END_LINE_RE.match(code):
                blocks.append(cur)
                cur = []
        cur.append(line)
    if cur:
        blocks.append(cur)

    windows: list[list[str]] = []
    cur_window: list[str] = []
    size = 0
    for block in blocks:
        block_len = sum(len(ln) + 1 for ln in block)
        if block_len > max_chars and not cur_window:
            windows.extend(_windows(block, max_chars))
            continue
        if cur_window and size + block_len > max_chars:
            windows.append(cur_window)
            cur_window = []
            size = 0
        cur_window.extend(block)
        size += block_len
    if cur_window:
        windows.append(cur_window)
    return windows


def _find_begin(lines: list[str], base_indent: int) -> int | None:
    """Index of the construct's ``begin`` line, or None.

    The construct's begin is aligned with the construct header; subprogram
    bodies declared in the declaration part are indented deeper, so the
    first column-aligned ``begin`` is the construct's own.
    """
    for i, line in enumerate(lines):
        if i == 0:
            continue
        if (
            line.split("--", 1)[0].strip() == "begin"
            and len(line) - len(line.lstrip()) == base_indent
        ):
            return i
    return None


def _split_structural(kind: str, part: list[str]) -> list[list[str]]:
    """Windows for one construct that exceeds MAX_CONTENT_CHARS.

    Architectures and package bodies split at their ``begin`` line — the
    declaration part first, then the statement part packed by statement —
    so oversized constructs are split along their own structure.
    Everything else is windowed on blank lines.
    """
    if kind in ("architecture", "package_body"):
        header_indent = len(part[0]) - len(part[0].lstrip())
        begin = _find_begin(part, header_indent)
        if begin is not None:
            declaration = part[:begin]
            statement = part[begin:]
            windows: list[list[str]] = (
                _windows(declaration, MAX_CONTENT_CHARS)
                if len("\n".join(declaration)) > MAX_CONTENT_CHARS
                else [declaration]
            )
            windows.extend(
                _split_statements(statement, header_indent, MAX_CONTENT_CHARS)
            )
            return windows
    return _windows(part, MAX_CONTENT_CHARS)


def _build_chunk(
    cfg: RepositoryConfig,
    file: str,
    commit: str,
    spec: ChunkSpec,
    part: list[str],
    abs_start: int,
    branch: str | None,
) -> Chunk:
    content = "\n".join(part)
    return Chunk(
        repository=cfg.name,
        branch=branch if branch is not None else cfg.ref,
        commit=commit,
        file=file,
        content_type=ContentType.SOURCE,
        language="vhdl",
        collection=CollectionName.HDL,
        symbol=spec.symbol,
        symbol_kind=spec.symbol_kind,
        start_line=abs_start,
        end_line=abs_start + len(part) - 1,
        content=content,
        library=spec.library,
        entity=spec.entity,
        architecture=spec.architecture,
        symbols=extract_identifiers(content),
    )


def _chunks_for_spec(
    cfg: RepositoryConfig,
    file: str,
    lines: list[str],
    commit: str,
    spec: ChunkSpec,
    branch: str | None = None,
) -> list[Chunk]:
    """One chunk, or — for a construct whose content exceeds
    MAX_CONTENT_CHARS — its structural split: several chunks that tile
    the construct's line range exactly. All parts keep the construct's
    symbol and kind; the distinct line ranges keep their point IDs
    (and canonical IDs) distinct and deterministic.
    """
    if not lines:
        raise ValueError(f"empty VHDL file {file!r}")
    start = max(1, min(spec.start_line, len(lines)))
    end = max(start, min(spec.end_line, len(lines)))
    part = lines[start - 1 : end]
    if len("\n".join(part)) <= MAX_CONTENT_CHARS:
        return [_build_chunk(cfg, file, commit, spec, part, start, branch)]
    chunks: list[Chunk] = []
    offset = start
    for group in _split_structural(spec.symbol_kind, part):
        chunks.append(_build_chunk(cfg, file, commit, spec, group, offset, branch))
        offset += len(group)
    return chunks


def chunk_vhdl_file(
    cfg: RepositoryConfig,
    file: str,
    content: str,
    commit: str,
    lsp_symbols: tuple[SymbolInfo, ...] | None = None,
    branch: str | None = None,
) -> list[Chunk]:
    """Chunk one VHDL file into :class:`Chunk` objects.

    ``lsp_symbols`` is the file's documentSymbol tree (None or empty
    when the language server was unavailable or the file has syntax
    errors) — in that case the structural scan takes over, and if no
    construct can be found the whole file is indexed as one chunk.
    """
    lines = content.splitlines()
    specs: list[ChunkSpec] = []
    if lsp_symbols:
        specs = _specs_from_symbols(lsp_symbols, lines)
    if not specs:
        specs = _specs_from_scan(lines)
    if not specs:
        stem = Path(file).stem
        specs = [ChunkSpec(stem, "file", 1, max(1, len(lines)))]
    library_match = next(
        (m.group(1) for line in lines if (m := _LIBRARY_RE.match(line))), None
    )
    specs = [
        spec
        if spec.library is not None
        else ChunkSpec(
            spec.symbol,
            spec.symbol_kind,
            spec.start_line,
            spec.end_line,
            library=library_match,
            entity=spec.entity,
            architecture=spec.architecture,
        )
        for spec in specs
    ]
    return [
        chunk
        for spec in specs
        for chunk in _chunks_for_spec(cfg, file, lines, commit, spec, branch)
    ]
