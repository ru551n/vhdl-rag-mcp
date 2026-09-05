"""Verilog/SystemVerilog semantic chunking: one chunk per meaningful construct.

The primary source is the Veridian ``documentSymbol`` result (exact line
ranges, hierarchy, plain names, standard LSP SymbolKinds). When the
language server is unavailable or produced no usable top-level symbols
(a syntax error yields a partial or empty tree, and Veridian does not
report ``always_*`` or generate blocks at all), a structural line
scanner finds top-level units and their inner always/function/task
bodies; when even that finds nothing, the whole file becomes one chunk
so no Verilog is ever lost from the index.

Constructs are stored with the *normalized* cross-language kind
(semantic model): module/program/interface map to ``design_unit``,
``always_ff``/``always_comb``/``always_latch``/``always`` map to
``process``; the server-native name is kept in ``native_symbol_kind``.
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

logger = logging.getLogger(__name__)

#: Minimum line span for an inner construct (always/function/task) to
#: earn its own chunk; smaller ones stay covered by the parent chunk.
MIN_INNER_SPAN = 5
#: Identifier cap per chunk (payload size bound).
MAX_SYMBOLS = 100

#: Verilog/SV `directive names (not identifiers).
VERILOG_DIRECTIVES = frozenset(
    """
    include define ifdef ifndef ifdef0 ifdef1 ifndef0 ifndef1
    iflt ifle ifgt ifge ifeq ifne endif endelse else
    timescale celldefine endcelldefine default_nettype unconnected_drive
    line resetall undefineall defineall keywords begin_keywords
    end_keywords pragma
    """.split()  # noqa: SIM905
)

VERILOG_KEYWORDS = frozenset(
    """
    always always_comb always_ff always_latch and assign automatic begin
    case casex cell config configuration const constraint continue
    cover covergroup cross default defparam disable do edge else end
    endcase endchecker endclocking endconfig endfunction endgenerate
    endinterface endmodule endpackage endparameter endprimitive
    endprogram endproperty endspecify endtable endtask enum event export
    extends extern final first_match for force foreach forever function
    generate genvar if iff ifnone illegal import initial inout input
    inside instance int integer interface join join_any join_none local
    localparam logic longint macro match medium modport module nand
    negedge new nmos nor not notified notify or output package parameter
    posedge primitive program property protected pull0 pull1 pure range
    reg repeat return rnmos rpmos rtran rtranif0 rtranif1 rtrireg rtrior
    rtriz scalared sequence shortint shortreal signed small specify
    specparam static string struct supply0 supply1 table task this time
    timeprecision timeunit tri tri0 tri1 triand trior trireg type typedef
    union unique0 unique unsigned use var vector virtual void wait
    wait_order weak0 weak1 while wildcard wire with within wor xnor xor
    """.split()  # noqa: SIM905
)

_IDENT_RE = re.compile(r"`?[A-Za-z_][A-Za-z0-9_$]*")
_BLOCK_OPEN_RE = re.compile(r"/\*")
_BLOCK_CLOSE_RE = re.compile(r"\*/")
_ALWAYS_START_RE = re.compile(r"^\s*always(?:_comb|_ff|_latch)?\b", re.I)
_FUNCTION_START_RE = re.compile(r"^\s*function\b", re.I)
_TASK_START_RE = re.compile(r"^\s*task\b", re.I)
_TOP_START_RE = re.compile(
    r"^\s*(module|interface|package|program)\s+([A-Za-z_][A-Za-z0-9_$]*)",
    re.I,
)
_TOP_ENDS: dict[str, re.Pattern[str]] = {
    "module": re.compile(r"^\s*endmodule\b", re.I),
    "interface": re.compile(r"^\s*endinterface\b", re.I),
    "package": re.compile(r"^\s*endpackage\b", re.I),
    "program": re.compile(r"^\s*endprogram\b", re.I),
}
_TOP_NORMALIZED: dict[str, str] = {
    "module": "design_unit",
    "interface": "design_unit",
    "package": "package",
    "program": "design_unit",
}
#: LSP SymbolKind ints (standard) that can stand at the top level of a
#: Verilog/SV file per Veridian.
_TOP_LEVEL_KINDS = frozenset({2, 4, 11})
#: LSP SymbolKind for inner functions/tasks (Function).
_INNER_FUNCTION_KIND = 12

_BEGIN_RE = re.compile(r"\bbegin\b", re.I)
_END_WORD_RE = re.compile(r"\bend\b", re.I)
_ENDFUNCTION_RE = re.compile(r"^\s*endfunction\b", re.I)
_ENDTASK_RE = re.compile(r"^\s*endtask\b", re.I)


@dataclass(frozen=True)
class ChunkSpec:
    """A chunk to emit: construct identity, line range, and parent context.

    Line numbers are 1-based and inclusive.
    """

    symbol: str
    symbol_kind: str
    native_kind: str
    start_line: int
    end_line: int
    module: str | None = None


def extract_identifiers(content: str) -> tuple[str, ...]:
    """Significant identifiers in Verilog/SV content (keywords removed).

    First-occurrence order, deduplicated, capped at MAX_SYMBOLS. Line
    comments (``//``) and block comments (``/* */``) are ignored;
    `` `directive`` names are kept (they are cross-referencing keys).
    """
    seen: dict[str, None] = {}
    depth = 0  # block-comment nesting across lines
    for raw_line in content.splitlines():
        if len(seen) >= MAX_SYMBOLS:
            break
        code, depth = _strip_block_comments(raw_line, depth)
        if depth > 0:
            continue  # still inside a block comment
        code = code.split("//", 1)[0]
        for match in _IDENT_RE.finditer(code):
            ident = match.group(0)
            if ident[0] == "`":
                ident = ident.lstrip("`")
                if ident.lower() in VERILOG_DIRECTIVES:
                    continue
            if ident and ident.lower() in VERILOG_KEYWORDS:
                continue
            if ident and ident not in seen:
                seen[ident] = None
                if len(seen) >= MAX_SYMBOLS:
                    break
    return tuple(seen)


def _strip_block_comments(line: str, depth: int) -> tuple[str, int]:
    """Remove /* */ spans from one line; returns (code, depth)."""
    out: list[str] = []
    pos = 0
    n = len(line)
    while pos < n:
        if depth > 0:
            # Inside a block comment: drop up to the closing */.
            close = _BLOCK_CLOSE_RE.search(line, pos)
            if close is None:
                break  # rest of the line is still inside the comment
            depth = 0
            pos = close.end()
            continue
        open_ = _BLOCK_OPEN_RE.search(line, pos)
        if open_ is None:
            out.append(line[pos:])
            break
        out.append(line[pos : open_.start()])
        depth = 1
        pos = open_.end()
    return "".join(out), depth


def _first_keyword_match(line: str) -> str | None:
    """The top-level unit kind on a line (module/interface/package/program)."""
    match = _TOP_START_RE.match(line)
    if match:
        return match.group(1).lower()
    return None


def _resolve_native(symbol: SymbolInfo, first_line: str) -> tuple[str, str]:
    """(normalized_kind, native_kind) for a top-level documentSymbol entry.

    Veridian uses plain names + standard LSP kinds; the LSP kind decides
    the normalized kind, and the declaration line disambiguates the
    native kind (module vs program vs interface vs package).
    """
    kind_map = {
        2: "design_unit",
        4: "package",
        11: "design_unit",
    }
    normalized = kind_map.get(symbol.kind, "design_unit")
    native = _first_keyword_match(first_line) or "module"
    if native == "package":
        normalized = "package"
        native = "package"
    elif native in ("module", "program", "interface"):
        normalized = _TOP_NORMALIZED[native]
    return normalized, native


def _specs_from_symbols(
    symbols: tuple[SymbolInfo, ...], lines: list[str]
) -> list[ChunkSpec]:
    """Chunk specs from a documentSymbol tree (0-based LSP line ranges).

    Veridian reports modules/packages/interfaces as top-level siblings;
    inner functions/tasks are children of the module. Anonymous tasks
    (empty name — a Veridian v0.1.0 quirk) are recovered from the
    declaration line.
    """
    specs: list[ChunkSpec] = []
    for symbol in symbols:
        if symbol.kind not in _TOP_LEVEL_KINDS:
            continue
        first_line = lines[symbol.start_line] if symbol.start_line < len(lines) else ""
        normalized, native = _resolve_native(symbol, first_line)
        module_name = symbol.name
        specs.append(
            ChunkSpec(
                symbol.name,
                normalized,
                native,
                symbol.start_line + 1,
                symbol.end_line + 1,
                module=None,
            )
        )
        for child in symbol.children:
            if child.kind != _INNER_FUNCTION_KIND:
                continue
            if child.end_line - child.start_line + 1 < MIN_INNER_SPAN:
                continue
            cline = lines[child.start_line] if child.start_line < len(lines) else ""
            cname, ckind = _resolve_inner_name_kind(child.name, cline)
            specs.append(
                ChunkSpec(
                    cname,
                    ckind,
                    ckind,
                    child.start_line + 1,
                    child.end_line + 1,
                    module=module_name,
                )
            )
    return specs


def _name_before_paren(line: str) -> str | None:
    """The identifier immediately before the argument list ``(`` (if any)."""
    idx = line.find("(")
    if idx < 0:
        return None
    match = re.search(r"([A-Za-z_][A-Za-z0-9_$]*)\s*$", line[:idx])
    return match.group(1) if match else None


def _subprogram_name(line: str, keyword: str) -> str:
    """Name of a function/task from its declaration line.

    Prefers the identifier immediately before the argument list (so the
    return type is not mistaken for the name); falls back to the last
    non-keyword identifier on the line for argument-less functions.
    """
    before = _name_before_paren(line)
    if before is not None and before.lower() != keyword:
        return before
    ids = [
        m.group(1)
        for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_$]*)", line)
        if m.group(1).lower() not in (keyword, "automatic")
    ]
    return ids[-1] if ids else keyword


def _resolve_inner_name_kind(name: str, first_line: str) -> tuple[str, str]:
    """(name, normalized_kind) for an inner function/task child.

    Veridian reports both as Function (kind 12) and may give an empty
    name for tasks. The LSP name wins when present; the declaration
    line recovers the name and decides the kind (task vs function).
    """
    kind = "task" if _TASK_START_RE.match(first_line) else "function"
    if name:
        return name, kind
    return _subprogram_name(first_line, kind), kind


# -- structural fallback (line scanner) -------------------------------------


def _specs_from_scan(lines: list[str]) -> list[ChunkSpec]:
    """Chunk specs from a structural scan (LSP unavailable or broken).

    Top-level units (module/interface/package/program) are closed by
    their named ``end<unit>``; inside a module/program/package, inner
    always blocks and function/task bodies get their own chunks (always
    blocks are closed by ``begin``/``end`` depth counting). The scanner
    is deliberately conservative: when in doubt it extends a construct
    to the end of the file rather than dropping it.
    """
    specs: list[ChunkSpec] = []
    n = len(lines)
    outer: tuple[str, str, int] | None = None  # native, name, start(0-based)
    # inner: (kind, name, native, start(0-based), end_match, begin_depth)
    inner: tuple[str, str, str, int, re.Pattern[str] | None, int] | None = None

    def close_inner(kind: str, name: str, native: str, start: int, end: int) -> None:
        if end - start + 1 < MIN_INNER_SPAN:
            return
        specs.append(
            ChunkSpec(
                name,
                kind,
                native,
                start + 1,
                end + 1,
                module=(outer[1] if outer else None),
            )
        )

    def close_outer(native: str, name: str, start: int, end: int) -> None:
        specs.append(
            ChunkSpec(
                name,
                _TOP_NORMALIZED[native],
                native,
                start + 1,
                end + 1,
                module=None,
            )
        )

    for i, raw in enumerate(lines):
        line = raw.split("//", 1)[0].strip()
        if inner is not None:
            if inner[0] == "process":
                depth = (
                    inner[5]
                    + len(_BEGIN_RE.findall(line))
                    - len(_END_WORD_RE.findall(line))
                )
                if depth <= 0 and _END_WORD_RE.search(line):
                    close_inner(inner[0], inner[1], inner[2], inner[3], i)
                    inner = None
                else:
                    inner = (inner[0], inner[1], inner[2], inner[3], inner[4], depth)
                continue
            if inner[4] is not None and inner[4].match(line):
                close_inner(inner[0], inner[1], inner[2], inner[3], i)
                inner = None
                continue
        if outer is None:
            kind = _first_keyword_match(line)
            if kind is not None:
                m = _TOP_START_RE.match(line)
                name = m.group(2) if m is not None else ""
                outer = (kind, name, i)
            continue
        # Inside a module/program/package: inner always/function/task.
        if outer[0] in ("module", "program", "package"):
            am = _ALWAYS_START_RE.match(line)
            fm = _FUNCTION_START_RE.match(line)
            tm = _TASK_START_RE.match(line)
            if am:
                label = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_$]*)\s*:", line)
                native = am.group(0).strip()
                name = label.group(1) if label else native
                depth = len(_BEGIN_RE.findall(line)) - len(_END_WORD_RE.findall(line))
                if depth <= 0 and _END_WORD_RE.search(line):
                    close_inner("process", name, native, i, i)
                else:
                    inner = ("process", name, native, i, None, depth)
                continue
            if fm:
                name = _subprogram_name(line, "function")
                inner = ("function", name, "function", i, _ENDFUNCTION_RE, 0)
                continue
            if tm:
                name = _subprogram_name(line, "task")
                inner = ("task", name, "task", i, _ENDTASK_RE, 0)
                continue
        end = _TOP_ENDS.get(outer[0])
        if end is not None and end.match(line):
            close_outer(outer[0], outer[1], outer[2], i)
            outer = None
    if outer is not None:
        close_outer(outer[0], outer[1], outer[2], n - 1)
        if inner is not None:
            close_inner(inner[0], inner[1], inner[2], inner[3], n - 1)
    return sorted(specs, key=lambda s: (s.start_line, s.end_line))


# -- chunk assembly -----------------------------------------------------------


def _make_chunk(
    cfg: RepositoryConfig,
    file: str,
    lines: list[str],
    commit: str,
    language: str,
    spec: ChunkSpec,
    branch: str | None = None,
) -> Chunk:
    if not lines:
        raise ValueError(f"empty {language} file {file!r}")
    start = max(1, min(spec.start_line, len(lines)))
    end = max(start, min(spec.end_line, len(lines)))
    text = "\n".join(lines[start - 1 : end])
    return Chunk(
        repository=cfg.name,
        branch=branch if branch is not None else cfg.ref,
        commit=commit,
        file=file,
        content_type=ContentType.SOURCE,
        language=language,
        collection=CollectionName.HDL,
        symbol=spec.symbol,
        symbol_kind=spec.symbol_kind,
        native_symbol_kind=spec.native_kind,
        start_line=start,
        end_line=end,
        content=text,
        module=spec.module,
        symbols=extract_identifiers(text),
    )


def chunk_verilog_file(
    cfg: RepositoryConfig,
    file: str,
    content: str,
    commit: str,
    language: str,
    lsp_symbols: tuple[SymbolInfo, ...] | None = None,
    branch: str | None = None,
) -> list[Chunk]:
    """Chunk one Verilog/SV file into :class:`Chunk` objects.

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
        specs = [ChunkSpec(stem, "file", "file", 1, max(1, len(lines)))]
    return [
        _make_chunk(cfg, file, lines, commit, language, spec, branch) for spec in specs
    ]
