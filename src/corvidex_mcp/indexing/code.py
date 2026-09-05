"""General source-code chunking: one chunk per top-level unit.

One tree-sitter-based strategy covers every language
(``tree-sitter-language-pack`` ships the grammars — no per-language
scanners in this module):

- top-level function/subroutine definitions and class units become
  chunks (name taken from the definition node's declarator/name field;
  decorators are included in the chunk range);
- uncovered top-level code (globals, typedefs, module-level
  statements) stays searchable as file-scope "gap" chunks — runs of
  >= 2 non-blank lines; comment lines and single-line runs
  (``#include`` etc.) are noise and are dropped;
- a file with no recognized definition node (headers,
  declaration-only files) or without a usable grammar becomes one
  whole-file chunk, so no code is ever lost from the index.

Every chunk's ``symbols`` lists the identifiers it references — the
cross-referencing key between the code collection and the VHDL/docs
collections.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Node, Parser
from tree_sitter_language_pack import get_language

from ..config import RepositoryConfig
from ..models import Chunk, CollectionName, ContentType
from .common import MAX_CONTENT_CHARS, clamp_and_join, extract_code_identifiers

logger = logging.getLogger(__name__)

#: Top-level unit node types for languages with explicit support; any
#: other language falls back to the generic ``*_definition`` rule.
_UNIT_TYPES: dict[str, frozenset[str]] = {
    "c": frozenset({"function_definition"}),
    "cpp": frozenset({"function_definition"}),
    "python": frozenset(
        {"function_definition", "class_definition", "decorated_definition"}
    ),
}
#: Generic rule: top-level nodes named "*_definition" (the C-family
#: type_definition is a declaration, not a definition with a body).
_GENERIC_UNIT_SUFFIX = "_definition"
_EXCLUDED_GENERIC_TYPE = "type_definition"

_PARSERS: dict[str, Parser] = {}


@dataclass(frozen=True)
class CodeUnit:
    """One top-level code unit (function/class) with its 1-based range."""

    name: str
    kind: str  # "function" | "class" | "file"
    start_line: int
    end_line: int


def _parser(language: str) -> Parser | None:
    """Lazily build (and cache) the tree-sitter parser for one grammar."""
    parser = _PARSERS.get(language)
    if parser is None:
        try:
            parser = Parser(get_language(language))
        except Exception:
            logger.warning(
                "no tree-sitter grammar for language %r; "
                "falling back to whole-file chunks",
                language,
            )
            return None
        _PARSERS[language] = parser
    return parser


def _text(node: Node) -> str:
    raw = node.text
    return raw.decode("utf-8", "replace") if raw is not None else ""


def _unit_name(node: Node) -> str | None:
    """Identifier of a definition node (None for anonymous definitions).

    C/C++: walk the declarator chain (``function_definition`` ->
    optional ``pointer_declarator`` -> ``function_declarator`` ->
    identifier). Python: the ``name`` field; a decorated definition is
    unwrapped to its inner def/class first.
    """
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                return _unit_name(child)
        return None
    decl = node.child_by_field_name("declarator")
    while decl is not None:
        if decl.type == "identifier":
            return _text(decl)
        decl = decl.child_by_field_name("declarator")
    name = node.child_by_field_name("name")
    if name is not None and name.type == "identifier":
        return _text(name)
    for child in node.children:
        if child.type == "identifier":
            return _text(child)
    return None


def _unit_kind(node: Node) -> str:
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type in ("function_definition", "class_definition"):
                return _unit_kind(child)
        return "function"
    return "class" if node.type == "class_definition" else "function"


def _is_unit(node: Node, language: str) -> bool:
    known = _UNIT_TYPES.get(language)
    if known is not None:
        return node.type in known
    return node.type.endswith(_GENERIC_UNIT_SUFFIX) and (
        node.type != _EXCLUDED_GENERIC_TYPE
    )


def _units_from_tree(root: Node, language: str) -> tuple[list[CodeUnit], set[int]]:
    """Top-level definition units, plus line numbers excluded from gaps.

    Excluded lines are top-level comment lines: comments are not code,
    and a comment block is not a gap unit.
    """
    units: list[CodeUnit] = []
    excluded: set[int] = set()
    for node in root.children:
        if node.type == "comment":
            excluded.update(range(node.start_point[0] + 1, node.end_point[0] + 1))
            continue
        if not _is_unit(node, language):
            continue
        name = _unit_name(node)
        if name is None:
            continue
        units.append(
            CodeUnit(
                name,
                _unit_kind(node),
                node.start_point[0] + 1,
                node.end_point[0] + 1,
            )
        )
    return units, excluded


def _gap_units(
    lines: list[str], stem: str, units: list[CodeUnit], excluded: set[int]
) -> list[CodeUnit]:
    """One file-scope unit per uncovered non-blank run of >= 2 lines.

    Top-level statements that no unit covers (module-level Python code,
    C globals/typedefs) stay searchable this way; single-line runs
    (``#include`` etc.) and comment lines are noise and are dropped.
    """
    covered = set(excluded)
    for unit in units:
        covered.update(range(unit.start_line, unit.end_line + 1))
    runs: list[tuple[int, int]] = []
    start: int | None = None
    prev_end: int | None = None
    for i in range(1, len(lines) + 1):
        if lines[i - 1].strip() and i not in covered:
            if start is None:
                start = i
            prev_end = i
        elif start is not None and prev_end is not None:
            if prev_end - start + 1 >= 2:
                runs.append((start, prev_end))
            start = None
            prev_end = None
    if start is not None and prev_end is not None and prev_end - start + 1 >= 2:
        runs.append((start, prev_end))
    return [CodeUnit(stem, "file", s, e) for s, e in runs]


def _fallback_units(lines: list[str], stem: str) -> list[CodeUnit]:
    """Whole-file fallback: the entire file as one unit.

    Used for files without recognizable top-level definitions (headers,
    declaration-only files) or without a usable grammar. Keeps every
    byte searchable; content is bounded by MAX_CONTENT_CHARS at chunk
    assembly.
    """
    return [CodeUnit(stem, "file", 1, len(lines))]


def chunk_code_file(
    cfg: RepositoryConfig,
    file: str,
    content: str,
    commit: str,
    language: str,
    content_type: ContentType = ContentType.CODE,
    collection: CollectionName = CollectionName.CODE,
    branch: str | None = None,
) -> list[Chunk]:
    """Chunk one source file into :class:`Chunk` objects (tree-sitter).

    ``content_type``/``collection`` default to general code but are
    overridable so the same generic parser can index HDL files (the
    graceful fallback when their dedicated analyzer is unavailable).
    """
    if not content.strip():
        return []
    lines = content.splitlines()
    stem = Path(file).stem
    units: list[CodeUnit] = []
    excluded: set[int] = set()
    parser = _parser(language)
    if parser is not None:
        root = parser.parse(content.encode("utf-8", "replace")).root_node
        units, excluded = _units_from_tree(root, language)
    if not units:
        # No recognized top-level units (or no grammar): the whole file
        # is one chunk.
        units = _fallback_units(lines, stem)
    else:
        units.extend(_gap_units(lines, stem, units, excluded))
        units.sort(key=lambda u: u.start_line)
    chunks: list[Chunk] = []
    for unit in units:
        start, end, text = clamp_and_join(lines, unit.start_line, unit.end_line)
        chunks.append(
            Chunk(
                repository=cfg.name,
                branch=branch if branch is not None else cfg.ref,
                commit=commit,
                file=file,
                content_type=content_type,
                language=language,
                collection=collection,
                symbol=unit.name,
                symbol_kind=unit.kind,
                start_line=start,
                end_line=end,
                content=text[:MAX_CONTENT_CHARS],
                symbols=extract_code_identifiers(text),
            )
        )
    return chunks
