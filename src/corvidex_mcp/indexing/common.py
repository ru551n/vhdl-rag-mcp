"""Shared helpers for the per-domain chunkers."""

from __future__ import annotations

import re

#: Upper bound on chunk content size (chars). Well within the embedding
#: models' 8192-token context, and small enough to stay useful as a
#: RAG unit.
MAX_CONTENT_CHARS = 12000
#: Identifier cap per chunk (payload size bound).
MAX_SYMBOLS = 100
#: Minimum line span for an inner construct (VHDL process/function/
#: procedure; Verilog/SV always/function/task) to earn its own chunk;
#: smaller ones stay covered by the parent chunk.
MIN_INNER_SPAN = 5

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
#: Dotted identifiers like ``foo.bar`` are kept whole; pure numbers and
#: single letters are dropped (noise for cross-referencing).


def clamp_and_join(lines: list[str], start: int, end: int) -> tuple[int, int, str]:
    """Clamp a construct's 1-based line range to the file's bounds and
    join the resulting lines.

    Returns ``(clamped_start, clamped_end, joined_text)``: the range
    clamp + join shared by every chunker that slices one construct's
    lines out of its file into chunk content.
    """
    clamped_start = max(1, min(start, len(lines)))
    clamped_end = max(clamped_start, min(end, len(lines)))
    return (
        clamped_start,
        clamped_end,
        "\n".join(lines[clamped_start - 1 : clamped_end]),
    )


def extract_code_identifiers(code: str) -> tuple[str, ...]:
    """Identifiers referenced/defined in a code snippet.

    Language-agnostic: first-occurrence order, deduplicated, capped at
    MAX_SYMBOLS. Comments are ignored (``//``, ``#``, ``--`` prefixes at
    token start). Used for documentation code fences and source-code
    chunks — the cross-referencing key that ties documentation to VHDL
    and code.
    """
    seen: dict[str, None] = {}
    for raw in _IDENT_RE.findall(code):
        ident = raw.rstrip(".")
        if len(ident) < 2 or ident.isdigit():
            continue
        if ident not in seen:
            seen[ident] = None
            if len(seen) >= MAX_SYMBOLS:
                break
    return tuple(seen)
