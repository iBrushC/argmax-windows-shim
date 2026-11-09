"""Sentence chunking for long-form generation (shim-internal)."""

from __future__ import annotations

import re

_SPLIT = re.compile(r"(?<=[.!?;])\s+|(?<=[。！？；])")


def split_into_sentences(text: str, min_chars: int = 20) -> list[str]:
    """Split at sentence boundaries, merging fragments shorter than ``min_chars``."""
    parts = [p.strip() for p in _SPLIT.split(text) if p and p.strip()]
    chunks: list[str] = []
    for p in parts:
        if chunks and len(chunks[-1]) < min_chars:
            chunks[-1] = f"{chunks[-1]} {p}"
        else:
            chunks.append(p)
    return chunks or [text]
