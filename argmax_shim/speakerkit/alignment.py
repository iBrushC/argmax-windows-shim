"""Word-to-speaker alignment helpers (shim-internal)."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Protocol


class _Seg(Protocol):
    speaker: int
    start: float
    end: float


def speaker_for_interval(start: float, end: float, segments: Iterable[_Seg], tolerance: float = 0.5) -> int | None:
    """Speaker with the most overlap with [start, end]; else the nearest segment within ``tolerance`` seconds."""
    overlap: dict[int, float] = defaultdict(float)
    nearest: tuple[float, int | None] = (float("inf"), None)
    for s in segments:
        ov = min(end, s.end) - max(start, s.start)
        if ov > 0:
            overlap[s.speaker] += ov
        else:
            dist = max(s.start - end, start - s.end)
            if dist < nearest[0]:
                nearest = (dist, s.speaker)
    if overlap:
        return max(overlap.items(), key=lambda kv: kv[1])[0]
    return nearest[1] if nearest[0] <= tolerance else None


def relabel_by_first_appearance(raw: list[tuple[str, float, float]]) -> list[tuple[int, float, float]]:
    """Map backend labels (e.g. "SPEAKER_03") to 0-based ints in order of first appearance."""
    mapping: dict[str, int] = {}
    out = []
    for label, start, end in sorted(raw, key=lambda r: r[1]):
        if label not in mapping:
            mapping[label] = len(mapping)
        out.append((mapping[label], start, end))
    return out
