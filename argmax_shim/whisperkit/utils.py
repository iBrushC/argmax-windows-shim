"""Result utilities."""

from __future__ import annotations

from dataclasses import replace

from .._reference import DOCS_FILE, mirrors
from .datatypes import TranscriptionResult, TranscriptionTimings


@mirrors("TranscriptionUtilities", module="WhisperKit", docs=DOCS_FILE)
class TranscriptionUtilities:
    @staticmethod
    @mirrors("TranscriptionUtilities.mergeTranscriptionResults(_:)", module="WhisperKit", docs=DOCS_FILE)
    def merge_transcription_results(results: list[TranscriptionResult | None] | None) -> TranscriptionResult:
        """Merge per-chunk results into one, renumbering segments."""
        items = [r for r in (results or []) if r is not None]
        if not items:
            return TranscriptionResult(text="", segments=[], language=None)
        segments = []
        for r in items:
            for s in r.segments:
                segments.append(replace(s, id=len(segments)))
        timings = TranscriptionTimings(
            input_audio_seconds=sum(r.timings.input_audio_seconds for r in items),
            full_pipeline=sum(r.timings.full_pipeline for r in items),
        )
        text = " ".join(r.text.strip() for r in items if r.text.strip())
        return TranscriptionResult(text=text, segments=segments, language=items[0].language, timings=timings)
