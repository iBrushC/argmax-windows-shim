"""SpeakerKit option and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .._reference import DOCS_DIARIZATION, DOCS_FILE, OSS_SPEAKERKIT, Fidelity, mirrors
from ..whisperkit.datatypes import TranscriptionResult, WordTiming
from .alignment import speaker_for_interval


@mirrors(
    "PyannoteDiarizationOptions",
    module="SpeakerKit",
    docs=OSS_SPEAKERKIT,
    fidelity=Fidelity.APPROXIMATE,
    notes="cluster_distance_threshold is not wired to pyannote.audio yet (TODO). "
    "min/max_number_of_speakers are shim-only.",
)
@dataclass
class PyannoteDiarizationOptions:
    number_of_speakers: int | None = None
    cluster_distance_threshold: float | None = None
    use_exclusive_reconciliation: bool = False
    # ---- shim-only ----
    min_number_of_speakers: int | None = None
    max_number_of_speakers: int | None = None


@mirrors("SpeakerInfoStrategy", module="SpeakerKit", docs=OSS_SPEAKERKIT, notes="Swift type name unverified; cases are .subsegment and .segment.")
class SpeakerInfoStrategy(str, Enum):
    SUBSEGMENT = "subsegment"
    """Split segments at word gaps / speaker changes, then assign speakers (default)."""
    SEGMENT = "segment"
    """Assign one speaker to each transcription segment as a whole."""


@mirrors(
    "DiarizationResult.segments element",
    module="SpeakerKit",
    docs=OSS_SPEAKERKIT,
    fidelity=Fidelity.APPROXIMATE,
    notes="Swift type name unverified. Speakers are 0-based ints ordered by first appearance.",
)
@dataclass(frozen=True)
class SpeakerSegment:
    speaker: int
    start: float
    end: float


@mirrors(
    "SpeakerSegment (from addSpeakerInfo)",
    module="SpeakerKit",
    docs=OSS_SPEAKERKIT,
    fidelity=Fidelity.APPROXIMATE,
    notes="Has .speaker and .text like the Swift type; words and timings are shim additions.",
)
@dataclass
class SpeakerTranscriptionSegment:
    speaker: int | None
    start: float
    end: float
    text: str
    words: list[WordTiming] = field(default_factory=list)


@mirrors("RTTMLine", module="SpeakerKit", docs=OSS_SPEAKERKIT, fidelity=Fidelity.APPROXIMATE)
@dataclass(frozen=True)
class RTTMLine:
    file_name: str
    start: float
    duration: float
    speaker: int
    channel: int = 1

    def __str__(self) -> str:
        return f"SPEAKER {self.file_name} {self.channel} {self.start:.3f} {self.duration:.3f} <NA> <NA> speaker_{self.speaker} <NA> <NA>"

    @property
    @mirrors("RTTMLine.description", module="SpeakerKit", docs=DOCS_DIARIZATION)
    def description(self) -> str:
        return str(self)


@mirrors("DiarizationTimings", module="SpeakerKit", fidelity=Fidelity.APPROXIMATE)
@dataclass
class DiarizationTimings:
    input_audio_seconds: float = 0.0
    full_pipeline: float = 0.0

    @property
    @mirrors("DiarizationTimings.debugDescription", module="SpeakerKit", docs=DOCS_DIARIZATION, fidelity=Fidelity.APPROXIMATE)
    def debug_description(self) -> str:
        return f"audio={self.input_audio_seconds:.2f}s pipeline={self.full_pipeline:.2f}s"


@mirrors("DiarizationResult", module="SpeakerKit", docs=OSS_SPEAKERKIT, fidelity=Fidelity.APPROXIMATE)
@dataclass
class DiarizationResult:
    segments: list[SpeakerSegment]
    timings: DiarizationTimings | None = None
    speaker_embeddings: np.ndarray | None = None
    """Per-speaker centroid embeddings when the backend provides them."""

    @property
    @mirrors("DiarizationResult.speakerCount", module="SpeakerKit", docs=OSS_SPEAKERKIT)
    def speaker_count(self) -> int:
        return len({s.speaker for s in self.segments})

    @mirrors(
        "DiarizationResult.addSpeakerInfo(to:strategy:)",
        module="SpeakerKit",
        docs=DOCS_FILE,
        fidelity=Fidelity.APPROXIMATE,
        notes="Requires word timestamps for .subsegment. max_word_gap and tolerance are shim parameters.",
    )
    def add_speaker_info(
        self,
        to: list[TranscriptionResult],
        strategy: SpeakerInfoStrategy = SpeakerInfoStrategy.SUBSEGMENT,
        max_word_gap: float = 0.8,
        tolerance: float = 0.5,
    ) -> list[list[SpeakerTranscriptionSegment]]:
        """Attach speakers to transcription segments. Returns one list per input result."""
        output: list[list[SpeakerTranscriptionSegment]] = []
        for result in to:
            groups: list[SpeakerTranscriptionSegment] = []
            for seg in result.segments:
                words = seg.words or []
                if strategy is SpeakerInfoStrategy.SEGMENT or not words:
                    spk = speaker_for_interval(seg.start, seg.end, self.segments, tolerance)
                    groups.append(SpeakerTranscriptionSegment(spk, seg.start, seg.end, seg.text, list(words)))
                    continue
                current: list[WordTiming] = []
                current_spk: int | None = None
                for w in words:
                    spk = speaker_for_interval(w.start, w.end, self.segments, tolerance)
                    gap = (w.start - current[-1].end) if current else 0.0
                    if current and (spk != current_spk or gap > max_word_gap):
                        groups.append(_group(current_spk, current))
                        current = []
                    if not current:
                        current_spk = spk
                    current.append(w)
                if current:
                    groups.append(_group(current_spk, current))
            output.append(groups)
        return output


def _group(speaker: int | None, words: list[WordTiming]) -> SpeakerTranscriptionSegment:
    return SpeakerTranscriptionSegment(
        speaker=speaker, start=words[0].start, end=words[-1].end, text="".join(w.word for w in words), words=list(words)
    )
