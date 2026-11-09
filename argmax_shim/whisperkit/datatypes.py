"""WhisperKit option and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .._reference import DOCS_FILE, OSS_WHISPERKIT, Fidelity, mirrors


@mirrors("DecodingTask", module="WhisperKit", docs=OSS_WHISPERKIT)
class DecodingTask(str, Enum):
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"


@mirrors("ChunkingStrategy", module="WhisperKit", docs=DOCS_FILE)
class ChunkingStrategy(str, Enum):
    NONE = "none"
    VAD = "vad"


@mirrors(
    "DecodingOptions",
    module="WhisperKit",
    docs=DOCS_FILE,
    fidelity=Fidelity.APPROXIMATE,
    notes="Subset of fields. Whisper-specific fields are ignored by Parakeet and Qwen3-ASR backends. "
    "prompt_text is shim-only (Swift uses promptTokens).",
)
@dataclass
class DecodingOptions:
    verbose: bool = False
    task: DecodingTask = DecodingTask.TRANSCRIBE
    language: str | None = None
    temperature: float = 0.0
    temperature_fallback_count: int = 5
    top_k: int = 5
    use_prefill_prompt: bool = True
    detect_language: bool | None = None
    skip_special_tokens: bool = False
    without_timestamps: bool = False
    word_timestamps: bool = False
    prompt_tokens: list[int] | None = None
    compression_ratio_threshold: float | None = 2.4
    log_prob_threshold: float | None = -1.0
    no_speech_threshold: float | None = 0.6
    chunking_strategy: ChunkingStrategy | None = None
    # ---- shim-only ----
    prompt_text: str | None = None
    """Text prompt passed to Whisper as initial_prompt (shim-only)."""

    @mirrors("(SDK-internal language default)", module="WhisperKit", docs=DOCS_FILE, fidelity=Fidelity.SHIM_ONLY)
    def resolved_language(self) -> str | None:
        """Argmax default: hint English unless ``detect_language`` is True (then auto-detect)."""
        if self.detect_language:
            return None
        return self.language or "en"


@mirrors("WordTiming", module="WhisperKit", docs=OSS_WHISPERKIT)
@dataclass
class WordTiming:
    word: str
    """Word text with a leading space, as WhisperKit emits it (so ''.join(words) reads naturally)."""
    tokens: list[int]
    start: float
    end: float
    probability: float = 1.0


@mirrors("TranscriptionSegment", module="WhisperKit", docs=OSS_WHISPERKIT, fidelity=Fidelity.APPROXIMATE)
@dataclass
class TranscriptionSegment:
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int] = field(default_factory=list)
    avg_logprob: float = 0.0
    compression_ratio: float = 1.0
    no_speech_prob: float = 0.0
    words: list[WordTiming] | None = None


@mirrors(
    "TranscriptionTimings",
    module="WhisperKit",
    docs=OSS_WHISPERKIT,
    fidelity=Fidelity.APPROXIMATE,
    notes="Only overall timings; the real struct breaks down encoder/decoder/prefill latency.",
)
@dataclass
class TranscriptionTimings:
    input_audio_seconds: float = 0.0
    full_pipeline: float = 0.0

    @property
    @mirrors("TranscriptionTimings.realTimeFactor", module="WhisperKit", fidelity=Fidelity.APPROXIMATE)
    def real_time_factor(self) -> float:
        return self.full_pipeline / self.input_audio_seconds if self.input_audio_seconds else 0.0


@mirrors("TranscriptionResult", module="WhisperKit", docs=DOCS_FILE)
@dataclass
class TranscriptionResult:
    text: str
    segments: list[TranscriptionSegment]
    language: str | None
    timings: TranscriptionTimings = field(default_factory=TranscriptionTimings)

    @property
    @mirrors(
        "result.segments.flatMap { $0.words ?? [] }",
        module="WhisperKit",
        fidelity=Fidelity.SHIM_ONLY,
        notes="Convenience accessor; write the flatMap explicitly in Swift.",
    )
    def all_words(self) -> list[WordTiming]:
        return [w for s in self.segments for w in (s.words or [])]


@mirrors(
    "TranscriptionProgress",
    module="WhisperKit",
    docs=OSS_WHISPERKIT,
    fidelity=Fidelity.APPROXIMATE,
    notes="The shim reports progress once, after decoding finishes.",
)
@dataclass
class TranscriptionProgress:
    timings: TranscriptionTimings
    text: str
    tokens: list[int] = field(default_factory=list)
