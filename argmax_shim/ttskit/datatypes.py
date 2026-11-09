"""TTSKit option and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .._reference import OSS_TTSKIT, Fidelity, mirrors


@mirrors(
    "TTSModelVariant (.qwen3TTS_0_6b, .qwen3TTS_1_7b)",
    module="TTSKit",
    docs=OSS_TTSKIT,
    fidelity=Fidelity.APPROXIMATE,
    notes="Swift type name unverified. KOKORO_82M, PIPER and SYSTEM are shim-only lightweight engines; "
    "replace them with .qwen3TTS_0_6b when swapping.",
)
class TTSModelVariant(str, Enum):
    QWEN3_TTS_0_6B = "qwen3TTS_0_6b"
    """Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice. Same model family as the real default."""
    QWEN3_TTS_1_7B = "qwen3TTS_1_7b"
    """Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice. Supports style instructions."""
    # ---- shim-only lightweight engines ----
    KOKORO_82M = "kokoro-82m"
    """hexgrad/Kokoro-82M. ~330 MB, good quality, runs fine on CPU. Recommended light default."""
    PIPER = "piper"
    """Piper ONNX voices (~60 MB each). Fastest neural option, robotic-but-clear."""
    SYSTEM = "system"
    """OS speech engine via pyttsx3 (SAPI5 on Windows, eSpeak on Linux). No download, lowest quality."""

    @property
    @mirrors("(none)", module="-", fidelity=Fidelity.SHIM_ONLY)
    def is_qwen(self) -> bool:
        return self in (TTSModelVariant.QWEN3_TTS_0_6B, TTSModelVariant.QWEN3_TTS_1_7B)


@mirrors(
    "Qwen3Speaker (.ryan, .aiden, .onoAnna, .sohee, .eric, .dylan, .serena, .vivian, .uncleFu)",
    module="TTSKit",
    docs=OSS_TTSKIT,
    notes="Swift type name unverified. Lightweight engines map each speaker to a similar-sounding voice.",
)
class Qwen3Speaker(str, Enum):
    RYAN = "Ryan"
    AIDEN = "Aiden"
    ONO_ANNA = "Ono_Anna"
    SOHEE = "Sohee"
    ERIC = "Eric"
    DYLAN = "Dylan"
    SERENA = "Serena"
    VIVIAN = "Vivian"
    UNCLE_FU = "Uncle_Fu"


@mirrors(
    "Qwen3Language (.english, .chinese, .japanese, .korean, .german, .french, .russian, .portuguese, .spanish, .italian)",
    module="TTSKit",
    docs=OSS_TTSKIT,
    notes="Swift type name unverified. Lightweight engines support a subset of languages.",
)
class Qwen3Language(str, Enum):
    ENGLISH = "English"
    CHINESE = "Chinese"
    JAPANESE = "Japanese"
    KOREAN = "Korean"
    GERMAN = "German"
    FRENCH = "French"
    RUSSIAN = "Russian"
    PORTUGUESE = "Portuguese"
    SPANISH = "Spanish"
    ITALIAN = "Italian"


@mirrors("TextChunkingStrategy (.sentence, .none)", module="TTSKit", docs=OSS_TTSKIT, notes="Swift type name unverified.")
class TextChunkingStrategy(str, Enum):
    NONE = "none"
    SENTENCE = "sentence"


@mirrors(
    "GenerationOptions",
    module="TTSKit",
    docs=OSS_TTSKIT,
    fidelity=Fidelity.APPROXIMATE,
    notes="Sampling fields apply to Qwen backends only. Chunks are generated sequentially; "
    "concurrent_worker_count is accepted and ignored.",
)
@dataclass
class GenerationOptions:
    temperature: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.05
    max_new_tokens: int = 245
    chunking_strategy: TextChunkingStrategy = TextChunkingStrategy.SENTENCE
    concurrent_worker_count: int | None = None
    instruction: str | None = None
    """Natural-language style instruction (1.7B only, as in TTSKit)."""


@mirrors("SpeechTimings", module="TTSKit", docs=OSS_TTSKIT, fidelity=Fidelity.APPROXIMATE, notes="Swift type name unverified.")
@dataclass
class SpeechTimings:
    full_pipeline: float = 0.0
    time_to_first_buffer: float | None = None


@mirrors("SpeechResult", module="TTSKit", docs=OSS_TTSKIT, fidelity=Fidelity.APPROXIMATE, notes="Swift type name unverified; fields audio, sampleRate, audioDuration match README.")
@dataclass
class SpeechResult:
    audio: np.ndarray
    sample_rate: int
    timings: SpeechTimings = field(default_factory=SpeechTimings)

    @property
    @mirrors("SpeechResult.audioDuration", module="TTSKit", docs=OSS_TTSKIT)
    def audio_duration(self) -> float:
        return len(self.audio) / self.sample_rate if self.sample_rate else 0.0


@mirrors(
    "GenerationProgress (progress callback argument)",
    module="TTSKit",
    docs=OSS_TTSKIT,
    fidelity=Fidelity.APPROXIMATE,
    notes="Shim delivers one progress event per text chunk (sentence), not per decoder step. "
    "sample_rate is a shim addition.",
)
@dataclass
class GenerationProgress:
    audio: np.ndarray
    step_time: float | None = None
    """Set on the first event only, like TTSKit's first-step timing."""
    sample_rate: int = 24_000


@mirrors(
    "PlaybackStrategy (.auto, .stream, .buffered(seconds:), .generateFirst)",
    module="TTSKit",
    docs=OSS_TTSKIT,
    fidelity=Fidelity.APPROXIMATE,
    notes="Buffering operates on sentence chunks rather than decoder frames.",
)
@dataclass(frozen=True)
class PlaybackStrategy:
    kind: str
    seconds: float = 0.0

    @classmethod
    @mirrors("PlaybackStrategy.auto", module="TTSKit", docs=OSS_TTSKIT)
    def auto(cls) -> "PlaybackStrategy":
        return cls("auto")

    @classmethod
    @mirrors("PlaybackStrategy.stream", module="TTSKit", docs=OSS_TTSKIT)
    def stream(cls) -> "PlaybackStrategy":
        return cls("stream")

    @classmethod
    @mirrors("PlaybackStrategy.buffered(seconds:)", module="TTSKit", docs=OSS_TTSKIT)
    def buffered(cls, seconds: float) -> "PlaybackStrategy":
        return cls("buffered", seconds)

    @classmethod
    @mirrors("PlaybackStrategy.generateFirst", module="TTSKit", docs=OSS_TTSKIT)
    def generate_first(cls) -> "PlaybackStrategy":
        return cls("generateFirst")


@mirrors("AudioFileFormat (.wav, .m4a)", module="TTSKit", docs=OSS_TTSKIT, notes="Swift type name unverified.")
class AudioFileFormat(str, Enum):
    WAV = "wav"
    M4A = "m4a"
