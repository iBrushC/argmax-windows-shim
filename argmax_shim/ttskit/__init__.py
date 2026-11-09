"""
Mirror of the Swift ``TTSKit`` module (open-source text-to-speech).

    from argmax_shim.ttskit import TTSKit, TTSKitConfig, TTSModelVariant

Default engine is Qwen3-TTS 0.6B, as in TTSKit. For laptops without a GPU use
TTSModelVariant.KOKORO_82M, .PIPER or .SYSTEM (shim-only lightweight engines).
"""

from .audio_output import AudioOutput
from .config import TTSKitConfig
from .datatypes import (
    AudioFileFormat,
    GenerationOptions,
    GenerationProgress,
    PlaybackStrategy,
    Qwen3Language,
    Qwen3Speaker,
    SpeechResult,
    SpeechTimings,
    TextChunkingStrategy,
    TTSModelVariant,
)
from .ttskit import TTSKit

__all__ = [
    "AudioFileFormat",
    "AudioOutput",
    "GenerationOptions",
    "GenerationProgress",
    "PlaybackStrategy",
    "Qwen3Language",
    "Qwen3Speaker",
    "SpeechResult",
    "SpeechTimings",
    "TTSKit",
    "TTSKitConfig",
    "TTSModelVariant",
    "TextChunkingStrategy",
]
