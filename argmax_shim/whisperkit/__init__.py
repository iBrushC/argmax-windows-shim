"""
Mirror of the Swift ``WhisperKit`` module (open-source speech-to-text).

    from argmax_shim.whisperkit import WhisperKit, WhisperKitConfig, DecodingOptions
"""

from ..core.audio import AudioDevice, AudioProcessor
from ..core.model_store import ModelStore, RepoType
from .config import LogLevel, WhisperKitConfig
from .datatypes import (
    ChunkingStrategy,
    DecodingOptions,
    DecodingTask,
    TranscriptionProgress,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionTimings,
    WordTiming,
)
from .utils import TranscriptionUtilities
from .whisperkit import ModelState, WhisperKit

__all__ = [
    "AudioDevice",
    "AudioProcessor",
    "ChunkingStrategy",
    "DecodingOptions",
    "DecodingTask",
    "LogLevel",
    "ModelState",
    "ModelStore",
    "RepoType",
    "TranscriptionProgress",
    "TranscriptionResult",
    "TranscriptionSegment",
    "TranscriptionTimings",
    "TranscriptionUtilities",
    "WhisperKit",
    "WhisperKitConfig",
    "WordTiming",
]
