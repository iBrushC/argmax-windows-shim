"""Shared building blocks used by every kit (audio I/O and model management)."""

from .audio import (
    SAMPLE_RATE,
    AudioDevice,
    AudioProcessor,
    AudioStream,
    AudioStreamContinuation,
    make_audio_stream,
)
from .errors import MissingDependencyError
from .model_store import ModelStore, RepoType, list_supported_models, resolve_asr_model

__all__ = [
    "SAMPLE_RATE",
    "AudioDevice",
    "AudioProcessor",
    "AudioStream",
    "AudioStreamContinuation",
    "make_audio_stream",
    "MissingDependencyError",
    "ModelStore",
    "RepoType",
    "list_supported_models",
    "resolve_asr_model",
]
