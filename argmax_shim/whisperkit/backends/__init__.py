"""Speech-to-text backends. Each one is imported lazily so missing extras only fail when used."""

from __future__ import annotations

from ...core.model_store import ASRBackendKind, ASRModelSpec
from .base import BaseASRBackend


def create_asr_backend(spec: ASRModelSpec, config) -> BaseASRBackend:
    if spec.backend is ASRBackendKind.FASTER_WHISPER:
        from .faster_whisper_backend import FasterWhisperBackend

        return FasterWhisperBackend(spec, config)
    if spec.backend is ASRBackendKind.NEMO_PARAKEET:
        from .nemo_parakeet_backend import NemoParakeetBackend

        return NemoParakeetBackend(spec, config)
    if spec.backend is ASRBackendKind.QWEN_ASR:
        from .qwen_asr_backend import QwenASRBackend

        return QwenASRBackend(spec, config)
    raise ValueError(f"No backend for {spec.backend}")


__all__ = ["BaseASRBackend", "create_asr_backend"]
