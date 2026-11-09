"""Text-to-speech engines, imported lazily so missing extras only fail when used."""

from __future__ import annotations

from ..datatypes import TTSModelVariant
from .base import TTSBackend


def create_tts_backend(config) -> TTSBackend:
    model = TTSModelVariant(config.model)
    if model.is_qwen:
        from .qwen_tts_backend import QwenTTSBackend

        return QwenTTSBackend(config)
    if model is TTSModelVariant.KOKORO_82M:
        from .kokoro_backend import KokoroBackend

        return KokoroBackend(config)
    if model is TTSModelVariant.PIPER:
        from .piper_backend import PiperBackend

        return PiperBackend(config)
    if model is TTSModelVariant.SYSTEM:
        from .system_backend import SystemBackend

        return SystemBackend(config)
    raise ValueError(f"Unsupported TTS model {model}")


__all__ = ["TTSBackend", "create_tts_backend"]
