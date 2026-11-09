"""
Piper ONNX voices (rhasspy/piper-voices on Hugging Face). Shim-only, very light, CPU-friendly.

One voice per language by default; override with TTSKitConfig(voice="en_US-ryan-high").
Japanese and Korean have no Piper voices and fall back to English.
"""

from __future__ import annotations

import logging

import numpy as np

from ...core.errors import MissingDependencyError, warn_once
from ..datatypes import GenerationOptions, Qwen3Language, Qwen3Speaker
from .base import TTSBackend

log = logging.getLogger(__name__)

DEFAULT_VOICE = {
    Qwen3Language.ENGLISH: "en_US-lessac-medium",
    Qwen3Language.GERMAN: "de_DE-thorsten-medium",
    Qwen3Language.FRENCH: "fr_FR-siwis-medium",
    Qwen3Language.SPANISH: "es_ES-davefx-medium",
    Qwen3Language.ITALIAN: "it_IT-riccardo-x_low",
    Qwen3Language.RUSSIAN: "ru_RU-irina-medium",
    Qwen3Language.PORTUGUESE: "pt_BR-faber-medium",
    Qwen3Language.CHINESE: "zh_CN-huayan-medium",
}


def voice_path(key: str) -> str:
    """'en_US-lessac-medium' -> 'en/en_US/lessac/medium/en_US-lessac-medium'."""
    locale, name, quality = key.split("-", 2)
    return f"{locale.split('_')[0]}/{locale}/{name}/{quality}/{key}"


class PiperBackend(TTSBackend):
    sample_rate = 22_050

    def __init__(self, config) -> None:
        super().__init__(config)
        self._voices: dict[str, object] = {}

    def load(self) -> None:
        try:
            import piper  # noqa: F401
        except ImportError as exc:
            raise MissingDependencyError("piper-tts", "tts-piper") from exc
        self._voice(self.config.voice or DEFAULT_VOICE[Qwen3Language.ENGLISH])

    def _voice(self, key: str):
        if key not in self._voices:
            from huggingface_hub import hf_hub_download
            from piper import PiperVoice

            model = hf_hub_download("rhasspy/piper-voices", voice_path(key) + ".onnx")
            hf_hub_download("rhasspy/piper-voices", voice_path(key) + ".onnx.json")
            self._voices[key] = PiperVoice.load(model)
        return self._voices[key]

    def unload(self) -> None:
        self._voices.clear()

    def synthesize(self, text: str, speaker: Qwen3Speaker, language: Qwen3Language, options: GenerationOptions):  # noqa: ARG002
        key = self.config.voice or DEFAULT_VOICE.get(language)
        if key is None:
            warn_once(log, f"piper-{language.value}", f"Piper has no default {language.value} voice; using English.")
            key = DEFAULT_VOICE[Qwen3Language.ENGLISH]
        voice = self._voice(key)
        rate = int(getattr(getattr(voice, "config", None), "sample_rate", self.sample_rate))
        chunks = list(voice.synthesize(text))
        if chunks and hasattr(chunks[0], "audio_float_array"):  # piper-tts >= 1.3
            audio = np.concatenate([c.audio_float_array for c in chunks])
            rate = int(getattr(chunks[0], "sample_rate", rate))
        else:  # older API: raw int16 PCM
            raw = b"".join(voice.synthesize_stream_raw(text))
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        self.sample_rate = rate
        return audio.astype(np.float32), rate
