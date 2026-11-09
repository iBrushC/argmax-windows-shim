"""
Kokoro-82M (hexgrad/Kokoro-82M) via the `kokoro` package. Shim-only lightweight engine.

Languages: English, Japanese*, Chinese*, Spanish, French, Italian, Portuguese.
(*needs `pip install misaki[ja]` / `misaki[zh]`). German, Russian and Korean are
not supported and fall back to English voices.
"""

from __future__ import annotations

import logging

import numpy as np

from ...core.device import resolve_torch_device
from ...core.errors import MissingDependencyError, warn_once
from ..datatypes import GenerationOptions, Qwen3Language, Qwen3Speaker
from .base import TTSBackend

log = logging.getLogger(__name__)

LANG_CODE = {
    Qwen3Language.ENGLISH: "a",
    Qwen3Language.JAPANESE: "j",
    Qwen3Language.CHINESE: "z",
    Qwen3Language.SPANISH: "e",
    Qwen3Language.FRENCH: "f",
    Qwen3Language.ITALIAN: "i",
    Qwen3Language.PORTUGUESE: "p",
}
DEFAULT_VOICE = {"a": "af_heart", "j": "jf_alpha", "z": "zf_xiaobei", "e": "ef_dora", "f": "ff_siwis", "i": "if_sara", "p": "pf_dora"}
SPEAKER_VOICE = {
    Qwen3Speaker.RYAN: "am_michael",
    Qwen3Speaker.AIDEN: "am_adam",
    Qwen3Speaker.ERIC: "am_eric",
    Qwen3Speaker.DYLAN: "am_puck",
    Qwen3Speaker.SERENA: "af_heart",
    Qwen3Speaker.VIVIAN: "af_bella",
    Qwen3Speaker.SOHEE: "af_nicole",
    Qwen3Speaker.ONO_ANNA: "jf_alpha",
    Qwen3Speaker.UNCLE_FU: "zm_yunxi",
}


class KokoroBackend(TTSBackend):
    sample_rate = 24_000

    def __init__(self, config) -> None:
        super().__init__(config)
        self._pipelines: dict[str, object] = {}

    def load(self) -> None:
        try:
            import kokoro  # noqa: F401
        except ImportError as exc:
            raise MissingDependencyError("kokoro", "tts-kokoro") from exc
        self._pipeline("a")

    def _pipeline(self, code: str):
        if code not in self._pipelines:
            from kokoro import KPipeline

            self._pipelines[code] = KPipeline(lang_code=code, repo_id="hexgrad/Kokoro-82M", device=resolve_torch_device(self.config.device))
        return self._pipelines[code]

    def unload(self) -> None:
        self._pipelines.clear()

    def synthesize(self, text: str, speaker: Qwen3Speaker, language: Qwen3Language, options: GenerationOptions):
        code = LANG_CODE.get(language)
        if code is None:
            warn_once(log, f"kokoro-{language.value}", f"Kokoro has no {language.value} voices; falling back to English.")
            code = "a"
        voice = self.config.voice or SPEAKER_VOICE.get(speaker, DEFAULT_VOICE[code])
        if not voice.startswith(code):
            voice = DEFAULT_VOICE[code]  # voice prefix encodes its language
        parts = []
        for _, _, audio in self._pipeline(code)(text, voice=voice, speed=1.0):
            if audio is not None:
                parts.append(audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio))
        out = np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)
        return out, self.sample_rate
