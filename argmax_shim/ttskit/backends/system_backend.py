"""
Operating-system TTS via pyttsx3 (SAPI5 on Windows, eSpeak NG on Linux). Shim-only.

No model download and near-zero memory, but robotic output. Language and
speaker mostly depend on the voices installed on the machine.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ...core.audio import resample
from ...core.errors import MissingDependencyError
from ..datatypes import GenerationOptions, Qwen3Language, Qwen3Speaker
from .base import TTSBackend


class SystemBackend(TTSBackend):
    sample_rate = 24_000

    def load(self) -> None:
        try:
            import pyttsx3
        except ImportError as exc:
            raise MissingDependencyError("pyttsx3", "tts-system") from exc
        self._engine = pyttsx3.init()
        if self.config.voice:
            for v in self._engine.getProperty("voices"):
                if self.config.voice.lower() in (v.id.lower(), (v.name or "").lower()):
                    self._engine.setProperty("voice", v.id)
                    break

    def synthesize(self, text: str, speaker: Qwen3Speaker, language: Qwen3Language, options: GenerationOptions):  # noqa: ARG002
        import soundfile as sf

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "out.wav")
            self._engine.save_to_file(text, path)
            self._engine.runAndWait()
            audio, rate = sf.read(path, dtype="float32", always_2d=True)
        return resample(audio.mean(axis=1), rate, self.sample_rate), self.sample_rate
