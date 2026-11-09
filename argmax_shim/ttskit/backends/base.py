"""Backend interface for TTS engines (shim-internal)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..datatypes import GenerationOptions, Qwen3Language, Qwen3Speaker


class TTSBackend(ABC):
    sample_rate: int = 24_000
    supports_instruction: bool = False

    def __init__(self, config) -> None:
        self.config = config

    @abstractmethod
    def load(self) -> None: ...

    def unload(self) -> None:
        pass

    @abstractmethod
    def synthesize(
        self, text: str, speaker: Qwen3Speaker, language: Qwen3Language, options: GenerationOptions
    ) -> tuple[np.ndarray, int]:
        """Return (mono float32 audio, sample_rate) for one text chunk."""
