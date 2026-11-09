"""Backend interface for speech-to-text engines (shim-internal)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ...core.model_store import ASRModelSpec
from ..datatypes import DecodingOptions, TranscriptionResult, TranscriptionSegment, WordTiming


class BaseASRBackend(ABC):
    name: str = "base"
    supports_language_hint: bool = True
    supports_native_vocabulary: bool = False
    """True when the model can take keyword context directly (Qwen3-ASR)."""

    def __init__(self, spec: ASRModelSpec, config) -> None:
        self.spec = spec
        self.config = config

    @abstractmethod
    def load(self) -> None: ...

    def unload(self) -> None:
        self._model = None

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        options: DecodingOptions,
        *,
        context_words: list[str] | None = None,
        streaming: bool = False,
    ) -> TranscriptionResult:
        """Transcribe 16 kHz mono float32 audio. Backends should always fill word timings when they can."""

    def detect_language(self, audio: np.ndarray) -> tuple[str, dict[str, float]]:
        raise NotImplementedError(f"{self.name} does not expose language probabilities")

    def _require(self):
        model = getattr(self, "_model", None)
        if model is None:
            raise RuntimeError("Model not loaded. Call load_models() first.")
        return model


def segments_from_words(
    words: list[WordTiming],
    text: str,
    seg_spans: list[tuple[float, float, str]] | None,
    duration: float,
) -> list[TranscriptionSegment]:
    """Group words into segments using backend-provided spans, or one segment if none."""
    if seg_spans:
        segments = []
        for i, (start, end, seg_text) in enumerate(seg_spans):
            seg_words = [w for w in words if start - 1e-3 <= (w.start + w.end) / 2 <= end + 1e-3]
            segments.append(
                TranscriptionSegment(id=i, seek=0, start=start, end=end, text=" " + seg_text.strip(), words=seg_words)
            )
        return segments
    start = words[0].start if words else 0.0
    end = words[-1].end if words else duration
    return [TranscriptionSegment(id=0, seek=0, start=start, end=end, text=" " + text.strip(), words=list(words))]


def leading_space(word: str) -> str:
    word = word.strip()
    return " " + word if word else word
