"""Fake model backends so the pipeline logic can be tested without downloading models."""

from __future__ import annotations

import numpy as np
import pytest

from argmax_shim.core.audio import SAMPLE_RATE
from argmax_shim.whisperkit.backends.base import BaseASRBackend, segments_from_words
from argmax_shim.whisperkit.datatypes import TranscriptionResult, WordTiming

SCRIPT = "the quick brown fox jumps over the lazy dog near the river bank today".split()
WORD_SECONDS = 0.4


class FakeASRBackend(BaseASRBackend):
    """Pretends word i is spoken during [0.4*i, 0.4*i + 0.3] seconds of the stream."""

    name = "fake"

    def __init__(self, spec, config, script=SCRIPT, offset_of=None):
        super().__init__(spec, config)
        self.script = script
        self.offset_of = offset_of  # callable returning the stream time of buffer start
        self._model = object()

    def load(self):
        pass

    def transcribe(self, audio, options, *, context_words=None, streaming=False):
        start = self.offset_of() if self.offset_of else 0.0
        dur = len(audio) / SAMPLE_RATE
        words = []
        for i, w in enumerate(self.script):
            ws, we = i * WORD_SECONDS, i * WORD_SECONDS + 0.3
            if ws >= start - 1e-6 and we <= start + dur + 1e-6:
                words.append(WordTiming(" " + w, [], ws - start, we - start))
        text = "".join(w.word for w in words)
        return TranscriptionResult(text=text.strip(), segments=segments_from_words(words, text, None, dur), language="en")


@pytest.fixture
def tone():
    def make(seconds: float) -> np.ndarray:
        t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
        return (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    return make
