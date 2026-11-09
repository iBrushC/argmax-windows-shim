"""
NVIDIA Parakeet TDT (v2 English, v3 25 European languages) via NeMo.

NeMo is the official runtime. It is heavy and best on Linux; on Windows, WSL2
or a community ONNX export of Parakeet is usually less painful.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from ...core.audio import SAMPLE_RATE
from ...core.device import resolve_torch_device
from ...core.errors import MissingDependencyError, warn_once
from ..datatypes import DecodingOptions, TranscriptionResult, WordTiming
from .base import BaseASRBackend, leading_space, segments_from_words

log = logging.getLogger(__name__)


class NemoParakeetBackend(BaseASRBackend):
    name = "nemo-parakeet"
    supports_language_hint = False  # Argmax docs: Parakeet only auto-detects and ignores hints.

    def load(self) -> None:
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError as exc:
            raise MissingDependencyError("nemo_toolkit[asr]", "parakeet") from exc
        folder = self.config.model_folder
        nemo_files = list(Path(folder).glob("*.nemo")) if folder else []
        if nemo_files:
            model = nemo_asr.models.ASRModel.restore_from(str(nemo_files[0]))
        else:
            model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.spec.hf_repo)
        model = model.to(resolve_torch_device(self.config.device))
        model.eval()
        self._model = model

    def _run(self, audio):
        model = self._require()
        try:
            out = model.transcribe([audio], batch_size=1, timestamps=True, verbose=False)
        except Exception:  # noqa: BLE001 - older NeMo builds only accept file paths
            import soundfile as sf

            with tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp) / "chunk.wav")
                sf.write(path, audio, SAMPLE_RATE)
                out = model.transcribe([path], batch_size=1, timestamps=True, verbose=False)
        if isinstance(out, tuple):
            out = out[0]
        return out[0]

    def transcribe(self, audio, options: DecodingOptions, *, context_words=None, streaming=False) -> TranscriptionResult:
        if options.language and not options.detect_language:
            warn_once(log, "parakeet-hint", "Parakeet ignores language hints (same as the Argmax SDK).")
        hyp = self._run(audio)
        text = hyp if isinstance(hyp, str) else getattr(hyp, "text", "")
        stamps = getattr(hyp, "timestamp", None) or {}
        words = [
            WordTiming(word=leading_space(str(w.get("word", ""))), tokens=[], start=float(w.get("start", 0.0)), end=float(w.get("end", 0.0)))
            for w in stamps.get("word", [])
        ]
        spans = [(float(s.get("start", 0.0)), float(s.get("end", 0.0)), str(s.get("segment", ""))) for s in stamps.get("segment", [])]
        segments = segments_from_words(words, text, spans, len(audio) / SAMPLE_RATE)
        # TODO: Parakeet v3 auto-detects language internally but NeMo does not expose a label.
        language = "en" if self.spec.english_only else None
        return TranscriptionResult(text=text.strip(), segments=segments, language=language)
