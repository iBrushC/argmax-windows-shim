"""OpenAI Whisper via faster-whisper (CTranslate2). Works on Windows, Linux, CPU and CUDA."""

from __future__ import annotations

import numpy as np

from ...core.errors import MissingDependencyError
from ...core.model_store import download_hf_snapshot
from ..datatypes import ChunkingStrategy, DecodingOptions, TranscriptionResult, TranscriptionSegment, WordTiming
from .base import BaseASRBackend


class FasterWhisperBackend(BaseASRBackend):
    name = "faster-whisper"
    supports_language_hint = True

    def load(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise MissingDependencyError("faster-whisper", "whisper") from exc
        device, compute_type = self._device()
        source = self.config.model_folder or download_hf_snapshot(self.spec.hf_repo, enabled=self.config.download)
        self._model = WhisperModel(str(source), device=device, compute_type=compute_type)

    def _device(self) -> tuple[str, str]:
        device = self.config.device
        if device == "auto":
            try:
                import ctranslate2

                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:  # noqa: BLE001
                device = "cpu"
        compute_type = self.config.compute_type
        if compute_type == "auto":
            compute_type = "float16" if device.startswith("cuda") else "int8"
        return device, compute_type

    def transcribe(self, audio, options: DecodingOptions, *, context_words=None, streaming=False) -> TranscriptionResult:
        model = self._require()
        language = "en" if self.spec.english_only else options.resolved_language()
        temps = tuple(min(1.0, round(options.temperature + 0.2 * i, 2)) for i in range(options.temperature_fallback_count + 1))
        segments_iter, info = model.transcribe(
            audio,
            language=language,
            task=options.task.value,
            temperature=temps,
            word_timestamps=True,
            vad_filter=(options.chunking_strategy is ChunkingStrategy.VAD) and not streaming,
            initial_prompt=options.prompt_text,
            hotwords=" ".join(context_words) if context_words else None,
            without_timestamps=options.without_timestamps,
            compression_ratio_threshold=options.compression_ratio_threshold,
            log_prob_threshold=options.log_prob_threshold,
            no_speech_threshold=options.no_speech_threshold,
            condition_on_previous_text=not streaming,
        )
        segments = []
        for s in segments_iter:
            words = [
                WordTiming(word=w.word, tokens=[], start=float(w.start), end=float(w.end), probability=float(w.probability))
                for w in (s.words or [])
            ]
            segments.append(
                TranscriptionSegment(
                    id=s.id,
                    seek=s.seek,
                    start=float(s.start),
                    end=float(s.end),
                    text=s.text,
                    tokens=list(s.tokens),
                    avg_logprob=float(s.avg_logprob),
                    compression_ratio=float(s.compression_ratio),
                    no_speech_prob=float(s.no_speech_prob),
                    words=words,
                )
            )
        return TranscriptionResult(text="".join(s.text for s in segments).strip(), segments=segments, language=info.language)

    def detect_language(self, audio: np.ndarray) -> tuple[str, dict[str, float]]:
        model = self._require()
        if self.spec.english_only:
            return "en", {"en": 1.0}
        try:
            language, _, all_probs = model.detect_language(audio)
            return language, {lang: float(p) for lang, p in all_probs}
        except (AttributeError, TypeError):
            _, info = model.transcribe(audio[: 30 * 16_000], language=None)
            probs = info.all_language_probs or [(info.language, info.language_probability)]
            return info.language, {lang: float(p) for lang, p in probs}
