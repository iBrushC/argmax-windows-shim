"""
Qwen3-ASR via the official `qwen-asr` package (transformers backend).

Word timestamps come from Qwen3-ForcedAligner-0.6B, which is loaded alongside
the ASR model. NOTE: `qwen-asr` and `qwen-tts` currently need different
transformers versions; keep them in separate environments.
"""

from __future__ import annotations

import logging

from ...core.audio import SAMPLE_RATE
from ...core.device import resolve_torch_device, torch_dtype_for
from ...core.errors import MissingDependencyError, warn_once
from ..datatypes import DecodingOptions, TranscriptionResult, WordTiming
from .base import BaseASRBackend, leading_space, segments_from_words

log = logging.getLogger(__name__)

ISO_TO_QWEN = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic", "de": "German", "fr": "French",
    "es": "Spanish", "pt": "Portuguese", "id": "Indonesian", "it": "Italian", "ko": "Korean", "ru": "Russian",
    "th": "Thai", "vi": "Vietnamese", "ja": "Japanese", "tr": "Turkish", "hi": "Hindi", "ms": "Malay",
    "nl": "Dutch", "sv": "Swedish", "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech",
    "tl": "Filipino", "fil": "Filipino", "fa": "Persian", "el": "Greek", "hu": "Hungarian",
    "mk": "Macedonian", "ro": "Romanian",
}
QWEN_TO_ISO = {v.lower(): k for k, v in ISO_TO_QWEN.items() if k != "fil"}


class QwenASRBackend(BaseASRBackend):
    name = "qwen-asr"
    supports_language_hint = True
    supports_native_vocabulary = True

    def load(self) -> None:
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise MissingDependencyError("qwen-asr", "qwen-asr") from exc
        device = resolve_torch_device(self.config.device)
        dtype = torch_dtype_for(device)
        device_map = "cuda:0" if device == "cuda" else device
        if device == "cpu":
            warn_once(log, "qwen-cpu", "Qwen3-ASR on CPU is slow. Parakeet or Whisper turbo are better CPU choices.")
        kwargs = dict(dtype=dtype, device_map=device_map, max_inference_batch_size=1, max_new_tokens=1024)
        aligner = self.spec.aux_repos.get("forced_aligner")
        if aligner:
            kwargs["forced_aligner"] = aligner
            kwargs["forced_aligner_kwargs"] = dict(dtype=dtype, device_map=device_map)
        self._model = Qwen3ASRModel.from_pretrained(self.config.model_folder or self.spec.hf_repo, **kwargs)

    def transcribe(self, audio, options: DecodingOptions, *, context_words=None, streaming=False) -> TranscriptionResult:
        model = self._require()
        iso = options.resolved_language()
        kwargs = dict(audio=(audio, SAMPLE_RATE), language=ISO_TO_QWEN.get(iso) if iso else None, return_time_stamps=True)
        if context_words:
            # TODO(verify): kwarg name for Qwen3-ASR context biasing in the qwen-asr package.
            kwargs["context"] = ", ".join(context_words)
        try:
            result = model.transcribe(**kwargs)[0]
        except TypeError:
            kwargs.pop("context", None)
            warn_once(log, "qwen-context", "This qwen-asr version rejected `context`; custom vocabulary is disabled.")
            result = model.transcribe(**kwargs)[0]
        words = _parse_timestamps(getattr(result, "time_stamps", None))
        language = getattr(result, "language", None)
        iso_out = QWEN_TO_ISO.get(str(language).lower(), language) if language else None
        text = str(getattr(result, "text", ""))
        return TranscriptionResult(
            text=text.strip(),
            segments=segments_from_words(words, text, None, len(audio) / SAMPLE_RATE),
            language=iso_out,
        )


def _field(item, *names, default=None):
    for n in names:
        if isinstance(item, dict) and n in item:
            return item[n]
        if hasattr(item, n):
            return getattr(item, n)
    return default


def _parse_timestamps(stamps) -> list[WordTiming]:
    """Defensive parser. TODO(verify): exact structure of Qwen3-ASR `time_stamps` items."""
    if not stamps:
        return []
    items = _field(stamps, "items", default=stamps)
    if isinstance(items, (list, tuple)) and items and isinstance(items[0], (list, tuple)):
        items = [x for group in items for x in group]  # one list per input audio -> flatten
    words = []
    for it in items or []:
        text = _field(it, "text", "word", "token", default="")
        start = _field(it, "start_time", "start", default=None)
        end = _field(it, "end_time", "end", default=None)
        if start is None or end is None:
            continue
        words.append(WordTiming(word=leading_space(str(text)), tokens=[], start=float(start), end=float(end)))
    return words
