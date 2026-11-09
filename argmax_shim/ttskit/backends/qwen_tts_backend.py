"""Qwen3-TTS CustomVoice models via the official `qwen-tts` package (same family TTSKit runs)."""

from __future__ import annotations

import logging

import numpy as np

from ...core.device import resolve_torch_device, torch_dtype_for
from ...core.errors import MissingDependencyError, warn_once
from ..datatypes import GenerationOptions, Qwen3Language, Qwen3Speaker, TTSModelVariant
from .base import TTSBackend

log = logging.getLogger(__name__)

REPOS = {
    TTSModelVariant.QWEN3_TTS_0_6B: "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    TTSModelVariant.QWEN3_TTS_1_7B: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
}


class QwenTTSBackend(TTSBackend):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.variant = TTSModelVariant(config.model)
        self.supports_instruction = self.variant is TTSModelVariant.QWEN3_TTS_1_7B
        self._model = None

    def load(self) -> None:
        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise MissingDependencyError("qwen-tts", "tts-qwen") from exc
        device = resolve_torch_device(self.config.device)
        if device == "cpu":
            warn_once(log, "qwen-tts-cpu", "Qwen3-TTS on CPU is slow; TTSModelVariant.KOKORO_82M is a lighter dev option.")
        self._model = Qwen3TTSModel.from_pretrained(
            self.config.model_folder or REPOS[self.variant],
            device_map="cuda:0" if device == "cuda" else device,
            dtype=torch_dtype_for(device),
        )

    def unload(self) -> None:
        self._model = None

    def synthesize(self, text: str, speaker: Qwen3Speaker, language: Qwen3Language, options: GenerationOptions):
        if self._model is None:
            raise RuntimeError("Model not loaded")
        kwargs = dict(text=text, language=language.value, speaker=self._speaker_name(speaker))
        if options.instruction and self.supports_instruction:
            kwargs["instruct"] = options.instruction
        sampling = dict(
            temperature=options.temperature,
            top_k=options.top_k,
            repetition_penalty=options.repetition_penalty,
            max_new_tokens=options.max_new_tokens,
        )
        try:
            wavs, sample_rate = self._model.generate_custom_voice(**kwargs, **sampling)
        except TypeError:  # TODO(verify): sampling kwarg names accepted by qwen-tts
            warn_once(log, "qwen-tts-sampling", "This qwen-tts version rejected sampling kwargs; using its defaults.")
            wavs, sample_rate = self._model.generate_custom_voice(**kwargs)
        self.sample_rate = int(sample_rate)
        return np.asarray(wavs[0], dtype=np.float32).reshape(-1), self.sample_rate

    def _speaker_name(self, speaker: Qwen3Speaker) -> str:
        supported = getattr(self._model, "get_supported_speakers", None)
        if callable(supported):
            for name in supported():
                if str(name).lower() == speaker.value.lower():
                    return str(name)
        return speaker.value
