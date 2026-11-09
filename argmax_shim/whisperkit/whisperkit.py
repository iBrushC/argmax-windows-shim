"""The WhisperKit speech-to-text pipeline."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np

from .._reference import DOCS_FILE, DOCS_MODELS, OSS_WHISPERKIT, Fidelity, mirrors
from ..core.audio import SAMPLE_RATE, AudioProcessor
from ..core.errors import warn_once
from ..core.model_store import infer_spec_from_folder, resolve_asr_model
from .backends import BaseASRBackend, create_asr_backend
from .config import WhisperKitConfig
from .datatypes import DecodingOptions, TranscriptionProgress, TranscriptionResult, TranscriptionTimings

log = logging.getLogger(__name__)


@mirrors("ModelState", module="WhisperKit", docs=OSS_WHISPERKIT)
class ModelState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    UNLOADING = "unloading"
    PREWARMED = "prewarmed"


@mirrors(
    "WhisperKit",
    module="WhisperKit",
    docs=OSS_WHISPERKIT,
    fidelity=Fidelity.APPROXIMATE,
    notes="Synchronous Python methods; the Swift methods are async/throwing. Same model names, "
    "but inference runs open-source checkpoints (faster-whisper / NeMo / qwen-asr).",
)
class WhisperKit:
    """Speech-to-text pipeline. Also the base class of :class:`argmax_shim.argmax.WhisperKitPro`."""

    @mirrors(
        "WhisperKit.init(_ config: WhisperKitConfig)",
        module="WhisperKit",
        docs=OSS_WHISPERKIT,
        notes="Loads models immediately unless config.load is False (then lazily on first use).",
    )
    def __init__(self, config: WhisperKitConfig | None = None) -> None:
        self.config = config or WhisperKitConfig()
        self.audio_processor = AudioProcessor()
        spec = None
        if self.config.model is None and self.config.model_folder:
            spec = infer_spec_from_folder(self.config.model_folder)
        self.model_spec = spec or resolve_asr_model(self.config.model)
        self.model_state = ModelState.UNLOADED
        self._backend: BaseASRBackend | None = None
        self._lock = threading.RLock()
        if self.config.load is not False:
            self.load_models()

    @property
    @mirrors("WhisperKit.modelVariant", module="WhisperKit", fidelity=Fidelity.APPROXIMATE, notes="Returns the Argmax model name string.")
    def model_variant(self) -> str:
        return self.model_spec.argmax_name

    @mirrors(
        "WhisperKit.loadModels(prewarmMode:)",
        module="WhisperKit",
        docs=DOCS_MODELS,
        notes="prewarm_mode is a no-op (no CoreML compilation step).",
    )
    def load_models(self, prewarm_mode: bool = False) -> None:  # noqa: ARG002
        with self._lock:
            if self._backend is not None:
                return
            self.model_state = ModelState.LOADING
            log.info("Loading %s via %s", self.model_spec.argmax_name, self.model_spec.backend.value)
            backend = create_asr_backend(self.model_spec, self.config)
            try:
                backend.load()
            except Exception:
                self.model_state = ModelState.UNLOADED
                raise
            self._backend = backend
            self.model_state = ModelState.LOADED

    @mirrors("WhisperKit.unloadModels()", module="WhisperKit", docs=OSS_WHISPERKIT)
    def unload_models(self) -> None:
        with self._lock:
            if self._backend is not None:
                self._backend.unload()
            self._backend = None
            self.model_state = ModelState.UNLOADED

    @mirrors(
        "WhisperKit.transcribe(audioPath:decodeOptions:callback:) / transcribe(audioArray:decodeOptions:callback:)",
        module="WhisperKit",
        docs=DOCS_FILE,
        fidelity=Fidelity.APPROXIMATE,
        notes="Pass exactly one of audio_path / audio_array. Returns one result for the whole input; "
        "callback fires once at the end and cannot cancel.",
    )
    def transcribe(
        self,
        audio_path: str | Path | None = None,
        audio_array: np.ndarray | list[float] | None = None,
        decode_options: DecodingOptions | None = None,
        callback: Callable[[TranscriptionProgress], bool | None] | None = None,
    ) -> list[TranscriptionResult]:
        if (audio_path is None) == (audio_array is None):
            raise ValueError("Pass exactly one of audio_path or audio_array.")
        if audio_path is not None:
            audio = AudioProcessor.load_audio_as_float_array(audio_path)
        else:
            audio = np.asarray(audio_array, dtype=np.float32).reshape(-1)
        result = self._transcribe_array(audio, decode_options or DecodingOptions())
        if callback is not None:
            callback(TranscriptionProgress(timings=result.timings, text=result.text))
        return [result]

    @mirrors(
        "WhisperKit.detectLanguage(audioPath:) / detectLangauge(audioArray:)",
        module="WhisperKit",
        docs=OSS_WHISPERKIT,
        fidelity=Fidelity.APPROXIMATE,
        notes="Whisper backends only. For Parakeet/Qwen use DecodingOptions(detect_language=True) and read result.language.",
    )
    def detect_language(
        self, audio_path: str | Path | None = None, audio_array: np.ndarray | None = None
    ) -> tuple[str, dict[str, float]]:
        if (audio_path is None) == (audio_array is None):
            raise ValueError("Pass exactly one of audio_path or audio_array.")
        audio = AudioProcessor.load_audio_as_float_array(audio_path) if audio_path is not None else np.asarray(audio_array, np.float32)
        backend = self._ensure_loaded()
        with self._lock:
            return backend.detect_language(audio)

    # ------------------------------------------------------------------
    # Internals shared with WhisperKitPro and the streaming session
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> BaseASRBackend:
        if self._backend is None:
            self.load_models()
        assert self._backend is not None
        return self._backend

    def _context_words(self) -> list[str] | None:
        """Hook for WhisperKitPro custom vocabulary."""
        return None

    def _post_process(self, result: TranscriptionResult) -> TranscriptionResult:
        """Hook for WhisperKitPro custom vocabulary."""
        return result

    def _transcribe_array(self, audio: np.ndarray, options: DecodingOptions, *, streaming: bool = False) -> TranscriptionResult:
        backend = self._ensure_loaded()
        if options.language and not options.detect_language and not backend.supports_language_hint:
            warn_once(log, f"hint-{backend.name}", f"{backend.name} ignores language hints; it always auto-detects.")
        started = time.perf_counter()
        with self._lock:
            result = backend.transcribe(audio, options, context_words=self._context_words(), streaming=streaming)
        result.timings = TranscriptionTimings(input_audio_seconds=len(audio) / SAMPLE_RATE, full_pipeline=time.perf_counter() - started)
        result = self._post_process(result)
        if not options.word_timestamps:
            result.segments = [replace(s, words=None) for s in result.segments]
        return result
