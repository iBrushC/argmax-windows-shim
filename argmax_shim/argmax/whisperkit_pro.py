"""WhisperKitPro: WhisperKit plus custom vocabulary and real-time streaming sessions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from .._reference import DOCS_FILE, DOCS_MODELS, DOCS_REALTIME, DOCS_VOCAB, Fidelity, mirrors
from ..core.errors import warn_once
from ..core.model_store import ASRBackendKind, RepoType
from ..whisperkit.config import WhisperKitConfig
from ..whisperkit.datatypes import DecodingOptions, TranscriptionResult
from ..whisperkit.utils import TranscriptionUtilities
from ..whisperkit.whisperkit import WhisperKit
from .custom_vocabulary import CustomVocabularyConfig, CustomVocabularyModelState, FuzzyKeywordCorrector, sanitize_vocabulary

log = logging.getLogger(__name__)


@mirrors(
    "ModelOptimizationMode",
    module="Argmax",
    docs=DOCS_MODELS,
    fidelity=Fidelity.NO_OP,
    notes="Device memory/latency trade-off; ignored by the shim.",
)
class ModelOptimizationMode(str, Enum):
    AUTO = "auto"
    LATENCY_OPTIMIZED = "latencyOptimized"
    MEMORY_OPTIMIZED = "memoryOptimized"


@mirrors(
    "StreamTranscriptionMode",
    module="Argmax",
    docs=DOCS_REALTIME,
    fidelity=Fidelity.APPROXIMATE,
    notes="silence_threshold is a linear RMS level here; the SDK's units may differ. "
    "battery_optimized() behaves like voice_triggered() (device-specific throttling is not mirrored).",
)
@dataclass(frozen=True)
class StreamTranscriptionMode:
    kind: str
    min_process_interval: float = 0.3
    silence_threshold: float = 0.008
    max_buffer_length: float = 20.0
    # ---- shim-only ----
    endpoint_silence: float = 0.8
    """Seconds of silence after which pending hypothesis text is confirmed (shim-only)."""

    @classmethod
    @mirrors("StreamTranscriptionMode.alwaysOn", module="Argmax", docs=DOCS_REALTIME)
    def always_on(cls, max_buffer_length: float = 20.0) -> "StreamTranscriptionMode":
        return cls("alwaysOn", min_process_interval=0.0, max_buffer_length=max_buffer_length)

    @classmethod
    @mirrors(
        "StreamTranscriptionMode.voiceTriggered(silenceThreshold:maxBufferLength:minProcessInterval:)",
        module="Argmax",
        docs=DOCS_REALTIME,
        fidelity=Fidelity.APPROXIMATE,
    )
    def voice_triggered(
        cls, min_process_interval: float = 0.3, silence_threshold: float = 0.008, max_buffer_length: float = 20.0
    ) -> "StreamTranscriptionMode":
        return cls("voiceTriggered", min_process_interval, silence_threshold, max_buffer_length)

    @classmethod
    @mirrors(
        "StreamTranscriptionMode.batteryOptimized",
        module="Argmax",
        docs=DOCS_REALTIME,
        fidelity=Fidelity.NO_OP,
        notes="Alias of voice_triggered() in the shim.",
    )
    def battery_optimized(cls) -> "StreamTranscriptionMode":
        return cls.voice_triggered()


@mirrors("DecodingOptionsPro", module="Argmax", docs=DOCS_REALTIME, fidelity=Fidelity.APPROXIMATE)
@dataclass
class DecodingOptionsPro:
    base: DecodingOptions = field(default_factory=DecodingOptions)
    transcribe_interval: float = 0.1
    stream_transcription_mode: StreamTranscriptionMode = field(default_factory=StreamTranscriptionMode.voice_triggered)
    align_timestamps_to_global: bool = False
    """Shim timestamps are always relative to stream start, so this flag has no effect."""


@mirrors(
    "WhisperKitProConfig",
    module="Argmax",
    docs=DOCS_FILE,
    fidelity=Fidelity.APPROXIMATE,
    notes="Inherits every WhisperKitConfig field (including shim-only device/compute_type).",
)
@dataclass
class WhisperKitProConfig(WhisperKitConfig):
    custom_vocabulary_config: CustomVocabularyConfig | None = None
    model_optimization_mode: ModelOptimizationMode = ModelOptimizationMode.AUTO

    @classmethod
    @mirrors("WhisperKitProConfig.qwen3ASR(optimization:)", module="Argmax", docs=DOCS_MODELS)
    def qwen3_asr(cls, optimization: ModelOptimizationMode = ModelOptimizationMode.AUTO, **kwargs) -> "WhisperKitProConfig":
        return cls(model="qwen3-asr-1.7b", model_repo=RepoType.QWEN_ASR_REPO, model_optimization_mode=optimization, **kwargs)


@mirrors("WhisperKitProUtils", module="Argmax", docs=DOCS_FILE)
class WhisperKitProUtils:
    @staticmethod
    @mirrors("WhisperKitProUtils.mergeTranscriptionResults(_:)", module="Argmax", docs=DOCS_FILE)
    def merge_transcription_results(results) -> TranscriptionResult:
        return TranscriptionUtilities.merge_transcription_results(results)


@mirrors(
    "WhisperKitPro",
    module="Argmax",
    docs=DOCS_FILE,
    fidelity=Fidelity.APPROXIMATE,
    notes="Subclass of WhisperKit, like the Swift class.",
)
class WhisperKitPro(WhisperKit):
    @mirrors("WhisperKitPro.init(_ config: WhisperKitProConfig)", module="Argmax", docs=DOCS_FILE)
    def __init__(self, config: WhisperKitProConfig | None = None) -> None:
        self.config: WhisperKitProConfig = config or WhisperKitProConfig()
        self._vocabulary: list[str] = []
        self._corrector: FuzzyKeywordCorrector | None = None
        super().__init__(self.config)
        vocab_cfg = self.config.custom_vocabulary_config
        if vocab_cfg is not None and vocab_cfg.words:
            self.set_custom_vocabulary(vocab_cfg.words)

    @mirrors(
        "WhisperKitPro.setCustomVocabulary(_:)",
        module="Argmax",
        docs=DOCS_VOCAB,
        fidelity=Fidelity.APPROXIMATE,
        notes="Whisper models: disabled, as in the SDK. Qwen3-ASR: passed as model context. "
        "Parakeet: fuzzy text correction instead of audio keyword spotting.",
    )
    def set_custom_vocabulary(self, words: list[str]) -> None:
        self._vocabulary = sanitize_vocabulary(list(words))
        threshold = self.config.custom_vocabulary_config.match_threshold if self.config.custom_vocabulary_config else 0.82
        self._corrector = FuzzyKeywordCorrector(self._vocabulary, threshold) if self._vocabulary else None
        if self._vocabulary and self.model_spec.backend is ASRBackendKind.FASTER_WHISPER:
            warn_once(log, "vocab-whisper", "Custom Vocabulary is disabled for Whisper models in the Argmax SDK; the shim does the same.")

    @property
    @mirrors("WhisperKitPro.customVocabularyModelState", module="Argmax", docs=DOCS_VOCAB, fidelity=Fidelity.APPROXIMATE)
    def custom_vocabulary_model_state(self) -> CustomVocabularyModelState:
        active = bool(self._vocabulary) and self.model_spec.backend is not ASRBackendKind.FASTER_WHISPER
        return CustomVocabularyModelState.LOADED if active else CustomVocabularyModelState.DISABLED

    @mirrors("WhisperKitPro.makeStreamSession(options:)", module="Argmax", docs=DOCS_REALTIME, fidelity=Fidelity.APPROXIMATE)
    def make_stream_session(self, options: DecodingOptionsPro | None = None):
        from .streaming import TranscribeStreamSession

        return TranscribeStreamSession(self, options or DecodingOptionsPro())

    # -- hooks used by WhisperKit._transcribe_array ---------------------------------
    def _context_words(self) -> list[str] | None:
        if self._vocabulary and self.model_spec.backend is ASRBackendKind.QWEN_ASR:
            return self._vocabulary
        return None

    def _post_process(self, result: TranscriptionResult) -> TranscriptionResult:
        if self._corrector is None or self.model_spec.backend is not ASRBackendKind.NEMO_PARAKEET:
            return result
        for seg in result.segments:
            if seg.words:
                seg.words, n = self._corrector.correct(seg.words)
                if n:
                    seg.text = "".join(w.word for w in seg.words)
        result.text = "".join(s.text for s in result.segments).strip()
        return result
