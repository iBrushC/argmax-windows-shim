"""
Mirror of the Swift ``Argmax`` module (Pro SDK).

Like ``import Argmax`` in Swift, this re-exports the open-source types the Pro
examples use (DecodingOptions, AudioProcessor, PyannoteConfig, ...).

    from argmax_shim.argmax import ArgmaxSDK, ArgmaxConfig, WhisperKitPro, WhisperKitProConfig
"""

from ..core.audio import AudioProcessor, AudioStream, AudioStreamContinuation, make_audio_stream
from ..core.model_store import ModelStore, RepoType
from ..speakerkit import (
    DiarizationResult,
    PyannoteConfig,
    PyannoteDiarizationOptions,
    SpeakerInfoStrategy,
    SpeakerKit,
)
from ..whisperkit import (
    ChunkingStrategy,
    DecodingOptions,
    DecodingTask,
    LogLevel,
    TranscriptionResult,
    TranscriptionUtilities,
    WhisperKit,
    WhisperKitConfig,
    WordTiming,
)
from .custom_vocabulary import CustomVocabularyConfig, CustomVocabularyModelState
from .live_transcriber import ArgmaxSource, CustomSource, LiveResult, LiveTranscriber, StreamType
from .sdk import ArgmaxConfig, ArgmaxSDK
from .speakerkit_pro import (
    SortformerConfig,
    SortformerDiarizationOptions,
    SortformerMode,
    SortformerStreamingConfig,
    SpeakerKitDiarizer,
    SpeakerKitPro,
    TranscribeDiarizeResult,
    TranscribeDiarizeResultType,
    TranscribeDiarizeStreamSession,
    WordWithSpeaker,
)
from .streaming import TranscribeStreamResult, TranscribeStreamSession
from .whisperkit_pro import (
    DecodingOptionsPro,
    ModelOptimizationMode,
    StreamTranscriptionMode,
    WhisperKitPro,
    WhisperKitProConfig,
    WhisperKitProUtils,
)

__all__ = [name for name in dir() if not name.startswith("_")]
