"""
Mirror of the Swift ``SpeakerKit`` module (open-source speaker diarization).

    from argmax_shim.speakerkit import SpeakerKit, PyannoteConfig, PyannoteDiarizationOptions
"""

from ..core.audio import AudioProcessor
from .config import PyannoteConfig
from .datatypes import (
    DiarizationResult,
    DiarizationTimings,
    PyannoteDiarizationOptions,
    RTTMLine,
    SpeakerInfoStrategy,
    SpeakerSegment,
    SpeakerTranscriptionSegment,
)
from .speakerkit import SpeakerKit

__all__ = [
    "AudioProcessor",
    "DiarizationResult",
    "DiarizationTimings",
    "PyannoteConfig",
    "PyannoteDiarizationOptions",
    "RTTMLine",
    "SpeakerInfoStrategy",
    "SpeakerKit",
    "SpeakerSegment",
    "SpeakerTranscriptionSegment",
]
