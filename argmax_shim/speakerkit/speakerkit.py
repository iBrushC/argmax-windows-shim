"""The SpeakerKit diarization pipeline."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .._reference import DOCS_DIARIZATION, OSS_SPEAKERKIT, Fidelity, mirrors
from ..core.audio import SAMPLE_RATE, AudioProcessor
from .config import PyannoteConfig
from .datatypes import DiarizationResult, DiarizationTimings, RTTMLine


@mirrors(
    "SpeakerKit",
    module="SpeakerKit",
    docs=OSS_SPEAKERKIT,
    fidelity=Fidelity.APPROXIMATE,
    notes="Synchronous Python methods. Base class of argmax_shim.argmax.SpeakerKitPro.",
)
class SpeakerKit:
    """Speaker diarization ("who spoke when")."""

    @mirrors(
        "SpeakerKit.init(_ config: PyannoteConfig)",
        module="SpeakerKit",
        docs=OSS_SPEAKERKIT,
        notes="Models load lazily on first diarize() unless config.load is True, as in Swift.",
    )
    def __init__(self, config=None) -> None:
        self.config = config or PyannoteConfig()
        self._backend = self.config._create_backend()
        if getattr(self.config, "load", None):
            self.load_models()

    @mirrors("SpeakerKit.loadModels()", module="SpeakerKit", docs=OSS_SPEAKERKIT)
    def load_models(self) -> None:
        self._backend.ensure_loaded()

    @mirrors("SpeakerKit.unloadModels()", module="SpeakerKit", docs=DOCS_DIARIZATION)
    def unload_models(self) -> None:
        self._backend.unload()

    @mirrors(
        "SpeakerKit.diarize(audioArray:options:)",
        module="SpeakerKit",
        docs=DOCS_DIARIZATION,
        fidelity=Fidelity.APPROXIMATE,
        notes="options: PyannoteDiarizationOptions or SortformerDiarizationOptions. audio_path is a shim convenience.",
    )
    def diarize(
        self,
        audio_array: np.ndarray | list[float] | None = None,
        options=None,
        *,
        audio_path: str | Path | None = None,
    ) -> DiarizationResult:
        if (audio_array is None) == (audio_path is None):
            raise ValueError("Pass exactly one of audio_array or audio_path.")
        audio = AudioProcessor.load_audio_as_float_array(audio_path) if audio_path is not None else np.asarray(audio_array, np.float32).reshape(-1)
        started = time.perf_counter()
        result = self._backend.diarize(audio, options)
        result.timings = DiarizationTimings(input_audio_seconds=len(audio) / SAMPLE_RATE, full_pipeline=time.perf_counter() - started)
        return result

    @staticmethod
    @mirrors(
        "SpeakerKit.generateRTTM(from:fileName:)",
        module="SpeakerKit",
        docs=OSS_SPEAKERKIT,
        notes="`from` is a Python keyword, so the first parameter is from_ (or pass positionally).",
    )
    def generate_rttm(from_: DiarizationResult, file_name: str) -> list[RTTMLine]:
        return [RTTMLine(file_name=file_name, start=s.start, duration=s.end - s.start, speaker=s.speaker) for s in from_.segments]
