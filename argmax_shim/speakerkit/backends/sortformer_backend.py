"""
NVIDIA Sortformer-family diarization via NeMo.

Default checkpoint is nvidia/Nemotron-3-Diarization (up to 8 speakers). If your
NeMo version cannot load it, set SortformerConfig(model_id=
"nvidia/diar_streaming_sortformer_4spk-v2.1") (4 speakers, older NeMo).
"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path

import numpy as np

from ...core.audio import SAMPLE_RATE
from ...core.device import resolve_torch_device
from ...core.errors import MissingDependencyError
from ..alignment import relabel_by_first_appearance
from ..datatypes import DiarizationResult, SpeakerSegment

log = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "nvidia/Nemotron-3-Diarization"
FALLBACK_MODEL_ID = "nvidia/diar_streaming_sortformer_4spk-v2.1"
_LINE = re.compile(r"^\s*([\d.]+)\s+([\d.]+)\s+(\S+)\s*$")


class SortformerBackend:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, model_folder: str | None = None, device: str = "auto") -> None:
        self.model_id = model_id
        self.model_folder = model_folder
        self.device = device
        self._model = None

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from nemo.collections.asr.models import SortformerEncLabelModel
        except ImportError as exc:
            raise MissingDependencyError("nemo_toolkit[asr]", "sortformer") from exc
        nemo_files = list(Path(self.model_folder).glob("*.nemo")) if self.model_folder else []
        try:
            if nemo_files:
                model = SortformerEncLabelModel.restore_from(str(nemo_files[0]), map_location="cpu")
            else:
                # TODO(verify): Nemotron-3-Diarization loading class/API in NeMo 3.x.
                model = SortformerEncLabelModel.from_pretrained(self.model_id)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load {self.model_id} with this NeMo version. Try SortformerConfig(model_id={FALLBACK_MODEL_ID!r})."
            ) from exc
        model = model.to(resolve_torch_device(self.device))
        model.eval()
        self._model = model

    def unload(self) -> None:
        self._model = None

    def diarize(self, audio: np.ndarray, options=None) -> DiarizationResult:  # noqa: ARG002
        import soundfile as sf

        self.ensure_loaded()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "audio.wav")
            sf.write(path, audio, SAMPLE_RATE)
            output = self._model.diarize(audio=[path], batch_size=1)
        lines = output[0] if output and isinstance(output[0], (list, tuple)) else output
        raw = []
        for line in lines or []:
            m = _LINE.match(str(line))
            if m:
                raw.append((m.group(3), float(m.group(1)), float(m.group(2))))
        return DiarizationResult(segments=[SpeakerSegment(s, a, b) for s, a, b in relabel_by_first_appearance(raw)])
