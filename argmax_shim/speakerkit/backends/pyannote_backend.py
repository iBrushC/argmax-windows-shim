"""pyannote.audio 4.x Community-1 pipeline."""

from __future__ import annotations

import logging
import os

import numpy as np

from ...core.audio import SAMPLE_RATE
from ...core.device import resolve_torch_device
from ...core.errors import MissingDependencyError, warn_once
from ..alignment import relabel_by_first_appearance
from ..datatypes import DiarizationResult, SpeakerSegment

log = logging.getLogger(__name__)

REPOS = {"community": "pyannote/speaker-diarization-community-1"}


class PyannoteBackend:
    def __init__(self, variant: str = "community", model_folder: str | None = None, hf_token: str | None = None, device: str = "auto") -> None:
        if variant == "precision":
            warn_once(
                log,
                "precision",
                "pyannote 'precision' (Precision-2) is a closed pyannoteAI model; the shim uses Community-1 instead.",
            )
            variant = "community"
        if variant not in REPOS:
            raise ValueError(f"Unknown pyannote variant {variant!r}; expected one of {list(REPOS)}")
        self.repo = model_folder or REPOS[variant]
        self.hf_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        self.device = device
        self._pipeline = None

    def ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        try:
            import torch
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise MissingDependencyError("pyannote.audio", "pyannote") from exc
        try:
            pipeline = Pipeline.from_pretrained(self.repo, token=self.hf_token)
        except TypeError:  # pyannote.audio 3.x used use_auth_token
            pipeline = Pipeline.from_pretrained(self.repo, use_auth_token=self.hf_token)
        if pipeline is None:
            raise RuntimeError(
                f"Could not load {self.repo}. Accept its terms on Hugging Face and set HF_TOKEN (or PyannoteConfig.hf_token)."
            )
        pipeline.to(torch.device(resolve_torch_device(self.device)))
        self._pipeline = pipeline

    def unload(self) -> None:
        self._pipeline = None

    def diarize(self, audio: np.ndarray, options=None) -> DiarizationResult:
        import torch

        self.ensure_loaded()
        kwargs = {}
        exclusive = False
        if options is not None:
            if getattr(options, "number_of_speakers", None):
                kwargs["num_speakers"] = options.number_of_speakers
            if getattr(options, "min_number_of_speakers", None):
                kwargs["min_speakers"] = options.min_number_of_speakers
            if getattr(options, "max_number_of_speakers", None):
                kwargs["max_speakers"] = options.max_number_of_speakers
            if getattr(options, "cluster_distance_threshold", None) is not None:
                warn_once(log, "cluster-threshold", "cluster_distance_threshold is not wired up in the shim yet (TODO).")
            exclusive = bool(getattr(options, "use_exclusive_reconciliation", False))
        waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
        output = self._pipeline({"waveform": waveform, "sample_rate": SAMPLE_RATE}, **kwargs)
        if exclusive and hasattr(output, "exclusive_speaker_diarization"):
            annotation = output.exclusive_speaker_diarization
        else:
            annotation = getattr(output, "speaker_diarization", output)
        raw = [(str(label), float(turn.start), float(turn.end)) for turn, _, label in annotation.itertracks(yield_label=True)]
        segments = [SpeakerSegment(spk, s, e) for spk, s, e in relabel_by_first_appearance(raw)]
        embeddings = getattr(output, "speaker_embeddings", None)
        return DiarizationResult(segments=segments, speaker_embeddings=np.asarray(embeddings) if embeddings is not None else None)
