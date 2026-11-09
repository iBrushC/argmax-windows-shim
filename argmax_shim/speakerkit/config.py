"""SpeakerKit configuration (Pyannote engine)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._reference import OSS_SPEAKERKIT, Fidelity, mirrors


@mirrors(
    "PyannoteConfig",
    module="SpeakerKit",
    docs=OSS_SPEAKERKIT,
    fidelity=Fidelity.APPROXIMATE,
    notes="variant, hf_token and device are shim-only. variant='precision' falls back to community "
    "(Precision-2 is closed-source).",
)
@dataclass
class PyannoteConfig:
    model_folder: str | None = None
    load: bool | None = None
    download: bool = True
    diarizer: Any = None
    """A loaded SpeakerKitDiarizer (Pro API, see DOCS_DIARIZATION)."""
    # ---- shim-only ----
    variant: str = "community"
    hf_token: str | None = None
    """Hugging Face token; pyannote models are gated. Falls back to the HF_TOKEN env var."""
    device: str = "auto"

    def _create_backend(self):
        if self.diarizer is not None:
            return self.diarizer._backend
        from .backends.pyannote_backend import PyannoteBackend

        return PyannoteBackend(variant=self.variant, model_folder=self.model_folder, hf_token=self.hf_token, device=self.device)


