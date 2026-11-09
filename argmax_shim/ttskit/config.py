"""TTSKit configuration."""

from __future__ import annotations

from dataclasses import dataclass

from .._reference import OSS_TTSKIT, Fidelity, mirrors
from .datatypes import TTSModelVariant


@mirrors(
    "TTSKitConfig",
    module="TTSKit",
    docs=OSS_TTSKIT,
    fidelity=Fidelity.APPROXIMATE,
    notes="speech_decoder_mode / multi_code_decoder_mode are CoreML-specific and ignored. "
    "device and voice are shim-only.",
)
@dataclass
class TTSKitConfig:
    model: TTSModelVariant = TTSModelVariant.QWEN3_TTS_0_6B
    model_folder: str | None = None
    speech_decoder_mode: str | None = None
    multi_code_decoder_mode: str | None = None
    load: bool | None = None
    # ---- shim-only ----
    device: str = "auto"
    voice: str | None = None
    """Engine-specific voice override, e.g. "af_heart" (Kokoro) or "en_US-lessac-medium" (Piper)."""
