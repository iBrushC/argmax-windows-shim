"""WhisperKit configuration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from .._reference import OSS_WHISPERKIT, Fidelity, mirrors
from ..core.model_store import RepoType


@mirrors(
    "Logging.LogLevel",
    module="WhisperKit",
    docs=OSS_WHISPERKIT,
    fidelity=Fidelity.APPROXIMATE,
    notes="Sets the level of the `argmax_shim` Python logger.",
)
class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    ERROR = "error"
    NONE = "none"

    @mirrors("(none)", module="-", fidelity=Fidelity.SHIM_ONLY)
    def apply(self) -> None:
        levels = {"debug": logging.DEBUG, "info": logging.INFO, "error": logging.ERROR, "none": logging.CRITICAL + 1}
        logger = logging.getLogger("argmax_shim")
        logger.setLevel(levels[self.value])
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())


@mirrors(
    "WhisperKitConfig",
    module="WhisperKit",
    docs=OSS_WHISPERKIT,
    fidelity=Fidelity.APPROXIMATE,
    notes="CoreML compute options, prewarm and tokenizer_folder are accepted and ignored. "
    "device and compute_type are shim-only.",
)
@dataclass
class WhisperKitConfig:
    model: str | None = None
    model_repo: RepoType | str | None = None
    model_folder: str | None = None
    tokenizer_folder: str | None = None
    verbose: bool = False
    log_level: LogLevel = LogLevel.NONE
    prewarm: bool | None = None
    load: bool | None = None
    download: bool = True
    # ---- shim-only ----
    device: str = "auto"
    """"auto" | "cpu" | "cuda" | "cuda:1" ... (shim-only)."""
    compute_type: str = "auto"
    """faster-whisper compute type: "auto" | "int8" | "float16" | "float32" (shim-only)."""

    def __post_init__(self) -> None:
        if self.verbose and self.log_level is LogLevel.NONE:
            self.log_level = LogLevel.INFO
        if self.log_level is not LogLevel.NONE:
            self.log_level.apply()
