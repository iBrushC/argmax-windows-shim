"""
Model registry and downloads.

Argmax model names (e.g. ``parakeet-v2_476MB``) are resolved to the upstream
open-source checkpoints on Hugging Face, so the same strings work in both the
shim and the real SDK.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

from .._reference import DOCS_MODELS, DOCS_MODELS_GALLERY, Fidelity, mirrors
from .errors import MissingDependencyError

log = logging.getLogger(__name__)


class ASRBackendKind(str, Enum):
    FASTER_WHISPER = "faster-whisper"
    NEMO_PARAKEET = "nemo-parakeet"
    QWEN_ASR = "qwen-asr"


@mirrors(
    "RepoType",
    module="Argmax",
    docs=DOCS_MODELS,
    fidelity=Fidelity.NO_OP,
    notes="Accepted for source compatibility. The shim resolves models by name, not by repo.",
)
class RepoType(str, Enum):
    OPEN_SOURCE_REPO = "argmaxinc/whisperkit-coreml"
    PRO_REPO = "argmaxinc/whisperkit-pro"
    PARAKEET_REPO = "argmaxinc/parakeetkit-pro"
    QWEN_ASR_REPO = "argmaxinc/qwenasrkit-pro"


@mirrors(
    "(Model Gallery entry)",
    module="Argmax",
    docs=DOCS_MODELS_GALLERY,
    fidelity=Fidelity.SHIM_ONLY,
    notes="Maps an Argmax model name to an upstream Hugging Face checkpoint and a Python backend.",
)
@dataclass(frozen=True)
class ASRModelSpec:
    argmax_name: str
    backend: ASRBackendKind
    hf_repo: str
    english_only: bool = False
    aux_repos: dict = field(default_factory=dict)
    notes: str = ""


_TURBO = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
_SPECS: list[tuple[tuple[str, ...], ASRModelSpec]] = [
    (
        ("large-v3-v20240930_626MB", "large-v3-v20240930_turbo", "large-v3-v20240930", "large-v3-turbo", "whisper-large-v3-turbo"),
        ASRModelSpec("large-v3-v20240930_626MB", ASRBackendKind.FASTER_WHISPER, _TURBO, notes="openai/whisper-large-v3-turbo via CTranslate2"),
    ),
    *[
        ((n,), ASRModelSpec(n, ASRBackendKind.FASTER_WHISPER, f"Systran/faster-whisper-{n}", english_only=n.endswith(".en")))
        for n in ("tiny", "tiny.en", "base", "base.en", "small", "small.en")
    ],
    (
        ("parakeet-v2", "parakeet-v2_476MB", "nvidia/parakeet-v2"),
        ASRModelSpec("parakeet-v2_476MB", ASRBackendKind.NEMO_PARAKEET, "nvidia/parakeet-tdt-0.6b-v2", english_only=True),
    ),
    (
        ("parakeet-v3", "parakeet-v3_494MB", "nvidia/parakeet-v3"),
        ASRModelSpec("parakeet-v3_494MB", ASRBackendKind.NEMO_PARAKEET, "nvidia/parakeet-tdt-0.6b-v3"),
    ),
    (
        ("qwen3-asr", "qwen3-asr-1.7b", "qwen/qwen3-asr-1.7b"),
        ASRModelSpec(
            "qwen3-asr-1.7b",
            ASRBackendKind.QWEN_ASR,
            "Qwen/Qwen3-ASR-1.7B",
            aux_repos={"forced_aligner": "Qwen/Qwen3-ForcedAligner-0.6B"},
        ),
    ),
    (
        ("qwen3-asr-0.6b",),
        ASRModelSpec(
            "qwen3-asr-0.6b",
            ASRBackendKind.QWEN_ASR,
            "Qwen/Qwen3-ASR-0.6B",
            aux_repos={"forced_aligner": "Qwen/Qwen3-ForcedAligner-0.6B"},
            notes="Shim-only lighter variant; not in the Argmax gallery.",
        ),
    ),
]
_TABLE: dict[str, ASRModelSpec] = {alias.lower(): spec for aliases, spec in _SPECS for alias in aliases}

DEFAULT_ASR_MODEL = "large-v3-v20240930_626MB"
"""Shim default. Whisper via faster-whisper is the most portable backend on Windows."""

_STRIP_PREFIXES = ("openai_whisper-", "openai/whisper-", "argmaxinc/")


@mirrors("(model name lookup inside WhisperKitConfig)", module="Argmax", docs=DOCS_MODELS, fidelity=Fidelity.SHIM_ONLY)
def resolve_asr_model(name: str | None) -> ASRModelSpec:
    """Resolve an Argmax model name (glob patterns allowed, like WhisperKit) to a spec."""
    key = (name or DEFAULT_ASR_MODEL).strip().lower()
    for prefix in _STRIP_PREFIXES:
        if key.startswith(prefix):
            key = key[len(prefix):]
    if key in _TABLE:
        return _TABLE[key]
    matches = {id(spec): spec for alias, spec in _TABLE.items() if fnmatch.fnmatch(alias, key)}
    if len(matches) == 1:
        return next(iter(matches.values()))
    if len(matches) > 1:
        raise ValueError(f"Model pattern {name!r} matches several models: {sorted(s.argmax_name for s in matches.values())}")
    raise ValueError(f"Unknown model {name!r}. Supported: {list_supported_models()}")


@mirrors("(none)", module="-", fidelity=Fidelity.SHIM_ONLY)
def infer_spec_from_folder(folder: str | Path) -> ASRModelSpec | None:
    """Best-effort: recognise a Hugging Face cache folder downloaded by ModelStore."""
    text = str(folder).replace("\\", "/").lower()
    for spec in {id(s): s for s in _TABLE.values()}.values():
        repo = spec.hf_repo.lower()
        if repo in text or ("models--" + repo.replace("/", "--")) in text:
            return spec
    return None


@mirrors("(Model Gallery)", module="Argmax", docs=DOCS_MODELS_GALLERY, fidelity=Fidelity.SHIM_ONLY)
def list_supported_models() -> list[str]:
    """Canonical Argmax names the shim understands."""
    return sorted({spec.argmax_name for spec in _TABLE.values()})


@mirrors("(none)", module="-", fidelity=Fidelity.SHIM_ONLY, notes="Shim-internal Hugging Face download helper.")
def download_hf_snapshot(repo_id: str, *, enabled: bool = True, cache_dir: str | None = None) -> str:
    """Download (or reuse) a Hugging Face snapshot and return its local path."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise MissingDependencyError("huggingface_hub", "core") from exc
    return snapshot_download(repo_id, local_files_only=not enabled, cache_dir=cache_dir)


@mirrors(
    "ModelStore",
    module="Argmax",
    docs=DOCS_MODELS,
    fidelity=Fidelity.APPROXIMATE,
    notes="Foreground Hugging Face downloads only. Background download, pause/resume, network-type "
    "restrictions and crash recovery are device features and are not mirrored.",
)
class ModelStore:
    """Downloads the upstream checkpoint behind an Argmax model name."""

    @mirrors("ModelStore(config:)", module="Argmax", docs=DOCS_MODELS, fidelity=Fidelity.APPROXIMATE, notes="cache_dir is shim-only.")
    def __init__(self, config: object | None = None, cache_dir: str | None = None) -> None:
        self.config = config
        self.cache_dir = cache_dir
        self.progress: float | None = None

    @mirrors(
        "ModelStore.downloadModel(name:repo:)",
        module="Argmax",
        docs=DOCS_MODELS,
        fidelity=Fidelity.APPROXIMATE,
        notes="Progress is reported via progress_callback instead of the Combine $progress publisher.",
    )
    def download_model(
        self,
        name: str,
        repo: RepoType | str | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> Path:
        """Download a model and return the local folder to pass as ``model_folder``."""
        if isinstance(repo, str) and repo not in {r.value for r in RepoType} and "ctckit" in repo:
            log.warning("Custom-vocabulary models (%s) are CoreML-only; the shim uses fuzzy matching instead.", repo)
        spec = resolve_asr_model(name)
        if progress_callback:
            progress_callback(0.0)
        path = download_hf_snapshot(spec.hf_repo, cache_dir=self.cache_dir)
        for aux in spec.aux_repos.values():
            download_hf_snapshot(aux, cache_dir=self.cache_dir)
        self.progress = 1.0
        if progress_callback:
            progress_callback(1.0)
        return Path(path)
