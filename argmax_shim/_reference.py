"""
Cross-reference machinery linking every shim declaration to its Argmax equivalent.

Usage, directly above a declaration::

    @mirrors("WhisperKit.transcribe(audioPath:decodeOptions:callback:)",
             module="WhisperKit", docs=OSS_WHISPERKIT)
    def transcribe(...): ...

The decorator appends an "Argmax equivalent" block to the docstring, sets
``__argmax_ref__`` on the object, and records it in ``REGISTRY`` so
``python -m argmax_shim.reference`` can emit MAPPING.md.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import Enum
from typing import Callable, TypeVar

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Documentation anchors
# ---------------------------------------------------------------------------
DOCS = "https://app.argmaxinc.com/docs"
DOCS_MODELS_GALLERY = f"{DOCS}/models"
DOCS_FILE = f"{DOCS}/examples/file-transcription"
DOCS_REALTIME = f"{DOCS}/examples/real-time-transcription"
DOCS_REALTIME_SPEAKERS = f"{DOCS}/examples/real-time-transcription#with-speakers"
DOCS_VOCAB = f"{DOCS}/examples/custom-vocabulary"
DOCS_DIARIZATION = f"{DOCS}/examples/speaker-diarization"
DOCS_MODELS = f"{DOCS}/guides/managing-models"
DOCS_INSTALL = f"{DOCS}/guides/upgrading-to-pro-sdk"
OSS = "https://github.com/argmaxinc/argmax-oss-swift"
OSS_WHISPERKIT = f"{OSS}#whisperkit"
OSS_SPEAKERKIT = f"{OSS}#speakerkit"
OSS_TTSKIT = f"{OSS}#ttskit"


class Fidelity(str, Enum):
    """How closely a shim declaration follows the real one."""

    EXACT = "exact"
    """Same inputs, outputs and semantics (modulo Python naming)."""
    APPROXIMATE = "approximate"
    """Same shape, but a different algorithm, reduced options or unverified details."""
    NO_OP = "no-op"
    """Accepted for source compatibility; does nothing in the shim."""
    SHIM_ONLY = "shim-only"
    """No Argmax equivalent. Delete or replace when swapping to the real SDK."""


@dataclass(frozen=True)
class ArgmaxRef:
    swift: str
    module: str
    docs: str | None = None
    fidelity: Fidelity = Fidelity.EXACT
    notes: str = ""


REGISTRY: dict[str, ArgmaxRef] = {}


def mirrors(
    swift: str,
    *,
    module: str,
    docs: str | None = None,
    fidelity: Fidelity = Fidelity.EXACT,
    notes: str = "",
) -> Callable[[T], T]:
    """Annotate a class, function or method with its Argmax SDK equivalent."""
    ref = ArgmaxRef(swift=swift, module=module, docs=docs, fidelity=fidelity, notes=notes)

    def decorate(obj: T) -> T:
        setattr(obj, "__argmax_ref__", ref)
        REGISTRY[f"{obj.__module__}.{obj.__qualname__}"] = ref  # type: ignore[attr-defined]
        block = ["", "", "Argmax equivalent", "-----------------", f"{module}: {swift}", f"Fidelity: {fidelity.value}"]
        if docs:
            block.append(f"Docs: {docs}")
        if notes:
            block.append(f"Notes: {notes}")
        base = inspect.cleandoc(obj.__doc__) if obj.__doc__ else ""
        obj.__doc__ = base + "\n".join(block)
        return obj

    return decorate
