"""
Custom Vocabulary.

The real SDK runs an auxiliary keyword-spotting model (canary-1b-v2 or
parakeet-tdt_ctc-110m) on the audio and overwrites the transcript when a keyword
is detected. The shim approximates this with text-level fuzzy matching on the
transcript words (FuzzyKeywordCorrector), and passes keywords natively to
Qwen3-ASR. Expect lower recall and more false positives than the real feature.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from .._reference import DOCS_VOCAB, Fidelity, mirrors
from ..whisperkit.datatypes import WordTiming

log = logging.getLogger(__name__)

_INVALID = re.compile(r"[^\w\s\-'.&+]", re.UNICODE)
_NORM = re.compile(r"[^\w]+", re.UNICODE)


@mirrors("CustomVocabularyModelRepo", module="Argmax", docs=DOCS_VOCAB, fidelity=Fidelity.NO_OP, notes="Swift type name unverified.")
@dataclass(frozen=True)
class CustomVocabularyModelRepo:
    repo_id: str
    models: tuple[str, ...]


@mirrors("CustomVocabularyModelState", module="Argmax", docs=DOCS_VOCAB, fidelity=Fidelity.APPROXIMATE, notes="Swift type name unverified.")
class CustomVocabularyModelState(str, Enum):
    DISABLED = "disabled"
    LOADED = "loaded"


@mirrors(
    "CustomVocabularyConfig",
    module="Argmax",
    docs=DOCS_VOCAB,
    fidelity=Fidelity.APPROXIMATE,
    notes="model_folder is accepted but no keyword-spotting model is loaded. match_threshold is shim-only.",
)
@dataclass
class CustomVocabularyConfig:
    words: list[str] | None = None
    model_folder: str | None = None
    # ---- shim-only ----
    match_threshold: float = field(default=0.82)
    """Minimum difflib similarity (0-1) for a fuzzy keyword replacement."""

    @staticmethod
    @mirrors("CustomVocabularyConfig.customVocabularyModelRepo()", module="Argmax", docs=DOCS_VOCAB, fidelity=Fidelity.NO_OP)
    def custom_vocabulary_model_repo() -> CustomVocabularyModelRepo:
        return CustomVocabularyModelRepo("argmaxinc/ctckit-pro", ("canary-1b-v2_474MB", "parakeet-tdt_ctc-110m"))


@mirrors("(validation inside setCustomVocabulary)", module="Argmax", docs=DOCS_VOCAB, fidelity=Fidelity.APPROXIMATE, notes="Called for you by set_custom_vocabulary.")
def sanitize_vocabulary(words: list[str]) -> list[str]:
    """Drop entries with invalid characters (as the SDK does) and warn about very short ones."""
    clean = []
    for w in words:
        w = w.strip()
        if not w:
            continue
        if _INVALID.search(w):
            log.warning("Dropping custom vocabulary entry with invalid characters: %r", w)
            continue
        if len(w) < 4:
            log.warning("Custom vocabulary entry %r is under 4 characters; accuracy may suffer.", w)
        clean.append(w)
    if len(clean) > 3000:
        log.warning("%d keywords registered; the SDK is tuned for up to 3000.", len(clean))
    return clean


def _norm(text: str) -> str:
    return _NORM.sub("", text.lower())


@mirrors(
    "(auxiliary keyword-spotting model)",
    module="Argmax",
    docs=DOCS_VOCAB,
    fidelity=Fidelity.SHIM_ONLY,
    notes="Text-level stand-in for the SDK's audio keyword spotting. Delete when swapping.",
)
class FuzzyKeywordCorrector:
    """Replace word spans that closely resemble a registered keyword."""

    def __init__(self, words: list[str], threshold: float = 0.82) -> None:
        self.threshold = threshold
        self.entries = [(kw, _norm(kw), max(1, len(kw.split()))) for kw in words if _norm(kw)]

    def correct(self, words: list[WordTiming]) -> tuple[list[WordTiming], int]:
        """Return corrected words and the number of replacements."""
        if not self.entries:
            return words, 0
        out: list[WordTiming] = []
        i = 0
        replaced = 0
        while i < len(words):
            best: tuple[float, int, str] | None = None
            for keyword, nkw, n_words in self.entries:
                for span in range(1, min(n_words + 2, 4) + 1):
                    window = words[i : i + span]
                    if len(window) < span:
                        break
                    joined = _norm("".join(w.word for w in window))
                    if not joined:
                        continue
                    score = difflib.SequenceMatcher(None, joined, nkw).ratio()
                    if score >= self.threshold and (best is None or score > best[0]):
                        best = (score, span, keyword)
            if best is None:
                out.append(words[i])
                i += 1
                continue
            _, span, keyword = best
            window = words[i : i + span]
            original = "".join(w.word for w in window).strip()
            if original != keyword:
                replaced += 1
            out.append(
                WordTiming(
                    word=" " + keyword,
                    tokens=[],
                    start=window[0].start,
                    end=window[-1].end,
                    probability=min(w.probability for w in window),
                )
            )
            i += span
        return out, replaced
