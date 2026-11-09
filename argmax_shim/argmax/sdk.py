"""SDK initialization (license). The shim needs no license, so these are no-ops kept for source compatibility."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .._reference import DOCS_FILE, DOCS_MODELS, Fidelity, mirrors

log = logging.getLogger(__name__)


@mirrors("ArgmaxConfig", module="Argmax", docs=DOCS_MODELS, fidelity=Fidelity.NO_OP)
@dataclass
class ArgmaxConfig:
    api_key: str = ""
    keychain_persistence: bool = True


@mirrors("ArgmaxSDK", module="Argmax", docs=DOCS_FILE, fidelity=Fidelity.NO_OP)
class ArgmaxSDK:
    config: ArgmaxConfig | None = None

    @classmethod
    @mirrors(
        "ArgmaxSDK.with(_ config: ArgmaxConfig)",
        module="Argmax",
        docs=DOCS_FILE,
        fidelity=Fidelity.NO_OP,
        notes="Named with_ because `with` is a Python keyword. The real call validates the license online "
        "(first use and every 30 days).",
    )
    def with_(cls, config: ArgmaxConfig) -> None:
        cls.config = config
        if config.api_key and not config.api_key.startswith("ax_"):
            log.warning("Argmax API keys start with 'ax_' (not 'axst_'). The shim ignores the key either way.")
