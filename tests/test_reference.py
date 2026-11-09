"""Every public declaration in the shim must point at its Argmax equivalent."""

import dataclasses
import importlib
import inspect

from argmax_shim._reference import Fidelity
from argmax_shim.reference import MODULES, load_all, mapping_markdown


def _missing() -> list[str]:
    load_all()
    missing = []
    for name in MODULES:
        mod = importlib.import_module(name)
        for attr, obj in vars(mod).items():
            if attr.startswith("_") or getattr(obj, "__module__", None) != mod.__name__:
                continue
            if not (inspect.isclass(obj) or inspect.isfunction(obj)):
                continue
            ref = getattr(obj, "__argmax_ref__", None)
            if ref is None and attr not in {"StreamSource", "ASRBackendKind"}:
                missing.append(f"{name}.{attr}")
            if not inspect.isclass(obj) or (ref is not None and ref.fidelity is Fidelity.SHIM_ONLY):
                continue  # members of shim-only classes need no Swift counterpart
            for m_name, member in vars(obj).items():
                if m_name.startswith("_") and m_name != "__init__":
                    continue
                if m_name == "__init__" and dataclasses.is_dataclass(obj):
                    continue  # memberwise init: covered by the class-level reference
                fn = member.fget if isinstance(member, property) else getattr(member, "__func__", member)
                if inspect.isfunction(fn) and not hasattr(fn, "__argmax_ref__"):
                    missing.append(f"{name}.{obj.__qualname__}.{m_name}")
    return missing


def test_every_public_declaration_is_mirrored():
    missing = _missing()
    assert not missing, "Add @mirrors(...) to: " + ", ".join(missing)


def test_mapping_generates():
    text = mapping_markdown()
    assert "WhisperKitPro.setCustomVocabulary" in text
    assert "TTSKit.generate" in text
