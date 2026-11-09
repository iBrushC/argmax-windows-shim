"""Compute device selection (shim-only; the real SDK targets the Neural Engine / NPU)."""

from __future__ import annotations


def resolve_torch_device(device: str = "auto") -> str:
    """Return "cuda" when a CUDA GPU is visible to torch, else "cpu", unless ``device`` is explicit."""
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def torch_dtype_for(device: str):
    """bfloat16 on CUDA, float32 on CPU."""
    import torch

    return torch.bfloat16 if device.startswith("cuda") else torch.float32
