"""Device selection helpers."""

from __future__ import annotations

import torch


def get_default_device() -> torch.device:
    """Prefer CUDA, then Apple Metal/MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_device(name: str = "auto") -> torch.device:
    """Resolve a requested device name and fail clearly if unavailable."""
    if name == "auto":
        return get_default_device()
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available in this PyTorch runtime.")
        return torch.device("mps")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be one of: auto, cuda, mps, cpu")
