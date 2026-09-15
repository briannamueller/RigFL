"""Resolve a device string to a ``torch.device``.

``auto`` prefers CUDA, then Apple MPS, then CPU, allowing the same configuration
to run across supported hardware.
"""

from __future__ import annotations

import torch


def resolve_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if spec == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but CUDA is not available")
    if spec == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device 'mps' requested but MPS is not available")
    return torch.device(spec)
