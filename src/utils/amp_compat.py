"""Mixed-precision shims that work across PyTorch 2.1 and 2.3+.

PyTorch 2.3 introduced the unified ``torch.amp.GradScaler('cuda', ...)`` and
``torch.amp.autocast('cuda', ...)`` APIs, deprecating the cuda-namespaced
``torch.cuda.amp.GradScaler/autocast``.  This module exposes a single
``get_grad_scaler`` and ``cuda_autocast`` that prefer the new API when
available and fall back transparently on PyTorch 2.1, which still ships
only the legacy namespace.

The public functions are drop-in replacements for the legacy calls used
across the project (no ``device_type`` keyword required).
"""

from __future__ import annotations

import torch


def get_grad_scaler(enabled: bool = True):
    """Return a CUDA ``GradScaler`` compatible with both APIs."""
    new_api = getattr(torch, "amp", None)
    if new_api is not None and hasattr(new_api, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            # PyTorch 2.1's torch.amp.GradScaler is an alias of the cuda
            # one and does not accept the device positional argument.
            pass
    return torch.cuda.amp.GradScaler(enabled=enabled)


def cuda_autocast(enabled: bool = True):
    """Return a CUDA autocast context compatible with both APIs."""
    new_api = getattr(torch, "amp", None)
    if new_api is not None and hasattr(new_api, "autocast"):
        try:
            return torch.amp.autocast("cuda", enabled=enabled)
        except TypeError:
            pass
    return torch.cuda.amp.autocast(enabled=enabled)
