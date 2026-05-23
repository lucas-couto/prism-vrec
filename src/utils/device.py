"""Runtime device resolution.

The ``device:`` field in ``configs/default.yaml`` accepts three values:

* ``"auto"``: pick ``cuda`` when a GPU is visible, otherwise ``cpu``.
  Default. One configuration works on a RunPod 4090, a lab server
  with a Titan V and a 16 GB Apple Silicon laptop.
* ``"cuda"``: request a GPU. Falls back to ``cpu`` with a warning if
  no GPU is detected, so a misconfigured host does not crash.
* ``"cpu"``: force CPU even on a GPU host. Useful for reproducing a
  CPU-only baseline or debugging without VRAM pressure.

:func:`resolve_device` is the single place where the config string
turns into the device string the steps pass to PyTorch.
"""

from __future__ import annotations

from src.utils.logging import get_logger

logger = get_logger(__name__)


def resolve_device(requested: str) -> str:
    """Map ``requested`` (``auto`` / ``cuda`` / ``cpu``) to a concrete
    device, falling back to ``cpu`` when the requested GPU is unavailable.

    Importing ``torch`` is deferred so this module is cheap to import
    from test code that does not need the full ML stack.
    """
    if requested == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError:
        logger.warning("torch unavailable, falling back to cpu")
        return "cpu"

    cuda_available = bool(torch.cuda.is_available())

    if requested == "auto":
        return "cuda" if cuda_available else "cpu"

    if requested == "cuda":
        if cuda_available:
            return "cuda"
        logger.warning("device='cuda' requested but no GPU is visible; using cpu instead")
        return "cpu"

    logger.warning("unknown device value %r, using cpu", requested)
    return "cpu"
