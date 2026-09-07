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


def cap_process_vram(n_workers: int = 1, *, vram_share: float) -> float:
    """Cap this process's VRAM and return the fraction applied.

    Sizing every budget off the whole card is what oversubscribes a
    shared GPU; capping the process makes :func:`vram_allowance_bytes`
    report the real allowance to every downstream budget.  The GPU that
    trains is the one that draws the researcher's desktop, so the
    remainder of the card is not slack: it is what keeps the
    workstation usable across a multi-day battery (an uncapped run froze
    the machine and, on 2026-09-01, tripped the display driver's
    watchdog).

    :param n_workers: Processes sharing the card.  ``1`` (the default,
        and the case for the single-worker battery and the ``evaluate``
        step) still caps, leaving the display its headroom.
    :param vram_share: ``resources.gpu.vram_share`` -- the share of the
        card the whole run may claim (``PRISM_VRAM_SHARE`` overrides it
        for one launch); split evenly across *n_workers*.  Every CUDA
        context lives *outside* the per-process cap, on top of it.
    :returns: The fraction applied, or ``0.0`` when there is no CUDA
        device to cap.
    """
    import torch

    if not torch.cuda.is_available():
        return 0.0
    fraction = vram_share / max(1, n_workers)
    torch.cuda.set_per_process_memory_fraction(fraction)
    return fraction


def vram_allowance_bytes(device=None) -> int:
    """Bytes of VRAM THIS process may use, honouring the per-process cap.

    ``torch.cuda.get_device_properties().total_memory`` reports the
    card, not the allowance: a worker capped by
    ``torch.cuda.set_per_process_memory_fraction`` sees the full total
    and overcommits (the 2026-08-24 OOM cascade — VNPR's eval chunk and
    the default ranking budget were both sized off the card while three
    workers shared it).  Every VRAM-derived budget must size off THIS
    value instead.

    :param device: CUDA device (index, str or ``torch.device``);
        ``None`` = current device.
    :returns: Allowance in bytes; ``0`` when CUDA is unavailable.
    """
    import torch

    if not torch.cuda.is_available():
        return 0
    try:
        total = torch.cuda.get_device_properties(device or 0).total_memory
        getter = getattr(torch.cuda, "get_per_process_memory_fraction", None)
        fraction = float(getter(device)) if getter is not None else 1.0
    except (RuntimeError, AssertionError):
        return 0
    return int(total * min(max(fraction, 0.0), 1.0))
