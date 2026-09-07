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

import math
import os

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


#: Share of the machine a run may consume, applied to VRAM here and
#: mirrored for RAM and CPU by ``docker-compose.yml`` (``mem_limit``,
#: ``cpus``).  The GPU that trains is the same one that draws the
#: researcher's desktop, so the remaining third is not slack: it is what
#: keeps the workstation usable across a multi-day battery.  Chosen by
#: the researcher on 2026-09-04, deliberately over throughput -- an
#: uncapped run froze the machine and, on 2026-09-01, tripped the
#: display driver's watchdog.  Lowered from 0.65 to 0.5 on 2026-09-06:
#: with the browser, IDE and another project's containers on the same
#: host, the 65% run share still pushed the desktop into swap.
DEFAULT_RUN_RESOURCE_SHARE = 0.5

#: Environment override of :data:`RUN_RESOURCE_SHARE`, forwarded by
#: ``docker-compose.yml`` like ``PRISM_MEM_LIMIT`` / ``PRISM_CPUS``.
#: Meant for unattended windows -- ``PRISM_VRAM_SHARE=0.95 docker compose
#: up -d`` overnight lets the amazon_women VNPR cells that do not fit
#: half the card run without touching the source (2026-09-07).  Anything
#: above ~0.85 leaves the desktop's own allocations at the driver's
#: mercy; use it only while nobody sits at the machine.
RUN_RESOURCE_SHARE_ENV = "PRISM_VRAM_SHARE"


def _resolve_run_resource_share() -> float:
    """``PRISM_VRAM_SHARE`` as a fraction in ``(0, 1]``, else the default.

    An unparsable or out-of-range value fails at import, in the spirit
    of the single validated environment boundary: a typo must not
    silently run the battery uncapped or at 5% of the card.
    """
    raw = os.environ.get(RUN_RESOURCE_SHARE_ENV)
    if raw is None or raw.strip() == "":
        return DEFAULT_RUN_RESOURCE_SHARE
    try:
        share = float(raw)
    except ValueError as exc:
        raise ValueError(f"{RUN_RESOURCE_SHARE_ENV}={raw!r} is not a number.") from exc
    if not math.isfinite(share) or not 0.0 < share <= 1.0:
        raise ValueError(f"{RUN_RESOURCE_SHARE_ENV}={raw!r} must be a fraction in (0, 1].")
    return share


RUN_RESOURCE_SHARE = _resolve_run_resource_share()

#: VRAM cap for a process that has the card to itself.
SOLO_PROCESS_VRAM_FRACTION = RUN_RESOURCE_SHARE

#: Total share of the card the training workers may claim between them.
#: Never above :data:`RUN_RESOURCE_SHARE`, and every CUDA context lives
#: *outside* the per-process cap, on top of it.
POOL_VRAM_FRACTION = RUN_RESOURCE_SHARE


def cap_process_vram(n_workers: int = 1) -> float:
    """Cap this process's VRAM and return the fraction applied.

    Sizing every budget off the whole card is what oversubscribes a
    shared GPU; capping the process makes :func:`vram_allowance_bytes`
    report the real allowance to every downstream budget.

    :param n_workers: Processes sharing the card.  ``1`` (the default,
        and the case for the single-worker battery and the ``evaluate``
        step) still caps, leaving the display its headroom.
    :returns: The fraction applied, or ``0.0`` when there is no CUDA
        device to cap.
    """
    import torch

    if not torch.cuda.is_available():
        return 0.0
    fraction = POOL_VRAM_FRACTION / n_workers if n_workers > 1 else SOLO_PROCESS_VRAM_FRACTION
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
