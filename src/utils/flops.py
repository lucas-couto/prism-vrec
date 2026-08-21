"""Analytic FLOP accounting for the model-bearing pipeline steps.

Measuring floating-point work *exactly* would mean running every
forward and backward pass under :class:`~torch.utils.flop_counter.
FlopCounterMode`, a ``TorchDispatchMode`` that intercepts each ATen
call.  That costs 10-30 % of wall-clock across the whole pipeline —
an unacceptable price for a number that, for this framework, is
knowable in advance.

Every backbone here runs at a fixed input resolution over a fixed
batch shape, so the FLOPs of one forward pass are a constant.  This
module measures that constant **once** per ``(model, input shape)``
with the dispatch counter, caches it, and from then on the hot loop
performs one multiplication:

    flops = flops_per_sample x items_processed

The counter follows the convention that one multiply-accumulate is
two FLOPs, which is what ``FlopCounterMode`` reports and what the
literature quotes (ResNet-50 at 224x224 measures 8.18 GFLOPs per
image here against a published 4.1 GMACs — the expected 2x).

Calibration is side-effect free: the probe forward runs under
``no_grad`` and with the module forced into eval mode, so it cannot
update BatchNorm running statistics or consume dropout randomness in
a training run it is only meant to describe.

Training multiplies by :data:`TRAINING_MULTIPLIER`.  The backward
pass computes gradients with respect to both inputs and weights, each
costing roughly one forward, hence the customary ``3 x forward`` for a
full training step.  It is an approximation — it ignores optimiser
arithmetic and recomputation under activation checkpointing — and it
is recorded as such in the manifest so nobody quotes it as measured.

The cache key is supplied by the caller (typically the extractor name
plus the batch shape) rather than derived from the module object, so
that two runs of the same backbone share one calibration and a change
in resolution correctly forces a new one.

**What counts as a FLOP here.**  ``FlopCounterMode`` attributes the
matmul-class operators — ``mm``/``addmm``/``bmm``, convolutions and
scaled-dot-product attention — and ignores elementwise arithmetic.
That is the same convention the literature uses when quoting "N
GFLOPs" for a backbone, and it is what makes the extractor numbers
comparable to published figures.  The consequence is that a pure
embedding-and-dot-product recommender (BPR-MF) measures as zero: its
arithmetic is elementwise and its cost is memory bandwidth, not
compute.  Such a step records no ``flops_per_s`` at all rather than a
misleading one; its throughput is described by ``items_per_s``
instead.  Attention-based recommenders (ACF) do dispatch matmuls and
are counted normally.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from typing import Any

from src.utils import telemetry
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Forward + backward-wrt-input + backward-wrt-weight ~ 3x forward.
TRAINING_MULTIPLIER = 3.0

_CACHE: dict[str, float] = {}
# Keys whose calibration failed or measured zero.  Cached as deliberately
# as the successes: without this, a model the counter cannot attribute
# (a factorisation recommender, say) would re-enter the dispatch counter
# on every batch — precisely the per-batch overhead this module exists
# to avoid.
_UNCOUNTABLE: set[str] = set()
_LOCK = threading.Lock()


def calibrate(key: str, model: Any, example: Any) -> float | None:
    """Return FLOPs for a *single sample* forward, measuring once per ``key``.

    ``model`` is any callable — an ``nn.Module`` or a bound method such
    as an extractor's component path.  ``example`` is either a batched
    input tensor or a tuple of them for multi-argument forwards, in
    which case it is splatted (``model(*example)``) and the batch size
    is read from the first element.  The measured total is divided by
    that batch size.

    Returns ``None`` when the measurement is impossible (no torch, a
    model that refuses the probe input) or when the counter attributes
    no matmul-class work, in which case the caller simply records no
    FLOPs — a missing number is better than a wrong one.
    """
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        if key in _UNCOUNTABLE:
            return None

    value = _measure(key, model, example)

    with _LOCK:
        if value is None:
            _UNCOUNTABLE.add(key)
        else:
            _CACHE[key] = value

    if value is None:
        return None
    logger.info("FLOP calibration: %s -> %.3f GFLOPs/sample", key, value / 1e9)
    return value


@contextlib.contextmanager
def _eval_mode(model: Any) -> Iterator[None]:
    """Force ``model`` into eval mode for the probe, then restore it.

    Calibration runs one extra forward.  On a module left in training
    mode that forward would update BatchNorm running statistics and
    consume dropout randomness — silently perturbing the very training
    run it is meant to describe.

    Restoration is **per submodule**, not a blanket ``model.train()``.
    Fine-tuning deliberately runs mixed modes: the backbone trains while
    frozen BatchNorm layers stay in eval so their running statistics do
    not drift (see ``FineTuner._set_train_mode``).  Re-enabling training
    on the whole tree would silently unfreeze them.

    Bound methods and plain callables have no mode to switch, so they
    pass through untouched.
    """
    if not hasattr(model, "modules") or not getattr(model, "training", False):
        yield
        return

    previous = [(m, m.training) for m in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, was_training in previous:
            module.training = was_training


def _measure(key: str, model: Any, example: Any) -> float | None:
    try:
        import torch
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None

    try:
        args = tuple(example) if isinstance(example, (tuple, list)) else (example,)
        batch = int(args[0].shape[0])
        if batch <= 0:
            return None
        counter = FlopCounterMode(display=False)
        # no_grad keeps the probe from building an autograd graph; the
        # backward cost is applied analytically via TRAINING_MULTIPLIER.
        with _eval_mode(model), counter, torch.no_grad():
            model(*args)
        total = counter.get_total_flops()
    except Exception as exc:  # noqa: BLE001 - calibration is best-effort
        logger.debug("FLOP calibration unavailable for %s: %r", key, exc)
        return None

    if not total:
        logger.debug("FLOP calibration for %s attributed no counted ops", key)
        return None
    return float(total) / batch


def record(key: str, n_items: int, *, training: bool = False) -> None:
    """Attribute ``n_items`` samples' worth of FLOPs to the running step.

    A no-op when ``key`` was never calibrated or telemetry is off, so
    hot loops can call it unconditionally.
    """
    if n_items <= 0:
        return
    with _LOCK:
        per_sample = _CACHE.get(key)
    if per_sample is None:
        return
    multiplier = TRAINING_MULTIPLIER if training else 1.0
    telemetry.add_flops(per_sample * n_items * multiplier)


def calibrated() -> dict[str, float]:
    """Snapshot of every calibration made, for the manifest header."""
    with _LOCK:
        return dict(_CACHE)


def reset_for_tests() -> None:
    """Clear the calibration cache.  Test-only escape hatch."""
    with _LOCK:
        _CACHE.clear()
        _UNCOUNTABLE.clear()
