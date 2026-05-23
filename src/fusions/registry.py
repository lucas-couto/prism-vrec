"""Registry for fusion strategies.

A fusion strategy is the recipe that takes a list of embedding matrices
``[Z1, Z2, ..., ZM]`` (each shape ``(N, dm)``) and produces a fused
matrix ``H`` of shape ``(N, Dh)``.  Each strategy is described by a
:class:`FusionSpec`:

* ``name`` — the registry key, also embedded in output filenames.
* ``fn`` — callable with signature
  ``fn(embeddings, normalize=True, **kwargs) -> np.ndarray``.
* ``equal_dim_required`` — whether all input matrices must share ``dm``.
* ``expand_grid`` — function ``cfg -> [(filename_suffix, fn_kwargs)]``
  that turns the strategy's YAML config block into a list of tasks.
  The default returns a single task with no extra kwargs and an empty
  filename suffix.

Adding a custom strategy
------------------------

::

    # plugins/fusions/my_fusion.py
    import numpy as np
    from src.fusions.registry import register_fusion_strategy

    def my_fusion(embeddings, normalize=True, *, alpha=0.5, **kwargs):
        # ... custom math
        return fused

    def expand_my(cfg):
        alphas = cfg.get("alpha", [0.5])
        if not isinstance(alphas, list):
            alphas = [alphas]
        return [(f"_a{a}", {"alpha": a}) for a in alphas]

    register_fusion_strategy(
        "my_fusion",
        my_fusion,
        equal_dim_required=True,
        expand_grid=expand_my,
    )

After this, ``"my_fusion"`` can be added to ``fusion_strategies_enabled``
in ``configs/fusion.yaml`` and the pipeline picks it up automatically.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

# Type aliases for clarity in plugin signatures.
FusionFn = Callable[..., np.ndarray]
ExpandGridFn = Callable[[dict], list[tuple[str, dict]]]


def _default_expand(_cfg: dict) -> list[tuple[str, dict]]:
    """Default grid expansion: one task, no extra kwargs, no filename suffix."""
    return [("", {})]


@dataclass(frozen=True)
class FusionSpec:
    """All metadata the pipeline needs to drive one fusion strategy.

    ``online=True`` marks a strategy whose fusion is performed at
    training time as a :class:`torch.nn.Module` co-trained with the
    recommender (see :mod:`src.fusions.online`).  Online strategies
    have no offline ``.npy`` output; instead the fuse step writes a
    JSON sidecar listing the component embeddings the trainer must
    load and stack.
    """

    name: str
    fn: FusionFn
    equal_dim_required: bool = True
    expand_grid: ExpandGridFn = field(default_factory=lambda: _default_expand)
    online: bool = False


_REGISTRY: dict[str, FusionSpec] = {}


def register_fusion_strategy(
    name: str,
    fn: FusionFn,
    *,
    equal_dim_required: bool = True,
    expand_grid: ExpandGridFn | None = None,
    online: bool = False,
) -> None:
    """Register a fusion strategy under ``name``.

    Re-registering an existing name overwrites the previous binding.
    """
    if not callable(fn):
        raise TypeError(
            f"register_fusion_strategy({name!r}): fn must be callable, got {type(fn).__name__}"
        )
    spec = FusionSpec(
        name=name,
        fn=fn,
        equal_dim_required=equal_dim_required,
        expand_grid=expand_grid if expand_grid is not None else _default_expand,
        online=online,
    )
    _REGISTRY[name] = spec


def is_online_strategy(name: str) -> bool:
    """Return ``True`` when *name* is a registered online (learned) fusion."""
    spec = _REGISTRY.get(name)
    return spec is not None and spec.online


def get_fusion_spec(name: str) -> FusionSpec:
    """Return the :class:`FusionSpec` registered under ``name``."""
    spec = _REGISTRY.get(name)
    if spec is None:
        raise KeyError(
            f"No fusion strategy registered for {name!r}.  "
            f"Available strategies: {registered_fusion_strategies()}.  "
            f"Register a custom strategy via "
            f"src.fusions.registry.register_fusion_strategy(name, fn, ...)."
        )
    return spec


def get_fusion_strategy(name: str, **kwargs) -> Callable:
    """Return the fusion callable bound to optional kwargs.

    Equivalent to ``functools.partial(get_fusion_spec(name).fn, **kwargs)``
    when ``kwargs`` is non-empty; otherwise returns the bare callable.
    Kept under the historical name so existing callers (e.g.
    ``src.steps.fuse``) work unchanged.
    """
    spec = get_fusion_spec(name)
    if not kwargs:
        return spec.fn
    from functools import partial

    return partial(spec.fn, **kwargs)


def registered_fusion_strategies() -> list[str]:
    """Return the sorted list of currently-registered strategy names."""
    return sorted(_REGISTRY)


def is_registered(name: str) -> bool:
    """Return True iff ``name`` is currently registered."""
    return name in _REGISTRY


def iter_specs() -> list[FusionSpec]:
    """Return the registered specs in name-sorted order."""
    return [_REGISTRY[name] for name in registered_fusion_strategies()]


__all__ = [
    "ExpandGridFn",
    "FusionFn",
    "FusionSpec",
    "get_fusion_spec",
    "get_fusion_strategy",
    "is_online_strategy",
    "is_registered",
    "iter_specs",
    "register_fusion_strategy",
    "registered_fusion_strategies",
]
