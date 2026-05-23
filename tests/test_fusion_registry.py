"""Plugin-contract tests for the fusion-strategy registry."""

from __future__ import annotations

import numpy as np
import pytest

from src.fusions.registry import (
    FusionSpec,
    get_fusion_spec,
    get_fusion_strategy,
    is_registered,
    register_fusion_strategy,
    registered_fusion_strategies,
)


def _identity_fusion(embeddings, normalize=True, **kwargs):  # noqa: ARG001
    """Trivial strategy used by these tests — returns the first matrix."""
    return embeddings[0]


def test_register_fusion_strategy_minimal() -> None:
    register_fusion_strategy("test_identity_min", _identity_fusion)
    assert is_registered("test_identity_min")
    spec = get_fusion_spec("test_identity_min")
    assert isinstance(spec, FusionSpec)
    assert spec.equal_dim_required is True
    assert callable(spec.expand_grid)
    assert spec.expand_grid({}) == [("", {})]


def test_register_fusion_strategy_custom_expand() -> None:
    def expand(_cfg):
        return [("_a", {"alpha": 0.1}), ("_b", {"alpha": 0.9})]

    register_fusion_strategy(
        "test_identity_expand",
        _identity_fusion,
        equal_dim_required=False,
        expand_grid=expand,
    )
    spec = get_fusion_spec("test_identity_expand")
    assert spec.equal_dim_required is False
    assert spec.expand_grid({}) == [("_a", {"alpha": 0.1}), ("_b", {"alpha": 0.9})]


def test_register_fusion_strategy_rejects_non_callable() -> None:
    with pytest.raises(TypeError, match="must be callable"):
        register_fusion_strategy("test_invalid_fn", 123)  # type: ignore[arg-type]


def test_get_fusion_spec_unknown_name_lists_available() -> None:
    register_fusion_strategy("test_listing_canary", _identity_fusion)
    with pytest.raises(KeyError) as excinfo:
        get_fusion_spec("definitely-unknown-fusion")
    message = str(excinfo.value)
    assert "definitely-unknown-fusion" in message
    assert "test_listing_canary" in message


def test_get_fusion_strategy_binds_kwargs() -> None:
    captured: dict = {}

    def fn(embeddings, normalize=True, *, alpha=0.0, **_):
        captured["alpha"] = alpha
        captured["normalize"] = normalize
        return embeddings[0]

    register_fusion_strategy("test_kwargs_binding", fn)
    bound = get_fusion_strategy("test_kwargs_binding", alpha=0.7)
    bound([np.zeros((2, 3))])
    assert captured == {"alpha": 0.7, "normalize": True}


def test_registered_fusion_strategies_is_sorted() -> None:
    names = registered_fusion_strategies()
    assert names == sorted(names)


def test_builtin_fusion_strategies_register_themselves() -> None:
    """Importing src.fusions must populate the registry with built-ins."""
    import src.fusions  # noqa: F401

    expected_subset = {"mean", "sum", "concat", "weighted_mean"}
    registered = set(registered_fusion_strategies())
    missing = expected_subset - registered
    assert not missing, f"built-in fusions not registered: {missing}"
