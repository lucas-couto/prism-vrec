"""Mathematical correctness tests for the built-in fusion strategies.

These tests pin the exact arithmetic each strategy performs on small
inputs so that future refactors of ``src/fusions/strategies.py`` cannot
silently change behaviour.  Normalisation is disabled where it would
get in the way of checking the raw operation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.fusions.registry import get_fusion_spec
from src.fusions.strategies import (
    fuse_concat,
    fuse_max_pool,
    fuse_mean,
    fuse_prod,
    fuse_sum,
    fuse_weighted_mean,
    l2_normalize,
)


def _e1() -> np.ndarray:
    return np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)


def _e2() -> np.ndarray:
    return np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)


def test_all_offline_strategies_emit_float32() -> None:
    # Regression: the weighted family accumulated in float64 and returned
    # it; the train gate validates artifacts against FEATURE_DTYPE
    # (float32) and refused the fused hybrids.
    from src.fusions.strategies import get_fusion_strategy

    rng = np.random.default_rng(0)
    sources = [rng.standard_normal((6, 8)).astype(np.float32) for _ in range(2)]
    for name in ("mean", "sum", "prod", "max_pool", "concat"):
        assert get_fusion_strategy(name)(sources, normalize=True).dtype == np.float32, name
    assert (
        get_fusion_strategy("weighted_mean", weights=[0.7, 0.3])(sources, normalize=True).dtype
        == np.float32
    )
    assert (
        get_fusion_strategy("attention_weighted", logits=[1.0, 0.0])(sources, normalize=True).dtype
        == np.float32
    )
    assert (
        get_fusion_strategy("gated", logits=[1.0, 0.0])(sources, normalize=True).dtype == np.float32
    )


def test_l2_normalize_unit_norm_per_row() -> None:
    x = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
    out = l2_normalize(x)

    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)


def test_fuse_mean_no_normalize() -> None:
    out = fuse_mean([_e1(), _e2()], normalize=False)

    expected = (_e1() + _e2()) / 2.0
    np.testing.assert_allclose(out, expected)


def test_fuse_sum_no_normalize() -> None:
    out = fuse_sum([_e1(), _e2()], normalize=False)

    expected = _e1() + _e2()
    np.testing.assert_allclose(out, expected)


def test_fuse_prod_no_normalize() -> None:
    out = fuse_prod([_e1(), _e2()], normalize=False)

    expected = _e1() * _e2()
    np.testing.assert_allclose(out, expected)


def test_fuse_max_pool_takes_elementwise_max() -> None:
    a = np.array([[1.0, 9.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 2.0], [7.0, 1.0]], dtype=np.float32)

    out = fuse_max_pool([a, b], normalize=False)

    expected = np.array([[5.0, 9.0], [7.0, 4.0]], dtype=np.float32)
    np.testing.assert_allclose(out, expected)


def test_fuse_concat_preserves_total_width() -> None:
    a = np.zeros((4, 3), dtype=np.float32)
    b = np.zeros((4, 5), dtype=np.float32)

    out = fuse_concat([a, b], normalize=False)

    assert out.shape == (4, 8)


def test_fuse_concat_allows_mixed_dims() -> None:
    """``equal_dim_required=False`` is the contract for concat."""
    a = np.array([[1.0, 2.0]], dtype=np.float32)
    b = np.array([[3.0, 4.0, 5.0]], dtype=np.float32)

    out = fuse_concat([a, b], normalize=False)

    np.testing.assert_allclose(out[0], [1.0, 2.0, 3.0, 4.0, 5.0])


def test_fuse_weighted_mean_default_weights_equal_to_mean() -> None:
    a = _e1()
    b = _e2()

    plain = fuse_mean([a, b], normalize=False)
    weighted = fuse_weighted_mean([a, b], normalize=False, weights=[0.5, 0.5])

    np.testing.assert_allclose(weighted, plain)


def test_fuse_weighted_mean_respects_weights() -> None:
    a = np.array([[2.0, 4.0]], dtype=np.float32)
    b = np.array([[6.0, 8.0]], dtype=np.float32)

    out = fuse_weighted_mean([a, b], normalize=False, weights=[0.25, 0.75])

    expected = 0.25 * a + 0.75 * b
    np.testing.assert_allclose(out, expected)


def test_equal_dim_required_strategies_raise_on_mismatch() -> None:
    a = np.zeros((3, 2), dtype=np.float32)
    b = np.zeros((3, 4), dtype=np.float32)

    for fn in (fuse_mean, fuse_sum, fuse_prod, fuse_max_pool):
        with pytest.raises(ValueError):
            fn([a, b], normalize=False)


def test_empty_embeddings_list_raises() -> None:
    for fn in (fuse_mean, fuse_sum, fuse_prod, fuse_max_pool, fuse_concat):
        with pytest.raises(ValueError):
            fn([], normalize=False)


# ---------------------------------------------------------------------
# Grid expansion: the kwargs a spec emits must be the ones its function
# consumes, or the sweep silently collapses to the default.
# ---------------------------------------------------------------------


def test_pca_per_model_grid_emits_the_kwarg_the_function_consumes() -> None:
    """``fuse_pca_per_model`` takes ``n_components``, not the YAML spelling.

    Emitting ``n_components_per_model`` left it in ``**kwargs``, where
    ``_warn_ignored_kwargs`` discarded it: a configured sweep of
    ``[32, 64, 128]`` wrote three identical 64-dim matrices under three
    different ``_nc`` filenames.
    """
    spec = get_fusion_spec("pca_per_model")

    grid = spec.expand_grid({"n_components_per_model": [32, 64, 128]})

    assert grid == [
        ("_nc32", {"n_components": 32}),
        ("_nc64", {"n_components": 64}),
        ("_nc128", {"n_components": 128}),
    ]


def test_pca_per_model_grid_kwarg_actually_changes_the_width() -> None:
    spec = get_fusion_spec("pca_per_model")
    [(_, kwargs)] = spec.expand_grid({"n_components_per_model": 3})
    sources = [
        np.random.default_rng(0).normal(size=(20, 8)).astype(np.float32),
        np.random.default_rng(1).normal(size=(20, 6)).astype(np.float32),
    ]

    fused = spec.fn(sources, normalize=True, train_items=np.arange(15), **kwargs)

    assert fused.shape == (20, 6)  # 2 sources x 3 components, not 2 x 64


def test_pca_grid_emits_the_kwarg_the_function_consumes() -> None:
    spec = get_fusion_spec("pca")

    grid = spec.expand_grid({"n_components": [16, 32]})

    assert grid == [("_nc16", {"n_components": 16}), ("_nc32", {"n_components": 32})]


def test_logit_strategies_grid_emits_the_kwarg_the_functions_consume() -> None:
    """``attention_weighted`` and ``gated`` expose their fixed logits
    through ``expand_grid`` so configured values reach the fusion fn
    (and the learned-alignment sidecar) instead of the uniform default."""
    for name in ("attention_weighted", "gated"):
        spec = get_fusion_spec(name)

        grid = spec.expand_grid({"logits": [[1.0, 0.0]]})

        assert grid == [("_l1_0", {"logits": [1.0, 0.0]})]


def test_logit_strategies_grid_defaults_to_uniform() -> None:
    for name in ("attention_weighted", "gated"):
        spec = get_fusion_spec(name)

        grid = spec.expand_grid({})

        assert grid == [("_l0_0", {"logits": [0.0, 0.0]})]


# ---------------------------------------------------------------------
# Canonical-config distinctness: the weighted-combination family only
# constitutes three distinct experimental conditions if the canonical
# values in configs/fusion.yaml keep the strategies apart.  The previous
# 0.5 / zero-logit canon collapsed all three into plain mean (decision
# 2026-08-22).
# ---------------------------------------------------------------------


def test_canonical_config_keeps_weighted_family_distinct() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "fusion.yaml"
    strategies_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))["strategies"]

    rng = np.random.default_rng(0)
    sources = [
        rng.normal(size=(12, 6)).astype(np.float32),
        rng.normal(size=(12, 6)).astype(np.float32),
    ]

    fused = {"mean": fuse_mean(sources, normalize=True)}
    for name in ("weighted_mean", "attention_weighted", "gated"):
        spec = get_fusion_spec(name)
        [(_suffix, kwargs)] = spec.expand_grid(strategies_cfg.get(name, {}))
        fused[name] = spec.fn(sources, normalize=True, **kwargs)

    names = list(fused)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not np.allclose(fused[a], fused[b], atol=1e-6), (
                f"canonical configs collapse {a!r} and {b!r} into the same "
                f"fused matrix — they are not distinct experimental conditions"
            )
