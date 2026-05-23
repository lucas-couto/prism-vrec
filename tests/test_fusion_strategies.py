"""Mathematical correctness tests for the built-in fusion strategies.

These tests pin the exact arithmetic each strategy performs on small
inputs so that future refactors of ``src/fusions/strategies.py`` cannot
silently change behaviour.  Normalisation is disabled where it would
get in the way of checking the raw operation.
"""

from __future__ import annotations

import numpy as np
import pytest

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
