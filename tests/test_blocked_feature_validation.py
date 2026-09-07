"""Feature validation reduces in row blocks; pools never assume free host memory (M06).

Finiteness and norms are reduced per block (no full-catalogue mask or
norm array), edge rows around block boundaries are still caught, the
stats equal the whole-array computation, and an unknown worker
footprint is charged a conservative floor instead of being read as
"no host memory needed".
"""

from __future__ import annotations

import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from src.steps.validate_features import (
    BLOCK_ROWS,
    FeatureValidationError,
    validate_matrix,
)
from src.utils import parallel as parallel_mod
from src.utils.parallel import UNKNOWN_WORKER_FOOTPRINT_BYTES, detect_max_workers

N, D = 1000, 8
BLOCK = 128


def _good(seed: int = 0) -> np.ndarray:
    matrix = np.random.default_rng(seed).standard_normal((N, D)).astype(np.float32)
    return matrix + 1.0


def _memmap(tmp_path: Path, matrix: np.ndarray) -> np.memmap:
    path = tmp_path / "features.npy"
    np.save(path, matrix)
    return np.load(path, mmap_mode="r")


class TestBlockedReduction:
    def test_stats_equal_the_whole_array_computation(self) -> None:
        matrix = _good()
        norms = np.linalg.norm(matrix, axis=1).astype(np.float64)

        stats = validate_matrix(matrix, label="ds/bb", expected_rows=N, block_rows=BLOCK)

        assert stats["block_rows"] == BLOCK and stats["n_blocks"] == -(-N // BLOCK)
        assert stats["norm_mean"] == pytest.approx(norms.mean(), rel=1e-9)
        assert stats["norm_std"] == pytest.approx(norms.std(), rel=1e-6)
        assert stats["norm_min"] == pytest.approx(norms.min())
        assert stats["norm_max"] == pytest.approx(norms.max())

    def test_block_size_does_not_change_the_verdict_or_the_stats(self) -> None:
        matrix = _good()
        reference = validate_matrix(matrix, label="x", expected_rows=N, block_rows=N)
        for block in (1, 7, BLOCK, N - 1, N + 5):
            stats = validate_matrix(matrix, label="x", expected_rows=N, block_rows=block)
            for key in ("norm_mean", "norm_std", "norm_min", "norm_max"):
                assert stats[key] == pytest.approx(reference[key], rel=1e-6), (block, key)

    @pytest.mark.parametrize("row", [0, BLOCK - 1, BLOCK, 2 * BLOCK + 1, N - 1])
    def test_non_finite_edge_rows_are_caught_with_their_global_index(self, row) -> None:
        matrix = _good()
        matrix[row, 3] = np.nan
        with pytest.raises(FeatureValidationError, match=rf"NaN/Inf, e.g. item_idx \[{row}\]"):
            validate_matrix(matrix, label="x", expected_rows=N, block_rows=BLOCK)

    @pytest.mark.parametrize("row", [BLOCK - 1, BLOCK, N - 1])
    def test_zero_norm_edge_rows_are_caught_with_their_global_index(self, row) -> None:
        matrix = _good()
        matrix[row] = 0.0
        with pytest.raises(FeatureValidationError, match=rf"item_idx \[{row}\]"):
            validate_matrix(matrix, label="x", expected_rows=N, block_rows=BLOCK)

    def test_error_counts_every_bad_row_but_quotes_at_most_ten(self) -> None:
        matrix = _good()
        bad = list(range(5, N, 37))
        matrix[bad, 0] = np.inf
        with pytest.raises(FeatureValidationError, match=rf"{len(bad)} row\(s\)") as err:
            validate_matrix(matrix, label="x", expected_rows=N, block_rows=BLOCK)
        assert str(bad[:10]) in str(err.value)

    def test_invalid_block_size_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            validate_matrix(_good(), label="x", expected_rows=N, block_rows=0)

    def test_default_block_is_bounded(self) -> None:
        assert 0 < BLOCK_ROWS <= 1 << 20


class TestBoundedHostTemporaries:
    def test_memmap_validation_allocates_per_block_not_per_catalogue(self, tmp_path) -> None:
        rows, dim, block = 200_000, 32, 2048
        big = np.ones((rows, dim), dtype=np.float32)
        matrix = _memmap(tmp_path, big)
        del big
        catalogue_bytes = rows * dim * 4

        tracemalloc.start()
        validate_matrix(matrix, label="x", expected_rows=rows, block_rows=block)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # One float32 block plus its boolean/norm reductions: far below the
        # catalogue and below the old full mask + full norm vector.
        assert peak < catalogue_bytes / 8
        assert peak < 8 * block * dim * 4


class TestUnknownFootprintIsNotFree:
    def test_cpu_pool_with_unknown_footprint_is_charged_the_floor(self, monkeypatch) -> None:
        seen: dict = {}

        def _plan(**kwargs):
            seen.update(kwargs)
            return 1

        monkeypatch.setattr(parallel_mod, "plan_pool_workers", _plan)
        detect_max_workers("cpu", 0, reserve_bytes=4 * 1024**3)

        assert seen["per_worker_bytes"] == UNKNOWN_WORKER_FOOTPRINT_BYTES > 0

    def test_known_footprint_is_passed_through(self, monkeypatch) -> None:
        seen: dict = {}
        monkeypatch.setattr(parallel_mod, "plan_pool_workers", lambda **kw: seen.update(kw) or 1)
        detect_max_workers("cpu", 3 * 1024**3, reserve_bytes=4 * 1024**3)
        assert seen["per_worker_bytes"] == 3 * 1024**3
