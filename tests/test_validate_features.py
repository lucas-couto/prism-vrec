"""Feature sanity gate (Task G).

Synthetic good/bad matrices exercise every check: shape, native dim,
dtype, NaN/Inf, zero-norm rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.steps.validate_features import (
    FeatureValidationError,
    gate_dataset_features,
    validate_backbone_feature,
    validate_matrix,
)

_N_ITEMS = 20
_DIM = 8


def _good(rng_seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    m = rng.standard_normal((_N_ITEMS, _DIM)).astype(np.float32)
    # guarantee no accidental zero-norm row
    m += 1.0
    return m


class TestValidateMatrix:
    def test_good_matrix_passes_and_returns_stats(self) -> None:
        stats = validate_matrix(_good(), label="ds/bb", expected_rows=_N_ITEMS, expected_dim=_DIM)
        assert stats["rows"] == _N_ITEMS and stats["dim"] == _DIM
        assert stats["norm_min"] > 0

    def test_wrong_row_count_raises(self) -> None:
        m = _good()[:-1]
        with pytest.raises(FeatureValidationError, match="row count"):
            validate_matrix(m, label="ds/bb", expected_rows=_N_ITEMS, expected_dim=_DIM)

    def test_wrong_dim_raises(self) -> None:
        with pytest.raises(FeatureValidationError, match="native dim"):
            validate_matrix(_good(), label="ds/bb", expected_rows=_N_ITEMS, expected_dim=_DIM + 1)

    def test_wrong_dtype_raises(self) -> None:
        m = _good().astype(np.float64)
        with pytest.raises(FeatureValidationError, match="dtype"):
            validate_matrix(m, label="ds/bb", expected_rows=_N_ITEMS, expected_dim=_DIM)

    def test_nan_raises_and_lists_item_idx(self) -> None:
        m = _good()
        m[3] = np.nan
        with pytest.raises(FeatureValidationError, match="NaN/Inf.*3"):
            validate_matrix(m, label="ds/bb", expected_rows=_N_ITEMS, expected_dim=_DIM)

    def test_inf_raises(self) -> None:
        m = _good()
        m[7, 0] = np.inf
        with pytest.raises(FeatureValidationError, match="NaN/Inf"):
            validate_matrix(m, label="ds/bb", expected_rows=_N_ITEMS, expected_dim=_DIM)

    def test_zero_norm_row_raises_and_lists_item_idx(self) -> None:
        m = _good()
        m[5] = 0.0
        with pytest.raises(FeatureValidationError, match=r"norm.*5"):
            validate_matrix(m, label="ds/bb", expected_rows=_N_ITEMS, expected_dim=_DIM)

    def test_dim_check_skipped_when_none(self) -> None:
        # Fused matrices pass expected_dim=None (dim depends on strategy).
        stats = validate_matrix(_good(), label="ds/hybrid", expected_rows=_N_ITEMS)
        assert stats["dim"] == _DIM


def _write_dataset(tmp_path: Path, matrix: np.ndarray) -> tuple[Path, Path, dict]:
    emb = tmp_path / "embeddings" / "ds"
    proc = tmp_path / "processed" / "ds"
    emb.mkdir(parents=True)
    proc.mkdir(parents=True)
    np.save(emb / "resnet50.npy", matrix)
    with open(proc / "item2idx.json", "w") as fh:
        json.dump({str(i): i for i in range(_N_ITEMS)}, fh)
    config = {"extractors": {"resnet50": {"raw_dim": _DIM}}, "extractors_enabled": ["resnet50"]}
    return tmp_path / "embeddings", tmp_path / "processed", config


class TestBackboneFeatureFromDisk:
    def test_valid_file_passes(self, tmp_path: Path) -> None:
        emb, proc, config = _write_dataset(tmp_path, _good())
        stats = validate_backbone_feature(
            "ds", "resnet50", config, embeddings_dir=emb, processed_dir=proc
        )
        assert stats["dim"] == _DIM

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        _, proc, config = _write_dataset(tmp_path, _good())
        with pytest.raises(FeatureValidationError, match="missing"):
            validate_backbone_feature(
                "ds", "vit_b16", config, embeddings_dir=tmp_path / "embeddings", processed_dir=proc
            )

    def test_gate_raises_on_a_bad_matrix(self, tmp_path: Path) -> None:
        bad = _good()
        bad[0] = np.nan
        emb, proc, config = _write_dataset(tmp_path, bad)
        with pytest.raises(FeatureValidationError):
            gate_dataset_features(["ds"], config, embeddings_dir=emb, processed_dir=proc)

    def test_gate_passes_on_good_matrix(self, tmp_path: Path) -> None:
        emb, proc, config = _write_dataset(tmp_path, _good())
        gate_dataset_features(["ds"], config, embeddings_dir=emb, processed_dir=proc)
