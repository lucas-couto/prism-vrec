"""Tests for the multi-seed feature: config parsing, path derivation, and
cross-seed aggregation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.reporting.aggregate_seeds import (
    aggregate_bootstrap_ci,
    aggregate_evaluation,
    write_cross_seed_aggregates,
)
from src.utils.config import derive_seed_config
from src.utils.config_schema import validate_config


class TestSchemaValidation:
    def test_seeds_default_is_none(self) -> None:
        cfg = validate_config({})
        assert cfg["seeds"] is None

    def test_seeds_accepts_unique_list(self) -> None:
        cfg = validate_config({"seeds": [42, 99, 7]})
        assert cfg["seeds"] == [42, 99, 7]

    def test_seeds_rejects_empty_list(self) -> None:
        with pytest.raises(Exception, match="seeds"):
            validate_config({"seeds": []})

    def test_seeds_rejects_duplicates(self) -> None:
        with pytest.raises(Exception, match="unique"):
            validate_config({"seeds": [42, 42]})

    def test_seeds_rejects_negative(self) -> None:
        with pytest.raises(Exception, match="non-negative"):
            validate_config({"seeds": [42, -1]})


class TestDeriveSeedConfig:
    def test_suffixes_results_and_checkpoints(self) -> None:
        cfg = {"seed": 42, "paths": {"results": "results", "checkpoints": "checkpoints"}}
        derived = derive_seed_config(cfg, 99)
        assert derived["seed"] == 99
        assert derived["paths"]["results"] == "results_seed99"
        assert derived["paths"]["checkpoints"] == "checkpoints_seed99"

    def test_preserves_shared_inputs(self) -> None:
        cfg = {
            "seed": 42,
            "paths": {
                "data_raw": "data/raw",
                "data_processed": "data/processed",
                "embeddings": "data/embeddings",
                "results": "results",
                "checkpoints": "checkpoints",
                "logs": "logs",
            },
        }
        derived = derive_seed_config(cfg, 99)
        # Inputs are seed-independent and stay at the base path
        assert derived["paths"]["data_raw"] == "data/raw"
        assert derived["paths"]["data_processed"] == "data/processed"
        assert derived["paths"]["embeddings"] == "data/embeddings"
        assert derived["paths"]["logs"] == "logs"

    def test_strips_seeds_list_from_derived(self) -> None:
        cfg = {"seed": 42, "seeds": [42, 99], "paths": {}}
        derived = derive_seed_config(cfg, 99)
        assert "seeds" not in derived

    def test_does_not_mutate_source(self) -> None:
        cfg = {"seed": 42, "paths": {"results": "results"}}
        derive_seed_config(cfg, 99)
        assert cfg["seed"] == 42
        assert cfg["paths"]["results"] == "results"

    def test_suffixes_sqlite_storage(self) -> None:
        cfg = {
            "seed": 42,
            "paths": {"results": "results", "checkpoints": "checkpoints"},
            "hp_search": {"optuna": {"storage": "sqlite:///optuna.db"}},
        }
        derived = derive_seed_config(cfg, 99)
        assert derived["hp_search"]["optuna"]["storage"] == "sqlite:///optuna_seed99.db"

    def test_leaves_non_sqlite_storage_alone(self) -> None:
        cfg = {
            "seed": 42,
            "paths": {},
            "hp_search": {"optuna": {"storage": "postgresql://host/db"}},
        }
        derived = derive_seed_config(cfg, 99)
        assert derived["hp_search"]["optuna"]["storage"] == "postgresql://host/db"


def _write_seed_csv(seed_dir: Path, filename: str, rows: list[dict]) -> None:
    (seed_dir / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(seed_dir / "tables" / filename, index=False)


class TestAggregateEvaluation:
    def test_computes_mean_std_across_seeds(self, tmp_path: Path) -> None:
        seed_dirs = []
        for seed, value in [(42, 0.5), (99, 0.6), (7, 0.55)]:
            d = tmp_path / f"results_seed{seed}"
            seed_dirs.append(d)
            _write_seed_csv(
                d,
                "evaluation_aggregated.csv",
                [
                    {
                        "dataset": "amazon_fashion",
                        "recommender": "vbpr",
                        "extractor": "resnet50",
                        "fusion": "none",
                        "condition": "frozen",
                        "metric": "ndcg",
                        "k": 10,
                        "mean": value,
                        "n_users": 100,
                    }
                ],
            )

        out = aggregate_evaluation(seed_dirs, seeds=[42, 99, 7])
        assert len(out) == 1
        row = out.iloc[0]
        assert row["n_seeds"] == 3
        assert row["mean_across_seeds"] == pytest.approx(0.55, abs=1e-9)
        assert row["min_across_seeds"] == pytest.approx(0.5)
        assert row["max_across_seeds"] == pytest.approx(0.6)
        assert row["std_across_seeds"] > 0

    def test_single_seed_gives_nan_std(self, tmp_path: Path) -> None:
        d = tmp_path / "results_seed42"
        _write_seed_csv(
            d,
            "evaluation_aggregated.csv",
            [
                {
                    "dataset": "x",
                    "recommender": "bpr",
                    "extractor": "none",
                    "fusion": "none",
                    "condition": "frozen",
                    "metric": "ndcg",
                    "k": 10,
                    "mean": 0.4,
                    "n_users": 100,
                }
            ],
        )
        out = aggregate_evaluation([d], seeds=[42])
        assert out.iloc[0]["n_seeds"] == 1
        assert pd.isna(out.iloc[0]["std_across_seeds"])

    def test_returns_empty_when_no_files(self, tmp_path: Path) -> None:
        out = aggregate_evaluation([tmp_path / "does_not_exist"], seeds=[42])
        assert out.empty


class TestAggregateBootstrapCi:
    def test_aggregates_ci_means(self, tmp_path: Path) -> None:
        seed_dirs = []
        for seed, mean in [(42, 0.5), (99, 0.6)]:
            d = tmp_path / f"results_seed{seed}"
            seed_dirs.append(d)
            _write_seed_csv(
                d,
                "bootstrap_ci.csv",
                [
                    {
                        "dataset": "x",
                        "recommender": "vbpr",
                        "extractor": "resnet50",
                        "fusion": "none",
                        "condition": "frozen",
                        "metric": "ndcg",
                        "k": 10,
                        "mean": mean,
                        "ci_lower": mean - 0.05,
                        "ci_upper": mean + 0.05,
                    }
                ],
            )
        out = aggregate_bootstrap_ci(seed_dirs, seeds=[42, 99])
        assert len(out) == 1
        assert out.iloc[0]["n_seeds"] == 2
        assert out.iloc[0]["mean_across_seeds"] == pytest.approx(0.55)


class TestWriteCrossSeedAggregates:
    def test_end_to_end(self, tmp_path: Path) -> None:
        seed_dirs = []
        for seed, value in [(42, 0.5), (99, 0.6)]:
            d = tmp_path / f"results_seed{seed}"
            seed_dirs.append(d)
            _write_seed_csv(
                d,
                "evaluation_aggregated.csv",
                [
                    {
                        "dataset": "x",
                        "recommender": "vbpr",
                        "extractor": "resnet50",
                        "fusion": "none",
                        "condition": "frozen",
                        "metric": "ndcg",
                        "k": 10,
                        "mean": value,
                        "n_users": 100,
                    }
                ],
            )
            _write_seed_csv(
                d,
                "bootstrap_ci.csv",
                [
                    {
                        "dataset": "x",
                        "recommender": "vbpr",
                        "extractor": "resnet50",
                        "fusion": "none",
                        "condition": "frozen",
                        "metric": "ndcg",
                        "k": 10,
                        "mean": value,
                        "ci_lower": value - 0.05,
                        "ci_upper": value + 0.05,
                    }
                ],
            )

        out_dir = tmp_path / "aggregated"
        written = write_cross_seed_aggregates(seed_dirs, out_dir, seeds=[42, 99])

        assert written["evaluation_multi_seed"].exists()
        assert written["bootstrap_ci_multi_seed"].exists()
        eval_df = pd.read_csv(written["evaluation_multi_seed"])
        assert len(eval_df) == 1
        assert eval_df.iloc[0]["n_seeds"] == 2
