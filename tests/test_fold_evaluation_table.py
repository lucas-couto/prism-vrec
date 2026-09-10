"""The folds owe the back half its input table.

Found 2026-09-09: with `folds.enabled` the folds replaced `evaluate` as
the producer of the per-user records but wrote only
``results/per_user/``; `statistical` then logged "Evaluation file not
found" and completed in under a second having tested nothing.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from src.folds.evaluation_table import UNAVAILABLE_FROM_FOLDS, write_evaluation_table


def _cell(results: Path, dataset: str, emb: str, model: str, ranks: list[int]) -> None:
    base = results / "per_user" / dataset
    base.mkdir(parents=True, exist_ok=True)
    key = f"{dataset}__{emb}__{model}__seed42"
    frame = pd.DataFrame(
        {
            "user_idx": range(len(ranks)),
            "rank": ranks,
            "n_candidates": [100] * len(ranks),
            "tie_block_size": [1] * len(ranks),
            "top_items": ["[1, 2, 3]"] * len(ranks),
        }
    )
    with gzip.open(base / f"{key}.csv.gz", "wt", newline="") as fh:
        frame.to_csv(fh, index=False)
    (base / f"{key}.meta.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "visual_config": emb,
                "recommender": model,
                "seed": 42,
                "n_users": len(ranks),
                "n_items": 100,
            }
        )
    )


def test_the_table_carries_one_row_per_user_per_cell(tmp_path: Path) -> None:
    _cell(tmp_path, "ds", "resnet50", "vbpr", [1, 4, 50])
    _cell(tmp_path, "ds", "resnet50", "bpr", [2, 99, 7])

    out = write_evaluation_table(tmp_path, "ds", "frozen", [10])

    table = pd.read_csv(out)
    assert out.name == "ds_evaluation_frozen.csv"
    assert len(table) == 6
    assert set(table["model_name"]) == {"vbpr", "bpr"}
    assert "user_id" in table.columns


def test_metrics_come_from_the_rank_alone(tmp_path: Path) -> None:
    """rank 1 -> ndcg 1.0; rank 4 -> 1/log2(5); rank 50 -> 0 at k=10."""
    _cell(tmp_path, "ds", "resnet50", "vbpr", [1, 4, 50])

    table = pd.read_csv(write_evaluation_table(tmp_path, "ds", "frozen", [10]))

    assert table["ndcg@10"].tolist() == pytest.approx([1.0, 0.4306765580733931, 0.0])
    assert table["recall@10"].tolist() == [1.0, 1.0, 0.0]
    assert table["precision@10"].tolist() == pytest.approx([0.1, 0.1, 0.0])


def test_every_configured_cutoff_gets_its_columns(tmp_path: Path) -> None:
    _cell(tmp_path, "ds", "resnet50", "vbpr", [1, 12])

    table = pd.read_csv(write_evaluation_table(tmp_path, "ds", "frozen", [5, 10, 20]))

    for k in (5, 10, 20):
        assert {f"precision@{k}", f"recall@{k}", f"f1@{k}", f"map@{k}", f"ndcg@{k}"} <= set(
            table.columns
        )
    assert table["ndcg@5"].tolist()[1] == 0.0  # rank 12 misses k=5
    assert table["ndcg@20"].tolist()[1] > 0.0


def test_columns_that_describe_one_model_instance_are_omitted_not_guessed(
    tmp_path: Path,
) -> None:
    _cell(tmp_path, "ds", "resnet50", "vbpr", [1])

    table = pd.read_csv(write_evaluation_table(tmp_path, "ds", "frozen", [10]))

    for column in UNAVAILABLE_FROM_FOLDS:
        assert column not in table.columns


def test_a_dataset_without_artifacts_is_not_an_error(tmp_path: Path) -> None:
    assert write_evaluation_table(tmp_path, "missing", "frozen", [10]) is None
