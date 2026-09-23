"""The consolidation must respect the run's scope and bounded memory.

Two defects found 2026-09-09 on the first real back-half run:

* it globbed the whole ``results/tables/`` directory, so a ``frozen``
  amazon_men run also consolidated amazon_fashion ``finetuned`` tables
  written two days earlier — silent contamination between runs;
* it melted each per-user table BEFORE aggregating, turning a
  2.57 M-row table with 30 metric columns into a 77 M-row intermediate
  on the way to a 2 025-row output, and was OOM-killed at the
  container's 16 GB limit.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.reporting import consolidate as consolidate_mod
from src.reporting.consolidate import consolidate_evaluation


def _table(tables: Path, dataset: str, condition: str, models: list[str], n_users: int) -> None:
    tables.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in models:
        for user in range(n_users):
            rows.append(
                {
                    "user_id": user,
                    "ndcg@10": (user + 1) / (n_users + 1),
                    "recall@10": 1.0 if user % 2 == 0 else 0.0,
                    "dataset": dataset,
                    "model_name": model,
                    "embedding_name": "resnet50",
                }
            )
    pd.DataFrame(rows).to_csv(tables / f"{dataset}_evaluation_{condition}.csv", index=False)


def test_a_table_outside_the_run_scope_is_skipped(tmp_path: Path) -> None:
    _table(tmp_path, "amazon_men", "frozen", ["vbpr"], 4)
    _table(tmp_path, "amazon_fashion", "finetuned", ["bpr"], 4)

    scoped = consolidate_evaluation(tmp_path, datasets={"amazon_men"}, conditions={"frozen"})

    assert set(scoped["dataset"]) == {"amazon_men"}
    assert "amazon_fashion" not in set(scoped["dataset"])


def test_no_scope_keeps_sweeping_everything(tmp_path: Path) -> None:
    _table(tmp_path, "amazon_men", "frozen", ["vbpr"], 4)
    _table(tmp_path, "amazon_fashion", "finetuned", ["bpr"], 4)

    swept = consolidate_evaluation(tmp_path)

    assert set(swept["dataset"]) == {"amazon_men", "amazon_fashion"}


def test_chunking_gives_the_same_numbers_as_one_pass(tmp_path: Path, monkeypatch) -> None:
    _table(tmp_path, "amazon_men", "frozen", ["vbpr", "bpr"], 50)

    whole = consolidate_evaluation(tmp_path, datasets={"amazon_men"}, conditions={"frozen"})
    monkeypatch.setattr(consolidate_mod, "EVALUATION_CHUNK_ROWS", 7)
    chunked = consolidate_evaluation(tmp_path, datasets={"amazon_men"}, conditions={"frozen"})

    key = ["recommender", "metric", "k"]
    left = whole.sort_values(key).reset_index(drop=True)
    right = chunked.sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(left[[*key, "n_users", "mean"]], right[[*key, "n_users", "mean"]])


def test_every_user_is_counted_exactly_once(tmp_path: Path, monkeypatch) -> None:
    _table(tmp_path, "amazon_men", "frozen", ["vbpr"], 33)
    monkeypatch.setattr(consolidate_mod, "EVALUATION_CHUNK_ROWS", 5)

    out = consolidate_evaluation(tmp_path, datasets={"amazon_men"}, conditions={"frozen"})

    assert set(out["n_users"]) == {33}, "a chunk boundary dropped or double-counted users"
