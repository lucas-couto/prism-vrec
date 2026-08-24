"""Integration tests for the post-hoc beyond-accuracy step (06b).

The step must consume ONLY the persisted per-user artifacts (top_items)
plus train-only statistics, merge the new columns into the same
evaluation tables the accuracy metrics live in, and honour the category
contract (Tradesy-style datasets get NO cat_entropy column).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.persistence import CellMetadata, write_cell_artifact
from src.steps import beyond_accuracy as ba_step

_DS = "toyset"
_N_USERS = 4
_N_ITEMS = 12
_SEED = 42
_KS = [2, 3]


def _write_processed(root: Path) -> None:
    base = root / _DS
    base.mkdir(parents=True)
    # Items 0..9 appear in train; items 10, 11 never do (zero popularity).
    rows = [(u, i) for u in range(_N_USERS) for i in range(10) if (u + i) % 2 == 0]
    pd.DataFrame(rows, columns=["user_idx", "item_idx"]).to_csv(base / "train.csv", index=False)
    (base / "user2idx.json").write_text(json.dumps({f"u{i}": i for i in range(_N_USERS)}))
    (base / "item2idx.json").write_text(json.dumps({f"i{i}": i for i in range(_N_ITEMS)}))


def _write_reference_embedding(root: Path) -> None:
    (root / _DS).mkdir(parents=True)
    rng = np.random.default_rng(0)
    np.save(root / _DS / "resnet50.npy", rng.standard_normal((_N_ITEMS, 8)).astype(np.float32))


def _top_items(offset: int) -> list[list[int]]:
    return [[(u + offset + j) % _N_ITEMS for j in range(5)] for u in range(_N_USERS)]


def _write_cell(results: Path, recommender: str, visual: str, offset: int) -> None:
    records = pd.DataFrame(
        {
            "user_idx": list(range(_N_USERS)),
            "rank": [1 + u for u in range(_N_USERS)],
            "n_candidates": [_N_ITEMS] * _N_USERS,
            "tie_block_size": [1] * _N_USERS,
            "top_items": _top_items(offset),
        }
    )
    metadata = CellMetadata(
        dataset=_DS,
        visual_config=visual,
        recommender=recommender,
        seed=_SEED,
        d=8,
        split="test",
        n_users=_N_USERS,
        n_items=_N_ITEMS,
    )
    write_cell_artifact(records, metadata, results)


def _write_eval_table(tables: Path, target: str, cells: list[tuple[str, str]]) -> Path:
    rows = []
    for recommender, visual in cells:
        for u in range(_N_USERS):
            rows.append(
                {
                    "user_id": u,
                    "recall@2": float(u % 2),
                    "ndcg@2": 0.5,
                    "dataset": _DS,
                    "model_name": recommender,
                    "embedding_name": visual,
                }
            )
    tables.mkdir(parents=True, exist_ok=True)
    path = tables / f"{_DS}_evaluation_{target}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _config(tmp_path: Path, *, expects_categories: bool | None = False) -> dict:
    contracts = {}
    if expects_categories is not None:
        contracts[_DS] = {"expects_categories": expects_categories}
    return {
        "seed": _SEED,
        "datasets": [_DS],
        "k_values": _KS,
        "dataset_contracts": contracts,
        "paths": {
            "data_processed": str(tmp_path / "processed"),
            "embeddings": str(tmp_path / "embeddings"),
            "results": str(tmp_path / "results"),
        },
        "beyond_accuracy": {"enabled": True, "reference_embedding": "resnet50"},
    }


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    _write_processed(tmp_path / "processed")
    _write_reference_embedding(tmp_path / "embeddings")
    results = tmp_path / "results"
    _write_cell(results, "bpr", "none", offset=0)
    _write_cell(results, "vbpr", "resnet50_D8", offset=3)
    _write_eval_table(results / "tables", "frozen", [("bpr", "none"), ("vbpr", "resnet50_D8")])
    return tmp_path


def _run(monkeypatch: pytest.MonkeyPatch, config: dict) -> None:
    monkeypatch.setattr(ba_step, "load_config", lambda: config)
    ba_step.run()


class TestBeyondAccuracyStep:
    def test_merges_columns_into_evaluation_table(self, workspace, monkeypatch) -> None:
        _run(monkeypatch, _config(workspace))

        table = pd.read_csv(workspace / "results/tables" / f"{_DS}_evaluation_frozen.csv")
        for k in _KS:
            assert f"efd@{k}" in table.columns
            assert f"ild@{k}" in table.columns
            assert f"icov@{k}" in table.columns
        # Existing accuracy columns and row count survive the merge.
        assert "recall@2" in table.columns
        assert len(table) == 2 * _N_USERS
        assert not table["efd@2"].isna().any()
        assert not table["ild@3"].isna().any()

    def test_icov_constant_per_cell_and_coverage_csv(self, workspace, monkeypatch) -> None:
        _run(monkeypatch, _config(workspace))

        table = pd.read_csv(workspace / "results/tables" / f"{_DS}_evaluation_frozen.csv")
        # Aggregate metric: one value per cell, replicated across users.
        assert (table.groupby(["model_name", "embedding_name"])["icov@2"].nunique() == 1).all()

        coverage = pd.read_csv(workspace / "results/tables" / f"{_DS}_beyond_accuracy_coverage.csv")
        assert set(coverage.columns) >= {"model_name", "embedding_name", "k", "icov"}
        assert len(coverage) == 2 * len(_KS)
        # bpr top-2 lists: users 0..3 recommend {0..4} pairwise -> hand value.
        tops = _top_items(0)
        expected = len({i for top in tops for i in top[:2]}) / 10  # 10 train items
        bpr_row = coverage[(coverage["model_name"] == "bpr") & (coverage["k"] == 2)]
        assert bpr_row["icov"].iloc[0] == pytest.approx(expected)

    def test_no_cat_entropy_for_contractless_dataset(self, workspace, monkeypatch) -> None:
        # expects_categories: false (Tradesy-style) -> explicit N/A.
        _run(monkeypatch, _config(workspace, expects_categories=False))
        table = pd.read_csv(workspace / "results/tables" / f"{_DS}_evaluation_frozen.csv")
        assert not any(c.startswith("cat_entropy@") for c in table.columns)

    def test_cat_entropy_written_when_contract_expects(self, workspace, monkeypatch) -> None:
        import src.data.categories as categories_mod

        cats = np.arange(_N_ITEMS) % 3
        monkeypatch.setattr(categories_mod, "item_category_array", lambda *a, **kw: cats)
        _run(monkeypatch, _config(workspace, expects_categories=True))
        table = pd.read_csv(workspace / "results/tables" / f"{_DS}_evaluation_frozen.csv")
        for k in _KS:
            assert f"cat_entropy@{k}" in table.columns
        assert not table["cat_entropy@2"].isna().any()

    def test_missing_reference_embedding_fails_loud(self, workspace, monkeypatch) -> None:
        (workspace / "embeddings" / _DS / "resnet50.npy").unlink()
        with pytest.raises(FileNotFoundError, match="reference embedding"):
            _run(monkeypatch, _config(workspace))

    def test_rerun_is_idempotent(self, workspace, monkeypatch) -> None:
        config = _config(workspace)
        _run(monkeypatch, config)
        first = pd.read_csv(workspace / "results/tables" / f"{_DS}_evaluation_frozen.csv")
        _run(monkeypatch, config)
        second = pd.read_csv(workspace / "results/tables" / f"{_DS}_evaluation_frozen.csv")
        assert list(first.columns) == list(second.columns)
        pd.testing.assert_frame_equal(first, second)

    def test_mean_table_refreshed_with_new_columns(self, workspace, monkeypatch) -> None:
        _run(monkeypatch, _config(workspace))
        mean = pd.read_csv(workspace / "results/tables" / f"{_DS}_evaluation_mean_frozen.csv")
        assert "efd@2" in mean.columns
        assert "ild@3" in mean.columns
        assert len(mean) == 2  # one row per cell

    def test_disabled_step_is_a_noop(self, workspace, monkeypatch) -> None:
        config = _config(workspace)
        config["beyond_accuracy"]["enabled"] = False
        before = pd.read_csv(workspace / "results/tables" / f"{_DS}_evaluation_frozen.csv")
        _run(monkeypatch, config)
        after = pd.read_csv(workspace / "results/tables" / f"{_DS}_evaluation_frozen.csv")
        pd.testing.assert_frame_equal(before, after)
