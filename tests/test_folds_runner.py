"""End-to-end K-fold runner on a synthetic dataset (CPU, fixed hyperparameters).

Exercises the whole chain — partition → frozen-hyperparameter training on
the other folds → fold-in of the held-out users → single-target ranking →
concatenation of the partial artifacts → manifest — the way
``python main.py --folds`` runs it, with a two-model battery (BPR, VBPR).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.persistence import read_cell_artifact
from src.folds.runner import manifest_path, run_folds

N_USERS, N_ITEMS, DV = 12, 30, 4


def _write_dataset(root: Path) -> tuple[str, str]:
    rng = np.random.default_rng(0)
    proc = root / "processed" / "synthetic"
    emb = root / "embeddings" / "synthetic"
    proc.mkdir(parents=True)
    emb.mkdir(parents=True)
    train, val, test = [], [], []
    for u in range(N_USERS):
        items = rng.choice(N_ITEMS, size=6, replace=False).tolist()
        train += [(u, i) for i in items[:4]]
        val.append((u, items[4]))
        test.append((u, items[5]))
    for name, rows in (("train", train), ("val", val), ("test", test)):
        pd.DataFrame(rows, columns=["user_idx", "item_idx"]).to_csv(
            proc / f"{name}.csv", index=False
        )
    (proc / "user2idx.json").write_text(json.dumps({str(i): i for i in range(N_USERS)}))
    (proc / "item2idx.json").write_text(json.dumps({str(i): i for i in range(N_ITEMS)}))
    np.save(emb / "resnet50.npy", rng.standard_normal((N_ITEMS, DV)).astype("float32"))
    return str(root / "processed"), str(root / "embeddings")


def _config(root: Path, processed: str, embeddings: str, *, enabled: bool = True) -> dict:
    return {
        "seed": 3,
        "device": "cpu",
        "datasets": ["synthetic"],
        "recommenders_enabled": ["bpr", "vbpr"],
        "extractors_enabled": ["resnet50"],
        "fusion_strategies_enabled": [],
        "embedding_variants": "native",
        "pipeline": {"condition": "frozen"},
        "paths": {
            "data_processed": processed,
            "embeddings": embeddings,
            "results": str(root / "results"),
            "checkpoints": str(root / "checkpoints"),
        },
        "common": {
            "total_dim": 4,
            "learning_rate": 0.05,
            "l2_reg": 0.0001,
            "epochs": 2,
            "batch_size": 8,
            "early_stopping_patience": 1,
            "early_stopping_metric": "ndcg@10",
            "eval_every_epochs": 1,
            "eval_sample_size": None,
        },
        "hp_search": {"strategy": "fixed", "workers": 1, "optuna": {"n_trials": 1}},
        "evaluation": {"protocol": "full_ranking"},
        "k_values": [5, 10],
        "folds": {
            "enabled": enabled,
            "k": 2,
            "seed": 7,
            "min_profile": 1,
            "fold_in": {"epochs": 1, "learning_rate": None, "batch_size": None},
        },
    }


@pytest.fixture()
def synthetic(tmp_path: Path) -> dict:
    processed, embeddings = _write_dataset(tmp_path)
    return _config(tmp_path, processed, embeddings)


class TestRunFolds:
    def test_should_refuse_to_run_when_folds_are_disabled(self, tmp_path: Path) -> None:
        processed, embeddings = _write_dataset(tmp_path)
        cfg = _config(tmp_path, processed, embeddings, enabled=False)

        with pytest.raises(RuntimeError, match="folds.enabled"):
            run_folds(cfg, cfg["paths"]["results"])

    def test_should_evaluate_every_user_exactly_once_across_the_folds(self, synthetic) -> None:
        results = Path(synthetic["paths"]["results"])

        manifest = run_folds(synthetic, results)

        assert manifest.summary()["done"] == 2 and manifest.summary()["failed"] == 0
        for cell_key, entry in manifest.cells.items():
            records_path = results / "per_user" / "synthetic" / f"{_artifact_key(entry)}.csv.gz"
            _, df = read_cell_artifact(records_path)
            assert sorted(df["user_idx"].tolist()) == list(range(N_USERS)), cell_key
            assert (df["rank"] >= 1).all()

    def test_should_record_origin_partition_folds_and_variability(self, synthetic) -> None:
        results = Path(synthetic["paths"]["results"])

        manifest = run_folds(synthetic, results)

        entry = next(iter(manifest.cells.values()))
        assert entry["hyperparam_origin"]["source"] == "fixed"
        assert entry["hyperparam_origin"]["hyperparams"]["latent_dim"] in (2, 4)
        assert entry["partition"]["fold_sizes"] == [6, 6]
        assert [f["seed"] for f in entry["folds"]] == [7, 8]
        assert all(f["fold_in"]["n_users"] == 6 for f in entry["folds"])
        assert set(entry["aggregate"]["between_fold_mean"]) >= {"recall@5", "ndcg@10"}
        assert entry["aggregate"]["n_users_total"] == N_USERS
        assert "combined" in entry["variability"]
        assert manifest_path(results).exists()

    def test_should_skip_cells_whose_artifact_already_exists(self, synthetic, monkeypatch) -> None:
        results = Path(synthetic["paths"]["results"])
        run_folds(synthetic, results)

        def _explode(*args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("cell re-executed despite an existing artifact")

        manifest = run_folds(synthetic, results, execute=_explode)

        assert manifest.summary()["done"] == 2
        assert all(
            e.get("note") == "fold artifact already present" for e in manifest.cells.values()
        )

    def test_should_not_skip_cell_whose_leave_one_out_artifact_shares_the_path(
        self, synthetic, monkeypatch
    ) -> None:
        results = Path(synthetic["paths"]["results"])
        run_folds(synthetic, results)
        for meta_path in results.glob("per_user/*/*.meta.json"):
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["fold"] = None  # what the leave-one-out evaluate step writes
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
        manifest_path(results).unlink()
        executed: list[str] = []

        def _record(cell, *args, **kwargs):
            executed.append(cell.key())
            return {}

        manifest = run_folds(synthetic, results, execute=_record)

        assert len(executed) == manifest.summary()["done"] == 2
        assert not any(
            e.get("note") == "fold artifact already present" for e in manifest.cells.values()
        )

    def test_should_tag_training_log_lines_with_the_fold(self, synthetic, caplog) -> None:
        """Every per-epoch ``timing`` line and a ``Fold i/k`` header name the fold."""
        from src.utils.logging import get_logger

        # Framework loggers do not propagate to the root logger (each owns
        # its console/file handlers), so caplog has to be attached to them.
        names = ["src.folds.runner", *(f"train_{m}" for m in synthetic["recommenders_enabled"])]
        for name in names:
            get_logger(name).addHandler(caplog.handler)
        caplog.set_level(logging.INFO)
        results = Path(synthetic["paths"]["results"])

        try:
            run_folds(synthetic, results)
        finally:
            for name in names:
                logging.getLogger(name).removeHandler(caplog.handler)

        messages = [r.getMessage() for r in caplog.records]
        k = synthetic["folds"]["k"]
        timing = [m for m in messages if m.startswith("timing dataset=")]
        assert timing, "no timing line captured"
        assert all("fold=" in m for m in timing), timing[:3]
        assert any(m.endswith(f"fold=1/{k}") for m in timing)
        assert any(m.startswith(f"Fold 1/{k} ") for m in messages)

    def test_should_keep_partial_artifacts_per_fold(self, synthetic) -> None:
        results = Path(synthetic["paths"]["results"])

        run_folds(synthetic, results)

        partials = sorted((results / "folds").glob("fold*/per_user/synthetic/*.csv.gz"))
        assert len(partials) == 4  # 2 cells x 2 folds
        sizes = [len(read_cell_artifact(p)[1]) for p in partials]
        assert all(size == N_USERS // 2 for size in sizes)


def _artifact_key(entry: dict) -> str:
    from src.evaluation.persistence import cell_key

    return cell_key(entry["dataset"], entry["visual_config"], entry["recommender"], 7)
