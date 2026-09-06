"""Training workers consume the parent's resolved configuration snapshot (E03, F05).

A spawned worker starts with fresh module globals, so ``load_config()``
inside it used to re-read the YAML defaults and silently drop the
``--config-dir`` / multi-seed / CLI overrides and the result/checkpoint
roots the parent had resolved.  The snapshot now travels through the
process arguments; the in-process path must not touch the loader at
all, and the spawned path must write under the snapshot's roots.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

import src.utils.config as config_mod
from src.utils.parallel import (
    TrainingJob,
    TrainingOrchestrator,
    _WorkerContext,
    checkpoint_root,
    grid_progress_path,
)

DATASET = "synthetic"
N_USERS, N_ITEMS = 6, 8
HP = {"learning_rate": 0.05, "latent_dim": 4, "l2_reg": 0.0}


def _write_dataset(root: Path) -> str:
    base = root / "processed" / DATASET
    base.mkdir(parents=True)
    train = [(u, (u + i) % N_ITEMS) for u in range(N_USERS) for i in range(3)]
    val = [(u, (u + 3) % N_ITEMS) for u in range(N_USERS)]
    test = [(u, (u + 4) % N_ITEMS) for u in range(N_USERS)]
    for name, rows in (("train", train), ("val", val), ("test", test)):
        pd.DataFrame(rows, columns=["user_idx", "item_idx"]).to_csv(
            base / f"{name}.csv", index=False
        )
    (base / "user2idx.json").write_text(json.dumps({str(u): u for u in range(N_USERS)}))
    (base / "item2idx.json").write_text(json.dumps({str(i): i for i in range(N_ITEMS)}))
    return str(root / "processed")


def _config(root: Path, processed: str) -> dict:
    return {
        "seed": 7,
        "device": "cpu",
        "paths": {
            "data_processed": processed,
            "embeddings": str(root / "embeddings"),
            "results": str(root / "custom_results"),
            "checkpoints": str(root / "custom_checkpoints"),
        },
        "common": {
            "epochs": 1,
            "batch_size": 4,
            "early_stopping_patience": 1,
            "early_stopping_metric": "ndcg@10",
            "eval_every_epochs": 1,
            "eval_sample_size": None,
        },
        "telemetry": {"enabled": False},
    }


def _job(processed: str) -> TrainingJob:
    return TrainingJob(
        dataset_name=DATASET,
        model_name="bpr",
        embedding_name="none",
        hyperparams=dict(HP),
        n_users=N_USERS,
        n_items=N_ITEMS,
        embeddings_path=None,
        processed_dir=processed,
        device="cpu",
    )


def _expected_artifacts(cfg: dict) -> tuple[Path, Path]:
    best = Path(cfg["paths"]["results"]) / "models" / DATASET / "bpr_none_best.pt"
    progress = Path(cfg["paths"]["checkpoints"]) / "grid_search" / f"{DATASET}_none_bpr.json"
    return best, progress


class TestInProcessWorker:
    def test_should_train_from_the_snapshot_without_touching_the_loader(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        processed = _write_dataset(tmp_path)
        cfg = _config(tmp_path, processed)

        def _forbidden(*args, **kwargs):
            raise AssertionError("worker reloaded the configuration from disk")

        monkeypatch.setattr(config_mod, "load_config", _forbidden)
        orchestrator = TrainingOrchestrator(
            n_workers=1, device="cpu", log_dir=str(tmp_path / "logs"), config=cfg
        )

        results = orchestrator.run([_job(processed)])

        assert [r["status"] for r in results] == ["ok"]
        best, progress = _expected_artifacts(cfg)
        assert best.exists(), "the winner must land under the snapshot's results root"
        assert progress.exists(), "grid progress must land under the snapshot's checkpoint root"
        assert not (tmp_path / "checkpoints").exists()
        assert not (tmp_path / "results").exists()

    def test_should_warn_and_fall_back_to_disk_when_no_snapshot_is_given(
        self, tmp_path, monkeypatch, caplog
    ) -> None:
        processed = _write_dataset(tmp_path)
        cfg = _config(tmp_path, processed)
        monkeypatch.setattr(config_mod, "load_config", lambda *a, **k: cfg)

        with caplog.at_level(logging.WARNING):
            context = _WorkerContext(1, logging.getLogger("worker_test"), None)

        assert context._config is cfg
        assert str(context._checkpoint_mgr.checkpoint_dir) == cfg["paths"]["checkpoints"]
        assert "no resolved configuration snapshot" in caplog.text

    def test_checkpoint_root_defaults_when_paths_are_absent(self) -> None:
        assert checkpoint_root({}) == "checkpoints"
        assert checkpoint_root({"paths": {"checkpoints": "/x/ckpt"}}) == "/x/ckpt"

    def test_grid_progress_path_follows_the_manager_root(self, tmp_path) -> None:
        from src.utils.checkpoint import CheckpointManager

        manager = CheckpointManager(str(tmp_path / "root"))
        path = grid_progress_path(manager, "ds_emb_model")
        manager.save_grid_search_progress("ds_emb_model", [{"hyperparams": {}}])

        assert path.exists()
        assert manager.load_grid_search_progress("ds_emb_model") == [{"hyperparams": {}}]


class TestSpawnedWorker:
    def test_should_receive_the_snapshot_across_spawn(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        processed = _write_dataset(tmp_path)
        cfg = _config(tmp_path, processed)
        # ``device="cuda"`` selects the spawned pool; the job itself trains
        # on CPU and the VRAM probe reports "unknown" without a device.
        orchestrator = TrainingOrchestrator(
            n_workers=2, device="cuda", log_dir=str(tmp_path / "logs"), config=cfg
        )

        results = orchestrator.run([_job(processed)])

        assert [r["status"] for r in results] == ["ok"]
        best, progress = _expected_artifacts(cfg)
        assert best.exists()
        assert progress.exists()
        assert not (tmp_path / "checkpoints").exists()
        assert not (tmp_path / "results").exists()
