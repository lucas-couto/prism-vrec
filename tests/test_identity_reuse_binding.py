"""Checkpoints and evaluation reuse are bound to the C02 identity (E04, Q13).

Same seed/config/data reuses; a changed seed, split, feature content,
hyperparameter budget or checkpoint bytes does not.  Legacy records
without a binding are identified and never reused.  The train step no
longer wipes every resume envelope at startup.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import src.steps.evaluate as ev
import src.steps.train as train_mod
from src.recommenders.bpr import BPR
from src.steps.train import build_job_list
from src.utils.checkpoint import CheckpointManager, ResumeStateError
from src.utils.identity import build_identity_context, clear_identity_cache, resolve_data_identity
from src.utils.parallel import TrainingOrchestrator
from src.utils.training import _save_best_model, resolve_training_identity, train_single_run

DATASET = "synthetic"
N_USERS, N_ITEMS, DIM = 6, 8, 3
HP = {"learning_rate": 0.05, "latent_dim": 4, "l2_reg": 0.0001}


def _write_dataset(root: Path, *, feature_seed: int = 0) -> tuple[str, str]:
    base = root / "processed" / DATASET
    base.mkdir(parents=True, exist_ok=True)
    train = [(u, (u + i) % N_ITEMS) for u in range(N_USERS) for i in range(3)]
    val = [(u, (u + 3) % N_ITEMS) for u in range(N_USERS)]
    test = [(u, (u + 4) % N_ITEMS) for u in range(N_USERS)]
    for name, rows in (("train", train), ("val", val), ("test", test)):
        pd.DataFrame(rows, columns=["user_idx", "item_idx"]).to_csv(
            base / f"{name}.csv", index=False
        )
    (base / "user2idx.json").write_text(json.dumps({str(u): u for u in range(N_USERS)}))
    (base / "item2idx.json").write_text(json.dumps({str(i): i for i in range(N_ITEMS)}))
    emb = root / "embeddings" / DATASET
    emb.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(feature_seed)
    np.save(emb / "resnet50.npy", rng.standard_normal((N_ITEMS, DIM)).astype(np.float32))
    return str(root / "processed"), str(root / "embeddings")


def _config(root: Path, processed: str, embeddings: str, *, seed: int = 7) -> dict:
    return {
        "seed": seed,
        "device": "cpu",
        "datasets": [DATASET],
        "recommenders_enabled": ["bpr"],
        "extractors_enabled": ["resnet50"],
        "fusion_strategies_enabled": [],
        "paths": {
            "data_processed": processed,
            "embeddings": embeddings,
            "results": str(root / "results"),
            "checkpoints": str(root / "checkpoints"),
        },
        "common": {
            "epochs": 1,
            "batch_size": 4,
            "early_stopping_patience": 1,
            "early_stopping_metric": "ndcg@10",
            "eval_every_epochs": 1,
            "eval_sample_size": None,
            "learning_rate": [0.05],
            "latent_dim": [4],
            "l2_reg": [0.0001],
        },
        "telemetry": {"enabled": False},
    }


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_identity_cache()
    yield
    clear_identity_cache()


class TestGridProgressReuse:
    def _run_once(self, tmp_path, cfg) -> list[dict]:
        jobs = build_job_list(
            "frozen", cfg, cfg["paths"]["data_processed"], cfg["paths"]["embeddings"], "cpu"
        )
        assert len(jobs) == 1 and jobs[0].data_identity is not None
        orchestrator = TrainingOrchestrator(
            n_workers=1, device="cpu", log_dir=str(tmp_path / "logs"), config=cfg
        )
        results = orchestrator.run(jobs)
        assert [r["status"] for r in results] == ["ok"]
        return jobs

    def test_same_identity_reuses_completed_work(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        processed, embeddings = _write_dataset(tmp_path)
        cfg = _config(tmp_path, processed, embeddings)
        self._run_once(tmp_path, cfg)

        again = build_job_list("frozen", cfg, processed, embeddings, "cpu")

        assert again == []

    @pytest.mark.parametrize("mutation", ["seed", "epochs", "split", "mapping"])
    def test_changed_identity_does_not_reuse_completed_work(
        self, tmp_path, monkeypatch, mutation
    ) -> None:
        monkeypatch.chdir(tmp_path)
        processed, embeddings = _write_dataset(tmp_path)
        cfg = _config(tmp_path, processed, embeddings)
        self._run_once(tmp_path, cfg)
        clear_identity_cache()
        if mutation == "seed":
            cfg["seed"] = 8
        elif mutation == "epochs":
            cfg["common"]["epochs"] = 2
        elif mutation == "split":
            val = Path(processed) / DATASET / "val.csv"
            val.write_text(val.read_text().replace("\n0,", "\n1,", 1))
        else:
            mapping = Path(processed) / DATASET / "item2idx.json"
            mapping.write_text(json.dumps({str(i): N_ITEMS - 1 - i for i in range(N_ITEMS)}))

        again = build_job_list("frozen", cfg, processed, embeddings, "cpu")

        assert len(again) == 1

    def test_legacy_entry_without_identity_is_not_reused(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        processed, embeddings = _write_dataset(tmp_path)
        cfg = _config(tmp_path, processed, embeddings)
        CheckpointManager(cfg["paths"]["checkpoints"]).save_grid_search_progress(
            f"{DATASET}_none_bpr", [{"hyperparams": dict(HP), "best_metric": 0.3}]
        )

        jobs = build_job_list("frozen", cfg, processed, embeddings, "cpu")

        assert len(jobs) == 1


class TestBestCheckpointScope:
    def _identity(self, tmp_path, processed, **overrides) -> dict:
        cfg = _config(tmp_path, processed, "")
        cfg.update({k: v for k, v in overrides.items() if k == "seed"})
        hp = overrides.get("hp", HP)
        data = resolve_data_identity(processed, DATASET, None)
        return resolve_training_identity(
            model_cls=BPR,
            model_name="bpr",
            dataset_name=DATASET,
            embedding_name="none",
            hyperparams=hp,
            config=cfg,
            identity_context=build_identity_context(data, condition="frozen"),
        )

    def _save(self, root, metric, identity) -> None:
        _save_best_model(
            {"w": torch.zeros(1)},
            dict(HP),
            metric,
            N_USERS,
            N_ITEMS,
            DATASET,
            "bpr",
            "none",
            "fp",
            results_root=root / "results",
            identity=identity,
        )

    def _stored_metric(self, root) -> float:
        path = root / "results" / "models" / DATASET / "bpr_none_best.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["identity"]["schema_version"] == 2
        assert payload["identity_digest"] and payload["selection_scope_digest"]
        return float(payload["best_metric"])

    def test_same_scope_keeps_the_better_trial(self, tmp_path) -> None:
        processed, _ = _write_dataset(tmp_path)
        self._save(tmp_path, 0.5, self._identity(tmp_path, processed))
        other_hp = {**HP, "latent_dim": 8}
        self._save(tmp_path, 0.1, self._identity(tmp_path, processed, hp=other_hp))

        assert self._stored_metric(tmp_path) == 0.5

    def test_another_seed_is_not_comparable_and_replaces_the_winner(self, tmp_path) -> None:
        processed, _ = _write_dataset(tmp_path)
        self._save(tmp_path, 0.5, self._identity(tmp_path, processed))
        self._save(tmp_path, 0.1, self._identity(tmp_path, processed, seed=99))

        assert self._stored_metric(tmp_path) == 0.1

    def test_legacy_best_without_scope_is_replaced(self, tmp_path) -> None:
        processed, _ = _write_dataset(tmp_path)
        _save_best_model(
            {"w": torch.zeros(1)},
            dict(HP),
            0.9,
            N_USERS,
            N_ITEMS,
            DATASET,
            "bpr",
            "none",
            "fp",
            results_root=tmp_path / "results",
        )
        self._save(tmp_path, 0.1, self._identity(tmp_path, processed))

        assert self._stored_metric(tmp_path) == 0.1


class TestResumeEnvelopeBinding:
    def _train(self, tmp_path, processed, context, manager) -> float:
        cfg = _config(tmp_path, processed, "")
        train = {u: {(u + i) % N_ITEMS for i in range(3)} for u in range(N_USERS)}
        val = {u: {(u + 3) % N_ITEMS} for u in range(N_USERS)}
        return train_single_run(
            model_cls=BPR,
            model_name="bpr",
            n_users=N_USERS,
            n_items=N_ITEMS,
            visual_embeddings=None,
            train_interactions=train,
            selection_interactions=val,
            hyperparams=dict(HP),
            config=cfg,
            checkpoint_mgr=manager,
            dataset_name=DATASET,
            embedding_name="none",
            device="cpu",
            identity_context=context,
        )

    def test_envelope_of_another_feature_content_is_refused(self, tmp_path, monkeypatch) -> None:
        processed, embeddings = _write_dataset(tmp_path, feature_seed=0)
        feature = Path(embeddings) / DATASET / "resnet50.npy"
        manager = CheckpointManager(str(tmp_path / "ckpt"))
        # Keep the envelope after the run so the next call must validate it.
        monkeypatch.setattr(manager, "clear_training_checkpoint", lambda run_id: None)
        first = build_identity_context(
            resolve_data_identity(processed, DATASET, feature), condition="frozen"
        )
        self._train(tmp_path, processed, first, manager)

        clear_identity_cache()
        _write_dataset(tmp_path, feature_seed=1)
        changed = build_identity_context(
            resolve_data_identity(processed, DATASET, feature), condition="frozen"
        )

        with pytest.raises(ResumeStateError, match="identity"):
            self._train(tmp_path, processed, changed, manager)

    def test_envelope_of_the_same_identity_validates(self, tmp_path, monkeypatch) -> None:
        """A normal completion deletes the trial file, so resume itself is the
        I04 genuine-kill test; here the envelope's identity must equal the
        digest recomputed from the same inputs and differ from any other."""
        from src.utils.checkpoint import validate_resume_envelope
        from src.utils.training import _training_resume_identity

        processed, embeddings = _write_dataset(tmp_path)
        feature = Path(embeddings) / DATASET / "resnet50.npy"
        manager = CheckpointManager(str(tmp_path / "ckpt"))
        monkeypatch.setattr(manager, "clear_training_checkpoint", lambda run_id: None)
        context = build_identity_context(
            resolve_data_identity(processed, DATASET, feature), condition="frozen"
        )
        self._train(tmp_path, processed, context, manager)
        run_id = CheckpointManager.get_run_id(DATASET, "none", "bpr", HP)
        envelope = manager.load_training_checkpoint(run_id)

        def _digest(ctx: dict) -> str:
            experiment = resolve_training_identity(
                model_cls=BPR,
                model_name="bpr",
                dataset_name=DATASET,
                embedding_name="none",
                hyperparams=dict(HP),
                config=_config(tmp_path, processed, ""),
                identity_context=ctx,
            )
            return _training_resume_identity(
                experiment,
                run_id=run_id,
                model_cls_name="BPR",
                n_users=N_USERS,
                n_items=N_ITEMS,
                job_seed=_job_seed(),
                fingerprint=envelope_fingerprint(tmp_path),
                use_cuda=False,
            )

        same = validate_resume_envelope(envelope, expected_identity=_digest(context), source="t")
        assert same["identity"] == _digest(context)
        other = build_identity_context(context["data"], condition="finetuned")
        with pytest.raises(ResumeStateError, match="identity"):
            validate_resume_envelope(envelope, expected_identity=_digest(other), source="t")


def _job_seed() -> int:
    from src.utils.training import _derive_job_seed

    return _derive_job_seed(7, DATASET, "bpr", "none", dict(HP))


def envelope_fingerprint(root: Path) -> str:
    from src.utils.training import selection_protocol_fingerprint

    return selection_protocol_fingerprint(
        dataset_name=DATASET,
        es_metric="ndcg@10",
        eval_sample_size=None,
        eval_sample_seed=7,
        tiebreak_seed=7,
        k_values=[10],
    )


class TestNoGlobalCleanup:
    def test_train_step_keeps_foreign_resume_envelopes(self, tmp_path, monkeypatch) -> None:
        processed, embeddings = _write_dataset(tmp_path)
        cfg = _config(tmp_path, processed, embeddings)
        foreign = Path(cfg["paths"]["checkpoints"]) / "training" / "other_run.pt"
        foreign.parent.mkdir(parents=True)
        foreign.write_bytes(b"belongs to another run")
        monkeypatch.setattr(train_mod, "load_config", lambda: cfg)
        monkeypatch.setattr(train_mod, "assert_dimension_parity", lambda c: None)
        monkeypatch.setattr(
            "src.recommenders.hp_budget.assert_uniform_budget", lambda c: None, raising=False
        )
        monkeypatch.setattr(
            "src.steps.validate_features.gate_dataset_features", lambda *a, **k: None
        )
        monkeypatch.setattr(train_mod, "_run_grid", lambda *a, **k: None)

        train_mod.run("frozen", workers=1, sequential=True)

        assert foreign.exists()


class TestEvaluationReuse:
    def _install(self, tmp_path, monkeypatch, calls: dict) -> dict:
        processed, embeddings = _write_dataset(tmp_path)
        cfg = _config(tmp_path, processed, embeddings)
        cfg["k_values"] = [10]
        monkeypatch.setattr(ev, "load_config", lambda: cfg)
        monkeypatch.setattr(ev, "resolve_device", lambda d: "cpu")
        monkeypatch.setattr(ev, "cap_process_vram", lambda *a, **k: None)
        monkeypatch.setattr(ev, "load_data", lambda p, d: (N_USERS, N_ITEMS, {}, {}, {}))
        winner = tmp_path / "bpr_none_best.pt"
        winner.write_bytes(b"winner-v1")
        monkeypatch.setattr(
            ev,
            "find_best_models",
            lambda d, **kw: [{"model_name": "bpr", "embedding_name": "none", "path": str(winner)}],
        )

        def _fake_cell(*a, **k):
            calls["n"] += 1
            return pd.DataFrame({"user_id": [1, 2], "ndcg@10": [0.5, 0.6]})

        monkeypatch.setattr(ev, "_evaluate_cell", _fake_cell)
        cfg["_winner"] = winner
        return cfg

    def test_same_checkpoint_and_identity_reuse(self, tmp_path, monkeypatch) -> None:
        calls = {"n": 0}
        self._install(tmp_path, monkeypatch, calls)
        ev.run("frozen")
        ev.run("frozen")
        assert calls["n"] == 1

    def test_changed_checkpoint_bytes_re_evaluate(self, tmp_path, monkeypatch) -> None:
        calls = {"n": 0}
        cfg = self._install(tmp_path, monkeypatch, calls)
        ev.run("frozen")
        cfg["_winner"].write_bytes(b"winner-v2")
        clear_identity_cache()
        ev.run("frozen")
        assert calls["n"] == 2

    def test_changed_seed_re_evaluates(self, tmp_path, monkeypatch) -> None:
        calls = {"n": 0}
        cfg = self._install(tmp_path, monkeypatch, calls)
        ev.run("frozen")
        cfg["seed"] = 8
        ev.run("frozen")
        assert calls["n"] == 2

    def test_legacy_done_row_is_not_a_completion(self, tmp_path, monkeypatch) -> None:
        calls = {"n": 0}
        cfg = self._install(tmp_path, monkeypatch, calls)
        tables = Path(cfg["paths"]["results"]) / "tables"
        tables.mkdir(parents=True)
        pd.DataFrame(
            [("frozen", "bpr", "none"), ("finetuned", "bpr", "none")],
            columns=["target", "model_name", "embedding_name"],
        ).to_csv(tables / f"{DATASET}_evaluation_done.csv", index=False)

        ev.run("frozen")

        assert calls["n"] == 1
        done = pd.read_csv(tables / f"{DATASET}_evaluation_done.csv", dtype=str)
        assert len(done) == 2 and done["checkpoint_digest"].notna().all()
