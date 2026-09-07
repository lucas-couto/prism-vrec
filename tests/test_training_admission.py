"""Host admission counts payloads, model state, workspace and workers (M05, Q10/Q11).

The budget is resolved conservatively (explicit config, cgroup, host,
4 GiB fallback — never unlimited); a job is charged its feature payload
(source bytes, not sidecar size), model + optimizer state, interaction
dicts and the host ranking workspace; the pool is sized so the aggregate
commitment fits; a job whose declared minimum exceeds the budget is
refused once, never launched.  The lazy feature path is opt-in and is
shown numerically identical to the dense one on a real training run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

import src.steps.train as train_mod
from src.recommenders.vbpr import VBPR
from src.steps.train import (
    TrainingJobsFailedError,
    estimate_job_bytes,
    feature_payload_bytes,
    lazy_features_for,
    plan_training_admission,
)
from src.utils import memory as memory_mod
from src.utils.checkpoint import CheckpointManager
from src.utils.memory import (
    AdmissionError,
    HostBudget,
    admit_workers,
    resolve_host_budget,
)
from src.utils.parallel import TrainingJob, TrainingOrchestrator
from src.utils.resources import resolve_resources
from src.utils.training import train_single_run

GB = 1024**3
N_USERS, N_ITEMS, DIM = 6, 8, 3


def _write_dataset(root: Path) -> tuple[str, str]:
    base = root / "processed" / "synthetic"
    base.mkdir(parents=True, exist_ok=True)
    train = [(u, (u + i) % N_ITEMS) for u in range(N_USERS) for i in range(3)]
    val = [(u, (u + 3) % N_ITEMS) for u in range(N_USERS)]
    for name, rows in (("train", train), ("val", val)):
        pd.DataFrame(rows, columns=["user_idx", "item_idx"]).to_csv(
            base / f"{name}.csv", index=False
        )
    (base / "user2idx.json").write_text(json.dumps({str(u): u for u in range(N_USERS)}))
    (base / "item2idx.json").write_text(json.dumps({str(i): i for i in range(N_ITEMS)}))
    emb = root / "embeddings" / "synthetic"
    emb.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    np.save(emb / "resnet50.npy", rng.standard_normal((N_ITEMS, DIM)).astype(np.float32))
    np.save(emb / "vit_b16.npy", rng.standard_normal((N_ITEMS, DIM + 1)).astype(np.float32))
    (emb / "hybrid_mean_learned_D2.json").write_text(
        json.dumps(
            {
                "strategy": "mean",
                "online": True,
                "alignment": "learned",
                "dim": 2,
                "components": ["resnet50.npy", "vit_b16.npy"],
                "normalize": True,
                "fusion_kwargs": {},
            }
        )
    )
    return str(root / "processed"), str(root / "embeddings")


def _job(processed: str, emb_path: str | None, **hp) -> TrainingJob:
    return TrainingJob(
        dataset_name="synthetic",
        model_name="vbpr" if emb_path else "bpr",
        embedding_name=Path(emb_path).stem if emb_path else "none",
        hyperparams={"learning_rate": 0.05, "latent_dim": 4, "l2_reg": 0.0, **hp},
        n_users=N_USERS,
        n_items=N_ITEMS,
        embeddings_path=emb_path,
        processed_dir=processed,
        device="cpu",
    )


class TestResourcesBlock:
    """The block itself is covered by tests/test_resources_config.py; here only
    what the admission path reads from it."""

    def test_defaults_are_conservative_and_dense(self) -> None:
        resources = resolve_resources({})
        assert resources.host.budget_bytes is None
        assert resources.host.headroom_bytes == 4 * GB
        assert resources.features.residency == "dense"

    def test_explicit_budget_wins_and_is_reported(self) -> None:
        budget = resolve_host_budget({"resources": {"host": {"budget_bytes": 3 * GB}}})
        assert (budget.limit_bytes, budget.source) == (3 * GB, "config")


class TestHostBudget:
    def test_cgroup_limit_is_reported_with_its_source(self, monkeypatch) -> None:
        monkeypatch.setattr(
            memory_mod, "_read_int_file", lambda p: 6 * GB if "memory.max" in str(p) else None
        )
        budget = resolve_host_budget({})
        assert (budget.limit_bytes, budget.source) == (6 * GB, "cgroup_v2")

    def test_unknown_budget_is_the_conservative_fallback_never_unlimited(self, monkeypatch) -> None:
        monkeypatch.setattr(memory_mod, "_read_int_file", lambda p: None)

        def _boom(name):
            raise OSError("no sysconf")

        monkeypatch.setattr(memory_mod.os, "sysconf", _boom)
        budget = resolve_host_budget({})
        assert (budget.limit_bytes, budget.source) == (4 * GB, "fallback")


class TestJobLedger:
    def test_sidecar_payload_is_its_sources_not_its_json_size(self, tmp_path) -> None:
        _, emb = _write_dataset(tmp_path)
        sidecar = Path(emb) / "synthetic" / "hybrid_mean_learned_D2.json"

        payload, width = feature_payload_bytes(sidecar)

        assert payload == N_ITEMS * (DIM + DIM + 1) * 4
        assert width == DIM + DIM + 1
        assert payload > sidecar.stat().st_size

    def test_estimate_counts_features_model_state_interactions_and_ranking(self, tmp_path) -> None:
        processed, emb = _write_dataset(tmp_path)
        job = _job(processed, str(Path(emb) / "synthetic" / "resnet50.npy"))

        estimate = estimate_job_bytes(job, processed, {}, lazy=False)

        assert estimate.feature_bytes == N_ITEMS * DIM * 4
        assert estimate.model_bytes == ((N_USERS + N_ITEMS) * 4 + DIM * 4) * 16
        assert estimate.interactions_bytes > 0 and estimate.ranking_bytes == 40 * N_ITEMS
        assert estimate.total == (
            estimate.base_bytes
            + estimate.feature_bytes
            + estimate.model_bytes
            + estimate.interactions_bytes
            + estimate.ranking_bytes
        )

    def test_lazy_residency_charges_a_bounded_block_not_the_catalogue(self, tmp_path) -> None:
        processed, emb = _write_dataset(tmp_path)
        big = Path(emb) / "synthetic" / "big.npy"
        np.save(big, np.zeros((50_000, 16), dtype=np.float32))
        job = _job(processed, str(big))

        dense = estimate_job_bytes(job, processed, {}, lazy=False)
        lazy = estimate_job_bytes(job, processed, {}, lazy=True)

        assert dense.feature_bytes == 50_000 * 16 * 4
        assert lazy.feature_bytes == 8192 * 16 * 4 * 2


class TestAdmitWorkers:
    def _budget(self, gb: float) -> HostBudget:
        return HostBudget(int(gb * GB), "config", None)

    def test_aggregate_commitment_caps_the_pool(self) -> None:
        plan = admit_workers(3 * GB, hard_cap=8, budget=self._budget(14), headroom_bytes=4 * GB)
        assert plan.admitted and plan.n_workers == 3

    def test_declared_minimum_above_budget_is_not_admitted(self) -> None:
        plan = admit_workers(5 * GB, hard_cap=8, budget=self._budget(8), headroom_bytes=4 * GB)
        assert not plan.admitted and plan.n_workers == 0
        assert "exceeds the usable budget" in plan.reason

    def test_unknown_footprint_admits_one_worker_conservatively(self) -> None:
        plan = admit_workers(0, hard_cap=8, budget=self._budget(64), headroom_bytes=4 * GB)
        assert plan.admitted and plan.n_workers == 1

    def test_max_workers_zero_admits_nothing(self) -> None:
        plan = admit_workers(
            GB, hard_cap=8, budget=self._budget(64), headroom_bytes=4 * GB, max_workers=0
        )
        assert not plan.admitted and plan.n_workers == 0

    def test_orchestrator_enforces_the_plan(self, tmp_path) -> None:
        plan = admit_workers(3 * GB, hard_cap=8, budget=self._budget(14), headroom_bytes=4 * GB)
        pool = TrainingOrchestrator(
            n_workers=8, device="cuda", log_dir=str(tmp_path), admission=plan
        )
        assert pool.n_workers == 3
        refused = admit_workers(50 * GB, hard_cap=1, budget=self._budget(8), headroom_bytes=4 * GB)
        with pytest.raises(AdmissionError):
            TrainingOrchestrator(
                n_workers=1, device="cpu", log_dir=str(tmp_path), admission=refused
            )


class TestPlanTrainingAdmission:
    def test_job_over_budget_is_refused_once_and_never_launched(
        self, tmp_path, monkeypatch
    ) -> None:
        processed, emb = _write_dataset(tmp_path)
        huge = Path(emb) / "synthetic" / "huge.npy"
        np.save(huge, np.zeros((N_ITEMS, 4), dtype=np.float32))
        config = {"resources": {"host": {"budget_bytes": 2 * GB, "headroom_bytes": 0}}}
        small = _job(processed, None)
        big = _job(processed, str(huge))
        monkeypatch.setattr(
            train_mod,
            "estimate_job_bytes",
            lambda job, p, c, lazy: train_mod.JobMemoryEstimate(
                feature_bytes=3 * GB if job is big else 0,
                model_bytes=0,
                interactions_bytes=0,
                ranking_bytes=0,
                base_bytes=GB,
            ),
        )

        admitted, refused, plan = plan_training_admission(
            [small, big], processed, config, requested_workers=1
        )

        assert admitted == [small]
        assert [job.job_id for job, _ in refused] == [big.job_id]
        assert "exceeds the usable host budget" in refused[0][1]
        assert plan.admitted and plan.n_workers == 1

    def test_grid_step_records_a_failed_outcome_without_launching(
        self, tmp_path, monkeypatch
    ) -> None:
        processed, emb = _write_dataset(tmp_path)
        big = _job(processed, str(Path(emb) / "synthetic" / "resnet50.npy"))
        config = {
            "device": "cpu",
            "seed": 1,
            "paths": {"data_processed": processed, "embeddings": emb, "results": str(tmp_path)},
            "recommenders_enabled": ["vbpr"],
            "resources": {"host": {"budget_bytes": 1, "headroom_bytes": 0}},
        }
        launched: list[int] = []

        class _Explode(TrainingOrchestrator):
            def __init__(self, **kwargs) -> None:
                launched.append(1)
                raise AssertionError("a refused job must not reach the pool")

        monkeypatch.setattr(train_mod, "build_job_list", lambda *a, **k: [big])
        monkeypatch.setattr(train_mod, "_cell_counts", lambda *a, **k: {"vbpr": 1})
        monkeypatch.setattr(train_mod, "_resolve_model_names", lambda config: ["vbpr"])
        monkeypatch.setattr(train_mod, "get_hyperparam_grid", lambda name, config: [{}])
        monkeypatch.setattr(train_mod, "TrainingOrchestrator", _Explode)

        with pytest.raises(TrainingJobsFailedError) as err:
            train_mod._run_grid("frozen", config, workers=1, sequential=True)

        assert launched == []
        assert err.value.failures[0]["status"] == "error"
        assert "not launched" in err.value.failures[0]["error"]

    def test_auto_residency_switches_a_large_payload_to_lazy(self, tmp_path) -> None:
        processed, emb = _write_dataset(tmp_path)
        feature = str(Path(emb) / "synthetic" / "resnet50.npy")
        tiny_budget = {
            "resources": {
                "host": {"budget_bytes": 2 * GB + 100, "headroom_bytes": 2 * GB},
                "features": {"residency": "auto"},
            }
        }
        lazy = {"resources": {"features": {"residency": "lazy"}}}
        auto = {"resources": {"features": {"residency": "auto"}}}
        assert lazy_features_for(tiny_budget, feature) is True
        assert lazy_features_for(lazy, feature) is True
        assert lazy_features_for({}, feature) is False
        assert lazy_features_for(auto, feature) is False


class TestLazyPathIsNumericallyIdentical:
    def _train(self, tmp_path, processed, emb_path, *, lazy: bool) -> tuple[float, dict]:
        from src.fusions import load_embedding

        train = {u: {(u + i) % N_ITEMS for i in range(3)} for u in range(N_USERS)}
        val = {u: {(u + 3) % N_ITEMS} for u in range(N_USERS)}
        config = {
            "seed": 3,
            "paths": {"results": str(tmp_path / ("lazy" if lazy else "dense"))},
            "common": {
                "epochs": 3,
                "batch_size": 4,
                "early_stopping_patience": 10,
                "early_stopping_metric": "ndcg@10",
                "eval_every_epochs": 1,
            },
        }
        value = train_single_run(
            model_cls=VBPR,
            model_name="vbpr",
            n_users=N_USERS,
            n_items=N_ITEMS,
            visual_embeddings=load_embedding(emb_path, lazy=lazy),
            train_interactions=train,
            selection_interactions=val,
            hyperparams={"learning_rate": 0.05, "latent_dim": 4, "visual_dim": 2, "l2_reg": 0.0},
            config=config,
            checkpoint_mgr=CheckpointManager(str(tmp_path / ("ckpt_lazy" if lazy else "ckpt"))),
            dataset_name="synthetic",
            embedding_name="resnet50",
            device="cpu",
        )
        best = torch.load(
            Path(config["paths"]["results"]) / "models" / "synthetic" / "vbpr_resnet50_best.pt",
            map_location="cpu",
            weights_only=False,
        )
        return value, best["model_state"]

    @pytest.mark.parametrize("artifact", ["resnet50.npy", "hybrid_mean_learned_D2.json"])
    def test_dense_and_lazy_training_agree_bit_for_bit(self, tmp_path, artifact) -> None:
        processed, emb = _write_dataset(tmp_path)
        path = str(Path(emb) / "synthetic" / artifact)

        dense_value, dense_state = self._train(tmp_path, processed, path, lazy=False)
        lazy_value, lazy_state = self._train(tmp_path, processed, path, lazy=True)

        assert dense_value == lazy_value
        assert dense_state.keys() == lazy_state.keys()
        for key in dense_state:
            assert torch.equal(dense_state[key], lazy_state[key]), key
