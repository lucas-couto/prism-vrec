"""S04 — opt-in bounded training diagnostics.

The hook is off by default, inert when off, and when on it records a
detached probe summary at init, at fixed optimizer steps and at every
validation, without changing the training trajectory.  VNPR exposes its
pre-ReLU branches so the "inactive ReLU" hypothesis can be measured
directly; a deliberately dead model (large negative dense bias) must show
up as zero active fraction, fully tied scores, zero gradients and a
legitimate zero metric — all as separate fields.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from src.recommenders.vnpr import VNPR
from src.utils.checkpoint import CheckpointManager
from src.utils.diagnostics import (
    DiagnosticsConfig,
    TrainingDiagnostics,
    build_probe,
    expected_random_metrics,
)
from src.utils.training import train_single_run

N_USERS, N_ITEMS, K, DV = 12, 30, 4, 6


def _interactions(seed: int = 0) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    train: dict[int, set[int]] = {}
    val: dict[int, set[int]] = {}
    for u in range(N_USERS):
        items = rng.choice(N_ITEMS, size=5, replace=False)
        train[u] = {int(i) for i in items[:4]}
        val[u] = {int(items[4])}
    return train, val


def _visual(seed: int = 0, scale: float = 1.0) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal((N_ITEMS, DV)) * scale).astype("float32")


def _config(tmp_path: Path, *, enabled: bool, epochs: int = 3) -> dict:
    return {
        "seed": 3,
        "paths": {"results": str(tmp_path / "results")},
        "common": {
            "epochs": epochs,
            "batch_size": 8,
            "early_stopping_patience": 10,
            "early_stopping_metric": "ndcg@10",
            "eval_every_epochs": 1,
        },
        "diagnostics": {
            "enabled": enabled,
            "probe_users": 5,
            "probe_items": 9,
            "probe_pairs": 6,
            "steps": [1, 3],
            "output_dir": str(tmp_path / "diag"),
        },
    }


def _run(tmp_path: Path, config: dict, visual: np.ndarray | None = None) -> tuple[float, VNPR]:
    train, val = _interactions()
    captured: dict[str, VNPR] = {}

    class Spy(VNPR):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            captured["model"] = self

    metric = train_single_run(
        model_cls=Spy,
        model_name="vnpr",
        n_users=N_USERS,
        n_items=N_ITEMS,
        visual_embeddings=_visual() if visual is None else visual,
        train_interactions=train,
        selection_interactions=val,
        hyperparams={"learning_rate": 0.01, "latent_dim": K, "l2_reg": 1e-4},
        config=config,
        checkpoint_mgr=CheckpointManager(str(tmp_path / "ckpt")),
        dataset_name="ds",
        embedding_name="native",
        device="cpu",
    )
    return metric, captured["model"]


class TestConfig:
    def test_disabled_by_default(self) -> None:
        assert DiagnosticsConfig.from_config({}) is None
        assert DiagnosticsConfig.from_config({"diagnostics": {"enabled": False}}) is None

    def test_enabled_block_is_parsed(self) -> None:
        cfg = DiagnosticsConfig.from_config({"diagnostics": {"enabled": True, "steps": [2, 5]}})

        assert cfg is not None and cfg.steps == (2, 5) and cfg.probe_users == 64

    def test_probe_is_seeded_bounded_and_negatives_avoid_train(self) -> None:
        train, _ = _interactions()
        cfg = DiagnosticsConfig(enabled=True, probe_users=4, probe_items=7, probe_pairs=5)

        a = build_probe(train, N_USERS, N_ITEMS, cfg)
        b = build_probe(train, N_USERS, N_ITEMS, cfg)

        assert a.users.shape == (4,) and a.items.shape == (7,) and a.pairs.shape == (5, 2)
        assert torch.equal(a.users, b.users) and torch.equal(a.neg, b.neg)
        assert all(int(n) not in train[int(u)] for u, n in zip(a.users, a.neg, strict=True))
        assert all(int(p) in train[int(u)] for u, p in zip(a.users, a.pos, strict=True))

    def test_random_baseline_is_the_loo_expectation(self) -> None:
        base = expected_random_metrics(100)

        assert base["hit@10"] == pytest.approx(0.1)
        assert base["ndcg@10"] == pytest.approx(
            sum(1 / math.log2(r + 1) for r in range(1, 11)) / 100
        )


class TestVNPRBranches:
    def test_preactivation_relu_equals_forward(self) -> None:
        torch.manual_seed(0)
        model = VNPR(N_USERS, N_ITEMS, _visual(), {"latent_dim": K}).eval()
        users, pos, neg = torch.arange(6), torch.arange(6, 12), torch.arange(12, 18)

        with torch.no_grad():
            pre = model.diagnostic_branches(users, pos, neg)
            r_pos, r_neg = model(users, pos, neg)

        torch.testing.assert_close(torch.relu(pre["pos"]), r_pos)
        torch.testing.assert_close(torch.relu(pre["neg"]), r_neg)
        assert bool((pre["pos"] < 0).any()) or bool((pre["neg"] < 0).any())


class TestHook:
    def test_off_by_default_writes_nothing_and_leaves_training_unchanged(
        self, tmp_path: Path
    ) -> None:
        metric_off, model_off = _run(tmp_path / "off", _config(tmp_path / "off", enabled=False))
        metric_on, model_on = _run(tmp_path / "on", _config(tmp_path / "on", enabled=True))

        assert not (tmp_path / "off" / "diag").exists()
        assert metric_off == metric_on
        for (name, p_off), (_, p_on) in zip(
            model_off.state_dict().items(), model_on.state_dict().items(), strict=True
        ):
            torch.testing.assert_close(p_off, p_on, rtol=0, atol=0, msg=name)

    def test_records_init_steps_and_validations_as_json(self, tmp_path: Path) -> None:
        _run(tmp_path, _config(tmp_path, enabled=True, epochs=2))

        [path] = list((tmp_path / "diag").glob("*.json"))
        payload = json.loads(path.read_text(encoding="utf-8"))
        phases = [r["phase"] for r in payload["records"]]
        assert phases == ["init", "step", "step", "validation", "validation"]
        assert [r["step"] for r in payload["records"][:3]] == [0, 1, 3]
        assert payload["identity"]["model"] == "vnpr"
        assert payload["identity"]["feature"]["shape"] == [N_ITEMS, DV]
        assert payload["random_baseline"]["hit@10"] == pytest.approx(10 / N_ITEMS)

        init = payload["records"][0]
        assert len(init["features"]["source_norm_quantiles"]) == 1
        assert len(init["features"]["fused_norm_quantiles"]) == 5
        assert set(init["train_branches"]) == {"pos", "neg"} == set(init["eval_branches"])
        assert 0.0 <= init["train_branches"]["pos"]["active_fraction"] <= 1.0
        assert {"bpr", "l2", "total", "finite"} <= set(init["loss"])
        assert init["loss"]["finite"] and init["loss"]["l2"] >= 0.0
        assert "dense.weight" in init["gradient_norms"]
        assert init["parameters"]["item_embedding.weight"]["delta_from_init"] == 0.0
        assert init["optimizer"]["steps_applied"] == 0

        step = payload["records"][2]
        assert step["optimizer"] == {
            "steps_attempted": 3,
            "steps_applied": 3,
            "steps_skipped": 0,
            "scaler_enabled": False,
            "scaler_scale": None,
        }
        assert step["parameters"]["dense.weight"]["delta_from_init"] > 0.0
        assert 0.0 < step["scores"]["unique_fraction"] <= 1.0

        val = payload["records"][3]
        assert {"zero_metric", "tie_frequency", "checkpoint_exists", "metric_value"} <= set(val)
        assert val["checkpoint_exists"] is True and val["has_valid_observation"] is True
        assert val["metric_present"] and val["metric_finite"]

    def test_dead_relu_model_is_measured_not_hidden(self, tmp_path: Path) -> None:
        """A huge negative dense bias kills both branches: every probe is flat."""
        torch.manual_seed(0)
        train, _ = _interactions()
        model = VNPR(N_USERS, N_ITEMS, _visual(), {"latent_dim": K, "l2_reg": 0.0})
        with torch.no_grad():
            model.dense.bias.fill_(-1e3)
        cfg = DiagnosticsConfig(enabled=True, probe_users=6, probe_items=10, probe_pairs=8)
        diag = TrainingDiagnostics(
            model,
            config=cfg,
            probe=build_probe(train, N_USERS, N_ITEMS, cfg),
            identity={"run_id": "dead"},
            output_path=tmp_path / "dead.json",
            n_users=N_USERS,
            n_items=N_ITEMS,
        )

        record = diag.record_init()
        val = diag.record_validation(
            epoch=0,
            step=0,
            metrics={"ndcg@10": 0.0, "recall@10": 0.0},
            es_metric="ndcg@10",
            checkpoint_exists=True,
            has_valid_observation=True,
        )

        assert record["train_branches"]["pos"]["active_fraction"] == 0.0
        assert record["train_branches"]["pos"]["max"] < 0.0
        assert record["scores"]["unique_fraction"] == pytest.approx(1 / 60)
        assert record["scores"]["all_tied_user_fraction"] == 1.0
        assert record["scores"]["pairwise_zero_fraction"] == 1.0
        assert record["gradient_norm_total"] == 0.0
        assert record["loss"]["bpr"] == pytest.approx(math.log(2.0))
        assert val["zero_metric"] is True and val["tie_frequency"] == 1.0
        assert val["checkpoint_exists"] is True
        assert diag.write().exists()

    def test_measurement_does_not_disturb_rng_parameters_or_gradients(self, tmp_path: Path) -> None:
        torch.manual_seed(1)
        train, _ = _interactions()
        model = VNPR(N_USERS, N_ITEMS, _visual(), {"latent_dim": K, "dropout": 0.5})
        cfg = DiagnosticsConfig(enabled=True, probe_users=4, probe_items=8, probe_pairs=4)
        diag = TrainingDiagnostics(
            model,
            config=cfg,
            probe=build_probe(train, N_USERS, N_ITEMS, cfg),
            identity={},
            output_path=tmp_path / "x.json",
            n_users=N_USERS,
            n_items=N_ITEMS,
        )
        before = {k: v.clone() for k, v in model.state_dict().items()}
        rng_before = torch.random.get_rng_state()
        model.train()

        diag.record_init()

        assert torch.equal(torch.random.get_rng_state(), rng_before)
        assert model.training is True
        assert all(p.grad is None for p in model.parameters())
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0, msg=name)
