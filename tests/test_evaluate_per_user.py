"""Tests for per-user evaluation output and battery routing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import src.steps.evaluate as ev
from src.steps.evaluate import (
    _append_cell,
    _load_done,
    _record_done,
    _route_targets,
)


class TestRouteTargets:
    def test_frozen_embedding_goes_to_frozen(self) -> None:
        assert _route_targets("deepstyle", "resnet50_D128") == ["frozen"]

    def test_finetuned_embedding_goes_to_finetuned(self) -> None:
        assert _route_targets("deepstyle", "resnet50_finetuned_D128") == ["finetuned"]

    def test_convnext_frozen_goes_to_frozen(self) -> None:
        assert _route_targets("vbpr", "convnext_base_D128") == ["frozen"]

    def test_hybrid_finetuned_goes_to_finetuned(self) -> None:
        assert _route_targets("avbpr", "hybrid_mean_finetuned_D128") == ["finetuned"]

    def test_bpr_none_goes_to_both(self) -> None:
        assert _route_targets("bpr", "none") == ["frozen", "finetuned"]


class TestSidecar:
    def test_load_done_missing_returns_empty(self, tmp_path: Path) -> None:
        assert _load_done(tmp_path / "absent.csv") == set()

    def test_record_then_load_roundtrip(self, tmp_path: Path) -> None:
        p = tmp_path / "done.csv"

        _record_done(p, [("frozen", "vbpr", "resnet50_D128")])
        _record_done(p, [("finetuned", "bpr", "none")])

        assert _load_done(p) == {
            ("frozen", "vbpr", "resnet50_D128"),
            ("finetuned", "bpr", "none"),
        }


class TestAppendCell:
    def test_creates_with_header_then_appends_without(self, tmp_path: Path) -> None:
        p = tmp_path / "amazon_x_evaluation_frozen.csv"

        _append_cell(pd.DataFrame({"user_id": [1, 2], "ndcg@10": [0.1, 0.2]}), p)
        _append_cell(pd.DataFrame({"user_id": [3], "ndcg@10": [0.3]}), p)

        out = pd.read_csv(p)
        assert list(out["user_id"]) == [1, 2, 3]
        assert list(out.columns) == ["user_id", "ndcg@10"]


class TestRunResume:
    def test_run_routes_per_user_and_resumes(self, tmp_path, monkeypatch) -> None:
        results = tmp_path / "results" / "tables"
        monkeypatch.chdir(tmp_path)

        monkeypatch.setattr(
            ev,
            "load_config",
            lambda: {
                "device": "cpu",
                "paths": {"data_processed": "p", "embeddings": "e"},
                "k_values": [10],
                "datasets": ["amazon_x"],
            },
        )
        monkeypatch.setattr(ev, "resolve_device", lambda d: "cpu")
        monkeypatch.setattr(ev, "load_data", lambda p, d: (2, 5, {}, {}))

        class _Eval:
            def __init__(self, *a, **k) -> None: ...

        monkeypatch.setattr(ev, "Evaluator", _Eval)
        monkeypatch.setattr(
            ev,
            "find_best_models",
            lambda d, **kw: [
                {
                    "model_name": "vbpr",
                    "embedding_name": "resnet50_finetuned_D128",
                    "path": "x.pt",
                }
            ],
        )

        calls = {"n": 0}

        def _fake_cell(*a, **k):
            calls["n"] += 1
            return pd.DataFrame({"user_id": [1, 2], "ndcg@10": [0.5, 0.6]})

        monkeypatch.setattr(ev, "_evaluate_cell", _fake_cell)

        ev.run("frozen")
        ev.run("frozen")

        ft = pd.read_csv(results / "amazon_x_evaluation_finetuned.csv")
        assert not (results / "amazon_x_evaluation_frozen.csv").exists()
        assert list(ft["user_id"]) == [1, 2]
        assert set(ft["model_name"]) == {"vbpr"}
        assert calls["n"] == 1
