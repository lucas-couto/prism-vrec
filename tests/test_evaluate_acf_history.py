"""Tests that evaluation feeds ACF the train-only user history (no leakage)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import src.steps.evaluate as ev
from src.steps.evaluate import _evaluate_cell, load_data


def _write_dataset(base: Path) -> None:
    base.mkdir(parents=True)
    pd.DataFrame({"user_idx": [0, 0, 1], "item_idx": [1, 2, 3]}).to_csv(base / "train.csv", index=False)
    pd.DataFrame({"user_idx": [0, 1], "item_idx": [5, 6]}).to_csv(base / "val.csv", index=False)
    pd.DataFrame({"user_idx": [0, 1], "item_idx": [8, 9]}).to_csv(base / "test.csv", index=False)
    (base / "user2idx.json").write_text('{"0": 0, "1": 1}')
    (base / "item2idx.json").write_text('{"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7, "i": 8, "j": 9}')


def test_load_data_returns_train_only_history_separate_from_seen(tmp_path: Path) -> None:
    _write_dataset(tmp_path / "amazon_x")

    n_users, n_items, seen, test, train_only = load_data(str(tmp_path), "amazon_x")

    # train-only excludes val items; seen (masking) includes them.
    assert train_only[0] == {1, 2}
    assert seen[0] == {1, 2, 5}
    assert 5 not in train_only[0]
    assert n_users == 2
    assert n_items == 10


def test_evaluate_cell_passes_train_only_history_to_history_model(tmp_path, monkeypatch) -> None:
    captured: dict = {}

    class _FakeModel:
        wants_history = True

        def __init__(self, *, train_interactions=None, **kwargs) -> None:
            captured["train_interactions"] = train_interactions

        def load_state_dict(self, state) -> None: ...

        def to(self, device):  # noqa: ANN001
            return self

    class _FakeSpec:
        cls = _FakeModel
        requires_visual = False

    class _FakeEvaluator:
        def evaluate_per_user(self, model, device):  # noqa: ANN001
            return pd.DataFrame({"user_id": [0], "ndcg@10": [0.5]})

    monkeypatch.setattr(ev, "get_recommender_spec", lambda name: _FakeSpec())
    monkeypatch.setattr(ev.torch, "load", lambda *a, **k: {"model_state": {}, "hyperparams": {}})

    train_only = {0: {1, 2}, 1: {3}}
    result = _evaluate_cell(
        {"model_name": "acf", "embedding_name": "none", "path": "x.pt"},
        "amazon_x",
        n_users=2,
        n_items=10,
        evaluator=_FakeEvaluator(),
        embeddings_dir="e",
        device="cpu",
        train_interactions=train_only,
    )

    assert result is not None
    assert captured["train_interactions"] is train_only
