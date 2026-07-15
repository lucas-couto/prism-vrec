"""Per-user persistence, derivation and paired loader (Task F)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.evaluation.derive_metrics import per_user_metrics
from src.evaluation.paired_loader import UserSetMismatchError, load_paired
from src.evaluation.persistence import CellMetadata, read_cell_artifact, write_cell_artifact
from src.evaluation.protocol import Evaluator


class _ScoreModel:
    """Deterministic model with a preset (n_users, n_items) score matrix."""

    def __init__(self, scores: np.ndarray) -> None:
        self._s = scores.astype(np.float32)

    def eval(self) -> None:
        pass

    def predict(self, user_id, item_ids):
        idx = item_ids.cpu().numpy() if isinstance(item_ids, torch.Tensor) else np.asarray(item_ids)
        return torch.tensor(self._s[user_id][idx])

    def predict_batch(self, user_ids, item_ids):
        return torch.tensor(self._s[np.ix_(user_ids.cpu().numpy(), item_ids.cpu().numpy())])


_NU, _NI = 40, 60
_KS = [5, 10, 20]


def _evaluator(seed: int = 7) -> Evaluator:
    train = {u: {u % 5} for u in range(_NU)}
    test = {u: {10 + (u % 40)} for u in range(_NU)}
    return Evaluator(train, test, _NI, k_values=_KS, tiebreak_seed=seed)


def _distinct_scores() -> np.ndarray:
    return np.random.default_rng(3).standard_normal((_NU, _NI))


def _tied_scores() -> np.ndarray:
    # Integer-quantised -> many exact ties, exercising the tie-break.
    return np.random.default_rng(1).integers(0, 4, size=(_NU, _NI)).astype(float)


class TestDeriveEquivalence:
    @pytest.mark.parametrize(
        "scores_fn", [_distinct_scores, _tied_scores], ids=["distinct", "tied"]
    )
    def test_derived_metrics_equal_online(self, scores_fn) -> None:
        from src.evaluation.derive_metrics import metrics_frame

        evaluator = _evaluator()
        model = _ScoreModel(scores_fn())

        online = evaluator.evaluate_per_user(model, device="cpu").sort_values("user_id")
        records = evaluator.per_user_records(model, device="cpu")
        derived = metrics_frame(records, _KS).sort_values("user_id")

        shared = [c for c in online.columns if c in derived.columns and c != "user_id"]
        assert len(shared) == len(_KS) * 5  # precision/recall/f1/map/ndcg per k
        for col in shared:
            np.testing.assert_allclose(online[col].to_numpy(), derived[col].to_numpy(), atol=1e-12)


class TestIdentities:
    def test_closed_form_at_k(self) -> None:
        ranks = np.array([1, 3, 7, 10, 25])
        m = per_user_metrics(ranks, k=10)
        assert list(m["hitrate"]) == [1, 1, 1, 1, 0]
        np.testing.assert_allclose(m["precision"], m["hitrate"] / 10)
        np.testing.assert_allclose(m["map"], m["mrr"])
        # ndcg for r=3: 1/log2(4) = 0.5
        assert m["ndcg"][1] == pytest.approx(0.5)

    def test_rejects_zero_indexed_ranks(self) -> None:
        with pytest.raises(ValueError, match="1-indexed"):
            per_user_metrics(np.array([0, 1]), k=5)


class TestRoundTrip:
    def test_write_read_records_identical(self, tmp_path) -> None:
        evaluator = _evaluator()
        records = evaluator.per_user_records(_ScoreModel(_distinct_scores()), device="cpu")
        meta = CellMetadata(
            dataset="synthetic",
            visual_config="resnet50",
            recommender="vbpr",
            seed=7,
            d=64,
            split="test",
            n_users=_NU,
            n_items=_NI,
        )

        path = write_cell_artifact(records, meta, tmp_path)
        read_meta, read_df = read_cell_artifact(path)

        assert read_meta["dataset"] == "synthetic" and read_meta["seed"] == 7
        assert list(read_df["rank"]) == list(records["rank"])
        assert read_df["top_items"].iloc[0] == records["top_items"].iloc[0]
        assert len(read_df["top_items"].iloc[0]) == 20


def _write_cell(tmp_path, evaluator, scores, recommender, visual_config, seed=7) -> None:
    records = evaluator.per_user_records(_ScoreModel(scores), device="cpu")
    meta = CellMetadata(
        dataset="synthetic",
        visual_config=visual_config,
        recommender=recommender,
        seed=seed,
        d=64,
        split="test",
        n_users=_NU,
        n_items=_NI,
    )
    write_cell_artifact(records, meta, tmp_path)


class TestPairedLoader:
    def test_builds_users_by_systems_matrix(self, tmp_path) -> None:
        ev = _evaluator()
        _write_cell(tmp_path, ev, _distinct_scores(), "bpr", "none")
        _write_cell(tmp_path, ev, _tied_scores(), "vbpr", "resnet50")

        matrix = load_paired(tmp_path, "synthetic", seed=7, metric="ndcg", k=10)

        assert matrix.shape == (_NU, 2)
        assert set(matrix.columns) == {"bpr__none", "vbpr__resnet50"}
        assert matrix.index.tolist() == sorted(range(_NU))

    def test_user_set_mismatch_raises(self, tmp_path) -> None:
        ev = _evaluator()
        _write_cell(tmp_path, ev, _distinct_scores(), "bpr", "none")

        # A cell over a DIFFERENT user set must not be silently intersected.
        train = {u: {u % 5} for u in range(_NU)}
        test = {u: {10 + (u % 40)} for u in range(10)}  # only 10 users
        ev2 = Evaluator(train, test, _NI, k_values=_KS, tiebreak_seed=7)
        _write_cell(tmp_path, ev2, _distinct_scores()[:10], "vbpr", "resnet50")

        with pytest.raises(UserSetMismatchError, match="does not match"):
            load_paired(tmp_path, "synthetic", seed=7, metric="ndcg", k=10)
