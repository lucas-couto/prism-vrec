"""SDD M04: golden full-ranking fixture and bounded device residency.

The fixture (TEST-PLAN §"Golden ranking fixture") has several users,
cold items no user ever touched, train-masked items, exact score ties
that include the held-out item, a catalogue whose final item block is
shorter than requested, and a user whose masked candidates are fewer
than the largest K.  The reference records come from the dense path;
every degraded execution layout — user-batch reduction, an early-pass
item OOM at offset 0, a mid-pass item OOM — must reproduce them exactly.

It also pins the M04 residency changes: catalogue ids and train masks
reach the device only inside the guarded chunk, in bounded pieces, and
the per-user CPU ranking is admitted against a host budget (design A:
reference full vector with a checked budget; see
``HostMemoryBudgetError``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.evaluation.protocol import (
    HOST_RANKING_BYTES_PER_ITEM,
    Evaluator,
    HostMemoryBudgetError,
    host_ranking_bytes,
)

N_ITEMS = 37  # 1024-item request -> one short final block; limit 5 -> 7 full + 1 short
N_USERS = 9
COLD_ITEMS = list(range(30, 37))
K_VALUES = [5, 10, 20]


def _train() -> dict[int, set[int]]:
    rng = np.random.default_rng(5)
    train = {u: set(rng.choice(30, size=6, replace=False).tolist()) for u in range(N_USERS)}
    train[8] = set(range(0, 30)) - {4}  # only 8 candidates survive the mask (< K=10, 20)
    return train


def _test(train: dict[int, set[int]]) -> dict[int, set[int]]:
    held = {}
    for u in range(N_USERS):
        pool = [i for i in range(30) if i not in train[u]]
        held[u] = {pool[u % len(pool)]}
    held[8] = {4}
    return held


class _GoldenModel(torch.nn.Module):
    """Quantised scores: many exact ties, some including the held-out."""

    def __init__(self, held: dict[int, set[int]]) -> None:
        super().__init__()
        rng = np.random.default_rng(11)
        table = rng.integers(0, 4, size=(N_USERS, N_ITEMS)).astype(np.float32)
        for u, items in held.items():
            h = next(iter(items))
            table[u, h] = 2.0
            table[u, (h + 1) % N_ITEMS] = 2.0  # exact tie with the held-out
            table[u, COLD_ITEMS[u % len(COLD_ITEMS)]] = 3.0  # a cold item ranks high
        self.register_buffer("table", torch.from_numpy(table))

    def predict(self, user_id: int, item_ids: torch.Tensor) -> torch.Tensor:
        return self.table[user_id][item_ids]

    def predict_batch(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        return self.table[user_ids][:, item_ids]


class _UserOOM(_GoldenModel):
    """User batches above *threshold* overflow; item blocks never do."""

    def __init__(self, held, threshold: int) -> None:
        super().__init__(held)
        self.threshold = threshold
        self.batches: list[int] = []

    def predict_batch(self, user_ids, item_ids):
        self.batches.append(len(user_ids))
        if len(user_ids) > self.threshold:
            raise torch.cuda.OutOfMemoryError("simulated user-batch OOM")
        return super().predict_batch(user_ids, item_ids)


class _ItemOOM(_GoldenModel):
    """Batched path always overflows; item blocks overflow per *limit_at*.

    ``limit_at(offset)`` returns the largest block accepted at that
    item offset, so early (offset 0) and mid-pass failures can be
    scripted.  Every accepted ``(user, item)`` is recorded.
    """

    def __init__(self, held, limit_at) -> None:
        super().__init__(held)
        self.limit_at = limit_at
        self.scored: list[tuple[int, int]] = []
        self.attempts: list[tuple[int, int]] = []

    def predict_batch(self, user_ids, item_ids):
        raise torch.cuda.OutOfMemoryError("simulated batched OOM")

    def predict(self, user_id, item_ids):
        offset = int(item_ids[0])
        self.attempts.append((offset, len(item_ids)))
        if len(item_ids) > self.limit_at(offset):
            raise torch.cuda.OutOfMemoryError("simulated item-block OOM")
        self.scored.extend((user_id, int(i)) for i in item_ids)
        return super().predict(user_id, item_ids)


@pytest.fixture()
def held() -> dict[int, set[int]]:
    return _test(_train())


def _evaluator(**kwargs) -> Evaluator:
    train = _train()
    return Evaluator(
        train_interactions=train,
        test_interactions=_test(train),
        n_items=N_ITEMS,
        k_values=K_VALUES,
        tiebreak_seed=3,
        ranking_budget_bytes=0,
        **kwargs,
    )


def _records(model, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics, records = _evaluator(**kwargs).evaluate_with_records(model, device="cpu")
    return metrics.sort_values("user_id").reset_index(drop=True), records.sort_values(
        "user_id"
    ).reset_index(drop=True)


def _assert_same(actual, reference) -> None:
    for frame, ref in zip(actual, reference, strict=True):
        pd.testing.assert_frame_equal(frame, ref)


class TestGoldenFixtureProperties:
    def test_reference_visits_cold_items_masks_train_and_carries_ties(self, held) -> None:
        metrics, records = _records(_GoldenModel(held))
        train = _train()

        for _, row in records.iterrows():
            u = int(row["user_id"])
            candidates = row["top_items"][: min(20, int(row["n_candidates"]))]
            assert not set(candidates) & train[u], "train items must be masked"
            assert row["n_candidates"] == N_ITEMS - len(train[u])
            assert row["tie_block_size"] >= 2, "held-out sits in an exact tie block"
        assert any(set(r["top_items"]) & set(COLD_ITEMS) for _, r in records.iterrows())
        assert int(records.loc[records["user_id"] == 8, "n_candidates"].iloc[0]) == 8
        assert set(metrics.columns) >= {f"ndcg@{k}" for k in K_VALUES}

    def test_single_and_batched_dense_paths_agree(self, held) -> None:
        model = _GoldenModel(held)
        evaluator = _evaluator()
        items = torch.arange(N_ITEMS)

        with torch.no_grad():
            batched = evaluator._evaluate_batched(model, items, 512, device=torch.device("cpu"))
            single = evaluator._evaluate_single(model, items, device=torch.device("cpu"))

        assert (
            pd.DataFrame(batched)
            .sort_values("user_id")
            .reset_index(drop=True)
            .equals(pd.DataFrame(single).sort_values("user_id").reset_index(drop=True))
        )


class TestDegradedLayoutsReproduceTheGolden:
    def test_user_batch_reduction(self, held) -> None:
        reference = _records(_GoldenModel(held))
        model = _UserOOM(held, threshold=2)

        actual = _records(model)

        assert max(model.batches) > 2 and min(model.batches) <= 2
        _assert_same(actual, reference)

    def test_early_pass_item_oom_at_offset_zero(self, held) -> None:
        reference = _records(_GoldenModel(held))
        model = _ItemOOM(held, limit_at=lambda offset: 4)

        actual = _records(model)

        _assert_same(actual, reference)
        assert model.attempts[0] == (0, N_ITEMS)  # first request: the whole catalogue
        assert sorted(model.scored) == [(u, i) for u in range(N_USERS) for i in range(N_ITEMS)]

    def test_mid_pass_item_oom_retries_only_the_failed_offset(self, held) -> None:
        reference = _records(_GoldenModel(held))
        model = _ItemOOM(held, limit_at=lambda offset: 2 if offset >= 20 else 8)

        actual = _records(model)

        _assert_same(actual, reference)
        # Every user: each item offset scored exactly once, in order,
        # with a final block shorter than the block size in force.
        per_user = {u: [i for uu, i in model.scored if uu == u] for u in range(N_USERS)}
        assert all(items == list(range(N_ITEMS)) for items in per_user.values())
        retried = [(o, n) for o, n in model.attempts if o >= 20 and n > 2]
        assert retried, "the mid-pass offset must have been attempted too large first"
        assert (0, 8) in model.attempts or (0, 9) in model.attempts or (0, 4) in model.attempts

    def test_one_item_block_that_cannot_fit_fails_without_records(self, held) -> None:
        model = _ItemOOM(held, limit_at=lambda offset: 0)

        with pytest.raises(torch.cuda.OutOfMemoryError):
            _records(model)


class TestBoundedDeviceResidency:
    def test_catalogue_ids_and_train_masks_stay_on_the_host_until_the_guarded_chunk(
        self, held
    ) -> None:
        evaluator = _evaluator()
        model = _GoldenModel(held)

        assert evaluator._catalogue_ids_gpu is None and evaluator._train_idx_host == {}
        with torch.no_grad():
            evaluator._evaluate_batched(model, torch.arange(N_ITEMS), 4, device=torch.device("cpu"))

        assert evaluator._catalogue_ids_gpu is not None
        assert all(isinstance(v, np.ndarray) for v in evaluator._train_idx_host.values())
        assert not hasattr(evaluator, "_train_idx_gpu")

    def test_chunk_mask_is_bounded_by_the_chunk_and_exact(self, held) -> None:
        evaluator = _evaluator()
        evaluator._ensure_train_idx_host()
        train = _train()

        rows, cols = evaluator._chunk_train_mask([2, 5])

        assert rows.size == len(train[2]) + len(train[5])
        assert {int(c) for r, c in zip(rows, cols, strict=True) if r == 0} == train[2]
        assert {int(c) for r, c in zip(rows, cols, strict=True) if r == 1} == train[5]

    def test_item_block_path_transfers_only_the_block(self, held) -> None:
        seen: list[int] = []

        class _Spy(_GoldenModel):
            def predict(self, user_id, item_ids):
                seen.append(len(item_ids))
                if len(item_ids) > 5:
                    raise torch.cuda.OutOfMemoryError("simulated")
                return super().predict(user_id, item_ids)

        evaluator = _evaluator()
        with torch.no_grad():
            evaluator._score_user_in_blocks(_Spy(held), 0, torch.arange(N_ITEMS))

        accepted = [n for n in seen if n <= 5]
        assert sum(accepted) == N_ITEMS, "every item scored exactly once"
        assert seen[-1] == 1, "37 = 9 blocks of 4 + a final block of 1"


class TestHostRankingBudget:
    def test_budget_is_linear_in_the_catalogue(self) -> None:
        assert host_ranking_bytes(N_ITEMS) == N_ITEMS * HOST_RANKING_BYTES_PER_ITEM
        assert host_ranking_bytes(0) == 0

    def test_explicit_budget_below_one_vector_fails_clearly(self, held) -> None:
        model = _ItemOOM(held, limit_at=lambda offset: N_ITEMS)
        evaluator = _evaluator(host_budget_bytes=host_ranking_bytes(N_ITEMS) - 1)

        with pytest.raises(HostMemoryBudgetError, match="n_items=37"):
            evaluator.evaluate_with_records(model, device="cpu")

    def test_explicit_budget_that_fits_produces_the_golden_records(self, held) -> None:
        reference = _records(_GoldenModel(held))

        actual = _records(
            _ItemOOM(held, limit_at=lambda offset: 6),
            host_budget_bytes=host_ranking_bytes(N_ITEMS),
        )

        _assert_same(actual, reference)

    def test_probed_budget_admits_a_small_catalogue(self, held) -> None:
        """``None`` probes the host; a 37-item vector always fits."""
        _records(_ItemOOM(held, limit_at=lambda offset: 6))

    def test_batched_path_does_not_admit_host_ranking(self, held) -> None:
        """The GPU-side path never sorts on the host, so the budget is not applied."""
        evaluator = _evaluator(host_budget_bytes=1)

        frame = evaluator.evaluate_per_user(_GoldenModel(held), device="cpu")

        assert len(frame) == N_USERS
