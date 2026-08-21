"""Tests for the memory-bounded user batching of the full-ranking path.

Scoring ``B`` users against ``N`` items allocates ``B * N`` elements
several times over, so a batch size that fits a 166K-item catalogue
overflows a 348K-item one — which is how a hyperparameter grid ended up
OOM-ing on every ``amazon_women`` and ``tradesy`` cell while
``amazon_fashion`` passed.

Two properties matter and are pinned here:

* the batch is derived from the catalogue size and the process's GPU
  allowance, not taken from the caller verbatim;
* **batching is an execution detail** — every row is ranked, masked and
  scored independently, so the metrics must be identical for any batch
  size, down to a single user per batch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.evaluation.protocol import (
    RANKING_BYTES_PER_ELEMENT,
    Evaluator,
    default_ranking_budget,
    plan_ranking_batch,
)

N_ITEMS = 40
N_USERS = 17


class _DeterministicModel(torch.nn.Module):
    """Scores with deliberate exact ties, so tie-breaking is exercised."""

    def __init__(self) -> None:
        super().__init__()
        rng = np.random.default_rng(7)
        # Coarse quantisation guarantees repeated scores within a row.
        table = rng.integers(0, 5, size=(N_USERS + 1, N_ITEMS)).astype(np.float32)
        self.register_buffer("table", torch.from_numpy(table))

    def predict(self, user_id: int, item_ids: torch.Tensor) -> torch.Tensor:
        return self.table[user_id][item_ids]

    def predict_batch(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        return self.table[user_ids][:, item_ids]


def _evaluator(**kwargs) -> Evaluator:
    rng = np.random.default_rng(3)
    train = {u: set(rng.choice(N_ITEMS, size=4, replace=False).tolist()) for u in range(N_USERS)}
    test = {u: {int((u * 7 + 3) % N_ITEMS)} for u in range(N_USERS)}
    return Evaluator(
        train_interactions=train,
        test_interactions=test,
        n_items=N_ITEMS,
        k_values=[5, 10],
        tiebreak_seed=11,
        **kwargs,
    )


class TestPlanRankingBatch:
    def test_budget_lowers_the_requested_batch(self):
        # Room for exactly 3 users over 1000 items.
        budget = 3 * 1000 * RANKING_BYTES_PER_ELEMENT

        assert plan_ranking_batch(512, 1000, budget) == 3

    def test_requested_value_is_an_upper_bound(self):
        budget = 10_000 * 1000 * RANKING_BYTES_PER_ELEMENT

        assert plan_ranking_batch(64, 1000, budget) == 64

    def test_bigger_catalogue_gets_a_smaller_batch(self):
        budget = 4 * 1024**3

        small = plan_ranking_batch(512, 166_270, budget)
        large = plan_ranking_batch(512, 347_591, budget)

        assert large < small

    def test_never_returns_zero(self):
        """One oversized user still has to be scored, not skipped."""
        assert plan_ranking_batch(512, 10_000_000, 1) == 1

    def test_unknown_budget_leaves_the_request_untouched(self):
        assert plan_ranking_batch(512, 1000, 0) == 512

    def test_empty_catalogue_does_not_divide_by_zero(self):
        assert plan_ranking_batch(512, 0, 4 * 1024**3) == 512

    def test_reference_case_fits_the_worker_allowance(self):
        """The amazon_women cell that OOM-ed at a fixed batch of 512.

        Three workers on a 16 GB card get ~6.25 GB each; half of that is
        the ranking's share.  The resulting batch must keep the peak
        inside the share.
        """
        share = int(16311 * 1024**2 * (1 / 3 + 0.05) * 0.5)

        batch = plan_ranking_batch(512, 347_591, share)

        assert batch < 512
        assert batch * 347_591 * RANKING_BYTES_PER_ELEMENT <= share


class TestDefaultRankingBudget:
    def test_cpu_yields_no_budget(self):
        """On CPU the caller's batch stands; host RAM is not the limit."""
        assert default_ranking_budget(torch.device("cpu")) == 0


class TestBatchingDoesNotChangeResults:
    def _frame(self, **kwargs) -> pd.DataFrame:
        evaluator = _evaluator(**kwargs)
        frame = evaluator.evaluate_per_user(_DeterministicModel(), device="cpu", batch_size=512)
        return frame.sort_values("user_id").reset_index(drop=True)

    def test_single_user_batches_match_one_big_batch(self):
        """The strongest form: B=1 must equal B=all."""
        budget = 1 * N_ITEMS * RANKING_BYTES_PER_ELEMENT  # forces B == 1

        one_at_a_time = self._frame(ranking_budget_bytes=budget)
        all_at_once = self._frame(ranking_budget_bytes=0)

        pd.testing.assert_frame_equal(one_at_a_time, all_at_once)

    def test_intermediate_batch_sizes_agree(self):
        frames = [
            self._frame(ranking_budget_bytes=b * N_ITEMS * RANKING_BYTES_PER_ELEMENT)
            for b in (2, 5, 13, 500)
        ]

        for frame in frames[1:]:
            pd.testing.assert_frame_equal(frames[0], frame)

    def test_batched_path_still_matches_the_single_user_path(self):
        """Cross-path agreement must survive the buffer refactor."""
        evaluator = _evaluator(ranking_budget_bytes=3 * N_ITEMS * RANKING_BYTES_PER_ELEMENT)
        model = _DeterministicModel()
        all_items = torch.arange(N_ITEMS)

        batched = evaluator._evaluate_batched(model, all_items, 512)
        single = evaluator._evaluate_single(model, all_items)

        by_user = {r["user_id"]: r for r in single}
        for record in batched:
            reference = by_user[record["user_id"]]
            for metric in ("recall@10", "ndcg@10", "precision@10"):
                assert record[metric] == reference[metric], (record["user_id"], metric)


class TestRecordsSurviveTheHeadSlice:
    def test_top_items_still_carries_twenty_entries(self):
        """``top_items`` persists 20 ids even though max_k is 10.

        The refactor slices the sort permutation before mapping it back
        to item ids; slicing to ``max_k`` instead of
        ``max(max_k, TOP_ITEMS_PERSISTED)`` would silently truncate this
        column from 20 entries to 10.
        """
        evaluator = _evaluator()

        records = evaluator.per_user_records(_DeterministicModel(), device="cpu")

        assert len(records["top_items"].iloc[0]) == min(20, N_ITEMS)

    def test_records_are_unaffected_by_batching(self):
        model = _DeterministicModel()
        tight = _evaluator(ranking_budget_bytes=N_ITEMS * RANKING_BYTES_PER_ELEMENT)
        loose = _evaluator(ranking_budget_bytes=0)

        pd.testing.assert_frame_equal(
            tight.per_user_records(model, device="cpu"),
            loose.per_user_records(model, device="cpu"),
        )
