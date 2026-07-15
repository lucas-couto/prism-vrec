"""Random seed-fixed tie-break + exact-tie instrumentation (Task 2).

Exact-score ties are broken by a fixed permutation seeded from the run's
global seed (``tiebreak_seed``), shared by every model of a run. When the
held-out is not tied the returned rank is identical to the old behaviour;
when it is tied, the rank follows the random key (seed-dependent), never
item id. Rank is recovered from ``map@k == 1/rank`` under leave-one-out
with ``k = n_items`` (always a hit).
"""

from __future__ import annotations

import numpy as np
import torch

from src.evaluation.protocol import Evaluator

N_ITEMS = 10


class _FixedScores:
    """Deterministic model returning a preset per-item score vector."""

    def __init__(self, scores: np.ndarray) -> None:
        self._scores = np.asarray(scores, dtype=np.float64)

    def eval(self) -> None:
        pass

    def predict(self, user_id: int, item_ids):
        idx = item_ids.cpu().numpy() if isinstance(item_ids, torch.Tensor) else np.asarray(item_ids)
        return torch.tensor(self._scores[idx], dtype=torch.float32)


def _ev(tiebreak_seed: int) -> Evaluator:
    return Evaluator(
        train_interactions={0: set()},
        test_interactions={0: {5}},
        n_items=N_ITEMS,
        k_values=[N_ITEMS],  # k = n_items so the held-out is always a hit
        tiebreak_seed=tiebreak_seed,
    )


def _heldout_rank(evaluator: Evaluator, scores: np.ndarray) -> int:
    df = evaluator.evaluate_per_user(_FixedScores(scores), device="cpu")
    mapk = float(df.loc[df["user_id"] == 0, f"map@{N_ITEMS}"].iloc[0])
    return round(1.0 / mapk)


# Held-out (item 5) tied with items 3 and 7 at score 1.0; item 9 sits
# strictly above; the rest strictly below.
_TIE_SCORES = np.zeros(N_ITEMS, dtype=np.float64)
_TIE_SCORES[[3, 5, 7]] = 1.0
_TIE_SCORES[9] = 2.0

# Distinct scores: held-out item 5 has exactly items 6,7,8,9 above it.
_DISTINCT_SCORES = np.arange(N_ITEMS, dtype=np.float64)


class TestPermutationDeterminism:
    def test_same_seed_same_permutation(self) -> None:
        assert np.array_equal(_ev(42)._tiebreak_key, _ev(42)._tiebreak_key)

    def test_different_seed_different_permutation(self) -> None:
        assert not np.array_equal(_ev(42)._tiebreak_key, _ev(43)._tiebreak_key)


class TestNonTiedInvariance:
    def test_untied_heldout_rank_is_seed_independent(self) -> None:
        # Item 5 has a unique score; items 6,7,8,9 are strictly above -> r=5.
        r42 = _heldout_rank(_ev(42), _DISTINCT_SCORES)
        r43 = _heldout_rank(_ev(43), _DISTINCT_SCORES)

        assert r42 == r43 == 5


class TestTiedHeldoutFollowsKey:
    def _expected_rank(self, evaluator: Evaluator) -> int:
        key = evaluator._tiebreak_key
        ahead_in_block = sum(1 for j in (3, 7) if key[j] < key[5])
        return 1 + 1 + ahead_in_block  # item 9 above + self + tied-ahead

    def test_rank_matches_key_formula_for_many_seeds(self) -> None:
        for seed in range(12):
            evaluator = _ev(seed)
            assert _heldout_rank(evaluator, _TIE_SCORES) == self._expected_rank(evaluator)

    def test_rank_varies_with_seed_and_is_not_fixed_by_id(self) -> None:
        ranks = {_heldout_rank(_ev(seed), _TIE_SCORES) for seed in range(20)}

        # A pure id tie-break would pin item 5 to rank 3 for every seed
        # (2nd of the block {3,5,7}); the random key must produce >1 value.
        assert len(ranks) > 1


class TestTieInstrumentation:
    def test_tie_block_size_is_measured(self) -> None:
        # Items 3, 5, 7 share the held-out's exact score -> block size 3.
        rows = _ev(42)._evaluate_single(_FixedScores(_TIE_SCORES), torch.arange(N_ITEMS))
        assert rows[0]["_tie_block_size"] == 3

    def test_untied_block_size_is_one(self) -> None:
        rows = _ev(42)._evaluate_single(_FixedScores(_DISTINCT_SCORES), torch.arange(N_ITEMS))
        assert rows[0]["_tie_block_size"] == 1

    def test_internal_column_stripped_from_public_df(self) -> None:
        df = _ev(42).evaluate_per_user(_FixedScores(_TIE_SCORES), device="cpu")
        # Diagnostic column must not leak into the metric matrix.
        assert "_tie_block_size" not in df.columns
