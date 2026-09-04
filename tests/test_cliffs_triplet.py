"""Paired effect size is reported as the win / loss / tie triplet, without a magnitude label.

The within-pair Cliff's delta ``(wins - losses) / n`` is a net quantity:
1% wins / 0% losses / 99% ties (A never loses) and 50.5% / 49.5% / 0%
(noise) both give δ = 0.01.  The triplet disambiguates them, and the
Romano et al. cut-offs — calibrated for the between-groups delta — are
not attached to the paired form (metrics audit, findings #1 and #2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.statistical import (
    PairedOutcomes,
    paired_cliffs_delta,
    paired_outcomes,
    pairwise_significance,
)

TRIPLET_COLUMNS = ("n_wins", "n_losses", "n_ties", "pct_wins", "pct_losses", "pct_ties")


def _vectors() -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(7)
    return [
        (np.array([1.0, 0.0, 0.5, 0.2]), np.array([0.0, 0.0, 0.5, 0.9])),
        (np.zeros(50), np.zeros(50)),
        (np.ones(10), np.zeros(10)),
        (rng.integers(0, 2, 400).astype(float), rng.integers(0, 2, 400).astype(float)),
        (rng.normal(size=300), rng.normal(size=300)),
    ]


class TestPairedOutcomes:
    @pytest.mark.parametrize("a,b", _vectors())
    def test_delta_equals_pct_wins_minus_pct_losses(self, a, b) -> None:
        out = paired_outcomes(a, b)

        assert out.delta == pytest.approx(out.pct_wins - out.pct_losses)
        assert paired_cliffs_delta(a, b) == pytest.approx(out.delta)

    @pytest.mark.parametrize("a,b", _vectors())
    def test_counts_close_the_denominator(self, a, b) -> None:
        out = paired_outcomes(a, b)

        assert out.n_wins + out.n_losses + out.n_ties == len(a) == out.n
        assert out.pct_wins + out.pct_losses + out.pct_ties == pytest.approx(1.0)

    def test_identical_vectors_are_all_ties_with_zero_delta(self) -> None:
        a = np.array([0.0, 1.0, 0.0, 0.3])

        out = paired_outcomes(a, a.copy())

        assert out == PairedOutcomes(n_wins=0, n_losses=0, n_ties=4)
        assert out.delta == 0.0

    def test_empty_or_mismatched_inputs_give_empty_outcomes(self) -> None:
        assert paired_outcomes([], []) == PairedOutcomes(0, 0, 0)
        assert paired_outcomes([1.0, 2.0], [1.0]) == PairedOutcomes(0, 0, 0)
        assert paired_cliffs_delta([], []) == 0.0

    def test_same_delta_different_triplets_are_distinguished(self) -> None:
        """Audit scenarios A (consistent, sparse) and B (noise) share δ = 0.01."""
        n = 1000
        consistent_a = np.r_[np.ones(10), np.zeros(n - 10)]
        consistent_b = np.zeros(n)
        noise_a = np.r_[np.ones(505), np.zeros(495)]
        noise_b = np.r_[np.zeros(505), np.ones(495)]

        consistent = paired_outcomes(consistent_a, consistent_b)
        noise = paired_outcomes(noise_a, noise_b)

        assert consistent.delta == pytest.approx(0.01)
        assert noise.delta == pytest.approx(0.01)
        assert (consistent.n_wins, consistent.n_losses, consistent.n_ties) == (10, 0, 990)
        assert (noise.n_wins, noise.n_losses, noise.n_ties) == (505, 495, 0)


def _eval_frame() -> pd.DataFrame:
    """Long-format per-user frame: two configs (same model, two embeddings)."""
    rng = np.random.default_rng(3)
    rows = []
    for user in range(60):
        for embedding in ("resnet50", "vit_b16"):
            rows.append(
                {
                    "user_id": user,
                    "model_name": "vbpr",
                    "embedding_name": embedding,
                    "recall@10": float(rng.integers(0, 2)),
                }
            )
    return pd.DataFrame(rows)


class TestPairwiseSignificanceColumns:
    def test_table_carries_the_triplet_and_no_magnitude_label(self) -> None:
        table = pairwise_significance(
            _eval_frame(), metric="recall@10", include_effect_size=True, n_iterations=20
        )

        assert set(TRIPLET_COLUMNS) <= set(table.columns)
        assert "cliffs_magnitude" not in table.columns
        row = table.iloc[0]
        assert row["n_wins"] + row["n_losses"] == row["n_nonzero_pairs"]
        assert row["n_wins"] + row["n_losses"] + row["n_ties"] == row["n_pairs"]
        assert row["cliffs_delta"] == pytest.approx(row["pct_wins"] - row["pct_losses"])
