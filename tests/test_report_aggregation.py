"""A1 + S8: the Markdown report must aggregate, never cherry-pick.

Per-user evaluation rows ranked directly report the best USERS (Top-N
full of ndcg = 1.0); the frozen-vs-finetuned pivot with
``aggfunc="max"`` let each condition win with a DIFFERENT embedding
(asymmetric best-of-N).  The report now collapses per-user rows to
config means and pairs conditions per (model, base embedding).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.report import (
    _aggregate_per_user_rows,
    _section_frozen_vs_finetuned,
    _section_top_n,
)


def _per_user_frame() -> pd.DataFrame:
    """Two configs × 4 users: config A mean 0.5, config B mean 0.25."""
    rows = []
    for user, val in enumerate([1.0, 1.0, 0.0, 0.0]):
        rows.append(
            {
                "dataset": "d",
                "condition": "frozen",
                "model_name": "vbpr",
                "embedding_name": "resnet50",
                "user_id": user,
                "ndcg@10": val,
            }
        )
    for user, val in enumerate([1.0, 0.0, 0.0, 0.0]):
        rows.append(
            {
                "dataset": "d",
                "condition": "frozen",
                "model_name": "vbpr",
                "embedding_name": "vit_b16",
                "user_id": user,
                "ndcg@10": val,
            }
        )
    return pd.DataFrame(rows)


class TestPerUserAggregation:
    def test_collapses_to_config_means(self) -> None:
        out = _aggregate_per_user_rows(_per_user_frame())

        assert len(out) == 2
        assert "user_id" not in out.columns
        by_emb = out.set_index("embedding_name")["ndcg@10"]
        assert by_emb["resnet50"] == 0.5
        assert by_emb["vit_b16"] == 0.25

    def test_top_n_ranks_configs_not_users(self) -> None:
        # Without aggregation the top row would be a lucky user at 1.0.
        df = _aggregate_per_user_rows(_per_user_frame())

        table = _section_top_n(df, "ndcg@10", 1)

        assert "0.5000" in table
        assert "1.0000" not in table

    def test_aggregated_input_passes_through(self) -> None:
        df = pd.DataFrame(
            {
                "dataset": ["d"],
                "condition": ["frozen"],
                "model_name": ["vbpr"],
                "embedding_name": ["resnet50"],
                "ndcg@10": [0.31],
            }
        )

        out = _aggregate_per_user_rows(df)

        pd.testing.assert_frame_equal(out, df)


class TestFrozenVsFinetunedPairing:
    def _frame(self) -> pd.DataFrame:
        # resnet50: frozen 0.30 vs finetuned 0.20 (FT LOSES on the pair).
        # vit_b16: only frozen exists, at a high 0.90 — the old
        # max-of-means compared 0.90 (frozen) against 0.20 (finetuned)
        # under model-level max and hid resnet50's story entirely.
        return pd.DataFrame(
            {
                "dataset": ["d", "d", "d"],
                "condition": ["frozen", "finetuned", "frozen"],
                "model_name": ["vbpr", "vbpr", "vbpr"],
                "embedding_name": ["resnet50", "resnet50_finetuned", "vit_b16"],
                "ndcg@10": [0.30, 0.20, 0.90],
            }
        )

    def test_pairs_by_base_embedding(self) -> None:
        table = _section_frozen_vs_finetuned(self._frame(), "ndcg@10")

        # The only pair is resnet50: delta = 0.20 - 0.30 = -0.10.
        assert "-0.1000" in table
        # The unpaired vit_b16 must not leak into the comparison.
        assert "0.9000" not in table

    def test_no_common_pair_yields_note(self) -> None:
        df = pd.DataFrame(
            {
                "dataset": ["d", "d"],
                "condition": ["frozen", "finetuned"],
                "model_name": ["vbpr", "vbpr"],
                "embedding_name": ["resnet50", "vit_b16_finetuned"],
                "ndcg@10": [0.3, 0.4],
            }
        )

        table = _section_frozen_vs_finetuned(df, "ndcg@10")

        assert "No (model, embedding) pair" in table


class TestPairedCliffsDelta:
    def test_bernoulli_anomaly_is_resolved(self) -> None:
        # S2's demonstrable anomaly: A~Bern(0.10) vs B~Bern(0.05) —
        # the between-groups delta reads ~0.05 (below every conventional
        # cut-off) despite a 2x hit rate.  The paired form scores the
        # discordant users, and the win/loss/tie triplet says so directly.
        from src.evaluation.statistical import (
            cliffs_delta,
            paired_cliffs_delta,
            paired_outcomes,
        )

        rng = np.random.default_rng(0)
        n = 50_000
        a = (rng.random(n) < 0.10).astype(float)
        b = (rng.random(n) < 0.05).astype(float)

        unpaired = cliffs_delta(a, b)
        paired = paired_cliffs_delta(a, b)
        outcomes = paired_outcomes(a, b)

        assert abs(unpaired) < 0.147  # the anomaly: "negligible" by any cut-off
        assert abs(paired) > abs(unpaired) * 0.9  # paired at least as informative
        assert outcomes.n_wins > 1.5 * outcomes.n_losses  # A wins most discordant users
        # Direction and construction: (wins - losses) / n.
        expected = (np.sum((a - b) > 0) - np.sum((a - b) < 0)) / n
        assert paired == expected

    def test_all_wins_is_one(self) -> None:
        from src.evaluation.statistical import paired_cliffs_delta

        assert paired_cliffs_delta([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]) == 1.0

    def test_ties_shrink_the_delta(self) -> None:
        from src.evaluation.statistical import paired_cliffs_delta

        # 1 win, 3 ties: delta = 1/4 (pratt-consistent denominator).
        assert paired_cliffs_delta([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]) == 0.25


class TestConsolidatedTables:
    """One file per (dataset, kind), metric identity carried in each row."""

    def test_tag_metric_prepends_identity_columns(self) -> None:
        import pandas as pd

        from src.steps.statistical import _tag_metric

        df = pd.DataFrame({"statistic": [1.0]})

        out = _tag_metric(df, "ndcg", "10")

        assert list(out.columns[:2]) == ["metric", "k"]
        assert out.loc[0, "metric"] == "ndcg"
        assert out.loc[0, "k"] == 10

    def test_evaluation_mean_table_written_per_config(self, tmp_path) -> None:
        import pandas as pd

        from src.steps.evaluate import _write_mean_table

        src = tmp_path / "ds_evaluation_frozen.csv"
        pd.DataFrame(
            {
                "user_id": [1, 2, 1, 2],
                "model_name": ["vbpr", "vbpr", "bpr", "bpr"],
                "embedding_name": ["resnet50", "resnet50", "none", "none"],
                "ndcg@10": [0.2, 0.4, 0.1, 0.1],
            }
        ).to_csv(src, index=False)

        _write_mean_table(tmp_path, "ds", "frozen")

        out = pd.read_csv(tmp_path / "ds_evaluation_mean_frozen.csv")
        assert len(out) == 2  # one row per config, not per user
        vbpr = out[out["model_name"] == "vbpr"].iloc[0]
        assert vbpr["ndcg@10"] == 0.30000000000000004 or abs(vbpr["ndcg@10"] - 0.3) < 1e-9
        assert vbpr["n_users"] == 2

    def test_mean_table_skips_aggregated_input(self, tmp_path) -> None:
        import pandas as pd

        from src.steps.evaluate import _write_mean_table

        src = tmp_path / "ds_evaluation_frozen.csv"
        pd.DataFrame(
            {"model_name": ["vbpr"], "embedding_name": ["resnet50"], "ndcg@10": [0.3]}
        ).to_csv(src, index=False)

        _write_mean_table(tmp_path, "ds", "frozen")

        assert not (tmp_path / "ds_evaluation_mean_frozen.csv").exists()
