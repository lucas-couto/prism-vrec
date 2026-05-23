"""Tests: statistical helpers compare by config (model_name + embedding_name).

Regression for the per-user bug where pivoting on ``model_name`` alone
raised ``ValueError: Index contains duplicate entries`` because a model
spans many embeddings.
"""

from __future__ import annotations

import pandas as pd

from src.evaluation.statistical import (
    _ensure_config,
    friedman_test,
    pairwise_significance,
    per_model_summary,
)


def _long(model_emb_pairs: list[tuple[str, str]], n_users: int) -> pd.DataFrame:
    """Build a rectangular long-format frame: every user in every cell."""
    rows = []
    for u in range(n_users):
        for i, (m, e) in enumerate(model_emb_pairs):
            rows.append(
                {
                    "user_id": u,
                    "model_name": m,
                    "embedding_name": e,
                    "ndcg@10": 0.1 * (i + 1) + 0.001 * u,
                }
            )
    return pd.DataFrame(rows)


class TestEnsureConfig:
    def test_builds_config_from_model_and_embedding(self) -> None:
        df = pd.DataFrame(
            {"user_id": [1], "model_name": ["vbpr"], "embedding_name": ["resnet50_D128"]}
        )

        out = _ensure_config(df)

        assert out["config"].tolist() == ["vbpr_resnet50_D128"]

    def test_deduplicates_baseline_duplicated_across_batteries(self) -> None:
        # bpr/none is written to both battery files -> two identical rows.
        df = pd.DataFrame(
            {
                "user_id": [7, 7],
                "model_name": ["bpr", "bpr"],
                "embedding_name": ["none", "none"],
                "ndcg@10": [0.3, 0.3],
            }
        )

        out = _ensure_config(df)

        assert len(out) == 1


class TestFriedmanByConfig:
    def test_does_not_crash_with_many_embeddings_per_model(self) -> None:
        df = _long(
            [
                ("deepstyle", "resnet50_D128"),
                ("deepstyle", "vit_b16_D128"),
                ("deepstyle", "convnext_base_D128"),
            ],
            n_users=5,
        )

        result = friedman_test(df, metric="ndcg@10", alpha=0.05)

        assert result["n_configs"] == 3
        assert result["p_value"] == result["p_value"]  # NaN check


class TestPairwiseByConfig:
    def test_returns_config_columns(self) -> None:
        df = _long(
            [("vbpr", "resnet50_D128"), ("vbpr", "resnet50_finetuned_D128")],
            n_users=6,
        )

        out = pairwise_significance(df, metric="ndcg@10", correction="holm")

        assert {"config_a", "config_b"} <= set(out.columns)
        assert out.iloc[0]["config_a"] == "vbpr_resnet50_D128"
        assert out.iloc[0]["config_b"] == "vbpr_resnet50_finetuned_D128"


class TestPerModelSummaryByConfig:
    def test_one_row_per_config_not_per_model(self) -> None:
        df = _long(
            [("deepstyle", "resnet50_D128"), ("deepstyle", "convnext_base_D128")],
            n_users=8,
        )

        out = per_model_summary(df, metric="ndcg@10", n_iterations=50)

        assert "config" in out.columns
        assert sorted(out["config"]) == [
            "deepstyle_convnext_base_D128",
            "deepstyle_resnet50_D128",
        ]
