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


class TestOmnibusAnnotation:
    """E4: the Friedman gate annotates pairwise rows, never suppresses them."""

    def _pairs_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "family": ["backbone_within_model", "frozen_vs_finetuned", "vs_baseline"],
                "group": ["model=vbpr,condition=frozen", "all", "all"],
                "config_a": ["a", "b", "c"],
                "config_b": ["x", "y", "z"],
            }
        )

    def test_joins_verdict_per_family_instance(self) -> None:
        from src.steps.statistical import _omnibus_column

        fried_rows = [
            {
                "family": "backbone_within_model",
                "group": "model=vbpr,condition=frozen",
                "p_value": 0.001,
                "significant": True,
            },
            # Friedman undefined for < 3 configs -> NaN p-value.
            {
                "family": "frozen_vs_finetuned",
                "group": "all",
                "p_value": float("nan"),
                "significant": False,
            },
        ]

        col = _omnibus_column(self._pairs_df(), fried_rows)

        assert col.iloc[0] is True
        assert pd.isna(col.iloc[1])  # undefined omnibus -> NaN, not False
        assert pd.isna(col.iloc[2])  # no Friedman row (disabled) -> NaN
        assert len(col) == 3  # every pairwise row kept (annotate, not suppress)

    def test_all_nan_when_friedman_disabled(self) -> None:
        from src.steps.statistical import _omnibus_column

        col = _omnibus_column(self._pairs_df(), fried_rows=[])

        assert col.isna().all()


class TestPairCollectionFamiliesSkipFriedman:
    """R1: no K-way Friedman over the pair-collection families' configs.

    ``vs_baseline`` / ``frozen_vs_finetuned`` bundle ~all dataset
    configs; an omnibus over them tests "everything is equivalent" and
    is trivially significant — those families must not appear in the
    Friedman output at all, and their pairwise rows carry
    ``omnibus_significant = NaN``.
    """

    def _instances_and_df(self):
        from src.evaluation.comparison_families import enumerate_family_instances

        pairs = [("bpr", "none")] + [
            (m, e) for m in ("vbpr", "deepstyle") for e in ("resnet50", "vit_b16", "cvt_13")
        ]
        df = _long(pairs, n_users=12)
        instances = enumerate_family_instances(df, ["backbone_within_model", "vs_baseline"])
        return instances, df

    def test_vs_baseline_emits_no_friedman_row(self) -> None:
        from src.steps.statistical import _family_friedman_rows

        instances, df = self._instances_and_df()
        assert any(i.family == "vs_baseline" for i in instances)  # family IS enumerated

        rows = _family_friedman_rows(df, instances, metric="ndcg@10", alpha=0.05)

        assert all(r["family"] != "vs_baseline" for r in rows)

    def test_one_dimension_family_omnibus_still_computed(self) -> None:
        from src.steps.statistical import _family_friedman_rows

        instances, df = self._instances_and_df()

        rows = _family_friedman_rows(df, instances, metric="ndcg@10", alpha=0.05)

        computed = [r for r in rows if r["family"] == "backbone_within_model"]
        assert computed and all(not pd.isna(r["p_value"]) for r in computed)

    def test_pairwise_rows_annotated_nan_for_vs_baseline(self) -> None:
        from src.steps.statistical import _family_friedman_rows, _omnibus_column

        instances, df = self._instances_and_df()
        rows = _family_friedman_rows(df, instances, metric="ndcg@10", alpha=0.05)
        pairs_df = pd.DataFrame(
            {
                "family": ["vs_baseline", "backbone_within_model"],
                "group": ["all", "model=vbpr,condition=frozen"],
            }
        )

        col = _omnibus_column(pairs_df, rows)

        assert pd.isna(col.iloc[0])  # pair-collection family -> NaN by construction
        assert col.iloc[1] in (True, False)  # homogeneous family -> real verdict


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
