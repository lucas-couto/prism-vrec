"""Tests for the long-format consolidation pipeline.

Covers the embedding-name parser, per-CSV reshape functions, and the
end-to-end ``write_consolidated`` flow on synthetic granular CSVs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.reporting import (
    classify_table_file,
    consolidate_bootstrap,
    consolidate_evaluation,
    consolidate_statistical_tests,
    evaluation_to_long,
    friedman_to_long,
    pairwise_to_long,
    parse_config,
    parse_embedding_name,
    summary_to_long,
    write_consolidated,
)

_KNOWN_RECS = ["bpr", "vbpr", "avbpr", "deepstyle", "vnpr"]


class TestParseEmbeddingName:
    def test_single_extractor_frozen(self) -> None:
        result = parse_embedding_name("clip_vitb32_D128")
        assert result == {
            "extractor": "clip_vitb32",
            "fusion": "none",
            "condition": "frozen",
            "embedding_dim": 128,
        }

    def test_single_extractor_finetuned(self) -> None:
        result = parse_embedding_name("vit_b16_finetuned_D128")
        assert result["extractor"] == "vit_b16"
        assert result["fusion"] == "none"
        assert result["condition"] == "finetuned"
        assert result["embedding_dim"] == 128

    def test_hybrid_fusion(self) -> None:
        result = parse_embedding_name("hybrid_adaptive_gated_D128")
        assert result["extractor"] == "hybrid"
        assert result["fusion"] == "adaptive_gated"
        assert result["condition"] == "frozen"

    def test_hybrid_fusion_finetuned(self) -> None:
        result = parse_embedding_name("hybrid_mean_finetuned_D128")
        assert result["extractor"] == "hybrid"
        assert result["fusion"] == "mean"
        assert result["condition"] == "finetuned"

    def test_hybrid_with_underscores_in_fusion(self) -> None:
        result = parse_embedding_name("hybrid_pca_per_model_nc64_D128")
        assert result["fusion"] == "pca_per_model_nc64"

    def test_hybrid_with_weight_suffix(self) -> None:
        result = parse_embedding_name("hybrid_weighted_mean_w0.5_D128")
        assert result["fusion"] == "weighted_mean_w0.5"

    def test_none_baseline(self) -> None:
        result = parse_embedding_name("none")
        assert result == {
            "extractor": "none",
            "fusion": "none",
            "condition": "both",
            "embedding_dim": None,
        }

    def test_extractor_with_underscore_digits(self) -> None:
        result = parse_embedding_name("coatnet_0_D128")
        assert result["extractor"] == "coatnet_0"
        assert result["fusion"] == "none"


class TestParseConfig:
    def test_recommender_with_simple_embedding(self) -> None:
        result = parse_config("vbpr_clip_vitb32_D128", _KNOWN_RECS)
        assert result["recommender"] == "vbpr"
        assert result["extractor"] == "clip_vitb32"

    def test_recommender_with_hybrid_embedding(self) -> None:
        result = parse_config("avbpr_hybrid_concat_finetuned_D128", _KNOWN_RECS)
        assert result["recommender"] == "avbpr"
        assert result["extractor"] == "hybrid"
        assert result["fusion"] == "concat"
        assert result["condition"] == "finetuned"

    def test_bpr_none(self) -> None:
        result = parse_config("bpr_none", _KNOWN_RECS)
        assert result["recommender"] == "bpr"
        assert result["extractor"] == "none"

    def test_unknown_recommender_falls_back(self) -> None:
        result = parse_config("mystery_resnet50_D128", _KNOWN_RECS)
        assert result["recommender"] == "unknown"


class TestEvaluationToLong:
    def test_melts_metric_columns(self) -> None:
        eval_df = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "model_name": ["vbpr", "avbpr", "vbpr", "avbpr"],
                "embedding_name": [
                    "clip_vitb32_D128",
                    "clip_vitb32_D128",
                    "clip_vitb32_D128",
                    "clip_vitb32_D128",
                ],
                "precision@5": [0.1, 0.2, 0.3, 0.4],
                "ndcg@10": [0.5, 0.6, 0.7, 0.8],
            }
        )

        long_df = evaluation_to_long(eval_df, dataset="amazon_fashion", condition="frozen")

        assert set(long_df["metric"].unique()) == {"precision", "ndcg"}
        assert set(long_df["k"].unique()) == {5, 10}
        assert len(long_df) == 8
        assert "recommender" in long_df.columns
        assert "extractor" in long_df.columns
        assert long_df["dataset"].unique().tolist() == ["amazon_fashion"]

    def test_no_metric_columns_returns_empty(self) -> None:
        eval_df = pd.DataFrame({"user_id": [1], "model_name": ["x"], "embedding_name": ["y"]})
        out = evaluation_to_long(eval_df, dataset="d", condition="frozen")
        assert out.empty


class TestSummaryToLong:
    def test_parses_config_and_appends_metadata(self) -> None:
        summary_df = pd.DataFrame(
            [
                {
                    "config": "vbpr_clip_vitb32_D128",
                    "n_users": 100,
                    "mean": 0.5,
                    "ci_lower": 0.4,
                    "ci_upper": 0.6,
                    "ci_width": 0.2,
                },
                {
                    "config": "bpr_none",
                    "n_users": 100,
                    "mean": 0.3,
                    "ci_lower": 0.25,
                    "ci_upper": 0.35,
                    "ci_width": 0.1,
                },
            ]
        )

        out = summary_to_long(
            summary_df,
            dataset="amazon_fashion",
            metric="ndcg",
            k=10,
            known_recommenders=_KNOWN_RECS,
        )

        assert out.loc[0, "recommender"] == "vbpr"
        assert out.loc[1, "recommender"] == "bpr"
        assert out["dataset"].unique().tolist() == ["amazon_fashion"]
        assert out["metric"].unique().tolist() == ["ndcg"]
        assert out["k"].unique().tolist() == [10]


class TestFriedmanToLong:
    def test_tags_with_dataset_metric_k(self) -> None:
        friedman_df = pd.DataFrame(
            [
                {
                    "statistic": 12.5,
                    "p_value": 0.001,
                    "significant": True,
                    "n_configs": 10,
                    "n_users": 1000,
                }
            ]
        )

        out = friedman_to_long(friedman_df, dataset="tradesy", metric="map", k=20)

        assert out.loc[0, "test_type"] == "friedman"
        assert out.loc[0, "dataset"] == "tradesy"
        assert out.loc[0, "metric"] == "map"
        assert out.loc[0, "k"] == 20


class TestPairwiseToLong:
    def test_splits_config_a_and_config_b(self) -> None:
        pairwise_df = pd.DataFrame(
            [
                {
                    "config_a": "vbpr_clip_vitb32_D128",
                    "config_b": "avbpr_hybrid_mean_finetuned_D128",
                    "statistic": 1.0,
                    "p_value": 0.02,
                    "corrected_p": 0.04,
                    "significant": True,
                    "mean_a": 0.5,
                    "mean_b": 0.55,
                    "cohens_d": 0.1,
                    "cliffs_delta": 0.05,
                    "cliffs_magnitude": "negligible",
                }
            ]
        )

        out = pairwise_to_long(
            pairwise_df,
            dataset="amazon_men",
            metric="precision",
            k=5,
            known_recommenders=_KNOWN_RECS,
        )

        assert out.loc[0, "recommender_a"] == "vbpr"
        assert out.loc[0, "extractor_a"] == "clip_vitb32"
        assert out.loc[0, "condition_a"] == "frozen"
        assert out.loc[0, "recommender_b"] == "avbpr"
        assert out.loc[0, "extractor_b"] == "hybrid"
        assert out.loc[0, "fusion_b"] == "mean"
        assert out.loc[0, "condition_b"] == "finetuned"
        assert out.loc[0, "test_type"] == "wilcoxon"


class TestClassifyTableFile:
    @pytest.mark.parametrize(
        "filename,expected_kind",
        [
            ("amazon_fashion_evaluation_frozen.csv", "evaluation"),
            ("amazon_men_evaluation_finetuned.csv", "evaluation"),
            ("amazon_women_summary_ndcg_at_10.csv", "summary"),
            ("tradesy_friedman_precision_at_20.csv", "friedman"),
            ("amazon_fashion_pairwise_map_at_5.csv", "pairwise"),
        ],
    )
    def test_recognised_patterns(self, filename: str, expected_kind: str) -> None:
        info = classify_table_file(Path(filename))
        assert info is not None
        assert info["kind"] == expected_kind

    def test_unrelated_files_return_none(self) -> None:
        assert classify_table_file(Path("amazon_fashion_evaluation_done.csv")) is None
        assert classify_table_file(Path("amazon_fashion_evaluation_combined.csv")) is None
        assert classify_table_file(Path("foo.csv")) is None


class TestWriteConsolidatedEndToEnd:
    def test_full_flow_on_synthetic_data(self, tmp_path: Path) -> None:
        tables = tmp_path / "tables"
        tables.mkdir()

        eval_df = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "model_name": ["vbpr", "avbpr", "vbpr", "avbpr"],
                "embedding_name": ["clip_vitb32_D128"] * 4,
                "precision@5": [0.1, 0.2, 0.3, 0.4],
                "ndcg@10": [0.5, 0.6, 0.7, 0.8],
            }
        )
        eval_df.to_csv(tables / "amazon_fashion_evaluation_frozen.csv", index=False)

        summary_df = pd.DataFrame(
            [
                {
                    "config": "vbpr_clip_vitb32_D128",
                    "n_users": 2,
                    "mean": 0.2,
                    "ci_lower": 0.1,
                    "ci_upper": 0.3,
                    "ci_width": 0.2,
                },
                {
                    "config": "avbpr_clip_vitb32_D128",
                    "n_users": 2,
                    "mean": 0.3,
                    "ci_lower": 0.2,
                    "ci_upper": 0.4,
                    "ci_width": 0.2,
                },
            ]
        )
        summary_df.to_csv(tables / "amazon_fashion_summary_ndcg_at_10.csv", index=False)

        friedman_df = pd.DataFrame(
            [{"statistic": 5.0, "p_value": 0.05, "significant": True, "n_configs": 2, "n_users": 2}]
        )
        friedman_df.to_csv(tables / "amazon_fashion_friedman_ndcg_at_10.csv", index=False)

        pairwise_df = pd.DataFrame(
            [
                {
                    "config_a": "vbpr_clip_vitb32_D128",
                    "config_b": "avbpr_clip_vitb32_D128",
                    "statistic": 1.0,
                    "p_value": 0.1,
                    "corrected_p": 0.1,
                    "significant": False,
                    "mean_a": 0.2,
                    "mean_b": 0.3,
                    "cohens_d": 0.5,
                    "cliffs_delta": 0.3,
                    "cliffs_magnitude": "small",
                }
            ]
        )
        pairwise_df.to_csv(tables / "amazon_fashion_pairwise_ndcg_at_10.csv", index=False)

        written = write_consolidated(tables)

        assert written["evaluation_aggregated"].exists()
        assert written["bootstrap_ci"].exists()
        assert written["statistical_tests"].exists()

        eval_out = pd.read_csv(written["evaluation_aggregated"])
        assert {
            "recommender",
            "extractor",
            "fusion",
            "condition",
            "metric",
            "k",
            "mean",
            "n_users",
        }.issubset(eval_out.columns)

        tests_out = pd.read_csv(written["statistical_tests"])
        assert set(tests_out["test_type"]) == {"friedman", "wilcoxon"}

    def test_empty_dir_writes_empty_csvs(self, tmp_path: Path) -> None:
        written = write_consolidated(tmp_path)
        for path in written.values():
            assert path.exists()
            assert path.stat().st_size >= 0


class TestConsolidateHelpers:
    def test_consolidate_evaluation_returns_empty_for_no_files(self, tmp_path: Path) -> None:
        assert consolidate_evaluation(tmp_path).empty

    def test_consolidate_bootstrap_returns_empty_for_no_files(self, tmp_path: Path) -> None:
        assert consolidate_bootstrap(tmp_path).empty

    def test_consolidate_statistical_tests_returns_empty_for_no_files(self, tmp_path: Path) -> None:
        assert consolidate_statistical_tests(tmp_path).empty
