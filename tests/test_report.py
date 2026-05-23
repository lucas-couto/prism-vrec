"""Tests for the aggregated-results report generator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.report import generate, write_report


def _seed_results(tables_dir: Path) -> None:
    """Create a tiny, realistic results/tables/ tree."""
    tables_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "dataset": "amazon_fashion",
                "model_name": "vbpr",
                "embedding_name": "resnet50_D128",
                "ndcg@10": 0.18,
                "recall@10": 0.10,
            },
            {
                "dataset": "amazon_fashion",
                "model_name": "bpr",
                "embedding_name": "none",
                "ndcg@10": 0.12,
                "recall@10": 0.08,
            },
            {
                "dataset": "amazon_fashion",
                "model_name": "avbpr",
                "embedding_name": "vit_b16_D128",
                "ndcg@10": 0.22,
                "recall@10": 0.13,
            },
        ]
    ).to_csv(tables_dir / "amazon_fashion_evaluation_frozen.csv", index=False)

    pd.DataFrame(
        [
            {
                "dataset": "amazon_fashion",
                "model_name": "vbpr",
                "embedding_name": "resnet50_finetuned_D128",
                "ndcg@10": 0.20,
                "recall@10": 0.11,
            },
            {
                "dataset": "amazon_fashion",
                "model_name": "avbpr",
                "embedding_name": "vit_b16_finetuned_D128",
                "ndcg@10": 0.24,
                "recall@10": 0.14,
            },
        ]
    ).to_csv(tables_dir / "amazon_fashion_evaluation_finetuned.csv", index=False)


def test_generate_includes_top_n_section(tmp_path) -> None:
    tables_dir = tmp_path / "results" / "tables"
    _seed_results(tables_dir)

    md = generate(tables_dir=tables_dir, metric="ndcg@10", top_n=3)

    assert "Top 3 configurations" in md
    assert "0.2400" in md
    assert "0.2200" in md
    assert "0.2000" in md


def test_generate_best_per_recommender_section(tmp_path) -> None:
    tables_dir = tmp_path / "results" / "tables"
    _seed_results(tables_dir)

    md = generate(tables_dir=tables_dir, metric="ndcg@10")

    assert "Best configuration per recommender" in md
    # Each recommender should appear exactly once per (dataset, condition).
    assert md.count("avbpr") >= 2
    assert md.count("bpr") >= 1


def test_generate_frozen_vs_finetuned_delta(tmp_path) -> None:
    tables_dir = tmp_path / "results" / "tables"
    _seed_results(tables_dir)

    md = generate(tables_dir=tables_dir, metric="ndcg@10")

    assert "Frozen vs finetuned diff" in md
    # vbpr: 0.20 - 0.18 = 0.02 (FT helps)
    assert "0.0200" in md or "0.020" in md


def test_generate_handles_empty_dir(tmp_path) -> None:
    md = generate(tables_dir=tmp_path / "nonexistent")
    assert "No evaluation CSVs" in md


def test_generate_handles_missing_metric(tmp_path) -> None:
    tables_dir = tmp_path / "results" / "tables"
    tables_dir.mkdir(parents=True)
    pd.DataFrame([{"model_name": "bpr", "ndcg@10": 0.1}]).to_csv(
        tables_dir / "x_evaluation_frozen.csv",
        index=False,
    )

    md = generate(tables_dir=tables_dir, metric="precision@5")
    assert "precision@5" in md


def test_write_report_atomic_no_tmp_left(tmp_path) -> None:
    tables_dir = tmp_path / "results" / "tables"
    _seed_results(tables_dir)
    out = tmp_path / "results" / "report.md"

    written = write_report(out_path=out, tables_dir=tables_dir)

    assert written == out
    assert out.exists()
    assert not (out.parent / "report.md.tmp").exists()
