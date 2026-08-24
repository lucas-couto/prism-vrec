"""Tests that component artifacts are routed only to component-consuming models.

Guards the additive ``requires_components`` enumeration split in
``src/steps/train.py``: ACF trains on ``*_comp`` stems, every other
recommender keeps its pooled pool unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.steps.train import (
    EnabledRecommenderHasNoCellsError,
    _cell_counts,
    assert_enabled_recommenders_have_cells,
    build_job_list,
    is_component_artifact,
)

DATASET = "synthetic"
N_ITEMS = 4


def _write_idx(path: Path, n: int) -> None:
    path.write_text(json.dumps({str(i): i for i in range(n)}))


def _setup(tmp_path: Path, *, with_comp: bool) -> tuple[str, str]:
    processed = tmp_path / "p" / DATASET
    processed.mkdir(parents=True)
    _write_idx(processed / "user2idx.json", 3)
    _write_idx(processed / "item2idx.json", N_ITEMS)

    emb = tmp_path / "e" / DATASET
    emb.mkdir(parents=True)
    np.save(emb / "vit_b16_D128.npy", np.zeros((N_ITEMS, 8), dtype="float32"))
    np.save(emb / "hybrid_mean_D128.npy", np.zeros((N_ITEMS, 8), dtype="float32"))
    if with_comp:
        np.save(emb / "vit_b16_D128_comp.npy", np.zeros((N_ITEMS, 7, 8), dtype="float32"))
    return str(tmp_path / "p"), str(tmp_path / "e")


def _config() -> dict:
    return {
        "datasets": [DATASET],
        "recommenders_enabled": ["vbpr", "acf"],
        "embedding_dims": ["D128"],
        # The fusion/extractor filters honour the config: stems on disk
        # only become cells when their strategy/backbone is enabled here.
        "fusion_strategies_enabled": ["mean"],
        "extractors_enabled": ["vit_b16"],
        "common": {
            "latent_dim": [64],
            "learning_rate": [0.001],
            "l2_reg": [0.0001],
            "visual_dim": [64],
        },
        "acf": {"att_hidden": [64], "max_history": [50]},
    }


def test_is_component_artifact_detects_comp_suffix() -> None:
    assert is_component_artifact("vit_b16_D128_comp")
    assert not is_component_artifact("vit_b16_D128")
    assert not is_component_artifact("hybrid_mean_D128")


def test_acf_gets_only_component_stems(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    processed, emb = _setup(tmp_path, with_comp=True)

    jobs = build_job_list("frozen", _config(), processed, emb, "cpu")

    acf_embeddings = {j.embedding_name for j in jobs if j.model_name == "acf"}
    assert acf_embeddings == {"vit_b16_D128_comp"}


def test_vbpr_excludes_component_stems(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    processed, emb = _setup(tmp_path, with_comp=True)

    jobs = build_job_list("frozen", _config(), processed, emb, "cpu")

    vbpr_embeddings = {j.embedding_name for j in jobs if j.model_name == "vbpr"}
    assert vbpr_embeddings == {"vit_b16_D128", "hybrid_mean_D128"}


class TestSilentDropoutGuard:
    """Audit D5: an enabled recommender with zero cells must fail loud."""

    def test_acf_without_comp_artifacts_raises_actionable_error(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        processed, emb = _setup(tmp_path, with_comp=False)

        counts = _cell_counts("frozen", _config(), processed, emb)

        assert counts["acf"] == 0
        assert counts["vbpr"] > 0
        with pytest.raises(EnabledRecommenderHasNoCellsError, match=r"acf.*comp"):
            assert_enabled_recommenders_have_cells(counts, "frozen")

    def test_passes_when_comp_artifacts_exist(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        processed, emb = _setup(tmp_path, with_comp=True)

        counts = _cell_counts("frozen", _config(), processed, emb)

        assert_enabled_recommenders_have_cells(counts, "frozen")  # no raise

    def test_globally_empty_enumeration_does_not_raise(self) -> None:
        # No artifacts at all (e.g. finetuned condition before finetuning):
        # handled by the existing "no pending jobs" paths, not per model.
        assert_enabled_recommenders_have_cells({"vbpr": 0, "acf": 0}, "frozen")

    def test_feature_blind_model_exempt_in_finetuned_condition(self) -> None:
        # Plain BPR only runs frozen; its absence from finetuned is by design.
        assert_enabled_recommenders_have_cells({"bpr": 0, "vbpr": 3}, "finetuned")

    def test_visual_model_with_zero_cells_raises_in_finetuned(self) -> None:
        with pytest.raises(EnabledRecommenderHasNoCellsError, match="vbpr"):
            assert_enabled_recommenders_have_cells({"vbpr": 0, "deepstyle": 2}, "finetuned")


def test_vbpr_pool_is_identical_with_and_without_comp_file(tmp_path, monkeypatch) -> None:
    """Reproducibility guard: adding a _comp file must not change vbpr's jobs."""
    (tmp_path / "a").mkdir()
    monkeypatch.chdir(tmp_path / "a")
    p1, e1 = _setup(tmp_path / "a", with_comp=False)
    baseline = {
        j.embedding_name
        for j in build_job_list("frozen", _config(), p1, e1, "cpu")
        if j.model_name == "vbpr"
    }

    (tmp_path / "b").mkdir()
    monkeypatch.chdir(tmp_path / "b")
    p2, e2 = _setup(tmp_path / "b", with_comp=True)
    with_comp = {
        j.embedding_name
        for j in build_job_list("frozen", _config(), p2, e2, "cpu")
        if j.model_name == "vbpr"
    }

    assert baseline == with_comp


class TestFusionFilterHonoursConfig:
    """Disabled fusion strategies must not become cells from stale disk artifacts."""

    def _names(self) -> list[str]:
        return [
            "vit_b16_D128",
            "hybrid_mean_p128",
            "hybrid_pca_nc128_p128",
            "hybrid_pca_per_model_nc64_p128",
            "hybrid_adaptive_gated_p128",
            "hybrid_gated_l1_0_p128",
        ]

    def test_empty_list_keeps_no_fusion_cells(self) -> None:
        from src.steps.train import filter_by_enabled_fusions

        kept = filter_by_enabled_fusions(self._names(), {"fusion_strategies_enabled": []})

        assert kept == ["vit_b16_D128"]

    def test_only_enabled_strategies_survive(self) -> None:
        from src.steps.train import filter_by_enabled_fusions

        kept = filter_by_enabled_fusions(
            self._names(), {"fusion_strategies_enabled": ["mean", "gated"]}
        )

        assert kept == ["vit_b16_D128", "hybrid_mean_p128", "hybrid_gated_l1_0_p128"]

    def test_prefix_collisions_resolve_by_longest_match(self) -> None:
        # pca enabled must NOT drag pca_per_model along, nor gated drag
        # adaptive_gated.
        from src.steps.train import filter_by_enabled_fusions

        kept = filter_by_enabled_fusions(
            self._names(), {"fusion_strategies_enabled": ["pca", "adaptive_gated"]}
        )

        assert kept == [
            "vit_b16_D128",
            "hybrid_pca_nc128_p128",
            "hybrid_adaptive_gated_p128",
        ]

    def test_unknown_hybrid_stem_is_excluded(self) -> None:
        from src.steps.train import filter_by_enabled_fusions

        kept = filter_by_enabled_fusions(
            ["hybrid_totally_unknown_thing"], {"fusion_strategies_enabled": ["mean"]}
        )

        assert kept == []

    def test_disabled_fusions_produce_no_jobs(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        processed_dir, embeddings_dir = _setup(tmp_path, with_comp=True)
        config = _config()
        config["fusion_strategies_enabled"] = []

        jobs = build_job_list("frozen", config, processed_dir, embeddings_dir, "cpu")

        assert not any(j.embedding_name.startswith("hybrid_") for j in jobs)
        assert any(j.embedding_name == "vit_b16_D128" for j in jobs)


class TestExtractorFilterHonoursConfig:
    """Disabled extractors must not become cells from stale disk artifacts."""

    def test_disabled_backbones_are_dropped(self) -> None:
        from src.steps.train import filter_by_enabled_extractors

        names = [
            "none",
            "resnet50_p128",
            "cvt_13_p128",
            "levit_256_p128",
            "resnet50_finetuned",
            "hybrid_mean_p128",
        ]

        kept = filter_by_enabled_extractors(names, {"extractors_enabled": ["resnet50"]})

        # baseline and hybrids pass through; only resnet50 stems survive.
        assert kept == ["none", "resnet50_p128", "resnet50_finetuned", "hybrid_mean_p128"]

    def test_empty_list_keeps_only_baseline_and_hybrids(self) -> None:
        from src.steps.train import filter_by_enabled_extractors

        kept = filter_by_enabled_extractors(
            ["none", "vit_b16", "hybrid_mean_p128"], {"extractors_enabled": []}
        )

        assert kept == ["none", "hybrid_mean_p128"]

    def test_unknown_stem_is_excluded(self) -> None:
        from src.steps.train import filter_by_enabled_extractors

        kept = filter_by_enabled_extractors(
            ["mystery_backbone_p128"], {"extractors_enabled": ["resnet50"]}
        )

        assert kept == []

    def test_jobs_respect_extractors_enabled(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        processed_dir, embeddings_dir = _setup(tmp_path, with_comp=True)
        config = _config()
        config["extractors_enabled"] = []  # vit_b16 artifacts on disk, disabled

        jobs = build_job_list("frozen", config, processed_dir, embeddings_dir, "cpu")

        assert not any(j.embedding_name.startswith("vit_b16") for j in jobs)
        # hybrid (enabled strategy) and the bpr baseline survive.
        assert any(j.embedding_name == "hybrid_mean_D128" for j in jobs)
