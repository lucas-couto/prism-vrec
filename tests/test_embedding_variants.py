"""Tests for the native / projected variant switches.

``projection:`` in extractors.yaml decides what gets *written*; these two
flags decide what gets *consumed* — ``embedding_variants`` in
recommenders.yaml for the training cells, ``extractor_variants`` in
fusion.yaml for the fusion sources.  Keeping them separate is what lets
one run write both families and train on one of them.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.steps.fuse import _resolve_extractor_variants
from src.steps.train import filter_by_variant


class TestRecommenderVariantFilter:
    NAMES = [
        "resnet50",
        "resnet50_p128",
        "vit_b16",
        "vit_b16_p128",
        "hybrid_mean",
        "hybrid_mean_p128",
    ]

    def test_both_keeps_everything(self):
        assert filter_by_variant(self.NAMES, "both") == self.NAMES

    def test_native_drops_the_projected_artifacts(self):
        assert filter_by_variant(self.NAMES, "native") == [
            "resnet50",
            "vit_b16",
            "hybrid_mean",
        ]

    def test_projected_drops_the_native_artifacts(self):
        assert filter_by_variant(self.NAMES, "projected") == [
            "resnet50_p128",
            "vit_b16_p128",
            "hybrid_mean_p128",
        ]

    def test_fusion_outputs_follow_their_sources(self):
        """A hybrid built from projected sources belongs to that family."""
        assert filter_by_variant(["hybrid_mean_p128"], "projected") == ["hybrid_mean_p128"]
        assert filter_by_variant(["hybrid_mean_p128"], "native") == []

    def test_the_non_visual_baseline_is_never_filtered_out(self):
        """Dropping "none" would silently remove plain BPR from the battery."""
        for variant in ("native", "projected", "both"):
            assert "none" in filter_by_variant(["none", "resnet50"], variant)

    def test_finetuned_artifacts_are_classified_by_their_projection(self):
        names = ["resnet50_finetuned", "resnet50_p128_finetuned"]

        assert filter_by_variant(names, "native") == ["resnet50_finetuned"]
        assert filter_by_variant(names, "projected") == ["resnet50_p128_finetuned"]


class TestFusionVariantResolution:
    EXTRACTORS = ["resnet50", "vit_b16"]

    def test_default_is_native_and_keeps_the_alignment(self):
        [(names, token, pre_aligned)] = _resolve_extractor_variants({}, self.EXTRACTORS)

        assert names == self.EXTRACTORS
        assert token == ""
        assert pre_aligned is False

    def test_projected_sources_bypass_the_alignment(self):
        config = {
            "extractor_variants": "projected",
            "projection": {"method": "pca", "dim": 128},
        }

        [(names, token, pre_aligned)] = _resolve_extractor_variants(config, self.EXTRACTORS)

        assert names == ["resnet50_p128", "vit_b16_p128"]
        assert token == "_p128"
        assert pre_aligned is True

    def test_both_yields_one_pass_per_variant(self):
        config = {
            "extractor_variants": "both",
            "projection": {"method": "random", "dim": 64},
        }

        variants = _resolve_extractor_variants(config, self.EXTRACTORS)

        assert [v[1] for v in variants] == ["", "_p64"]
        assert [v[2] for v in variants] == [False, True]

    def test_projected_without_a_projection_fails_loudly(self):
        config = {"extractor_variants": "projected"}

        with pytest.raises(ValueError, match="projection.method 'none'"):
            _resolve_extractor_variants(config, self.EXTRACTORS)

    def test_mismatched_widths_fail_loudly(self):
        """Sources at different widths do not share a space; fusing them is meaningless."""
        config = {
            "extractor_variants": "projected",
            "projection": {"method": "random", "dim": 128},
            "extractors": {"vit_b16": {"projection": {"dim": 64}}},
        }

        with pytest.raises(ValueError, match="different widths"):
            _resolve_extractor_variants(config, self.EXTRACTORS)

    def test_unknown_variant_is_rejected(self):
        with pytest.raises(ValueError, match="extractor_variants"):
            _resolve_extractor_variants({"extractor_variants": "projeted"}, self.EXTRACTORS)


class TestFusionTaskRouting:
    """The task list a projected variant produces, end to end."""

    def _dataset(self, tmp_path, names):
        dataset_dir = tmp_path / "amazon_fashion"
        dataset_dir.mkdir(parents=True)
        for name in names:
            np.save(dataset_dir / f"{name}.npy", np.zeros((8, 32), dtype=np.float32))
        processed = tmp_path / "processed" / "amazon_fashion"
        processed.mkdir(parents=True)
        (processed / "train.csv").write_text("user_idx,item_idx\n0,0\n1,1\n")
        return dataset_dir, str(tmp_path), str(tmp_path / "processed")

    def test_projected_sources_produce_offline_tasks_with_no_alignment(self, tmp_path):
        from src.steps.fuse import _collect_fusion_tasks

        _, emb_dir, processed = self._dataset(tmp_path, ["resnet50_p32", "vit_b16_p32"])

        tasks = _collect_fusion_tasks(
            "amazon_fashion",
            emb_dir,
            processed,
            ["resnet50_p32", "vit_b16_p32"],
            {},
            True,
            {"mean"},
            "learned",
            128,
            variant_token="_p32",
            pre_aligned=True,
        )

        [task] = tasks
        # Offline .npy, not a learned-alignment .json sidecar.
        assert task["output_path"].endswith("hybrid_mean_p32.npy")
        assert "sidecar_payload" not in task

    def test_native_sources_still_route_through_the_alignment(self, tmp_path):
        from src.steps.fuse import _collect_fusion_tasks

        _, emb_dir, processed = self._dataset(tmp_path, ["resnet50", "vit_b16"])

        tasks = _collect_fusion_tasks(
            "amazon_fashion",
            emb_dir,
            processed,
            ["resnet50", "vit_b16"],
            {},
            True,
            {"mean"},
            "learned",
            128,
        )

        [task] = tasks
        assert task["output_path"].endswith("hybrid_mean_learned_D128.json")
        assert task["sidecar_payload"]["alignment"] == "learned"

    def test_the_variant_token_keeps_both_outputs_apart(self, tmp_path):
        from src.steps.fuse import _collect_fusion_tasks

        _, emb_dir, processed = self._dataset(
            tmp_path, ["resnet50", "vit_b16", "resnet50_p32", "vit_b16_p32"]
        )
        common = dict(fusion_config={}, normalize=True, enabled_strategies={"concat"})

        native = _collect_fusion_tasks(
            "amazon_fashion",
            emb_dir,
            processed,
            ["resnet50", "vit_b16"],
            common["fusion_config"],
            common["normalize"],
            common["enabled_strategies"],
            "learned",
            128,
        )
        projected = _collect_fusion_tasks(
            "amazon_fashion",
            emb_dir,
            processed,
            ["resnet50_p32", "vit_b16_p32"],
            common["fusion_config"],
            common["normalize"],
            common["enabled_strategies"],
            "learned",
            128,
            variant_token="_p32",
            pre_aligned=True,
        )

        assert native[0]["output_path"] != projected[0]["output_path"]
        assert projected[0]["output_path"].endswith("hybrid_concat_p32.npy")
