"""Evaluate must not resurrect checkpoints of disabled models/backbones.

Third instance of the disk-vs-config leak: the models directory
accumulates ``*_best.pt`` across runs, and the evaluate step used to
evaluate every parseable checkpoint regardless of what the current
config enables.
"""

from __future__ import annotations

from src.utils.variant_filters import checkpoint_matches_config, embedding_matches_config


def _config(**overrides) -> dict:
    base = {
        "recommenders_enabled": ["bpr", "vbpr"],
        "extractors_enabled": ["resnet50"],
        "fusion_strategies_enabled": ["mean"],
        "embedding_variants": "both",
    }
    base.update(overrides)
    return base


class TestEmbeddingMatchesConfig:
    def test_baseline_always_matches(self) -> None:
        assert embedding_matches_config("none", _config(extractors_enabled=[]))

    def test_enabled_backbone_matches_disabled_does_not(self) -> None:
        cfg = _config()

        assert embedding_matches_config("resnet50_p128", cfg)
        assert not embedding_matches_config("clip_vitb32_p128", cfg)
        assert not embedding_matches_config("coatnet_0_p128", cfg)

    def test_fusion_gated_by_strategy_list(self) -> None:
        cfg = _config()

        assert embedding_matches_config("hybrid_mean_p128", cfg)
        assert not embedding_matches_config("hybrid_gated_l1_0_p128", cfg)

    def test_variant_gate_applies(self) -> None:
        cfg = _config(embedding_variants="projected")

        assert embedding_matches_config("resnet50_p128", cfg)
        assert not embedding_matches_config("resnet50", cfg)  # native excluded
        assert embedding_matches_config("none", cfg)  # baseline exempt

    def test_unknown_stems_never_match(self) -> None:
        cfg = _config()

        assert not embedding_matches_config("mystery_backbone", cfg)
        assert not embedding_matches_config("hybrid_mystery_strategy", cfg)


class TestCheckpointMatchesConfig:
    def test_disabled_recommender_is_skipped(self) -> None:
        info = {"model_name": "deepstyle", "embedding_name": "resnet50_p128"}

        assert not checkpoint_matches_config(info, _config())

    def test_enabled_cell_passes(self) -> None:
        info = {"model_name": "vbpr", "embedding_name": "resnet50_p128"}

        assert checkpoint_matches_config(info, _config())

    def test_stale_backbone_checkpoint_is_skipped(self) -> None:
        # The exact leak from the 2026-08-24 run: clip/coatnet _best.pt
        # from an earlier run, extractors_enabled reduced to resnet50.
        info = {"model_name": "vbpr", "embedding_name": "clip_vitb32_p128"}

        assert not checkpoint_matches_config(info, _config())


class TestFeatureGateHonoursConfig:
    """The train gate must not scan or fail on artifacts outside the config."""

    def _setup(self, tmp_path):
        import json

        import numpy as np

        proc = tmp_path / "p" / "ds"
        emb = tmp_path / "e" / "ds"
        proc.mkdir(parents=True)
        emb.mkdir(parents=True)
        (proc / "item2idx.json").write_text(json.dumps({str(i): i for i in range(4)}))
        # Enabled backbone: valid float32 with non-zero norms.
        rng = np.random.default_rng(0)
        np.save(emb / "resnet50.npy", rng.standard_normal((4, 8)).astype(np.float32))
        # Disabled backbone: poisoned (float64 would fail the gate).
        np.save(emb / "clip_vitb32.npy", np.zeros((4, 8), dtype=np.float64))
        # Disabled fusion: poisoned too.
        np.save(emb / "hybrid_gated_l1_0.npy", np.zeros((4, 8), dtype=np.float64))
        # Component artifact (fp16, 3-D): out of the gate's scope.
        np.save(emb / "resnet50_comp.npy", np.zeros((4, 3, 8), dtype=np.float16))
        return str(tmp_path / "p"), str(tmp_path / "e")

    def test_out_of_config_artifacts_neither_scan_nor_fail(self, tmp_path) -> None:
        from src.steps.validate_features import gate_dataset_features

        proc, emb = self._setup(tmp_path)
        config = {
            "extractors_enabled": ["resnet50"],
            "fusion_strategies_enabled": [],
            "embedding_variants": "both",
            "extractors": {"resnet50": {"raw_dim": 8}},
        }

        # Poisoned out-of-config artifacts must not raise.
        gate_dataset_features(["ds"], config, embeddings_dir=emb, processed_dir=proc)

    def test_in_config_artifact_still_fails_loud(self, tmp_path) -> None:
        import numpy as np
        import pytest

        from src.steps.validate_features import (
            FeatureValidationError,
            gate_dataset_features,
        )

        proc, emb = self._setup(tmp_path)
        # Poison the ENABLED backbone: the gate must still catch it.
        np.save(f"{emb}/ds/resnet50.npy", np.zeros((4, 8), dtype=np.float64))
        config = {
            "extractors_enabled": ["resnet50"],
            "fusion_strategies_enabled": [],
            "embedding_variants": "both",
            "extractors": {"resnet50": {"raw_dim": 8}},
        }

        with pytest.raises(FeatureValidationError):
            gate_dataset_features(["ds"], config, embeddings_dir=emb, processed_dir=proc)
