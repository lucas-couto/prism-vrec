"""v2 protocol contract tests: native dims, alignment, PCA train-only fit.

All synthetic and CPU-only — no backbone downloads.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.fusions.online import LearnedAlignmentFusion, RaggedSources, load_embedding
from src.fusions.strategies import fuse_pca, fuse_pca_per_model, pca_align

N_ITEMS = 12
D1, D2 = 20, 8  # differing native dims


def _sources() -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [
        rng.standard_normal((N_ITEMS, D1)).astype("float32"),
        rng.standard_normal((N_ITEMS, D2)).astype("float32"),
    ]


class TestPcaTrainOnlyFit:
    def test_fit_rows_change_the_transform(self) -> None:
        # Fitting on a subset must generally differ from fitting on all
        # rows — proving the train_items argument is actually honoured.
        sources = _sources()
        train = np.arange(6)

        full = fuse_pca(sources, n_components=4, train_items=None)
        train_only = fuse_pca(sources, n_components=4, train_items=train)

        assert full.shape == train_only.shape == (N_ITEMS, 4)
        assert not np.allclose(full, train_only)

    def test_pca_per_model_concatenates(self) -> None:
        out = fuse_pca_per_model(_sources(), n_components=3, train_items=np.arange(6))

        # per-source PCA to k then CONCATENATION -> M*k dims.
        assert out.shape == (N_ITEMS, 2 * 3)

    def test_pca_align_brings_sources_to_common_dim(self) -> None:
        aligned = pca_align(_sources(), 5, train_items=np.arange(6))

        assert [a.shape for a in aligned] == [(N_ITEMS, 5), (N_ITEMS, 5)]

    def test_deterministic_given_seed(self) -> None:
        a = fuse_pca(_sources(), n_components=4, train_items=np.arange(6), random_state=42)
        b = fuse_pca(_sources(), n_components=4, train_items=np.arange(6), random_state=42)

        assert np.array_equal(a, b)


class TestLearnedAlignmentFusion:
    @pytest.mark.parametrize(
        "strategy",
        ["mean", "sum", "prod", "max_pool", "attention_weighted", "gated", "adaptive_gated"],
    )
    def test_ops_produce_aligned_dim_and_backprop(self, strategy: str) -> None:
        torch.manual_seed(0)
        module = LearnedAlignmentFusion([D1, D2], dim=6, strategy=strategy)
        concat = torch.randn(4, D1 + D2)

        out = module(concat)
        out.sum().backward()

        assert out.shape == (4, 6)
        assert torch.isfinite(out).all()
        assert module.projections[0].weight.grad is not None

    def test_weighted_mean_uses_fixed_weights(self) -> None:
        module = LearnedAlignmentFusion(
            [D1, D2], dim=6, strategy="weighted_mean", weights=[0.7, 0.3]
        )
        assert torch.allclose(module.fixed_weights, torch.tensor([0.7, 0.3]))

    def test_unknown_op_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            LearnedAlignmentFusion([D1, D2], dim=6, strategy="warp")

    def test_adaptive_gated_requires_two_sources(self) -> None:
        with pytest.raises(ValueError, match="2 sources"):
            LearnedAlignmentFusion([D1, D2, D2], dim=6, strategy="adaptive_gated")


class TestLoadEmbeddingSidecars:
    def test_learned_sidecar_returns_ragged_sources(self, tmp_path) -> None:
        sources = _sources()
        np.save(tmp_path / "resnet50.npy", sources[0])
        np.save(tmp_path / "vit_b16.npy", sources[1])
        sidecar = {
            "strategy": "mean",
            "online": True,
            "alignment": "learned",
            "dim": 6,
            "components": ["resnet50.npy", "vit_b16.npy"],
            "normalize": True,
            "fusion_kwargs": {},
        }
        path = tmp_path / "hybrid_mean_learned_D6.json"
        path.write_text(json.dumps(sidecar))

        arr = load_embedding(path)

        assert isinstance(arr, RaggedSources)
        assert arr.shape == (N_ITEMS, D1 + D2)
        assert arr.source_dims == [D1, D2]
        assert arr.aligned_dim == 6
        assert arr.strategy == "mean"

    def test_single_component_sidecar_degenerates_to_passthrough(self, tmp_path) -> None:
        # The smoke profile fuses a single extractor by design; the loader
        # must accept the M=1 sidecar (warning, not error).
        sources = _sources()
        np.save(tmp_path / "resnet50.npy", sources[0])
        sidecar = {
            "strategy": "mean",
            "online": True,
            "alignment": "learned",
            "dim": 6,
            "components": ["resnet50.npy"],
            "normalize": True,
            "fusion_kwargs": {},
        }
        path = tmp_path / "hybrid_mean_learned_D6.json"
        path.write_text(json.dumps(sidecar))

        arr = load_embedding(path)

        assert isinstance(arr, RaggedSources)
        assert arr.shape == (N_ITEMS, D1)
        assert arr.source_dims == [D1]

        fusion = LearnedAlignmentFusion([D1], dim=6, strategy="mean")
        out = fusion(torch.from_numpy(sources[0]))
        assert out.shape == (N_ITEMS, 6)

    def test_empty_sidecar_fails_loudly(self, tmp_path) -> None:
        path = tmp_path / "hybrid_mean_learned_D6.json"
        path.write_text(json.dumps({"strategy": "mean", "components": []}))

        with pytest.raises(ValueError, match="no components"):
            load_embedding(path)

    def test_equal_dim_sidecar_still_stacks_3d(self, tmp_path) -> None:
        rng = np.random.default_rng(1)
        a = rng.standard_normal((N_ITEMS, 5)).astype("float32")
        b = rng.standard_normal((N_ITEMS, 5)).astype("float32")
        np.save(tmp_path / "a.npy", a)
        np.save(tmp_path / "b.npy", b)
        sidecar = {"strategy": "adaptive_gated", "online": True, "components": ["a.npy", "b.npy"]}
        path = tmp_path / "hybrid_adaptive_gated_pca_D5.json"
        path.write_text(json.dumps(sidecar))

        arr = load_embedding(path)

        assert arr.shape == (N_ITEMS, 2, 5)

    def test_meta_sidecar_mismatch_fails_loudly(self, tmp_path) -> None:
        rng = np.random.default_rng(2)
        np.save(tmp_path / "resnet50.npy", rng.standard_normal((N_ITEMS, 16)))
        (tmp_path / "resnet50.meta.json").write_text(
            json.dumps({"name": "resnet50", "native_dim": 2048})
        )

        with pytest.raises(ValueError, match="native_dim"):
            load_embedding(tmp_path / "resnet50.npy")

    def test_meta_sidecar_match_passes(self, tmp_path) -> None:
        rng = np.random.default_rng(3)
        np.save(tmp_path / "resnet50.npy", rng.standard_normal((N_ITEMS, 16)))
        (tmp_path / "resnet50.meta.json").write_text(
            json.dumps({"name": "resnet50", "native_dim": 16})
        )

        arr = load_embedding(tmp_path / "resnet50.npy")

        assert arr.shape == (N_ITEMS, 16)


class TestRecommenderWithNativeDims:
    """Mudança 2 validation: two native dims train with the same d."""

    @pytest.mark.parametrize("native_dim", [D1, D2])
    def test_same_d_across_different_native_dims(self, native_dim: int) -> None:
        from src.recommenders.vbpr import VBPR

        rng = np.random.default_rng(0)
        visual = rng.standard_normal((N_ITEMS, native_dim)).astype("float32")
        model = VBPR(
            4, N_ITEMS, visual_embeddings=visual, config={"latent_dim": 4, "visual_dim": 6}
        )

        users = torch.tensor([0, 1])
        loss = model.bpr_loss(*model(users, torch.tensor([1, 2]), torch.tensor([3, 4])))
        loss.backward()

        # E maps native_dim -> d and is trainable.
        assert model.visual_projection.in_features == native_dim
        assert model.visual_projection.out_features == 6
        assert model.visual_projection.weight.grad is not None
        assert torch.isfinite(loss)

    def test_ragged_learned_alignment_through_recommender(self) -> None:
        from src.recommenders.vbpr import VBPR

        concat = np.concatenate(_sources(), axis=1)
        ragged = RaggedSources(concat, source_dims=[D1, D2], strategy="mean", aligned_dim=6)
        model = VBPR(
            4, N_ITEMS, visual_embeddings=ragged, config={"latent_dim": 4, "visual_dim": 5}
        )

        loss = model.bpr_loss(
            *model(torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([3, 4]))
        )
        loss.backward()

        # Alignment projections are co-trained; E consumes the aligned dim.
        assert model.visual_dim_raw == 6
        align = model._online_fusion
        assert isinstance(align, LearnedAlignmentFusion)
        assert align.projections[0].weight.grad is not None
        assert torch.isfinite(loss)
