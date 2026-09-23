"""Uniform cost labels resolve backbones and fusion for every stem shape.

A hybrid stem does not name its backbones, so ``train`` / ``evaluate``
cost cells read them back from the provenance sidecars — through a
projection and down to the native sources.
"""

from __future__ import annotations

from src.utils.cost_labels import embedding_labels, extractor_of, source_labels
from src.utils.identity import write_provenance


def _npy_source(name: str) -> dict:
    return {"kind": "npy", "name": name, "dtype": "float32", "shape": [4, 2], "sha256": "0"}


def _fusion(dataset_dir, artifact: str, strategy: str, sources: list[str]) -> None:
    write_provenance(
        dataset_dir / artifact,
        {"kind": "fusion", "strategy": strategy, "sources": [_npy_source(s) for s in sources]},
    )


class TestExtractorOf:
    def test_should_resolve_projected_stem_to_its_backbone(self):
        assert extractor_of("clip_vitb32_pcaw128") == "clip_vitb32"

    def test_should_return_none_for_unknown_stem(self):
        assert extractor_of("not_a_backbone") is None


class TestEmbeddingLabels:
    def test_should_label_baseline_with_no_extractor(self):
        labels = embedding_labels("none")

        assert labels == {"embedding": "none", "extractors": [], "fusion": None}

    def test_should_label_native_stem_with_its_backbone(self):
        labels = embedding_labels("resnet50")

        assert labels["extractors"] == ["resnet50"]
        assert labels["fusion"] is None

    def test_should_read_offline_hybrid_backbones_from_provenance(self, tmp_path):
        _fusion(tmp_path, "hybrid_concat.npy", "concat", ["resnet50.npy", "vit_b16.npy"])

        labels = embedding_labels("hybrid_concat", tmp_path / "hybrid_concat.npy")

        assert labels["extractors"] == ["resnet50", "vit_b16"]
        assert labels["fusion"] == "concat"

    def test_should_read_online_hybrid_backbones_from_json_sidecar(self, tmp_path):
        stem = "hybrid_max_pool_learned_D128"
        _fusion(tmp_path, f"{stem}.json", "max_pool", ["resnet50.npy", "vit_b16.npy"])

        labels = embedding_labels(stem, tmp_path / f"{stem}.json")

        assert labels["extractors"] == ["resnet50", "vit_b16"]
        assert labels["fusion"] == "max_pool"

    def test_should_walk_through_projection_to_the_fused_backbones(self, tmp_path):
        _fusion(tmp_path, "hybrid_pca_nc128.npy", "pca", ["resnet50.npy", "vit_b16.npy"])
        write_provenance(
            tmp_path / "hybrid_pca_nc128_pcaw128.npy",
            {"kind": "projection", "source": _npy_source("hybrid_pca_nc128.npy")},
        )

        labels = embedding_labels(
            "hybrid_pca_nc128_pcaw128", tmp_path / "hybrid_pca_nc128_pcaw128.npy"
        )

        assert labels["extractors"] == ["resnet50", "vit_b16"]
        assert labels["fusion"] == "pca"

    def test_should_leave_hybrid_unresolved_when_sidecar_is_missing(self, tmp_path):
        labels = embedding_labels("hybrid_mean", tmp_path / "hybrid_mean.npy")

        assert labels["extractors"] == []
        assert labels["fusion"] is None


class TestSourceLabels:
    def test_should_deduplicate_backbones_in_first_seen_order(self, tmp_path):
        paths = [tmp_path / "vit_b16.npy", tmp_path / "resnet50_comp.npy", tmp_path / "vit_b16.npy"]

        assert source_labels(paths) == ["vit_b16", "resnet50"]
