"""Tests for the fixed linear projection configured in extractors.yaml.

The v2 contract has extraction emit native dims, leaving the mapping to
a common space to the recommender's learned E or to the fuse step's
alignment.  The optional ``projection:`` block adds a third route: one
fixed linear map per artifact, written *alongside* the native features
so element-wise fusion can consume equal-dim sources with nothing
learned online.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.extractors.projection import (
    ProjectionConfig,
    ensure_projected,
    projected_path,
    resolve_projection_config,
)


@pytest.fixture
def native(tmp_path) -> Path:
    """A ``(200, 512)`` native pooled artifact on disk."""
    rng = np.random.default_rng(0)
    path = tmp_path / "resnet50.npy"
    np.save(path, rng.standard_normal((200, 512)).astype(np.float32))
    return path


class TestConfigResolution:
    def test_absent_block_means_native_only(self):
        assert resolve_projection_config({}, "resnet50") is None

    def test_method_none_means_native_only(self):
        config = {"projection": {"method": "none", "dim": 128}}

        assert resolve_projection_config(config, "resnet50") is None

    def test_global_block_applies_to_every_extractor(self):
        config = {"projection": {"method": "random", "dim": 128, "seed": 7}}

        resolved = resolve_projection_config(config, "vit_b16")

        assert resolved == ProjectionConfig(method="random", dim=128, seed=7)

    def test_per_extractor_block_overrides_the_global_one(self):
        config = {
            "projection": {"method": "random", "dim": 128},
            "extractors": {"cvt_13": {"projection": {"dim": 64}}},
        }

        assert resolve_projection_config(config, "cvt_13").dim == 64
        assert resolve_projection_config(config, "resnet50").dim == 128

    def test_an_extractor_can_opt_out_of_a_global_projection(self):
        config = {
            "projection": {"method": "pca", "dim": 128},
            "extractors": {"resnet50": {"projection": {"method": "none"}}},
        }

        assert resolve_projection_config(config, "resnet50") is None

    def test_unknown_method_is_rejected(self):
        with pytest.raises(ValueError, match="projection.method"):
            resolve_projection_config({"projection": {"method": "umap"}}, "resnet50")

    def test_unknown_key_is_rejected(self):
        """A typo must fail loudly instead of silently reverting to default."""
        with pytest.raises(ValueError, match="unknown keys"):
            resolve_projection_config({"projection": {"methd": "random"}}, "resnet50")


class TestArtifactNaming:
    def test_token_follows_the_extractor_name(self, tmp_path):
        assert projected_path(tmp_path / "resnet50.npy", 128).name == "resnet50_p128.npy"

    def test_token_precedes_the_finetuned_marker(self, tmp_path):
        """So fuse's `{extractor}{condition_suffix}` resolves in both conditions."""
        out = projected_path(tmp_path / "resnet50_finetuned.npy", 128)

        assert out.name == "resnet50_p128_finetuned.npy"


class TestRandomProjection:
    def test_writes_the_requested_dim(self, native):
        cfg = ProjectionConfig(method="random", dim=128)

        written = ensure_projected(native, cfg)

        assert np.load(written).shape == (200, 128)

    def test_is_reproducible_across_runs(self, native, tmp_path):
        cfg = ProjectionConfig(method="random", dim=64, seed=7)
        first = np.load(ensure_projected(native, cfg))

        second_source = tmp_path / "copy" / "resnet50.npy"
        second_source.parent.mkdir()
        np.save(second_source, np.load(native))
        second = np.load(ensure_projected(second_source, cfg))

        np.testing.assert_array_equal(first, second)

    def test_different_extractors_get_different_matrices(self, tmp_path):
        """One shared matrix would correlate the projected spaces by construction."""
        rng = np.random.default_rng(0)
        rows = rng.standard_normal((50, 512)).astype(np.float32)
        cfg = ProjectionConfig(method="random", dim=32, seed=7)
        a_path, b_path = tmp_path / "resnet50.npy", tmp_path / "vit_b16.npy"
        np.save(a_path, rows)
        np.save(b_path, rows)

        a = np.load(ensure_projected(a_path, cfg))
        b = np.load(ensure_projected(b_path, cfg))

        assert not np.allclose(a, b)

    def test_needs_no_fit_set(self, native):
        """Data-independent: it cannot leak val/test items because it never reads them."""
        assert ensure_projected(native, ProjectionConfig(method="random", dim=32)) is not None


class TestPcaProjection:
    def test_writes_the_requested_dim(self, native):
        cfg = ProjectionConfig(method="pca", dim=64)

        written = ensure_projected(native, cfg, train_items=list(range(150)))

        assert np.load(written).shape == (200, 64)

    def test_fit_uses_only_the_train_items(self, native):
        """A different fit set must produce a different basis."""
        cfg = ProjectionConfig(method="pca", dim=16)
        first = np.load(ensure_projected(native, cfg, train_items=list(range(100))))

        second_source = native.with_name("vit_b16.npy")
        np.save(second_source, np.load(native))
        second = np.load(ensure_projected(second_source, cfg, train_items=list(range(100, 200))))

        assert not np.allclose(first, second)

    def test_without_a_fit_set_it_fails_loudly(self, native):
        with pytest.raises(ValueError, match="train-item"):
            ensure_projected(native, ProjectionConfig(method="pca", dim=64))


class TestContract:
    def test_the_native_artifact_is_never_modified(self, native):
        before = np.load(native).copy()

        ensure_projected(native, ProjectionConfig(method="random", dim=64))

        np.testing.assert_array_equal(np.load(native), before)

    def test_an_existing_projection_is_left_alone(self, native):
        cfg = ProjectionConfig(method="random", dim=64)
        first = ensure_projected(native, cfg)

        assert ensure_projected(native, cfg) is None
        assert first.exists()

    def test_every_extractor_lands_in_the_same_space(self, tmp_path):
        """The point of the feature: differing native dims, one shared width."""
        rng = np.random.default_rng(0)
        cfg = ProjectionConfig(method="random", dim=128)
        widths = {"resnet50": 2048, "vit_b16": 768, "cvt_13": 384}

        shapes = []
        for name, native_dim in widths.items():
            path = tmp_path / f"{name}.npy"
            np.save(path, rng.standard_normal((40, native_dim)).astype(np.float32))
            shapes.append(np.load(ensure_projected(path, cfg)).shape)

        assert shapes == [(40, 128)] * 3

    def test_projecting_upward_fails_loudly(self, tmp_path):
        path = tmp_path / "cvt_13.npy"
        np.save(path, np.zeros((10, 64), dtype=np.float32))

        with pytest.raises(ValueError, match="must not exceed"):
            ensure_projected(path, ProjectionConfig(method="random", dim=128))

    def test_the_projector_is_persisted_next_to_the_artifact(self, native):
        cfg = ProjectionConfig(method="random", dim=64, seed=7)

        written = ensure_projected(native, cfg)

        stored = np.load(written.with_suffix(".proj.npz"))
        assert stored["matrix"].shape == (512, 64)
        meta = json.loads(written.with_suffix(".proj.json").read_text())
        assert meta["method"] == "random"
        assert meta["dim"] == 64
        assert meta["fit"] == "data-independent"

    def test_the_persisted_matrix_reproduces_the_artifact(self, native):
        """The projector on disk IS the map that was applied — auditable."""
        cfg = ProjectionConfig(method="random", dim=64)
        written = ensure_projected(native, cfg)

        matrix = np.load(written.with_suffix(".proj.npz"))["matrix"]
        expected = np.load(native) @ matrix

        np.testing.assert_allclose(np.load(written), expected, rtol=1e-5, atol=1e-5)

    def test_chunking_does_not_change_the_result(self, tmp_path):
        """Peak memory is a function of the chunk; the output is not."""
        rng = np.random.default_rng(1)
        rows = rng.standard_normal((100, 256)).astype(np.float32)
        cfg = ProjectionConfig(method="random", dim=32, seed=3)
        # Same file NAME in two directories: the matrix is derived from the
        # artifact name, so both runs share it and the arrays are comparable.
        paths = []
        for sub in ("a", "b"):
            path = tmp_path / sub / "resnet50.npy"
            path.parent.mkdir()
            np.save(path, rows)
            paths.append(path)

        whole = np.load(ensure_projected(paths[0], cfg, chunk_rows=10_000))
        chunked = np.load(ensure_projected(paths[1], cfg, chunk_rows=7))

        # Not bit-identical: BLAS blocks a (100, 256) matmul differently
        # from a (7, 256) one, which reorders float32 accumulation. The
        # residual is ~1e-6, i.e. rounding, not a different projection.
        np.testing.assert_allclose(whole, chunked, rtol=1e-5, atol=1e-5)

    def test_the_random_matrix_survives_a_new_interpreter(self, native):
        """Derivation must not depend on PYTHONHASHSEED, which is salted per process."""
        import subprocess
        import sys

        cfg = ProjectionConfig(method="random", dim=32, seed=5)
        first = np.load(ensure_projected(native, cfg).with_suffix(".proj.npz"))["matrix"]

        script = (
            "import numpy as np;"
            "from src.extractors.projection import _random_matrix;"
            "np.save('%s', _random_matrix(512, 32, 5, 'resnet50_p32'))"
        ) % (native.parent / "second.npy")
        subprocess.run([sys.executable, "-c", script], check=True)
        second = np.load(native.parent / "second.npy")

        np.testing.assert_array_equal(first, second)


class TestExtractStepIntegration:
    """The hook in ``steps.extract``, exercised without a backbone."""

    def _native(self, tmp_path, name="resnet50", dim=512):
        path = tmp_path / f"{name}.npy"
        rng = np.random.default_rng(2)
        np.save(path, rng.standard_normal((30, dim)).astype(np.float32))
        meta = {"name": name, "native_dim": dim, "kind": "pooled", "weights_id": "IMAGENET1K_V2"}
        path.with_suffix(".meta.json").write_text(json.dumps(meta))
        return path

    def test_projecting_an_extracted_catalogue_needs_no_backbone(self, tmp_path):
        """The reason the hook does not take an extractor instance."""
        from src.steps.extract import _project_pooled

        native = self._native(tmp_path)

        assert _project_pooled(native, ProjectionConfig(method="random", dim=64), None) is True
        assert np.load(tmp_path / "resnet50_p64.npy").shape == (30, 64)

    def test_the_sidecar_declares_the_projected_width(self, tmp_path):
        """Otherwise the loader's meta cross-check rejects the artifact."""
        from src.steps.extract import _project_pooled

        native = self._native(tmp_path)
        _project_pooled(native, ProjectionConfig(method="random", dim=64), None)

        meta = json.loads((tmp_path / "resnet50_p64.meta.json").read_text())
        assert meta["native_dim"] == 64
        assert meta["source_native_dim"] == 512
        assert meta["name"] == "resnet50_p64"
        assert meta["projection"] == {"method": "random", "dim": 64, "source": "resnet50.npy"}

    def test_the_sidecar_passes_the_loader_cross_check(self, tmp_path):
        from src.fusions.online import _validate_against_meta
        from src.steps.extract import _project_pooled

        native = self._native(tmp_path)
        _project_pooled(native, ProjectionConfig(method="random", dim=64), None)
        projected = tmp_path / "resnet50_p64.npy"

        _validate_against_meta(projected, np.load(projected))

    def test_backbone_provenance_is_carried_over(self, tmp_path):
        """A projected artifact must still say which weights produced it."""
        from src.steps.extract import _project_pooled

        native = self._native(tmp_path)
        _project_pooled(native, ProjectionConfig(method="random", dim=64), None)

        meta = json.loads((tmp_path / "resnet50_p64.meta.json").read_text())
        assert meta["weights_id"] == "IMAGENET1K_V2"

    def test_no_projection_configured_writes_nothing(self, tmp_path):
        from src.steps.extract import _project_pooled

        native = self._native(tmp_path)

        assert _project_pooled(native, None, None) is False
        assert list(tmp_path.glob("*_p*.npy")) == []

    def test_projected_artifacts_are_discovered_as_embeddings(self, tmp_path):
        """train/evaluate pick them up by globbing, so they need no registration."""
        from src.steps.extract import _project_pooled
        from src.steps.train import get_embedding_files

        dataset_dir = tmp_path / "amazon_fashion"
        dataset_dir.mkdir()
        for name, dim in (("resnet50", 512), ("vit_b16", 768)):
            _project_pooled(
                self._native(dataset_dir, name, dim),
                ProjectionConfig(method="random", dim=128),
                None,
            )

        stems = get_embedding_files(str(tmp_path), "amazon_fashion")

        assert "resnet50_p128" in stems
        assert "vit_b16_p128" in stems
        assert "resnet50" in stems
