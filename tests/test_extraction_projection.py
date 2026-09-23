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
from src.steps.extract import _extract_for_config


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

    def test_needs_fit_covers_every_pca_variant(self):
        # Regression: extract/finetune resolved the fit set only for
        # method == "pca", so pca_whitened crashed the extract step with
        # "needs the train-item fit set; none was provided".
        assert ProjectionConfig(method="pca", dim=8).needs_fit is True
        assert ProjectionConfig(method="pca_whitened", dim=8).needs_fit is True
        assert ProjectionConfig(method="random", dim=8).needs_fit is False

    def test_unknown_method_is_rejected(self):
        with pytest.raises(ValueError, match="projection.method"):
            resolve_projection_config({"projection": {"method": "umap"}}, "resnet50")

    def test_unknown_key_is_rejected(self):
        """A typo must fail loudly instead of silently reverting to default."""
        with pytest.raises(ValueError, match="unknown keys"):
            resolve_projection_config({"projection": {"methd": "random"}}, "resnet50")


class TestArtifactNaming:
    def test_token_carries_the_method_and_the_width(self, tmp_path):
        """Width alone is not an identity: two methods share a width.

        The old ``p<dim>`` token let a ``pca`` artifact and a
        ``pca_whitened`` one collide on the same filename, so they could
        not coexist in a run and the stale one was silently reused
        (2026-09-10).
        """
        cases = {
            "pca": "resnet50_pca128.npy",
            "pca_whitened": "resnet50_pcaw128.npy",
            "random": "resnet50_rand128.npy",
        }
        for method, expected in cases.items():
            cfg = ProjectionConfig(method=method, dim=128, seed=42)

            assert projected_path(tmp_path / "resnet50.npy", cfg).name == expected

    def test_two_methods_of_one_width_do_not_collide(self, tmp_path):
        source = tmp_path / "resnet50.npy"
        plain = projected_path(source, ProjectionConfig(method="pca", dim=128, seed=42))
        whitened = projected_path(source, ProjectionConfig(method="pca_whitened", dim=128, seed=42))

        assert plain != whitened

    def test_token_precedes_the_finetuned_marker(self, tmp_path):
        """So fuse's `{extractor}{condition_suffix}` resolves in both conditions."""
        cfg = ProjectionConfig(method="pca", dim=128, seed=42)
        out = projected_path(tmp_path / "resnet50_finetuned.npy", cfg)

        assert out.name == "resnet50_pca128_finetuned.npy"


class TestNameClassification:
    """``src.utils.artifact_names`` owns the parse side of the token."""

    def test_every_method_token_is_recognised_as_projected(self):
        from src.utils.artifact_names import is_projected_artifact

        for name in ("resnet50_pca128", "resnet50_pcaw128", "resnet50_rand64"):
            assert is_projected_artifact(name), name

    def test_the_legacy_width_only_token_is_still_recognised(self):
        """Read-only compatibility, and the failure it prevents.

        A leftover ``resnet50_p128.npy`` is no longer written, but if one
        survives it must not be globbed up as a native backbone of its
        own -- it would enter the statistical families beside the real
        ResNet-50 as a separate backbone.
        """
        from src.utils.artifact_names import is_projected_artifact

        assert is_projected_artifact("resnet50_p128")

    def test_the_width_is_parsed_from_either_token(self):
        from src.utils.artifact_names import projection_dim

        assert projection_dim("resnet50_pcaw128") == 128
        assert projection_dim("resnet50_p128") == 128
        assert projection_dim("resnet50") is None

    def test_the_method_is_parsed_from_the_new_token_only(self):
        from src.utils.artifact_names import projection_method

        assert projection_method("resnet50_pcaw128") == "pca_whitened"
        assert projection_method("resnet50_pca128") == "pca"
        assert projection_method("resnet50_rand64") == "random"
        assert projection_method("resnet50_p128") is None
        assert projection_method("resnet50") is None

    def test_an_extractor_named_like_a_token_is_not_mistaken(self):
        from src.utils.artifact_names import is_projected_artifact

        assert not is_projected_artifact("clip_patch")
        assert not is_projected_artifact("hybrid_pca_nc128")


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


class TestPcaWhitenedProjection:
    def test_writes_the_requested_dim(self, native):
        cfg = ProjectionConfig(method="pca_whitened", dim=64)

        written = ensure_projected(native, cfg, train_items=list(range(150)))

        assert np.load(written).shape == (200, 64)

    def test_train_components_have_unit_variance(self, native):
        """Whitening's defining property, checked on the fit rows."""
        train = list(range(150))
        cfg = ProjectionConfig(method="pca_whitened", dim=16)

        projected = np.load(ensure_projected(native, cfg, train_items=train))

        variances = projected[train].var(axis=0, ddof=1)
        np.testing.assert_allclose(variances, np.ones(16), rtol=1e-3)

    def test_shares_the_pca_basis_up_to_scale(self, native):
        """pca_whitened is pca with rescaled columns — same directions."""
        train = list(range(150))
        plain = np.load(
            ensure_projected(native, ProjectionConfig(method="pca", dim=8), train_items=train)
        )

        second_source = native.with_name("vit_b16.npy")
        np.save(second_source, np.load(native))
        white = np.load(
            ensure_projected(
                second_source, ProjectionConfig(method="pca_whitened", dim=8), train_items=train
            )
        )

        # Column-wise correlation ±1: identical direction, different scale.
        for j in range(8):
            corr = np.corrcoef(plain[:, j], white[:, j])[0, 1]
            assert abs(abs(corr) - 1.0) < 1e-4

    def test_near_zero_variance_component_is_zeroed_not_amplified(self, tmp_path):
        """A direction the train set does not vary along must map to 0."""
        rng = np.random.default_rng(5)
        rows = rng.standard_normal((100, 8)).astype(np.float32)
        rows[:, 7] = 3.0  # constant column: zero variance in every basis
        path = tmp_path / "resnet50.npy"
        np.save(path, rows)
        cfg = ProjectionConfig(method="pca_whitened", dim=8)

        projected = np.load(ensure_projected(path, cfg, train_items=list(range(80))))

        assert np.isfinite(projected).all()
        # The dead direction survives as an all-zero component, not inf/NaN.
        variances = projected[:80].var(axis=0, ddof=1)
        assert (variances < 1e-6).sum() >= 1

    def test_without_a_fit_set_it_fails_loudly(self, native):
        with pytest.raises(ValueError, match="train-item"):
            ensure_projected(native, ProjectionConfig(method="pca_whitened", dim=64))

    def test_projector_metadata_records_the_method(self, native):
        cfg = ProjectionConfig(method="pca_whitened", dim=32)

        written = ensure_projected(native, cfg, train_items=list(range(150)))

        meta = json.loads(written.with_suffix(".proj.json").read_text())
        assert meta["method"] == "pca_whitened"
        assert meta["fit"] == "train items only"


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
            "np.save('%s', _random_matrix(512, 32, 5, 'resnet50_rand32'))"
        ) % (native.parent / "second.npy")
        subprocess.run([sys.executable, "-c", script], check=True)
        second = np.load(native.parent / "second.npy")

        np.testing.assert_array_equal(first, second)


class _NoBackbone:
    """Sentinel extractor: instantiating it means the cell re-extracted.

    A cell whose pooled artifact is already on disk must reach the
    projection without loading a backbone, so any construction here is
    itself the failure.
    """

    supports_components = False

    def __init__(self, *args, **kwargs):  # pragma: no cover - guard
        raise AssertionError("the backbone must not be instantiated for a projection-only cell")


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

        assert (
            _project_pooled(native, ProjectionConfig(method="random", dim=64, seed=42), None)
            is True
        )
        assert np.load(tmp_path / "resnet50_rand64.npy").shape == (30, 64)

    def test_the_sidecar_declares_the_projected_width(self, tmp_path):
        """Otherwise the loader's meta cross-check rejects the artifact."""
        from src.steps.extract import _project_pooled

        native = self._native(tmp_path)
        _project_pooled(native, ProjectionConfig(method="random", dim=64), None)

        meta = json.loads((tmp_path / "resnet50_rand64.meta.json").read_text())
        assert meta["native_dim"] == 64
        assert meta["source_native_dim"] == 512
        assert meta["name"] == "resnet50_rand64"
        assert meta["projection"] == {"method": "random", "dim": 64, "source": "resnet50.npy"}

    def test_the_sidecar_passes_the_loader_cross_check(self, tmp_path):
        from src.fusions.online import _validate_against_meta
        from src.steps.extract import _project_pooled

        native = self._native(tmp_path)
        _project_pooled(native, ProjectionConfig(method="random", dim=64), None)
        projected = tmp_path / "resnet50_rand64.npy"

        _validate_against_meta(projected, np.load(projected))

    def test_backbone_provenance_is_carried_over(self, tmp_path):
        """A projected artifact must still say which weights produced it."""
        from src.steps.extract import _project_pooled

        native = self._native(tmp_path)
        _project_pooled(native, ProjectionConfig(method="random", dim=64), None)

        meta = json.loads((tmp_path / "resnet50_rand64.meta.json").read_text())
        assert meta["weights_id"] == "IMAGENET1K_V2"

    def test_no_projection_configured_writes_nothing(self, tmp_path):
        from src.steps.extract import _project_pooled

        native = self._native(tmp_path)

        assert _project_pooled(native, None, None) is False
        assert list(tmp_path.glob("*_p*.npy")) == []

    def test_a_projection_of_another_recipe_is_not_reused_by_name(self, tmp_path):
        """The token encodes name and width, never the whole recipe.

        ``_extract_for_config`` used to decide by the mere EXISTENCE of
        the projected artifact.  The name cannot carry the fit set or
        the seed, so a cell configured for another recipe was skipped
        whole: ``ensure_projected`` never ran, and with it the
        provenance check that would have caught the mismatch.  The stale
        array then fed training as if it were the configured one
        (amazon_women probe, 2026-09-10).
        """
        from src.utils.identity import ArtifactProvenanceError

        self._native(tmp_path)
        fit = list(range(30))
        _extract_for_config(
            extractor_cls=_NoBackbone,
            extractor_name="resnet50",
            dataset_name=tmp_path.name,
            image_dir="",
            item_ids=[],
            embeddings_dir=str(tmp_path.parent),
            batch_size=1,
            checkpoint_every=1,
            device="cpu",
            config={},
            projection=ProjectionConfig(method="pca", dim=64, seed=42),
            train_items=fit,
        )
        assert (tmp_path / "resnet50_pca64.npy").exists()

        # The METHOD no longer collides -- it is in the name since
        # 2026-09-10, which removes that class outright.  What still
        # shares one path is everything else in the recipe, and the
        # skip predicate must not decide those by existence either.
        with pytest.raises(ArtifactProvenanceError, match="seed"):
            _extract_for_config(
                extractor_cls=_NoBackbone,
                extractor_name="resnet50",
                dataset_name=tmp_path.name,
                image_dir="",
                item_ids=[],
                embeddings_dir=str(tmp_path.parent),
                batch_size=1,
                checkpoint_every=1,
                device="cpu",
                config={},
                projection=ProjectionConfig(method="pca", dim=64, seed=7),
                train_items=fit,
            )

    def test_the_projector_sidecar_is_not_a_phantom_embedding(self, tmp_path):
        """`.proj.json` describes a projector; it is not an online fusion.

        The embedding glob reads `hybrid_*.json` as online-fusion
        sidecars.  Projecting a FUSION output (2026-09-10) put a
        `.proj.json` next to one for the first time, and every such file
        became an embedding named `hybrid_*_pcaw128.proj` whose jobs
        could only fail: "online sidecar ... lists no components; cannot
        stack".
        """
        from src.steps.extract import _project_pooled
        from src.steps.train import get_embedding_files

        dataset_dir = tmp_path / "amazon_men"
        dataset_dir.mkdir()
        fusion = dataset_dir / "hybrid_concat.npy"
        rng = np.random.default_rng(3)
        np.save(fusion, rng.standard_normal((30, 128)).astype(np.float32))
        _project_pooled(fusion, ProjectionConfig(method="random", dim=64, seed=42), None)

        stems = get_embedding_files(str(tmp_path), "amazon_men")

        assert (dataset_dir / "hybrid_concat_rand64.proj.json").exists(), "fixture guard"
        assert "hybrid_concat_rand64" in stems
        assert not any(stem.endswith(".proj") for stem in stems), stems

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

        assert "resnet50_rand128" in stems
        assert "vit_b16_rand128" in stems
        assert "resnet50" in stems
