"""Projection and fusion reuse validate their ingredients, not just the output name (E05, Q13).

An existing projected / fused artifact is reused only when its
provenance record (source content, fit set, recipe, dimensions, seed)
equals what the current call would produce.  Any changed ingredient is
refused; identical content moved to another directory is reused; a
legacy artifact without a record is reused unverified with a warning.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

import src.steps.fuse as fuse_mod
from src.extractors.projection import ProjectionConfig, ensure_projected, projected_path
from src.utils.identity import (
    PROVENANCE_LEGACY,
    ArtifactProvenanceError,
    check_provenance,
    clear_identity_cache,
    provenance_path,
    read_provenance,
)

N, D = 40, 8
TRAIN = list(range(0, N, 2))


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_identity_cache()
    yield
    clear_identity_cache()


def _native(root: Path, name: str = "resnet50", *, seed: int = 0, dim: int = D) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.npy"
    np.save(path, np.random.default_rng(seed).standard_normal((N, dim)).astype(np.float32))
    return path


class TestProjectionReuse:
    def test_second_call_reuses_a_verified_artifact(self, tmp_path) -> None:
        source = _native(tmp_path)
        cfg = ProjectionConfig(method="pca", dim=4)

        first = ensure_projected(source, cfg, TRAIN)
        second = ensure_projected(source, cfg, TRAIN)

        assert first is not None and second is None
        record = read_provenance(projected_path(source, 4))
        assert record["kind"] == "projection" and record["fit_set_digest"]

    def test_changed_source_content_is_refused(self, tmp_path) -> None:
        source = _native(tmp_path)
        cfg = ProjectionConfig(method="random", dim=4)
        ensure_projected(source, cfg, None)
        clear_identity_cache()
        _native(tmp_path, seed=1)

        with pytest.raises(ArtifactProvenanceError, match="source"):
            ensure_projected(source, cfg, None)

    def test_changed_fit_set_is_refused(self, tmp_path) -> None:
        source = _native(tmp_path)
        cfg = ProjectionConfig(method="pca", dim=4)
        ensure_projected(source, cfg, TRAIN)

        with pytest.raises(ArtifactProvenanceError, match="fit_set_digest"):
            ensure_projected(source, cfg, TRAIN[:-1])

    def test_changed_seed_is_refused(self, tmp_path) -> None:
        source = _native(tmp_path)
        ensure_projected(source, ProjectionConfig(method="random", dim=4, seed=1), None)

        with pytest.raises(ArtifactProvenanceError, match="seed"):
            ensure_projected(source, ProjectionConfig(method="random", dim=4, seed=2), None)

    def test_changed_method_is_refused(self, tmp_path) -> None:
        source = _native(tmp_path)
        ensure_projected(source, ProjectionConfig(method="pca", dim=4), TRAIN)

        with pytest.raises(ArtifactProvenanceError, match="method"):
            ensure_projected(source, ProjectionConfig(method="pca_whitened", dim=4), TRAIN)

    def test_identical_content_moved_elsewhere_is_reused(self, tmp_path) -> None:
        source = _native(tmp_path / "a")
        cfg = ProjectionConfig(method="pca", dim=4)
        ensure_projected(source, cfg, TRAIN)
        shutil.copytree(tmp_path / "a", tmp_path / "b")

        assert ensure_projected(tmp_path / "b" / "resnet50.npy", cfg, TRAIN) is None

    def test_legacy_artifact_without_record_is_reused_unverified(self, tmp_path) -> None:
        from src.extractors.projection import projection_provenance

        source = _native(tmp_path)
        cfg = ProjectionConfig(method="random", dim=4)
        ensure_projected(source, cfg, None)
        output = projected_path(source, 4)
        provenance_path(output).unlink()

        status = check_provenance(output, projection_provenance(source, cfg, None), label="p")

        assert status == PROVENANCE_LEGACY
        assert ensure_projected(source, cfg, None) is None
        assert output.exists()


def _task(paths: list[Path], out: Path, **extra) -> dict:
    task = {
        "strategy_name": "concat",
        "output_path": str(out),
        "emb_list_paths": [str(p) for p in paths],
        "normalize": True,
        **extra,
    }
    task["provenance"] = fuse_mod.task_provenance(task)
    return task


class TestFusionReuse:
    def _sources(self, root: Path) -> list[Path]:
        return [_native(root, "resnet50", seed=0), _native(root, "vit_b16", seed=1, dim=D + 2)]

    def test_output_with_matching_record_is_reused(self, tmp_path) -> None:
        paths = self._sources(tmp_path)
        task = _task(paths, tmp_path / "hybrid_concat.npy")

        assert not fuse_mod._reusable(task)
        assert fuse_mod._fuse_single(**task)
        assert fuse_mod._reusable(_task(paths, tmp_path / "hybrid_concat.npy"))

    def test_changed_source_content_is_refused(self, tmp_path) -> None:
        paths = self._sources(tmp_path)
        fuse_mod._fuse_single(**_task(paths, tmp_path / "hybrid_concat.npy"))
        clear_identity_cache()
        _native(tmp_path, "vit_b16", seed=5, dim=D + 2)

        with pytest.raises(ArtifactProvenanceError, match="sources"):
            fuse_mod._reusable(_task(paths, tmp_path / "hybrid_concat.npy"))

    def test_changed_normalisation_is_refused(self, tmp_path) -> None:
        paths = self._sources(tmp_path)
        fuse_mod._fuse_single(**_task(paths, tmp_path / "hybrid_concat.npy"))

        with pytest.raises(ArtifactProvenanceError, match="normalize"):
            fuse_mod._reusable(_task(paths, tmp_path / "hybrid_concat.npy", normalize=False))

    def test_changed_fit_set_is_refused_for_pca(self, tmp_path) -> None:
        paths = self._sources(tmp_path)
        out = tmp_path / "hybrid_pca.npy"
        fuse_mod._fuse_single(
            **_task(paths, out, strategy_name="pca", train_items=TRAIN, n_components=4)
        )

        with pytest.raises(ArtifactProvenanceError, match="fit_set_digest"):
            fuse_mod._reusable(
                _task(paths, out, strategy_name="pca", train_items=TRAIN[1:], n_components=4)
            )

    def test_sidecar_recipe_change_is_refused(self, tmp_path) -> None:
        paths = self._sources(tmp_path)
        out = tmp_path / "hybrid_mean_learned_D4.json"
        sidecar = {
            "strategy": "mean",
            "online": True,
            "alignment": "learned",
            "dim": 4,
            "components": [p.name for p in paths],
            "normalize": True,
        }
        fuse_mod._fuse_single(**_task(paths, out, strategy_name="mean", sidecar_payload=sidecar))

        changed = {**sidecar, "dim": 8}
        with pytest.raises(ArtifactProvenanceError, match="sidecar"):
            fuse_mod._reusable(_task(paths, out, strategy_name="mean", sidecar_payload=changed))

    def test_legacy_output_is_reused_unverified(self, tmp_path) -> None:
        paths = self._sources(tmp_path)
        out = tmp_path / "hybrid_concat.npy"
        fuse_mod._fuse_single(**_task(paths, out))
        provenance_path(out).unlink()

        assert (
            check_provenance(out, _task(paths, out)["provenance"], label="f") == PROVENANCE_LEGACY
        )
        assert fuse_mod._reusable(_task(paths, out))

    def test_provenance_is_written_before_the_output(self, tmp_path, monkeypatch) -> None:
        paths = self._sources(tmp_path)
        out = tmp_path / "hybrid_concat.npy"

        def _boom(*a, **k):
            raise RuntimeError("crash before the output is committed")

        monkeypatch.setattr(fuse_mod, "run_streamed", _boom)
        with pytest.raises(RuntimeError):
            fuse_mod._fuse_single(**_task(paths, out))

        assert provenance_path(out).exists() and not out.exists()
        assert not fuse_mod._reusable(_task(paths, out))
