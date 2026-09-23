"""The projection pass over OFFLINE fusion outputs.

The NPR paper reduces the visual feature offline, before the model sees
it, and VNPR is the only recommender here without a learned projection
to absorb the input scale (Niu et al. WSDM 2018, section 5).  For a
single backbone the reduction is written by the extract step; for a
fusion it has to happen after the fusion, over the vector the model
actually receives -- whitening each source and then mixing them is a
different transform from whitening the mixture.

Only offline fusions have an artifact to project.  With
``alignment: learned`` most strategies are JSON sidecars whose fusion
runs inside the recommender at train time; those carry no array here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.extractors.projection import ProjectionConfig, projected_path


@pytest.fixture
def dataset_dir(tmp_path) -> Path:
    """One offline fusion, one online sidecar, one component artifact."""
    root = tmp_path / "embeddings" / "amazon_men"
    root.mkdir(parents=True)
    rng = np.random.default_rng(0)
    np.save(root / "hybrid_concat.npy", rng.standard_normal((40, 96)).astype(np.float32))
    (root / "hybrid_mean_learned_D128.json").write_text(json.dumps({"strategy": "mean"}))
    np.save(root / "hybrid_concat_comp.npy", rng.standard_normal((40, 4, 96)).astype(np.float32))
    return root


class TestFusionOutputProjection:
    def test_an_offline_fusion_is_projected(self, dataset_dir):
        from src.steps.fuse import project_fusion_outputs

        cfg = ProjectionConfig(method="pca_whitened", dim=32, seed=42)

        project_fusion_outputs(dataset_dir, cfg, train_items=list(range(40)))

        out = projected_path(dataset_dir / "hybrid_concat.npy", cfg)
        assert out.exists()
        assert np.load(out).shape == (40, 32)

    def test_an_online_sidecar_has_no_array_to_project(self, dataset_dir):
        from src.steps.fuse import project_fusion_outputs

        cfg = ProjectionConfig(method="pca_whitened", dim=32, seed=42)

        project_fusion_outputs(dataset_dir, cfg, train_items=list(range(40)))

        assert not list(dataset_dir.glob("hybrid_mean_learned_D128_pcaw32*"))

    def test_a_component_fusion_is_left_alone(self, dataset_dir):
        """Component artifacts are (n_items, R, D); the projector reads rows."""
        from src.steps.fuse import project_fusion_outputs

        cfg = ProjectionConfig(method="pca_whitened", dim=32, seed=42)

        project_fusion_outputs(dataset_dir, cfg, train_items=list(range(40)))

        assert not list(dataset_dir.glob("hybrid_concat_comp_pcaw32*"))

    def test_no_projection_configured_writes_nothing(self, dataset_dir):
        from src.steps.fuse import project_fusion_outputs

        project_fusion_outputs(dataset_dir, None, train_items=None)

        assert not list(dataset_dir.glob("*pcaw*"))
        assert not list(dataset_dir.glob("*pca32*"))

    def test_it_runs_even_when_every_fusion_already_existed(self, dataset_dir, monkeypatch):
        """The early return of `run()` must not skip the projection.

        `run()` returns as soon as nothing is pending, so a projection
        hooked after the worker pool would never fire on a rerun -- the
        one-path-recovery shape that produced several defects in this
        pipeline (2026-09-09).
        """
        import src.steps.fuse as fuse

        calls: list[Path] = []
        monkeypatch.setattr(
            fuse, "project_fusion_outputs", lambda d, cfg, train_items: calls.append(d)
        )
        monkeypatch.setattr(fuse, "_collect_fusion_tasks", lambda *a, **k: [])
        monkeypatch.setattr(fuse, "gate_backbone_features", lambda *a, **k: None, raising=False)

        cfg = {
            "paths": {"embeddings": str(dataset_dir.parent), "data_processed": "unused"},
            "datasets": ["amazon_men"],
            "fusion_strategies_enabled": ["concat"],
            # `random` needs no fit set, so the assertion stays on the
            # hook firing rather than on a fixture of train splits.
            "projection": {"method": "random", "dim": 32, "seed": 42},
        }
        monkeypatch.setattr(fuse, "load_config", lambda: cfg)

        fuse.run("frozen")

        assert calls, "projection must run on the all-cached path too"
