"""Tests that component artifacts are routed only to component-consuming models.

Guards the additive ``requires_components`` enumeration split in
``src/steps/train.py``: ACF trains on ``*_comp`` stems, every other
recommender keeps its pooled pool unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.steps.train import build_job_list, is_component_artifact

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
