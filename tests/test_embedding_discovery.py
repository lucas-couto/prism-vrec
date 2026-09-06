"""Provenance sidecars are never mistaken for embeddings.

``<artifact>.provenance.json`` (E05) is written next to every fusion
output, including the ``hybrid_*.json`` online sidecars.  On 2026-09-06
the discovery glob turned ``hybrid_x.json.provenance.json`` into a
phantom embedding ``hybrid_x.json.provenance`` and 92 battery jobs
failed on "sidecar lists no components".
"""

from __future__ import annotations

import json

import numpy as np

from src.steps.train import get_embedding_files
from src.utils.identity import provenance_path


def _write(tmp_path, dataset: str) -> None:
    d = tmp_path / dataset
    d.mkdir()
    np.save(d / "resnet50.npy", np.zeros((3, 4), dtype=np.float32))
    np.save(d / "hybrid_concat.npy", np.zeros((3, 8), dtype=np.float32))
    (d / "hybrid_sum_learned_D128.json").write_text(json.dumps({"components": ["resnet50.npy"]}))
    for artifact in ("hybrid_concat.npy", "hybrid_sum_learned_D128.json"):
        provenance_path(d / artifact).write_text(json.dumps({"kind": "fusion"}))
    (d / "resnet50.meta.json").write_text("{}")
    (d / "resnet50_ids.json").write_text("[]")


def test_provenance_sidecars_are_not_embeddings(tmp_path):
    _write(tmp_path, "amazon_fashion")

    stems = get_embedding_files(str(tmp_path), "amazon_fashion")

    assert stems == ["hybrid_concat", "hybrid_sum_learned_D128", "resnet50"]
    assert not any("provenance" in s for s in stems)
