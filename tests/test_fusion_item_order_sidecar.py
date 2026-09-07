"""Offline fusions inherit the item-order digest of their sources.

A fusion is row-wise over its inputs, so its rows are its sources' rows;
before this change the fuse step wrote provenance but no ``item_order``
sidecar, and ``validate_features`` reported every fresh ``hybrid_*.npy``
as a legacy artifact ("re-extract to prove row i == item_idx i").
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.steps.fuse import _fuse_single
from src.steps.validate_features import (
    ALIGNMENT_UNVERIFIED,
    ALIGNMENT_VERIFIED,
    verify_item_order,
)
from src.utils.item_order import item_order_metadata

N = 12
IDS = [str(i) for i in range(N)]


def _source(tmp_path: Path, name: str, dim: int, *, stamped: bool, ids=IDS) -> str:
    path = tmp_path / f"{name}.npy"
    np.save(path, np.random.default_rng(dim).standard_normal((N, dim)).astype(np.float32))
    meta = {"name": name, "kind": "pooled"}
    if stamped:
        meta["item_order"] = item_order_metadata(ids)
    (tmp_path / f"{name}.meta.json").write_text(json.dumps(meta))
    return str(path)


@pytest.mark.parametrize("strategy", ["concat", "pca", "mean"])
def test_fusion_inherits_a_common_digest(tmp_path: Path, strategy: str) -> None:
    dim = 6 if strategy == "mean" else 4
    sources = [_source(tmp_path, "a", dim, stamped=True), _source(tmp_path, "b", 6, stamped=True)]
    out = tmp_path / f"hybrid_{strategy}.npy"
    kwargs = {"n_components": 3} if strategy == "pca" else {}

    _fuse_single(strategy, str(out), sources, True, train_items=list(range(N)), **kwargs)

    meta = json.loads((tmp_path / f"hybrid_{strategy}.meta.json").read_text())
    assert meta["item_order"] == item_order_metadata(IDS)
    assert verify_item_order(out, label="t", expected_ids=IDS) == ALIGNMENT_VERIFIED


def test_fusion_stays_unverified_when_a_source_is_unstamped(tmp_path: Path) -> None:
    sources = [_source(tmp_path, "a", 4, stamped=True), _source(tmp_path, "b", 4, stamped=False)]
    out = tmp_path / "hybrid_concat.npy"

    _fuse_single("concat", str(out), sources, True, train_items=list(range(N)))

    assert not (tmp_path / "hybrid_concat.meta.json").exists()
    assert verify_item_order(out, label="t", expected_ids=IDS) == ALIGNMENT_UNVERIFIED


def test_fusion_stays_unverified_when_sources_disagree(tmp_path: Path) -> None:
    other = list(reversed(IDS))
    sources = [
        _source(tmp_path, "a", 4, stamped=True),
        _source(tmp_path, "b", 4, stamped=True, ids=other),
    ]
    out = tmp_path / "hybrid_concat.npy"

    _fuse_single("concat", str(out), sources, True, train_items=list(range(N)))

    assert not (tmp_path / "hybrid_concat.meta.json").exists()


def test_reused_fusion_gets_the_sidecar_on_the_next_run(tmp_path: Path) -> None:
    from src.steps.fuse import _inherit_item_order

    sources = [_source(tmp_path, "a", 4, stamped=True), _source(tmp_path, "b", 4, stamped=True)]
    out = tmp_path / "hybrid_concat.npy"
    np.save(out, np.zeros((N, 8), dtype=np.float32))  # written before the sidecar existed

    _inherit_item_order(out, sources, "concat")
    first = (tmp_path / "hybrid_concat.meta.json").read_text()
    _inherit_item_order(out, sources, "concat")

    assert (tmp_path / "hybrid_concat.meta.json").read_text() == first
    assert verify_item_order(out, label="t", expected_ids=IDS) == ALIGNMENT_VERIFIED
