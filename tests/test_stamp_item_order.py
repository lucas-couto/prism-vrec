"""``scripts/stamp_item_order.py`` nests the digest where the validator reads it.

Its first version (2026-09-06) merged the block's keys into the top
level of ``<stem>.meta.json``; every artifact stayed "alignment
unverified" and carried three stray keys.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from src.steps.validate_features import ALIGNMENT_VERIFIED, verify_item_order
from src.utils.item_order import canonical_item_order

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "stamp_item_order.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("stamp_item_order", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dataset(tmp_path: Path, *, insertion_reversed: bool, stray_keys: bool) -> tuple[Path, Path]:
    n = 6
    processed = tmp_path / "processed" / "toy"
    processed.mkdir(parents=True)
    ids = [str(i) for i in range(n)]
    order = list(reversed(range(n))) if insertion_reversed else list(range(n))
    processed.joinpath("item2idx.json").write_text(json.dumps({str(i): i for i in order}))
    emb = tmp_path / "embeddings" / "toy"
    emb.mkdir(parents=True)
    np.save(emb / "resnet50.npy", np.zeros((n, 3), dtype=np.float32))
    emb.joinpath("resnet50_ids.json").write_text(json.dumps(ids))
    meta = {"name": "resnet50", "kind": "pooled"}
    if stray_keys:
        meta.update({"schema_version": 1, "n_items": n, "digest": "stale"})
    emb.joinpath("resnet50.meta.json").write_text(json.dumps(meta))
    return emb, processed.parent


def test_stamp_nests_the_block_and_the_validator_accepts_it(tmp_path):
    emb, processed = _dataset(tmp_path, insertion_reversed=True, stray_keys=True)
    script = _load_script()

    lines = script._stamp_dataset(emb, processed)

    meta = json.loads((emb / "resnet50.meta.json").read_text())
    assert lines == ["toy/resnet50: stamped (6 rows)"]
    assert set(meta) == {"name", "kind", "item_order"}
    expected = canonical_item_order(json.loads((processed / "toy" / "item2idx.json").read_text()))
    assert verify_item_order(emb / "resnet50.npy", label="toy/resnet50", expected_ids=expected) == (
        ALIGNMENT_VERIFIED
    )
    assert script._stamp_dataset(emb, processed) == ["toy/resnet50: already stamped"]


def test_unprovable_order_is_left_unstamped(tmp_path):
    emb, processed = _dataset(tmp_path, insertion_reversed=False, stray_keys=False)
    (emb / "resnet50_ids.json").write_text(json.dumps(["5", "4", "3", "2", "1", "0"]))
    script = _load_script()

    lines = script._stamp_dataset(emb, processed)

    assert "NOT PROVABLE" in lines[0]
    assert "item_order" not in json.loads((emb / "resnet50.meta.json").read_text())
