"""Stamp a provable ``item_order`` block onto legacy extractor sidecars.

Feature artifacts written before 3.0.0rc1 carry no ``item_order`` digest,
so ``validate_features`` reports them as *alignment unverified*.  Every
extraction also wrote ``<stem>_ids.json``: the item ids in the exact row
order the extractor processed.  When that recorded order equals the
canonical order derived from ``item2idx.json`` (row i == item whose mapped
value is i) and the matrix has exactly that many rows, alignment is proven
by the artifact's own record and the digest can be added without
re-extracting.  Anything that does not match is left untouched and
reported; nothing is ever relabelled on faith.

Usage (inside the container)::

    python scripts/stamp_item_order.py --embeddings data/embeddings --processed data/processed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.utils.atomic_io import atomic_write
from src.utils.item_order import canonical_item_order, item_order_metadata


def _write_json(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2)
    atomic_write(lambda tmp: Path(tmp).write_text(text, encoding="utf-8"), path)


def _stamp_dataset(embeddings_dir: Path, processed_dir: Path) -> list[str]:
    dataset = embeddings_dir.name
    item2idx = json.loads((processed_dir / dataset / "item2idx.json").read_text(encoding="utf-8"))
    expected = canonical_item_order(item2idx)
    lines: list[str] = []
    for meta_path in sorted(embeddings_dir.glob("*.meta.json")):
        stem = meta_path.name[: -len(".meta.json")]
        npy_path = embeddings_dir / f"{stem}.npy"
        ids_path = embeddings_dir / f"{stem}_ids.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if "item_order" in meta:
            lines.append(f"{dataset}/{stem}: already stamped")
            continue
        if not npy_path.exists() or not ids_path.exists():
            lines.append(f"{dataset}/{stem}: SKIPPED (missing .npy or _ids.json)")
            continue
        recorded = [str(i) for i in json.loads(ids_path.read_text(encoding="utf-8"))]
        n_rows = int(np.load(npy_path, mmap_mode="r").shape[0])
        if recorded != expected or n_rows != len(expected):
            lines.append(
                f"{dataset}/{stem}: NOT PROVABLE (rows={n_rows}, ids={len(recorded)}, "
                f"expected={len(expected)}, order_equal={recorded == expected}) — left unstamped"
            )
            continue
        meta.update(item_order_metadata(expected))
        _write_json(meta_path, meta)
        lines.append(f"{dataset}/{stem}: stamped ({n_rows} rows)")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--embeddings", default="data/embeddings")
    parser.add_argument("--processed", default="data/processed")
    args = parser.parse_args()
    for dataset_dir in sorted(Path(args.embeddings).iterdir()):
        if dataset_dir.is_dir():
            print("\n".join(_stamp_dataset(dataset_dir, Path(args.processed))))


if __name__ == "__main__":
    main()
