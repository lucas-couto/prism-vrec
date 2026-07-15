"""Per-user persistence at final evaluation (Task F).

Every evaluated cell writes two artifacts under
``<results>/per_user/<dataset>/``:

* ``<cell_key>.csv.gz`` — one row per test user: ``user_idx``, ``rank``
  (held-out, 1-indexed, post-mask/post-tiebreak), ``n_candidates``,
  ``tie_block_size``, ``top_items`` (JSON list of the first 20 item_idx).
* ``<cell_key>.meta.json`` — the cell metadata contract below.

**Format choice:** csv.gz, not parquet.  parquet would pull in pyarrow
(a large binary dependency absent from the pinned environment); the
per-cell tables (tens–hundreds of thousands of rows) compress to a few
MB as gzip'd CSV, and staying stdlib-only keeps the reproducibility
lock small.  ``top_items`` is stored as a JSON string (CSV has no list
type) and parsed back on read.

The metadata container is defined here (the contract); the evaluate step
fills the evaluation-side fields and the battery runner (Task I) fills
git/env/duration fields.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

#: Bumped when the evaluation protocol changes in a way that makes ranks
#: non-comparable across versions (masking, tie-break, selection).
EVAL_PROTOCOL_VERSION = "2.3"

_RECORD_COLUMNS = ["user_idx", "rank", "n_candidates", "tie_block_size", "top_items"]


@dataclass
class CellMetadata:
    """Metadata contract for one evaluated cell.

    Evaluate fills the identity + protocol fields; the runner (Task I)
    fills ``git_sha`` / ``git_dirty`` / ``env`` / ``gpu`` / ``durations``
    / ``config_hash`` (left at defaults here).
    """

    dataset: str
    visual_config: str  # extractor | fusion name | "none" (BPR)
    recommender: str
    seed: int
    d: int  # latent dim
    split: str  # "test"
    eval_protocol_version: str = EVAL_PROTOCOL_VERSION
    n_users: int = 0
    n_items: int = 0
    # Runner-filled (Task I); defaults keep the contract valid standalone.
    git_sha: str | None = None
    git_dirty: bool | None = None
    config_hash: str | None = None
    timestamp: str | None = None
    durations: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)
    gpu: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def cell_key(dataset: str, visual_config: str, recommender: str, seed: int) -> str:
    """Deterministic, filesystem-safe artifact key for a cell."""
    parts = [dataset, visual_config, recommender, f"seed{seed}"]
    safe = ["".join(c if c.isalnum() or c in "-." else "_" for c in p) for p in parts]
    return "__".join(safe)


def _dir_for(out_dir: str | Path, dataset: str) -> Path:
    return Path(out_dir) / "per_user" / dataset


def artifact_paths(out_dir: str | Path, metadata: CellMetadata) -> tuple[Path, Path]:
    """Return ``(records_csv_gz, meta_json)`` paths for a cell."""
    key = cell_key(metadata.dataset, metadata.visual_config, metadata.recommender, metadata.seed)
    base = _dir_for(out_dir, metadata.dataset)
    return base / f"{key}.csv.gz", base / f"{key}.meta.json"


def write_cell_artifact(
    records: pd.DataFrame,
    metadata: CellMetadata,
    out_dir: str | Path,
) -> Path:
    """Write the per-user records + metadata; return the records path."""
    records_path, meta_path = artifact_paths(out_dir, metadata)
    records_path.parent.mkdir(parents=True, exist_ok=True)

    df = records.rename(columns={"user_id": "user_idx"}).copy()
    missing = [c for c in _RECORD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"records missing columns {missing}; got {list(df.columns)}.")
    df = df[_RECORD_COLUMNS]
    df["top_items"] = df["top_items"].map(json.dumps)
    df.to_csv(records_path, index=False, compression="gzip")

    meta_path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")
    return records_path


def read_cell_artifact(records_path: str | Path) -> tuple[dict, pd.DataFrame]:
    """Read ``(metadata dict, records DataFrame)`` for a cell.

    ``top_items`` is parsed back into a Python list.
    """
    records_path = Path(records_path)
    meta_path = records_path.with_name(records_path.name.replace(".csv.gz", ".meta.json"))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    with gzip.open(records_path, "rt", encoding="utf-8") as fh:
        df = pd.read_csv(fh)
    df["top_items"] = df["top_items"].map(json.loads)
    return metadata, df
