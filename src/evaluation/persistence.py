"""Per-user persistence at final evaluation (Task F, publication E06/C05).

Every evaluated cell writes two artifacts under
``<results>/per_user/<dataset>/``:

* ``<cell_key>.csv.gz`` — one row per test user: ``user_idx``, ``rank``
  (held-out, 1-indexed, post-mask/post-tiebreak), ``n_candidates``,
  ``tie_block_size``, ``top_items`` (JSON list of the first 20 item_idx).
* ``<cell_key>.meta.json`` — the cell metadata contract below, plus the
  completion block of the generation it points at.

**Format choice:** csv.gz, not parquet.  parquet would pull in pyarrow
(a large binary dependency absent from the pinned environment); the
per-cell tables (tens–hundreds of thousands of rows) compress to a few
MB as gzip'd CSV, and staying stdlib-only keeps the reproducibility
lock small.  ``top_items`` is stored as a JSON string (CSV has no list
type) and parsed back on read.

**Publication (E06).**  A cell is published as an immutable *generation*
under ``<dataset>/.generations/<cell_key>/<generation_id>/`` holding
``records.csv.gz``, ``meta.json`` and ``manifest.json`` (payload
digests, row count, expected-user digest, identity, producer version).
The generation is fsynced, then the records file is linked into the
canonical path and finally the canonical ``.meta.json`` — the
*completion pointer* — is replaced atomically.  Readers accept a cell
as complete only when the pointer's completion block matches the
canonical records byte for byte; a crash at any boundary therefore
leaves either the previous complete generation, a mixed pair that every
reader rejects as torn, or an incomplete generation, never a silently
accepted mixture.  A per-cell lock file rejects concurrent publishers.

The metadata container is defined here (the contract); the evaluate step
fills the evaluation-side fields and the battery runner (Task I) fills
git/env/duration fields.
"""

from __future__ import annotations

import fcntl
import gzip
import json
import os
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.atomic_io import _fsync_dir, _fsync_path, atomic_write
from src.utils.checkpoint import file_digest
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Bumped when the evaluation protocol changes in a way that makes ranks
#: non-comparable across versions (masking, tie-break, selection).
EVAL_PROTOCOL_VERSION = "2.3"

#: Schema of the ``completion`` block a published ``.meta.json`` carries.
COMPLETION_SCHEMA_VERSION = 1

_RECORD_COLUMNS = ["user_idx", "rank", "n_candidates", "tie_block_size", "top_items"]
_GENERATIONS_DIR = ".generations"
_RECORDS_FILE = "records.csv.gz"
_META_FILE = "meta.json"
_MANIFEST_FILE = "manifest.json"

#: Publication boundaries, in order; tests inject faults after each one.
STAGE_PAYLOAD = "payload"
STAGE_METADATA = "metadata"
STAGE_RECORDS_PUBLISHED = "records_published"
STAGE_POINTER = "pointer"


class ArtifactIntegrityError(RuntimeError):
    """A per-user artifact is torn, incomplete or inconsistent with its pointer.

    Raised by the readers instead of accepting a mixed generation (a
    records file from one publication next to the metadata of another),
    a row count that disagrees with the completion block or a missing
    payload.
    """


class ConcurrentPublicationError(RuntimeError):
    """Another process is publishing the same cell right now."""


def _stage(name: str) -> None:
    """Publication boundary hook (no-op; fault-injection tests replace it)."""


@dataclass
class CellMetadata:
    """Metadata contract for one evaluated cell.

    Evaluate fills the identity + protocol fields; the runner (Task I)
    fills ``git_sha`` / ``git_dirty`` / ``env`` / ``gpu`` / ``durations``
    / ``config_hash`` (left at defaults here).

    ``fold`` records the K-fold provenance of the artifact and is ``None``
    for leave-one-out cells (every ``.meta.json`` written before the field
    existed reads back as ``None``).  Two shapes are used:

    * partial artifact of one fold (``folds/fold<i>/...``)::

          {"index": i, "k": K, "seed": <fold seed>, "n_users": n_i}

    * concatenated cell artifact (canonical location, no ``index``)::

          {"k": K, "seeds": [seed_0, ..., seed_{K-1}],
           "n_users_per_fold": [n_0, ..., n_{K-1}]}
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
    # K-fold provenance (see class docstring); None for leave-one-out cells.
    fold: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CellCompletion:
    """The validated completion block of a published cell (C05)."""

    generation_id: str
    records_sha256: str
    row_count: int
    expected_user_digest: str
    identity_digest: str | None
    producer_version: str | None
    published_at: str | None

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


def user_digest(user_ids: Iterable[int]) -> str:
    """SHA-256 of the sorted user ids (the ``expected_user_digest`` of a cell)."""
    from src.evaluation.paired_validation import user_population_digest

    return user_population_digest(np.asarray(list(user_ids), dtype=np.int64))


def _prepare_records(records: pd.DataFrame) -> pd.DataFrame:
    df = records.rename(columns={"user_id": "user_idx"}).copy()
    missing = [c for c in _RECORD_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"records missing columns {missing}; got {list(df.columns)}.")
    df = df[_RECORD_COLUMNS]
    df["top_items"] = df["top_items"].map(json.dumps)
    return df


def _write_durable(path: Path, write) -> None:
    write(path)
    _fsync_path(path)


def _publish_file(source: Path, destination: Path) -> None:
    """Expose an immutable generation file at *destination* atomically.

    Hard-links the generation file (same inode, no second copy) into a
    sibling temp name and ``os.replace``-s it over the destination; a
    filesystem without hard links gets a byte copy through
    :func:`atomic_write`.
    """
    tmp = destination.with_name(f"{destination.name}.{os.getpid()}.publish")
    try:
        tmp.unlink(missing_ok=True)
        os.link(source, tmp)
    except OSError:
        atomic_write(lambda t: Path(t).write_bytes(source.read_bytes()), destination)
        return
    try:
        os.replace(tmp, destination)
    finally:
        tmp.unlink(missing_ok=True)
    _fsync_dir(destination.parent)


class _CellLock:
    """Per-cell writer lock; a competing publisher is rejected, not queued."""

    def __init__(self, records_path: Path) -> None:
        self._path = records_path.with_name(f"{records_path.name}.lock")
        self._handle = None

    def __enter__(self) -> _CellLock:
        self._handle = open(self._path, "w")  # noqa: SIM115 — released in __exit__
        try:
            fcntl.flock(self._handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._handle.close()
            raise ConcurrentPublicationError(
                f"another process is publishing {self._path.name[: -len('.lock')]}"
            ) from exc
        return self

    def __exit__(self, *exc_info) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle, fcntl.LOCK_UN)
            self._handle.close()


def _producer_version() -> str:
    from src import __version__

    return __version__


def write_cell_artifact(
    records: pd.DataFrame,
    metadata: CellMetadata,
    out_dir: str | Path,
    *,
    expected_users: Iterable[int] | None = None,
    identity_digest: str | None = None,
) -> Path:
    """Publish the per-user records + metadata as one generation; return the records path.

    ``expected_users`` is the user population the cell was supposed to
    evaluate (its digest is recorded and checked against the records);
    ``None`` records the digest of the users actually present, which
    only proves self-consistency.  ``identity_digest`` is the C02
    evaluation identity the generation was produced under.
    """
    records_path, meta_path = artifact_paths(out_dir, metadata)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    df = _prepare_records(records)
    users = df["user_idx"].to_numpy(dtype=np.int64)
    expected = user_digest(expected_users) if expected_users is not None else user_digest(users)
    actual = user_digest(users)
    if actual != expected:
        raise ArtifactIntegrityError(
            f"{records_path.name}: records cover a different user population than "
            "expected; refusing to publish an incomplete cell."
        )

    with _CellLock(records_path):
        generation = _new_generation_dir(records_path)
        # 1. immutable payload
        gen_records = generation / _RECORDS_FILE
        _write_durable(gen_records, lambda p: df.to_csv(p, index=False, compression="gzip"))
        _stage(STAGE_PAYLOAD)
        # 2. metadata + manifest (digests computed from the committed bytes)
        completion = CellCompletion(
            generation_id=generation.name,
            records_sha256=file_digest(gen_records),
            row_count=int(len(df)),
            expected_user_digest=expected,
            identity_digest=identity_digest,
            producer_version=_producer_version(),
            published_at=datetime.now(UTC).isoformat(),
        )
        meta = {
            **metadata.to_dict(),
            "row_count": completion.row_count,
            "expected_user_digest": completion.expected_user_digest,
            "completion": {"schema_version": COMPLETION_SCHEMA_VERSION, **completion.to_dict()},
        }
        meta_text = json.dumps(meta, indent=2)
        _write_durable(generation / _META_FILE, lambda p: p.write_text(meta_text, encoding="utf-8"))
        manifest_text = json.dumps(
            {
                "schema_version": COMPLETION_SCHEMA_VERSION,
                "cell": records_path.name[: -len(".csv.gz")],
                "payloads": {_RECORDS_FILE: completion.records_sha256},
                **completion.to_dict(),
            },
            indent=2,
        )
        _write_durable(
            generation / _MANIFEST_FILE, lambda p: p.write_text(manifest_text, encoding="utf-8")
        )
        _fsync_dir(generation)
        _stage(STAGE_METADATA)
        # 3. expose the payload at its canonical path
        _publish_file(gen_records, records_path)
        _stage(STAGE_RECORDS_PUBLISHED)
        # 4. completion pointer
        _publish_file(generation / _META_FILE, meta_path)
        _stage(STAGE_POINTER)
    return records_path


def _new_generation_dir(records_path: Path) -> Path:
    key = records_path.name[: -len(".csv.gz")]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    generation = records_path.parent / _GENERATIONS_DIR / key / f"{stamp}-{uuid.uuid4().hex[:8]}"
    generation.mkdir(parents=True, exist_ok=False)
    return generation


def _meta_path_of(records_path: Path) -> Path:
    return records_path.with_name(records_path.name.replace(".csv.gz", ".meta.json"))


#: Keys a published ``.meta.json`` carries on top of :class:`CellMetadata`.
COMPLETION_KEYS: tuple[str, ...] = ("row_count", "expected_user_digest", "completion")


def contract_fields(metadata: dict) -> dict:
    """The :class:`CellMetadata` fields of a metadata dict, without the completion keys."""
    return {k: v for k, v in metadata.items() if k not in COMPLETION_KEYS}


def read_completion(metadata: dict) -> CellCompletion | None:
    """The completion block of a metadata dict, or ``None`` for a legacy artifact."""
    block = metadata.get("completion")
    if block is None:
        return None
    if not isinstance(block, dict) or block.get("schema_version") != COMPLETION_SCHEMA_VERSION:
        raise ArtifactIntegrityError(f"unsupported completion block {block!r}.")
    return CellCompletion(
        generation_id=str(block["generation_id"]),
        records_sha256=str(block["records_sha256"]),
        row_count=int(block["row_count"]),
        expected_user_digest=str(block["expected_user_digest"]),
        identity_digest=block.get("identity_digest"),
        producer_version=block.get("producer_version"),
        published_at=block.get("published_at"),
    )


def validate_cell_artifact(records_path: str | Path) -> CellCompletion | None:
    """Validate the canonical pair before anyone accepts it as complete.

    :returns: The completion block, or ``None`` for a legacy artifact
        (metadata without a completion block — identified, never
        assumed complete).
    :raises ArtifactIntegrityError: Missing payload/metadata, a pointer
        whose recorded digest differs from the canonical records file
        (torn / mixed generation), or unreadable metadata.
    """
    records_path = Path(records_path)
    meta_path = _meta_path_of(records_path)
    if not meta_path.exists():
        raise ArtifactIntegrityError(f"{records_path.name}: completion pointer missing.")
    if not records_path.exists():
        raise ArtifactIntegrityError(f"{records_path.name}: records payload missing.")
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ArtifactIntegrityError(f"{meta_path.name}: unreadable metadata ({exc}).") from exc
    completion = read_completion(metadata)
    if completion is None:
        logger.warning(
            "%s: legacy per-user artifact without a completion block; its completeness "
            "cannot be verified.",
            records_path.name,
        )
        return None
    actual = file_digest(records_path)
    if actual != completion.records_sha256:
        raise ArtifactIntegrityError(
            f"{records_path.name}: records digest {actual[:12]}... does not match the "
            f"completion pointer ({completion.records_sha256[:12]}...); the artifact is "
            "torn (payload and metadata from different generations)."
        )
    return completion


def read_cell_artifact(records_path: str | Path) -> tuple[dict, pd.DataFrame]:
    """Read ``(metadata dict, records DataFrame)`` for a cell.

    ``top_items`` is parsed back into a Python list.  The pair is
    validated first (:func:`validate_cell_artifact`); a torn artifact
    raises :class:`ArtifactIntegrityError`, a legacy one is returned
    with a warning.
    """
    records_path = Path(records_path)
    completion = validate_cell_artifact(records_path)
    metadata = json.loads(_meta_path_of(records_path).read_text(encoding="utf-8"))
    with gzip.open(records_path, "rt", encoding="utf-8") as fh:
        df = pd.read_csv(fh)
    df["top_items"] = df["top_items"].map(json.loads)
    if completion is not None and len(df) != completion.row_count:
        raise ArtifactIntegrityError(
            f"{records_path.name}: {len(df)} rows but the completion block declares "
            f"{completion.row_count}."
        )
    return metadata, df


def list_generations(records_path: str | Path) -> list[Path]:
    """Every generation directory ever published for a cell (never deleted here)."""
    records_path = Path(records_path)
    key = records_path.name[: -len(".csv.gz")]
    base = records_path.parent / _GENERATIONS_DIR / key
    return sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
