"""Canonical scientific identity (C02, schema version 2).

An experiment is identified by *what* it computes, never by *where* or
*how fast*: the dataset (its item/user mappings and split files), the
feature artifact (content plus the recipe and content of every source it
was built from), the registered model and its implementation version,
the effective hyperparameters, the selection budget, the seed, the
evaluation protocol and the condition/fold.  Block sizes, worker
counts, devices, filesystem roots and wall-clock time are execution
metadata and are deliberately absent from every payload built here.

The identity of a payload is the SHA-256 of its canonical JSON form:
sorted object keys, compact separators, ASCII output, finite numbers
only.  Type normalisation is strict and documented by
``tests/test_identity.py``: booleans stay booleans, integers stay
integers, floats stay floats (``1`` and ``1.0`` are different values),
NumPy scalars/arrays and tuples are converted to their Python
equivalents, ``Path`` becomes its POSIX string, ``NaN``/``Inf`` and any
other type raise :class:`IdentityError` instead of being stringified.

Large files are hashed by streaming (:func:`stream_digest`); ordered
mappings (``item2idx.json``, ``user2idx.json``) are hashed by numeric
index (:func:`mapping_digest`), so the JSON insertion order never
changes the identity.  Modification times are never used as identity;
they only key an in-process cache in front of the content digests.
"""

from __future__ import annotations

import hashlib
import json
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.checkpoint import file_digest
from src.utils.item_order import canonical_item_order, item_order_digest

#: Schema version of every payload produced by :func:`experiment_identity`.
IDENTITY_SCHEMA_VERSION = 2

#: Split files that give a *training/selection* run its data identity.
#: The final evaluation adds the held-out test split explicitly.
SELECTION_SPLITS: tuple[str, ...] = ("train", "val")
EVALUATION_SPLITS: tuple[str, ...] = ("train", "val", "test")

_SIDECAR_RECIPE_KEYS = ("strategy", "online", "alignment", "dim", "normalize", "fusion_kwargs")


class IdentityError(ValueError):
    """A payload cannot be canonicalised or a file's identity cannot be read."""


# ---------------------------------------------------------------------
# Canonical JSON
# ---------------------------------------------------------------------


def _normalize(value: Any, path: str) -> Any:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        if not np.isfinite(number):
            raise IdentityError(f"non-finite number at {path}: {value!r}")
        return number
    if isinstance(value, np.ndarray):
        return [_normalize(v, f"{path}[{i}]") for i, v in enumerate(value.tolist())]
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise IdentityError(f"non-string key at {path}: {key!r}")
            out[key] = _normalize(item, f"{path}.{key}")
        return out
    if isinstance(value, Sequence | set | frozenset):
        items = sorted(value) if isinstance(value, set | frozenset) else list(value)
        return [_normalize(v, f"{path}[{i}]") for i, v in enumerate(items)]
    raise IdentityError(f"unsupported value type at {path}: {type(value).__name__}")


def canonical_json(payload: Any) -> str:
    """Canonical JSON text of *payload* (sorted keys, compact, finite numbers).

    :raises IdentityError: On non-finite numbers, non-string keys or a
        value type without a canonical form.
    """
    return json.dumps(
        _normalize(payload, "$"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def canonical_digest(payload: Any) -> str:
    """SHA-256 hex digest of :func:`canonical_json`."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------
# Content digests (cached per process by path + size + mtime)
# ---------------------------------------------------------------------

_CACHE: dict[tuple[str, int, int], Any] = {}


def _cache_key(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))


def _cached(path: Path, compute) -> Any:
    key = _cache_key(path)
    if key not in _CACHE:
        _CACHE[key] = compute()
    return _CACHE[key]


def clear_identity_cache() -> None:
    """Drop the in-process digest cache (tests that rewrite files in place)."""
    _CACHE.clear()


def stream_digest(path: str | Path) -> str:
    """SHA-256 of the bytes of *path*, streamed; cached per process."""
    file = Path(path)
    if not file.is_file():
        raise IdentityError(f"cannot digest missing file {file}")
    return _cached(file, lambda: file_digest(file))


def mapping_digest(mapping: Mapping[str, Any]) -> str:
    """Digest of an ``id -> index`` mapping ordered by its numeric index.

    Insertion order is irrelevant: ``{"b": 1, "a": 0}`` and
    ``{"a": 0, "b": 1}`` share one digest.  Holes, duplicates and
    non-integer indices raise :class:`IdentityError`.
    """
    try:
        return item_order_digest(canonical_item_order(mapping))
    except ValueError as exc:
        raise IdentityError(str(exc)) from exc


def mapping_file_digest(path: str | Path) -> tuple[int, str]:
    """``(n_entries, digest)`` of a JSON ``id -> index`` mapping file."""
    file = Path(path)
    if not file.is_file():
        raise IdentityError(f"mapping file missing: {file}")

    def _compute() -> tuple[int, str]:
        mapping = json.loads(file.read_text(encoding="utf-8"))
        return len(mapping), mapping_digest(mapping)

    return _cached(file, _compute)


def split_digest(processed_dir: str | Path, dataset: str, splits: Sequence[str]) -> str:
    """Digest of the named split files of *dataset* (streamed bytes)."""
    base = Path(processed_dir) / dataset
    return canonical_digest({name: stream_digest(base / f"{name}.csv") for name in splits})


def _npy_header(path: Path) -> tuple[list[int], str]:
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        reader = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0
        )
        shape, _fortran, dtype = reader(handle)
    return [int(s) for s in shape], str(np.dtype(dtype))


def feature_recipe(path: str | Path) -> dict[str, Any]:
    """Recursive content + recipe description of a feature artifact.

    A ``.npy`` contributes its shape, dtype and streamed digest; an
    online-fusion ``.json`` sidecar contributes its recipe keys and, in
    declared order, the recipe of every component it references.  The
    artifact's own file name is part of the recipe (it names the
    extractor / fusion); its directory is not.
    """
    file = Path(path)
    if not file.is_file():
        raise IdentityError(f"feature artifact missing: {file}")
    if file.suffix == ".json":
        return _cached(file, lambda: _sidecar_recipe(file))
    if file.suffix == ".npy":
        return _cached(file, lambda: _npy_recipe(file))
    raise IdentityError(f"unsupported feature artifact type: {file}")


def _npy_recipe(file: Path) -> dict[str, Any]:
    shape, dtype = _npy_header(file)
    return {
        "kind": "npy",
        "name": file.name,
        "shape": shape,
        "dtype": dtype,
        "sha256": file_digest(file),
    }


def _sidecar_recipe(file: Path) -> dict[str, Any]:
    sidecar = json.loads(file.read_text(encoding="utf-8"))
    components = sidecar.get("components") or []
    if not components:
        raise IdentityError(f"sidecar {file} lists no components")
    return {
        "kind": "sidecar",
        "name": file.name,
        "recipe": {k: sidecar[k] for k in _SIDECAR_RECIPE_KEYS if k in sidecar},
        "components": [feature_recipe(file.parent / name) for name in components],
    }


def feature_digest(path: str | Path) -> str:
    """Digest of :func:`feature_recipe` — content plus recursive recipes."""
    return canonical_digest(feature_recipe(path))


# ---------------------------------------------------------------------
# Data identity
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class DataIdentity:
    """Content identity of one dataset (and optionally one feature artifact)."""

    dataset: str
    dataset_digest: str
    item_mapping_digest: str
    split_digest: str
    splits: tuple[str, ...]
    feature_name: str | None = None
    feature_digest: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "dataset_digest": self.dataset_digest,
            "item_mapping_digest": self.item_mapping_digest,
            "split_digest": self.split_digest,
            "splits": list(self.splits),
            "feature_name": self.feature_name,
            "feature_digest": self.feature_digest,
        }


def resolve_data_identity(
    processed_dir: str | Path,
    dataset: str,
    feature_path: str | Path | None = None,
    *,
    splits: Sequence[str] = SELECTION_SPLITS,
) -> DataIdentity:
    """Compute the :class:`DataIdentity` of *dataset* under *processed_dir*.

    The dataset digest covers the dataset name, both mappings (by numeric
    index) and their sizes; the split digest covers the requested split
    files; the feature digest covers the artifact recursively.  Paths do
    not enter any digest, so identical content under another root
    yields the same identity.
    """
    base = Path(processed_dir) / dataset
    n_users, user_digest = mapping_file_digest(base / "user2idx.json")
    n_items, item_digest = mapping_file_digest(base / "item2idx.json")
    dataset_digest = canonical_digest(
        {
            "dataset": dataset,
            "n_users": n_users,
            "n_items": n_items,
            "user_mapping_digest": user_digest,
            "item_mapping_digest": item_digest,
        }
    )
    feature_name = Path(feature_path).name if feature_path is not None else None
    return DataIdentity(
        dataset=dataset,
        dataset_digest=dataset_digest,
        item_mapping_digest=item_digest,
        split_digest=split_digest(processed_dir, dataset, tuple(splits)),
        splits=tuple(splits),
        feature_name=feature_name,
        feature_digest=feature_digest(feature_path) if feature_path is not None else None,
    )


def build_identity_context(
    data: DataIdentity | Mapping[str, Any] | None,
    *,
    condition: str,
    fold: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Package what a training call needs to bind its identity (E03/E04).

    ``data`` may be a :class:`DataIdentity` or its payload (the shape a
    spawned worker receives).  ``condition`` is ``frozen``/``finetuned``;
    ``fold`` carries ``{"index", "k", "partition_seed", "min_profile"}``
    for a K-fold training.
    """
    payload = data.to_payload() if isinstance(data, DataIdentity) else data
    return {
        "data": dict(payload) if payload is not None else None,
        "condition": condition,
        "fold": dict(fold) if fold is not None else None,
    }


# ---------------------------------------------------------------------
# Experiment identity
# ---------------------------------------------------------------------


def implementation_digest(model_cls: type) -> str:
    """Identity of the model implementation: import path plus framework version."""
    from src import __version__

    return canonical_digest(
        {
            "module": model_cls.__module__,
            "qualname": model_cls.__qualname__,
            "version": __version__,
        }
    )


def experiment_identity(
    *,
    data: DataIdentity | Mapping[str, Any] | None,
    model_name: str,
    implementation: str,
    hyperparams: Mapping[str, Any],
    selection_budget: Mapping[str, Any] | None,
    seed: int,
    protocol: Mapping[str, Any],
    condition: str,
    fold: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the C02 identity payload (schema version 2).

    ``data`` may be ``None`` when the caller could not resolve the
    dataset/feature content; the payload then records
    ``"data_identity": "unresolved"`` explicitly instead of guessing.
    """
    payload = data.to_payload() if isinstance(data, DataIdentity) else data
    if payload is None:
        data_fields: dict[str, Any] = {
            "data_identity": "unresolved",
            "dataset_digest": None,
            "item_mapping_digest": None,
            "split_digest": None,
            "feature_digest": None,
        }
    else:
        data_fields = {
            "data_identity": "resolved",
            "dataset": payload.get("dataset"),
            "dataset_digest": payload["dataset_digest"],
            "item_mapping_digest": payload["item_mapping_digest"],
            "split_digest": payload["split_digest"],
            "splits": list(payload.get("splits") or []),
            "feature_name": payload.get("feature_name"),
            "feature_digest": payload.get("feature_digest"),
        }
    return {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        **data_fields,
        "model_name": model_name,
        "implementation_digest": implementation,
        "effective_hyperparams": dict(hyperparams),
        "selection_budget": dict(selection_budget) if selection_budget is not None else None,
        "seed": int(seed),
        "protocol": dict(protocol),
        "condition": condition,
        "fold": dict(fold) if fold is not None else None,
    }


def selection_scope(identity: Mapping[str, Any]) -> dict[str, Any]:
    """The identity minus the hyperparameters: the arena a ``_best.pt`` competes in.

    Trials of one cell legitimately differ in ``effective_hyperparams``;
    everything else must agree for their selection metrics to be
    comparable.
    """
    return {k: v for k, v in identity.items() if k != "effective_hyperparams"}


def selection_scope_digest(identity: Mapping[str, Any]) -> str:
    """Digest of :func:`selection_scope`."""
    return canonical_digest(selection_scope(identity))


def condition_of(embedding_name: str) -> str:
    """``finetuned`` when the artifact stem carries the fine-tuned marker, else ``frozen``."""
    from src.utils.artifact_names import is_finetuned_artifact

    return "finetuned" if is_finetuned_artifact(embedding_name) else "frozen"
