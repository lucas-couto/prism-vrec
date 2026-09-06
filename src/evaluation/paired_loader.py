"""Paired per-user matrix loader for the statistical pipeline (Task F).

Given ``(dataset, seed, metric, k)``, assemble the users x systems matrix
aligned by ``user_idx`` from the per-cell artifacts written at final
evaluation.  This is the interface the statistical tests (Friedman,
Wilcoxon + Holm, Cliff's delta, paired bootstrap) consume — the tests
themselves are out of scope here.

Hard rule: if the user sets across cells do NOT match exactly, raise —
never intersect silently.  A quietly truncated pairing is a statistical
bug (it changes what is being compared).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.derive_metrics import per_user_metrics
from src.evaluation.paired_validation import ProvenanceMismatchError, validate_cell_records
from src.evaluation.persistence import read_cell_artifact


class UserSetMismatchError(RuntimeError):
    """Raised when cells to be paired do not share the exact same users."""


def _system_key(metadata: dict) -> str:
    return f"{metadata['recommender']}__{metadata['visual_config']}"


def discover_cells(per_user_dir: str | Path, dataset: str, seed: int) -> list[Path]:
    """All record artifacts for ``(dataset, seed)`` (checked via metadata)."""
    base = Path(per_user_dir) / "per_user" / dataset
    if not base.is_dir():
        return []
    hits: list[Path] = []
    for path in sorted(base.glob("*.csv.gz")):
        metadata, _ = read_cell_artifact(path)
        if int(metadata.get("seed")) == int(seed):
            hits.append(path)
    return hits


def load_paired(
    per_user_dir: str | Path,
    dataset: str,
    seed: int,
    metric: str,
    k: int,
) -> pd.DataFrame:
    """Users x systems matrix of ``metric@k``, aligned by ``user_idx``.

    Raises :class:`UserSetMismatchError` if any two cells disagree on the
    set of users, and a
    :class:`~src.evaluation.paired_validation.PairedValidationError` when
    a cell repeats a user, has non-finite ranks, disagrees with its own
    completion fields (``row_count`` / ``expected_user_digest`` when the
    metadata carries them) or differs from the others in split, protocol
    version or catalogue size (R04).
    """
    paths = discover_cells(per_user_dir, dataset, seed)
    if not paths:
        raise FileNotFoundError(f"no per-user artifacts for dataset={dataset!r} seed={seed}.")

    columns: dict[str, pd.Series] = {}
    reference_users: np.ndarray | None = None
    reference_key = ""
    reference_provenance: dict | None = None
    for path in paths:
        metadata, records = read_cell_artifact(path)
        key = _system_key(metadata)
        validate_cell_records(metadata, records, key)
        provenance = _cell_provenance(metadata, dataset)
        if reference_provenance is None:
            reference_provenance = provenance
        elif provenance != reference_provenance:
            raise ProvenanceMismatchError(
                f"cell {key} carries provenance {provenance}, but {reference_key} carries "
                f"{reference_provenance} for dataset={dataset} seed={seed}. Cells of "
                "different protocols/splits cannot be paired (Q07)."
            )
        users = records["user_idx"].to_numpy()
        sorted_users = np.sort(users)
        if reference_users is None:
            reference_users = sorted_users
            reference_key = key
        elif not np.array_equal(sorted_users, reference_users):
            raise UserSetMismatchError(
                f"user set of cell {_system_key(metadata)} "
                f"({len(users)} users) does not match {reference_key} "
                f"({len(reference_users)} users) for dataset={dataset} seed={seed}. "
                f"Refusing to intersect silently."
            )
        values = per_user_metrics(records["rank"].to_numpy(), k)[metric]
        columns[key] = pd.Series(values, index=users)

    return pd.DataFrame(columns).sort_index()


def _cell_provenance(metadata: dict, dataset: str) -> dict:
    """The identity fields every paired cell must share (R04).

    The fold policy is deliberately NOT part of it: a concatenated K-fold
    cell keeps the user as the unit and pairs with a leave-one-out cell
    of the same split by design (docs/protocol.md §3b).
    """
    if str(metadata.get("dataset")) != str(dataset):
        raise ProvenanceMismatchError(
            f"cell {_system_key(metadata)} is filed under dataset={dataset!r} but its "
            f"metadata says {metadata.get('dataset')!r}."
        )
    return {
        "split": metadata.get("split"),
        "eval_protocol_version": metadata.get("eval_protocol_version"),
        "n_items": metadata.get("n_items"),
    }
