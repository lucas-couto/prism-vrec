"""Aggregation of per-fold evaluation artifacts into one cell artifact.

Each cell ``(dataset, visual_config, recommender)`` runs ``K`` folds.  On
fold ``k`` only the users assigned to that fold are evaluated (after
fold-in) against exactly one target item, producing a *partial* per-user
artifact with ``n_k`` rows in the same format the leave-one-out evaluate
step writes (see :mod:`src.evaluation.persistence`).

Because every user is evaluated exactly once across the ``K`` folds, the
partials are **concatenated** — never averaged — into a single artifact
at the cell's canonical location.  The statistical pipeline (paired
Wilcoxon + Holm, Cliff's delta) keeps pairing by *user* and never needs
to know folds existed.  The between-fold mean / standard deviation
reported by :class:`FoldAggregate` is descriptive variability only.

Layout under ``out_dir``::

    folds/fold<k>/per_user/<dataset>/<cell_key>.csv.gz    partial records
    folds/fold<k>/per_user/<dataset>/<cell_key>.meta.json cell metadata + fold
    per_user/<dataset>/<cell_key>.csv.gz                  concatenated cell

Fold provenance lives in the ``fold`` field of ``CellMetadata`` (see
:class:`src.evaluation.persistence.CellMetadata`): the partial carries
``{"index", "k", "seed", "n_users"}``; the concatenated artifact carries
``{"k", "seeds", "n_users_per_fold"}``.  There is no separate sidecar, so
a partial cannot drift from its own provenance on an interrupted resume.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.derive_metrics import aggregate as _aggregate_metrics
from src.evaluation.persistence import (
    CellMetadata,
    artifact_paths,
    read_cell_artifact,
    write_cell_artifact,
)

FOLDS_SUBDIR = "folds"
DEFAULT_K_VALUES: tuple[int, ...] = (5, 10, 20)
_REPORTED_METRICS: tuple[str, ...] = ("recall", "ndcg")


@dataclass(frozen=True)
class FoldAggregate:
    """Descriptive between-fold summary of one concatenated cell.

    :param n_users_total: Rows of the concatenated artifact (``sum(n_users_per_fold)``).
    :param n_users_per_fold: Evaluated users on each fold, in fold order.
    :param per_fold_metrics: ``{metric@k: mean over that fold's users}`` per fold.
    :param between_fold_mean: Mean of ``per_fold_metrics`` across folds.
    :param between_fold_std: Sample standard deviation (``ddof=1``) across folds.
    """

    n_users_total: int
    n_users_per_fold: list[int]
    per_fold_metrics: list[dict[str, float]]
    between_fold_mean: dict[str, float]
    between_fold_std: dict[str, float]

    def to_dict(self) -> dict:
        """Manifest-friendly plain dictionary."""
        return {
            "n_users_total": self.n_users_total,
            "n_users_per_fold": list(self.n_users_per_fold),
            "per_fold_metrics": [dict(m) for m in self.per_fold_metrics],
            "between_fold_mean": dict(self.between_fold_mean),
            "between_fold_std": dict(self.between_fold_std),
        }


def fold_dir(out_dir: str | Path, fold_index: int) -> Path:
    """Root directory of the partial artifacts of fold ``fold_index``."""
    return Path(out_dir) / FOLDS_SUBDIR / f"fold{fold_index}"


def write_fold_artifact(
    records: pd.DataFrame,
    metadata: CellMetadata,
    fold_index: int,
    out_dir: str | Path,
    *,
    k: int,
    fold_seed: int,
) -> Path:
    """Persist the partial per-user artifact of one fold.

    Reuses :func:`write_cell_artifact` under ``<out_dir>/folds/fold<i>/``
    with ``metadata.fold`` set to the partial provenance
    ``{"index", "k", "seed", "n_users"}``.

    :param records: Per-user records (``user_id``/``user_idx``, ``rank``, ...).
    :param metadata: Cell metadata (identical across the cell's folds).
    :param fold_index: Fold index in ``[0, k)``.
    :param out_dir: Results root; the cell's canonical location is derived from it.
    :param k: Number of folds of the plan.
    :param fold_seed: Seed the fold ran under.
    :returns: Path of the partial records file.
    :raises ValueError: If ``fold_index`` is outside ``[0, k)`` or ``records`` lacks a column.
    """
    if fold_index < 0:
        raise ValueError(f"fold_index must be >= 0, got {fold_index}")
    if fold_index >= k:
        raise ValueError(f"fold_index must be < k={k}, got {fold_index}")
    fold = {
        "index": int(fold_index),
        "k": int(k),
        "seed": int(fold_seed),
        "n_users": int(len(records)),
    }
    partial = replace(metadata, fold=fold)
    return write_cell_artifact(records, partial, fold_dir(out_dir, fold_index))


def _read_partial(
    out_dir: str | Path, metadata: CellMetadata, fold_index: int, k: int
) -> tuple[pd.DataFrame, dict]:
    """Read one partial and return ``(records, fold provenance)`` after validation."""
    records_path, _ = artifact_paths(fold_dir(out_dir, fold_index), metadata)
    if not records_path.exists():
        raise FileNotFoundError(f"missing partial artifact for fold {fold_index}: {records_path}")
    meta, records = read_cell_artifact(records_path)
    fold = meta.get("fold")
    if not isinstance(fold, dict) or "index" not in fold:
        raise ValueError(f"metadata of {records_path} carries no fold provenance")
    if int(fold["index"]) != fold_index:
        raise ValueError(
            f"metadata of {records_path} claims fold {fold['index']}, expected {fold_index}"
        )
    if int(fold["k"]) != k:
        raise ValueError(f"metadata of {records_path} claims k={fold['k']}, expected k={k}")
    if records.empty:
        raise ValueError(f"fold {fold_index} evaluated no users: {records_path}")
    return records, fold


def _assert_disjoint(partials: list[pd.DataFrame]) -> None:
    seen: set[int] = set()
    for fold_index, records in enumerate(partials):
        users = set(int(u) for u in records["user_idx"])
        if len(users) != len(records):
            raise ValueError(f"fold {fold_index} contains duplicate user_idx rows")
        repeated = seen & users
        if repeated:
            raise ValueError(
                f"user_idx {sorted(repeated)[:5]} evaluated on fold {fold_index} "
                f"and on an earlier fold; folds must be mutually disjoint"
            )
        seen |= users


def _between_folds(partials: list[pd.DataFrame], k_values: list[int]) -> FoldAggregate:
    per_fold = [
        {
            name: value
            for name, value in _aggregate_metrics(p, k_values).items()
            if name.split("@")[0] in _REPORTED_METRICS
        }
        for p in partials
    ]
    names = list(per_fold[0])
    stacked = {name: np.array([m[name] for m in per_fold], dtype=np.float64) for name in names}
    return FoldAggregate(
        n_users_total=int(sum(len(p) for p in partials)),
        n_users_per_fold=[int(len(p)) for p in partials],
        per_fold_metrics=per_fold,
        between_fold_mean={n: float(v.mean()) for n, v in stacked.items()},
        between_fold_std={n: float(v.std(ddof=1)) for n, v in stacked.items()},
    )


def concatenate_fold_artifacts(
    out_dir: str | Path,
    metadata: CellMetadata,
    k: int,
    k_values: list[int] | None = None,
) -> tuple[Path, FoldAggregate]:
    """Concatenate the ``k`` partial artifacts of a cell into its canonical artifact.

    Reads every ``folds/fold<i>`` partial, checks that all ``k`` exist and
    that their ``user_idx`` sets are mutually disjoint, sorts the union by
    ``user_idx`` and writes it with :func:`write_cell_artifact` to
    ``artifact_paths(out_dir, metadata)`` — the exact format the
    leave-one-out evaluate step produces, so the paired loader consumes
    it unchanged.  The written metadata carries
    ``fold = {"k", "seeds", "n_users_per_fold"}`` as provenance.

    :param out_dir: Results root holding ``folds/`` and the ``per_user/`` target.
    :param metadata: Cell metadata written alongside the concatenated records.
    :param k: Number of folds expected (``>= 2``).
    :param k_values: Cut-offs for the descriptive per-fold metrics (default 5/10/20).
    :returns: ``(concatenated records path, FoldAggregate)``.
    :raises ValueError: If ``k < 2``, a fold is empty, a partial's provenance
        disagrees with its location, or users repeat across folds.
    :raises FileNotFoundError: If any of the ``k`` partial artifacts is missing.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2, got {k}")
    cut_offs = list(k_values) if k_values is not None else list(DEFAULT_K_VALUES)
    read = [_read_partial(out_dir, metadata, i, k) for i in range(k)]
    partials = [records for records, _ in read]
    _assert_disjoint(partials)

    concatenated = (
        pd.concat(partials, ignore_index=True).sort_values("user_idx").reset_index(drop=True)
    )
    fold = {
        "k": int(k),
        "seeds": [int(f["seed"]) for _, f in read],
        "n_users_per_fold": [int(len(p)) for p in partials],
    }
    records_path = write_cell_artifact(concatenated, replace(metadata, fold=fold), out_dir)
    return records_path, _between_folds(partials, cut_offs)
