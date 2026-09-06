"""Observation-key validation for paired per-user statistics (R04 / Q17).

A paired test is only meaningful over a set of *uniquely keyed*
scientific observations: one value per ``(provenance, config, user)``.
The evaluation tables the statistical step consumes are rebuildable
views assembled by appending cells (and, in ``condition="all"``, by
concatenating two battery files), so the frame must be validated before
any pivot:

* **Provenance** columns (``dataset``, ``seed``, ``split``, ``protocol``,
  ``eval_protocol_version``, ...) must be single-valued across the
  frame — a paired comparison across seeds or protocols is a different
  analysis, never an implicit one (Q07).
* **Config identity** columns (``visual_input_dim``,
  ``n_trainable_params``, ...) must be constant within a config: two
  different checkpoints writing rows under one label is a mixed identity.
* **Duplicates** of a ``(user, config)`` key are accepted once only when
  every value matches AND the duplicate is the intentional one — the
  non-visual baseline (``embedding_name == "none"``) written to both
  battery files.  A changed value under the same key is a conflict; an
  identical duplicate of a visual cell is an unexplained (torn) append.
  Neither is ever resolved by keeping the first row.
* **Populations**: the users of the two configs of a pair must be equal.
  A restricted population is a separately declared analysis
  (``population="declared_intersection"``) that reports its exclusions.

Every error subclasses :class:`PairedValidationError` (a ``ValueError``,
matching the contract the callers already raise for bad inputs).

The provenance column list already includes the fields a future
generation/provenance manifest (CONTRACTS.md C05) would project onto the
rows (``split_digest``, ``generation_id``); they are validated whenever
present and ignored when absent, so no signature changes when they land.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

#: Columns that identify the experiment the whole frame comes from.
#: Every one present must be single-valued.
PROVENANCE_COLUMNS: tuple[str, ...] = (
    "dataset",
    "seed",
    "split",
    "protocol",
    "eval_protocol_version",
    "fold_policy",
    "split_digest",
    "generation_id",
)

#: Columns that identify the artifact behind ONE config; constant within it.
CONFIG_IDENTITY_COLUMNS: tuple[str, ...] = (
    "visual_input_dim",
    "n_trainable_params",
    "d",
    "checkpoint_digest",
)

#: Columns that say which file a row was read from, not what it measures.
SOURCE_COLUMNS: tuple[str, ...] = ("condition",)

#: The only cell intentionally written to more than one condition file.
INTENTIONAL_DUPLICATE_EMBEDDING = "none"

POPULATION_STRICT = "strict"
POPULATION_DECLARED_INTERSECTION = "declared_intersection"
POPULATION_POLICIES: tuple[str, ...] = (POPULATION_STRICT, POPULATION_DECLARED_INTERSECTION)

_KEY_COLUMNS = ("user_id", "config")
_LABEL_COLUMNS = ("model_name", "embedding_name")
_MAX_LISTED = 5


class PairedValidationError(ValueError):
    """Base class: the frame is not a valid set of paired observations."""


class ObservationConflictError(PairedValidationError):
    """Same observation key, different values."""


class DuplicateObservationError(PairedValidationError):
    """Identical duplicate rows that no intentional sharing explains."""


class ProvenanceMismatchError(PairedValidationError):
    """Rows of different seeds / protocols / identities mixed in one frame."""


class UserPopulationMismatchError(PairedValidationError):
    """The configs of a paired comparison were evaluated on different users."""


class InvalidMetricValueError(PairedValidationError):
    """A metric value is missing or non-finite (Q06)."""


def config_key(df: pd.DataFrame) -> pd.Series:
    """``model_name + "_" + embedding_name`` (the project-wide cell identity)."""
    if "embedding_name" in df.columns:
        return df["model_name"].astype(str) + "_" + df["embedding_name"].astype(str)
    return df["model_name"].astype(str)


def _listed(values: object) -> str:
    items = sorted(values) if not isinstance(values, list) else values
    head = ", ".join(str(v) for v in items[:_MAX_LISTED])
    more = len(items) - _MAX_LISTED
    return head + (f", ... (+{more} more)" if more > 0 else "")


def _check_provenance(df: pd.DataFrame) -> None:
    for column in PROVENANCE_COLUMNS:
        if column not in df.columns:
            continue
        distinct = df[column].dropna().unique()
        if len(distinct) > 1:
            raise ProvenanceMismatchError(
                f"paired observations mix {len(distinct)} values of {column!r}: "
                f"{_listed(list(distinct))}. A comparison across {column} values is a "
                "separate analysis (aggregate per value first); refusing to pool them."
            )


def _check_config_identity(df: pd.DataFrame) -> None:
    present = [c for c in CONFIG_IDENTITY_COLUMNS if c in df.columns]
    if not present:
        return
    n_distinct = df.groupby("config", sort=False)[present].nunique(dropna=True)
    mixed = n_distinct[(n_distinct > 1).any(axis=1)]
    if not mixed.empty:
        detail = "; ".join(
            f"{config}: " + ", ".join(f"{c}={int(n)} values" for c, n in row.items() if n > 1)
            for config, row in mixed.iterrows()
        )
        raise ProvenanceMismatchError(
            f"mixed identity within a config (rows of more than one artifact share "
            f"the same label): {detail}."
        )


def _resolve_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop provably intentional identical duplicates; fail on anything else."""
    key = list(_KEY_COLUMNS)
    dup_mask = df.duplicated(subset=key, keep=False)
    if not dup_mask.any():
        return df
    dups = df[dup_mask]
    value_columns = [c for c in df.columns if c not in key + list(SOURCE_COLUMNS)]
    identical = dups.duplicated(subset=key + value_columns, keep=False)
    conflicting = dups[~identical]
    if not conflicting.empty:
        keys = conflicting[key].drop_duplicates()
        pairs = [f"user {u} / {c}" for u, c in zip(keys["user_id"], keys["config"], strict=True)]
        raise ObservationConflictError(
            f"{len(keys)} observation key(s) carry conflicting values (same user and "
            f"config, different metrics): {_listed(pairs)}. Refusing to pick a row."
        )
    _check_intentional(dups)
    return df.drop_duplicates(subset=key, keep="first")


def _check_intentional(dups: pd.DataFrame) -> None:
    if "embedding_name" in dups.columns:
        visual = dups[dups["embedding_name"].astype(str) != INTENTIONAL_DUPLICATE_EMBEDDING]
        if not visual.empty:
            configs = visual["config"].unique()
            raise DuplicateObservationError(
                f"identical duplicate rows for visual config(s) {_listed(list(configs))}: "
                "only the non-visual baseline is written to more than one condition "
                "file; a repeated visual cell is a torn append, not an intentional share."
            )
    if "condition" not in dups.columns:
        return
    per_key = dups.groupby(list(_KEY_COLUMNS), sort=False)["condition"]
    same_source = per_key.nunique() < per_key.size()
    if same_source.any():
        keys = same_source[same_source].index[:_MAX_LISTED]
        raise DuplicateObservationError(
            f"identical duplicate rows within one condition file for "
            f"{_listed([f'user {u} / {c}' for u, c in keys])}: the baseline is shared "
            "across condition files, never repeated inside one."
        )


def validate_observations(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return *results_df* with a ``config`` column and one row per observation key.

    Adds ``config``; validates provenance homogeneity, per-config identity
    and duplicates as described in the module docstring; returns the frame
    sorted by ``(config, user_id)`` so downstream bootstrap draws do not
    depend on the caller's row order.  Idempotent: validating the output
    again returns an equal frame.  Frames without ``user_id`` (aggregated
    tables) only get the ``config`` column.

    :raises PairedValidationError: On conflicts, unexplained duplicates,
        mixed provenance or mixed identity.
    """
    out = results_df.copy()
    out["config"] = config_key(out)
    if "user_id" not in out.columns:
        return out
    _check_provenance(out)
    _check_config_identity(out)
    out = _resolve_duplicates(out)
    return out.sort_values(["config", "user_id"], kind="mergesort").reset_index(drop=True)


def check_finite(df: pd.DataFrame, metric: str) -> None:
    """Fail when *metric* has a missing or non-finite value (Q06).

    :raises InvalidMetricValueError: With the offending configs and count.
    """
    values = pd.to_numeric(df[metric], errors="coerce")
    bad = ~np.isfinite(values.to_numpy(dtype=float))
    if not bad.any():
        return
    configs = df.loc[bad, "config"].unique() if "config" in df.columns else []
    raise InvalidMetricValueError(
        f"{int(bad.sum())} value(s) of {metric!r} are missing or non-finite "
        f"(configs: {_listed(list(configs))}). A missing metric is a failed cell, "
        "not a zero and not an absent user."
    )


def check_population_policy(population: str) -> None:
    """Reject an unknown population policy name."""
    if population not in POPULATION_POLICIES:
        raise PairedValidationError(
            f"population must be one of {list(POPULATION_POLICIES)}; got {population!r}"
        )


def align_pair(
    pivot: pd.DataFrame, config_a: str, config_b: str, population: str
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Paired score vectors of two configs plus the users excluded from each.

    ``pivot`` is users x configs (NaN where a config has no row for a
    user).  Under ``"strict"`` any missing user fails; under
    ``"declared_intersection"`` the shared users are kept and the number
    missing from each side is returned.

    :returns: ``(scores_a, scores_b, n_excluded_a, n_excluded_b)``.
    :raises UserPopulationMismatchError: Strict policy with unequal populations.
    """
    pair = pivot[[config_a, config_b]]
    missing_a = pair[config_a].isna()
    missing_b = pair[config_b].isna()
    n_excluded_a = int(missing_a.sum())
    n_excluded_b = int(missing_b.sum())
    if population == POPULATION_STRICT and (n_excluded_a or n_excluded_b):
        raise UserPopulationMismatchError(
            f"user populations differ for the pair ({config_a}, {config_b}): "
            f"{n_excluded_a} user(s) absent from {config_a} "
            f"[{_listed(list(pair.index[missing_a]))}], "
            f"{n_excluded_b} absent from {config_b} "
            f"[{_listed(list(pair.index[missing_b]))}]. Refusing to intersect silently; "
            f"declare population={POPULATION_DECLARED_INTERSECTION!r} for a restricted analysis."
        )
    valid = pair.dropna()
    return (
        valid[config_a].to_numpy(dtype=float),
        valid[config_b].to_numpy(dtype=float),
        n_excluded_a,
        n_excluded_b,
    )


def require_equal_populations(pivot: pd.DataFrame) -> None:
    """Every config column of ``pivot`` must cover the same users.

    :raises UserPopulationMismatchError: Listing the deviating configs.
    """
    counts = pivot.notna().sum(axis=0)
    n_users = len(pivot)
    short = counts[counts != n_users]
    if short.empty:
        return
    detail = ", ".join(f"{c}: {int(n)}/{n_users}" for c, n in short.items())
    raise UserPopulationMismatchError(
        f"user populations differ across configs ({detail}); a joint test needs one "
        "shared population. Refusing to intersect silently."
    )


def user_population_digest(user_ids: np.ndarray) -> str:
    """SHA-256 of the sorted user ids, the digest a completion manifest carries."""
    ordered = np.sort(np.asarray(user_ids, dtype=np.int64))
    return hashlib.sha256(ordered.tobytes()).hexdigest()


def validate_cell_records(metadata: dict, records: pd.DataFrame, label: str) -> None:
    """Check one per-user cell artifact before it enters a paired matrix.

    Users must be unique, ranks finite, and — when the metadata carries
    the completion fields of a generation manifest (``row_count``,
    ``expected_user_digest``) — the records must match them.  Metadata
    without those fields is a legacy artifact and is accepted as such.

    :raises DuplicateObservationError: Repeated ``user_idx`` rows.
    :raises InvalidMetricValueError: Non-finite ranks or a size mismatch.
    :raises ProvenanceMismatchError: Records disagree with the manifest digest.
    """
    users = records["user_idx"].to_numpy()
    if len(np.unique(users)) != len(users):
        repeated = pd.Series(users)
        repeated = repeated[repeated.duplicated()].unique()
        raise DuplicateObservationError(
            f"cell {label} has repeated user_idx rows: {_listed(list(repeated))}."
        )
    ranks = pd.to_numeric(records["rank"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(ranks)):
        raise InvalidMetricValueError(f"cell {label} has non-finite ranks.")
    row_count = metadata.get("row_count")
    if row_count is not None and int(row_count) != len(records):
        raise InvalidMetricValueError(
            f"cell {label} declares row_count={row_count} but has {len(records)} rows."
        )
    expected = metadata.get("expected_user_digest")
    if expected is not None and expected != user_population_digest(users):
        raise ProvenanceMismatchError(
            f"cell {label}: user population does not match its declared "
            f"expected_user_digest; the artifact is incomplete or from another split."
        )
