"""Cell reconciliation and output partitioning for the statistical step (R05).

Before a dataset's tests run, the step establishes what the report is a
report OF:

* which cells it was expected to contain — the evaluate step's
  completion record (``{dataset}_evaluation_done.csv``) when present,
  otherwise the table itself (a legacy run, recorded as such);
* which of those are complete: present in the table and evaluated on the
  reference user population (the union of users across cells);
* the run seed and how many distinct seeds the rows carry;
* the provenance the rows share (protocol, split, ...).

The result is written next to the tables as
``{dataset}_{condition}[_restricted]_integrity.json`` BEFORE any test runs,
so a rejected report still leaves its reasons on disk.  Every output of
the step is partitioned by the same stem, so a ``frozen`` invocation can
never overwrite an ``all`` one, nor a restricted analysis a strict one.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from src.evaluation.paired_validation import (
    POPULATION_STRICT,
    PROVENANCE_COLUMNS,
    PairedValidationError,
    validate_observations,
)

EXPECTED_SOURCE_DONE_MARKER = "done_marker"
EXPECTED_SOURCE_TABLE = "table"
_RESTRICTED_SUFFIX = "restricted"


def partition_stem(dataset: str, condition: str, population: str) -> str:
    """``{dataset}_{condition}[_restricted]`` — the prefix of every output file."""
    if population == POPULATION_STRICT:
        return f"{dataset}_{condition}"
    return f"{dataset}_{condition}_{_RESTRICTED_SUFFIX}"


@dataclass(frozen=True)
class ReportIntegrity:
    """What one dataset/condition report covers, and what it had to leave out."""

    dataset: str
    condition: str
    population_policy: str
    seed: int | None
    seed_source: str
    n_seeds_distinct: int
    provenance: dict[str, object]
    expected_source: str
    n_users_reference: int
    n_users_per_cell: dict[str, int]
    cells_expected: list[str]
    cells_completed: list[str]
    cells_missing: list[str]
    cells_excluded: dict[str, str] = field(default_factory=dict)

    @property
    def n_cells_expected(self) -> int:
        return len(self.cells_expected)

    @property
    def n_cells_completed(self) -> int:
        return len(self.cells_completed)

    def to_dict(self) -> dict:
        """JSON-friendly dictionary including the derived counts."""
        return {
            **asdict(self),
            "n_cells_expected": self.n_cells_expected,
            "n_cells_completed": self.n_cells_completed,
            "n_cells_missing": len(self.cells_missing),
            "n_cells_excluded": len(self.cells_excluded),
        }


def _expected_cells(
    tables_dir: Path, dataset: str, condition: str, observed: list[str]
) -> tuple[list[str], str]:
    """Cells the report must contain, from the evaluate step's done marker."""
    done_path = tables_dir / f"{dataset}_evaluation_done.csv"
    if not done_path.exists():
        return sorted(observed), EXPECTED_SOURCE_TABLE
    done = pd.read_csv(done_path)
    if done.empty:
        return sorted(observed), EXPECTED_SOURCE_TABLE
    targets = ("frozen", "finetuned") if condition == "all" else (condition,)
    done = done[done["target"].isin(targets)]
    keys = done["model_name"].astype(str) + "_" + done["embedding_name"].astype(str)
    return sorted(set(keys)), EXPECTED_SOURCE_DONE_MARKER


def _seed_of(df: pd.DataFrame, run_seed: int | None) -> tuple[int | None, str, int]:
    if "seed" in df.columns:
        seeds = df["seed"].dropna().unique()
        first = int(seeds[0]) if len(seeds) else run_seed
        return first, "table", int(len(seeds))
    return run_seed, "run_config", 1 if run_seed is not None else 0


def _shared_provenance(df: pd.DataFrame) -> dict[str, object]:
    out: dict[str, object] = {}
    for column in PROVENANCE_COLUMNS:
        if column not in df.columns:
            continue
        values = df[column].dropna().unique()
        if len(values) == 1:
            value = values[0]
            out[column] = value.item() if hasattr(value, "item") else value
    return out


def reconcile(
    eval_df: pd.DataFrame,
    tables_dir: Path,
    dataset: str,
    condition: str,
    *,
    population: str,
    run_seed: int | None,
) -> tuple[pd.DataFrame, ReportIntegrity]:
    """Validate the table and reconcile its cells against the expected set.

    :returns: ``(validated frame, integrity record)``.  The frame has one
        row per observation key (see :func:`validate_observations`).
    :raises PairedValidationError: On conflicts, mixed provenance/identity.
    """
    validated = validate_observations(eval_df)
    users_per_cell = validated.groupby("config")["user_id"].agg(lambda s: set(s))
    reference = set().union(*users_per_cell.tolist()) if len(users_per_cell) else set()
    observed = sorted(users_per_cell.index)
    expected, source = _expected_cells(tables_dir, dataset, condition, observed)
    missing = sorted(set(expected) - set(observed))
    excluded: dict[str, str] = {}
    for config in observed:
        n_cell = len(users_per_cell[config])
        if n_cell != len(reference):
            excluded[config] = (
                f"evaluated on {n_cell} of {len(reference)} reference users "
                "(population differs from the other cells)"
            )
    completed = [c for c in expected if c in set(observed) and c not in excluded]
    seed, seed_source, n_seeds = _seed_of(validated, run_seed)
    integrity = ReportIntegrity(
        dataset=dataset,
        condition=condition,
        population_policy=population,
        seed=seed,
        seed_source=seed_source,
        n_seeds_distinct=n_seeds,
        provenance=_shared_provenance(validated),
        expected_source=source,
        n_users_reference=len(reference),
        n_users_per_cell={c: len(users_per_cell[c]) for c in observed},
        cells_expected=list(expected),
        cells_completed=completed,
        cells_missing=missing,
        cells_excluded=excluded,
    )
    return validated, integrity


def write_integrity(tables_dir: Path, stem: str, integrity: ReportIntegrity) -> Path:
    """Write ``{stem}_integrity.json`` and return its path."""
    path = tables_dir / f"{stem}_integrity.json"
    path.write_text(json.dumps(integrity.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return path


def enforce(integrity: ReportIntegrity) -> None:
    """Reject incomplete required cells; under strict, reject unequal populations.

    Missing cells (recorded as done, absent from the table) always fail:
    a published aggregate must not silently drop a cell that exists.  A
    cell evaluated on fewer users fails under ``strict``; under
    ``declared_intersection`` it stays and every pair reports its
    exclusions.

    :raises PairedValidationError: With the offending cells and reasons.
    """
    if integrity.cells_missing:
        raise PairedValidationError(
            f"{integrity.dataset}/{integrity.condition}: {len(integrity.cells_missing)} cell(s) "
            f"recorded as evaluated but absent from the table: "
            f"{', '.join(integrity.cells_missing)}. Re-run evaluate for them or remove them "
            "from the completion record; the report cannot silently omit them."
        )
    if integrity.population_policy == POPULATION_STRICT and integrity.cells_excluded:
        reasons = "; ".join(f"{c}: {r}" for c, r in integrity.cells_excluded.items())
        raise PairedValidationError(
            f"{integrity.dataset}/{integrity.condition}: cells evaluated on a different user "
            f"population ({reasons}). A paired comparison needs one shared population; "
            "declare statistical.population: declared_intersection for a restricted "
            "analysis that reports its exclusions."
        )
