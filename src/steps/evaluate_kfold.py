"""K-fold branch of the evaluate step (``folds.enabled: true``).

Runs the user-level K-fold protocol (:func:`src.folds.runner.run_folds`)
over every battery cell — K trainings with the frozen winners, fold-in of
the held-out users, per-fold held-out ranking, concatenation into the
canonical per-user artifact, manifest at ``results/folds/manifest.json``
— and then materialises, from the concatenated artifacts, the battery
tables the downstream steps consume:

* ``results/tables/{dataset}_evaluation_{frozen|finetuned}.csv`` — one
  row per (cell, user) with the same five metric families the
  single-split evaluator writes, derived from the persisted held-out
  rank (:mod:`src.evaluation.derive_metrics`) and routed by embedding
  exactly like the single-split rows;
* ``{dataset}_evaluation_done.csv`` — the completion record the
  statistical step reconciles its expected cells against (R05), bound
  to the fold-plan digest;
* ``{dataset}_evaluation_mean_{target}.csv`` — the mean view.

The single-split evaluator never runs while folds are enabled, so the
canonical per-user artifact has exactly one writer per run.  The step is
idempotent: ``run_folds`` skips cells whose artifact validates for the
current plan and the tables are upserted (rows of a cell replace the
cell's previous rows, never append to them).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.battery.manifest import BatteryManifest, require_complete
from src.evaluation.derive_metrics import metrics_frame
from src.evaluation.persistence import CellMetadata, artifact_paths, read_cell_artifact
from src.steps.evaluate import (
    _append_cell,
    _done_path,
    _record_done,
    _route_targets,
    _write_mean_table,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Metric families the single-split evaluator writes; the K-fold tables
#: carry the same ones so both protocols feed identical downstream code.
_METRIC_FAMILIES: tuple[str, ...] = ("precision", "recall", "f1", "map", "ndcg")


def run_kfold(config: dict, results_root: Path) -> BatteryManifest:
    """Run the K-fold protocol and materialise the battery tables.

    :param config: The merged run configuration (``folds.enabled`` true).
    :param results_root: ``paths.results`` of the run.
    :returns: The fold manifest, every cell ``done``.
    :raises IncompleteRunError: If any cell failed; the tables are not
        materialised from a partial cell set.
    """
    from src.folds.runner import run_folds

    manifest = run_folds(config, results_root)
    require_complete(manifest, label="K-fold evaluation")
    materialise_tables(config, results_root, manifest)
    return manifest


def materialise_tables(config: dict, results_root: Path, manifest: BatteryManifest) -> None:
    """Rebuild the battery tables and the completion record from the fold artifacts.

    :param config: The merged run configuration.
    :param results_root: ``paths.results`` of the run.
    :param manifest: The fold manifest listing the evaluated cells.
    :raises RuntimeError: If a cell's artifact carries no concatenated
        K-fold provenance (a torn or foreign artifact at the canonical path).
    """
    tables_dir = Path(results_root) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    k_values = list(config.get("k_values", [5, 10, 20]))
    seed = int(config["folds"]["seed"])
    touched: dict[str, set[str]] = {}
    for entry in manifest.cells.values():
        dataset = str(entry["dataset"])
        model_name = str(entry["recommender"])
        embedding_name = str(entry["visual_config"])
        rows, binding = _cell_rows(results_root, entry, seed, k_values)
        targets = _route_targets(model_name, embedding_name)
        for target in targets:
            _append_cell(rows, tables_dir / f"{dataset}_evaluation_{target}.csv")
        recorded = [(target, model_name, embedding_name) for target in targets]
        _record_done(
            _done_path(tables_dir, dataset), recorded, bindings=dict.fromkeys(recorded, binding)
        )
        touched.setdefault(dataset, set()).update(targets)
        logger.info("  %s/%s: %d users -> %s", model_name, embedding_name, len(rows), targets)
    for dataset, targets in touched.items():
        for target in sorted(targets):
            _write_mean_table(tables_dir, dataset, target)


def _cell_rows(
    results_root: Path, entry: dict, seed: int, k_values: list[int]
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Per-user metric rows of one concatenated fold artifact plus its done binding."""
    metadata_key = CellMetadata(
        dataset=str(entry["dataset"]),
        visual_config=str(entry["visual_config"]),
        recommender=str(entry["recommender"]),
        seed=seed,
        d=0,
        split="test",
    )
    records_path, _ = artifact_paths(results_root, metadata_key)
    metadata, records = read_cell_artifact(records_path)
    fold = metadata.get("fold")
    if not isinstance(fold, dict) or "k" not in fold or "index" in fold:
        raise RuntimeError(
            f"{records_path}: the artifact at the canonical path carries no concatenated "
            "K-fold provenance; refusing to build the battery tables from it."
        )
    frame = metrics_frame(records, k_values)
    kept = ["user_id", *[c for c in frame.columns if c.split("@")[0] in _METRIC_FAMILIES]]
    rows = frame[kept].assign(
        protocol="full_ranking",
        d=int(metadata.get("d", 0)),
        fold_policy=f"kfold_k{int(fold['k'])}",
        dataset=metadata_key.dataset,
        model_name=metadata_key.recommender,
        embedding_name=metadata_key.visual_config,
    )
    # No single winner exists under K-fold (one per fold), so the
    # checkpoint binding stays empty: a later single-split run can never
    # mistake this completion for one of its own (see ``_binding_matches``).
    binding = {"checkpoint_digest": "", "identity_digest": str(metadata.get("config_hash") or "")}
    return rows, binding
