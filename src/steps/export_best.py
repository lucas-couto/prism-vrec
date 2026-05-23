"""Step 08 — Export winning hyperparameters as a single JSON.

Walks ``<paths.results>/models/<dataset>/*_best.pt`` and consolidates
the ``hyperparams`` + ``best_metric`` payload of each checkpoint into
``<paths.results>/best_hyperparams.json``.  Works uniformly across
both hyperparameter-search backends (grid and Optuna) because each
backend writes the same ``_best.pt`` payload via
``src.utils.training._save_best_model``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.recommenders.registry import registered_recommender_names
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _parse_checkpoint_stem(stem: str, known_models: list[str]) -> tuple[str, str] | None:
    """Split ``{model_name}_{embedding_name}`` filename stem.

    Recommender names may contain underscores (e.g. ``uniform_noise``),
    so the boundary cannot be inferred positionally — we match the
    longest registered recommender name as the *prefix*.  Returns
    ``(model_name, embedding_name)`` or ``None`` when no registered
    model matches.
    """
    for candidate in known_models:
        if stem == candidate:
            return candidate, "none"
        if stem.startswith(candidate + "_"):
            return candidate, stem[len(candidate) + 1 :]
    return None


def _natural_sort_key(key: str) -> list:
    """Natural sort so output JSON keys read in a stable, human-friendly order."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", key)]


def export_best_hyperparams(models_root: Path, output_path: Path) -> dict:
    """Walk ``models_root`` and write the consolidated summary JSON.

    Each ``<dataset>/<model>_<embedding>_best.pt`` checkpoint is
    expected to carry a ``hyperparams`` dict and a ``best_metric``
    float (the payload :func:`src.utils.training._save_best_model`
    writes).  Cells whose checkpoint cannot be parsed or loaded are
    logged and skipped; the rest are aggregated into

        {dataset: {model: {embedding: {hyperparams, best_metric}}}}
    """
    if not models_root.exists():
        logger.warning("Models directory not found: %s", models_root)
        return {}

    try:
        import torch
    except ImportError:
        logger.warning("torch unavailable; cannot read checkpoint payloads.")
        return {}

    known_models = sorted(registered_recommender_names(), key=len, reverse=True)

    summary: dict[str, dict[str, dict[str, dict]]] = {}
    n_exported = 0
    n_skipped = 0

    for dataset_dir in sorted(p for p in models_root.iterdir() if p.is_dir()):
        dataset = dataset_dir.name
        for pt_path in sorted(dataset_dir.glob("*_best.pt")):
            parsed = _parse_checkpoint_stem(
                pt_path.stem.replace("_best", ""),
                known_models,
            )
            if parsed is None:
                logger.warning("Could not parse checkpoint stem: %s", pt_path.name)
                n_skipped += 1
                continue
            model_name, embedding_name = parsed

            try:
                payload = torch.load(pt_path, map_location="cpu", weights_only=False)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load %s: %s", pt_path, exc)
                n_skipped += 1
                continue

            hyperparams = payload.get("hyperparams") or {}
            best_metric = float(payload.get("best_metric", 0.0))
            summary.setdefault(dataset, {}).setdefault(model_name, {})[embedding_name] = {
                "hyperparams": hyperparams,
                "best_metric": best_metric,
            }
            n_exported += 1

    sorted_summary: dict = {}
    for dataset in sorted(summary.keys(), key=_natural_sort_key):
        sorted_summary[dataset] = {}
        for model in sorted(summary[dataset].keys(), key=_natural_sort_key):
            sorted_summary[dataset][model] = {
                emb: summary[dataset][model][emb]
                for emb in sorted(summary[dataset][model].keys(), key=_natural_sort_key)
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(sorted_summary, fh, indent=2, sort_keys=False)

    logger.info(
        "Exported best hyperparams for %d cell(s) to %s (%d skipped).",
        n_exported,
        output_path,
        n_skipped,
    )
    return sorted_summary


def run(output: str | None = None) -> None:
    """Build the winning-hyperparameters summary file.

    Reads ``<paths.results>/models/`` and writes
    ``<paths.results>/best_hyperparams.json`` (or the explicit
    ``output`` path when supplied).  Honours the configured results
    root so swapped config profiles (e.g. ``configs/smoke/``) land
    their summary in the expected place.
    """
    config = load_config()
    results_root = Path(config.get("paths", {}).get("results", "results"))
    models_root = results_root / "models"
    output_path = Path(output) if output else results_root / "best_hyperparams.json"
    export_best_hyperparams(models_root, output_path)
