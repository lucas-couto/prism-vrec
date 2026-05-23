"""Step 03c, Post-hoc evaluation of fine-tuned extractors.

Runs after :mod:`src.steps.finetune` and produces, for every
``(extractor, dataset)`` combination that has a v2 checkpoint, a JSON
report with top-1 / top-5 / macro-F1 / per-class metrics / confusion
matrix on the deterministic validation split.

Reports land at ``results/finetuning/{dataset}_{extractor}.json``.

Skip-if-output-exists is honoured per combination, so the step is
cheap to re-run (it only computes what is missing).  Datasets without
category labels (Tradesy) are skipped, there is nothing to evaluate
since the post-hoc test requires labelled validation images.

Failures are isolated per combination so a single missing checkpoint
or layer-name mismatch does not abort the rest of the queue.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import torch

from src.data import dvbpr  # noqa: F401
from src.data.base import get_dataset_provider
from src.extractors import is_registered
from src.finetuning.evaluator import (
    CheckpointMissingHeadError,
    FineTuningEvaluator,
)
from src.utils.config import load_config
from src.utils.dataloader import resolve_dataloader_settings
from src.utils.device import resolve_device
from src.utils.logging import get_logger
from src.utils.seed import set_seed
from src.utils.timing import time_cell

logger = get_logger(__name__)


def _report_path(results_dir: str, dataset_name: str, extractor_name: str) -> Path:
    return Path(results_dir) / "finetuning" / f"{dataset_name}_{extractor_name}.json"


def _checkpoint_path(extractor_name: str, dataset_name: str) -> Path:
    return Path(f"checkpoints/finetuning/{extractor_name}_{dataset_name}_best.pt")


def _evaluate_one(
    extractor_name: str,
    dataset_name: str,
    image_dir: str,
    checkpoint: Path,
    output_path: Path,
    device: str,
    batch_size: int,
    num_workers: int,
) -> None:
    evaluator = FineTuningEvaluator(
        extractor_name=extractor_name,
        dataset_name=dataset_name,
        image_dir=image_dir,
        checkpoint_path=checkpoint,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    report = evaluator.evaluate()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2)
    tmp.rename(output_path)

    metrics = report.metrics
    logger.info(
        "  %s × %s, top1=%.4f top%d=%.4f macroF1=%.4f (n_val=%d)",
        extractor_name,
        dataset_name,
        metrics["top1_acc"],
        report.n_classes if report.n_classes <= 5 else 5,
        metrics.get(f"top{min(5, report.n_classes)}_acc", 0.0),
        metrics["macro_f1"],
        report.n_val_samples,
    )


def run() -> None:
    """Evaluate every fine-tuned ``(extractor, dataset)`` checkpoint.

    Failures are isolated per combination so the queue survives a
    missing checkpoint, a legacy (head-less) format, or a transient
    OOM.
    """
    config = load_config()
    set_seed(config["seed"])

    device = resolve_device(config["device"])
    raw_dir = config["paths"]["data_raw"]
    results_dir = config["paths"]["results"]
    datasets = config.get("datasets", [])
    ft_config = config.get("finetuning", {})
    ft_extractors = list(ft_config.get("extractors", []))

    enabled = config.get("extractors_enabled") or []
    if not enabled:
        logger.info(
            "evaluate_finetuning step skipped: extractors_enabled is empty.",
        )
        return
    if not datasets:
        logger.info("evaluate_finetuning step skipped: datasets list is empty.")
        return
    if not ft_extractors:
        logger.info(
            "evaluate_finetuning step skipped: finetuning.extractors is empty.",
        )
        return
    enabled_set = set(enabled)
    ft_extractors = [name for name in ft_extractors if name in enabled_set]
    if not ft_extractors:
        logger.info(
            "evaluate_finetuning step skipped: no fine-tunable extractor "
            "is also in extractors_enabled.",
        )
        return

    eval_settings = resolve_dataloader_settings(config)
    batch_size = eval_settings.batch_size
    num_workers = eval_settings.num_workers

    succeeded: list[str] = []
    skipped: list[str] = []
    failed: list[tuple[str, str]] = []

    for dataset_name in datasets:
        provider = get_dataset_provider(dataset_name)
        categories = provider.load_categories()
        if categories is None or provider.num_categories() <= 0:
            logger.info(
                "  %s has no category labels, nothing to evaluate, skipping.",
                dataset_name,
            )
            continue

        image_dir = f"{raw_dir}/{dataset_name}/images"
        logger.info("=== Evaluating fine-tunings on %s ===", dataset_name)

        for ext_name in ft_extractors:
            job_label = f"{ext_name} × {dataset_name}"

            if not is_registered(ext_name):
                logger.warning("  Extractor %s not registered, skipping.", ext_name)
                failed.append((job_label, "extractor not registered"))
                continue

            output_path = _report_path(results_dir, dataset_name, ext_name)
            if output_path.exists():
                logger.info("  %s, report exists, skipping.", job_label)
                skipped.append(job_label)
                continue

            checkpoint = _checkpoint_path(ext_name, dataset_name)
            if not checkpoint.exists():
                logger.warning(
                    "  %s, no checkpoint at %s, skipping.",
                    job_label,
                    checkpoint,
                )
                skipped.append(job_label)
                continue

            try:
                with time_cell(
                    "evaluate_finetuning",
                    dataset=dataset_name,
                    extractor=ext_name,
                ):
                    _evaluate_one(
                        extractor_name=ext_name,
                        dataset_name=dataset_name,
                        image_dir=image_dir,
                        checkpoint=checkpoint,
                        output_path=output_path,
                        device=device,
                        batch_size=batch_size,
                        num_workers=num_workers,
                    )
                succeeded.append(job_label)
            except CheckpointMissingHeadError as exc:
                logger.warning(
                    "  %s, legacy checkpoint without head: %s",
                    job_label,
                    exc,
                )
                skipped.append(job_label)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001, isolate per combination
                logger.error(
                    "  FAILED: %s, %s: %s\n%s",
                    job_label,
                    type(exc).__name__,
                    exc,
                    traceback.format_exc(),
                )
                failed.append((job_label, f"{type(exc).__name__}: {exc}"))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    logger.info(
        "Fine-tuning evaluation complete. succeeded=%d skipped=%d failed=%d",
        len(succeeded),
        len(skipped),
        len(failed),
    )
    if failed:
        logger.warning("Failed evaluations:")
        for job_label, err in failed:
            logger.warning("  - %s, %s", job_label, err)
