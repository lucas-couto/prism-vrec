"""Step 04 — Fuse pairs of visual embeddings using ten strategies.

Operates on the primary fusion pair declared in
``configs/extractors.yaml`` (default: ``resnet50`` + ``vit_b16``) and
runs every strategy listed in ``configs/fusion.yaml`` in parallel via a
``ProcessPoolExecutor``.

Two conditions are supported:

* ``frozen``    → fuses the original frozen embeddings (no suffix).
* ``finetuned`` → fuses the ``_finetuned`` embeddings produced by step 03b.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from src.fusions import (
    get_fusion_strategy,
    is_registered,
    iter_specs,
    registered_fusion_strategies,
)
from src.utils.atomic_io import atomic_np_save
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _strategies_map(config: dict) -> dict:
    """Return the ``strategy_name -> config block`` map from *config*.

    ``load_config`` merges every yaml flat, so the fusion strategy grid
    lives at top-level ``config["strategies"]``.  Returning it here (and
    looking strategies up directly) avoids the historical double
    ``.get("strategies")`` that silently discarded the configured grid.
    """
    return config.get("strategies", {})


def _fuse_single(
    strategy_name: str,
    output_path: str,
    emb_list_paths: list[str],
    normalize: bool,
    online: bool = False,
    **kwargs,
) -> str | None:
    """Execute a single fusion and save the result. Pickled by ProcessPool.

    For *online* strategies (``adaptive_gated``) we do not run any
    fusion offline; we only persist a small JSON sidecar listing the
    component embeddings.  The training step re-reads this sidecar
    and stacks the components into a 3-D buffer before constructing
    the recommender.
    """
    import json

    out = Path(output_path)
    if out.exists():
        return None

    out.parent.mkdir(parents=True, exist_ok=True)

    if online:
        sidecar = {
            "strategy": strategy_name,
            "online": True,
            "components": [Path(p).name for p in emb_list_paths],
            "normalize": normalize,
        }
        out.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        return f"{strategy_name} (online): sidecar written -> {out}"

    emb_list = [np.load(p) for p in emb_list_paths]
    fuse_fn = get_fusion_strategy(strategy_name, **kwargs)
    fused = fuse_fn(emb_list, normalize=normalize)
    atomic_np_save(fused, out)
    return f"{strategy_name}: {fused.shape} -> {out}"


def _collect_fusion_tasks(
    dataset_name: str,
    embeddings_dir: str,
    extractors: list[str],
    projection_dims: list[int],
    fusion_config: dict,
    normalize: bool,
    enabled_strategies: set[str],
    suffix: str = "",
) -> list[dict]:
    """Build the list of fusion tasks for a single dataset.

    Strategy-agnostic: every registered strategy declares its grid via
    :meth:`FusionSpec.expand_grid`, which returns
    ``[(filename_suffix, fn_kwargs), ...]``.  This function only knows
    about the I/O layout — never about strategy-specific hyperparameters.
    """
    tasks: list[dict] = []

    for dim in projection_dims:
        emb_paths: list[str] = []
        for ext in extractors:
            p = Path(embeddings_dir) / dataset_name / f"{ext}{suffix}_D{dim}.npy"
            if p.exists():
                emb_paths.append(str(p))

        if len(emb_paths) < 2:
            continue

        for spec in iter_specs():
            if spec.name not in enabled_strategies:
                continue

            # Online strategies (e.g. adaptive_gated) are co-trained with
            # the recommender — no offline ``.npy`` is produced.  We
            # write a tiny JSON sidecar instead, listing the component
            # embeddings the trainer must stack into a (n_items, M, D)
            # buffer at load time.
            if spec.online:
                sidecar = (
                    Path(embeddings_dir) / dataset_name / f"hybrid_{spec.name}{suffix}_D{dim}.json"
                )
                tasks.append(
                    dict(
                        strategy_name=spec.name,
                        output_path=str(sidecar),
                        emb_list_paths=emb_paths,
                        normalize=normalize,
                        online=True,
                    )
                )
                continue

            strat_config = fusion_config.get(spec.name, {})
            try:
                grid = spec.expand_grid(strat_config) or [("", {})]
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "expand_grid failed for strategy %r: %s — skipping.",
                    spec.name,
                    exc,
                )
                continue

            for filename_suffix, fn_kwargs in grid:
                out = str(
                    Path(embeddings_dir)
                    / dataset_name
                    / f"hybrid_{spec.name}{filename_suffix}{suffix}_D{dim}.npy"
                )
                tasks.append(
                    dict(
                        strategy_name=spec.name,
                        output_path=out,
                        emb_list_paths=emb_paths,
                        normalize=normalize,
                        online=False,
                        **fn_kwargs,
                    )
                )

    return tasks


def run(condition: str = "frozen") -> None:
    """Run all fusion strategies for the given condition.

    Parameters
    ----------
    condition:
        ``"frozen"`` or ``"finetuned"``.  Selects the suffix appended to
        the input embedding filenames and to the output ``hybrid_*``
        files.
    """
    if condition not in {"frozen", "finetuned"}:
        raise ValueError(f"condition must be 'frozen' or 'finetuned', got {condition!r}")

    suffix = "_finetuned" if condition == "finetuned" else ""

    config = load_config()
    embeddings_dir = config["paths"]["embeddings"]
    fusion_config = _strategies_map(config)
    normalize = config.get("normalize_before_fusion", True)
    projection_dims = config.get("projection_dims", [64, 128, 256])
    datasets = config.get("datasets", [])
    extractors = config.get("fusion_extractors", ["resnet50", "vit_b16"])

    enabled_list = config.get("fusion_strategies_enabled") or []
    if not enabled_list:
        logger.info(
            "fuse step skipped: fusion_strategies_enabled is empty in "
            "configs/fusion.yaml. Set the list to e.g. [mean, concat] to "
            "enable fusion or leave it empty to opt out entirely.",
        )
        return
    if not datasets:
        logger.info("fuse step skipped: datasets list is empty in configs/default.yaml.")
        return

    enabled_strategies = set(enabled_list)
    unknown = [name for name in enabled_list if not is_registered(name)]
    if unknown:
        logger.warning(
            "fusion_strategies_enabled lists unregistered strategies "
            "(skipped): %s. Registered strategies: %s",
            ", ".join(sorted(unknown)),
            ", ".join(registered_fusion_strategies()),
        )
        enabled_strategies -= set(unknown)
    if not enabled_strategies:
        logger.info(
            "fuse step skipped: every name in fusion_strategies_enabled is "
            "absent from the registry.",
        )
        return

    logger.info("Condition: %s (suffix=%r)", condition, suffix)

    all_tasks: list[dict] = []
    for dataset_name in datasets:
        tasks = _collect_fusion_tasks(
            dataset_name,
            embeddings_dir,
            extractors,
            projection_dims,
            fusion_config,
            normalize,
            enabled_strategies,
            suffix=suffix,
        )
        all_tasks.extend(tasks)

    pending = [t for t in all_tasks if not Path(t["output_path"]).exists()]
    skipped = len(all_tasks) - len(pending)
    if skipped:
        logger.info("Skipping %d already existing fusions.", skipped)

    if not pending:
        logger.info("All fusions already exist.")
        return

    logger.info("Running %d fusions in parallel...", len(pending))

    n_workers = min(len(pending), os.cpu_count() or 4)
    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_fuse_single, **task): task for task in pending}
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if result:
                logger.info("  [%d/%d] %s", completed, len(pending), result)

    logger.info("Embedding fusion complete.")
