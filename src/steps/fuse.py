"""Step 04 — Fuse pairs of visual embeddings (v2: native-dim sources).

Operates on the primary fusion pair declared in
``configs/extractors.yaml`` (default: ``resnet50`` + ``vit_b16``), whose
native dimensionalities differ (2048 vs 768).

Strategy families and dimensionality (Pipeline B of the v2 protocol):

* **Element-wise / weighted** (``equal_dim_required=True``): require an
  alignment step first.  The alignment method is itself an experimental
  variable, configured under ``alignment:`` in ``configs/fusion.yaml``:

  - ``learned`` (default) — per-source ``Linear(D_i -> D)`` co-trained
    with the recommender via the BPR loss (the fusion analogue of the
    recommender's projection ``E``).  No offline ``.npy``; a JSON
    sidecar lists the native components and the training step builds a
    :class:`src.fusions.online.LearnedAlignmentFusion`.
  - ``pca`` — per-source PCA to ``D`` (fit ONLY on train items), then
    the element-wise op runs offline as before.

* **Concatenation family** (``equal_dim_required=False``): operate on
  the native dims directly.  ``concat`` -> 2816-d; ``pca`` (joint) and
  ``pca_per_model`` fit their PCA only on train items.

Two conditions are supported:

* ``frozen``    → fuses the original frozen embeddings (no suffix).
* ``finetuned`` → fuses the ``_finetuned`` embeddings produced by step 03b.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from src.fusions import (
    get_fusion_strategy,
    is_registered,
    iter_specs,
    registered_fusion_strategies,
)
from src.fusions.strategies import pca_align
from src.utils.atomic_io import atomic_np_save, atomic_write
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

_PCA_STRATEGIES = {"pca", "pca_per_model"}


def _strategies_map(config: dict) -> dict:
    """Return the ``strategy_name -> config block`` map from *config*."""
    return config.get("strategies", {})


def _alignment_config(config: dict) -> tuple[str, int]:
    """Read the alignment method/dim used by equal-dim fusion strategies."""
    block = config.get("alignment") or {}
    method = block.get("method", "learned")
    if method not in {"learned", "pca"}:
        raise ValueError(f"fusion alignment.method must be 'learned' or 'pca', got {method!r}")
    return method, int(block.get("dim", 128))


def _train_item_indices(processed_dir: str, dataset_name: str) -> list[int]:
    """Item indices with at least one *training* interaction.

    The PCA fit set: fitting on all catalogue items would leak items
    that only occur in validation/test interactions into the learned
    components.
    """
    train_csv = Path(processed_dir) / dataset_name / "train.csv"
    df = pd.read_csv(train_csv, usecols=["item_idx"])
    return sorted(int(i) for i in df["item_idx"].unique())


def _fuse_single(
    strategy_name: str,
    output_path: str,
    emb_list_paths: list[str],
    normalize: bool,
    train_items: list[int] | None = None,
    sidecar_payload: dict | None = None,
    **kwargs,
) -> str | None:
    """Execute a single fusion and save the result. Pickled by ProcessPool.

    When ``sidecar_payload`` is given, no offline fusion runs — the JSON
    sidecar is written for the training step to build the online module
    (learned alignment or adaptive_gated).
    """
    out = Path(output_path)
    if out.exists():
        return None

    out.parent.mkdir(parents=True, exist_ok=True)

    if sidecar_payload is not None:
        payload = json.dumps(sidecar_payload, indent=2)
        atomic_write(lambda tmp: Path(tmp).write_text(payload, encoding="utf-8"), out)
        return f"{strategy_name} (online): sidecar written -> {out}"

    emb_list = [np.load(p) for p in emb_list_paths]
    if strategy_name in _PCA_STRATEGIES:
        kwargs["train_items"] = np.asarray(train_items) if train_items is not None else None
    fuse_fn = get_fusion_strategy(strategy_name, **kwargs)
    fused = fuse_fn(emb_list, normalize=normalize)
    atomic_np_save(fused, out)
    return f"{strategy_name}: {fused.shape} -> {out}"


def _ensure_pca_aligned_sources(
    dataset_dir: Path,
    extractors: list[str],
    suffix: str,
    dim: int,
    train_items: list[int],
) -> list[Path] | None:
    """Materialise per-source PCA-aligned matrices, once per dataset/dim.

    Writes ``<ext><suffix>_pcaD<dim>.npy`` next to the native features.
    Fit is train-item-only (see :func:`pca_align`).  Returns the aligned
    paths, or ``None`` when a native source is missing.
    """
    native_paths = [dataset_dir / f"{ext}{suffix}.npy" for ext in extractors]
    if not all(p.exists() for p in native_paths):
        return None

    aligned_paths = [dataset_dir / f"{ext}{suffix}_pcaD{dim}.npy" for ext in extractors]
    if all(p.exists() for p in aligned_paths):
        return aligned_paths

    natives = [np.load(p) for p in native_paths]
    aligned = pca_align(natives, dim, train_items=np.asarray(train_items))
    for arr, path in zip(aligned, aligned_paths, strict=True):
        atomic_np_save(arr.astype(np.float32), path)
        logger.info("  pca-aligned source written: %s %s", path.name, arr.shape)
    return aligned_paths


def _collect_fusion_tasks(
    dataset_name: str,
    embeddings_dir: str,
    processed_dir: str,
    extractors: list[str],
    fusion_config: dict,
    normalize: bool,
    enabled_strategies: set[str],
    alignment_method: str,
    alignment_dim: int,
    suffix: str = "",
) -> list[dict]:
    """Build the list of fusion tasks for a single dataset.

    Strategy-agnostic: every registered strategy declares its grid via
    :meth:`FusionSpec.expand_grid`.  This function only knows the I/O
    layout and the alignment routing — never strategy-specific math.
    """
    tasks: list[dict] = []
    dataset_dir = Path(embeddings_dir) / dataset_name

    native_paths = [dataset_dir / f"{ext}{suffix}.npy" for ext in extractors]
    if not all(p.exists() for p in native_paths):
        logger.info(
            "  %s%s: fusion sources missing (%s) — skipping dataset.",
            dataset_name,
            suffix,
            ", ".join(p.name for p in native_paths if not p.exists()),
        )
        return tasks
    native_path_strs = [str(p) for p in native_paths]

    train_items = _train_item_indices(processed_dir, dataset_name)
    aligned_paths: list[Path] | None = None
    if alignment_method == "pca" and any(
        s.equal_dim_required for s in iter_specs() if s.name in enabled_strategies
    ):
        aligned_paths = _ensure_pca_aligned_sources(
            dataset_dir, extractors, suffix, alignment_dim, train_items
        )

    for spec in iter_specs():
        if spec.name not in enabled_strategies:
            continue

        strat_config = fusion_config.get(spec.name, {})
        try:
            grid = spec.expand_grid(strat_config) or [("", {})]
        except Exception as exc:  # noqa: BLE001
            logger.error("expand_grid failed for strategy %r: %s — skipping.", spec.name, exc)
            continue

        if not spec.equal_dim_required:
            # Concatenation family: operates on native dims directly.
            for fsuffix, fn_kwargs in grid:
                out = dataset_dir / f"hybrid_{spec.name}{fsuffix}{suffix}.npy"
                tasks.append(
                    {
                        "strategy_name": spec.name,
                        "output_path": str(out),
                        "emb_list_paths": native_path_strs,
                        "normalize": normalize,
                        "train_items": train_items if spec.name in _PCA_STRATEGIES else None,
                        **fn_kwargs,
                    }
                )
            continue

        # Equal-dim family: needs alignment (learned or pca).
        if alignment_method == "learned":
            for fsuffix, fn_kwargs in grid:
                out = dataset_dir / (
                    f"hybrid_{spec.name}{fsuffix}_learned{suffix}_D{alignment_dim}.json"
                )
                sidecar = {
                    "strategy": spec.name,
                    "online": True,
                    "alignment": "learned",
                    "dim": alignment_dim,
                    "components": [p.name for p in native_paths],
                    "normalize": normalize,
                    "fusion_kwargs": fn_kwargs,
                }
                tasks.append(
                    {
                        "strategy_name": spec.name,
                        "output_path": str(out),
                        "emb_list_paths": native_path_strs,
                        "normalize": normalize,
                        "sidecar_payload": sidecar,
                    }
                )
            continue

        # alignment_method == "pca"
        if aligned_paths is None:
            continue
        if spec.online:
            # adaptive_gated over pca-aligned equal-dim sources: classic
            # 3-D stacked sidecar consumed by AdaptiveGatedFusion.
            out = dataset_dir / f"hybrid_{spec.name}_pca{suffix}_D{alignment_dim}.json"
            sidecar = {
                "strategy": spec.name,
                "online": True,
                "alignment": "pca",
                "components": [p.name for p in aligned_paths],
                "normalize": normalize,
            }
            tasks.append(
                {
                    "strategy_name": spec.name,
                    "output_path": str(out),
                    "emb_list_paths": [str(p) for p in aligned_paths],
                    "normalize": normalize,
                    "sidecar_payload": sidecar,
                }
            )
            continue

        for fsuffix, fn_kwargs in grid:
            out = dataset_dir / (f"hybrid_{spec.name}{fsuffix}_pca{suffix}_D{alignment_dim}.npy")
            tasks.append(
                {
                    "strategy_name": spec.name,
                    "output_path": str(out),
                    "emb_list_paths": [str(p) for p in aligned_paths],
                    "normalize": normalize,
                    **fn_kwargs,
                }
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
    processed_dir = config["paths"]["data_processed"]
    fusion_config = _strategies_map(config)
    normalize = config.get("normalize_before_fusion", True)
    datasets = config.get("datasets", [])
    extractors = config.get("fusion_extractors", ["resnet50", "vit_b16"])
    alignment_method, alignment_dim = _alignment_config(config)

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

    logger.info(
        "Condition: %s (suffix=%r) | alignment: %s D=%d",
        condition,
        suffix,
        alignment_method,
        alignment_dim,
    )

    all_tasks: list[dict] = []
    for dataset_name in datasets:
        tasks = _collect_fusion_tasks(
            dataset_name,
            embeddings_dir,
            processed_dir,
            extractors,
            fusion_config,
            normalize,
            enabled_strategies,
            alignment_method,
            alignment_dim,
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
