"""Feature sanity gate (Task G).

The battery trains over 32 backbone feature matrices (8 backbones x 4
datasets) plus fused matrices.  A corrupt matrix (NaN, zeroed
placeholder rows, wrong shape/dtype) does not crash training — it burns
cloud credit producing silently wrong numbers.  This module fails loud
BEFORE consumption, in the same spirit as the ``expects_categories``
contract: a violation raises, never warns.

Native dims are read from the extractor registry (``configs/extractors``
``raw_dim``) — the single source of truth, not duplicated here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Canonical on-disk feature dtype (see ``src/extractors/base.py``).
FEATURE_DTYPE = np.float32
#: Rows with L2 norm below this are treated as empty (placeholder image).
NORM_EPS = 1e-8


class FeatureValidationError(RuntimeError):
    """Raised when a feature matrix fails a sanity check."""


def _n_items(processed_dir: str | Path, dataset: str) -> int:
    with open(Path(processed_dir) / dataset / "item2idx.json", encoding="utf-8") as fh:
        return len(json.load(fh))


def _raw_dim(backbone: str, config: dict) -> int | None:
    """Native dim from the extractor registry (single source of truth)."""
    return config.get("extractors", {}).get(backbone, {}).get("raw_dim")


def validate_matrix(
    matrix: np.ndarray,
    *,
    label: str,
    expected_rows: int,
    expected_dim: int | None = None,
    eps: float = NORM_EPS,
) -> dict:
    """Validate one feature matrix, returning its stats or raising.

    Positional premise (audit 1): on-disk row ``i`` must be ``item_idx``
    ``i``.  Only the row COUNT is verifiable here (asserted exactly); the
    order is a documented invariant of the extraction step, logged below.
    """
    if matrix.ndim != 2:
        raise FeatureValidationError(f"{label}: expected a 2-D matrix, got shape {matrix.shape}.")
    if matrix.shape[0] != expected_rows:
        raise FeatureValidationError(
            f"{label}: row count {matrix.shape[0]} != n_items {expected_rows} "
            f"(row i must be item_idx i — positional invariant)."
        )
    if expected_dim is not None and matrix.shape[1] != expected_dim:
        raise FeatureValidationError(
            f"{label}: feature dim {matrix.shape[1]} != backbone native dim "
            f"{expected_dim} (configs/extractors.yaml raw_dim)."
        )
    if matrix.dtype != FEATURE_DTYPE:
        raise FeatureValidationError(f"{label}: dtype {matrix.dtype} != {np.dtype(FEATURE_DTYPE)}.")

    finite_rows = np.isfinite(matrix).all(axis=1)
    if not finite_rows.all():
        bad = np.where(~finite_rows)[0]
        raise FeatureValidationError(
            f"{label}: {bad.size} row(s) contain NaN/Inf, e.g. item_idx {bad[:10].tolist()}."
        )

    norms = np.linalg.norm(matrix, axis=1)
    zero_rows = np.where(norms < eps)[0]
    if zero_rows.size:
        raise FeatureValidationError(
            f"{label}: {zero_rows.size} row(s) with L2 norm < {eps} "
            f"(zeroed/placeholder), e.g. item_idx {zero_rows[:10].tolist()}."
        )

    stats = {
        "rows": int(matrix.shape[0]),
        "dim": int(matrix.shape[1]),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
    }
    logger.info(
        "%s: OK (rows=%d, dim=%d, norm mean=%.4f std=%.4f min=%.4f max=%.4f) "
        "[positional invariant: row i == item_idx i]",
        label,
        stats["rows"],
        stats["dim"],
        stats["norm_mean"],
        stats["norm_std"],
        stats["norm_min"],
        stats["norm_max"],
    )
    return stats


def validate_backbone_feature(
    dataset: str,
    backbone: str,
    config: dict,
    *,
    embeddings_dir: str | Path,
    processed_dir: str | Path,
    suffix: str = "",
) -> dict:
    """Load and validate ``<dataset>/<backbone><suffix>.npy``."""
    path = Path(embeddings_dir) / dataset / f"{backbone}{suffix}.npy"
    label = f"{dataset}/{backbone}{suffix}"
    if not path.exists():
        raise FeatureValidationError(f"{label}: feature file missing at {path}.")
    matrix = np.load(path)
    return validate_matrix(
        matrix,
        label=label,
        expected_rows=_n_items(processed_dir, dataset),
        expected_dim=_raw_dim(backbone, config),
    )


def validate_fused_feature(
    path: str | Path,
    dataset: str,
    *,
    processed_dir: str | Path,
) -> dict:
    """Validate a fused matrix (post-fuse, pre-train).

    Fused dims depend on the strategy, so only the row count is checked
    against ``n_items``; NaN/Inf and zero-norm rows are still fatal.
    """
    path = Path(path)
    label = f"{dataset}/{path.name}"
    if not path.exists():
        raise FeatureValidationError(f"{label}: fused feature missing at {path}.")
    matrix = np.load(path)
    return validate_matrix(
        matrix,
        label=label,
        expected_rows=_n_items(processed_dir, dataset),
        expected_dim=None,
    )


def gate_backbone_features(
    datasets: list[str],
    backbones: list[str],
    config: dict,
    *,
    embeddings_dir: str | Path,
    processed_dir: str | Path,
    suffix: str = "",
) -> None:
    """Pipeline gate: validate every present ``(dataset, backbone)`` feature.

    Missing files are skipped (a step may run before all backbones are
    extracted); present files must pass.  Raises on the first failure.
    """
    for dataset in datasets:
        for backbone in backbones:
            path = Path(embeddings_dir) / dataset / f"{backbone}{suffix}.npy"
            if not path.exists():
                continue
            validate_backbone_feature(
                dataset,
                backbone,
                config,
                embeddings_dir=embeddings_dir,
                processed_dir=processed_dir,
                suffix=suffix,
            )


def gate_dataset_features(
    datasets: list[str],
    config: dict,
    *,
    embeddings_dir: str | Path,
    processed_dir: str | Path,
) -> None:
    """Pipeline gate for train: validate every ``.npy`` a dataset ships.

    Backbone matrices (stem is an enabled extractor) are dim-checked
    against ``raw_dim``; fused ``hybrid_*`` matrices are validated for
    row count / NaN / zero-norm only (their dim depends on the strategy).
    Online-fusion ``.json`` sidecars carry no matrix and are skipped.
    """
    enabled_extractors = set(config.get("extractors_enabled", []))
    for dataset in datasets:
        ds_dir = Path(embeddings_dir) / dataset
        if not ds_dir.is_dir():
            continue
        for path in sorted(ds_dir.glob("*.npy")):
            stem = path.stem
            # Strip the finetuned suffix when matching the extractor name.
            base = stem[: -len("_finetuned")] if stem.endswith("_finetuned") else stem
            if base in enabled_extractors:
                validate_backbone_feature(
                    dataset,
                    base,
                    config,
                    embeddings_dir=embeddings_dir,
                    processed_dir=processed_dir,
                    suffix="_finetuned" if stem.endswith("_finetuned") else "",
                )
            else:
                validate_fused_feature(path, dataset, processed_dir=processed_dir)


def run(dataset: str | None = None, backbone: str | None = None) -> int:
    """CLI entry (``--validate-features``): validate all or one pair.

    Returns a process exit code (0 = all OK, 1 = at least one failure).
    """
    from src.utils.config import load_config

    config = load_config()
    embeddings_dir = config["paths"]["embeddings"]
    processed_dir = config["paths"]["data_processed"]
    datasets = [dataset] if dataset else config.get("datasets", [])
    backbones = [backbone] if backbone else config.get("extractors_enabled", [])

    failures: list[str] = []
    checked = 0
    for ds in datasets:
        for bb in backbones:
            path = Path(embeddings_dir) / ds / f"{bb}.npy"
            if not path.exists():
                logger.info("%s/%s: not extracted yet, skipping.", ds, bb)
                continue
            checked += 1
            try:
                validate_backbone_feature(
                    ds, bb, config, embeddings_dir=embeddings_dir, processed_dir=processed_dir
                )
            except FeatureValidationError as exc:
                logger.error("FEATURE INVALID: %s", exc)
                failures.append(str(exc))

    logger.info("validate-features: %d matrices checked, %d failed.", checked, len(failures))
    return 1 if failures else 0
