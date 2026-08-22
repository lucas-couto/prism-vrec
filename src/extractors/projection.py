"""Fixed linear projection of native embeddings to a common dimension.

The v2 contract has extraction emit each backbone's **native**
dimensionality (ResNet-50 -> 2048, ViT-B/16 -> 768, ...), leaving the
mapping to a common space to the learned projection ``E`` inside each
recommender, or to ``alignment:`` in ``configs/fusion.yaml`` for the
element-wise fusion family.

This module adds an *optional* third route, configured under
``projection:`` in ``configs/extractors.yaml``: one fixed linear map
per artifact, applied once and written next to the native features as
``<extractor>_p<dim>.npy``.  Every projected artifact then lives in the
same ``dim``-dimensional space, so element-wise fusion (``mean``,
``sum``, ``prod``, ``max_pool``, ...) consumes them directly, with no
alignment learned online and no PCA fit inside the fusion step.

Two methods, both linear and both fixed before any recommender sees a
gradient:

``random``
    A semi-orthogonal matrix drawn from a seeded RNG (QR of a Gaussian).
    Data-independent: it never looks at the catalogue, so it cannot leak
    validation or test items, and the same seed reproduces it exactly on
    any machine.  Distances are preserved approximately
    (Johnson-Lindenstrauss).

``pca``
    Principal components fit on **train items only**, mirroring
    ``alignment.method: pca``.  Preserves more variance than a random
    map, at the cost of being dataset-dependent: the projector is fit
    per ``(dataset, artifact)`` and is not transferable between them.

The native artifact is never modified.  Projection is additive, so a
run comparing native against projected embeddings needs no re-extraction
of the backbone.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.artifact_names import FINETUNED_MARKER
from src.utils.atomic_io import atomic_np_memmap_save, atomic_write
from src.utils.logging import get_logger

logger = get_logger(__name__)

#: Rows transformed per pass.  Matches the fusion streaming executor, so
#: peak memory is a function of the chunk rather than the catalogue.
CHUNK_ROWS = 8192

METHODS = ("none", "random", "pca")


@dataclass(frozen=True)
class ProjectionConfig:
    """Resolved ``projection:`` block for one extractor.

    :param method: ``"random"`` or ``"pca"``.  ``"none"`` never reaches
        here — :func:`resolve_projection_config` returns ``None``.
    :param dim: Target dimensionality shared by every projected artifact.
    :param seed: RNG seed for ``random``; ignored by ``pca``, which is
        deterministic given its fit rows.
    """

    method: str
    dim: int
    seed: int = 42


def _coerce_block(block: Any, source: str) -> dict[str, Any]:
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise ValueError(f"{source} must be a mapping, got {type(block).__name__}")
    unknown = set(block) - {"method", "dim", "seed"}
    if unknown:
        raise ValueError(f"{source} has unknown keys: {sorted(unknown)}")
    return block


def resolve_projection_config(config: dict, extractor_name: str) -> ProjectionConfig | None:
    """Merge the global ``projection:`` block with a per-extractor override.

    The global block under ``projection:`` applies to every extractor;
    an ``extractors.<name>.projection`` block overrides any of its keys
    for that extractor alone, which is how one backbone opts out
    (``method: none``) or projects to a different width.

    :param config: The merged framework config.
    :param extractor_name: Extractor the projection is resolved for.
    :returns: The resolved config, or ``None`` when this extractor emits
        native features only.
    :raises ValueError: On an unknown method, a non-positive dim, or an
        unknown key in either block.
    """
    merged = dict(_coerce_block(config.get("projection"), "projection"))
    per_extractor = (config.get("extractors") or {}).get(extractor_name) or {}
    merged.update(
        _coerce_block(
            per_extractor.get("projection"),
            f"extractors.{extractor_name}.projection",
        )
    )

    method = str(merged.get("method", "none"))
    if method not in METHODS:
        raise ValueError(f"projection.method must be one of {list(METHODS)}, got {method!r}")
    if method == "none":
        return None

    dim = int(merged.get("dim", 128))
    if dim < 1:
        raise ValueError(f"projection.dim must be >= 1, got {dim}")
    return ProjectionConfig(method=method, dim=dim, seed=int(merged.get("seed", 42)))


def _random_matrix(native_dim: int, dim: int, seed: int, name: str) -> np.ndarray:
    """Semi-orthogonal ``(native_dim, dim)`` matrix from a seeded RNG.

    Orthonormal columns via QR, so the map is an isometry up to the
    scale factor that L2 normalisation removes downstream anyway.  The
    seed is mixed with the artifact name, otherwise every extractor
    would share one matrix and their projected features would be
    correlated by construction.
    """
    # ``hash()`` is salted per interpreter process (PYTHONHASHSEED), so it
    # would give a different matrix on every run — the one thing this
    # method promises not to do.  A stable digest keeps the derivation
    # reproducible across processes and machines.
    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    mixed = (seed + int.from_bytes(digest, "big")) % (2**32)
    rng = np.random.default_rng(mixed)
    gaussian = rng.standard_normal((native_dim, dim), dtype=np.float64)
    q, _ = np.linalg.qr(gaussian)
    return np.ascontiguousarray(q[:, :dim], dtype=np.float32)


def _pca_matrix(
    source: np.ndarray,
    dim: int,
    train_items: np.ndarray,
    seed: int,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Train-only PCA basis as an explicit ``(components, mean)`` pair.

    Returned as matrices rather than a fitted estimator so the projector
    persisted on disk is a plain linear map — auditable, and applicable
    without scikit-learn in the loop.
    """
    from src.fusions import fit_pca_on_rows

    fit_rows = np.asarray(source[train_items], dtype=np.float32)
    k = min(dim, *fit_rows.shape, int(source.shape[1]))
    pca = fit_pca_on_rows(fit_rows, k, seed, f"projection[{name}]", copy=False)
    return (
        np.ascontiguousarray(pca.components_.T, dtype=np.float32),
        np.ascontiguousarray(pca.mean_, dtype=np.float32),
    )


def _write_projector(path: Path, matrix: np.ndarray, mean: np.ndarray | None, cfg, name) -> None:
    """Persist the linear map next to the artifact it produced.

    The projected ``.npy`` alone does not say how it was built.  Saving
    ``W`` (and the PCA mean) makes the artifact auditable and lets a
    later run project new rows into the *same* space instead of fitting
    a second, subtly different basis.
    """
    payload: dict[str, np.ndarray] = {"matrix": matrix}
    if mean is not None:
        payload["mean"] = mean

    def _save(tmp: str) -> None:
        # Through a file object: ``np.savez`` appends ``.npz`` to a *path*
        # argument, which would write next to the temp file instead of to it.
        with open(tmp, "wb") as handle:
            np.savez(handle, **payload)

    atomic_write(_save, path)

    meta = {
        "artifact": name,
        "method": cfg.method,
        "dim": int(matrix.shape[1]),
        "native_dim": int(matrix.shape[0]),
        "seed": cfg.seed if cfg.method == "random" else None,
        "fit": "train items only" if cfg.method == "pca" else "data-independent",
    }
    text = json.dumps(meta, indent=2)
    atomic_write(
        lambda tmp: Path(tmp).write_text(text, encoding="utf-8"),
        path.with_suffix(".json"),
    )


def projected_path(source_npy: Path, dim: int) -> Path:
    """Destination of the projected artifact for *source_npy*.

    The ``_p<dim>`` token goes immediately after the extractor name and
    *before* ``_finetuned``, so the projected artifacts of both
    conditions compose with the ``{extractor}{condition_suffix}`` naming
    the fuse step builds its source paths from: setting
    ``fusion_extractors: [resnet50_p128, vit_b16_p128]`` then resolves to
    ``resnet50_p128.npy`` for the frozen condition and
    ``resnet50_p128_finetuned.npy`` for the fine-tuned one.
    """
    stem = source_npy.stem
    if stem.endswith(FINETUNED_MARKER):
        stem = f"{stem[: -len(FINETUNED_MARKER)]}_p{dim}{FINETUNED_MARKER}"
    else:
        stem = f"{stem}_p{dim}"
    return source_npy.with_name(f"{stem}.npy")


def ensure_projected(
    source_npy: str | Path,
    cfg: ProjectionConfig,
    train_items: np.ndarray | list[int] | None = None,
    *,
    chunk_rows: int = CHUNK_ROWS,
) -> Path | None:
    """Project *source_npy* to ``cfg.dim``, writing the artifact if absent.

    Idempotent: an existing projected artifact is left alone, so the
    projection of an already-extracted catalogue costs one no-op check
    rather than a re-extraction.

    The source is read through a memory map and transformed in chunks of
    *chunk_rows*, so peak memory is a function of the chunk, not of the
    catalogue.

    :param source_npy: Native ``.npy`` to project.
    :param cfg: Resolved projection config (method, dim, seed).
    :param train_items: Item indices forming the PCA fit set.  Required
        by ``method: pca``, ignored by ``method: random``.
    :param chunk_rows: Rows transformed per pass.
    :returns: The path written, or ``None`` when it already existed.
    :raises ValueError: When ``method: pca`` is configured without a fit
        set, or when the source is narrower than the requested dim.
    """
    source_npy = Path(source_npy)
    output = projected_path(source_npy, cfg.dim)
    if output.exists():
        return None

    source = np.load(source_npy, mmap_mode="r")
    if source.ndim != 2:
        raise ValueError(
            f"{source_npy}: projection expects a 2-D pooled matrix, got shape {source.shape}."
        )
    native_dim = int(source.shape[1])
    if native_dim < cfg.dim:
        raise ValueError(
            f"{source_npy}: cannot project {native_dim}-d features up to "
            f"{cfg.dim}-d — projection.dim must not exceed the narrowest "
            f"backbone's native dim."
        )

    name = output.stem
    mean: np.ndarray | None = None
    if cfg.method == "random":
        matrix = _random_matrix(native_dim, cfg.dim, cfg.seed, name)
    else:
        if train_items is None or len(train_items) == 0:
            raise ValueError(
                f"{source_npy}: projection.method 'pca' needs the train-item "
                f"fit set; none was provided."
            )
        matrix, mean = _pca_matrix(source, cfg.dim, np.asarray(train_items), cfg.seed, name)

    shape = (int(source.shape[0]), int(matrix.shape[1]))

    def _fill(out: np.memmap) -> None:
        for start in range(0, shape[0], chunk_rows):
            rows = np.asarray(source[start : start + chunk_rows], dtype=np.float32)
            if mean is not None:
                rows = rows - mean
            out[start : start + chunk_rows] = rows @ matrix

    atomic_np_memmap_save(output, dtype=np.float32, shape=shape, fill=_fill)
    _write_projector(output.with_suffix(".proj.npz"), matrix, mean, cfg, name)
    logger.info(
        "  projected %s -> %s %s (%s, dim %d)",
        source_npy.name,
        output.name,
        shape,
        cfg.method,
        cfg.dim,
    )
    return output
