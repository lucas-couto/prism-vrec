"""Step 03, Extract frozen visual embeddings.

Iterates over every registered extractor × dataset × projection dim and
writes the resulting ``.npy`` files to ``data/embeddings/<dataset>/``.
Idempotent: existing outputs are detected and skipped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.extractors import (
    get_extractor_class,
    is_registered,
    registered_extractor_names,
)
from src.extractors.projection import (
    ProjectionConfig,
    ensure_projected,
    projected_path,
    resolve_projection_config,
)
from src.utils.atomic_io import atomic_write
from src.utils.checkpoint import CheckpointManager
from src.utils.config import load_config
from src.utils.dataloader import resolve_dataloader_settings
from src.utils.device import resolve_device
from src.utils.logging import get_logger
from src.utils.seed import set_seed
from src.utils.splits import train_item_indices
from src.utils.timing import time_cell

logger = get_logger(__name__)


class ImageDataset(Dataset):
    """Loader for per-item JPEGs already extracted to disk.

    Performs a single ``os.listdir()`` to filter items that have an
    image on disk, distributed filesystems (NFS, MooseFS) make a naive
    ``Path.exists()`` per item prohibitively slow.
    """

    def __init__(self, image_dir: str, item_ids: list, transform=None) -> None:
        self.image_dir = Path(image_dir)
        self.item_ids = item_ids
        self.transform = transform
        self.valid_items: list = []
        self.valid_paths: list = []

        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        files_by_stem: dict[str, Path] = {}
        try:
            for name in os.listdir(self.image_dir):
                stem, ext = os.path.splitext(name)
                if ext.lower() in valid_exts and stem not in files_by_stem:
                    files_by_stem[stem] = self.image_dir / name
        except (FileNotFoundError, NotADirectoryError) as exc:
            logger.warning(
                "Image directory %s is missing or not a directory (%s); dataset will be empty.",
                self.image_dir,
                exc,
            )

        for item_id in item_ids:
            path = files_by_stem.get(str(item_id))
            if path is not None:
                self.valid_items.append(item_id)
                self.valid_paths.append(path)

    def __len__(self) -> int:
        return len(self.valid_items)

    def __getitem__(self, idx: int):
        img = Image.open(self.valid_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.valid_items[idx]


def get_item_ids(processed_dir: str, dataset_name: str) -> list:
    """Load the ordered item id list for a dataset from ``item2idx.json``."""
    item2idx_path = Path(processed_dir) / dataset_name / "item2idx.json"
    with open(item2idx_path) as f:
        item2idx = json.load(f)
    return list(item2idx.keys())


def _write_meta(extractor, extractor_name: str, npy_path: Path, extra: dict | None = None) -> None:
    """Write the ``<stem>.meta.json`` sidecar next to a feature file.

    The metadata is what makes the artifact reproducible and lets the
    loader know the input dimension without inferring it from the shape:
    backbone name, native dimensionality, extraction point, exact
    pretrained-weights id, and the transform recipe.
    """
    meta = {"name": extractor_name, **extractor.metadata()}
    if extra:
        meta.update(extra)
    meta_path = npy_path.with_suffix("").with_suffix(".meta.json")
    payload = json.dumps(meta, indent=2)
    atomic_write(lambda tmp: Path(tmp).write_text(payload, encoding="utf-8"), meta_path)


def _project_pooled(
    pooled_path: Path,
    projection: ProjectionConfig | None,
    train_items,
) -> bool:
    """Write the fixed-dim projection of a pooled artifact, if configured.

    Deliberately independent of a live extractor instance: the source's
    own ``.meta.json`` carries the backbone provenance, so projecting a
    catalogue that was extracted on an earlier run never pays for
    loading the backbone weights again.

    The sidecar declares the *projected* width under ``native_dim``,
    because that is what a loader must expect from this file; the
    backbone's own width is preserved under ``source_native_dim`` so the
    artifact still says where it came from.

    :returns: ``True`` when a projected artifact was written.
    """
    if projection is None:
        return False

    written = ensure_projected(pooled_path, projection, train_items)
    if written is None:
        return False

    source_meta_path = pooled_path.with_suffix("").with_suffix(".meta.json")
    meta: dict = {}
    if source_meta_path.exists():
        meta = json.loads(source_meta_path.read_text(encoding="utf-8"))

    meta.update(
        {
            "name": written.stem,
            "kind": "pooled",
            "source_native_dim": meta.get("native_dim"),
            "native_dim": projection.dim,
            "projection": {
                "method": projection.method,
                "dim": projection.dim,
                "source": pooled_path.name,
            },
        }
    )
    payload = json.dumps(meta, indent=2)
    meta_path = written.with_suffix("").with_suffix(".meta.json")
    atomic_write(lambda tmp: Path(tmp).write_text(payload, encoding="utf-8"), meta_path)
    return True


def _validate_weights_id(extractor, extractor_name: str, config: dict) -> None:
    """Fail loudly when the declared weights tag is absent from the model's id.

    ``configs/extractors.yaml`` declares each backbone's ``weights`` tag
    (e.g. ``IMAGENET1K_V2``); the authoritative value is the extractor's
    hardcoded ``weights_id``.  The key used to be decorative — editing
    it changed nothing — which is a fidelity trap: like ``raw_dim``, a
    declaration the code does not honour must at least be checked.
    """
    declared = config.get("extractors", {}).get(extractor_name, {}).get("weights")
    if declared and str(declared) not in getattr(extractor, "weights_id", ""):
        raise RuntimeError(
            f"{extractor_name}: configs/extractors.yaml declares weights="
            f"{declared!r} but the extractor is built with weights_id="
            f"{extractor.weights_id!r}. The code is authoritative — fix the "
            "config (the declared tag must appear in the real weights id)."
        )


def _validate_native_dim(extractor, extractor_name: str, config: dict) -> None:
    """Fail loudly when the probed native dim contradicts the config.

    ``configs/extractors.yaml`` declares each backbone's expected
    ``raw_dim``.  The authoritative value is the one READ from the model
    (probe forward); a mismatch means the config (or the model wiring)
    is wrong and must be fixed before anything is extracted.
    """
    declared = config.get("extractors", {}).get(extractor_name, {}).get("raw_dim")
    if declared is not None and int(declared) != extractor.native_dim:
        raise RuntimeError(
            f"{extractor_name}: probed native_dim={extractor.native_dim} but "
            f"configs/extractors.yaml declares raw_dim={declared}. The model "
            "is authoritative — fix the config (dims are read, never assumed)."
        )


def _extract_for_config(
    extractor_cls,
    extractor_name: str,
    dataset_name: str,
    image_dir: str,
    item_ids: list,
    embeddings_dir: str,
    batch_size: int,
    checkpoint_every: int,
    device: str,
    config: dict,
    extract_components: bool = False,
    projection: ProjectionConfig | None = None,
    train_items: list[int] | None = None,
    component_grid: int | None = None,
) -> bool:
    """Extract native-dim embeddings for a single ``(extractor, dataset)`` cell.

    Writes the pooled ``<extractor>.npy`` at the backbone's native
    dimensionality plus a ``<extractor>.meta.json`` sidecar.  When
    ``extract_components`` is set and the extractor advertises
    ``supports_components``, additionally writes the 3-D
    ``<extractor>_comp.npy`` (native per-item components) consumed by
    ACF.  Both outputs are skipped independently when already present.

    When ``projection`` is set, a fixed-dim ``<extractor>_p<dim>.npy`` is
    written alongside the native pooled artifact.  It is skipped
    independently too, and a cell that needs *only* the projection never
    instantiates the backbone: projecting an already-extracted catalogue
    costs a linear pass over the matrix, not a re-extraction.

    :returns:
        ``True`` when something was actually extracted, ``False`` when
        every output already existed.  The caller uses this to keep
        no-op cells out of ``step_timings.json``.
    """
    out_dir = Path(embeddings_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pooled_path = out_dir / f"{extractor_name}.npy"
    comp_path = out_dir / f"{extractor_name}_comp.npy"

    want_components = extract_components and getattr(extractor_cls, "supports_components", False)
    need_pooled = not pooled_path.exists()
    need_components = want_components and not comp_path.exists()
    need_projection = (
        projection is not None and not projected_path(pooled_path, projection.dim).exists()
    )

    if not need_pooled and not need_components and not need_projection:
        logger.info("  %s: already exists, skipping.", extractor_name)
        return False

    if not need_pooled and not need_components:
        # Only the projection is missing.  The native features are already
        # on disk and the projection is a linear map over them, so the
        # backbone is never loaded.
        return _project_pooled(pooled_path, projection, train_items)

    logger.info("  Extracting %s (native dim)...", extractor_name)
    extractor = extractor_cls(device=device)
    extractor.component_grid = component_grid
    _validate_native_dim(extractor, extractor_name, config)
    _validate_weights_id(extractor, extractor_name, config)

    dataset = ImageDataset(image_dir, item_ids, transform=extractor.transform)
    if len(dataset) == 0:
        # Fail loudly: extracting over zero items would write a degenerate
        # .npy that downstream steps then skip forever as "already exists".
        raise RuntimeError(
            f"No images found in {image_dir} for dataset '{dataset_name}' "
            f"({len(item_ids)} items expected). Check paths.data_raw."
        )
    extract_settings = resolve_dataloader_settings(load_config())
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=extract_settings.num_workers,
    )

    # Honour the configured checkpoints path (e.g. checkpoints/validation) instead
    # of a fixed 'checkpoints/' — otherwise a run under a different profile
    # resumes from another profile's stale extraction checkpoint.
    checkpoints_dir = config.get("paths", {}).get("checkpoints", "checkpoints")
    ckpt_base = f"{checkpoints_dir}/extraction/{dataset_name}_{extractor_name}"
    Path(ckpt_base).parent.mkdir(parents=True, exist_ok=True)

    if need_pooled:
        embeddings, extracted_ids = extractor.extract_batch(
            dataloader,
            checkpoint_path=ckpt_base,
            save_every=checkpoint_every,
        )
        extractor.save(embeddings, extracted_ids, str(pooled_path))
        _write_meta(extractor, extractor_name, pooled_path, {"kind": "pooled"})
        logger.info(
            "  %s: native pooled saved to %s (%s)", extractor_name, pooled_path, embeddings.shape
        )

    if need_components:
        # The streaming part-file lives NEXT TO the final artifact (not in
        # the checkpoints dir): data/ and checkpoints/ are separate bind
        # mounts in the container, and same-filesystem placement is what
        # lets save_components finalise by atomic rename instead of
        # rewriting a catalogue-sized matrix.
        components, comp_ids = extractor.extract_components_batch(
            dataloader,
            checkpoint_path=str(comp_path.with_suffix("")),
            save_every=checkpoint_every,
        )
        extractor.save_components(components, comp_ids, str(comp_path))
        _write_meta(
            extractor,
            extractor_name,
            comp_path,
            {
                "kind": "components",
                "n_components": int(components.shape[1]),
                "component_grid": component_grid,
                "pooling": "adaptive_avg" if component_grid is not None else None,
            },
        )
        logger.info(
            "  %s: native components saved to %s (%s)",
            extractor_name,
            comp_path,
            components.shape,
        )

    _project_pooled(pooled_path, projection, train_items)

    return True


def _components_needed(config: dict) -> bool:
    """Return ``True`` when component artifacts must be extracted.

    An explicit ``extract_components: true`` in the config always wins.
    Otherwise component extraction is auto-enabled when any recommender
    in ``recommenders_enabled`` declares ``requires_components`` (e.g.
    ACF) — so enabling such a model can never strand the train step on
    ``EnabledRecommenderHasNoCellsError`` just because the flag was off.
    Unregistered names are skipped here; the train step already fails
    loud on them.
    """
    if bool(config.get("extract_components", False)):
        return True

    # Local import: pulling in the recommender package registers every
    # model spec, and doing it lazily keeps extract importable without it.
    from src.recommenders import get_recommender_spec

    for name in config.get("recommenders_enabled") or []:
        try:
            spec = get_recommender_spec(name)
        except KeyError:
            continue
        if spec.requires_components:
            logger.info(
                "extract_components auto-enabled: recommender %r requires "
                "component embeddings (*_comp.npy).",
                name,
            )
            return True
    return False


def run() -> None:
    """Extract native-dim embeddings for every configured ``(extractor, dataset)``."""
    config = load_config()
    set_seed(config["seed"])

    device = resolve_device(config["device"])
    processed_dir = config["paths"]["data_processed"]
    embeddings_dir = config["paths"]["embeddings"]
    batch_size = config.get("batch_size", 64)
    checkpoint_every = config.get("checkpoint_every", 500)
    datasets = config.get("datasets", [])
    extract_components = _components_needed(config)
    component_grid = config.get("component_grid")
    if extract_components:
        logger.info(
            "component extraction: %s",
            f"pooled to a {component_grid}x{component_grid} grid (component_grid)"
            if component_grid
            else "native region map (component_grid unset)",
        )

    # Instantiating the manager guarantees the on-disk directories exist.
    CheckpointManager()

    enabled = config.get("extractors_enabled")
    if not enabled:
        logger.info(
            "extract step skipped: extractors_enabled is empty in "
            "configs/extractors.yaml. Add at least one name "
            "(e.g. resnet50) to enable extraction.",
        )
        return
    if not datasets:
        logger.info("extract step skipped: datasets list is empty in configs/default.yaml.")
        return

    unknown = [name for name in enabled if not is_registered(name)]
    if unknown:
        logger.warning(
            "extractors_enabled lists unregistered names (skipped): %s. Registered extractors: %s",
            ", ".join(sorted(unknown)),
            ", ".join(registered_extractor_names()),
        )
    extractors = {name: get_extractor_class(name) for name in enabled if is_registered(name)}
    if not extractors:
        logger.info(
            "extract step skipped: no extractor in extractors_enabled is registered.",
        )
        return

    for dataset_name in datasets:
        logger.info("=== Dataset: %s ===", dataset_name)

        item_ids = get_item_ids(processed_dir, dataset_name)
        image_dir = f"{config['paths']['data_raw']}/{dataset_name}/images"
        # Read once per dataset, and only when some extractor actually
        # asks for a train-only PCA projection.
        train_items: list[int] | None = None

        for extractor_name, extractor_cls in extractors.items():
            projection = resolve_projection_config(config, extractor_name)
            if projection is not None and projection.needs_fit and train_items is None:
                train_items = train_item_indices(processed_dir, dataset_name)

            with time_cell(
                "extract",
                dataset=dataset_name,
                extractor=extractor_name,
            ) as cell:
                did_work = _extract_for_config(
                    extractor_cls=extractor_cls,
                    extractor_name=extractor_name,
                    dataset_name=dataset_name,
                    image_dir=image_dir,
                    item_ids=item_ids,
                    embeddings_dir=embeddings_dir,
                    batch_size=batch_size,
                    checkpoint_every=checkpoint_every,
                    device=device,
                    config=config,
                    extract_components=extract_components,
                    projection=projection,
                    train_items=train_items,
                    component_grid=component_grid,
                )
                if not did_work:
                    cell.skip("embeddings already on disk")

    logger.info("Embedding extraction complete.")
