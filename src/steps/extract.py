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
from src.utils.atomic_io import atomic_write
from src.utils.checkpoint import CheckpointManager
from src.utils.config import load_config
from src.utils.dataloader import resolve_dataloader_settings
from src.utils.device import resolve_device
from src.utils.logging import get_logger
from src.utils.seed import set_seed
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
) -> None:
    """Extract native-dim embeddings for a single ``(extractor, dataset)`` cell.

    Writes the pooled ``<extractor>.npy`` at the backbone's native
    dimensionality plus a ``<extractor>.meta.json`` sidecar.  When
    ``extract_components`` is set and the extractor advertises
    ``supports_components``, additionally writes the 3-D
    ``<extractor>_comp.npy`` (native per-item components) consumed by
    ACF.  Both outputs are skipped independently when already present.
    """
    out_dir = Path(embeddings_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pooled_path = out_dir / f"{extractor_name}.npy"
    comp_path = out_dir / f"{extractor_name}_comp.npy"

    want_components = extract_components and getattr(extractor_cls, "supports_components", False)
    need_pooled = not pooled_path.exists()
    need_components = want_components and not comp_path.exists()

    if not need_pooled and not need_components:
        logger.info("  %s: already exists, skipping.", extractor_name)
        return

    logger.info("  Extracting %s (native dim)...", extractor_name)
    extractor = extractor_cls(device=device)
    _validate_native_dim(extractor, extractor_name, config)

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

    # Honour the configured checkpoints path (e.g. checkpoints/smoke) instead
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
        components, comp_ids = extractor.extract_components_batch(
            dataloader,
            checkpoint_path=f"{ckpt_base}_comp",
            save_every=checkpoint_every,
        )
        extractor.save_components(components, comp_ids, str(comp_path))
        _write_meta(
            extractor,
            extractor_name,
            comp_path,
            {"kind": "components", "n_components": int(components.shape[1])},
        )
        logger.info(
            "  %s: native components saved to %s (%s)",
            extractor_name,
            comp_path,
            components.shape,
        )


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
    extract_components = bool(config.get("extract_components", False))

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

        for extractor_name, extractor_cls in extractors.items():
            with time_cell(
                "extract",
                dataset=dataset_name,
                extractor=extractor_name,
            ):
                _extract_for_config(
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
                )

    logger.info("Embedding extraction complete.")
