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
        except (FileNotFoundError, NotADirectoryError):
            pass

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


def _extract_for_config(
    extractor_cls,
    extractor_name: str,
    dataset_name: str,
    dim: int,
    image_dir: str,
    item_ids: list,
    embeddings_dir: str,
    batch_size: int,
    checkpoint_every: int,
    device: str,
) -> None:
    """Extract embeddings for a single ``(extractor, dataset, dim)`` cell."""
    output_path = Path(embeddings_dir) / dataset_name / f"{extractor_name}_D{dim}.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        logger.info("  %s D=%d: already exists, skipping.", extractor_name, dim)
        return

    logger.info("  Extracting %s D=%d...", extractor_name, dim)

    extractor = extractor_cls(device=device, output_dim=dim)

    dataset = ImageDataset(image_dir, item_ids, transform=extractor.transform)
    extract_settings = resolve_dataloader_settings(load_config())
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=extract_settings.num_workers,
    )

    ckpt_path = f"checkpoints/extraction/{dataset_name}_{extractor_name}_D{dim}"
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)

    embeddings, extracted_ids = extractor.extract_batch(
        dataloader,
        checkpoint_path=ckpt_path,
        save_every=checkpoint_every,
    )

    extractor.save(embeddings, extracted_ids, str(output_path))
    logger.info(
        "  %s D=%d: saved to %s (%s)",
        extractor_name,
        dim,
        output_path,
        embeddings.shape,
    )


def run() -> None:
    """Extract embeddings for every configured ``(extractor, dataset, dim)``."""
    config = load_config()
    set_seed(config["seed"])

    device = resolve_device(config["device"])
    processed_dir = config["paths"]["data_processed"]
    embeddings_dir = config["paths"]["embeddings"]
    projection_dims = config.get("projection_dims", [64, 128, 256])
    batch_size = config.get("batch_size", 64)
    checkpoint_every = config.get("checkpoint_every", 500)
    datasets = config.get("datasets", ["amazon_fashion", "amazon_women", "amazon_men"])

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
            for dim in projection_dims:
                with time_cell(
                    "extract",
                    dataset=dataset_name,
                    extractor=extractor_name,
                    dim=dim,
                ):
                    _extract_for_config(
                        extractor_cls=extractor_cls,
                        extractor_name=extractor_name,
                        dataset_name=dataset_name,
                        dim=dim,
                        image_dir=image_dir,
                        item_ids=item_ids,
                        embeddings_dir=embeddings_dir,
                        batch_size=batch_size,
                        checkpoint_every=checkpoint_every,
                        device=device,
                    )

    logger.info("Embedding extraction complete.")
