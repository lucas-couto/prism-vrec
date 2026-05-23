"""Step 02 — Materialise the pipeline's canonical on-disk layout.

For every dataset listed in ``configs/default.yaml`` this step calls the
registered :class:`DatasetProvider` to:

* write ``data/processed/<name>/{train,val,test}.csv``
* write ``data/processed/<name>/{user,item}2idx.json``
* extract per-item images into ``data/raw/<name>/images/``

The downstream steps (extract, finetune, fuse, train, evaluate) are
dataset-agnostic and only read this canonical layout.
"""

from pathlib import Path

from src.data import dvbpr  # noqa: F401
from src.data.base import get_dataset_provider
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run() -> None:
    """Process every dataset listed in ``configs/default.yaml``.

    Splits already on disk are detected via the presence of
    ``train.csv``/``val.csv``/``test.csv`` and skipped.  Image
    extraction is skipped when the destination directory already
    contains JPEG files.
    """
    config = load_config()
    raw_dir = config["paths"]["data_raw"]
    processed_dir = config["paths"]["data_processed"]
    datasets = config.get("datasets", [])

    for dataset_name in datasets:
        output_dir = Path(processed_dir) / dataset_name

        if (
            (output_dir / "train.csv").exists()
            and (output_dir / "val.csv").exists()
            and (output_dir / "test.csv").exists()
        ):
            logger.info("%s: already processed, skipping.", dataset_name)
        else:
            logger.info("=== Preprocessing %s ===", dataset_name)
            provider = get_dataset_provider(dataset_name)
            provider.save_processed(output_dir)
            logger.info("%s: preprocessing complete.", dataset_name)

        image_dir = Path(raw_dir) / dataset_name / "images"
        n_existing = len(list(image_dir.glob("*.jpg"))) if image_dir.exists() else 0
        if n_existing > 0:
            logger.info(
                "%s: %d images already extracted, skipping.",
                dataset_name,
                n_existing,
            )
        else:
            logger.info("=== Extracting images %s ===", dataset_name)
            provider = get_dataset_provider(dataset_name)
            provider.extract_images(image_dir)
            logger.info("%s: image extraction complete.", dataset_name)

    logger.info("All datasets preprocessed.")
