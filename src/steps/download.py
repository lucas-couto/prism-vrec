"""Step 01 — Download every configured dataset into ``data/raw``.

Resolves each name listed in ``datasets:`` through the
:mod:`src.data.base` registry, so swapping in a new dataset requires
nothing more than registering a new :class:`DatasetProvider`.
"""

from src.data import dvbpr  # noqa: F401
from src.data.base import get_dataset_provider
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run() -> None:
    """Download every dataset listed in ``configs/default.yaml``.

    Each dataset is materialised through its registered
    :class:`DatasetProvider`.  Providers are expected to be idempotent
    and to validate already-downloaded files (size / checksum) so a
    re-run skips work that has already been done correctly.
    """
    config = load_config()
    datasets = config.get("datasets", [])

    for dataset_name in datasets:
        logger.info("=== Downloading %s ===", dataset_name)
        provider = get_dataset_provider(dataset_name)
        provider.download()
        logger.info("%s: download complete.", dataset_name)

    logger.info("All downloads complete.")
