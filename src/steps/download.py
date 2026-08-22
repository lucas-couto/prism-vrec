"""Step 01 — Download every configured dataset into ``data/raw``.

Resolves each name listed in ``datasets:`` through the
:mod:`src.data.base` registry, so swapping in a new dataset requires
nothing more than registering a new :class:`DatasetProvider`.

Each dataset is timed as its own cell, so ``step_timings.json`` answers
"which dataset cost the download hour" instead of reporting one opaque
window for all of them.  Every cell also carries the dataset's weight
on disk, which is the number a researcher planning storage or a mirror
actually wants.
"""

from pathlib import Path

from src.data import dvbpr  # noqa: F401
from src.data.base import get_dataset_provider
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.timing import time_cell

logger = get_logger(__name__)


def _dir_size_mb(path: Path) -> float:
    """Total size of every file under *path*, in MB (0.0 if absent).

    Symlinks are counted at their target's size only when the target is
    a regular file; broken links and unreadable entries are skipped
    rather than aborting a download that otherwise succeeded.

    :param path: Directory to measure, typically a provider's raw dir.
    :returns: Size in megabytes, rounded to three decimals.
    """
    total = 0
    for entry in path.rglob("*"):
        try:
            if entry.is_file():
                total += entry.stat().st_size
        except OSError:
            continue
    return round(total / 1e6, 3)


def run() -> None:
    """Download every dataset listed in ``configs/default.yaml``.

    Each dataset is materialised through its registered
    :class:`DatasetProvider`.  Providers are expected to be idempotent
    and to validate already-downloaded files (size / checksum) so a
    re-run skips work that has already been done correctly.

    A dataset already on disk is still recorded rather than skipped:
    re-validating a multi-gigabyte archive is real wall-time, and it is
    exactly the window a reader is looking for when a "no-op" re-run
    takes twenty minutes.  The ``downloaded_mb`` label separates the
    two cases — it is ``0.0`` when nothing new came over the network.
    """
    config = load_config()
    datasets = config.get("datasets", [])

    for dataset_name in datasets:
        logger.info("=== Downloading %s ===", dataset_name)
        provider = get_dataset_provider(dataset_name)
        raw_dir = Path(provider.raw_dir)
        size_before = _dir_size_mb(raw_dir)

        with time_cell("download", dataset=dataset_name) as cell:
            provider.download()
            size_after = _dir_size_mb(raw_dir)
            cell.label(size_mb=size_after, downloaded_mb=round(size_after - size_before, 3))

        logger.info(
            "%s: download complete (%.1f MB on disk, %.1f MB fetched).",
            dataset_name,
            size_after,
            size_after - size_before,
        )

    logger.info("All downloads complete.")
