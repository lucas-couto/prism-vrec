"""Auto-discovery for user-provided datasets under ``plugins/datasets/``.

Drop a dataset directory under ``plugins/datasets/<name>/`` and the
pipeline picks it up automatically — no Python code, no edits to
``src/data/``.

Two layouts are supported.

Layout A — files already on disk
--------------------------------

::

    plugins/datasets/<name>/
    ├── interactions.csv      # columns: user_id, item_id, image_path
    └── images/               # one image per item (any extension)
        ├── it_001.jpg
        ├── it_002.jpg
        └── ...

The CSV is the user's interaction log: one row per ``(user, item)``
pair.  ``image_path`` is interpreted relative to the same directory
as ``interactions.csv``.

Layout B — describe URLs to download
------------------------------------

::

    plugins/datasets/<name>/
    └── source.yaml

with::

    interactions:
      url: https://example.com/interactions.csv
    images:
      url: https://example.com/images.tar.gz
    # Optional metadata
    categories:
      url: https://example.com/categories.json

The first run downloads the URLs into ``data/raw/<name>/`` and from
there behaves exactly like Layout A.  ``source.yaml`` may also point
at local paths via ``path: /absolute/path`` — useful when the dataset
is mounted from a Docker volume.

Plugging the dataset into the pipeline
--------------------------------------

After creating the directory, add ``"<name>"`` to the ``datasets:``
list in ``configs/default.yaml`` and run the pipeline.  The
auto-registration is opt-in by name: dropping a directory is not
enough, the YAML must explicitly mention it.  This keeps reruns
predictable and avoids accidentally pulling in scratch data.
"""

from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Callable

import yaml

from src.data.base import (
    DatasetProvider,
    register_dataset_provider,
    registered_dataset_names,
)
from src.data.example_csv import CSVDatasetProvider
from src.utils.logging import get_logger

logger = get_logger(__name__)


_USER_DATASETS_ROOT = Path("plugins/datasets")
_REQUIRED_DIRECT = ("interactions.csv", "images")
_SOURCE_FILE = "source.yaml"


def scan_user_datasets(root: Path | str = _USER_DATASETS_ROOT) -> list[str]:
    """Walk ``root`` and register every well-formed dataset directory.

    Returns the list of names that were registered (already-registered
    names are skipped silently so a re-import does not raise).  Does
    not register names that collide with built-in providers — built-in
    wins.
    """
    root = Path(root)
    if not root.exists():
        return []

    builtin = set(registered_dataset_names())
    registered: list[str] = []

    for entry in sorted(p for p in root.iterdir() if p.is_dir()):
        name = entry.name
        # Skip hidden directories and convention-private examples.  Any
        # directory whose name starts with ``.`` or ``_`` is reserved for
        # scaffolding / templates and is never auto-registered.
        if name.startswith((".", "_")):
            continue
        if name in builtin:
            logger.warning(
                "plugins/datasets/%s: name collides with a built-in provider; "
                "skipping (rename the directory if you wanted to override).",
                name,
            )
            continue

        factory = _resolve_factory(name, entry)
        if factory is None:
            continue

        register_dataset_provider(name, factory)
        registered.append(name)

    if registered:
        logger.info(
            "Auto-registered %d user dataset(s) under %s: %s",
            len(registered),
            root,
            ", ".join(registered),
        )
    return registered


def _resolve_factory(name: str, directory: Path) -> Callable[[], DatasetProvider] | None:
    """Inspect *directory* and return a provider factory or None."""
    direct = _check_direct_layout(directory)
    if direct is not None:
        csv_path, images_dir, categories_csv = direct
        return lambda: CSVDatasetProvider(
            name=name,
            interactions_csv=csv_path,
            images_dir=images_dir,
            categories_csv=categories_csv,
        )

    source_yaml = directory / _SOURCE_FILE
    if source_yaml.exists():
        return lambda: _build_from_source_yaml(name, directory, source_yaml)

    logger.warning(
        "plugins/datasets/%s: missing %s and missing %s — skipping.",
        name,
        "/".join(_REQUIRED_DIRECT),
        _SOURCE_FILE,
    )
    return None


def _check_direct_layout(directory: Path) -> tuple[Path, Path, Path | None] | None:
    """Return ``(interactions_csv, images_dir, categories_csv)`` when present.

    ``categories_csv`` is ``None`` when the optional ``categories.csv``
    file is absent — the dataset is still usable, just not for the
    fine-tuning step.
    """
    csv = directory / "interactions.csv"
    img = directory / "images"
    if csv.exists() and img.is_dir():
        cat = directory / "categories.csv"
        return csv, img, (cat if cat.exists() else None)
    return None


def _build_from_source_yaml(
    name: str,
    directory: Path,
    source_yaml: Path,
) -> DatasetProvider:
    """Materialise the dataset from a ``source.yaml`` declaration.

    The download (or copy from local path) only runs once per machine;
    after the files land in ``data/raw/<name>/`` the behaviour matches
    the direct layout.
    """
    raw_dir = Path("data/raw") / name
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_target = raw_dir / "interactions.csv"
    images_target = raw_dir / "images"

    if csv_target.exists() and images_target.is_dir():
        logger.info(
            "%s: source.yaml already materialised at %s, skipping fetch.",
            name,
            raw_dir,
        )
    else:
        with open(source_yaml, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        _fetch_resource(cfg.get("interactions", {}), csv_target)
        _fetch_resource(cfg.get("images", {}), images_target, expect_dir=True)

    return CSVDatasetProvider(
        name=name,
        interactions_csv=csv_target,
        images_dir=images_target,
    )


def _fetch_resource(spec: dict, target: Path, expect_dir: bool = False) -> None:
    """Resolve a ``{url: ...}`` or ``{path: ...}`` spec into ``target``.

    For ``url`` entries the file is downloaded with HTTP-Range resume
    support; tarballs / zip files are extracted in place when
    ``expect_dir`` is True.
    """
    if not spec:
        raise ValueError(
            f"source.yaml entry for {target.name!r} is missing — specify either 'url' or 'path'."
        )

    if "path" in spec:
        src = Path(spec["path"]).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Local path declared in source.yaml does not exist: {src}")
        if expect_dir:
            if src.is_dir():
                shutil.copytree(src, target, dirs_exist_ok=True)
            else:
                target.mkdir(parents=True, exist_ok=True)
                _extract_archive(src, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
        return

    if "url" not in spec:
        raise ValueError(f"source.yaml entry for {target.name!r} must contain 'url' or 'path'.")

    url = spec["url"]
    if expect_dir:
        target.mkdir(parents=True, exist_ok=True)
        archive_path = target.with_suffix(target.suffix + ".download")
        _stream_download(url, archive_path)
        _extract_archive(archive_path, target)
        archive_path.unlink(missing_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        _stream_download(url, target)


def _stream_download(url: str, target: Path) -> None:
    """Download ``url`` into ``target`` with a basic progress log."""
    import requests
    from tqdm import tqdm

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".partial")

    logger.info("Downloading %s -> %s", url, target)
    written = 0
    with requests.get(url, stream=True, timeout=600) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", "0") or 0)
        with (
            open(partial, "wb") as fout,
            tqdm(total=total or None, unit="B", unit_scale=True, desc=target.name) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=1 << 20):
                fout.write(chunk)
                written += len(chunk)
                pbar.update(len(chunk))

    # Verify completeness before promoting the .partial file. A
    # connection that ends early without raising would otherwise leave a
    # truncated file at the final name, and the exists()-based skip would
    # never re-fetch it.
    if total and written != total:
        partial.unlink(missing_ok=True)
        raise OSError(
            f"Incomplete download of {url}: got {written} of {total} bytes. Re-run to retry."
        )
    partial.rename(target)


def _extract_archive(archive: Path, target_dir: Path) -> None:
    """Best-effort extraction for tar / zip archives into ``target_dir``."""
    target_dir.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(target_dir)
        logger.info("Extracted %s into %s", archive.name, target_dir)
        return

    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            # The 'data' filter (path-traversal protection) exists on
            # CPython 3.12+ and 3.11.4+; on 3.11.0-3.11.3 the keyword is
            # absent and raises TypeError, so fall back to a plain
            # extract (plugin archives are trusted local content).
            try:
                tf.extractall(target_dir, filter="data")
            except TypeError:
                tf.extractall(target_dir)
        logger.info("Extracted %s into %s", archive.name, target_dir)
        return

    raise ValueError(
        f"Could not extract {archive}: not a recognised archive (zip/tar). "
        f"If you have a custom format, write a DatasetProvider subclass instead."
    )


scan_user_datasets()
