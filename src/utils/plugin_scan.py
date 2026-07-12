"""Shared plugin auto-discovery used by every ``plugins/<domain>/`` scanner.

The extractor, recommender and fusion domains each expose a public
``scan_user_*()`` that imports user modules so their import-time
``register_*`` calls populate the corresponding registry.  The mechanics
are identical across domains, so they live here once; each domain module
is a one-line wrapper that fixes ``package``, ``plugins_dir`` and the
human-readable ``kind``.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


def scan_plugins(package: str, plugins_dir: Path, kind: str) -> list[str]:
    """Import every non-private module under *plugins_dir* exactly once.

    Each plugin module is responsible for calling its domain's
    ``register_*`` function at import time; this function only triggers
    the imports and never inspects what was registered.

    Args:
        package: Dotted package name (e.g. ``"plugins.extractors"``).
        plugins_dir: Filesystem path to that package.
        kind: Singular noun for log messages (e.g. ``"extractor"``).

    Returns:
        The list of imported module names (informational). Modules whose
        name starts with ``_`` are skipped (private / scratch files).
    """
    if not plugins_dir.is_dir():
        return []

    # pkgutil needs a valid package; create __init__.py if the user did
    # not. A read-only filesystem (Docker volume mount) must not break
    # startup, so log and skip discovery rather than raising.
    init = plugins_dir / "__init__.py"
    if not init.exists():
        try:
            init.write_text(
                f'"""User-supplied {kind}s. Auto-discovered at import."""\n',
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning(
                "Cannot create %s (%r); skipping %s plugin discovery.",
                init,
                exc,
                kind,
            )
            return []

    imported: list[str] = []
    pkg = importlib.import_module(package)
    for module_info in pkgutil.iter_modules(pkg.__path__):
        name = module_info.name
        if name.startswith("_"):
            continue
        full_name = f"{package}.{name}"
        try:
            importlib.import_module(full_name)
            imported.append(name)
        except Exception as exc:  # noqa: BLE001 — one bad plugin must not block the rest
            logger.error(
                "Failed to import user %s %s: %s — skipping.",
                kind,
                full_name,
                exc,
            )

    if imported:
        logger.info(
            "Auto-registered %d user %s module(s): %s",
            len(imported),
            kind,
            ", ".join(imported),
        )
    return imported
