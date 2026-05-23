"""Auto-discovery for user-supplied fusion strategies under ``plugins/fusions/``.

Drop a Python module into ``plugins/fusions/<your_strategy>.py`` that
calls :func:`register_fusion_strategy` at import time and the pipeline
picks it up automatically.

See :mod:`src.fusions.registry` for the API contract.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


# auto_register.py lives at src/fusions/auto_register.py — the repo
# root is three parents up.  ``plugins/fusions/`` sits at the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PLUGINS_PACKAGE = "plugins.fusions"
_PLUGINS_DIR = _REPO_ROOT / "plugins" / "fusions"


def scan_user_fusion_strategies() -> list[str]:
    """Import every module under ``plugins/fusions/`` once.

    Each plugin module is responsible for calling
    :func:`register_fusion_strategy` at import time.
    """
    if not _PLUGINS_DIR.is_dir():
        return []

    init = _PLUGINS_DIR / "__init__.py"
    if not init.exists():
        try:
            init.write_text(
                '"""User-supplied fusion strategies. Auto-discovered at import."""\n',
                encoding="utf-8",
            )
        except OSError:
            return []

    imported: list[str] = []
    package = importlib.import_module(_PLUGINS_PACKAGE)
    for module_info in pkgutil.iter_modules(package.__path__):
        name = module_info.name
        if name.startswith("_"):
            continue
        full_name = f"{_PLUGINS_PACKAGE}.{name}"
        try:
            importlib.import_module(full_name)
            imported.append(name)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to import user fusion strategy %s: %s — skipping.",
                full_name,
                exc,
            )

    if imported:
        logger.info(
            "Auto-registered %d user fusion module(s): %s",
            len(imported),
            ", ".join(imported),
        )
    return imported
