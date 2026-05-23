"""Auto-discovery for user-supplied recommenders under ``plugins/recommenders/``.

Drop a Python module into ``plugins/recommenders/<your_model>.py`` that
calls :func:`register_recommender` at import time and the pipeline picks
it up automatically.

See :mod:`src.recommenders.registry` for the API contract.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


# auto_register.py lives at src/recommenders/auto_register.py — the repo
# root is three parents up.  ``plugins/recommenders/`` sits at the repo
# root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PLUGINS_PACKAGE = "plugins.recommenders"
_PLUGINS_DIR = _REPO_ROOT / "plugins" / "recommenders"


def scan_user_recommenders() -> list[str]:
    """Import every module under ``plugins/recommenders/`` once.

    Each plugin module is responsible for calling
    :func:`register_recommender` at import time.
    """
    if not _PLUGINS_DIR.is_dir():
        return []

    init = _PLUGINS_DIR / "__init__.py"
    if not init.exists():
        try:
            init.write_text(
                '"""User-supplied recommenders. Auto-discovered at import."""\n',
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
                "Failed to import user recommender %s: %s — skipping.",
                full_name,
                exc,
            )

    if imported:
        logger.info(
            "Auto-registered %d user recommender module(s): %s",
            len(imported),
            ", ".join(imported),
        )
    return imported
