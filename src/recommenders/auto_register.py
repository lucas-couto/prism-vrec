"""Auto-discovery for user-supplied recommenders under ``plugins/recommenders/``.

Drop a Python module into ``plugins/recommenders/<your_model>.py`` that
calls :func:`register_recommender` at import time and the pipeline picks
it up automatically.

See :mod:`src.recommenders.registry` for the API contract.
"""

from __future__ import annotations

from pathlib import Path

from src.utils.plugin_scan import scan_plugins

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
    return scan_plugins(_PLUGINS_PACKAGE, _PLUGINS_DIR, "recommender")
