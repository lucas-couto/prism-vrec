"""Auto-discovery for user-supplied fusion strategies under ``plugins/fusions/``.

Drop a Python module into ``plugins/fusions/<your_strategy>.py`` that
calls :func:`register_fusion_strategy` at import time and the pipeline
picks it up automatically.

See :mod:`src.fusions.registry` for the API contract.
"""

from __future__ import annotations

from pathlib import Path

from src.utils.plugin_scan import scan_plugins

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
    return scan_plugins(_PLUGINS_PACKAGE, _PLUGINS_DIR, "fusion strategy")
