"""Auto-discovery for user-supplied extractors under ``plugins/extractors/``.

Drop a Python module into ``plugins/extractors/<your_extractor>.py`` that
calls :func:`register_extractor` at import time and the pipeline picks
it up automatically — no edits to ``src/extractors/__init__.py``.

Layout::

    plugins/extractors/
    ├── __init__.py            # may be empty
    ├── my_extractor.py        # calls register_extractor("my_ext", MyExtractor)
    └── another.py

Minimal example (``plugins/extractors/my_extractor.py``)::

    import torch.nn as nn
    from torchvision.models import resnet18, ResNet18_Weights
    from src.extractors.base import BaseExtractor
    from src.extractors.registry import register_extractor

    class MyExtractor(BaseExtractor):
        def _build_model(self):
            backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            backbone.fc = nn.Linear(backbone.fc.in_features, self.output_dim)
            return backbone.to(self.device).eval()

        def _build_transform(self):
            return ResNet18_Weights.IMAGENET1K_V1.transforms()

    register_extractor("my_resnet18", MyExtractor)

Then add ``"my_resnet18"`` to ``extractors_enabled`` in
``configs/extractors.yaml`` and run the pipeline.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


# auto_register.py lives at src/extractors/auto_register.py — the repo
# root is three parents up.  ``plugins/extractors/`` sits at the repo
# root so user code is physically separated from the framework.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PLUGINS_PACKAGE = "plugins.extractors"
_PLUGINS_DIR = _REPO_ROOT / "plugins" / "extractors"


def scan_user_extractors() -> list[str]:
    """Import every module under ``plugins/extractors/`` once.

    Each plugin module is responsible for calling
    :func:`register_extractor` at import time.  This function does not
    inspect what was registered — it only triggers the imports.

    Returns the list of imported module names (informational).  Modules
    starting with ``_`` are skipped (private / scratch files).
    """
    if not _PLUGINS_DIR.is_dir():
        return []

    # Make sure the package itself imports cleanly even if the user did
    # not create an __init__.py — pkgutil needs a valid package path.
    init = _PLUGINS_DIR / "__init__.py"
    if not init.exists():
        try:
            init.write_text(
                '"""User-supplied extractors. Auto-discovered at import."""\n',
                encoding="utf-8",
            )
        except OSError:
            # Read-only filesystems (Docker volume mounts) should not
            # break startup; just skip the discovery in that case.
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
                "Failed to import user extractor %s: %s — skipping.",
                full_name,
                exc,
            )

    if imported:
        logger.info(
            "Auto-registered %d user extractor module(s): %s",
            len(imported),
            ", ".join(imported),
        )
    return imported
