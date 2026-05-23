"""Shared pytest fixtures and bootstrap for the framework test suite.

The tests live alongside (not inside) the ``src/`` package so the import
machinery is exercised the same way as in the running pipeline.  We add
the repository root to ``sys.path`` once here so every test module can
just ``import src.*`` without per-file boilerplate.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
