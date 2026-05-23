"""Pipeline steps.

Each module exposes a single ``run`` callable so the pipeline can be
composed programmatically by ``main.py`` instead of going through the
filesystem with subprocess calls.

The steps are intentionally independent: every ``run`` function loads
the configuration, performs its work, and returns ``None``.  Skipping
already-completed work is the responsibility of each step (via
checkpoint files, ``output_path.exists()`` guards, etc.).
"""

from src.steps import (
    download,
    evaluate,
    evaluate_finetuning,
    export_best,
    extract,
    finetune,
    fuse,
    preprocess,
    statistical,
    train,
)

__all__ = [
    "download",
    "preprocess",
    "extract",
    "finetune",
    "evaluate_finetuning",
    "fuse",
    "train",
    "evaluate",
    "statistical",
    "export_best",
]
