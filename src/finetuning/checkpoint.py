"""Persistence helpers for fine-tuned extractor checkpoints.

The checkpoint is a ``dict`` saved with :func:`torch.save` containing:

* ``backbone`` — state_dict of the backbone *without* the classification
  head (filtered by removing keys starting with ``projection.``).  This is
  what the rest of the pipeline consumes when re-extracting embeddings or
  transferring to a category-less dataset such as Tradesy.
* ``head`` — state_dict of the classification head only (``projection.*``
  keys).  Required by the post-hoc evaluator to compute top-K, F1 and
  confusion matrix.
* ``metadata`` — extractor name, dataset name, n_classes, in_features,
  best_val_acc, epochs_trained, early_stopped, split_seed, format_version.

A flat state_dict (the previous on-disk format) is still loadable through
:func:`load_finetuned`; in that case ``head`` and ``metadata`` come back
as ``None``.  The evaluator must handle the missing-head case explicitly.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_FORMAT_VERSION = "v2"
HEAD_PREFIX = "projection."


@dataclass(frozen=True)
class FineTuningMetadata:
    """Bookkeeping fields stored alongside the weights.

    All values are JSON-serialisable so the metadata can also be exported
    to a sidecar file when convenient.
    """

    extractor_name: str
    dataset_name: str
    n_classes: int
    in_features: int
    best_val_acc: float
    epochs_trained: int
    early_stopped: bool
    split_seed: int
    format_version: str = CHECKPOINT_FORMAT_VERSION
    extra: dict[str, Any] = field(default_factory=dict)


def split_state_dict(
    full_state: dict[str, torch.Tensor],
) -> tuple[OrderedDict, OrderedDict]:
    """Partition a model state_dict into (backbone_state, head_state).

    Keys starting with :data:`HEAD_PREFIX` go into ``head_state``; every
    other key goes into ``backbone_state``.  Insertion order is preserved
    in both halves so loading them back yields a model identical to the
    one that was saved.
    """
    backbone_state: OrderedDict = OrderedDict()
    head_state: OrderedDict = OrderedDict()
    for key, tensor in full_state.items():
        if key.startswith(HEAD_PREFIX):
            head_state[key] = tensor
        else:
            backbone_state[key] = tensor
    return backbone_state, head_state


def save_finetuned(
    path: str | Path,
    backbone_state: dict[str, torch.Tensor],
    head_state: dict[str, torch.Tensor],
    metadata: FineTuningMetadata,
) -> None:
    """Atomically write a fine-tuning checkpoint to *path*.

    The file is first written to a sibling ``.tmp`` and then renamed, so
    a crash mid-save never leaves a corrupt checkpoint behind.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "backbone": backbone_state,
        "head": head_state,
        "metadata": asdict(metadata),
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.rename(target)


def load_finetuned(
    path: str | Path,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None, dict[str, Any] | None]:
    """Load a fine-tuning checkpoint, transparently handling both formats.

    Returns
    -------
    (backbone_state, head_state, metadata)
        ``head_state`` and ``metadata`` are ``None`` for the legacy flat
        format (a state_dict saved directly without the wrapping dict).
    """
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if (
        isinstance(raw, dict)
        and "backbone" in raw
        and "head" in raw
        and not isinstance(raw["backbone"], torch.Tensor)
    ):
        return raw["backbone"], raw["head"], raw.get("metadata")
    return raw, None, None


def is_legacy_checkpoint(path: str | Path) -> bool:
    """Return ``True`` when *path* contains a flat state_dict (no head)."""
    _, head_state, _ = load_finetuned(path)
    return head_state is None
