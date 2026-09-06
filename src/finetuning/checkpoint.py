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

from src.utils.atomic_io import atomic_write
from src.utils.checkpoint import (
    RESUME_ENVELOPE_VERSION,
    ResumeStateError,
    file_digest,
    identity_digest,
    validate_best_ref,
    validate_resume_envelope,
)

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

    Uses :func:`src.utils.atomic_io.atomic_write` (fsync + retried
    replace) so a crash mid-save never leaves a corrupt checkpoint and
    the rename survives networked-filesystem dirent lag.
    """
    payload = {
        "backbone": backbone_state,
        "head": head_state,
        "metadata": asdict(metadata),
    }
    atomic_write(lambda tmp: torch.save(payload, tmp), path)


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


# --- Resume envelope (C04, additive) -----------------------------------------
#
# The per-epoch RESUME checkpoint of :meth:`FineTuner.train` is distinct
# from the persistent fine-tuned weights above.  Version 2 binds the
# current state (model/optimizer/scheduler/scaler/RNG/epoch/patience) to
# a run identity and references the HISTORICAL best weights, kept in a
# sibling file committed (and digest-bound) before the envelope that
# names it.  A legacy envelope (no version) carries no best weights, so a
# resume from it cannot reproduce the uninterrupted run and is refused.

BEST_WEIGHTS_SUFFIX = ".best.pt"


def finetune_resume_identity(
    *,
    extractor_name: str,
    n_classes: int,
    in_features: int,
    unfreeze_prefixes: list[str],
    config: dict[str, Any],
    use_amp: bool,
) -> str:
    """Digest of what a fine-tuning resume envelope must agree on.

    Architecture identity (extractor, head size, trainable prefixes), the
    optimisation budget (learning rate, weight decay, ``epochs_max`` —
    it also fixes the cosine schedule — and patience) and the AMP regime.
    """
    return identity_digest(
        {
            "schema": 1,
            "extractor_name": extractor_name,
            "n_classes": n_classes,
            "in_features": in_features,
            "unfreeze_prefixes": list(unfreeze_prefixes),
            "config": {
                key: config.get(key)
                for key in ("learning_rate", "weight_decay", "epochs_max", "patience")
            },
            "amp": use_amp,
        }
    )


def best_weights_path(checkpoint_path: str | Path) -> Path:
    """Sibling file holding the historical best weights of a resume envelope."""
    path = Path(checkpoint_path)
    return path.with_name(path.name + BEST_WEIGHTS_SUFFIX)


def save_best_weights(
    path: str | Path,
    state: dict[str, torch.Tensor],
    *,
    identity: str,
    epoch: int,
    val_acc: float,
) -> dict[str, str]:
    """Atomically commit *state* (CPU tensors) and return its ``best_ref``.

    Must be called BEFORE the envelope that references the file, so the
    envelope can only ever point at bytes that exist.
    """
    payload = {"identity": identity, "epoch": epoch, "val_acc": val_acc, "model_state": state}
    atomic_write(lambda tmp: torch.save(payload, tmp), path)
    return {"path": str(path), "digest": file_digest(path)}


def load_best_weights(best_ref: Any, *, identity: str, source: str) -> dict[str, torch.Tensor]:
    """Validate a ``best_ref`` and return the referenced weights (CPU)."""
    path = validate_best_ref(best_ref, source=source)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("identity") != identity:
        raise ResumeStateError(f"{source}: best-weights file {path} belongs to another run.")
    return payload["model_state"]


def save_resume_envelope(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically write a v2 resume envelope (stamps ``envelope_version``)."""
    envelope = {**payload, "envelope_version": RESUME_ENVELOPE_VERSION}
    atomic_write(lambda tmp: torch.save(envelope, tmp), path)


def load_resume_envelope(path: str | Path, *, identity: str) -> dict[str, Any]:
    """Load and validate a v2 resume envelope bound to *identity*.

    Raises :class:`ResumeStateError` for a legacy envelope, a version or
    identity mismatch, or a missing required key.
    """
    raw = torch.load(path, map_location="cpu", weights_only=False)
    ckpt = validate_resume_envelope(raw, expected_identity=identity, source=f"resume {path}")
    for key in ("scheduler_state", "scaler_state", "epochs_no_improve"):
        if key not in ckpt:
            raise ResumeStateError(f"resume {path}: envelope is missing {[key]}.")
    return ckpt
