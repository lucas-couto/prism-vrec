"""Checkpoint manager with atomic writes.

All save operations write to a temporary file first and then rename it
into place.  This guarantees that a checkpoint file is either complete
or absent -- never partially written -- which protects against data
loss on power failure or unexpected termination.
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.utils.atomic_io import atomic_write

#: Version of the resume-envelope schema written by
#: :meth:`CheckpointManager.save_training_checkpoint` when an ``identity``
#: is supplied (and by the fine-tuning trainer).  Version 1 is the
#: implicit legacy shape without the field: it carries neither identity
#: nor a reference to the historical best weights, so it cannot be
#: resumed faithfully and is refused rather than guessed.
RESUME_ENVELOPE_VERSION = 2

#: Keys every v2 resume envelope must carry.
RESUME_ENVELOPE_REQUIRED_KEYS: tuple[str, ...] = (
    "envelope_version",
    "identity",
    "epoch",
    "model_state",
    "optimizer_state",
    "rng_states",
    "has_valid_observation",
    "best_metric",
    "best_epoch",
    "best_ref",
)


class ResumeStateError(RuntimeError):
    """A resume checkpoint cannot be trusted to continue the run.

    Raised for a legacy envelope (no version / no identity), an identity
    that does not match the run about to resume, a truncated payload or
    a referenced best-weights file that is missing or altered.  The
    caller must rerun the trial cleanly; the state is never patched
    from the current weights.
    """


class BestCheckpointError(RuntimeError):
    """A ``_best``/trial-best checkpoint is absent, unreadable or invalid.

    A selection winner that cannot be loaded is a failed run, never a
    silent skip: the cell would otherwise vanish from the expected set.
    """


class CheckpointManager:
    """Manages extraction, training, grid-search and evaluation checkpoints.

    Parameters
    ----------
    checkpoint_dir:
        Root directory for all checkpoint files.  Sub-directories
        (``extraction/``, ``training/``, ``grid_search/``) are created
        automatically as needed.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints") -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _atomic_save_torch(state: dict, path: Path) -> None:
        """Save *state* via :func:`torch.save` atomically and durably.

        Uses :func:`atomic_write` (fsync + retried ``os.replace``) so a
        networked-FS dirent-propagation lag does not turn a checkpoint
        save into a fatal ``FileNotFoundError`` on rename.
        """
        atomic_write(lambda tmp: torch.save(state, tmp), path)

    @staticmethod
    def _atomic_save_json(data: Any, path: Path) -> None:
        """Serialise *data* to JSON atomically and durably."""

        def _write(tmp: str) -> None:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)

        atomic_write(_write, path)

    def _extraction_dir(self) -> Path:
        d = self.checkpoint_dir / "extraction"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_extraction_progress(
        self,
        extractor_name: str,
        dataset: str,
        dim: int,
        batch_idx: int,
        partial_embeddings: np.ndarray,
        partial_ids: list,
    ) -> None:
        """Persist extraction progress so it can be resumed later.

        The checkpoint is stored as a ``.pt`` file under
        ``checkpoints/extraction/``.
        """
        ckpt_path = self._extraction_dir() / f"{extractor_name}_{dataset}_dim{dim}.pt"
        state = {
            "extractor_name": extractor_name,
            "dataset": dataset,
            "dim": dim,
            "batch_idx": batch_idx,
            "partial_embeddings": partial_embeddings,
            "partial_ids": partial_ids,
        }
        self._atomic_save_torch(state, ckpt_path)

    def load_extraction_progress(self, extractor_name: str, dataset: str, dim: int) -> dict | None:
        """Load a previously saved extraction checkpoint.

        Returns ``None`` if no checkpoint exists.
        """
        ckpt_path = self._extraction_dir() / f"{extractor_name}_{dataset}_dim{dim}.pt"
        if not ckpt_path.exists():
            return None
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)

    def _training_dir(self) -> Path:
        d = self.checkpoint_dir / "training"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def get_run_id(dataset: str, embedding: str, model: str, hyperparams: dict) -> str:
        """Compute a deterministic run identifier from the configuration.

        The identifier is the first 12 hex characters of the MD5 hash of
        a canonical JSON representation of the inputs.
        """
        config_dict = {
            "dataset": dataset,
            "embedding": embedding,
            "model": model,
            "hyperparams": hyperparams,
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode("utf-8")).hexdigest()[:12]

    def save_training_checkpoint(
        self,
        run_id: str,
        epoch: int,
        model_state: dict,
        optimizer_state: dict,
        best_metric: float,
        rng_states: dict,
        epochs_without_improvement: int = 0,
        *,
        identity: str | None = None,
        has_valid_observation: bool = False,
        best_epoch: int | None = None,
        best_ref: dict | None = None,
        scaler_state: dict | None = None,
    ) -> None:
        """Save a training checkpoint that can be used to resume later.

        Parameters
        ----------
        run_id:
            Unique identifier for the training run (see
            :meth:`get_run_id`).
        epoch:
            The epoch number that was just completed.
        model_state:
            The ``state_dict`` of the model.
        optimizer_state:
            The ``state_dict`` of the optimizer.
        best_metric:
            The best evaluation metric observed so far.
        rng_states:
            Dictionary containing RNG states.  Callers should populate
            this with ``torch``, ``numpy``, ``random`` and (optionally)
            ``cuda`` states for full reproducibility.
        epochs_without_improvement:
            Early stopping counter to restore on resume.
        identity:
            Digest of the scientific identity of the run (see
            :func:`identity_digest`).  When given, the file is written as
            a versioned v2 envelope carrying the remaining keyword fields;
            when ``None`` the legacy shape is written unchanged.
        has_valid_observation:
            Whether a finite selection metric has been observed yet.
            Distinguishes "no observation" from a legitimate ``0.0``.
        best_epoch:
            Epoch of the historical best observation (``None`` if none).
        best_ref:
            ``{"path": str, "digest": str}`` of the best-weights file,
            which must be committed BEFORE this envelope references it.
        scaler_state:
            AMP ``GradScaler`` state (empty dict when disabled).
        """
        ckpt_path = self._training_dir() / f"{run_id}.pt"
        state = {
            "run_id": run_id,
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "best_metric": best_metric,
            "epochs_without_improvement": epochs_without_improvement,
            "rng_states": rng_states,
        }
        if identity is not None:
            state.update(
                envelope_version=RESUME_ENVELOPE_VERSION,
                identity=identity,
                has_valid_observation=bool(has_valid_observation),
                best_epoch=best_epoch,
                best_ref=best_ref,
                scaler_state=scaler_state if scaler_state is not None else {},
            )
        self._atomic_save_torch(state, ckpt_path)

    def load_training_checkpoint(self, run_id: str) -> dict | None:
        """Load a training checkpoint.

        Returns ``None`` when no checkpoint exists for the given
        *run_id*.
        """
        ckpt_path = self._training_dir() / f"{run_id}.pt"
        if not ckpt_path.exists():
            return None
        return torch.load(ckpt_path, map_location="cpu", weights_only=False)

    def clear_training_checkpoint(self, run_id: str) -> None:
        """Delete the training checkpoint for *run_id*, if it exists."""
        ckpt_path = self._training_dir() / f"{run_id}.pt"
        # missing_ok avoids a TOCTOU FileNotFoundError when two processes
        # clear the same run_id concurrently.
        ckpt_path.unlink(missing_ok=True)

    def clear_all_training_checkpoints(self) -> int:
        """Delete every file in ``training/``. Returns the number removed.

        Intended for invocation at process startup. A ``SIGKILL`` (e.g.
        from a watchdog on memory preemption) bypasses the
        ``try/finally`` cleanup in :func:`train_single_run`, leaving
        orphaned ``.pt`` and ``.pt.tmp`` files behind. Optuna's RDB
        tracks completed trials independently, so any file left here
        is dead weight: the next trial samples new hyperparameters,
        producing a different ``run_id`` that ignores the stale file.
        """
        training_dir = self._training_dir()
        count = 0
        for entry in training_dir.iterdir():
            if entry.is_file():
                entry.unlink()
                count += 1
        return count

    def _grid_search_dir(self) -> Path:
        d = self.checkpoint_dir / "grid_search"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_grid_search_progress(self, experiment_key: str, completed_configs: list[dict]) -> None:
        """Save the list of completed grid-search configurations."""
        ckpt_path = self._grid_search_dir() / f"{experiment_key}.json"
        self._atomic_save_json(completed_configs, ckpt_path)

    def load_grid_search_progress(self, experiment_key: str) -> list[dict]:
        """Load previously completed grid-search configurations.

        Returns an empty list when no checkpoint exists.
        """
        ckpt_path = self._grid_search_dir() / f"{experiment_key}.json"
        if not ckpt_path.exists():
            return []
        with open(ckpt_path, encoding="utf-8") as fh:
            return json.load(fh)

    def save_evaluation_progress(self, results_so_far: list[dict], path: str) -> None:
        """Save intermediate evaluation results.

        Parameters
        ----------
        results_so_far:
            List of result dictionaries accumulated so far.
        path:
            Destination file path (JSON).
        """
        self._atomic_save_json(results_so_far, Path(path))

    def load_evaluation_progress(self, path: str) -> list[dict]:
        """Load previously saved evaluation results.

        Returns an empty list when *path* does not exist.
        """
        p = Path(path)
        if not p.exists():
            return []
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)


def capture_rng_states() -> dict:
    """Capture the current RNG states for all relevant libraries.

    The returned dictionary can be passed directly to
    :meth:`CheckpointManager.save_training_checkpoint` as
    ``rng_states``.
    """
    states: dict[str, Any] = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def restore_rng_states(rng_states: dict) -> None:
    """Restore RNG states previously captured by :func:`capture_rng_states`."""
    if "random" in rng_states:
        random.setstate(rng_states["random"])
    if "numpy" in rng_states:
        np.random.set_state(rng_states["numpy"])
    if "torch" in rng_states:
        torch.random.set_rng_state(rng_states["torch"])
    if "cuda" in rng_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_states["cuda"])


def identity_digest(payload: dict) -> str:
    """SHA-256 of the canonical JSON form of *payload* (sorted keys).

    Non-JSON values (e.g. ``Path``) are stringified.  Used to bind resume
    envelopes and best-weight files to the run that produced them.
    """
    key = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def file_digest(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of the bytes of *path*, streamed in ``chunk_size`` blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_resume_envelope(ckpt: Any, *, expected_identity: str, source: str) -> dict:
    """Check that *ckpt* is a v2 envelope bound to ``expected_identity``.

    Returns the envelope.  Raises :class:`ResumeStateError` for a legacy
    shape (no ``envelope_version``), an unsupported version, a missing
    required key or an identity mismatch.  A legacy envelope's historical
    best state is unknown, so it is refused rather than reconstructed.
    """
    if not isinstance(ckpt, dict) or "envelope_version" not in ckpt:
        raise ResumeStateError(
            f"{source}: legacy resume checkpoint without envelope_version; its "
            "historical best state is unknown. Delete it and rerun the trial cleanly."
        )
    version = ckpt["envelope_version"]
    if version != RESUME_ENVELOPE_VERSION:
        raise ResumeStateError(
            f"{source}: unsupported resume envelope version {version!r} "
            f"(expected {RESUME_ENVELOPE_VERSION})."
        )
    missing = [k for k in RESUME_ENVELOPE_REQUIRED_KEYS if k not in ckpt]
    if missing:
        raise ResumeStateError(f"{source}: resume envelope is missing {missing}.")
    if ckpt["identity"] != expected_identity:
        raise ResumeStateError(
            f"{source}: resume envelope identity {ckpt['identity'][:12]}... does not "
            f"match this run ({expected_identity[:12]}...); refusing to reuse it."
        )
    return ckpt


def validate_best_ref(best_ref: Any, *, source: str) -> Path:
    """Check that a ``best_ref`` names an existing, unaltered file.

    Returns its path.  Raises :class:`ResumeStateError` when the reference
    is malformed, the file is missing or its digest differs from the one
    recorded when the envelope was written.
    """
    if not isinstance(best_ref, dict) or "path" not in best_ref or "digest" not in best_ref:
        raise ResumeStateError(f"{source}: resume envelope has a malformed best_ref {best_ref!r}.")
    path = Path(best_ref["path"])
    if not path.exists():
        raise ResumeStateError(
            f"{source}: referenced best-weights file {path} is missing; the "
            "historical best cannot be recovered from the current weights."
        )
    actual = file_digest(path)
    if actual != best_ref["digest"]:
        raise ResumeStateError(
            f"{source}: referenced best-weights file {path} was altered "
            f"(digest {actual[:12]}... != {best_ref['digest'][:12]}...)."
        )
    return path


#: Keys a promoted / promotable best checkpoint must carry.
BEST_PAYLOAD_REQUIRED_KEYS: tuple[str, ...] = (
    "model_state",
    "hyperparams",
    "best_metric",
    "n_users",
    "n_items",
    "selection_fingerprint",
)


def load_best_checkpoint(path: str | Path, *, map_location: str = "cpu") -> dict:
    """Load and validate a best checkpoint written by the training step.

    Raises :class:`BestCheckpointError` when the file is absent, cannot
    be deserialised, lacks a required key or carries a non-finite
    ``best_metric``.  A finite ``0.0`` is a legitimate winner.
    """
    path = Path(path)
    if not path.exists():
        raise BestCheckpointError(f"best checkpoint {path} does not exist.")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except (RuntimeError, EOFError, OSError, ValueError, KeyError, pickle.UnpicklingError) as exc:
        raise BestCheckpointError(f"best checkpoint {path} is unreadable: {exc!r}") from exc
    if not isinstance(payload, dict):
        raise BestCheckpointError(f"best checkpoint {path} is not a payload dict.")
    if "model_state" not in payload and all(isinstance(v, torch.Tensor) for v in payload.values()):
        # Identified explicitly, never guessed: a flat state_dict carries
        # neither hyperparameters nor selection identity, so the model it
        # belongs to cannot be reconstructed with any confidence.
        raise BestCheckpointError(
            f"best checkpoint {path} is a legacy flat state_dict without hyperparams "
            "or selection identity; retrain the cell instead of guessing its dimensions."
        )
    missing = [k for k in BEST_PAYLOAD_REQUIRED_KEYS if k not in payload]
    if missing:
        raise BestCheckpointError(f"best checkpoint {path} is missing {missing}.")
    metric = payload["best_metric"]
    if isinstance(metric, bool) or not isinstance(metric, int | float) or not math.isfinite(metric):
        raise BestCheckpointError(f"best checkpoint {path} has invalid best_metric {metric!r}.")
    return payload
