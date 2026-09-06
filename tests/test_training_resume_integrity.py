"""I04 — recommender training resume restores current AND historical state.

The interruption is genuine: a child process (spawn context) runs
``train_single_run`` with a ``CheckpointManager`` that calls
``os._exit`` right after the epoch-two envelope is durably written, so
the ``finally`` cleanup never runs and the files stay on disk exactly as
a SIGKILL would leave them.  The parent then resumes with a fresh model
and must reproduce the uninterrupted CPU run bit for bit: same returned
selection value, same promoted winner, same final current weights.

Everything is real (tiny BPR-MF, full-ranking selection ``Evaluator``,
Adam, sampler, persistence); nothing about the model is mocked.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path

import pytest
import torch

from src.recommenders.bpr import BPR
from src.utils.checkpoint import CheckpointManager, ResumeStateError
from src.utils.training import train_single_run

TRAIN = {0: {0, 1}, 1: {2, 3}, 2: {4}, 3: {5}, 4: {0, 5}}
VAL = {0: {2}, 1: {0}, 2: {1}, 3: {3}, 4: {2}}
N_USERS, N_ITEMS = 5, 6
HYPERPARAMS = {"learning_rate": 0.05, "latent_dim": 4, "l2_reg": 0.0}
EPOCHS = 4
KILL_AFTER_EPOCH = 1


class RecordingManager(CheckpointManager):
    """Real manager that also remembers every epoch it durably saved."""

    def __init__(self, checkpoint_dir: str) -> None:
        super().__init__(checkpoint_dir)
        self.saved_epochs: list[int] = []
        self.last_state: dict | None = None

    def save_training_checkpoint(self, run_id: str, epoch: int, model_state: dict, **kw) -> None:
        super().save_training_checkpoint(run_id, epoch, model_state, **kw)
        self.saved_epochs.append(epoch)
        self.last_state = {k: v.detach().clone() for k, v in model_state.items()}


class KillAfterSaveManager(RecordingManager):
    """Exits the process right after the durable save of ``KILL_AFTER_EPOCH``."""

    def save_training_checkpoint(self, run_id: str, epoch: int, model_state: dict, **kw) -> None:
        super().save_training_checkpoint(run_id, epoch, model_state, **kw)
        if epoch == KILL_AFTER_EPOCH:
            os._exit(0)  # no ``finally``: files stay exactly as a SIGKILL leaves them


def _config(root: Path, *, seed: int = 1, epochs: int = EPOCHS) -> dict:
    return {
        "seed": seed,
        "paths": {"results": str(root / "results")},
        "common": {
            "epochs": epochs,
            "batch_size": 4,
            "early_stopping_patience": 100,
            "early_stopping_metric": "ndcg@10",
            "eval_every_epochs": 1,
        },
    }


def _run(root: Path, manager: CheckpointManager, config: dict) -> float:
    return train_single_run(
        model_cls=BPR,
        model_name="bpr",
        n_users=N_USERS,
        n_items=N_ITEMS,
        visual_embeddings=None,
        train_interactions=TRAIN,
        selection_interactions=VAL,
        hyperparams=HYPERPARAMS,
        config=config,
        checkpoint_mgr=manager,
        dataset_name="ds",
        embedding_name="none",
        device="cpu",
    )


def _child_killed_run(root: str) -> None:
    torch.set_num_threads(1)
    _run(Path(root), KillAfterSaveManager(str(Path(root) / "ckpt")), _config(Path(root)))
    os._exit(3)  # unreachable: the manager must have exited first


def _run_id() -> str:
    return CheckpointManager.get_run_id("ds", "none", "bpr", HYPERPARAMS)


def _resume_file(root: Path) -> Path:
    return root / "ckpt" / "training" / f"{_run_id()}.pt"


def _trial_file(root: Path) -> Path:
    return root / "results" / "models" / "ds" / f"bpr_none_trial_{_run_id()}.pt"


def _best_payload(root: Path) -> dict:
    path = root / "results" / "models" / "ds" / "bpr_none_best.pt"
    return torch.load(path, map_location="cpu", weights_only=False)


def _assert_equal_state(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), f"weights diverge at {key}"


@pytest.fixture
def killed(tmp_path: Path) -> Path:
    """Run the interrupted branch in a child process; return its root."""
    root = tmp_path / "interrupted"
    root.mkdir()
    proc = mp.get_context("spawn").Process(target=_child_killed_run, args=(str(root),))
    proc.start()
    proc.join(timeout=300)
    assert proc.exitcode == 0, f"child did not exit through the injected kill ({proc.exitcode})"
    assert _resume_file(root).exists(), "the resume envelope must survive the kill"
    assert _trial_file(root).exists(), "the trial-local best must survive the kill"
    return root


class TestGenuineInterruptionMatchesContinuousRun:
    def test_resumed_run_reproduces_continuous_cpu_run(self, tmp_path, killed) -> None:
        torch.set_num_threads(1)
        continuous_root = tmp_path / "continuous"
        continuous_mgr = RecordingManager(str(continuous_root / "ckpt"))
        continuous_value = _run(continuous_root, continuous_mgr, _config(continuous_root))

        resumed_mgr = RecordingManager(str(killed / "ckpt"))
        resumed_value = _run(killed, resumed_mgr, _config(killed))

        # Resumed from the completed epoch boundary: only the remaining
        # epochs ran, from restored optimizer / RNG / early-stop state.
        assert resumed_mgr.saved_epochs == list(range(KILL_AFTER_EPOCH + 1, EPOCHS))
        assert continuous_mgr.saved_epochs == list(range(EPOCHS))
        assert resumed_value == continuous_value
        # Historical best: same promoted winner (metric and weights).
        best_c, best_r = _best_payload(continuous_root), _best_payload(killed)
        assert best_r["best_metric"] == best_c["best_metric"]
        _assert_equal_state(best_r["model_state"], best_c["model_state"])
        # Current trajectory: same final weights after the last epoch.
        _assert_equal_state(resumed_mgr.last_state, continuous_mgr.last_state)
        # Cleanup after normal completion.
        assert not _resume_file(killed).exists() and not _trial_file(killed).exists()

    def test_envelope_carries_the_v2_fields(self, killed) -> None:
        ckpt = torch.load(_resume_file(killed), map_location="cpu", weights_only=False)

        assert ckpt["envelope_version"] == 2
        assert ckpt["epoch"] == KILL_AFTER_EPOCH
        assert ckpt["has_valid_observation"] is True
        assert ckpt["best_ref"]["path"] == str(_trial_file(killed))
        assert ckpt["scaler_state"] == {}  # CPU: disabled scaler, saved anyway
        assert {"optimizer_state", "rng_states", "epochs_without_improvement"} <= set(ckpt)


class TestResumeRefusesUntrustedState:
    def test_incompatible_identity_is_rejected(self, killed) -> None:
        # Same run_id (dataset/model/embedding/hyperparams), different seed.
        with pytest.raises(ResumeStateError, match="identity"):
            _run(killed, RecordingManager(str(killed / "ckpt")), _config(killed, seed=2))

    def test_changed_selection_budget_is_rejected(self, killed) -> None:
        with pytest.raises(ResumeStateError, match="identity"):
            _run(killed, RecordingManager(str(killed / "ckpt")), _config(killed, epochs=8))

    def test_missing_referenced_best_file_is_rejected(self, killed) -> None:
        _trial_file(killed).unlink()

        with pytest.raises(ResumeStateError, match="missing"):
            _run(killed, RecordingManager(str(killed / "ckpt")), _config(killed))

    def test_altered_referenced_best_file_is_rejected(self, killed) -> None:
        _trial_file(killed).write_bytes(b"tampered")

        with pytest.raises(ResumeStateError, match="altered"):
            _run(killed, RecordingManager(str(killed / "ckpt")), _config(killed))

    def test_legacy_envelope_is_rejected_not_guessed(self, tmp_path) -> None:
        mgr = CheckpointManager(str(tmp_path / "ckpt"))
        model = BPR(N_USERS, N_ITEMS, None, {"latent_dim": 4})
        mgr.save_training_checkpoint(
            _run_id(),
            epoch=1,
            model_state=model.state_dict(),
            optimizer_state={},
            best_metric=0.0,
            rng_states={},
        )

        with pytest.raises(ResumeStateError, match="legacy"):
            _run(tmp_path, mgr, _config(tmp_path))
