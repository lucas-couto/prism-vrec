"""I03 — fine-tuning resume must restore the HISTORICAL best, not only the current state.

F07: the resume checkpoint carried the current weights and ``best_acc``
but not the weights that produced ``best_acc``.  After a genuine
interruption (exception right after the durable save, before normal
cleanup), a resumed run with no later improvement returned the LAST
weights while reporting the earlier best accuracy.

The interruption here is genuine: the exception is injected inside the
per-epoch save path after the envelope has been committed, the trainer
never reaches its cleanup, and a NEW trainer instance resumes from the
file on disk.  Validation accuracies are scripted so the improvement
pattern is under control; everything else (optimizer, scheduler, RNG,
persistence) is real.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import src.finetuning.checkpoint as ft_ckpt
from src.finetuning.checkpoint import best_weights_path
from src.finetuning.trainer import FineTuner, FineTuningResult
from src.utils.checkpoint import ResumeStateError


class _ToyBackbone(nn.Module):
    def __init__(self, in_dim: int = 8, hidden_dim: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.projection(self.features(x))


class _Interrupt(RuntimeError):
    """Injected right after a durable save; stands in for a kill."""


def _loaders() -> tuple[DataLoader, DataLoader]:
    torch.manual_seed(0)
    x = torch.randn(16, 8)
    y = torch.randint(0, 3, (16,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=4, shuffle=True), DataLoader(ds, batch_size=4)


def _trainer(*, n_classes: int = 3, epochs_max: int = 4) -> FineTuner:
    torch.manual_seed(0)
    return FineTuner(
        backbone=_ToyBackbone(),
        extractor_name="toy",
        n_classes=n_classes,
        unfreeze_prefixes=["features.0"],
        device="cpu",
        config={"epochs_max": epochs_max, "patience": 10, "learning_rate": 1e-2},
    )


def _script(trainer: FineTuner, values: list[float]) -> None:
    it = iter(values)
    trainer._validate = lambda *a, **k: next(it)


def _train(
    trainer: FineTuner,
    ckpt: Path,
    monkeypatch,
    *,
    interrupt_after_envelopes: int | None = None,
) -> tuple[FineTuningResult | None, dict | None]:
    """Run ``train``; return (result, last envelope model_state written).

    With ``interrupt_after_envelopes=n`` the n-th envelope save is
    followed by an injected exception, leaving the files on disk.
    """
    real_write = ft_ckpt.atomic_write
    envelopes = {"n": 0, "last_state": None}

    def _write(fn, path, *args, **kwargs):
        real_write(fn, path, *args, **kwargs)
        if Path(path) != ckpt:
            return
        envelopes["n"] += 1
        saved = torch.load(ckpt, map_location="cpu", weights_only=False)
        envelopes["last_state"] = {k: v.clone() for k, v in saved["model_state"].items()}
        if interrupt_after_envelopes is not None and envelopes["n"] == interrupt_after_envelopes:
            raise _Interrupt(f"killed after envelope {envelopes['n']}")

    monkeypatch.setattr(ft_ckpt, "atomic_write", _write)
    if interrupt_after_envelopes is None:
        return trainer.train(*_loaders(), checkpoint_path=str(ckpt)), envelopes["last_state"]
    with pytest.raises(_Interrupt):
        trainer.train(*_loaders(), checkpoint_path=str(ckpt))
    return None, envelopes["last_state"]


def _assert_equal_state(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), f"weights diverge at {key}"


def _interrupt_after_epoch_two(tmp_path: Path, monkeypatch) -> Path:
    """Best at epoch one (0.9), worse at epoch two (0.5), killed after its save."""
    ckpt = tmp_path / "ft.pt"
    trainer = _trainer()
    _script(trainer, [0.9, 0.5])
    _train(trainer, ckpt, monkeypatch, interrupt_after_envelopes=2)
    assert ckpt.exists(), "the resume envelope must survive the interruption"
    assert best_weights_path(ckpt).exists()
    return ckpt


class TestResumeReturnsHistoricalBest:
    def test_no_later_improvement_returns_epoch_one_best(self, tmp_path, monkeypatch) -> None:
        continuous = _trainer()
        _script(continuous, [0.9, 0.5, 0.3, 0.2])
        result_full, last_full = _train(continuous, tmp_path / "full.pt", monkeypatch)

        ckpt = _interrupt_after_epoch_two(tmp_path, monkeypatch)
        resumed = _trainer()
        _script(resumed, [0.3, 0.2])
        result_resumed, last_resumed = _train(resumed, ckpt, monkeypatch)

        assert result_full.best_val_acc == result_resumed.best_val_acc == 0.9
        assert result_resumed.epochs_trained == 4
        # Returned weights: the epoch-one best, bit-identical to the
        # continuous run's (which never lost them).
        _assert_equal_state(result_resumed.model.state_dict(), result_full.model.state_dict())
        # Current trajectory: the last envelope (epoch four) matches too.
        _assert_equal_state(last_resumed, last_full)
        # The returned model is NOT the last epoch's weights.
        assert any(
            not torch.equal(last_resumed[k], v)
            for k, v in result_resumed.model.state_dict().items()
        )
        assert not ckpt.exists() and not best_weights_path(ckpt).exists()

    def test_later_improvement_replaces_best(self, tmp_path, monkeypatch) -> None:
        continuous = _trainer()
        _script(continuous, [0.9, 0.5, 0.95, 0.2])
        result_full, _ = _train(continuous, tmp_path / "full.pt", monkeypatch)

        ckpt = _interrupt_after_epoch_two(tmp_path, monkeypatch)
        resumed = _trainer()
        _script(resumed, [0.95, 0.2])
        result_resumed, _ = _train(resumed, ckpt, monkeypatch)

        assert result_full.best_val_acc == result_resumed.best_val_acc == 0.95
        _assert_equal_state(result_resumed.model.state_dict(), result_full.model.state_dict())


class TestResumeRefusesUntrustedState:
    def test_incompatible_identity_is_rejected(self, tmp_path, monkeypatch) -> None:
        ckpt = _interrupt_after_epoch_two(tmp_path, monkeypatch)
        other = _trainer(n_classes=4)
        _script(other, [0.3, 0.2])

        with pytest.raises(ResumeStateError, match="identity"):
            other.train(*_loaders(), checkpoint_path=str(ckpt))

    def test_missing_referenced_best_file_is_rejected(self, tmp_path, monkeypatch) -> None:
        ckpt = _interrupt_after_epoch_two(tmp_path, monkeypatch)
        best_weights_path(ckpt).unlink()
        resumed = _trainer()
        _script(resumed, [0.3, 0.2])

        with pytest.raises(ResumeStateError, match="missing"):
            resumed.train(*_loaders(), checkpoint_path=str(ckpt))

    def test_altered_best_file_is_rejected(self, tmp_path, monkeypatch) -> None:
        ckpt = _interrupt_after_epoch_two(tmp_path, monkeypatch)
        best_weights_path(ckpt).write_bytes(b"tampered")
        resumed = _trainer()
        _script(resumed, [0.3, 0.2])

        with pytest.raises(ResumeStateError, match="altered"):
            resumed.train(*_loaders(), checkpoint_path=str(ckpt))

    def test_legacy_envelope_is_rejected_not_guessed(self, tmp_path) -> None:
        ckpt = tmp_path / "legacy.pt"
        trainer = _trainer()
        torch.save(
            {
                "model_state": trainer.model.state_dict(),
                "optimizer_state": {},
                "scheduler_state": {},
                "epoch": 1,
                "best_acc": 0.9,
                "epochs_no_improve": 1,
            },
            ckpt,
        )

        with pytest.raises(ResumeStateError, match="legacy"):
            trainer.train(*_loaders(), checkpoint_path=str(ckpt))
