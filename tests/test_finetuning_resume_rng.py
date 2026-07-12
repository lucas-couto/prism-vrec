"""Fine-tuning resume must be bit-identical to an uninterrupted run.

The resume checkpoint persists RNG states so a run interrupted after
epoch k and resumed draws the same shuffle / augmentation sequence as a
run that never stopped.  These tests use a tiny synthetic backbone and a
tensor-backed dataset — no real extractor, no image files, CPU only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.finetuning.trainer import FineTuner


class _ToyBackbone(nn.Module):
    def __init__(self, in_dim: int = 8, hidden_dim: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.projection(self.features(x))


def _loaders() -> tuple[DataLoader, DataLoader]:
    torch.manual_seed(0)
    x = torch.randn(16, 8)
    y = torch.randint(0, 3, (16,))
    ds = TensorDataset(x, y)
    # shuffle=True on the train loader is exactly what makes RNG state
    # matter across a resume boundary.
    return DataLoader(ds, batch_size=4, shuffle=True), DataLoader(ds, batch_size=4)


def _trainer(unfreeze: list[str] | None = None) -> FineTuner:
    return FineTuner(
        backbone=_ToyBackbone(),
        extractor_name="toy",
        n_classes=3,
        unfreeze_prefixes=unfreeze or ["features.0"],
        device="cpu",
        config={"epochs_max": 4, "patience": 10, "learning_rate": 1e-2},
    )


def test_resume_checkpoint_contains_rng_states(tmp_path, monkeypatch) -> None:
    # train() removes the resume checkpoint on clean completion; suppress
    # that single cleanup unlink so the last per-epoch checkpoint survives
    # for inspection. atomic_write only unlinks its temp on the error path,
    # so a successful save is unaffected.
    ckpt = tmp_path / "ft_ckpt.pt"
    trainer = _trainer()
    trainer.config["epochs_max"] = 2

    from pathlib import Path as _P

    monkeypatch.setattr(_P, "unlink", lambda self, *a, **k: None)
    trainer.train(*_loaders(), checkpoint_path=str(ckpt))

    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "rng_states" in saved
    assert set(saved["rng_states"]) >= {"random", "numpy", "torch"}
    assert "scaler_state" in saved


def test_resumed_run_matches_uninterrupted_run(tmp_path) -> None:
    # Uninterrupted 4-epoch run.
    full = _trainer()
    result_full = full.train(*_loaders(), checkpoint_path=str(tmp_path / "full.pt"))

    # Interrupted run: 2 epochs, keep the checkpoint, then resume for the
    # remaining epochs into a fresh trainer.
    interrupted = _trainer()
    interrupted.config["epochs_max"] = 2
    ckpt = tmp_path / "resume.pt"
    interrupted.train(*_loaders(), checkpoint_path=str(ckpt))

    resumed = _trainer()  # epochs_max back to 4
    result_resumed = resumed.train(*_loaders(), checkpoint_path=str(ckpt))

    # Same final weights (bit-identical) => same best val accuracy and
    # identical projection weights.
    assert result_full.best_val_acc == result_resumed.best_val_acc
    for (kf, vf), (kr, vr) in zip(
        result_full.model.state_dict().items(),
        result_resumed.model.state_dict().items(),
        strict=True,
    ):
        assert kf == kr
        assert torch.equal(vf, vr), f"weights diverge at {kf}"
