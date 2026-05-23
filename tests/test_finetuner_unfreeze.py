"""Unit tests for the FineTuner's freeze/unfreeze accounting.

The trainer freezes every backbone parameter, then re-enables only the
modules whose names start with one of the configured ``unfreeze_prefixes``
(plus the freshly-installed classification head).  These tests pin that
behaviour against a tiny synthetic backbone — no real extractor is
loaded, so the suite stays fast and CPU-only.
"""

from __future__ import annotations

import torch.nn as nn

from src.finetuning.trainer import FineTuner


class _ToyBackbone(nn.Module):
    """Two-layer ``features`` + a 1-Linear ``projection`` head.

    Mirrors the contract documented on
    :class:`src.extractors.base.BaseExtractor`: the last submodule is
    named ``projection`` and exposes ``in_features`` so the trainer can
    swap it for a classification head.
    """

    def __init__(self, in_dim: int = 8, hidden_dim: int = 4) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),  # features.0
            nn.ReLU(),  # features.1
            nn.Linear(hidden_dim, hidden_dim),  # features.2
        )
        self.projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.projection(self.features(x))


def _new_trainer(unfreeze_prefixes: list[str]) -> FineTuner:
    return FineTuner(
        backbone=_ToyBackbone(),
        extractor_name="toy",
        n_classes=3,
        unfreeze_prefixes=unfreeze_prefixes,
        device="cpu",
        config={"epochs_max": 1, "patience": 1},
    )


def _frozen_param_names(model: nn.Module) -> set[str]:
    return {n for n, p in model.named_parameters() if not p.requires_grad}


def _trainable_param_names(model: nn.Module) -> set[str]:
    return {n for n, p in model.named_parameters() if p.requires_grad}


def test_no_prefixes_only_head_trains() -> None:
    trainer = _new_trainer(unfreeze_prefixes=[])

    trainable = _trainable_param_names(trainer.model)
    assert trainable == {"projection.weight", "projection.bias"}


def test_single_prefix_unfreezes_only_matching_layer() -> None:
    trainer = _new_trainer(unfreeze_prefixes=["features.2"])

    trainable = _trainable_param_names(trainer.model)
    assert "features.2.weight" in trainable
    assert "features.2.bias" in trainable
    assert "projection.weight" in trainable
    assert "features.0.weight" not in trainable


def test_multiple_prefixes_compose() -> None:
    trainer = _new_trainer(unfreeze_prefixes=["features.0", "features.2"])

    trainable = _trainable_param_names(trainer.model)
    expected = {
        "features.0.weight",
        "features.0.bias",
        "features.2.weight",
        "features.2.bias",
        "projection.weight",
        "projection.bias",
    }
    assert trainable == expected


def test_classification_head_replaces_original_projection() -> None:
    trainer = _new_trainer(unfreeze_prefixes=[])

    assert isinstance(trainer.model.projection, nn.Linear)
    assert trainer.model.projection.out_features == 3
    # in_features is captured before replacement and exposed for the
    # post-hoc evaluator to rebuild the head.
    assert trainer._proj_in_features == trainer.model.projection.in_features


def test_prefix_that_matches_nothing_leaves_only_head_trainable() -> None:
    trainer = _new_trainer(unfreeze_prefixes=["nonexistent.layer.99"])

    trainable = _trainable_param_names(trainer.model)
    assert trainable == {"projection.weight", "projection.bias"}

    frozen = _frozen_param_names(trainer.model)
    assert "features.0.weight" in frozen
    assert "features.2.weight" in frozen
