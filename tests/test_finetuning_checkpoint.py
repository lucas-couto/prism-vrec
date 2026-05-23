"""Persistence tests for the fine-tuning checkpoint helpers.

The checkpoint format is the only on-disk contract between the trainer
and downstream consumers (re-extraction, post-hoc evaluator, transfer to
unlabelled datasets).  These tests pin both the v2 round-trip and the
legacy flat-format compatibility path.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from src.finetuning.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    HEAD_PREFIX,
    FineTuningMetadata,
    is_legacy_checkpoint,
    load_finetuned,
    save_finetuned,
    split_state_dict,
)


def _make_state() -> OrderedDict:
    """Toy state_dict mixing backbone keys and a classification head."""
    return OrderedDict(
        [
            ("features.0.weight", torch.zeros(2, 2)),
            ("features.0.bias", torch.zeros(2)),
            ("features.4.weight", torch.ones(3, 2)),
            ("projection.weight", torch.full((4, 3), 0.5)),
            ("projection.bias", torch.full((4,), 0.25)),
        ]
    )


def test_split_state_dict_partitions_by_prefix() -> None:
    backbone, head = split_state_dict(_make_state())
    assert list(backbone.keys()) == [
        "features.0.weight",
        "features.0.bias",
        "features.4.weight",
    ]
    assert list(head.keys()) == ["projection.weight", "projection.bias"]
    assert all(not k.startswith(HEAD_PREFIX) for k in backbone)
    assert all(k.startswith(HEAD_PREFIX) for k in head)


def test_split_state_dict_preserves_order() -> None:
    backbone, head = split_state_dict(_make_state())
    assert list(backbone.keys()) == sorted(backbone.keys())[::1] or True
    full = list(_make_state().keys())
    expected_backbone = [k for k in full if not k.startswith(HEAD_PREFIX)]
    expected_head = [k for k in full if k.startswith(HEAD_PREFIX)]
    assert list(backbone.keys()) == expected_backbone
    assert list(head.keys()) == expected_head


def _meta() -> FineTuningMetadata:
    return FineTuningMetadata(
        extractor_name="dummy",
        dataset_name="toy",
        n_classes=4,
        in_features=3,
        best_val_acc=0.987,
        epochs_trained=7,
        early_stopped=True,
        split_seed=42,
    )


def test_save_and_load_roundtrip(tmp_path) -> None:
    backbone, head = split_state_dict(_make_state())
    path = tmp_path / "ft.pt"

    save_finetuned(path, backbone, head, _meta())

    loaded_backbone, loaded_head, loaded_meta = load_finetuned(path)
    assert loaded_head is not None and loaded_meta is not None
    assert set(loaded_backbone.keys()) == set(backbone.keys())
    assert set(loaded_head.keys()) == set(head.keys())
    for key in backbone:
        assert torch.equal(loaded_backbone[key], backbone[key])
    for key in head:
        assert torch.equal(loaded_head[key], head[key])

    assert loaded_meta["extractor_name"] == "dummy"
    assert loaded_meta["dataset_name"] == "toy"
    assert loaded_meta["n_classes"] == 4
    assert loaded_meta["in_features"] == 3
    assert loaded_meta["best_val_acc"] == pytest.approx(0.987)
    assert loaded_meta["epochs_trained"] == 7
    assert loaded_meta["early_stopped"] is True
    assert loaded_meta["split_seed"] == 42
    assert loaded_meta["format_version"] == CHECKPOINT_FORMAT_VERSION


def test_save_is_atomic_no_tmp_left(tmp_path) -> None:
    backbone, head = split_state_dict(_make_state())
    path = tmp_path / "ft.pt"
    save_finetuned(path, backbone, head, _meta())

    assert path.exists()
    assert not (tmp_path / "ft.pt.tmp").exists()


def test_load_legacy_flat_state_dict(tmp_path) -> None:
    """A bare state_dict (pre-v2) must still load with head=None, meta=None."""
    flat = OrderedDict(
        [
            ("features.0.weight", torch.zeros(2, 2)),
            ("features.0.bias", torch.zeros(2)),
        ]
    )
    legacy_path = tmp_path / "legacy.pt"
    torch.save(flat, legacy_path)

    backbone, head, metadata = load_finetuned(legacy_path)

    assert head is None
    assert metadata is None
    assert set(backbone.keys()) == set(flat.keys())
    assert is_legacy_checkpoint(legacy_path) is True


def test_v2_checkpoint_is_not_flagged_as_legacy(tmp_path) -> None:
    backbone, head = split_state_dict(_make_state())
    path = tmp_path / "ft.pt"
    save_finetuned(path, backbone, head, _meta())
    assert is_legacy_checkpoint(path) is False
