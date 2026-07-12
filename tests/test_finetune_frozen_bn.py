"""Pin the LeViT-256 fine-tuning failure mode (frozen-BN corruption).

Root cause (investigation "Parte C"): ``FineTuner.train`` used a bare
``model.train()``, which puts every BatchNorm layer in train mode — so
BN layers in the FROZEN stages kept updating ``running_mean`` /
``running_var`` on the fine-tuning data even though their weights were
frozen.  BN-dense backbones (LeViT-256 is BN-everywhere) had their
frozen stages silently rewritten; measured drift on the LeViT stem BN
exceeded 12 sigma after a single epoch.  A missing ``drop_last`` made
it worse: a final train batch of exactly 1 sample injects single-image
statistics at momentum 0.1.

The hypotheses about timm head handling (dual ``head``/``head_dist``,
``NormLinear``, distilled tuple forward) do NOT apply to the v2 code:
the wrapper builds timm models with ``num_classes=0`` (both heads become
``nn.Identity``) and attaches its own ``projection``; those invariants
are pinned by the LeViT integration test below.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.finetuning.trainer import FineTuner
from src.steps.finetune import drop_degenerate_tail

FT_CONFIG = {
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs_max": 1,
    "patience": 1,
}


class _ToyBNBackbone(nn.Module):
    """Minimal BN-bearing backbone honouring the extractor contract."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4))
        self.stage = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.BatchNorm2d(4))
        self.projection = nn.Identity()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stage(self.stem(x))
        return self.projection(feat.mean(dim=(2, 3)))


def _loader(n: int, batch_size: int, n_classes: int = 3) -> DataLoader:
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(n, 3, 8, 8, generator=generator)
    y = torch.randint(0, n_classes, (n,), generator=generator)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def _bn_stats(module: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    bn = module[1]
    return bn.running_mean.clone(), bn.running_var.clone()


class TestFrozenBatchNormPreserved:
    def test_frozen_stage_bn_running_stats_do_not_drift(self) -> None:
        backbone = _ToyBNBackbone()
        stem_mean_before, stem_var_before = _bn_stats(backbone.stem)

        ft = FineTuner(
            backbone=backbone,
            extractor_name="toy",
            n_classes=3,
            unfreeze_prefixes=["stage"],
            device="cpu",
            config=FT_CONFIG,
            in_features=4,
        )
        ft.train(_loader(9, 4), _loader(4, 4), checkpoint_path=None)

        stem_mean_after, stem_var_after = _bn_stats(ft.model.stem)
        torch.testing.assert_close(stem_mean_after, stem_mean_before)
        torch.testing.assert_close(stem_var_after, stem_var_before)

    def test_unfrozen_stage_bn_running_stats_do_update(self) -> None:
        backbone = _ToyBNBackbone()
        stage_mean_before, _ = _bn_stats(backbone.stage)

        ft = FineTuner(
            backbone=backbone,
            extractor_name="toy",
            n_classes=3,
            unfreeze_prefixes=["stage"],
            device="cpu",
            config=FT_CONFIG,
            in_features=4,
        )
        ft.train(_loader(9, 4), _loader(4, 4), checkpoint_path=None)

        stage_mean_after, _ = _bn_stats(ft.model.stage)
        assert not torch.allclose(stage_mean_after, stage_mean_before)

    def test_size_one_final_batch_trains_without_error(self) -> None:
        backbone = _ToyBNBackbone()
        ft = FineTuner(
            backbone=backbone,
            extractor_name="toy",
            n_classes=3,
            unfreeze_prefixes=["stage"],
            device="cpu",
            config=FT_CONFIG,
            in_features=4,
        )
        # 9 items / batch 4 -> final batch of 1 (drop_last absent here on
        # purpose: the trainer itself must survive a degenerate batch).
        result = ft.train(_loader(9, 4), _loader(4, 4), checkpoint_path=None)
        assert result.epochs_trained == 1


class TestDropDegenerateTail:
    @pytest.mark.parametrize(
        ("n_items", "batch_size", "expected"),
        [
            (129, 128, True),  # tail of exactly 1 -> drop
            (5, 2, True),  # tail of exactly 1 -> drop
            (161, 128, False),  # tail of 33 -> keep
            (128, 128, False),  # single full batch -> keep
            (100, 128, False),  # dataset smaller than one batch -> keep
            (1, 128, False),  # never drop the entire dataset
            (4, 2, False),  # even split -> keep
        ],
    )
    def test_drop_rule(self, n_items: int, batch_size: int, expected: bool) -> None:
        assert drop_degenerate_tail(n_items, batch_size) is expected


class TestLeViTFineTuneIntegration:
    """Pin the refuted head hypotheses + the BN fix on the real LeViT arch."""

    @pytest.fixture(scope="class")
    def levit_backbone(self) -> nn.Module:
        timm = pytest.importorskip("timm")

        class _Backbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                # pretrained=False: architecture-only, no network access.
                self.backbone = timm.create_model("levit_256", pretrained=False, num_classes=0)
                for param in self.backbone.parameters():
                    param.requires_grad = False
                self.projection = nn.Identity()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.projection(self.backbone(x))

        return _Backbone()

    def test_num_classes_zero_leaves_identity_heads(self, levit_backbone: nn.Module) -> None:
        # H1/H2: with num_classes=0 BOTH timm heads are Identity — there
        # is no NormLinear left to mis-replace.
        assert isinstance(levit_backbone.backbone.head, nn.Identity)
        assert isinstance(levit_backbone.backbone.head_dist, nn.Identity)

    def test_train_mode_forward_returns_tensor(self, levit_backbone: nn.Module) -> None:
        # H5: distilled_training defaults to False -> no tuple output.
        assert levit_backbone.backbone.distilled_training is False
        levit_backbone.train()
        out = levit_backbone(torch.randn(1, 3, 224, 224))
        assert torch.is_tensor(out)
        assert out.shape == (1, 512)  # H4: native dim is 512, not 256

    def test_finetune_size_one_batch_preserves_frozen_bn(self, levit_backbone: nn.Module) -> None:
        state = levit_backbone.state_dict()
        frozen_bn_keys = [
            k for k in state if "running_mean" in k and not k.startswith("backbone.stages.2")
        ]
        assert frozen_bn_keys, "LeViT must expose frozen-stage BN buffers"
        before = {k: state[k].clone() for k in frozen_bn_keys}

        ft = FineTuner(
            backbone=levit_backbone,
            extractor_name="levit_256",
            n_classes=3,
            unfreeze_prefixes=["backbone.stages.2"],
            device="cpu",
            config=FT_CONFIG,
            in_features=512,
        )
        generator = torch.Generator().manual_seed(0)
        x = torch.randn(5, 3, 224, 224, generator=generator)
        y = torch.randint(0, 3, (5,), generator=generator)
        train = DataLoader(TensorDataset(x, y), batch_size=2)  # final batch of 1
        val = DataLoader(TensorDataset(x, y), batch_size=2)

        ft.train(train, val, checkpoint_path=None)

        after = ft.model.state_dict()
        for key in frozen_bn_keys:
            torch.testing.assert_close(after[key], before[key], msg=f"{key} drifted")
