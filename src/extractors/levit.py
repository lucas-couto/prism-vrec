import timm
import torch
import torch.nn as nn

from src.extractors.base import BaseExtractor, timm_canonical_transform


class _LeViTBackbone(nn.Module):
    """LeViT-256 backbone, frozen, native 512-d output.

    The "256" in the model name is the width of the FIRST stage, not the
    final feature size — the pooled output is 512-d (read from the model
    at probe time, never hardcoded).

    ``projection`` defaults to identity so extraction emits the native
    pooled feature; the fine-tuner replaces it with a classification
    head.
    """

    def __init__(self):
        super().__init__()

        # num_classes=0 removes the classification head
        self.backbone = timm.create_model("levit_256", pretrained=True, num_classes=0)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.backbone(x))

    def forward_components(self, x: torch.Tensor) -> torch.Tensor:
        """Return native final-stage tokens ``(B, M, 512)``."""
        feat = self.backbone.forward_features(x)  # (B, N, 512) or (B, C, H, W)
        if feat.dim() == 4:
            feat = feat.flatten(2).transpose(1, 2)
        return self.projection(feat)


class LeViTExtractor(BaseExtractor):
    """Visual feature extractor based on LeViT-256 (native 512-d).

    LeViT uses initial convolutional stages followed by Transformer
    blocks in a sequential architecture optimized for fast inference.

    Parameters
    ----------
    device : str
        Device to run inference on.
    """

    unfreeze_prefixes = ["backbone.stages.2"]

    #: LeViT exposes its final-stage token sequence for ACF.
    supports_components = True

    backbone_cls = _LeViTBackbone
    extraction_point = "pooled final-stage tokens (timm num_classes=0)"
    weights_id = "timm levit_256.fb_dist_in1k"

    def _build_transform(self):
        # Canonical recipe resolved from the checkpoint's pretrained
        # config (crop_pct 0.9, bicubic, ImageNet norm).
        return timm_canonical_transform(self.model.backbone)
