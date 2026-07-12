import timm
import torch
import torch.nn as nn

from src.extractors.base import BaseExtractor


class _LeViTBackbone(nn.Module):
    """LeViT-256 backbone followed by a trainable projection."""

    def __init__(self, output_dim: int):
        super().__init__()

        # num_classes=0 removes the classification head
        self.backbone = timm.create_model("levit_256", pretrained=True, num_classes=0)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # LeViT-256 output dim = 512
        self.projection = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.projection(x)

    def forward_components(self, x: torch.Tensor) -> torch.Tensor:
        """Return projected final-stage tokens ``(B, M, output_dim)``."""
        feat = self.backbone.forward_features(x)  # (B, N, 512) or (B, C, H, W)
        if feat.dim() == 4:
            feat = feat.flatten(2).transpose(1, 2)
        return self.projection(feat)


class LeViTExtractor(BaseExtractor):
    """Visual feature extractor based on LeViT-256.

    LeViT uses initial convolutional stages followed by Transformer
    blocks in a sequential architecture optimized for fast inference.

    Parameters
    ----------
    device : str
        Device to run inference on.
    output_dim : int
        Dimensionality of the output embedding.
    """

    unfreeze_prefixes = ["backbone.stages.2"]

    #: LeViT exposes its final-stage token sequence for ACF.
    supports_components = True

    backbone_cls = _LeViTBackbone
