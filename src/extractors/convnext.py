import timm
import torch
import torch.nn as nn

from src.extractors.base import BaseExtractor


class _ConvNeXtBackbone(nn.Module):
    """ConvNeXt-Base backbone followed by a trainable projection."""

    def __init__(self, output_dim: int):
        super().__init__()

        # num_classes=0 removes the classification head and returns the
        # 1024-d pooled feature vector directly.
        self.backbone = timm.create_model(
            "convnext_base.fb_in22k_ft_in1k",
            pretrained=True,
            num_classes=0,
        )
        for param in self.backbone.parameters():
            param.requires_grad = False

        # ConvNeXt-Base output dim = 1024
        self.projection = nn.Sequential(
            nn.Linear(1024, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.projection(x)

    def forward_components(self, x: torch.Tensor) -> torch.Tensor:
        """Return projected spatial cells ``(B, H*W, output_dim)`` (7x7=49)."""
        feat = self.backbone.forward_features(x)  # (B, 1024, H, W)
        tokens = feat.flatten(2).transpose(1, 2)  # (B, H*W, 1024)
        return self.projection(tokens)


class ConvNeXtExtractor(BaseExtractor):
    """Visual feature extractor based on ConvNeXt-Base.

    ConvNeXt is a pure CNN that adopts ViT-inspired design choices
    (large kernels, layer normalization, GELU, inverted bottlenecks)
    without using self-attention.  It serves as a modern pure-CNN
    baseline at the same parameter scale (~89M) as ViT-B/16, CLIP
    ViT-B/32 and DINOv2 ViT-B/14.

    Parameters
    ----------
    device : str
        Device to run inference on.
    output_dim : int
        Dimensionality of the output embedding.
    """

    unfreeze_prefixes = ["backbone.stages.3"]

    #: ConvNeXt exposes its final 7x7=49 spatial cells for ACF.
    supports_components = True

    backbone_cls = _ConvNeXtBackbone
