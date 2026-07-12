import timm
import torch
import torch.nn as nn

from src.extractors.base import BaseExtractor, timm_canonical_transform


class _ConvNeXtBackbone(nn.Module):
    """ConvNeXt-Base backbone, frozen, native 1024-d output.

    ``projection`` defaults to identity so extraction emits the native
    pooled feature; the fine-tuner replaces it with a classification
    head.
    """

    def __init__(self):
        super().__init__()

        # num_classes=0 removes the classification head and returns the
        # pooled feature vector directly.
        self.backbone = timm.create_model(
            "convnext_base.fb_in22k_ft_in1k",
            pretrained=True,
            num_classes=0,
        )
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.backbone(x))

    def forward_components(self, x: torch.Tensor) -> torch.Tensor:
        """Return native spatial cells ``(B, H*W, 1024)`` (7x7=49)."""
        feat = self.backbone.forward_features(x)  # (B, 1024, H, W)
        tokens = feat.flatten(2).transpose(1, 2)  # (B, H*W, 1024)
        return self.projection(tokens)


class ConvNeXtExtractor(BaseExtractor):
    """Visual feature extractor based on ConvNeXt-Base (native 1024-d).

    ConvNeXt is a pure CNN that adopts ViT-inspired design choices
    (large kernels, layer normalization, GELU, inverted bottlenecks)
    without using self-attention.  It serves as a modern pure-CNN
    baseline at the same parameter scale (~89M) as ViT-B/16, CLIP
    ViT-B/32 and DINOv2 ViT-B/14.

    Parameters
    ----------
    device : str
        Device to run inference on.
    """

    unfreeze_prefixes = ["backbone.stages.3"]

    #: ConvNeXt exposes its final 7x7=49 spatial cells for ACF.
    supports_components = True

    backbone_cls = _ConvNeXtBackbone
    extraction_point = "global average pool (timm num_classes=0)"
    weights_id = "timm convnext_base.fb_in22k_ft_in1k"

    def _build_transform(self):
        # Canonical recipe resolved from the checkpoint's pretrained
        # config (crop_pct 0.875, bicubic, ImageNet norm).
        return timm_canonical_transform(self.model.backbone)
