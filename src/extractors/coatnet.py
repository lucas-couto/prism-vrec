import timm
import torch
import torch.nn as nn

from src.extractors.base import BaseExtractor, timm_canonical_transform


class _CoAtNetBackbone(nn.Module):
    """CoAtNet-0 backbone, frozen, native 768-d output.

    ``projection`` defaults to identity so extraction emits the native
    pooled feature; the fine-tuner replaces it with a classification
    head.
    """

    def __init__(self):
        super().__init__()

        # num_classes=0 removes the classification head
        self.backbone = timm.create_model("coatnet_0_rw_224", pretrained=True, num_classes=0)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.backbone(x))

    def forward_components(self, x: torch.Tensor) -> torch.Tensor:
        """Return native spatial cells ``(B, H*W, 768)`` (7x7=49)."""
        feat = self.backbone.forward_features(x)  # (B, 768, H, W)
        tokens = feat.flatten(2).transpose(1, 2)  # (B, H*W, 768)
        return self.projection(tokens)


class CoAtNetExtractor(BaseExtractor):
    """Visual feature extractor based on CoAtNet-0 (native 768-d).

    CoAtNet combines depthwise convolutions with self-attention modules
    in a unified architecture, mixing both operations within the same
    stages rather than using them sequentially.

    Parameters
    ----------
    device : str
        Device to run inference on.
    """

    unfreeze_prefixes = ["backbone.stages.3"]

    #: CoAtNet exposes its final 7x7=49 spatial cells for ACF.
    supports_components = True

    backbone_cls = _CoAtNetBackbone
    extraction_point = "global average pool (timm num_classes=0)"
    weights_id = "timm coatnet_0_rw_224.sw_in1k"

    def _build_transform(self):
        # Canonical recipe resolved from the checkpoint's pretrained
        # config. NOTE: this tag normalises with mean/std 0.5 (NOT
        # ImageNet), crop_pct 0.95, bicubic — v1.x applied ImageNet norm
        # here, silently degrading the features.
        return timm_canonical_transform(self.model.backbone)
