import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from src.extractors.base import BaseExtractor


class _ResNet50Backbone(nn.Module):
    """ResNet-50 backbone (up to avgpool), frozen, native 2048-d output.

    ``projection`` defaults to identity so extraction emits the native
    pooled feature; the fine-tuner replaces it with a classification
    head.
    """

    def __init__(self):
        super().__init__()

        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        for param in backbone.parameters():
            param.requires_grad = False

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )

        self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.projection(x)

    def forward_components(self, x: torch.Tensor) -> torch.Tensor:
        """Return native per-cell features ``(B, H*W, 2048)``.

        Uses the conv5 spatial map (``2048×7×7`` for 224² input, i.e.
        ``M=49`` components) before the global average pool.
        """
        x = self.features[:-1](x)  # drop avgpool -> (B, 2048, H, W)
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, 2048)
        return self.projection(x)


class ResNet50Extractor(BaseExtractor):
    """Visual feature extractor based on ResNet-50 (native 2048-d).

    Uses ``torchvision.models.resnet50`` with ImageNet-V2 weights.  The
    classification head is removed; the saved feature is the native
    pooled output (2048-d).

    Parameters
    ----------
    device : str
        Device to run inference on (e.g. ``"cuda"`` or ``"cpu"``).
    """

    # ``layer4`` (the last ResNet block) is unfrozen during fine-tuning.
    unfreeze_prefixes = ["features.8"]

    #: ResNet exposes its conv5 spatial map (``M=49`` cells) for ACF.
    supports_components = True

    backbone_cls = _ResNet50Backbone
    extraction_point = "global average pool (after layer4)"
    weights_id = "torchvision resnet50 IMAGENET1K_V2"

    def _build_transform(self):
        # Canonical recipe shipped with the weights (IMAGENET1K_V2):
        # resize 232 -> center-crop 224, bilinear, ImageNet norm.  Read
        # from the weights object, never hand-written.
        return ResNet50_Weights.IMAGENET1K_V2.transforms()
