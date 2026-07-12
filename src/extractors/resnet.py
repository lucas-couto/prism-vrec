import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from src.extractors.base import BaseExtractor


class _ResNet50Backbone(nn.Module):
    """ResNet-50 backbone (up to avgpool) followed by a trainable projection."""

    def __init__(self, output_dim: int):
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

        self.projection = nn.Sequential(
            nn.Linear(2048, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.projection(x)

    def forward_components(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-cell projected features ``(B, H*W, output_dim)``.

        Uses the conv5 spatial map (``2048×7×7`` for 224² input, i.e.
        ``M=49`` components) before the global average pool, projected
        through the same trainable ``projection`` as the pooled path.
        """
        x = self.features[:-1](x)  # drop avgpool -> (B, 2048, H, W)
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, 2048)
        return self.projection(x)


class ResNet50Extractor(BaseExtractor):
    """Visual feature extractor based on ResNet-50.

    Uses ``torchvision.models.resnet50`` with ImageNet-V2 weights.  The
    classification head is removed and a trainable ``Linear + ReLU``
    projection maps the 2048-dim pooled features to ``output_dim``.

    Parameters
    ----------
    device : str
        Device to run inference on (e.g. ``"cuda"`` or ``"cpu"``).
    output_dim : int
        Dimensionality of the output embedding.
    """

    # ``layer4`` (the last ResNet block) is unfrozen during fine-tuning.
    unfreeze_prefixes = ["features.8"]

    #: ResNet exposes its conv5 spatial map (``M=49`` cells) for ACF.
    supports_components = True

    backbone_cls = _ResNet50Backbone
