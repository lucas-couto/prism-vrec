import torch
import torch.nn as nn
from torchvision import transforms
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

    def __init__(self, device: str = "cuda", output_dim: int = 128):
        super().__init__(device=device, output_dim=output_dim)
        self.model = self._build_model()
        self.transform = self._build_transform()

    def _build_model(self) -> nn.Module:
        model = _ResNet50Backbone(output_dim=self.output_dim)
        model = model.to(self.device)
        model.eval()
        return model

    def _build_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
