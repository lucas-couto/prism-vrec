import timm
import torch
import torch.nn as nn
from torchvision import transforms

from src.extractors.base import BaseExtractor


class _CoAtNetBackbone(nn.Module):
    """CoAtNet-0 backbone followed by a trainable projection."""

    def __init__(self, output_dim: int):
        super().__init__()

        # num_classes=0 removes the classification head
        self.backbone = timm.create_model("coatnet_0_rw_224", pretrained=True, num_classes=0)
        for param in self.backbone.parameters():
            param.requires_grad = False

        # CoAtNet-0 output dim = 768
        self.projection = nn.Sequential(
            nn.Linear(768, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.projection(x)


class CoAtNetExtractor(BaseExtractor):
    """Visual feature extractor based on CoAtNet-0.

    CoAtNet combines depthwise convolutions with self-attention modules
    in a unified architecture, mixing both operations within the same
    stages rather than using them sequentially.

    Parameters
    ----------
    device : str
        Device to run inference on.
    output_dim : int
        Dimensionality of the output embedding.
    """

    unfreeze_prefixes = ["backbone.stages.3"]

    def __init__(self, device: str = "cuda", output_dim: int = 128):
        super().__init__(device=device, output_dim=output_dim)
        self.model = self._build_model()
        self.transform = self._build_transform()

    def _build_model(self) -> nn.Module:
        model = _CoAtNetBackbone(output_dim=self.output_dim)
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
