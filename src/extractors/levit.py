import timm
import torch
import torch.nn as nn
from torchvision import transforms

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

    def __init__(self, device: str = "cuda", output_dim: int = 128):
        super().__init__(device=device, output_dim=output_dim)
        self.model = self._build_model()
        self.transform = self._build_transform()

    def _forward_components(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.forward_components(images)

    def _build_model(self) -> nn.Module:
        model = _LeViTBackbone(output_dim=self.output_dim)
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
