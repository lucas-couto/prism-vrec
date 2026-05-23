import timm
import torch
import torch.nn as nn
from torchvision import transforms

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

    def __init__(self, device: str = "cuda", output_dim: int = 128):
        super().__init__(device=device, output_dim=output_dim)
        self.model = self._build_model()
        self.transform = self._build_transform()

    def _build_model(self) -> nn.Module:
        model = _ConvNeXtBackbone(output_dim=self.output_dim)
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
