import torch
import torch.nn as nn
from torchvision import transforms
from transformers import CvtModel

from src.extractors.base import BaseExtractor


class _CvTBackbone(nn.Module):
    """CvT-13 backbone followed by a trainable projection."""

    def __init__(self, output_dim: int):
        super().__init__()

        self.backbone = CvtModel.from_pretrained("microsoft/cvt-13")
        for param in self.backbone.parameters():
            param.requires_grad = False

        # CvT-13 output dim = 384
        self.projection = nn.Sequential(
            nn.Linear(384, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(x)
        x = outputs.cls_token_value.squeeze(1)
        return self.projection(x)


class CvTExtractor(BaseExtractor):
    """Visual feature extractor based on CvT-13.

    CvT replaces the linear patch projection in ViT with convolutional
    token embeddings, introducing local spatial context (CNN inductive
    bias) directly inside the Transformer architecture.

    Parameters
    ----------
    device : str
        Device to run inference on.
    output_dim : int
        Dimensionality of the output embedding.
    """

    unfreeze_prefixes = ["backbone.stages.2"]

    def __init__(self, device: str = "cuda", output_dim: int = 128):
        super().__init__(device=device, output_dim=output_dim)
        self.model = self._build_model()
        self.transform = self._build_transform()

    def _build_model(self) -> nn.Module:
        model = _CvTBackbone(output_dim=self.output_dim)
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
