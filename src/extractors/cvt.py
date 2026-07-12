import torch
import torch.nn as nn
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

    def forward_components(self, x: torch.Tensor) -> torch.Tensor:
        """Return projected spatial tokens ``(B, H*W, output_dim)``."""
        feat = self.backbone(x).last_hidden_state  # (B, 384, H, W)
        tokens = feat.flatten(2).transpose(1, 2)  # (B, H*W, 384)
        return self.projection(tokens)


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

    #: CvT exposes its final-stage spatial tokens (14x14=196) for ACF.
    supports_components = True

    backbone_cls = _CvTBackbone
