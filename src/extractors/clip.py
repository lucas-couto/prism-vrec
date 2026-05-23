import torch
import torch.nn as nn

from src.extractors.base import BaseExtractor


class _CLIPVisualBackbone(nn.Module):
    """CLIP ViT-B/32 visual encoder followed by a trainable projection."""

    def __init__(self, output_dim: int):
        super().__init__()
        import open_clip

        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k",
        )
        self.visual = model.visual

        for param in self.visual.parameters():
            param.requires_grad = False

        # CLIP ViT-B/32 output dim = 512
        self.projection = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.visual(x)
        return self.projection(x)


class CLIPExtractor(BaseExtractor):
    """Visual feature extractor based on CLIP ViT-B/32.

    Uses ``open_clip`` with LAION-2B pretrained weights.  Only the visual
    encoder is used.  The 512-dim output is projected to ``output_dim``
    via a trainable ``Linear + ReLU`` layer.

    Parameters
    ----------
    device : str
        Device to run inference on (e.g. ``"cuda"`` or ``"cpu"``).
    output_dim : int
        Dimensionality of the output embedding.
    """

    def __init__(self, device: str = "cuda", output_dim: int = 128):
        super().__init__(device=device, output_dim=output_dim)
        self._backbone = _CLIPVisualBackbone(output_dim=self.output_dim)
        self.model = self._build_model()
        self.transform = self._build_transform()

    def _build_model(self) -> nn.Module:
        model = self._backbone
        model = model.to(self.device)
        model.eval()
        return model

    def _build_transform(self):
        return self._backbone.preprocess
