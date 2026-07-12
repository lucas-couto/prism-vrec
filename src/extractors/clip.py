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

    def forward_components(self, x: torch.Tensor) -> torch.Tensor:
        """Return projected patch tokens ``(B, M, output_dim)`` (49 for B/32).

        ``output_tokens`` makes open_clip's visual encoder return the
        per-patch sequence (pre-projection ``width``); we apply the same
        ``visual.proj`` the pooled path uses so components live in the
        512-d CLIP space before the trainable projection.
        """
        self.visual.output_tokens = True
        try:
            _, tokens = self.visual(x)  # tokens: (B, M, width)
        finally:
            self.visual.output_tokens = False
        proj = getattr(self.visual, "proj", None)
        if proj is not None:
            tokens = tokens @ proj  # (B, M, 512)
        return self.projection(tokens)


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

    #: CLIP exposes its 49 patch tokens (via open_clip output_tokens) for ACF.
    supports_components = True

    backbone_cls = _CLIPVisualBackbone

    def _build_transform(self):
        # The transform is the open_clip preprocess carried on the
        # backbone that the base _build_model already constructed.
        return self.model.preprocess
