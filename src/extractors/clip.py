import torch
import torch.nn as nn

from src.extractors.base import BaseExtractor


class _CLIPVisualBackbone(nn.Module):
    """CLIP ViT-B/32 visual encoder, frozen, native 512-d projected output.

    The feature is the **projected** output (the same space
    ``encode_image`` returns, aligned with the text tower), not the
    768-d pre-projection encoder width — a deliberate, declared choice:
    it is how CLIP is canonically used as a feature extractor in
    practice.  ``projection`` defaults to identity so extraction emits
    that native feature; the fine-tuner replaces it with a
    classification head.
    """

    def __init__(self):
        super().__init__()
        import open_clip

        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_s34b_b79k",
        )
        self.visual = model.visual

        for param in self.visual.parameters():
            param.requires_grad = False

        self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.visual(x))

    def forward_components(self, x: torch.Tensor) -> torch.Tensor:
        """Return native patch tokens ``(B, M, 512)`` (49 for B/32).

        ``output_tokens`` makes open_clip's visual encoder return the
        per-patch sequence (pre-projection ``width``); we apply the same
        ``visual.proj`` the pooled path uses so components live in the
        512-d CLIP space, consistent with the pooled feature.
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
    """Visual feature extractor based on CLIP ViT-B/32 (native 512-d).

    Uses ``open_clip`` with LAION-2B pretrained weights.  Only the
    visual encoder is used; the saved feature is the projected 512-d
    output (the ``encode_image`` space).

    Parameters
    ----------
    device : str
        Device to run inference on (e.g. ``"cuda"`` or ``"cpu"``).
    """

    #: CLIP exposes its 49 patch tokens (via open_clip output_tokens) for ACF.
    supports_components = True

    backbone_cls = _CLIPVisualBackbone
    extraction_point = "projected visual output (encode_image space, post visual.proj)"
    weights_id = "open_clip ViT-B-32 laion2b_s34b_b79k"

    def _build_transform(self):
        # Canonical: the open_clip preprocess returned alongside the
        # weights (resize 224 bicubic -> crop 224, CLIP normalisation).
        return self.model.preprocess
