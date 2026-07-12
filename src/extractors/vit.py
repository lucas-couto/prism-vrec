import timm
import torch
import torch.nn as nn

from src.extractors.base import BaseExtractor, timm_canonical_transform


class _ViTBackbone(nn.Module):
    """ViT-B/16 backbone ([CLS] token), frozen, native 768-d output.

    ``projection`` defaults to identity so extraction emits the native
    [CLS] feature; the fine-tuner replaces it with a classification
    head.
    """

    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model("vit_base_patch16_224", pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # timm ViT returns (B, num_tokens, dim); take [CLS] at position 0
        x = self.backbone.forward_features(x)
        if x.dim() == 3:
            x = x[:, 0]
        return self.projection(x)

    def forward_components(self, x: torch.Tensor) -> torch.Tensor:
        """Return native patch tokens ``(B, M, 768)`` (no [CLS])."""
        feats = self.backbone.forward_features(x)  # (B, T, 768)
        prefix = getattr(self.backbone, "num_prefix_tokens", 1)
        tokens = feats[:, prefix:] if feats.dim() == 3 else feats
        return self.projection(tokens)


class ViTExtractor(BaseExtractor):
    """Visual feature extractor based on ViT-B/16 (native 768-d, [CLS]).

    Uses ``timm.create_model('vit_base_patch16_224', pretrained=True)``.
    The saved feature is the native [CLS] token representation (768-d).

    Parameters
    ----------
    device : str
        Device to run inference on (e.g. ``"cuda"`` or ``"cpu"``).
    """

    unfreeze_prefixes = ["backbone.blocks.11", "backbone.blocks.10"]

    #: ViT exposes its 196 patch tokens (before [CLS] pooling) for ACF.
    supports_components = True

    backbone_cls = _ViTBackbone
    extraction_point = "CLS token (last block)"
    weights_id = "timm vit_base_patch16_224.augreg2_in21k_ft_in1k"

    def _build_transform(self):
        # Canonical recipe resolved from the checkpoint's pretrained
        # config. NOTE: this tag normalises with mean/std 0.5 (NOT
        # ImageNet), crop_pct 0.9, bicubic — v1.x applied ImageNet norm
        # here, silently degrading the features.
        return timm_canonical_transform(self.model.backbone)
