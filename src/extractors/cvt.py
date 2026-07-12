import torch
import torch.nn as nn
from transformers import CvtModel

from src.extractors.base import BaseExtractor, HFProcessorTransform


class _CvTBackbone(nn.Module):
    """CvT-13 backbone, frozen, native 384-d output (CLS token).

    CvT's final hidden state is a spatial map ``(B, 384, 14, 14)``; the
    pooled feature used here is the model's own **CLS token** (384-d),
    not a flatten of the map.  ``projection`` defaults to identity so
    extraction emits the native feature; the fine-tuner replaces it
    with a classification head.
    """

    def __init__(self):
        super().__init__()

        self.backbone = CvtModel.from_pretrained("microsoft/cvt-13")
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(x)
        x = outputs.cls_token_value.squeeze(1)
        return self.projection(x)

    def forward_components(self, x: torch.Tensor) -> torch.Tensor:
        """Return native spatial tokens ``(B, H*W, 384)`` (14x14=196)."""
        feat = self.backbone(x).last_hidden_state  # (B, 384, H, W)
        tokens = feat.flatten(2).transpose(1, 2)  # (B, H*W, 384)
        return self.projection(tokens)


class CvTExtractor(BaseExtractor):
    """Visual feature extractor based on CvT-13 (native 384-d, CLS token).

    CvT replaces the linear patch projection in ViT with convolutional
    token embeddings, introducing local spatial context (CNN inductive
    bias) directly inside the Transformer architecture.

    Parameters
    ----------
    device : str
        Device to run inference on.
    """

    unfreeze_prefixes = ["backbone.stages.2"]

    #: CvT exposes its final-stage spatial tokens (14x14=196) for ACF.
    supports_components = True

    backbone_cls = _CvTBackbone
    extraction_point = "CLS token (final stage; NOT a flatten of the 14x14 map)"
    weights_id = "huggingface microsoft/cvt-13 (224px checkpoint)"

    def _build_transform(self):
        # Canonical processor shipped with the checkpoint (ConvNext-style:
        # crop_pct 0.875 -> 224, bicubic, ImageNet norm), resolved via
        # AutoImageProcessor instead of a hand-written Compose.
        from transformers import AutoImageProcessor

        return HFProcessorTransform(AutoImageProcessor.from_pretrained("microsoft/cvt-13"))
