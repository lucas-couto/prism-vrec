import warnings

import torch
import torch.nn as nn
from torchvision import transforms

from src.extractors.base import BaseExtractor, _imagenet_transform

# Pinned commit of facebookresearch/dinov2 (default-branch HEAD at pin
# time). An unpinned hub load tracks the remote branch, so an upstream
# push could silently change the backbone code and break bit-identical
# reproducibility of extracted embeddings.
_DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"


class _DINOv2Backbone(nn.Module):
    """DINOv2 ViT-B/14 backbone ([CLS] token) followed by a trainable projection."""

    def __init__(self, output_dim: int):
        super().__init__()

        # The vendored DINOv2 modules (swiglu_ffn, attention, block) emit
        # `UserWarning: xFormers is not available` at import time on
        # builds without the optional xFormers extension.  We silence
        # only this exact message and only for the duration of the hub
        # load — unrelated UserWarnings still propagate.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"xFormers is not available.*",
                category=UserWarning,
            )
            self.backbone = torch.hub.load(
                f"facebookresearch/dinov2:{_DINOV2_COMMIT}",
                "dinov2_vitb14",
            )
        for param in self.backbone.parameters():
            param.requires_grad = False

        # DINOv2 ViT-B/14 output dim = 768
        self.projection = nn.Sequential(
            nn.Linear(768, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)  # CLS token by default
        return self.projection(x)

    def forward_components(self, x: torch.Tensor) -> torch.Tensor:
        """Return projected patch tokens ``(B, M, output_dim)`` (256 for B/14).

        DINOv2 exposes ``get_intermediate_layers`` which returns the
        last block's patch-token sequence (no [CLS]) at the backbone's
        768-d width — the same space the pooled [CLS] path projects from.
        """
        tokens = self.backbone.get_intermediate_layers(x, n=1)[0]  # (B, M, 768)
        return self.projection(tokens)


class DINOv2Extractor(BaseExtractor):
    """Visual feature extractor based on DINOv2 ViT-B/14.

    Uses ``torch.hub`` to load the DINOv2 ViT-B/14 model from
    ``facebookresearch/dinov2``.  The [CLS] token (768-dim) is projected
    to ``output_dim`` via a trainable ``Linear + ReLU`` layer.

    Parameters
    ----------
    device : str
        Device to run inference on (e.g. ``"cuda"`` or ``"cpu"``).
    output_dim : int
        Dimensionality of the output embedding.
    """

    #: DINOv2 exposes its 256 patch tokens (get_intermediate_layers) for ACF.
    supports_components = True

    backbone_cls = _DINOv2Backbone

    def _build_transform(self) -> transforms.Compose:
        # DINOv2 was trained with bicubic resizing (the rest of the
        # framework uses the default bilinear).
        return _imagenet_transform(interpolation=transforms.InterpolationMode.BICUBIC)
