import warnings

import torch
import torch.nn as nn
from torchvision import transforms

from src.extractors.base import IMAGENET_MEAN, IMAGENET_STD, BaseExtractor

# Pinned commit of facebookresearch/dinov2 (default-branch HEAD at pin
# time). An unpinned hub load tracks the remote branch, so an upstream
# push could silently change the backbone code and break bit-identical
# reproducibility of extracted embeddings.
_DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"


class _DINOv2Backbone(nn.Module):
    """DINOv2 ViT-B/14 backbone ([CLS] token), frozen, native 768-d output.

    ``projection`` defaults to identity so extraction emits the native
    [CLS] feature; the fine-tuner replaces it with a classification
    head.
    """

    def __init__(self):
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

        self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)  # CLS token by default
        return self.projection(x)

    def forward_components(self, x: torch.Tensor) -> torch.Tensor:
        """Return native patch tokens ``(B, M, 768)`` (256 for B/14).

        DINOv2 exposes ``get_intermediate_layers`` which returns the
        last block's patch-token sequence (no [CLS]) at the backbone's
        768-d width — the same space the pooled [CLS] path comes from.
        """
        tokens = self.backbone.get_intermediate_layers(x, n=1)[0]  # (B, M, 768)
        return self.projection(tokens)


class DINOv2Extractor(BaseExtractor):
    """Visual feature extractor based on DINOv2 ViT-B/14 (native 768-d, [CLS]).

    Uses ``torch.hub`` to load the DINOv2 ViT-B/14 model from
    ``facebookresearch/dinov2`` at a pinned commit.  The saved feature
    is the native [CLS] token representation (768-d).

    Parameters
    ----------
    device : str
        Device to run inference on (e.g. ``"cuda"`` or ``"cpu"``).
    """

    #: DINOv2 exposes its 256 patch tokens (get_intermediate_layers) for ACF.
    supports_components = True

    backbone_cls = _DINOv2Backbone
    extraction_point = "CLS token (last block)"
    weights_id = f"torch.hub facebookresearch/dinov2@{_DINOV2_COMMIT[:12]} dinov2_vitb14"

    def _build_transform(self) -> transforms.Compose:
        # torch.hub ships no transform for DINOv2, so this is the one
        # hand-built recipe: the reference eval pipeline (resize 256
        # bicubic -> center-crop 224, ImageNet norm), matching the
        # `make_classification_eval_transform` in the DINOv2 repo.
        # 224 is a multiple of the 14px patch, as required.
        return transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
            ]
        )
