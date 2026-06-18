import warnings

import torch
import torch.nn as nn
from torchvision import transforms

from src.extractors.base import BaseExtractor


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
                "facebookresearch/dinov2",
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

    def __init__(self, device: str = "cuda", output_dim: int = 128):
        super().__init__(device=device, output_dim=output_dim)
        self.model = self._build_model()
        self.transform = self._build_transform()

    def _forward_components(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.forward_components(images)

    def _build_model(self) -> nn.Module:
        model = _DINOv2Backbone(output_dim=self.output_dim)
        model = model.to(self.device)
        model.eval()
        return model

    def _build_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(
                    (224, 224),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
