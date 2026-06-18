import timm
import torch
import torch.nn as nn
from torchvision import transforms

from src.extractors.base import BaseExtractor


class _ViTBackbone(nn.Module):
    """ViT-B/16 backbone ([CLS] token) followed by a trainable projection."""

    def __init__(self, output_dim: int):
        super().__init__()

        self.backbone = timm.create_model("vit_base_patch16_224", pretrained=True)
        for param in self.backbone.parameters():
            param.requires_grad = False

        cls_dim = self.backbone.num_features  # 768

        self.projection = nn.Sequential(
            nn.Linear(cls_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # timm ViT returns (B, num_tokens, dim); take [CLS] at position 0
        x = self.backbone.forward_features(x)
        if x.dim() == 3:
            x = x[:, 0]
        return self.projection(x)

    def forward_components(self, x: torch.Tensor) -> torch.Tensor:
        """Return projected patch tokens ``(B, M, output_dim)`` (no [CLS])."""
        feats = self.backbone.forward_features(x)  # (B, T, 768)
        prefix = getattr(self.backbone, "num_prefix_tokens", 1)
        tokens = feats[:, prefix:] if feats.dim() == 3 else feats
        return self.projection(tokens)


class ViTExtractor(BaseExtractor):
    """Visual feature extractor based on ViT-B/16.

    Uses ``timm.create_model('vit_base_patch16_224', pretrained=True)``.
    The [CLS] token representation (768-dim) is projected to ``output_dim``
    via a trainable ``Linear + ReLU`` layer.

    Parameters
    ----------
    device : str
        Device to run inference on (e.g. ``"cuda"`` or ``"cpu"``).
    output_dim : int
        Dimensionality of the output embedding.
    """

    unfreeze_prefixes = ["backbone.blocks.11", "backbone.blocks.10"]

    #: ViT exposes its 196 patch tokens (before [CLS] pooling) for ACF.
    supports_components = True

    def __init__(self, device: str = "cuda", output_dim: int = 128):
        super().__init__(device=device, output_dim=output_dim)
        self.model = self._build_model()
        self.transform = self._build_transform()

    def _forward_components(self, images: torch.Tensor) -> torch.Tensor:
        return self.model.forward_components(images)

    def _build_model(self) -> nn.Module:
        model = _ViTBackbone(output_dim=self.output_dim)
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
