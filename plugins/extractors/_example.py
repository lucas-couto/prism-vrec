"""Example extractor plugin: ResNet-18 with the last block fine-tunable.

How to use this file
--------------------
1. Copy ``_example.py`` to a new file with a descriptive name, e.g.
   ``plugins/extractors/my_resnet18.py``.  The leading underscore on
   *this* file is what keeps the auto-discovery from importing it —
   files (and dataset directories) starting with ``_`` are skipped on
   purpose so the example never registers itself.
2. Rename the class and the ``register_extractor("my_resnet18", ...)``
   key to whatever you want to expose.
3. Add the same key to ``configs/extractors.yaml -> extractors_enabled``
   and run the pipeline.

The example follows the contract documented on
:class:`src.extractors.base.BaseExtractor`: the backbone exposes a
``projection`` submodule whose ``in_features`` matches the pooled-feature
size, so the fine-tuner can swap it for a classification head.

Full guide: ``docs/extending.md``.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from src.extractors.base import BaseExtractor
from src.extractors.registry import register_extractor


class _ResNet18Backbone(nn.Module):
    """ResNet-18 backbone (everything up to avgpool) + trainable projection."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Freeze the entire backbone by default; the FineTuner re-enables
        # whatever is matched by ``unfreeze_prefixes`` on the extractor.
        for param in backbone.parameters():
            param.requires_grad = False

        # Strip the final FC; keep everything else including avgpool.
        # children()[-1] is the original FC layer.
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.projection = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.projection(x)


class ResNet18Extractor(BaseExtractor):
    """ResNet-18 extractor, registered as ``my_resnet18``."""

    # Unfreeze the last residual block (children[-2] in the Sequential
    # above corresponds to ResNet's ``layer4``).  Adjust the prefix if
    # you change the architecture wrapping.
    unfreeze_prefixes = ["features.7"]

    def __init__(self, device: str = "cuda", output_dim: int = 128) -> None:
        super().__init__(device=device, output_dim=output_dim)
        self.model = self._build_model()
        self.transform = self._build_transform()

    def _build_model(self) -> nn.Module:
        model = _ResNet18Backbone(output_dim=self.output_dim).to(self.device)
        model.eval()
        return model

    def _build_transform(self):
        return ResNet18_Weights.IMAGENET1K_V1.transforms()


# Side effect on import: add this extractor to the global registry.
register_extractor("my_resnet18", ResNet18Extractor)
