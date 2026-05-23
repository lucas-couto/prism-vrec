"""Example recommender plugin: deterministic uniform-noise scorer.

How to use this file
--------------------
1. Copy ``_example.py`` to a new file with a descriptive name, e.g.
   ``plugins/recommenders/my_model.py``.  The leading underscore on
   *this* file is what keeps the auto-discovery from importing it —
   files (and dataset directories) starting with ``_`` are skipped on
   purpose so the example never registers itself.
2. Rename the class and the ``register_recommender("uniform_noise"
   ...)`` key.
3. Add the same key to ``configs/recommenders.yaml ->
   recommenders_enabled`` and run the pipeline.

This recommender is intentionally trivial: it returns uniform-noise
scores.  Use it as a sanity floor in your benchmarks — every legitimate
recommender should beat it on every metric.

Two contract details to keep in mind when writing your own:

* The :class:`BaseRecommender` constructor wires ``visual_embeddings``
  into a non-trainable buffer named ``visual_features``.  Read it as
  ``self.visual_features`` inside ``forward`` / ``predict``.  When
  ``requires_visual=False`` (as in this example) the buffer is ``None``.
* ``forward`` returns ``(score_pos, score_neg)`` and ``predict`` returns
  scores for an entire candidate list.  The trainer takes care of BPR
  loss + L2 reg via :meth:`BaseRecommender.bpr_loss`.

Full guide: ``docs/extending.md``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.recommenders.base import BaseRecommender
from src.recommenders.registry import register_recommender


class UniformNoiseRecommender(BaseRecommender):
    """Uniform-noise scorer; serves as a ranking sanity floor."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        visual_embeddings,
        config: dict,
    ) -> None:
        super().__init__(n_users, n_items, visual_embeddings, config)
        # A trainable scalar so the optimiser has at least one parameter.
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos = torch.rand_like(user_ids, dtype=torch.float32) + self.dummy
        neg = torch.rand_like(user_ids, dtype=torch.float32) + self.dummy
        return pos, neg

    def predict(self, user_id: int, item_ids: torch.Tensor) -> torch.Tensor:
        return torch.rand(item_ids.shape[0], device=self.dummy.device) + self.dummy


register_recommender(
    "uniform_noise",
    UniformNoiseRecommender,
    priority=0,
    requires_visual=False,
    uses_visual_dim=False,
)
