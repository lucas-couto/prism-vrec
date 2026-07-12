"""DeepStyle -- Style-aware recommendation (Liu, Wu & Wang, SIGIR 2017).

Faithful to the paper's formulation:

    theta_i  = E f_i - c_{cat(i)}          (style = projected visual - category)
    y_hat_ui = gamma_u^T gamma_i + s_u^T theta_i + beta_i

where ``E`` is a LINEAR projection (not an MLP) mapping the native
visual feature to the style space, and ``c_{cat(i)}`` is a LEARNED
embedding shared by every item of the same category, trained jointly
by the BPR loss.  Category labels come from the data (the same labels
the fine-tuning step consumes); the model never infers them.

Datasets without category labels (e.g. Tradesy) run with a single null
category for every item.  Subtracting the same vector from all items
cancels in the BPR pairwise difference, so the model **analytically
degenerates to VBPR** — an expected, declared property of the method
on unlabelled data, not a failure (logged at construction).

References
----------
Liu, Q., Wu, S., Wang, L. (2017). DeepStyle: Learning User Preferences
for Visual Recommendation.  SIGIR.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.recommenders._scoring import LinearVisualScoreMixin
from src.recommenders.base import BaseRecommender
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DeepStyle(LinearVisualScoreMixin, BaseRecommender):
    """DeepStyle with linear projection and learned category embeddings.

    Parameters
    ----------
    n_users, n_items:
        Vocabulary sizes.
    visual_embeddings:
        Pre-extracted visual features of shape ``(n_items, D_v)``.
    config:
        Must contain ``latent_dim`` (k) and ``style_dim`` (k_s).
        ``l2_reg`` is optional (default 0).
    item_categories:
        ``(n_items,)`` int array mapping each item to its category
        index (built once before training from the dataset's labels).
        ``None`` for unlabelled datasets — a single null category is
        used and the model degenerates to VBPR (see module docstring).
    """

    #: The training/evaluation steps pass ``item_categories`` only to
    #: models that declare this flag (mirrors ``wants_history``).
    wants_categories = True

    def __init__(
        self,
        n_users: int,
        n_items: int,
        visual_embeddings: np.ndarray | None = None,
        config: dict | None = None,
        *,
        item_categories: np.ndarray | None = None,
    ) -> None:
        config = config or {}
        super().__init__(n_users, n_items, visual_embeddings, config)

        k: int = config["latent_dim"]
        ks: int = config["style_dim"]

        if self.visual_features is None:
            raise RuntimeError("DeepStyle requires visual embeddings")
        dv: int = self.visual_dim_raw

        if item_categories is None:
            # Expected degeneration, not an error: with one category the
            # same vector is subtracted from every item, which cancels in
            # the BPR pairwise difference — DeepStyle == VBPR here.
            logger.info(
                "DeepStyle: dataset has no category labels; using a single "
                "null category for all items. The model analytically "
                "degenerates to VBPR on this dataset (expected, declared)."
            )
            cat_idx = torch.zeros(n_items, dtype=torch.long)
            n_categories = 1
        else:
            cat_idx = torch.as_tensor(np.asarray(item_categories), dtype=torch.long)
            if cat_idx.shape != (n_items,):
                raise ValueError(
                    f"item_categories must have shape ({n_items},), got {tuple(cat_idx.shape)}."
                )
            n_categories = int(cat_idx.max().item()) + 1
        self.n_categories = n_categories
        self.register_buffer("item_category_idx", cat_idx, persistent=False)

        self.user_embedding = nn.Embedding(n_users, k)
        self.item_embedding = nn.Embedding(n_items, k)
        self.item_bias = nn.Embedding(n_items, 1)

        self.style_user_embedding = nn.Embedding(n_users, ks)  # s_u
        self.visual_projection = nn.Linear(dv, ks, bias=False)  # E (linear, per the paper)
        self.category_embedding = nn.Embedding(n_categories, ks)  # c_{cat}

        self._init_embedding(self.user_embedding)
        self._init_embedding(self.item_embedding)
        self._init_embedding(self.style_user_embedding)
        self._init_embedding(self.category_embedding)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.xavier_uniform_(self.visual_projection.weight)

        self._item_proj_cache: torch.Tensor | None = None

    def _visual_user_table(self) -> nn.Embedding:
        return self.style_user_embedding

    def _item_visual_term(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Style vector per item: ``E f_i - c_{cat(i)}`` (projected space).

        Both the projection and the category lookup are batched tensor
        indexing.  Cache is bypassed when an online fusion is active
        because the gate's output changes every optimisation step.
        """
        cache_eligible = self._online_fusion is None
        if (
            cache_eligible
            and self._item_proj_cache is not None
            and item_ids.shape[0] == self.n_items
        ):
            return self._item_proj_cache
        f_i = self._resolve_visual(item_ids)
        proj = self.visual_projection(f_i) - self.category_embedding(
            self.item_category_idx[item_ids]
        )
        if cache_eligible and item_ids.shape[0] == self.n_items:
            self._item_proj_cache = proj
        return proj
