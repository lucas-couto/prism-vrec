"""DeepStyle -- Style-aware recommendation (Liu, Wu & Wang, SIGIR 2017).

Paper formulation (Eqs. 2-3)::

    s_i      = E v_i - l_i                 (style = projected visual - category)
    y_hat_ui = p_u^T (s_i + q_i)

with ``p_u, q_i, s_i, l_i`` all in ``R^d``: ONE user vector ``p_u``
dotted against both the latent item factor ``q_i`` and the style
vector ``s_i``, a single dimension ``d`` for every factor, and NO item
bias.  ``E`` is a LINEAR projection (not an MLP) from the native visual
feature to ``R^d``; ``l_i = l_{cat(i)}`` is a LEARNED embedding shared
by every item of the same category, trained jointly by the BPR loss.
Category labels come from the data (the same labels the fine-tuning
step consumes); the model never infers them.  Eq. 6 regularises every
parameter with a single ``λ``, so all tables and ``E`` sit under the
``l2_reg`` key.

Datasets without category labels (e.g. Tradesy) run with a single null
category ``l`` for every item.  The term ``p_u^T l`` is then constant
per user and cancels in the BPR pairwise difference, so the model
**analytically degenerates** — an expected, declared property of the
method on unlabelled data, not a failure (logged at construction).
The target of that degeneration is a RESTRICTED VBPR — visual user
weights tied to the latent ones (``γ_u ≡ θ_u``), ``k_v = k``, no item
bias ``β_i`` and no visual bias ``β'`` — not the production VBPR,
which keeps a separate ``θ_u`` and both biases.

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
        Must contain ``latent_dim`` (``d``, shared by every factor).
        ``l2_reg`` is optional (default 0).  A ``style_dim`` entry is
        accepted only when it equals ``latent_dim``: the paper defines a
        single ``d`` and no separate style dimension.
    item_categories:
        ``(n_items,)`` int array mapping each item to its category
        index (built once before training from the dataset's labels).
        ``None`` for unlabelled datasets — a single null category is
        used and the model degenerates to a restricted VBPR (see module
        docstring).
    """

    #: The training/evaluation steps pass ``item_categories`` only to
    #: models that declare this flag (mirrors ``wants_history``).
    wants_categories = True

    #: BPR-Opt L2 under the paper's single ``λ`` (Eq. 6): the one user
    #: table ``p_u`` and the item table ``q_i`` are gathered per batch;
    #: category rows ``l_{cat(i)}`` are gathered per touched item in
    #: :meth:`_l2_gathered_terms` (listing the table here only excludes
    #: its full matrix from the shared term).  The dense projection
    #: ``E`` stays shared (every triple touches it).  All keys resolve to
    #: ``l2_reg`` — no per-group ``λ``.
    _L2_USER_TABLES = ("user_embedding",)
    _L2_EXTRA_GATHERED_TABLES = ("category_embedding",)
    #: Fold-in: ``p_u`` is the single user-indexed table (base default).
    _USER_TABLES = ("user_embedding",)

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
        style_dim = config.get("style_dim")
        if style_dim is not None and int(style_dim) != int(k):
            raise ValueError(
                "DeepStyle (Liu et al., 2017) uses a single dimension d for p_u, q_i, "
                f"s_i and l_i; got style_dim={style_dim} != latent_dim={k}. "
                "Drop style_dim from the config or set it equal to latent_dim."
            )

        if self.visual_features is None:
            raise RuntimeError("DeepStyle requires visual embeddings")
        dv: int = self.visual_dim_raw

        cat_idx, n_categories = self._resolve_categories(n_items, item_categories)
        self.n_categories = n_categories
        self.register_buffer("item_category_idx", cat_idx, persistent=False)

        self.user_embedding = nn.Embedding(n_users, k)  # p_u
        self.item_embedding = nn.Embedding(n_items, k)  # q_i
        self.visual_projection = nn.Linear(dv, k, bias=False)  # E (linear, per the paper)
        self.category_embedding = nn.Embedding(n_categories, k)  # l_{cat}

        self._init_embedding(self.user_embedding)
        self._init_embedding(self.item_embedding)
        self._init_embedding(self.category_embedding)
        nn.init.xavier_uniform_(self.visual_projection.weight)

        self._item_proj_cache: torch.Tensor | None = None

    @staticmethod
    def _resolve_categories(
        n_items: int, item_categories: np.ndarray | None
    ) -> tuple[torch.Tensor, int]:
        """Validate the category labels, or build the single null category."""
        if item_categories is None:
            # Expected degeneration, not an error: with one category the
            # same vector is subtracted from every item, which cancels in
            # the BPR pairwise difference — DeepStyle == restricted VBPR.
            logger.info(
                "DeepStyle: dataset has no category labels; using a single "
                "null category for all items. The model analytically "
                "degenerates to a restricted VBPR on this dataset (expected, declared)."
            )
            return torch.zeros(n_items, dtype=torch.long), 1
        cat_idx = torch.as_tensor(np.asarray(item_categories), dtype=torch.long)
        if cat_idx.shape != (n_items,):
            raise ValueError(
                f"item_categories must have shape ({n_items},), got {tuple(cat_idx.shape)}."
            )
        return cat_idx, int(cat_idx.max().item()) + 1

    def _l2_gathered_terms(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
    ) -> list[tuple[str, torch.Tensor]]:
        """Add the category rows ``l_{cat(i)}`` of the touched items (single ``λ``)."""
        terms = super()._l2_gathered_terms(user_ids, pos_item_ids, neg_item_ids)
        terms.append(("l2_reg", self.category_embedding(self.item_category_idx[pos_item_ids])))
        terms.append(("l2_reg", self.category_embedding(self.item_category_idx[neg_item_ids])))
        return terms

    def _visual_user_table(self) -> nn.Embedding:
        """The paper has one user vector: ``p_u`` weighs both ``q_i`` and ``s_i``."""
        return self.user_embedding

    def _item_visual_term(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Style vector per item: ``s_i = E v_i - l_{cat(i)}`` (projected space).

        Both the projection and the category lookup are batched tensor
        indexing.  Cache is bypassed when an online fusion is active
        because the gate's output changes every optimisation step.
        """

        def _style(ids: torch.Tensor) -> torch.Tensor:
            return self.visual_projection(self._resolve_visual(ids)) - self.category_embedding(
                self.item_category_idx[ids]
            )

        return self._full_catalog_cache(item_ids, _style)
