"""Abstract base recommender with BPR loss, L2 regularization, and checkpointing."""

from __future__ import annotations

import abc

import numpy as np
import torch
import torch.nn as nn


class BaseRecommender(nn.Module, abc.ABC):
    """Base class for all BPR-based recommendation models.

    Subclasses must implement :meth:`forward` (training scores) and
    :meth:`predict` (inference scores).  Common functionality such as BPR
    loss computation, L2 regularisation, and checkpoint management lives
    here.

    Parameters
    ----------
    n_users:
        Total number of users.
    n_items:
        Total number of items.
    visual_embeddings:
        Pre-extracted visual feature matrix of shape ``(n_items, D_v)``.
        Registered as a non-trainable buffer.  May be ``None`` for models
        that do not use visual features (e.g. plain BPR).
    config:
        Model-specific configuration dictionary (latent_dim, l2_reg, etc.).
    train_interactions:
        Optional ``{user_idx: set(item_idx)}`` of training interactions.
        Only models that set :attr:`wants_history` (e.g. ACF, whose
        item-level attention operates over the user profile) consume it;
        for every other model it defaults to ``None`` and is ignored, so
        existing behaviour is bit-identical.
    """

    #: ``True`` for models whose 3-D visual buffer holds per-item
    #: *components* (e.g. ACF, ``M`` spatial tokens) rather than the two
    #: stacked sources of an online fusion.  When ``True`` the base class
    #: keeps the raw 3-D buffer and does NOT instantiate an online fusion
    #: module (which assumes exactly two sources).
    consumes_raw_components: bool = False

    #: ``True`` for models that need each user's training history at
    #: construction time (e.g. ACF item-level attention).  The training
    #: and evaluation steps pass ``train_interactions`` only to such
    #: models, so models that do not accept the keyword are untouched.
    wants_history: bool = False

    def __init__(
        self,
        n_users: int,
        n_items: int,
        visual_embeddings: np.ndarray | None,
        config: dict,
        *,
        train_interactions: dict[int, set[int]] | None = None,
    ) -> None:
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.config = config
        self.train_interactions = train_interactions

        # Register visual embeddings as a non-trainable buffer if provided.
        # Three layouts are accepted:
        #   2-D ``(n_items, D)``    — pre-fused or single-source embeddings
        #                             (the long-standing default).
        #   3-D ``(n_items, M, D)`` — M equal-dim source embeddings stacked
        #                             along axis=1, ready to feed an online
        #                             fusion module (e.g. adaptive_gated).
        #   RaggedSources          — 2-D concat of M native sources with
        #                             differing dims + metadata; drives a
        #                             LearnedAlignmentFusion (per-source
        #                             learned projections, alignment=learned).
        # ``self.visual_dim_raw`` always reports the dimension the model's
        # learned projection E consumes, regardless of the layout.
        self._online_fusion: nn.Module | None = None
        source_dims = getattr(visual_embeddings, "source_dims", None)
        if visual_embeddings is not None and source_dims:
            from src.fusions.online import LearnedAlignmentFusion  # avoid cycle

            arr = torch.FloatTensor(np.asarray(visual_embeddings))
            self.register_buffer("visual_features", arr, persistent=False)
            self.visual_dim_raw = int(visual_embeddings.aligned_dim)
            self._online_fusion = LearnedAlignmentFusion(
                source_dims=list(source_dims),
                dim=int(visual_embeddings.aligned_dim),
                strategy=visual_embeddings.strategy,
                normalize=bool(visual_embeddings.normalize),
                **visual_embeddings.fusion_kwargs,
            )
        elif visual_embeddings is not None:
            arr = torch.FloatTensor(visual_embeddings)
            if arr.dim() == 3 and not self.consumes_raw_components:
                self.register_buffer("visual_features", arr, persistent=False)
                self.visual_dim_raw = int(arr.shape[-1])
                self._init_online_fusion(int(arr.shape[1]), self.visual_dim_raw, config)
            elif arr.dim() == 3:
                # Raw component buffer (n_items, M, D): the consuming model
                # (e.g. ACF) applies its own component attention; no online
                # fusion module is created.
                self.register_buffer("visual_features", arr, persistent=False)
                self.visual_dim_raw = int(arr.shape[-1])
            elif arr.dim() == 2:
                self.register_buffer("visual_features", arr, persistent=False)
                self.visual_dim_raw = int(arr.shape[-1])
            else:
                raise ValueError(
                    f"visual_embeddings must be 2-D (n_items, D) or 3-D "
                    f"(n_items, M, D); got shape {tuple(arr.shape)}.",
                )
        else:
            self.visual_features: torch.Tensor | None = None
            self.visual_dim_raw = 0

    def _init_online_fusion(self, n_sources: int, dim: int, config: dict) -> None:
        """Instantiate the online fusion module declared in ``config``.

        Triggered when ``visual_embeddings`` is 3-D.  ``config`` may
        carry the strategy name under ``visual_fusion_strategy`` (set
        by the train step from the JSON sidecar).  Defaults to
        ``adaptive_gated`` when omitted but the buffer is 3-D.
        """
        from src.fusions import online_module_for  # local import — avoids cycle

        strategy = config.get("visual_fusion_strategy", "adaptive_gated")
        self._online_fusion = online_module_for(strategy, dim=dim)
        if n_sources != 2:
            raise ValueError(
                f"Online fusion {strategy!r} expects 2 source embeddings, got {n_sources}.",
            )

    def _resolve_visual(self, item_ids: torch.Tensor) -> torch.Tensor:
        """Return per-item visual features as a ``(B, D)`` tensor.

        Hides the 2-D vs 3-D distinction from concrete recommenders:
        when the buffer is 3-D, the configured online fusion module
        is applied to produce the fused representation; otherwise the
        buffer is indexed directly.
        """
        if self.visual_features is None:
            raise RuntimeError("This recommender was instantiated without visual_embeddings.")

        if self._online_fusion is None:
            return self.visual_features[item_ids]

        from src.fusions.online import LearnedAlignmentFusion  # avoid cycle

        if isinstance(self._online_fusion, LearnedAlignmentFusion):
            # 2-D ragged concat buffer: the module splits by source_dims,
            # projects each native source to the aligned dim and fuses.
            return self._online_fusion(self.visual_features[item_ids])

        # 3-D buffer: features[item_ids] has shape (B, M, D).
        stacked = self.visual_features[item_ids]
        e1 = stacked[:, 0, :]
        e2 = stacked[:, 1, :]
        return self._online_fusion(e1, e2)

    @abc.abstractmethod
    def forward(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute positive and negative scores for a BPR training step.

        Returns
        -------
        score_pos:
            Predicted scores for positive (user, item) pairs.
        score_neg:
            Predicted scores for negative (user, item) pairs.
        """
        ...

    @abc.abstractmethod
    def predict(self, user_id: int, item_ids: torch.Tensor) -> torch.Tensor:
        """Score a list of candidate items for a single user.

        Parameters
        ----------
        user_id:
            Scalar user index.
        item_ids:
            1-D tensor of candidate item indices.

        Returns
        -------
        torch.Tensor
            Predicted scores, same length as *item_ids*.
        """
        ...

    def bpr_loss(self, score_pos: torch.Tensor, score_neg: torch.Tensor) -> torch.Tensor:
        """BPR pairwise loss with L2 regularisation.

        .. math::
            \\mathcal{L} = -\\frac{1}{N}\\sum \\log\\sigma(\\hat y_{pos}
            - \\hat y_{neg}) + \\lambda \\|\\Theta\\|^2
        """
        bpr = -torch.log(torch.sigmoid(score_pos - score_neg) + 1e-10).mean()
        reg = self.config.get("l2_reg", 0.0) * self.l2_reg()
        return bpr + reg

    def l2_reg(self) -> torch.Tensor:
        """Sum of squared L2 norms over all trainable parameters."""
        reg = torch.tensor(0.0, device=self._device())
        for param in self.parameters():
            if param.requires_grad:
                reg = reg + param.norm(2).pow(2)
        return reg

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        best_metric: float,
        config_hash: str,
    ) -> None:
        """Persist complete training state for later resumption."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metric": best_metric,
                "config_hash": config_hash,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> dict:
        """Restore model weights and return the full checkpoint dict.

        ``weights_only=False`` is intentional: framework-produced
        training checkpoints carry a hyperparams dict + optimiser
        state alongside the state_dict.  Trusted source.
        """
        checkpoint = torch.load(path, map_location=self._device(), weights_only=False)
        self.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint

    def _device(self) -> torch.device:
        """Infer the current device from the first parameter (or CPU)."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _init_embedding(embedding: nn.Embedding) -> None:
        """Xavier-uniform initialisation for an embedding table."""
        nn.init.xavier_uniform_(embedding.weight)
