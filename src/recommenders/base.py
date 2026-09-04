"""Abstract base recommender with BPR loss, L2 regularization, and checkpointing."""

from __future__ import annotations

import abc

import numpy as np
import torch
import torch.nn as nn


def _record_bpr_batch(module: BaseRecommender, args: tuple) -> None:
    """Forward pre-hook: remember the (user, pos, neg) index batch.

    :meth:`BaseRecommender.bpr_loss` keeps its fixed
    ``(score_pos, score_neg)`` signature, so the batch indices needed by
    the BPR-Opt regulariser are captured transparently whenever the
    module is called with the standard triple of index tensors.
    """
    if len(args) == 3:
        module._last_batch = args


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

    #: Embedding tables whose rows are gathered per batch for the
    #: BPR-Opt L2 penalty (see :meth:`l2_reg`).  ``_L2_USER_TABLES`` are
    #: indexed by the batch's user ids, ``_L2_ITEM_TABLES`` by both the
    #: positive and the negative item ids.  Missing attributes are
    #: skipped, so a model lacking e.g. ``item_bias`` needs no override.
    #: ``_L2_EXTRA_GATHERED_TABLES`` names tables whose rows are gathered
    #: by a model-specific :meth:`_l2_gathered_terms` override — listing
    #: them here only excludes their full weight matrix from the
    #: shared-parameter term.
    _L2_USER_TABLES: tuple[str, ...] = ("user_embedding",)
    _L2_ITEM_TABLES: tuple[str, ...] = ("item_embedding", "item_bias")
    _L2_EXTRA_GATHERED_TABLES: tuple[str, ...] = ()

    #: Every ``nn.Embedding`` attribute indexed by USER id — the rows a
    #: fold-in routine (:func:`src.folds.foldin.fold_in_users`) re-
    #: initialises and optimises for a held-out user while the rest of
    #: the model stays frozen.  DISTINCT from :attr:`_L2_USER_TABLES`,
    #: which only governs regularisation (a model may own a user table
    #: it does not regularise per row; VNPR did until 2026-09-04, when
    #: its whole-matrix L2 proved fatal under Adam — see its module
    #: docstring).  Models whose per-user state is NOT a
    #: parameter (ACF's history buffers) refresh it through
    #: :meth:`rebuild_user_state` instead.  A test guards that every
    #: ``nn.Embedding`` with ``num_embeddings == n_users`` is listed.
    _USER_TABLES: tuple[str, ...] = ("user_embedding",)

    #: ``λ`` config key per parameter group, so each model reproduces the
    #: regularisation its paper declares.  Keys are an attribute name or
    #: an ``(attribute, role)`` pair with role ``"user"`` / ``"pos"`` /
    #: ``"neg"`` (gathered rows) or ``"shared"`` (dense parameters);
    #: anything unlisted uses the single ``"l2_reg"`` key.
    _L2_LAMBDA_KEYS: dict[str | tuple[str, str], str] = {}
    #: Value of a ``λ`` key absent from the config (e.g. VBPR's
    #: ``λ_E = 0``); keys absent here too fall back to ``config["l2_reg"]``.
    _L2_LAMBDA_DEFAULTS: dict[str, float] = {}
    #: Top-level attribute names (parameters or submodules) excluded from
    #: every L2 term — e.g. ACF's attention nets, which its objective
    #: (Eq. 5) leaves unpenalised.
    _L2_UNREGULARIZED: tuple[str, ...] = ()

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

        # Batch recorded by the forward pre-hook; consumed by the
        # BPR-Opt regulariser in bpr_loss / l2_reg.
        self._last_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.register_forward_pre_hook(_record_bpr_batch)

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

    def rebuild_user_state(self, interactions: dict[int, set[int]]) -> None:
        """Refresh NON-parametric per-user state from ``interactions``.

        Hook for the fold-in routine.  Models whose user representation
        is entirely parametric (the :attr:`_USER_TABLES` rows) have
        nothing to do — the default is a no-op.  Models that derive
        per-user state from the interaction history at construction
        time (ACF's ``history_items`` / ``history_mask`` buffers)
        override it to update ONLY the rows of the users present in
        ``interactions``, leaving every other user untouched.

        Parameters
        ----------
        interactions:
            ``{user_idx: set(item_idx)}`` — the profile of the users
            being folded in.
        """

    def bpr_loss(self, score_pos: torch.Tensor, score_neg: torch.Tensor) -> torch.Tensor:
        """BPR pairwise loss plus the model's L2 regulariser.

        .. math::
            \\mathcal{L} = -\\frac{1}{N}\\sum_{(u,i,j)}
            \\log\\sigma(\\hat y_{ui} - \\hat y_{uj})
            + \\sum_g \\lambda_g \\Big( \\frac{1}{N}\\sum_{(u,i,j)}
            \\|\\theta^{g}_{(u,i,j)}\\|^2 + \\|\\Theta^{g}_{shared}\\|^2 \\Big)

        Following BPR-Opt (Rendle et al., 2009), the L2 term penalises
        the parameters *touched* by each sampled triple ``(u, i, j)``:

        * ``θ_(u,i,j)`` — the embedding rows gathered for the triple
          (user row, positive/negative item rows, plus model-specific
          per-row extras such as visual-user rows, category rows, or
          ACF history rows).  Their squared norms are summed *as
          gathered* and mean-reduced over the ``N`` triples, matching
          the mean-reduced log-loss.  A row appearing in two triples of
          the batch is therefore penalised once per occurrence.
        * ``Θ_shared`` — dense parameters that participate in *every*
          triple's score function (visual projections, MLPs, attention
          nets, online-fusion modules).  These are penalised in full
          each step.

        Every parameter group ``g`` carries its own constant
        ``λ_g`` so each model reproduces the regularisation its paper
        declares (BPR-MF's ``λ_W / λ_H+ / λ_H-``, VBPR's ``λ_Θ / λ_β /
        λ_E``, ACF's unpenalised attention nets, ...): see
        :attr:`_L2_LAMBDA_KEYS`, :attr:`_L2_LAMBDA_DEFAULTS` and
        :attr:`_L2_UNREGULARIZED`.  The single ``l2_reg`` key remains
        the default group every other key falls back to.

        The gathered term needs the ``(user, pos, neg)`` index batch of
        the preceding forward call — recorded transparently by a forward
        pre-hook, keeping this signature unchanged.  When the loss is
        invoked without a prior forward, only the shared term applies.
        """
        # F.logsigmoid is the numerically stable form of the documented
        # -ln σ(x̂_uij): the old log(sigmoid(x)+1e-10) saturated at ≈23
        # for strongly-wrong pairs and ZEROED their gradient (F9).
        bpr = -torch.nn.functional.logsigmoid(score_pos - score_neg).mean()
        return bpr + self.l2_reg()

    def l2_reg(self) -> torch.Tensor:
        """Weighted BPR-Opt regulariser: ``Σ_g λ_g (gathered_g / N + shared_g)``.

        Without a recorded batch only the shared terms are returned.  The
        recorded batch is CONSUMED: a second call without a new forward
        falls back to the shared terms instead of silently reusing a
        stale batch (R11).
        """
        reg = torch.tensor(0.0, device=self._device())
        for key, term in self._l2_shared_terms():
            reg = reg + self._l2_lambda(key) * term
        batch, self._last_batch = self._last_batch, None
        if batch is None:
            return reg
        user_ids, pos_item_ids, neg_item_ids = batch
        batch_size = user_ids.shape[0]
        for key, rows in self._l2_gathered_terms(user_ids, pos_item_ids, neg_item_ids):
            reg = reg + self._l2_lambda(key) * rows.pow(2).sum() / batch_size
        return reg

    def _l2_lambda_key(self, name: str, role: str) -> str:
        """Config key of the ``λ`` governing ``name`` in ``role``.

        ``role`` is ``"user"`` / ``"pos"`` / ``"neg"`` for gathered rows
        and ``"shared"`` for dense parameters.  Lookup order:
        ``(name, role)`` → ``name`` → ``"l2_reg"``.
        """
        keys = self._L2_LAMBDA_KEYS
        return keys.get((name, role), keys.get(name, "l2_reg"))

    def _l2_lambda(self, key: str) -> float:
        """Value of the ``λ`` under ``key``: config → class default → ``l2_reg``."""
        if key in self.config:
            return float(self.config[key])
        if key in self._L2_LAMBDA_DEFAULTS:
            return float(self._L2_LAMBDA_DEFAULTS[key])
        return float(self.config.get("l2_reg", 0.0))

    def _l2_gathered_terms(
        self,
        user_ids: torch.Tensor,
        pos_item_ids: torch.Tensor,
        neg_item_ids: torch.Tensor,
    ) -> list[tuple[str, torch.Tensor]]:
        """``(λ key, rows)`` pairs of the embedding rows touched by the batch.

        Default: rows of :attr:`_L2_USER_TABLES` for the batch users and
        rows of :attr:`_L2_ITEM_TABLES` for the positive and negative
        items.  Models with additional per-triple rows (e.g. category or
        history rows) extend the returned list.
        """
        terms: list[tuple[str, torch.Tensor]] = []
        for name in self._L2_USER_TABLES:
            table = getattr(self, name, None)
            if table is not None:
                terms.append((self._l2_lambda_key(name, "user"), table(user_ids)))
        for name in self._L2_ITEM_TABLES:
            table = getattr(self, name, None)
            if table is not None:
                terms.append((self._l2_lambda_key(name, "pos"), table(pos_item_ids)))
                terms.append((self._l2_lambda_key(name, "neg"), table(neg_item_ids)))
        return terms

    def _l2_shared_terms(self) -> list[tuple[str, torch.Tensor]]:
        """``(λ key, ‖param‖²)`` for every dense parameter shared by all triples.

        Every trainable parameter except the weight matrices of the
        row-gathered embedding tables (penalised per batch row in
        :meth:`_l2_gathered_terms`) and anything under
        :attr:`_L2_UNREGULARIZED`.  The key is resolved from the
        parameter's top-level attribute name with role ``"shared"``.
        """
        gathered_names = (
            *self._L2_USER_TABLES,
            *self._L2_ITEM_TABLES,
            *self._L2_EXTRA_GATHERED_TABLES,
        )
        excluded: set[int] = set()
        for name in gathered_names:
            table = getattr(self, name, None)
            if table is not None:
                excluded.add(id(table.weight))
        terms: list[tuple[str, torch.Tensor]] = []
        for full_name, param in self.named_parameters():
            top = full_name.split(".", 1)[0]
            if not param.requires_grad or id(param) in excluded:
                continue
            if top in self._L2_UNREGULARIZED:
                continue
            terms.append((self._l2_lambda_key(top, "shared"), param.pow(2).sum()))
        return terms

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
