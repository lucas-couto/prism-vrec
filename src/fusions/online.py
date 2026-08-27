"""Online (learned) fusion modules.

Unlike the ten offline strategies under :mod:`src.fusions.strategies`,
which produce a fixed ``(n_items, D)`` matrix once and write it to
disk, online fusion modules are :class:`torch.nn.Module` subclasses
whose parameters are trained jointly with the recommender via the BPR
loss.  They receive the M source embeddings *separately* at every
forward pass and produce the fused representation on the fly.

Implemented here
----------------
* :class:`AdaptiveGatedFusion` — per-item per-dimension gated fusion
  of two equal-dimensional embeddings, as defined in the qualification
  document of this dissertation (Couto, 2026).

How online fusions plug into the framework
------------------------------------------
1. The :mod:`src.fusions.strategies` registry marks the strategy with
   ``online=True``.  The offline ``fn`` is a placeholder that raises
   :class:`NotImplementedError` — calling an online strategy as if it
   were offline is a programming error.
2. :func:`src.steps.fuse._collect_fusion_tasks` skips online strategies
   (no ``.npy`` to write) and instead emits a small JSON sidecar at
   ``data/embeddings/<dataset>/hybrid_<strategy>_<dim>.json``
   describing which source embeddings are needed at training time.
3. :func:`src.steps.train` loads the two component ``.npy`` files,
   stacks them into a ``(n_items, M, D)`` array, and passes that to
   the recommender constructor instead of the usual ``(n_items, D)``.
4. :class:`src.recommenders.base.BaseRecommender` detects a 3-D buffer,
   instantiates the matching online module on construction, and
   exposes ``_resolve_visual(item_ids) -> (B, D)`` so each recommender
   stays agnostic to whether features come from an offline ``.npy`` or
   from a learned fusion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class RaggedSources(np.ndarray):
    """Concatenated native sources ``(n_items, sum(D_i))`` + metadata.

    Produced by :func:`load_embedding` for learned-alignment fusion
    sidecars.  The array itself is the axis-1 concatenation of the M
    native source matrices; the attached attributes let the recommender
    build a :class:`LearnedAlignmentFusion` that splits, projects and
    combines them at every forward pass:

    * ``source_dims`` — the native dim of each source, in concat order.
    * ``strategy`` — the element-wise fusion op to apply after
      projection (``mean``/``sum``/.../``adaptive_gated``).
    * ``aligned_dim`` — the target dimension D of the learned alignment.
    * ``normalize`` — whether projected sources are L2-normalised
      before the op (mirrors ``normalize_before_fusion``).
    * ``fusion_kwargs`` — strategy hyperparameters (e.g. fixed
      ``weights`` for weighted_mean).
    """

    def __new__(
        cls,
        arr: np.ndarray,
        *,
        source_dims: list[int],
        strategy: str,
        aligned_dim: int,
        normalize: bool = True,
        fusion_kwargs: dict | None = None,
    ) -> RaggedSources:
        obj = np.asarray(arr).view(cls)
        obj.source_dims = list(source_dims)
        obj.strategy = str(strategy)
        obj.aligned_dim = int(aligned_dim)
        obj.normalize = bool(normalize)
        obj.fusion_kwargs = dict(fusion_kwargs or {})
        return obj

    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return
        self.source_dims = getattr(obj, "source_dims", [])
        self.strategy = getattr(obj, "strategy", "")
        self.aligned_dim = getattr(obj, "aligned_dim", 0)
        self.normalize = getattr(obj, "normalize", True)
        self.fusion_kwargs = getattr(obj, "fusion_kwargs", {})


class LearnedAlignmentFusion(nn.Module):
    """Learned alignment + element-wise fusion of differing-dim sources.

    The v2 fusion analogue of the recommender's projection ``E``: each
    native source (e.g. ResNet-50 2048-d, ViT-B/16 768-d) gets its own
    learned ``Linear(D_i -> D)`` trained jointly with the recommender by
    the BPR loss; the configured element-wise op then combines the
    aligned sources.  This is the ``alignment.method = learned`` path;
    the non-supervised counterpart is offline
    :func:`src.fusions.strategies.pca_align`.

    Supported ops mirror the offline equal-dim family: ``mean``,
    ``sum``, ``prod``, ``max_pool``, ``weighted_mean`` (fixed weights
    from config), ``attention_weighted`` (softmax over fixed
    configurable logits), ``gated`` (normalised sigmoids over fixed
    configurable logits) and ``adaptive_gated`` (per-item gate MLP;
    2 sources only).  Per the thesis spec, the whole weighted-
    combination family uses FIXED convex weights — only the
    per-source projections and ``adaptive_gated``'s MLP are learned.
    """

    # METHODOLOGICAL GUARANTEE — identical projection across fusions.
    # The per-source alignment is a plain ``Linear(D_i -> dim)`` with
    # bias, Xavier-uniform init, no activation, no normalisation layer,
    # and it is built once, in ``build_projections``, for every strategy
    # this class supports.  ``dim`` comes from the single
    # ``alignment.dim`` config key, so mean/sum/prod/max_pool/
    # weighted_mean/attention_weighted/gated/adaptive_gated all get
    # projections of exactly the same architecture and capacity.  Each
    # trained model learns its own weights (unavoidable and intended),
    # but when two fusions are compared the fusion *operation* is the
    # only structural difference between them — the projection is a
    # controlled variable, never a confound.  Do NOT add per-strategy
    # branches that change the projection (extra layers, different
    # dims, dropout) without revisiting this guarantee.
    _SIMPLE_OPS = ("mean", "sum", "prod", "max_pool")
    _LOGIT_OPS = ("attention_weighted", "gated")

    def __init__(
        self,
        source_dims: list[int],
        dim: int,
        strategy: str = "mean",
        *,
        normalize: bool = True,
        weights: list[float] | None = None,
        logits: list[float] | None = None,
    ) -> None:
        super().__init__()
        if not source_dims:
            raise ValueError("LearnedAlignmentFusion needs at least 1 source.")
        self.source_dims = list(source_dims)
        self.strategy = strategy
        self.normalize = normalize
        self.projections = self.build_projections(source_dims, dim)

        m = len(source_dims)
        if strategy == "weighted_mean":
            w = weights if weights is not None else [1.0 / m] * m
            if len(w) != m:
                raise ValueError(f"weighted_mean needs {m} weights, got {len(w)}.")
            total = float(sum(w))
            self.register_buffer("fixed_weights", torch.tensor([x / total for x in w]))
        elif strategy in self._LOGIT_OPS:
            # FIXED configurable logits (thesis spec), mirroring how
            # weighted_mean handles fixed_weights: a non-trainable
            # buffer, never a Parameter.  Defaults to uniform (logits 0).
            lg = logits if logits is not None else [0.0] * m
            if len(lg) != m:
                raise ValueError(f"{strategy} needs {m} logits, got {len(lg)}.")
            self.register_buffer("fixed_logits", torch.tensor([float(x) for x in lg]))
        elif strategy == "adaptive_gated":
            if m != 2:
                raise ValueError("adaptive_gated supports exactly 2 sources.")
            self.gate = AdaptiveGatedFusion(dim=dim)
        elif strategy not in self._SIMPLE_OPS:
            raise ValueError(f"Unknown learned-alignment fusion op: {strategy!r}.")

    @staticmethod
    def build_projections(source_dims: list[int], dim: int) -> nn.ModuleList:
        """The one and only projection architecture (see class comment).

        One ``Linear(D_i -> dim)`` per source, Xavier-uniform weight,
        zero bias.  Strategy-agnostic by construction: the caller never
        passes the strategy, so no branch can specialise it.
        """
        projections = nn.ModuleList(nn.Linear(d, dim) for d in source_dims)
        for proj in projections:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.zeros_(proj.bias)
        return projections

    def _aligned(self, concat: torch.Tensor) -> list[torch.Tensor]:
        parts = torch.split(concat, self.source_dims, dim=-1)
        aligned = [proj(part) for proj, part in zip(self.projections, parts, strict=True)]
        if self.normalize:
            aligned = [nn.functional.normalize(a, p=2, dim=-1, eps=1e-12) for a in aligned]
        return aligned

    def forward(self, concat: torch.Tensor) -> torch.Tensor:
        aligned = self._aligned(concat)

        if self.strategy == "adaptive_gated":
            return self.gate(aligned[0], aligned[1])

        stacked = torch.stack(aligned, dim=0)  # (M, B, D)
        if self.strategy == "mean":
            return stacked.mean(dim=0)
        if self.strategy == "sum":
            return stacked.sum(dim=0)
        if self.strategy == "prod":
            return stacked.prod(dim=0)
        if self.strategy == "max_pool":
            return stacked.max(dim=0).values
        if self.strategy == "weighted_mean":
            w = self.fixed_weights.view(-1, 1, 1)
            return (stacked * w).sum(dim=0)
        if self.strategy == "attention_weighted":
            alphas = torch.softmax(self.fixed_logits, dim=0).view(-1, 1, 1)
            return (stacked * alphas).sum(dim=0)
        if self.strategy == "gated":
            gates = torch.sigmoid(self.fixed_logits)
            gates = (gates / gates.sum()).view(-1, 1, 1)
            return (stacked * gates).sum(dim=0)
        raise RuntimeError(f"unreachable op {self.strategy!r}")


class AdaptiveGatedFusion(nn.Module):
    """Per-item per-dimension gated fusion of two equal-dim embeddings.

    Following the formal specification of the
    ``adaptive_gated`` strategy in the qualification document:

    .. math::
        \\mathbf{g}_i &= \\sigma\\!\\bigl(
            \\mathrm{MLP}_{\\mathrm{gate}}\\bigl(
                [\\mathbf{e}_i^{(1)} \\Vert \\mathbf{e}_i^{(2)}]
            \\bigr)
        \\bigr) \\in [0, 1]^D \\\\
        \\mathbf{h}_i &= \\mathbf{g}_i \\odot \\mathbf{e}_i^{(1)}
            + (1 - \\mathbf{g}_i) \\odot \\mathbf{e}_i^{(2)}

    where ``MLP_gate`` is ``Linear(2D → D) → ReLU → Linear(D → D)``
    followed by a sigmoid on the output, and ``odot`` denotes the
    element-wise product.  The FINAL linear layer is zero-initialised
    so the initial gate is ``sigmoid(0) = 0.5`` everywhere — training
    starts from the uniform (plain ``mean``) fusion.

    Parameters
    ----------
    dim:
        Per-source embedding dimensionality ``D``.  Both inputs must
        share this value.
    hidden_dim:
        Width of the MLP's hidden layer.  Defaults to ``dim`` (matches
        the qualification's ``2D -> D -> D`` architecture).
    """

    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}.")
        self.dim = dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else dim

        self.gate = nn.Sequential(
            nn.Linear(2 * dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, dim),
        )
        # Sigmoid is applied separately so the linear layer is exposed
        # for diagnostics (``gate[-1].weight`` retrieves the gate's
        # final layer).

        # Zero-initialise the FINAL linear layer (weight and bias), per
        # the qualification spec, so that at step 0:
        #   gate[-1](·) = 0  →  sigmoid(0) = 0.5  (uniform fusion)
        # The first layer keeps its default (Kaiming-uniform) init, so
        # its ReLU activations are non-zero from the start: gate[-1]
        # receives a non-zero gradient on the first backward, and once
        # its weight moves off zero, gradients flow into gate[0] too —
        # no dead-gradient pathology.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the gated fusion.

        Parameters
        ----------
        e1, e2:
            Source embeddings, each shape ``(..., dim)`` — typically
            ``(batch, dim)`` during training.

        Returns
        -------
        torch.Tensor
            Fused tensor of shape ``(..., dim)``.
        """
        if e1.shape != e2.shape:
            raise ValueError(
                f"Inputs to AdaptiveGatedFusion must share shape; "
                f"got {tuple(e1.shape)} vs {tuple(e2.shape)}.",
            )
        if e1.shape[-1] != self.dim:
            raise ValueError(
                f"Inputs trailing dim must equal self.dim={self.dim}; got {e1.shape[-1]}.",
            )

        gate_input = torch.cat([e1, e2], dim=-1)
        g = torch.sigmoid(self.gate(gate_input))
        return g * e1 + (1.0 - g) * e2

    def gate_values(
        self,
        e1: torch.Tensor,
        e2: torch.Tensor,
    ) -> torch.Tensor:
        """Return the raw per-dimension gate values without applying them.

        Useful for analysis and figures — the per-dimension gate
        distribution is the most direct visualisation of *which*
        extractor the model attends to for each item.
        """
        gate_input = torch.cat([e1, e2], dim=-1)
        return torch.sigmoid(self.gate(gate_input))


# Maps an online strategy name to the nn.Module that implements it. The
# error message below derives its list from this dict so the two never
# drift apart when a new online strategy is added.
_ONLINE_MODULES: dict[str, type[nn.Module]] = {
    "adaptive_gated": AdaptiveGatedFusion,
}


def online_module_for(strategy_name: str, dim: int) -> nn.Module:
    """Factory: instantiate the matching online module for a strategy."""
    factory = _ONLINE_MODULES.get(strategy_name)
    if factory is None:
        raise ValueError(
            f"Unknown online fusion strategy: {strategy_name!r}. "
            f"Available online strategies: {sorted(_ONLINE_MODULES)}.",
        )
    return factory(dim=dim)


def load_embedding(path: str | Path):
    """Load a visual-embedding artefact, transparently handling sidecars.

    Two on-disk formats are accepted:

    * ``.npy`` — direct numpy load; returns a ``(n_items, D)`` array.
      Standard for every offline fusion strategy and for plain
      single-extractor embeddings.
    * ``.json`` — sidecar produced by :func:`src.steps.fuse._fuse_single`
      for online strategies.  Lists the component embeddings; this
      function loads each one, validates they share shape, and stacks
      them along ``axis=1`` into ``(n_items, M, D)`` ready to feed an
      online fusion module.

    Returns
    -------
    np.ndarray
        Either 2-D ``(n_items, D)`` or 3-D ``(n_items, M, D)``.
    """
    import json
    from pathlib import Path

    import numpy as np

    p = Path(path)
    if p.suffix == ".json":
        sidecar = json.loads(p.read_text(encoding="utf-8"))
        components = sidecar.get("components") or []
        if not components:
            raise ValueError(
                f"online sidecar {p} lists no components; cannot stack.",
            )
        if len(components) == 1:
            # Degenerate but well-defined: a single-source fusion (e.g. the
            # smoke profile) reduces to one learned projection / an M=1
            # stack. Real grids always fuse >= 2 sources.
            from src.utils.logging import get_logger

            get_logger(__name__).warning(
                "online sidecar %s has a single component (%s); "
                "fusion degenerates to a passthrough of that source.",
                p,
                components[0],
            )
        loaded = []
        for fname in components:
            comp_path = p.parent / fname
            if not comp_path.exists():
                raise FileNotFoundError(
                    f"sidecar {p} references missing component {comp_path}",
                )
            loaded.append(np.load(comp_path))

        if sidecar.get("alignment") == "learned":
            # Native sources with differing dims: concatenate along the
            # feature axis and attach the metadata the recommender needs
            # to build a LearnedAlignmentFusion (per-source learned
            # projections co-trained via BPR).
            n_rows = {arr.shape[0] for arr in loaded}
            if len(n_rows) != 1:
                raise ValueError(
                    f"sidecar {p}: components disagree on n_items ({sorted(n_rows)}).",
                )
            return RaggedSources(
                np.concatenate(loaded, axis=1),
                source_dims=[int(arr.shape[1]) for arr in loaded],
                strategy=sidecar["strategy"],
                aligned_dim=int(sidecar["dim"]),
                normalize=bool(sidecar.get("normalize", True)),
                fusion_kwargs=sidecar.get("fusion_kwargs") or {},
            )

        first_shape = loaded[0].shape
        for fname, arr in zip(components, loaded, strict=True):
            if arr.shape != first_shape:
                raise ValueError(
                    f"sidecar {p}: component {fname} has shape "
                    f"{arr.shape}, expected {first_shape}.",
                )
        return np.stack(loaded, axis=1)  # (n_items, M, D)

    arr = np.load(p)
    _validate_against_meta(p, arr)
    return arr


def _validate_against_meta(npy_path, arr) -> None:
    """Cross-check a feature file against its ``.meta.json`` sidecar.

    Fails loudly when the loaded array does not correspond to the
    backbone declared in the metadata (wrong native_dim or a stem/name
    mismatch) — the exact silent-mixup the sidecar exists to prevent.
    Files without a sidecar (fusion outputs, legacy artifacts) pass.
    """
    import json

    meta_path = npy_path.with_suffix("").with_suffix(".meta.json")
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    declared_dim = meta.get("native_dim")
    if declared_dim is not None and arr.shape[-1] != int(declared_dim):
        raise ValueError(
            f"{npy_path}: array last dim is {arr.shape[-1]} but its "
            f"metadata declares native_dim={declared_dim} "
            f"(extractor={meta.get('name')}). The features on disk do not "
            "match the backbone they claim to be — re-extract."
        )
    declared_name = meta.get("name")
    if declared_name is not None and npy_path.stem not in {
        declared_name,
        f"{declared_name}_comp",
    }:
        raise ValueError(
            f"{npy_path}: file stem {npy_path.stem!r} does not match the "
            f"extractor name {declared_name!r} declared in {meta_path.name}."
        )


__all__ = [
    "AdaptiveGatedFusion",
    "LearnedAlignmentFusion",
    "RaggedSources",
    "load_embedding",
    "online_module_for",
]
