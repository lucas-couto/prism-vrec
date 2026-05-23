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

import torch
import torch.nn as nn


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
    element-wise product.

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
            # Tanh is used instead of ReLU because its derivative at x=0
            # is exactly 1 (not 0).  This matters because the first linear
            # layer (gate.0) is zero-initialised so its pre-activation
            # output is identically 0 at step 0.  With ReLU the subgradient
            # at 0 is 0, which would permanently block gradients from
            # reaching gate.0.weight via the chain rule through gate[-1].
            # Tanh(0)=0 preserves the initial gate = sigmoid(0) = 0.5
            # (uniform fusion property), while Tanh'(0)=1 ensures
            # gate.0.weight receives a non-zero gradient from the first
            # backward pass.
            nn.Tanh(),
            nn.Linear(self.hidden_dim, dim),
        )
        # Sigmoid is applied separately so the linear layer is exposed
        # for diagnostics (``gate[-1].weight`` retrieves the gate's
        # final layer).

        # Initialise the FIRST linear layer with zeros so that at step 0:
        #   gate[0](cat) = 0  →  Tanh(0) = 0  →  gate[-1](0) = bias[-1] = 0
        #   →  sigmoid(0) = 0.5  (uniform fusion)
        # The final layer keeps its Kaiming-uniform weight so that
        # gate[-1].weight.T is non-zero, allowing gradients to propagate
        # back through Tanh'(0)=1 into gate[0].weight from the first step.
        nn.init.zeros_(self.gate[0].weight)
        nn.init.zeros_(self.gate[0].bias)
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


def online_module_for(strategy_name: str, dim: int) -> nn.Module:
    """Factory: instantiate the matching online module for a strategy."""
    if strategy_name == "adaptive_gated":
        return AdaptiveGatedFusion(dim=dim)
    raise ValueError(
        f"Unknown online fusion strategy: {strategy_name!r}. "
        f"Available online strategies: ['adaptive_gated'].",
    )


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
        if len(components) < 2:
            raise ValueError(
                f"online sidecar {p} has fewer than 2 components "
                f"(got {len(components)}); cannot stack.",
            )
        loaded = []
        first_shape = None
        for fname in components:
            comp_path = p.parent / fname
            if not comp_path.exists():
                raise FileNotFoundError(
                    f"sidecar {p} references missing component {comp_path}",
                )
            arr = np.load(comp_path)
            if first_shape is None:
                first_shape = arr.shape
            elif arr.shape != first_shape:
                raise ValueError(
                    f"sidecar {p}: component {fname} has shape "
                    f"{arr.shape}, expected {first_shape}.",
                )
            loaded.append(arr)
        return np.stack(loaded, axis=1)  # (n_items, M, D)

    return np.load(p)


__all__ = [
    "AdaptiveGatedFusion",
    "load_embedding",
    "online_module_for",
]
