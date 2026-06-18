# Learned (online) fusion

Most of the framework's fusion strategies are *offline*: they produce
a fixed `(n_items, D)` embedding once, write it to disk, and the
recommender consumes it as a non-trainable buffer.  The
``adaptive_gated`` strategy is *online* — its parameters are updated
via backprop alongside the recommender's, so the fusion adapts to
the downstream BPR objective.

## Specification

For each item *i* with two source embeddings
\(\mathbf{e}_i^{(1)}, \mathbf{e}_i^{(2)} \in \mathbb{R}^D\),
the fusion module produces a per-dimension gate vector

```
g_i = σ(MLP_gate([e_i^(1) ‖ e_i^(2)]))   ∈ [0, 1]^D
h_i = g_i ⊙ e_i^(1) + (1 - g_i) ⊙ e_i^(2)
```

where ``MLP_gate`` is ``Linear(2D → D) → ReLU → Linear(D → D)`` and
the sigmoid is applied to the final layer's output.  The final
linear layer is zero-initialised so the *initial* gate is uniform
(``0.5`` everywhere) — the first epoch begins with the same
representation as plain ``mean`` fusion.

The gate's parameters are members of the recommender's ``state_dict``
and are saved/loaded with every checkpoint.

## How to enable

In ``configs/fusion.yaml``:

```yaml
fusion_strategies_enabled:
  - adaptive_gated   # add to the list

strategies:
  adaptive_gated: {}
```

Run the pipeline normally:

```bash
python main.py --from fuse        # builds the JSON sidecar (no .npy)
python main.py --from train       # the recommender picks up the
                                  # 3-D buffer and instantiates the
                                  # gate module automatically
```

## On-disk artefacts

* No ``hybrid_adaptive_gated_<dim>.npy`` is written.
* Instead, ``data/embeddings/<dataset>/hybrid_adaptive_gated_<dim>.json``
  is created — a small sidecar listing the component embeddings the
  trainer must stack.
* At training time, ``src.fusions.load_embedding`` reads the sidecar,
  loads each component (e.g. ``resnet50_D128.npy`` and
  ``vit_b16_D128.npy``), and stacks them into ``(n_items, 2, D)``.
* The recommender (``BaseRecommender``) detects the 3-D buffer in its
  constructor, instantiates the matching online module, and exposes
  ``_resolve_visual(item_ids) -> (B, D)`` so concrete recommenders
  remain agnostic to whether features come from an offline ``.npy`` or
  an online module.

> **Not every 3-D buffer is an online fusion.** A 3-D buffer here means
> *two stacked source embeddings* to be fused (``M == 2``). ACF also
> receives a 3-D buffer, but its ``M`` axis holds an item's *components*
> (49–256 spatial cells / patch tokens), not fusion sources. ACF sets
> the class attribute ``consumes_raw_components = True`` so the base
> class keeps the raw buffer and skips online-fusion instantiation —
> the model runs its own component-level attention instead of
> ``_resolve_visual``. See ``docs/extending.md`` § 3.4.

## Caching caveat

Recommenders such as VBPR, AVBPR, VNPR and DeepStyle cache the
projected visual features for the full item catalogue (used by
``predict_batch`` during evaluation).  When an online fusion is
active the cache is **bypassed** — the gate's output depends on
trainable parameters and would otherwise return stale values
across optimisation steps.  Evaluation cost is therefore slightly
higher with ``adaptive_gated`` than with offline strategies, but
correctness is preserved.

## Adding a new online fusion

1. Subclass :class:`torch.nn.Module` under ``src/fusions/online.py``
   (or in a plugin directory) with a ``forward(e1, e2, ...)`` method
   that returns the fused tensor.
2. Add a branch to ``online_module_for(name, dim)`` mapping the
   strategy name to your module.
3. Register the strategy with ``online=True``:

   ```python
   from src.fusions.registry import register_fusion_strategy

   def _placeholder(*_a, **_kw):
       raise NotImplementedError("online strategy")

   register_fusion_strategy(
       "my_learned_fusion",
       _placeholder,
       equal_dim_required=True,
       online=True,
   )
   ```

4. The rest of the pipeline (sidecar writer, embedding loader,
   recommender wiring) needs no changes — the framework treats every
   ``online=True`` strategy uniformly.

## Citation

This implementation follows the specification of the
``adaptive_gated`` strategy in the qualification document of
**Couto, L. (2026), "Hybrid Visual Recommendation for Fashion"**,
chapter "Methodology — Fusion strategies".
