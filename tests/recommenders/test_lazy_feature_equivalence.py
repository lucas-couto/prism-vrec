"""SDD M02/M03: lazy (bounded) feature sources reproduce the dense models.

For every built-in visual recommender — VBPR, AVBPR, DeepStyle, VNPR
(pooled, stacked online fusion and learned alignment) and ACF (fp16
components) — a model fed a :class:`FeatureSource` must yield the same
scores, loss and gradients as the same model fed the dense array, while
registering no raw feature buffer (so ``model.to`` moves none) and
reading only the rows a forward pass asks for.

Tolerances: rtol=1e-5 / atol=1e-7 for fp32 scores, the existing VNPR
item-block tolerance; gradients use the same bound.  No op-specific
loosening was needed on CPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.feature_source import (
    ArrayFeatureSource,
    ConcatFeatureSource,
    NpyFeatureSource,
    StackedFeatureSource,
)
from src.fusions.online import RaggedSources
from src.recommenders.acf import ACF
from src.recommenders.avbpr import AVBPR
from src.recommenders.deepstyle import DeepStyle
from src.recommenders.vbpr import VBPR
from src.recommenders.vnpr import VNPR

N_USERS, N_ITEMS, DV, M_COMP = 7, 23, 6, 3
RTOL, ATOL = 1e-5, 1e-7
USERS = torch.tensor([0, 3, 3, 6, 1])
POS = torch.tensor([2, 9, 9, 22, 4])
NEG = torch.tensor([9, 0, 15, 4, 4])
SUBSET = torch.tensor([22, 4, 4, 0, 13])
HISTORY = {0: {2, 5, 9}, 1: {4}, 2: set(), 3: {9, 10, 11, 12, 13, 14}, 6: {22, 0}}
CATEGORIES = np.arange(N_ITEMS) % 4


def _pooled(seed: int = 0, d: int = DV, n: int = N_ITEMS) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, d)).astype(np.float32)


class _Recording:
    """Delegating source that records every ``read_rows`` request."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.requests: list[np.ndarray] = []

    @property
    def shape(self):
        return self.inner.shape

    @property
    def dtype(self):
        return self.inner.dtype

    def read_rows(self, item_ids):
        self.requests.append(np.asarray(item_ids).copy())
        return self.inner.read_rows(item_ids)

    def close(self) -> None:
        self.inner.close()

    def __getattr__(self, name):  # recipe attributes of a concat source
        return getattr(self.inner, name)


# ------------------------------------------------------------ feature variants
def _variant(kind: str, tmp_path: Path):
    """``(dense_array, lazy_source)`` for one feature layout."""
    if kind == "pooled":
        arr = _pooled(1)
        path = tmp_path / "resnet50.npy"
        np.save(path, arr)
        return arr, NpyFeatureSource(path)
    if kind == "stacked":
        a, b = _pooled(2), _pooled(3)
        return np.stack([a, b], axis=1), StackedFeatureSource(
            [ArrayFeatureSource(a), ArrayFeatureSource(b)]
        )
    if kind.startswith("learned:"):
        strategy = kind.split(":", 1)[1]
        a, b = _pooled(4, d=DV + 2), _pooled(5, d=DV - 1)
        kwargs = {"logits": [0.4, -0.4]} if strategy == "sigmoid_gated" else {}
        dense = RaggedSources(
            np.concatenate([a, b], axis=1),
            source_dims=[DV + 2, DV - 1],
            strategy=strategy,
            aligned_dim=DV,
            fusion_kwargs=kwargs,
        )
        lazy = ConcatFeatureSource(
            [ArrayFeatureSource(a), ArrayFeatureSource(b)],
            strategy=strategy,
            aligned_dim=DV,
            fusion_kwargs=kwargs,
        )
        return dense, lazy
    if kind == "components":
        arr = np.random.default_rng(6).standard_normal((N_ITEMS, M_COMP, 4)).astype(np.float16)
        path = tmp_path / "resnet50_comp.npy"
        np.save(path, arr)
        return arr, NpyFeatureSource(path)
    raise ValueError(kind)


def _build(model_cls, features, **extra):
    torch.manual_seed(0)
    if model_cls is VBPR:
        return VBPR(N_USERS, N_ITEMS, features, {"latent_dim": 4, "visual_dim": 3, "l2_reg": 1e-3})
    if model_cls is AVBPR:
        cfg = {"latent_dim": 4, "visual_dim": 3, "att_hidden": 5, "l2_reg": 1e-3}
        return AVBPR(N_USERS, N_ITEMS, features, cfg)
    if model_cls is DeepStyle:
        cfg = {"latent_dim": 4, "l2_reg": 1e-3}
        return DeepStyle(N_USERS, N_ITEMS, features, cfg, item_categories=CATEGORIES)
    if model_cls is VNPR:
        return VNPR(N_USERS, N_ITEMS, features, {"latent_dim": 4, "l2_reg": 1e-3})
    if model_cls is ACF:
        cfg = {
            "latent_dim": 4,
            "visual_dim": 3,
            "att_hidden": 5,
            "max_history": 4,
            "l2_reg": 1e-3,
            "history_seed": 0,
        }
        return ACF(N_USERS, N_ITEMS, features, cfg, train_interactions=HISTORY)
    raise ValueError(model_cls)


def _pair(model_cls, kind: str, tmp_path: Path, *, block: int = 5):
    """Dense and lazy twins sharing one state; lazy forced to block at *block*."""
    dense_feats, lazy_feats = _variant(kind, tmp_path)
    dense = _build(model_cls, dense_feats)
    lazy = _build(model_cls, lazy_feats)
    lazy.load_state_dict(dense.state_dict())
    lazy._LAZY_ITEM_BLOCK = block
    return dense, lazy


CASES = [
    (VBPR, "pooled"),
    (VBPR, "stacked"),
    (VBPR, "learned:mean"),
    (AVBPR, "pooled"),
    (AVBPR, "learned:sigmoid_gated"),
    (DeepStyle, "pooled"),
    (DeepStyle, "stacked"),
    (VNPR, "pooled"),
    (VNPR, "stacked"),
    (VNPR, "learned:adaptive_gated"),
    (ACF, "components"),
]
IDS = [f"{cls.__name__}-{kind}" for cls, kind in CASES]


def _close(a: torch.Tensor, b: torch.Tensor, what: str) -> None:
    torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL, msg=lambda m: f"{what}: {m}")


# ---------------------------------------------------------------- equivalence
@pytest.mark.parametrize(("model_cls", "kind"), CASES, ids=IDS)
def test_eval_scores_match_dense(model_cls, kind, tmp_path: Path) -> None:
    dense, lazy = _pair(model_cls, kind, tmp_path)
    dense.eval()
    lazy.eval()
    all_items = torch.arange(N_ITEMS)

    with torch.no_grad():
        for u in (0, 3, 6):
            _close(lazy.predict(u, all_items), dense.predict(u, all_items), f"predict({u})")
        _close(
            lazy.predict_batch(USERS, all_items),
            dense.predict_batch(USERS, all_items),
            "predict_batch(all)",
        )
        _close(
            lazy.predict_batch(USERS, SUBSET),
            dense.predict_batch(USERS, SUBSET),
            "predict_batch(subset)",
        )


@pytest.mark.parametrize(("model_cls", "kind"), CASES, ids=IDS)
def test_training_scores_loss_and_gradients_match_dense(model_cls, kind, tmp_path: Path) -> None:
    dense, lazy = _pair(model_cls, kind, tmp_path)
    dense.train()
    lazy.train()

    pos_d, neg_d = dense(USERS, POS, NEG)
    pos_l, neg_l = lazy(USERS, POS, NEG)
    loss_d = dense.bpr_loss(pos_d, neg_d)
    loss_l = lazy.bpr_loss(pos_l, neg_l)
    loss_d.backward()
    loss_l.backward()

    _close(pos_l, pos_d, "score_pos")
    _close(neg_l, neg_d, "score_neg")
    _close(loss_l, loss_d, "loss")
    grads_d = {n: p.grad for n, p in dense.named_parameters()}
    grads_l = {n: p.grad for n, p in lazy.named_parameters()}
    assert grads_d.keys() == grads_l.keys()
    for name, grad in grads_d.items():
        if grad is None:
            assert grads_l[name] is None, name
            continue
        _close(grads_l[name], grad, f"grad {name}")


@pytest.mark.parametrize(
    ("model_cls", "kind"),
    [(VBPR, "learned:mean"), (VNPR, "learned:adaptive_gated"), (VNPR, "stacked")],
    ids=["VBPR-learned", "VNPR-learned-gated", "VNPR-stacked-gated"],
)
def test_online_fusion_stays_differentiable_under_lazy_features(model_cls, kind, tmp_path) -> None:
    """No ``detach``: the fusion parameters receive the same non-zero gradient."""
    dense, lazy = _pair(model_cls, kind, tmp_path)
    dense.train()
    lazy.train()

    dense.bpr_loss(*dense(USERS, POS, NEG)).backward()
    lazy.bpr_loss(*lazy(USERS, POS, NEG)).backward()

    fusion_grads = [
        (n, p.grad) for n, p in dense.named_parameters() if n.startswith("_online_fusion")
    ]
    assert fusion_grads, "the fusion module must own trainable parameters"
    lazy_params = dict(lazy.named_parameters())
    assert any(g is not None and g.abs().sum() > 0 for _, g in fusion_grads)
    for name, grad in fusion_grads:
        _close(lazy_params[name].grad, grad, f"fusion grad {name}")


# ------------------------------------------------------------- residency
@pytest.mark.parametrize(("model_cls", "kind"), CASES, ids=IDS)
def test_lazy_model_registers_no_raw_feature_buffer(model_cls, kind, tmp_path: Path) -> None:
    dense, lazy = _pair(model_cls, kind, tmp_path)

    lazy.to("cpu")  # must not need, nor move, a catalogue-sized buffer

    assert "visual_features" in dict(dense.named_buffers())
    assert "visual_features" not in dict(lazy.named_buffers())
    assert lazy.visual_features is None and lazy.is_lazy_visual
    assert lazy.visual_shape == tuple(dense.visual_features.shape)
    assert lazy.visual_dim_raw == dense.visual_dim_raw


def test_buffer_bytes_do_not_grow_with_the_catalogue(tmp_path: Path) -> None:
    def _buffer_bytes(n_items: int) -> int:
        src = ArrayFeatureSource(_pooled(9, n=n_items))
        model = VBPR(N_USERS, n_items, src, {"latent_dim": 4, "visual_dim": 3})
        return sum(b.numel() * b.element_size() for b in model.buffers())

    assert _buffer_bytes(N_ITEMS) == _buffer_bytes(4 * N_ITEMS)


@pytest.mark.parametrize(
    ("model_cls", "kind"), [(VBPR, "pooled"), (VNPR, "learned:mean"), (DeepStyle, "stacked")]
)
def test_forward_reads_only_the_batch_items(model_cls, kind, tmp_path: Path) -> None:
    _, lazy_feats = _variant(kind, tmp_path)
    spy = _Recording(lazy_feats)
    model = _build(model_cls, spy)
    model.train()

    model(USERS, POS, NEG)

    requested = np.concatenate(spy.requests)
    expected = set(POS.tolist()) | set(NEG.tolist())
    assert set(requested.tolist()) == expected
    assert max(len(r) for r in spy.requests) <= len(expected)


def test_acf_forward_reads_history_items_once_per_batch(tmp_path: Path) -> None:
    _, lazy_feats = _variant("components", tmp_path)
    spy = _Recording(lazy_feats)
    model = _build(ACF, spy)
    model.train()

    model(USERS, POS, NEG)

    hist = model.history_items[USERS][model.history_mask[USERS]].tolist()
    padding = {0} if not bool(model.history_mask[USERS].all()) else set()
    assert len(spy.requests) == 1  # unique ids, one read
    assert set(spy.requests[0].tolist()) == set(hist) | padding
    assert len(spy.requests[0]) == len(set(spy.requests[0].tolist()))


# --------------------------------------------------------------------- ACF/M03
def test_acf_repeated_history_rows_gather_exactly(tmp_path: Path) -> None:
    dense, lazy = _pair(ACF, "components", tmp_path, block=2)
    index = torch.tensor([[9, 9, 0, 22], [0, 0, 0, 0], [22, 9, 5, 5]])

    rows_lazy = lazy._raw_visual_rows(index)
    rows_dense = dense._raw_visual_rows(index)

    assert rows_lazy.dtype == torch.float16 == rows_dense.dtype
    assert rows_lazy.shape == (3, 4, M_COMP, 4)
    assert torch.equal(rows_lazy, rows_dense)


def test_acf_gradients_match_with_duplicate_users_and_history_positives(tmp_path) -> None:
    dense, lazy = _pair(ACF, "components", tmp_path, block=2)
    users = torch.tensor([0, 0, 3, 3, 6])
    pos = torch.tensor([2, 9, 9, 13, 22])  # every positive sits in its user's history
    neg = torch.tensor([9, 2, 13, 9, 0])
    dense.train()
    lazy.train()

    dense.bpr_loss(*dense(users, pos, neg)).backward()
    lazy.bpr_loss(*lazy(users, pos, neg)).backward()

    for (name, p_d), (_, p_l) in zip(
        dense.named_parameters(), lazy.named_parameters(), strict=True
    ):
        if p_d.grad is None:
            assert p_l.grad is None, name
            continue
        _close(p_l.grad, p_d.grad, f"grad {name}")
    assert dense.comp_projection.weight.grad.abs().sum() > 0


def test_acf_lazy_mode_holds_no_catalogue_projection_by_default(tmp_path: Path) -> None:
    dense, lazy = _pair(ACF, "components", tmp_path, block=4)
    dense.eval()
    lazy.eval()
    items = torch.arange(N_ITEMS)

    with torch.no_grad():
        ref = dense.predict_batch(USERS, items)
        out = lazy.predict_batch(USERS, items)

    assert dense._comp_cache is not None  # dense keeps the historical cache
    assert lazy._comp_cache is None  # constrained: projected per history block
    _close(out, ref, "predict_batch without cache")


def test_acf_lazy_cache_is_optional_and_bounded_by_bytes(tmp_path: Path) -> None:
    dense, lazy = _pair(ACF, "components", tmp_path, block=4)
    dense.eval()
    lazy.eval()
    items = torch.arange(N_ITEMS)
    needed = lazy._catalogue_projection_bytes()

    with torch.no_grad():
        lazy.LAZY_DERIVED_CACHE_MAX_BYTES = needed - 1
        lazy.predict_batch(USERS, items)
        assert lazy._comp_cache is None
        lazy.LAZY_DERIVED_CACHE_MAX_BYTES = needed
        out = lazy.predict_batch(USERS, items)
        ref = dense.predict_batch(USERS, items)

    assert lazy._comp_cache is not None and lazy._comp_cache.shape == (N_ITEMS, M_COMP, 3)
    _close(out, ref, "predict_batch with admitted cache")


def test_acf_dense_cache_over_its_limit_falls_back_to_on_demand(tmp_path: Path) -> None:
    dense, _ = _pair(ACF, "components", tmp_path)
    dense.eval()
    items = torch.arange(N_ITEMS)
    with torch.no_grad():
        ref = dense.predict_batch(USERS, items)
        dense.train()
        dense.eval()
        dense.DERIVED_CACHE_MAX_BYTES = 0
        out = dense.predict_batch(USERS, items)

    assert dense._comp_cache is None
    _close(out, ref, "over-limit dense")


def test_acf_projection_cache_is_invalidated_by_a_parameter_update_without_train(
    tmp_path: Path,
) -> None:
    """An optimiser step bumps the weight's version counter: no stale ``W_c f``."""
    dense, _ = _pair(ACF, "components", tmp_path)
    dense.eval()
    items = torch.arange(N_ITEMS)

    with torch.no_grad():
        before = dense.predict_batch(USERS, items)
        stale = dense._comp_cache
        dense.comp_projection.weight.add_(0.5)  # in-place, as optimizer.step() does
        after = dense.predict_batch(USERS, items)
        reference = _build(ACF, _variant("components", tmp_path)[0])
        reference.load_state_dict(dense.state_dict())
        reference.eval()
        fresh = reference.predict_batch(USERS, items)

    assert stale is not None and dense._comp_cache is not stale
    assert not torch.allclose(before, after)
    _close(after, fresh, "post-update predict_batch")


# ------------------------------------------------------------------- VNPR/M03
def test_vnpr_dense_catalogue_cache_is_a_zero_copy_alias(tmp_path: Path) -> None:
    dense, lazy = _pair(VNPR, "pooled", tmp_path, block=4)
    dense.eval()
    lazy.eval()
    items = torch.arange(N_ITEMS)

    with torch.no_grad():
        ref = dense.predict_batch(USERS, items)
        out = lazy.predict_batch(USERS, items)

    assert dense._catalogue_visual_cache is dense.visual_features
    assert lazy._catalogue_visual_cache is None
    _close(out, ref, "blocked VNPR predict_batch")
    dense.train()
    assert dense._catalogue_visual_cache is None


def test_vnpr_blocked_predict_batch_covers_a_short_final_block(tmp_path: Path) -> None:
    dense, lazy = _pair(VNPR, "pooled", tmp_path, block=N_ITEMS - 1)
    dense.eval()
    lazy.eval()
    items = torch.arange(N_ITEMS)
    subset = torch.tensor(
        [5, 5, 1, 0, 2, 7, 9, 22, 21, 20, 3, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1]
    )

    with torch.no_grad():
        _close(lazy.predict_batch(USERS, items), dense.predict_batch(USERS, items), "all")
        _close(lazy.predict_batch(USERS, subset), dense.predict_batch(USERS, subset), "subset")
