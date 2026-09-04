"""VNPR fidelity to Niu, Caverlee & Lu (WSDM 2018), Sections 4.1-4.3 and 5.2.

Every assertion is written against the paper's equations computed by
hand: mirrored branches with two item tables, element-wise product
merge, one-neuron ReLU dense, inference as the average of the two
branches and dropout on the embeddings.  Regularisation is the ONE
declared divergence: BPR-Opt gathered rows instead of the paper's whole
embedding matrices — under Adam the whole-matrix penalty drives every
rarely-gathered row to exactly zero (measured 2026-09-04, see
``docs/protocol.md`` §"VNPR regularisation"), leaving a visual-only model.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.recommenders.vnpr import VNPR

N_USERS, N_ITEMS, K, DV = 30, 200, 6, 24


def _visual(seed: int = 0, shape: tuple[int, ...] = (N_ITEMS, DV)) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(shape).astype("float32")


def _make(config: dict | None = None, visual: np.ndarray | None = None) -> VNPR:
    torch.manual_seed(0)
    cfg = {"latent_dim": K, "l2_reg": 1e-4, **(config or {})}
    return VNPR(
        N_USERS, N_ITEMS, visual_embeddings=_visual() if visual is None else visual, config=cfg
    )


@pytest.fixture()
def model() -> VNPR:
    return _make().eval()


def _manual_branch(
    model: VNPR, users: torch.Tensor, items: torch.Tensor, neg: bool
) -> torch.Tensor:
    """ReLU(w^T [p ∘ q, v ∘ f] + b) with q from W_i (neg=False) or W_i' (neg=True)."""
    table = model.item_embedding_neg if neg else model.item_embedding
    p = model.user_embedding.weight[users]
    q = table.weight[items]
    v = model.visual_user_embedding.weight[users]
    f = model.visual_features[items]
    merged = torch.cat([p * q, v * f], dim=-1)
    w = model.dense.weight.squeeze(0)
    return torch.relu(merged @ w + model.dense.bias)


def test_forward_matches_paper_branches_with_positive_and_negative_item_tables(model: VNPR) -> None:
    users = torch.arange(8)
    pos = torch.arange(10, 18)
    neg = torch.arange(50, 58)

    with torch.no_grad():
        r_pos, r_neg = model(users, pos, neg)

    assert torch.allclose(r_pos, _manual_branch(model, users, pos, neg=False), atol=1e-6)
    assert torch.allclose(r_neg, _manual_branch(model, users, neg, neg=True), atol=1e-6)
    assert not torch.allclose(r_neg, _manual_branch(model, users, neg, neg=False))


def test_predict_is_the_mean_of_both_branches_fed_the_same_item(model: VNPR) -> None:
    items = torch.arange(N_ITEMS)
    users = torch.full((N_ITEMS,), 3, dtype=torch.long)

    with torch.no_grad():
        out = model.predict(3, items)

    expected = 0.5 * (
        _manual_branch(model, users, items, neg=False)
        + _manual_branch(model, users, items, neg=True)
    )
    assert torch.allclose(out, expected, atol=1e-6)


def test_predict_batch_matches_predict_and_preserves_ranking(model: VNPR) -> None:
    users = torch.arange(N_USERS)
    items = torch.arange(N_ITEMS)

    with torch.no_grad():
        ref = torch.stack([model.predict(int(u), items) for u in users])
        out = model.predict_batch(users, items)

    assert out.shape == (N_USERS, N_ITEMS)
    assert torch.allclose(ref, out, atol=1e-5, rtol=1e-5)
    r_ref = torch.sort(ref, dim=1, descending=True, stable=True).indices
    r_out = torch.sort(out, dim=1, descending=True, stable=True).indices
    for u, p in (r_ref != r_out).nonzero().tolist():
        gap = abs(ref[u, r_ref[u, p]] - ref[u, r_out[u, p]]).item()
        assert gap < 1e-6, f"ranking swap at a non-tie gap ({gap:.2e})"


def test_predict_batch_on_item_subset_matches_predict(model: VNPR) -> None:
    users = torch.arange(3)
    items = torch.tensor([0, 7, 42, 199])

    with torch.no_grad():
        ref = torch.stack([model.predict(int(u), items) for u in users])
        out = model.predict_batch(users, items)

    assert model._catalogue_visual_cache is None
    assert torch.allclose(ref, out, atol=1e-5, rtol=1e-5)


def test_has_no_mlp_hidden_layers_or_item_bias(model: VNPR) -> None:
    assert not hasattr(model, "mlp")
    assert not hasattr(model, "hidden_layers")
    assert not hasattr(model, "item_bias")
    assert not hasattr(model, "visual_transform")


def test_dense_layer_is_a_single_neuron_over_k_plus_dv(model: VNPR) -> None:
    assert model.dense.out_features == 1
    assert model.dense.in_features == K + DV


def test_l2_penalises_only_the_gathered_rows_and_not_the_dense_layer() -> None:
    """BPR-Opt reading: ``W_i`` rows of the positives, ``W_i'`` rows of the
    negatives, ``W_u`` / ``W_v`` rows of the users; nothing else."""
    model = _make({"l2_reg": 1e-2}).eval()
    users, pos, neg = torch.arange(4), torch.arange(4), torch.arange(4, 8)
    untouched_item = N_ITEMS - 1  # never in pos / neg
    untouched_user = N_USERS - 1

    def reg() -> float:
        model(users, pos, neg)  # records the batch l2_reg() consumes
        return model.l2_reg().item()

    with torch.no_grad():
        reg_0 = reg()
        model.item_embedding_neg.weight[untouched_item] *= 10.0
        model.item_embedding.weight[untouched_item] *= 10.0
        model.user_embedding.weight[untouched_user] *= 10.0
        model.visual_user_embedding.weight[untouched_user] *= 10.0
        model.dense.weight *= 10.0
        reg_untouched = reg()
        model.item_embedding.weight[pos[0]] *= 10.0
        reg_pos = reg()
        model.item_embedding_neg.weight[neg[0]] *= 10.0
        reg_neg = reg()
        model.item_embedding_neg.weight[pos[0]] *= 10.0  # W_i' row of a POSITIVE
        reg_cross = reg()

    assert reg_untouched == pytest.approx(reg_0)
    assert reg_pos > reg_untouched
    assert reg_neg > reg_pos
    assert reg_cross == pytest.approx(reg_neg), "W_i' is gathered by the negatives only"


def test_adam_training_keeps_rows_outside_the_batch_at_their_initial_norm() -> None:
    """Regression for the 2026-09-04 collapse: with the paper's whole-matrix
    L2, Adam moved every rarely-gathered row ~lr per step towards zero
    regardless of ``λ``, so ``W_u``/``W_i``/``W_i'`` trained to EXACTLY 0.
    Under BPR-Opt a row outside the batch receives no gradient at all."""
    model = _make({"l2_reg": 1e-4, "dropout": 0.0}).train()
    users, pos, neg = torch.arange(4), torch.arange(4), torch.arange(4, 8)
    untouched_user, untouched_item = N_USERS - 1, N_ITEMS - 1
    tables = (model.user_embedding, model.visual_user_embedding)
    item_tables = (model.item_embedding, model.item_embedding_neg)
    before_user = [t.weight[untouched_user].detach().clone() for t in tables]
    before_item = [t.weight[untouched_item].detach().clone() for t in item_tables]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(60):
        optimizer.zero_grad()
        loss = model.bpr_loss(*model(users, pos, neg)) + model.l2_reg()
        loss.backward()
        optimizer.step()

    for t, b in zip(tables, before_user, strict=True):
        assert torch.equal(t.weight[untouched_user].detach(), b)
    for t, b in zip(item_tables, before_item, strict=True):
        assert torch.equal(t.weight[untouched_item].detach(), b)
    assert model.user_embedding.weight[users].norm(dim=1).min().item() > 0.0


def test_dropout_perturbs_training_forward_and_is_inert_in_eval() -> None:
    model = _make({"dropout": 0.5})
    users, pos, neg = torch.arange(N_USERS), torch.arange(N_USERS), torch.arange(60, 90)

    model.train()
    with torch.no_grad():
        first, _ = model(users, pos, neg)
        second, _ = model(users, pos, neg)
    model.eval()
    with torch.no_grad():
        eval_first, _ = model(users, pos, neg)
        eval_second, _ = model(users, pos, neg)

    assert not torch.equal(first, second)
    assert torch.equal(eval_first, eval_second)
    assert torch.allclose(eval_first, _manual_branch(model, users, pos, neg=False), atol=1e-6)


def test_online_fusion_scores_all_paths_without_using_the_cache() -> None:
    model = _make(visual=_visual(1, (N_ITEMS, 2, DV))).eval()
    users, pos, neg = torch.arange(5), torch.arange(5), torch.arange(20, 25)
    items = torch.arange(N_ITEMS)

    with torch.no_grad():
        r_pos, r_neg = model(users, pos, neg)
        ref = torch.stack([model.predict(int(u), items) for u in users])
        out = model.predict_batch(users, items)

    assert model._online_fusion is not None
    assert r_pos.shape == r_neg.shape == (5,)
    assert torch.allclose(ref, out, atol=1e-5, rtol=1e-5)
    assert model._catalogue_visual_cache is None


def test_catalogue_cache_is_populated_in_eval_and_invalidated_by_train(model: VNPR) -> None:
    users, items = torch.arange(2), torch.arange(N_ITEMS)

    with torch.no_grad():
        model.predict_batch(users, items)
    assert model._catalogue_visual_cache is not None

    model.train()
    assert model._catalogue_visual_cache is None


def test_state_dict_holds_exactly_the_paper_parameters(model: VNPR) -> None:
    expected = {
        "user_embedding.weight",
        "item_embedding.weight",
        "item_embedding_neg.weight",
        "visual_user_embedding.weight",
        "dense.weight",
        "dense.bias",
    }

    assert set(model.state_dict()) == expected
    assert model.visual_user_embedding.embedding_dim == DV


def test_missing_visual_embeddings_are_rejected() -> None:
    with pytest.raises(RuntimeError):
        VNPR(N_USERS, N_ITEMS, visual_embeddings=None, config={"latent_dim": K})
