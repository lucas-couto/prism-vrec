"""VBPR / AVBPR visual bias ``beta'^T f_i`` (He & McAuley 2016, Eq. 4).

Pins: (a) a zero ``beta'`` leaves scores bit-identical to the model
without the term; (b) a non-zero ``beta'`` shifts every scoring path by
exactly ``f_i @ beta'``; (c) the three scoring paths agree with and
without an online fusion; (d)-(e) the paper's three ``lambda`` keys
route the right parameters; (f) the bias cache only serves eval;
(g) ``state_dict`` gains exactly the ``visual_bias`` key.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.recommenders.avbpr import AVBPR
from src.recommenders.base import BaseRecommender
from src.recommenders.vbpr import VBPR

# N_ITEMS != len(POS) + len(NEG): the full-catalogue caches key on length.
N_USERS, N_ITEMS, D_V = 3, 8, 5
USERS = torch.tensor([0, 1, 2])
POS = torch.tensor([0, 2, 4])
NEG = torch.tensor([1, 3, 5])
ALL_ITEMS = torch.arange(N_ITEMS)


def _features(online: bool) -> np.ndarray:
    rng = np.random.default_rng(0)
    shape = (N_ITEMS, 2, D_V) if online else (N_ITEMS, D_V)
    return rng.standard_normal(shape).astype("float32")


def _build(model_cls: type[BaseRecommender], config: dict, *, online: bool = False):
    torch.manual_seed(0)
    base = {"latent_dim": 4, "visual_dim": 3, "att_hidden": 4}
    return model_cls(N_USERS, N_ITEMS, _features(online), {**base, **config})


def _set_visual_bias(model: BaseRecommender, value: float) -> None:
    with torch.no_grad():
        model.visual_bias.fill_(value)


def _manual_predict(model: BaseRecommender, user_id: int, item_ids: torch.Tensor) -> torch.Tensor:
    """Score without the visual bias, in the mixin's exact op order."""
    gamma_u = model.user_embedding.weight[user_id]
    alpha_u = model.visual_user_embedding.weight[user_id]
    gamma_i = model.item_embedding(item_ids)
    theta_i = model._item_visual_term(item_ids)
    beta_i = model.item_bias(item_ids).squeeze(-1)
    bias = torch.zeros(item_ids.shape[0]) + beta_i
    return (gamma_u * gamma_i).sum(-1) + (alpha_u * theta_i).sum(-1) + bias


MODELS = pytest.mark.parametrize("model_cls", [VBPR, AVBPR], ids=["vbpr", "avbpr"])


@MODELS
def test_zero_visual_bias_scores_bit_identical_to_model_without_term(model_cls) -> None:
    model = _build(model_cls, {"l2_reg": 0.0})
    model.eval()

    predicted = model.predict(1, ALL_ITEMS)
    manual = _manual_predict(model, 1, ALL_ITEMS)

    assert torch.all(model.visual_bias == 0)
    assert torch.equal(predicted, manual)


@MODELS
def test_visual_bias_is_zero_initialised(model_cls) -> None:
    model = _build(model_cls, {"l2_reg": 0.0})

    assert model.visual_bias.shape == (D_V,)
    assert torch.all(model.visual_bias == 0)


@MODELS
@pytest.mark.parametrize("online", [False, True], ids=["plain", "online_fusion"])
def test_nonzero_visual_bias_shifts_every_path_by_f_dot_beta(model_cls, online: bool) -> None:
    model = _build(model_cls, {"l2_reg": 0.0}, online=online)
    model.eval()
    with torch.no_grad():
        base_fwd = model(USERS, POS, NEG)
        base_pred = model.predict(1, ALL_ITEMS)
        base_batch = model.predict_batch(USERS, ALL_ITEMS)
    torch.manual_seed(1)
    with torch.no_grad():
        model.visual_bias.copy_(torch.randn(D_V))
        shift = model._resolve_visual(ALL_ITEMS) @ model.visual_bias
    model.train().eval()  # parameters changed: drop the eval caches

    with torch.no_grad():
        got_fwd = model(USERS, POS, NEG)
        got_pred = model.predict(1, ALL_ITEMS)
        got_batch = model.predict_batch(USERS, ALL_ITEMS)

    torch.testing.assert_close(got_fwd[0], base_fwd[0] + shift[POS])
    torch.testing.assert_close(got_fwd[1], base_fwd[1] + shift[NEG])
    torch.testing.assert_close(got_pred, base_pred + shift)
    torch.testing.assert_close(got_batch, base_batch + shift.unsqueeze(0))


@MODELS
@pytest.mark.parametrize("online", [False, True], ids=["plain", "online_fusion"])
@pytest.mark.parametrize("training", [True, False], ids=["train_mode", "eval_mode"])
def test_predict_predict_batch_and_forward_agree(model_cls, online: bool, training: bool) -> None:
    model = _build(model_cls, {"l2_reg": 0.0}, online=online)
    _set_visual_bias(model, 0.7)
    model.train(training)

    with torch.no_grad():
        score_pos, score_neg = model(USERS, POS, NEG)
        batch = model.predict_batch(USERS, ALL_ITEMS)
        single = torch.stack([model.predict(u, ALL_ITEMS) for u in range(N_USERS)])

    torch.testing.assert_close(single, batch)
    torch.testing.assert_close(score_pos, batch[USERS, POS])
    torch.testing.assert_close(score_neg, batch[USERS, NEG])


def _loss(model: BaseRecommender) -> float:
    return model.bpr_loss(*model(USERS, POS, NEG)).item()


@MODELS
def test_projection_is_unregularised_by_default_lambda_e_zero(model_cls) -> None:
    model = _build(model_cls, {"l2_reg": 0.1})
    with torch.no_grad():
        model.visual_user_embedding.weight.zero_()  # W no longer reaches the score
    baseline = _loss(model)

    with torch.no_grad():
        model.visual_projection.weight.mul_(10.0)
    perturbed = _loss(model)

    assert perturbed == baseline


@MODELS
def test_projection_is_regularised_when_l2_reg_projection_is_given(model_cls) -> None:
    model = _build(model_cls, {"l2_reg": 0.1, "l2_reg_projection": 0.1})
    with torch.no_grad():
        model.visual_user_embedding.weight.zero_()
    baseline = _loss(model)
    w_before = model.visual_projection.weight.pow(2).sum().item()

    with torch.no_grad():
        model.visual_projection.weight.mul_(10.0)
    perturbed = _loss(model)

    expected_delta = 0.1 * (100.0 - 1.0) * w_before
    assert math.isclose(perturbed - baseline, expected_delta, rel_tol=1e-4)


@MODELS
@pytest.mark.parametrize(
    ("config", "expected_lambda"),
    [({"l2_reg": 0.1, "l2_reg_visual_bias": 0.5}, 0.5), ({"l2_reg": 0.1}, 0.1)],
    ids=["dedicated_key", "falls_back_to_l2_reg"],
)
def test_visual_bias_regularised_under_its_key_or_l2_reg(
    model_cls, config: dict, expected_lambda: float
) -> None:
    model = _build(model_cls, config)
    with torch.no_grad():
        model.visual_features.zero_()  # beta' cannot reach the score
    baseline = _loss(model)

    _set_visual_bias(model, 2.0)
    perturbed = _loss(model)

    expected_delta = expected_lambda * D_V * 2.0**2
    assert math.isclose(perturbed - baseline, expected_delta, rel_tol=1e-5)


@MODELS
def test_visual_bias_cache_only_serves_eval_and_is_cleared_by_train(model_cls) -> None:
    model = _build(model_cls, {"l2_reg": 0.0})
    _set_visual_bias(model, 0.3)

    model.train()
    model.predict(0, ALL_ITEMS)
    train_cache = model._item_visual_bias_cache
    model.eval()
    model.predict(0, ALL_ITEMS)
    eval_cache = model._item_visual_bias_cache
    model.train()
    after_train = model._item_visual_bias_cache

    assert train_cache is None
    assert eval_cache is not None and eval_cache.shape == (N_ITEMS,)
    assert after_train is None


@MODELS
def test_visual_bias_cache_is_bypassed_with_online_fusion(model_cls) -> None:
    model = _build(model_cls, {"l2_reg": 0.0}, online=True)
    model.eval()

    model.predict(0, ALL_ITEMS)

    assert model._item_visual_bias_cache is None


def test_vbpr_state_dict_gains_exactly_the_visual_bias_key() -> None:
    model = _build(VBPR, {"l2_reg": 0.0})

    keys = set(model.state_dict())

    assert keys == {
        "user_embedding.weight",
        "item_embedding.weight",
        "item_bias.weight",
        "visual_user_embedding.weight",
        "visual_projection.weight",
        "visual_bias",
    }
    assert model.state_dict()["visual_bias"].shape == (D_V,)


def test_avbpr_state_dict_gains_exactly_the_visual_bias_key() -> None:
    model = _build(AVBPR, {"l2_reg": 0.0})

    keys = set(model.state_dict())
    non_attention = {k for k in keys if not k.startswith("attention_net.")}

    assert non_attention == {
        "user_embedding.weight",
        "item_embedding.weight",
        "item_bias.weight",
        "visual_user_embedding.weight",
        "visual_projection.weight",
        "visual_bias",
    }
    assert len(keys - non_attention) == 4  # two Linear layers x (weight, bias)
