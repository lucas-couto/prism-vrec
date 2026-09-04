"""Fold-in guarantees (``src.folds.foldin``) over every built-in recommender.

Each test builds a small model, folds in a subset of users from a
profile, and checks the invariants the K-fold protocol relies on:
frozen parameters stay bit-identical, only the profile users' rows in
the declared user tables move, the target set never reaches the
sampler, the model's flags are restored, and the run is deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.folds import foldin as foldin_mod
from src.folds.foldin import FoldInConfig, FoldInReport, fold_in_users
from src.fusions.online import RaggedSources
from src.recommenders import BPR, get_recommender_spec, registered_recommender_names
from src.recommenders.base import BaseRecommender
from src.utils import training as training_mod
from src.utils.amp_compat import get_grad_scaler
from src.utils.training import BPRBatchSampler, bpr_step

N_USERS = 7
N_ITEMS = 11
N_COMPONENTS = 3
RAW_DIM = 5
L2_REG = 1e-2
BUILTIN = ("bpr", "vbpr", "avbpr", "deepstyle", "vnpr", "acf")
#: ``<model>+<strategy>``: the visual models driven by a
#: :class:`LearnedAlignmentFusion` (learned per-source projections;
#: ``adaptive_gated`` adds a gate MLP, ``sigmoid_gated`` a fixed-logit buffer).
FUSED = tuple(
    f"{name}+{strategy}"
    for name in ("vbpr", "avbpr", "deepstyle", "vnpr")
    for strategy in ("adaptive_gated", "sigmoid_gated")
)
ALL = BUILTIN + FUSED

TRAIN = {0: {0, 1}, 1: {2, 3, 4}, 2: {5}, 3: {6, 7}}
PROFILE = {4: {1, 2, 8}, 5: {3, 9, 10}}
TARGET = {4: {0}, 5: {5}}
OTHER_USERS = [u for u in range(N_USERS) if u not in PROFILE]
CONFIG = FoldInConfig(epochs=3, learning_rate=0.05, batch_size=4, seed=11)


def _config() -> dict:
    return {
        "latent_dim": 4,
        "visual_dim": 3,
        "att_hidden": 6,
        "max_history": 4,
        "l2_reg": L2_REG,
        "dropout": 0.0,
        "history_seed": 3,
    }


def _visual(spec) -> np.ndarray | None:
    if not spec.requires_visual:
        return None
    rng = np.random.default_rng(0)
    shape = (N_ITEMS, N_COMPONENTS, RAW_DIM) if spec.requires_components else (N_ITEMS, RAW_DIM)
    return rng.standard_normal(shape).astype("float32")


def _ragged(strategy: str) -> RaggedSources:
    """Two native sources of differing dims, concatenated, for a learned fusion."""
    rng = np.random.default_rng(0)
    dims = [RAW_DIM, RAW_DIM - 1]
    arr = rng.standard_normal((N_ITEMS, sum(dims))).astype("float32")
    kwargs = {"logits": [0.3, -0.3]} if strategy == "sigmoid_gated" else None
    return RaggedSources(
        arr, source_dims=dims, strategy=strategy, aligned_dim=RAW_DIM, fusion_kwargs=kwargs
    )


def _build(name: str, seed: int = 0) -> BaseRecommender:
    """Small model of ``name`` — constructor kwargs from the spec's flags, not the name.

    ``"<model>+<strategy>"`` builds ``model`` over a learned-alignment
    fusion with that strategy (see :data:`FUSED`).
    """
    name, _, strategy = name.partition("+")
    spec = get_recommender_spec(name)
    kwargs: dict = {}
    if getattr(spec.cls, "wants_history", False):
        kwargs["train_interactions"] = {u: set(s) for u, s in TRAIN.items()}
    if getattr(spec.cls, "wants_categories", False):
        kwargs["item_categories"] = None
    visual = _ragged(strategy) if strategy else _visual(spec)
    torch.manual_seed(seed)
    return spec.cls(N_USERS, N_ITEMS, visual_embeddings=visual, config=_config(), **kwargs)


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _user_table_keys(model: BaseRecommender) -> set[str]:
    return {f"{name}.weight" for name in model._USER_TABLES}


# ------------------------------------------------------------ coverage guard
@pytest.mark.parametrize("name", BUILTIN)
def test_every_user_indexed_embedding_is_declared_in_user_tables(name: str) -> None:
    model = _build(name)

    undeclared = [
        attr
        for attr, module in model.named_modules()
        if isinstance(module, nn.Embedding)
        and module.num_embeddings == N_USERS
        and attr not in model._USER_TABLES
    ]

    assert undeclared == [], f"{name}: user-indexed tables missing from _USER_TABLES"
    assert all(hasattr(model, t) for t in model._USER_TABLES)


def test_builtin_list_matches_registry() -> None:
    assert set(BUILTIN) <= set(registered_recommender_names())


# ------------------------------------------------------- parameter isolation
@pytest.mark.parametrize("name", ALL)
def test_fold_in_leaves_every_non_user_parameter_bit_identical(name: str) -> None:
    model = _build(name)
    before = _snapshot(model)

    fold_in_users(model, PROFILE, CONFIG, n_items=N_ITEMS, device="cpu")

    after = _snapshot(model)
    user_keys = _user_table_keys(model)
    for key, tensor in before.items():
        if key not in user_keys:
            assert torch.equal(tensor, after[key]), f"{name}: {key} changed"


@pytest.mark.parametrize("name", ALL)
def test_fold_in_leaves_rows_of_users_outside_the_profile_untouched(name: str) -> None:
    model = _build(name)
    before = _snapshot(model)

    fold_in_users(model, PROFILE, CONFIG, n_items=N_ITEMS, device="cpu")

    after = _snapshot(model)
    for key in _user_table_keys(model):
        assert torch.equal(before[key][OTHER_USERS], after[key][OTHER_USERS]), f"{name}: {key}"


# ------------------------------------------------- row mask: negative proof
def _disable_row_mask(monkeypatch) -> None:
    """Replace the row-mask hook by an identity hook (same handle protocol)."""
    monkeypatch.setattr(
        foldin_mod,
        "_row_mask_hook",
        lambda table, rows: table.weight.register_hook(lambda grad: grad),
    )


def _whole_matrix_vnpr(monkeypatch) -> BaseRecommender:
    """A VNPR regularised like its paper (whole matrices, nothing gathered).

    Production VNPR moved to BPR-Opt on 2026-09-04 (whole-matrix L2 under
    Adam trained ``W_u``/``W_i``/``W_i'`` to exactly zero); the paper
    reading is re-created here only to prove the row-mask guarantee
    against the one regulariser that CAN leak into rows outside the
    profile.
    """
    from src.recommenders.vnpr import VNPR

    monkeypatch.setattr(VNPR, "_L2_USER_TABLES", ())
    monkeypatch.setattr(VNPR, "_L2_ITEM_TABLES", ())
    return _build("vnpr")


def test_whole_matrix_l2_puts_a_dense_gradient_on_every_user_row(monkeypatch) -> None:
    """Numerical justification of ``L2_REG``.

    Under a whole-matrix regulariser the shared term is
    ``λ(‖W_u‖² + ‖W_v‖²)`` and its gradient is ``2λW`` on every row —
    profile or not.  With Xavier rows of magnitude ~0.1–0.7 and
    ``λ = 1e-2`` that gradient is ~1e-3 to ~1e-2: orders of magnitude
    above float32 resolution, and Adam's first step is ``≈ lr`` for ANY
    non-zero gradient, so the mask is the only thing keeping those rows
    still.
    """
    model = _whole_matrix_vnpr(monkeypatch)
    for param in model.parameters():
        param.requires_grad_(False)
    weights = {t: getattr(model, t).weight.requires_grad_(True) for t in model._USER_TABLES}

    model.l2_reg().backward()

    for name, weight in weights.items():
        grad = weight.grad[OTHER_USERS]
        assert torch.allclose(grad, 2 * L2_REG * weight.detach()[OTHER_USERS]), name
        assert grad.abs().max().item() > 1e-3, name
        assert bool((grad.abs().sum(dim=1) > 0).all()), f"{name}: a row got no gradient"


@pytest.mark.parametrize("name", (*ALL, "vnpr:whole_matrix"))
def test_without_the_row_mask_only_a_whole_matrix_l2_leaks_outside_the_profile(
    name: str, monkeypatch
) -> None:
    """Negative proof: the strong test above DOES detect the leak.

    Hook removed, a whole-matrix regulariser (the paper reading of VNPR,
    re-created via :func:`_whole_matrix_vnpr`) moves the rows outside the
    profile; every production model — VNPR included since it moved to
    BPR-Opt — keeps them bit-identical because its gradient on the user
    tables is sparse (gathered rows only).
    """
    whole_matrix = name == "vnpr:whole_matrix"
    model = _whole_matrix_vnpr(monkeypatch) if whole_matrix else _build(name)
    before = _snapshot(model)
    _disable_row_mask(monkeypatch)

    fold_in_users(model, PROFILE, CONFIG, n_items=N_ITEMS, device="cpu")

    after = _snapshot(model)
    drift = {
        key: (before[key][OTHER_USERS] - after[key][OTHER_USERS]).abs().max().item()
        for key in _user_table_keys(model)
    }
    if whole_matrix:
        assert all(value > 1e-2 for value in drift.values()), drift
    else:
        assert all(value == 0.0 for value in drift.values()), drift


# --------------------------------------------- trainable set during the loop
@pytest.mark.parametrize("name", ALL)
def test_only_user_table_weights_are_trainable_and_optimised(name: str, monkeypatch) -> None:
    """Spy on the optimiser: ``requires_grad`` set == Adam params == user tables."""
    model = _build(name)
    expected_ids = {id(getattr(model, t).weight) for t in model._USER_TABLES}
    seen: dict = {}
    original_adam = torch.optim.Adam

    def spy(params, **kwargs):
        params = list(params)
        seen["trainable"] = {n for n, p in model.named_parameters() if p.requires_grad}
        optimizer = original_adam(params, **kwargs)
        seen["groups"] = [[id(p) for p in g["params"]] for g in optimizer.param_groups]
        return optimizer

    monkeypatch.setattr(torch.optim, "Adam", spy)

    fold_in_users(model, PROFILE, CONFIG, n_items=N_ITEMS, device="cpu")

    assert seen["trainable"] == _user_table_keys(model)
    assert len(seen["groups"]) == 1
    assert set(seen["groups"][0]) == expected_ids
    assert len(seen["groups"][0]) == len(expected_ids)


@pytest.mark.parametrize("name", FUSED)
def test_online_fusion_is_registered_and_fully_frozen_during_fold_in(name: str) -> None:
    """The fusion module's params AND buffers are reached by the freeze."""
    model = _build(name)
    fusion_params = {n for n, _ in model.named_parameters() if n.startswith("_online_fusion.")}
    fusion_buffers = {n for n, _ in model.named_buffers() if n.startswith("_online_fusion.")}
    assert fusion_params, "fusion projections must be registered parameters"
    if name.endswith("+sigmoid_gated"):
        assert "_online_fusion.fixed_logits" in fusion_buffers
    before = {n: t.detach().clone() for n, t in model.named_parameters()}
    before.update({n: t.detach().clone() for n, t in model.named_buffers()})

    fold_in_users(model, PROFILE, CONFIG, n_items=N_ITEMS, device="cpu")

    for key in fusion_params | fusion_buffers:
        tensor = dict(model.named_parameters()).get(key, dict(model.named_buffers()).get(key))
        assert torch.equal(before[key], tensor), f"{name}: {key} changed"


@pytest.mark.parametrize("name", BUILTIN)
def test_fold_in_changes_rows_of_profile_users(name: str) -> None:
    model = _build(name)
    before = _snapshot(model)
    rows = sorted(PROFILE)

    report = fold_in_users(model, PROFILE, CONFIG, n_items=N_ITEMS, device="cpu")

    after = _snapshot(model)
    assert report.user_tables == model._USER_TABLES
    for key in _user_table_keys(model):
        assert not torch.equal(before[key][rows], after[key][rows]), f"{name}: {key}"


# ----------------------------------------------------------- target isolation
@pytest.mark.parametrize("name", ALL)
def test_target_items_never_reach_the_sampler(name: str, monkeypatch) -> None:
    model = _build(name)
    seen: list[dict] = []
    original_init = BPRBatchSampler.__init__

    def spy(self, train_interactions, *args, **kwargs):
        seen.append({u: set(s) for u, s in train_interactions.items()})
        original_init(self, train_interactions, *args, **kwargs)

    monkeypatch.setattr(BPRBatchSampler, "__init__", spy)

    fold_in_users(model, PROFILE, CONFIG, n_items=N_ITEMS, device="cpu")

    assert len(seen) == 1
    assert seen[0] == PROFILE
    for user, targets in TARGET.items():
        assert seen[0][user].isdisjoint(targets), f"{name}: target leaked for user {user}"


def test_target_item_can_be_drawn_as_negative_expected_behaviour() -> None:
    """The target is outside the profile, so it is a legitimate negative.

    With a profile covering every item but the target, the sampler has
    no other choice: every negative for that user IS the target.  This
    mirrors the training protocol, where a held-out item may be sampled
    as a negative — expected, not a leak.
    """
    user, target = 4, 0
    profile = {user: set(range(N_ITEMS)) - {target}}
    sampler = BPRBatchSampler(profile, N_ITEMS, batch_size=64, seed=CONFIG.seed)

    negatives = torch.cat([neg for _, _, neg in sampler.epoch(0)])

    assert negatives.numel() == N_ITEMS - 1
    assert bool((negatives == target).all())


# ------------------------------------------------------------ flag restoration
@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("training_mode", [True, False])
def test_fold_in_restores_requires_grad_and_train_mode(name: str, training_mode: bool) -> None:
    model = _build(name)
    model.item_embedding.weight.requires_grad_(False)
    model.train(training_mode)
    expected = {n: p.requires_grad for n, p in model.named_parameters()}

    fold_in_users(model, PROFILE, CONFIG, n_items=N_ITEMS, device="cpu")

    assert model.training is training_mode
    assert {n: p.requires_grad for n, p in model.named_parameters()} == expected


# ---------------------------------------------------------------- determinism
@pytest.mark.parametrize("name", BUILTIN)
def test_fold_in_is_deterministic_given_the_seed(name: str) -> None:
    first, second = _build(name, seed=5), _build(name, seed=5)

    report_a = fold_in_users(first, PROFILE, CONFIG, n_items=N_ITEMS, device="cpu")
    report_b = fold_in_users(second, PROFILE, CONFIG, n_items=N_ITEMS, device="cpu")

    assert report_a == report_b
    for key, tensor in _snapshot(first).items():
        assert torch.equal(tensor, second.state_dict()[key]), f"{name}: {key}"


def test_different_seed_reinitialises_rows_differently() -> None:
    first, second = _build("bpr", seed=5), _build("bpr", seed=5)
    other = FoldInConfig(epochs=1, learning_rate=0.05, batch_size=4, seed=CONFIG.seed + 1)

    fold_in_users(first, PROFILE, CONFIG, n_items=N_ITEMS, device="cpu")
    fold_in_users(second, PROFILE, other, n_items=N_ITEMS, device="cpu")

    rows = sorted(PROFILE)
    assert not torch.equal(first.user_embedding.weight[rows], second.user_embedding.weight[rows])


# ------------------------------------------------------------------- report
def test_report_counts_users_interactions_and_epochs() -> None:
    model = _build("bpr")

    report = fold_in_users(model, PROFILE, CONFIG, n_items=N_ITEMS, device="cpu")

    assert report == FoldInReport(
        n_users=2,
        n_interactions=6,
        epochs=CONFIG.epochs,
        final_loss=report.final_loss,
        user_tables=("user_embedding",),
    )
    assert np.isfinite(report.final_loss)


def test_fold_in_lowers_the_profile_loss() -> None:
    model = _build("bpr")
    short = FoldInConfig(epochs=1, learning_rate=0.05, batch_size=4, seed=CONFIG.seed)
    long = FoldInConfig(epochs=40, learning_rate=0.05, batch_size=4, seed=CONFIG.seed)

    loss_short = fold_in_users(_build("bpr"), PROFILE, short, n_items=N_ITEMS, device="cpu")
    loss_long = fold_in_users(model, PROFILE, long, n_items=N_ITEMS, device="cpu")

    assert loss_long.final_loss < loss_short.final_loss


@pytest.mark.parametrize(
    "profile",
    [{}, {4: set()}, {N_USERS: {1}}, {4: {N_ITEMS}}],
    ids=["empty", "empty-set", "user-out-of-range", "item-out-of-range"],
)
def test_fold_in_rejects_invalid_profiles(profile: dict) -> None:
    with pytest.raises(ValueError):
        fold_in_users(_build("bpr"), profile, CONFIG, n_items=N_ITEMS, device="cpu")


@pytest.mark.parametrize("field", ["epochs", "batch_size", "learning_rate"])
def test_config_rejects_non_positive_budget(field: str) -> None:
    kwargs = {"epochs": 1, "learning_rate": 0.1, "batch_size": 1, "seed": 0, field: 0}
    with pytest.raises(ValueError):
        FoldInConfig(**kwargs)


# -------------------------------------------------------- rebuild_user_state
def test_base_rebuild_user_state_is_a_no_op() -> None:
    model = _build("bpr")
    before = _snapshot(model)

    assert model.rebuild_user_state(PROFILE) is None
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())


def test_acf_rebuild_user_state_rewrites_only_the_given_users_history_rows() -> None:
    model = _build("acf")
    items_before = model.history_items.clone()
    mask_before = model.history_mask.clone()
    long_profile = {4: set(range(N_ITEMS)), 5: {3, 9}, 1: set()}

    model.rebuild_user_state(long_profile)

    untouched = [u for u in range(N_USERS) if u not in long_profile]
    assert torch.equal(model.history_items[untouched], items_before[untouched])
    assert torch.equal(model.history_mask[untouched], mask_before[untouched])
    horizon = model.history_items.shape[1]
    expected_4 = model._select_history(4, sorted(range(N_ITEMS)), horizon)
    assert model.history_items[4].tolist() == expected_4
    assert bool(model.history_mask[4].all())
    assert model.history_items[5, :2].tolist() == [3, 9]
    assert model.history_mask[5].tolist() == [True, True, False, False]
    assert not bool(model.history_mask[1].any())


def test_acf_fold_in_leaves_history_of_other_users_untouched() -> None:
    model = _build("acf")
    items_before = model.history_items.clone()

    fold_in_users(model, PROFILE, CONFIG, n_items=N_ITEMS, device="cpu")

    assert torch.equal(model.history_items[OTHER_USERS], items_before[OTHER_USERS])
    assert set(model.history_items[4][model.history_mask[4]].tolist()) == PROFILE[4]


# ------------------------------------------------------------------ bpr_step
def test_bpr_step_returns_detached_loss_and_updates_parameters() -> None:
    torch.manual_seed(0)
    model = BPR(N_USERS, N_ITEMS, config={"latent_dim": 4, "l2_reg": 0.0})
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    before = model.user_embedding.weight.detach().clone()
    users, pos, neg = torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([3, 4])

    loss = bpr_step(
        model,
        optimizer,
        get_grad_scaler(enabled=False),
        users,
        pos,
        neg,
        device="cpu",
        use_cuda=False,
    )

    assert loss.requires_grad is False
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert not torch.equal(before, model.user_embedding.weight)
    assert training_mod.bpr_step is bpr_step
