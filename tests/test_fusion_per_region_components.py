"""Per-region fusion of the ACF component artifacts.

The fusion family is applied region by region to the
``(n_items, R, D)`` component maps so that ACF answers the same
``fusion_within_model`` question as the pooled recommenders while
keeping its per-region attention (``docs/protocol.md``, "ACF per-region
fusion").  These tests pin the three properties that make the artifact
usable: the layout survives, one fitted basis is shared by every region,
and a PCA fit still sees training items only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.fusions.online import RaggedSources, load_embedding
from src.recommenders.acf import ACF
from src.steps.fuse import _collect_fusion_tasks, _fuse_component_rows, _fuse_components_enabled
from src.utils.artifact_names import is_component_artifact

N_ITEMS, REGIONS, D_A, D_B = 24, 4, 32, 16


def _sources(tmp_path: Path) -> list[str]:
    rng = np.random.default_rng(0)
    paths = []
    for name, dim in (("resnet50_comp.npy", D_A), ("vit_b16_comp.npy", D_B)):
        arr = rng.standard_normal((N_ITEMS, REGIONS, dim)).astype("float16")
        np.save(tmp_path / name, arr)
        paths.append(str(tmp_path / name))
    return paths


def test_fused_components_keep_the_item_and_region_layout(tmp_path: Path) -> None:
    paths = _sources(tmp_path)

    fused = _fuse_component_rows("concat", paths, normalize=True, train_items=None)

    assert fused.shape == (N_ITEMS, REGIONS, D_A + D_B)
    assert fused.dtype == np.float16, "the fp16 the component grid was sized on must survive"


def test_each_region_is_fused_independently_of_the_others(tmp_path: Path) -> None:
    """Region ``r`` of item ``i`` may depend on that region's sources only."""
    paths = _sources(tmp_path)
    baseline = _fuse_component_rows("concat", paths, normalize=True, train_items=None)

    perturbed = np.load(paths[0])
    perturbed[7, 1] += 5.0
    np.save(paths[0], perturbed)
    after = _fuse_component_rows("concat", paths, normalize=True, train_items=None)

    assert not np.allclose(after[7, 1], baseline[7, 1]), "the touched region must change"
    changed = ~np.isclose(after, baseline).all(axis=-1)
    assert changed.sum() == 1 and changed[7, 1], "no other item or region may move"


def test_one_pca_basis_is_shared_by_every_region(tmp_path: Path) -> None:
    """Regions must land in a common space, or ACF's component attention
    compares coordinates that mean different things region by region.

    Each region is given its own offset: under ONE shared basis those
    offsets survive the projection and the region centroids stay apart,
    while a per-region basis would centre every region on its own mean
    and collapse them onto each other.
    """
    rng = np.random.default_rng(0)
    paths = []
    offsets = np.arange(REGIONS, dtype="float32") * 10.0
    for name, dim in (("resnet50_comp.npy", D_A), ("vit_b16_comp.npy", D_B)):
        arr = rng.standard_normal((N_ITEMS, REGIONS, dim)).astype("float32")
        arr[:, :, 0] += offsets[None, :]
        np.save(tmp_path / name, arr.astype("float16"))
        paths.append(str(tmp_path / name))

    fused = _fuse_component_rows(
        "pca", paths, normalize=False, train_items=list(range(N_ITEMS)), n_components=8
    ).astype("float64")

    centroids = np.stack([fused[:, r, :].mean(axis=0) for r in range(REGIONS)])
    separation = np.linalg.norm(centroids[1:] - centroids[:-1], axis=1).min()
    within_region = max(
        float(np.linalg.norm(fused[:, r, :] - centroids[r], axis=1).mean()) for r in range(REGIONS)
    )
    assert separation > within_region, (
        "region centroids collapsed onto each other - the regions were projected by different bases"
    )


def test_pca_fit_never_sees_a_held_out_item(tmp_path: Path) -> None:
    """``train_items`` must be expanded to the rows those items own."""
    paths = _sources(tmp_path)
    train_items = list(range(N_ITEMS // 2))

    fitted = _fuse_component_rows(
        "pca", paths, normalize=True, train_items=train_items, n_components=8
    )
    poisoned = np.load(paths[0])
    poisoned[N_ITEMS - 1] += 50.0  # a held-out item only
    np.save(paths[0], poisoned)
    unchanged = _fuse_component_rows(
        "pca", paths, normalize=True, train_items=train_items, n_components=8
    )

    np.testing.assert_allclose(
        fitted[: N_ITEMS // 2].astype("float32"),
        unchanged[: N_ITEMS // 2].astype("float32"),
        atol=1e-2,
        err_msg="a held-out item moved the PCA basis — the fit set leaked",
    )


def test_component_outputs_stay_routable_to_component_models(tmp_path: Path) -> None:
    embeddings_dir = tmp_path / "embeddings"
    dataset_dir = embeddings_dir / "ds"
    dataset_dir.mkdir(parents=True)
    _sources(dataset_dir)
    processed = tmp_path / "processed" / "ds"
    processed.mkdir(parents=True)
    (processed / "train.csv").write_text("user_idx,item_idx\n0,1\n1,2\n", encoding="utf-8")

    tasks = _collect_fusion_tasks(
        "ds",
        str(embeddings_dir),
        str(tmp_path / "processed"),
        ["resnet50", "vit_b16"],
        {"concat": {}, "mean": {}},
        normalize=True,
        enabled_strategies={"concat", "mean"},
        alignment_method="learned",
        alignment_dim=12,
        component=True,
    )

    assert tasks, "the component pass produced no task"
    for task in tasks:
        stem = Path(task["output_path"]).stem
        assert is_component_artifact(stem), f"{stem} would be routed to pooled models"
        assert all(src.endswith("_comp.npy") for src in task["emb_list_paths"])


def test_the_component_pass_follows_the_recommender_roster() -> None:
    assert _fuse_components_enabled({"recommenders_enabled": ["acf"]}) is True
    assert _fuse_components_enabled({"recommenders_enabled": ["bpr", "vbpr"]}) is False


def test_acf_consumes_a_per_region_learned_fusion(tmp_path: Path) -> None:
    """End to end: ACF keeps R regions and projects the ALIGNED dim."""
    paths = _sources(tmp_path)
    (tmp_path / "sidecar.json").write_text(
        '{"strategy": "mean", "online": true, "alignment": "learned", "dim": 12,'
        ' "components": ["resnet50_comp.npy", "vit_b16_comp.npy"], "normalize": true}',
        encoding="utf-8",
    )
    source = load_embedding(tmp_path / "sidecar.json")
    assert isinstance(source, RaggedSources)
    assert source.shape == (N_ITEMS, REGIONS, D_A + D_B)
    assert list(source.source_dims) == [D_A, D_B]
    assert Path(paths[0]).exists()

    train = {u: {(u * 3) % N_ITEMS, (u * 5) % N_ITEMS} for u in range(8)}
    model = ACF(
        8,
        N_ITEMS,
        visual_embeddings=source,
        config={"latent_dim": 8, "att_hidden": 8, "max_history": 5, "l2_reg": 1e-4},
        train_interactions=train,
    )

    assert model.n_components == REGIONS, "per-region attention lost its regions"
    assert model.visual_dim_raw == 12, "W_c must consume the aligned dim, not the concat"
    assert model.comp_projection.in_features == 12

    users, pos, neg = torch.arange(4), torch.arange(4), torch.arange(4, 8)
    score_pos, score_neg = model(users, pos, neg)
    loss = model.bpr_loss(score_pos, score_neg)
    loss.backward()

    fusion_grads = [p.grad for p in model._online_fusion.parameters() if p.requires_grad]
    assert fusion_grads and any(g is not None and g.abs().sum() > 0 for g in fusion_grads), (
        "the per-region projections must be trained by the BPR loss"
    )


def test_acf_catalogue_cache_is_invalidated_by_a_fusion_step(tmp_path: Path) -> None:
    """The fusion sits before ``W_c``, so its updates must not be cached over."""
    _sources(tmp_path)
    (tmp_path / "sidecar.json").write_text(
        '{"strategy": "mean", "online": true, "alignment": "learned", "dim": 12,'
        ' "components": ["resnet50_comp.npy", "vit_b16_comp.npy"], "normalize": true}',
        encoding="utf-8",
    )
    train = {u: {(u * 3) % N_ITEMS} for u in range(8)}
    model = ACF(
        8,
        N_ITEMS,
        visual_embeddings=load_embedding(tmp_path / "sidecar.json"),
        config={"latent_dim": 8, "att_hidden": 8, "max_history": 5, "l2_reg": 1e-4},
        train_interactions=train,
    ).eval()

    before = model._catalogue_projection()
    assert before is not None
    # The aligned sources are L2-normalised, so a pure rescale would be
    # invisible; perturb the projection's direction instead.
    with torch.no_grad():
        weight = next(iter(model._online_fusion.projections)).weight
        weight += torch.randn_like(weight)
    after = model._catalogue_projection()

    assert after is not None
    assert not torch.allclose(before, after), "a stale projection cache survived a fusion update"


@pytest.mark.parametrize("strategy", ["concat", "pca"])
def test_component_fusion_rejects_pooled_two_dimensional_sources(
    tmp_path: Path, strategy: str
) -> None:
    for name, dim in (("a.npy", D_A), ("b.npy", D_B)):
        np.save(tmp_path / name, np.zeros((N_ITEMS, dim), dtype="float16"))

    with pytest.raises(ValueError, match="3-D"):
        _fuse_component_rows(
            strategy,
            [str(tmp_path / "a.npy"), str(tmp_path / "b.npy")],
            normalize=True,
            train_items=None,
        )


def test_the_component_flag_is_not_a_fusion_ingredient(tmp_path: Path) -> None:
    """A pooled fusion's provenance must be byte-identical to the record
    written before the component pass existed, or every ``hybrid_*`` file
    already on disk is refused as built from different ingredients."""
    from src.steps.fuse import task_provenance

    for name, dim in (("a.npy", D_A), ("b.npy", D_B)):
        np.save(tmp_path / name, np.zeros((N_ITEMS, dim), dtype="float32"))
    task = {
        "strategy_name": "concat",
        "output_path": str(tmp_path / "hybrid_concat.npy"),
        "emb_list_paths": [str(tmp_path / "a.npy"), str(tmp_path / "b.npy")],
        "normalize": True,
        "train_items": None,
    }

    without_flag = task_provenance(dict(task))
    with_flag = task_provenance({**task, "component": False})
    as_component = task_provenance({**task, "component": True})

    assert with_flag == without_flag
    assert as_component == without_flag, "the route must not enter the digest"


def test_the_lazy_loader_accepts_per_region_component_sources(tmp_path: Path) -> None:
    """The lazy path must carry the same recipe as the eager one.

    Found 2026-09-09 on the first real ACF run: ``ConcatFeatureSource``
    refused any 3-D source, so every learned-alignment component sidecar
    failed the moment ``features.residency`` was set to ``lazy`` -- which
    is exactly the setting that keeps ACF inside its memory budget.
    """
    _sources(tmp_path)
    (tmp_path / "sidecar.json").write_text(
        '{"strategy": "mean", "online": true, "alignment": "learned", "dim": 12,'
        ' "components": ["resnet50_comp.npy", "vit_b16_comp.npy"], "normalize": true}',
        encoding="utf-8",
    )

    source = load_embedding(tmp_path / "sidecar.json", lazy=True)

    assert tuple(source.shape) == (N_ITEMS, REGIONS, D_A + D_B)
    assert list(source.source_dims) == [D_A, D_B]
    rows = source.read_rows(np.array([3, 1, 3]))
    assert rows.shape == (3, REGIONS, D_A + D_B)

    eager = load_embedding(tmp_path / "sidecar.json")
    np.testing.assert_allclose(
        rows.astype("float32"),
        np.asarray(eager)[[3, 1, 3]].astype("float32"),
        err_msg="lazy rows must equal the eager array for the same ids",
    )


def test_dense_and_lazy_acf_agree_on_the_projected_components(tmp_path: Path) -> None:
    """Found 2026-09-09: the lazy path resolves through ``_resolve_visual``,
    which ALREADY applies the per-region fusion, so ACF must not apply it a
    second time.  Doing so split an aligned-width tensor by the native
    ``source_dims`` and raised on the first real lazy run."""
    _sources(tmp_path)
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text(
        '{"strategy": "mean", "online": true, "alignment": "learned", "dim": 12,'
        ' "components": ["resnet50_comp.npy", "vit_b16_comp.npy"], "normalize": true}',
        encoding="utf-8",
    )
    train = {u: {(u * 3) % N_ITEMS, (u * 5) % N_ITEMS} for u in range(8)}
    config = {"latent_dim": 8, "att_hidden": 8, "max_history": 5, "l2_reg": 1e-4}
    history = torch.tensor([[0, 1, 2], [3, 4, 5]])

    def projected(lazy: bool) -> torch.Tensor:
        torch.manual_seed(0)
        model = ACF(
            8,
            N_ITEMS,
            visual_embeddings=load_embedding(sidecar, lazy=lazy),
            config=config,
            train_interactions=train,
        ).train()
        with torch.no_grad():
            return model._projected_components(history)

    dense, lazy = projected(False), projected(True)

    assert dense.shape == (2, 3, REGIONS, 8)
    torch.testing.assert_close(dense, lazy)


@pytest.mark.parametrize(
    "strategy, kwargs",
    [
        ("mean", {}),
        ("sum", {}),
        ("prod", {}),
        ("max_pool", {}),
        ("weighted_mean", {"weights": [0.3, 0.7]}),
        ("softmax_weighted", {"logits": [1.0, 0.0]}),
        ("sigmoid_gated", {"logits": [1.0, 0.0]}),
        ("adaptive_gated", {}),
    ],
)
def test_every_strategy_fuses_per_region_rows(strategy: str, kwargs: dict) -> None:
    """EVERY enabled strategy must accept ``(B, R, sum(D_i))``, not just the
    plain reductions.

    Found 2026-09-09: the four weight-carrying strategies reshaped their
    per-source weights to a fixed rank, which broadcast against the wrong
    axes once a region axis existed.  They were the only ones that failed
    on the first real ACF run, and the unit tests missed it because they
    exercised ``mean`` alone.
    """
    from src.fusions.online import LearnedAlignmentFusion

    torch.manual_seed(0)
    fusion = LearnedAlignmentFusion(
        source_dims=[D_A, D_B], dim=12, strategy=strategy, normalize=True, **kwargs
    )
    pooled = torch.randn(5, D_A + D_B)
    per_region = torch.randn(5, REGIONS, D_A + D_B)

    assert fusion(pooled).shape == (5, 12)
    assert fusion(per_region).shape == (5, REGIONS, 12)

    # Region r of the batch must equal the pooled result on those rows:
    # the fusion is last-axis-wise and must not mix regions.
    torch.testing.assert_close(fusion(per_region)[:, 1, :], fusion(per_region[:, 1, :]))
