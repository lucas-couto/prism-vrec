"""Methodological guarantee: identical learned projection across fusions.

The equal-dim fusion family must share ONE projection architecture,
sized by the single ``alignment.dim`` key, trained end-to-end by BPR,
and the concatenation family must never carry it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.fusions.online import LearnedAlignmentFusion, RaggedSources
from src.fusions.registry import get_fusion_spec
from src.steps.fuse import _collect_fusion_tasks
from src.utils.config_schema import AlignmentConfig

N_ITEMS = 12
D1, D2 = 20, 8
DIM = 6
EQUAL_DIM_OPS = [
    "mean",
    "sum",
    "prod",
    "max_pool",
    "weighted_mean",
    "attention_weighted",
    "gated",
    "adaptive_gated",
]
CONCAT_OPS = ["concat", "pca", "pca_per_model"]


def _projection_signature(module: LearnedAlignmentFusion) -> list[tuple]:
    return [
        (type(p).__name__, tuple(p.weight.shape), tuple(p.bias.shape), p.bias is not None)
        for p in module.projections
    ]


def _ragged(strategy: str) -> RaggedSources:
    rng = np.random.default_rng(0)
    concat = rng.standard_normal((N_ITEMS, D1 + D2)).astype("float32")
    return RaggedSources(concat, source_dims=[D1, D2], strategy=strategy, aligned_dim=DIM)


class TestIdenticalProjectionAcrossFusions:
    def test_every_equal_dim_op_builds_the_same_projection_architecture(self) -> None:
        signatures = {
            op: _projection_signature(LearnedAlignmentFusion([D1, D2], dim=DIM, strategy=op))
            for op in EQUAL_DIM_OPS
        }

        reference = signatures["mean"]
        assert reference == [
            ("Linear", (DIM, D1), (DIM,), True),
            ("Linear", (DIM, D2), (DIM,), True),
        ]
        assert all(sig == reference for sig in signatures.values()), signatures

    def test_projection_parameter_count_is_strategy_independent(self) -> None:
        counts = {
            op: sum(
                p.numel()
                for p in LearnedAlignmentFusion(
                    [D1, D2], dim=DIM, strategy=op
                ).projections.parameters()
            )
            for op in EQUAL_DIM_OPS
        }

        assert len(set(counts.values())) == 1, counts
        assert counts["mean"] == (D1 + 1) * DIM + (D2 + 1) * DIM

    def test_each_source_has_its_own_matrix(self) -> None:
        module = LearnedAlignmentFusion([D1, D2], dim=DIM, strategy="sum")

        assert module.projections[0].weight is not module.projections[1].weight
        assert module.projections[0].in_features == D1
        assert module.projections[1].in_features == D2


class TestProjectionsAreTrainedByBpr:
    @pytest.mark.parametrize("strategy", EQUAL_DIM_OPS)
    def test_all_projection_params_are_in_optimizer_and_get_gradient(self, strategy: str) -> None:
        from src.recommenders.vbpr import VBPR

        torch.manual_seed(0)
        kwargs = {"weights": [0.7, 0.3]} if strategy == "weighted_mean" else {}
        ragged = _ragged(strategy)
        ragged.fusion_kwargs = kwargs
        model = VBPR(
            4, N_ITEMS, visual_embeddings=ragged, config={"latent_dim": 4, "visual_dim": 5}
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        proj_params = list(model._online_fusion.projections.parameters())
        optimizer_params = {id(p) for g in optimizer.param_groups for p in g["params"]}
        before = [p.detach().clone() for p in proj_params]

        loss = model.bpr_loss(
            *model(torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]))
        )
        loss.backward()
        optimizer.step()

        assert len(proj_params) == 4  # 2 sources x (weight, bias)
        assert all(id(p) in optimizer_params for p in proj_params)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in proj_params)
        assert all(not torch.equal(b, p.detach()) for b, p in zip(before, proj_params, strict=True))


class TestDimIsReadFromConfig:
    def test_alignment_dim_flows_from_config_to_sidecar_to_module(self, tmp_path) -> None:
        cfg = AlignmentConfig(method="learned", dim=DIM)
        ds = tmp_path / "emb" / "ds"
        ds.mkdir(parents=True)
        np.save(ds / "a.npy", np.zeros((N_ITEMS, D1), dtype="float32"))
        np.save(ds / "b.npy", np.zeros((N_ITEMS, D2), dtype="float32"))
        proc = tmp_path / "proc" / "ds"
        proc.mkdir(parents=True)
        (proc / "train.csv").write_text("user_idx,item_idx\n0,0\n1,1\n")

        tasks = _collect_fusion_tasks(
            "ds",
            str(tmp_path / "emb"),
            str(tmp_path / "proc"),
            ["a", "b"],
            {},
            True,
            set(EQUAL_DIM_OPS),
            cfg.method,
            cfg.dim,
        )

        sidecars = [t["sidecar_payload"] for t in tasks]
        assert {s["strategy"] for s in sidecars} == set(EQUAL_DIM_OPS)
        assert {s["dim"] for s in sidecars} == {DIM}
        assert all(s["alignment"] == "learned" for s in sidecars)
        for s in sidecars:
            module = LearnedAlignmentFusion([D1, D2], dim=s["dim"], strategy=s["strategy"])
            assert all(p.out_features == DIM for p in module.projections)

    def test_schema_default_and_bounds(self) -> None:
        assert AlignmentConfig().method == "learned"
        with pytest.raises(ValueError):
            AlignmentConfig(dim=0)


class TestConcatFamilyIsUntouched:
    @pytest.mark.parametrize("strategy", CONCAT_OPS)
    def test_concat_family_is_offline_and_dim_agnostic(self, strategy: str) -> None:
        spec = get_fusion_spec(strategy)

        assert spec.equal_dim_required is False
        assert spec.online is False

    def test_concat_keeps_native_dims_and_recommender_has_no_projection(self) -> None:
        from src.fusions.strategies import fuse_concat
        from src.recommenders.vbpr import VBPR

        rng = np.random.default_rng(0)
        sources = [rng.standard_normal((N_ITEMS, d)).astype("float32") for d in (D1, D2)]

        fused = fuse_concat(sources)
        model = VBPR(4, N_ITEMS, visual_embeddings=fused, config={"latent_dim": 4, "visual_dim": 5})

        assert fused.shape == (N_ITEMS, D1 + D2)
        assert model._online_fusion is None
        assert model.visual_dim_raw == D1 + D2
