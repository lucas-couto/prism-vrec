"""Comparison families (C1) + paired-diff CI (C4) + effect-size policy (C3).

The correction's ``m`` must equal the family size (e.g. C(8,2)=28 for
backbones within one recommender), never the dataset-wide pair count;
families never mix two experimental dimensions in one pair; and the
paired-difference bootstrap CI must agree with the Wilcoxon verdict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.comparison_families import (
    DEFAULT_FAMILIES,
    enumerate_family_instances,
)
from src.evaluation.statistical import bootstrap_diff_ci, pairwise_significance, wilcoxon_test

BACKBONES = ["resnet50", "vit_b16", "cvt_13", "coatnet_0"]
MODELS = ["vbpr", "deepstyle"]


def _cells() -> pd.DataFrame:
    rows = [{"model_name": "bpr", "embedding_name": "none"}]
    for m in MODELS:
        for b in BACKBONES:
            rows.append({"model_name": m, "embedding_name": b})
        rows.append({"model_name": m, "embedding_name": "hybrid_mean_learned_D128"})
        rows.append({"model_name": m, "embedding_name": "hybrid_concat"})
    rows.append({"model_name": "acf", "embedding_name": "resnet50_comp"})
    rows.append({"model_name": "vbpr", "embedding_name": "resnet50_finetuned"})
    return pd.DataFrame(rows)


class TestFamilyEnumeration:
    def test_backbone_within_model_has_c_n_2_pairs(self) -> None:
        instances = enumerate_family_instances(_cells(), ["backbone_within_model"])

        vbpr_frozen = next(i for i in instances if i.group == "model=vbpr,condition=frozen")
        n = len(BACKBONES)
        assert len(vbpr_frozen.pairs) == n * (n - 1) // 2

    def test_no_pair_varies_two_dimensions(self) -> None:
        # The fixed dimension is the recommender or the BASE backbone —
        # ACF consumes `<backbone>_comp` by construction, so within
        # model_within_backbone the artifact string differs while the
        # backbone identity (the dimension under study) stays fixed.
        from src.evaluation.comparison_families import _backbone_base

        instances = enumerate_family_instances(_cells(), list(DEFAULT_FAMILIES))

        cells = _cells()
        emb_of = {
            f"{m}_{e}": e for m, e in zip(cells["model_name"], cells["embedding_name"], strict=True)
        }
        model_of = {
            f"{m}_{e}": m for m, e in zip(cells["model_name"], cells["embedding_name"], strict=True)
        }
        for inst in instances:
            for a, b in inst.pairs:
                same_model = model_of[a] == model_of[b]
                same_backbone = _backbone_base(emb_of[a]) == _backbone_base(emb_of[b])
                if inst.family == "frozen_vs_finetuned":
                    assert same_model and same_backbone
                elif inst.family == "vs_baseline":
                    # By design every pair anchors on the pure-BPR baseline.
                    assert b == "bpr_none"
                else:
                    assert same_model or same_backbone, (
                        f"{inst.family}: pair ({a}, {b}) varies model AND backbone"
                    )

    def test_frozen_never_paired_with_finetuned_within_backbone_family(self) -> None:
        instances = enumerate_family_instances(_cells(), ["backbone_within_model"])

        for inst in instances:
            for a, b in inst.pairs:
                assert ("finetuned" in a) == ("finetuned" in b)

    def test_frozen_vs_finetuned_is_one_instance_for_the_whole_family(self) -> None:
        instances = enumerate_family_instances(_cells(), ["frozen_vs_finetuned"])

        assert len(instances) == 1
        inst = instances[0]
        assert inst.group == "all"
        assert inst.pairs == (("vbpr_resnet50", "vbpr_resnet50_finetuned"),)

    def test_frozen_vs_finetuned_holm_m_is_number_of_pairs(self) -> None:
        # Two (model, backbone) cells present in both conditions must
        # land in ONE instance so Holm runs with m=2, not twice with m=1.
        cells = pd.concat(
            [
                _cells(),
                pd.DataFrame([{"model_name": "deepstyle", "embedding_name": "vit_b16_finetuned"}]),
            ],
            ignore_index=True,
        )

        instances = enumerate_family_instances(cells, ["frozen_vs_finetuned"])

        assert len(instances) == 1
        assert len(instances[0].pairs) == 2
        assert ("deepstyle_vit_b16", "deepstyle_vit_b16_finetuned") in instances[0].pairs

    def test_frozen_vs_finetuned_duplicate_condition_raises(self) -> None:
        # Two frozen artifacts collapsing to the same (model, base) would
        # silently overwrite each other; the builder must refuse instead.
        cells = pd.DataFrame(
            [
                {"model_name": "vbpr", "embedding_name": "resnet50"},
                {"model_name": "vbpr", "embedding_name": "resnet50_comp"},
                {"model_name": "vbpr", "embedding_name": "resnet50_finetuned"},
            ]
        )

        with pytest.raises(ValueError, match="Duplicate 'frozen' embedding"):
            enumerate_family_instances(cells, ["frozen_vs_finetuned"])

    def test_component_artifacts_group_with_base_backbone(self) -> None:
        instances = enumerate_family_instances(_cells(), ["model_within_backbone"])

        resnet = next(i for i in instances if i.group == "backbone=resnet50,condition=frozen")
        assert "acf_resnet50_comp" in resnet.configs
        assert "vbpr_resnet50" in resnet.configs

    def test_vs_baseline_is_one_instance_pairing_every_config(self) -> None:
        cells = _cells()
        instances = enumerate_family_instances(cells, ["vs_baseline"])

        assert len(instances) == 1
        inst = instances[0]
        assert inst.group == "all"
        n_configs = len(cells)  # includes the baseline itself
        assert len(inst.pairs) == n_configs - 1
        # (config, baseline) ordering so diff_mean = config - baseline.
        assert all(b == "bpr_none" for _, b in inst.pairs)
        assert all(a != "bpr_none" for a, _ in inst.pairs)
        assert ("vbpr_resnet50", "bpr_none") in inst.pairs
        assert ("acf_resnet50_comp", "bpr_none") in inst.pairs

    def test_vs_baseline_absent_baseline_yields_no_instance(self) -> None:
        cells = _cells()
        cells = cells[~((cells["model_name"] == "bpr") & (cells["embedding_name"] == "none"))]

        instances = enumerate_family_instances(cells, ["vs_baseline"])

        assert instances == []

    def test_vs_baseline_is_part_of_default_families(self) -> None:
        assert "vs_baseline" in DEFAULT_FAMILIES

    def test_pair_collection_families_declare_no_omnibus(self) -> None:
        # R1: a K-way Friedman over these instances' configs would test
        # "all dataset configs are equivalent" — a gate that never gates.
        instances = enumerate_family_instances(_cells(), ["vs_baseline", "frozen_vs_finetuned"])

        assert instances
        assert all(inst.omnibus_defined is False for inst in instances)

    def test_one_dimension_families_keep_the_omnibus(self) -> None:
        instances = enumerate_family_instances(
            _cells(),
            ["backbone_within_model", "model_within_backbone", "fusion_within_model"],
        )

        assert instances
        assert all(inst.omnibus_defined is True for inst in instances)

    def test_unknown_family_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown comparison families"):
            enumerate_family_instances(_cells(), ["nonsense"])


def _long_df(configs: list[tuple[str, str]], n_users: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        for i, (m, e) in enumerate(configs):
            rows.append(
                {
                    "user_id": u,
                    "model_name": m,
                    "embedding_name": e,
                    # config i+1 strictly better in expectation
                    "ndcg@10": float(rng.random() < 0.1 * (i + 1)),
                }
            )
    return pd.DataFrame(rows)


class TestFamilyScopedCorrection:
    def test_holm_m_equals_family_size_not_dataset_pairs(self) -> None:
        configs = [("vbpr", b) for b in BACKBONES] + [("deepstyle", b) for b in BACKBONES]
        df = _long_df(configs, n_users=200)
        pairs = [("vbpr_resnet50", "vbpr_vit_b16"), ("vbpr_cvt_13", "vbpr_coatnet_0")]

        out = pairwise_significance(
            df,
            metric="ndcg@10",
            pairs=pairs,
            family="backbone_within_model",
            group="model=vbpr,condition=frozen",
            correction="bonferroni",  # deterministic p*m, easiest to audit
        )

        assert (out["n_comparisons_in_family"] == 2).all()
        assert (out["family"] == "backbone_within_model").all()
        # Bonferroni multiplies by len(pairs)=2, NOT by C(8,2)=28.
        for _, row in out.iterrows():
            assert row["corrected_p"] == pytest.approx(min(row["p_value"] * 2, 1.0))

    def test_pairs_absent_from_results_raise(self) -> None:
        df = _long_df([("vbpr", "resnet50")], n_users=10)

        with pytest.raises(ValueError, match="absent from results"):
            pairwise_significance(df, metric="ndcg@10", pairs=[("vbpr_resnet50", "vbpr_ghost")])


class TestPairedDiffCI:
    def test_diff_ci_agrees_with_wilcoxon_on_clear_effect(self) -> None:
        rng = np.random.default_rng(1)
        a = (rng.random(600) < 0.6).astype(float)
        b = (rng.random(600) < 0.2).astype(float)

        _, p = wilcoxon_test(a, b)
        diff_mean, lo, hi = bootstrap_diff_ci(a, b, seed=42)

        assert p < 0.05
        assert lo > 0.0  # CI of the paired difference excludes zero
        assert lo <= diff_mean <= hi

    def test_diff_ci_columns_present_in_pairwise_output(self) -> None:
        df = _long_df([("vbpr", "resnet50"), ("vbpr", "vit_b16")], n_users=100)

        out = pairwise_significance(df, metric="ndcg@10", diff_ci=True)

        assert {"diff_mean", "diff_ci_lower", "diff_ci_upper"} <= set(out.columns)

    def test_diff_ci_is_deterministic_given_seed(self) -> None:
        df = _long_df([("vbpr", "resnet50"), ("vbpr", "vit_b16")], n_users=100)

        out1 = pairwise_significance(df, metric="ndcg@10", seed=42)
        out2 = pairwise_significance(df, metric="ndcg@10", seed=42)

        assert out1["diff_ci_lower"].tolist() == out2["diff_ci_lower"].tolist()


class TestEffectSizePolicy:
    def test_cohens_d_absent_by_default_cliffs_present(self) -> None:
        df = _long_df([("vbpr", "resnet50"), ("vbpr", "vit_b16")], n_users=50)

        out = pairwise_significance(df, metric="ndcg@10")

        assert "cliffs_delta" in out.columns
        assert "cohens_d" not in out.columns

    def test_cohens_d_available_on_request(self) -> None:
        df = _long_df([("vbpr", "resnet50"), ("vbpr", "vit_b16")], n_users=50)

        out = pairwise_significance(df, metric="ndcg@10", include_cohens_d=True)

        assert "cohens_d" in out.columns
