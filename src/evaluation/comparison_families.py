"""Comparison families for the statistical analysis (C1).

A multiple-comparison correction must be applied WITHIN the family of
comparisons that a research question defines — not over the Cartesian
product of every config in a dataset.  With ~77 configs, all-pairs Holm
runs with ``m ≈ 2900`` and rejects everything for an artificial reason:
most of those pairs (e.g. ``vbpr_resnet50`` vs ``acf_dinov2``) vary two
experimental dimensions at once and answer no question.

Each family below fixes every dimension except one:

* ``backbone_within_model`` — which backbone extracts best?  Varies the
  extractor, fixes the recommender (one instance per recommender ×
  condition; ``m = C(n_backbones, 2)``).
* ``model_within_backbone`` — which recommender is best?  Varies the
  recommender, fixes the backbone (one instance per backbone ×
  condition; component artifacts are grouped with their base backbone).
* ``fusion_within_model`` — which fusion strategy is best?  Varies the
  fusion artifact, fixes the recommender.
* ``frozen_vs_finetuned`` — does fine-tuning help?  ONE instance per
  dataset containing every (recommender, base embedding) pair present
  in both conditions, so Holm corrects across all of them
  (``m = n_pairs`` — the family is one research question, not one
  question per config).
* ``vs_baseline`` — does the visual signal help at all?  ONE instance
  per dataset pairing every config with the pure-BPR baseline
  (``bpr_none``); Holm corrects across all of them (``m = n_configs``).

``all_pairs`` remains available as an EXPLORATORY option and is never
part of the default set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import pandas as pd

from src.utils.artifact_names import (
    COMPONENT_SUFFIX,
    FINETUNED_MARKER,
    FUSION_PREFIX,
    is_finetuned_artifact,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_FAMILIES = (
    "backbone_within_model",
    "model_within_backbone",
    "fusion_within_model",
    "frozen_vs_finetuned",
    "vs_baseline",
)
VALID_FAMILIES = DEFAULT_FAMILIES + ("all_pairs",)

BASELINE_MODEL = "bpr"
BASELINE_EMBEDDING = "none"


@dataclass(frozen=True)
class FamilyInstance:
    """One independent correction unit: a family applied to one group.

    ``pairs`` lists the ``(config_a, config_b)`` keys to test; Holm runs
    over exactly this set (``m = len(pairs)``).

    ``omnibus_defined`` says whether a K-way Friedman omnibus over
    ``configs`` answers this family's research question.  It is False
    for the pair-collection families (``frozen_vs_finetuned``,
    ``vs_baseline``): their instances bundle many independent
    two-treatment questions for the Holm correction, so the only K-way
    hypothesis available — "all ~n configs of the dataset are
    equivalent" — is trivially false and the gate would never gate.
    The statistical step skips Friedman for such instances and reports
    ``omnibus_significant = NaN`` on their pairwise rows.
    """

    family: str
    group: str
    pairs: tuple[tuple[str, str], ...]
    configs: tuple[str, ...] = field(default=())
    omnibus_defined: bool = True


def _config_key(model: str, embedding: str) -> str:
    """Same identity rule as ``statistical._ensure_config``."""
    return f"{model}_{embedding}"


def _embedding_kind(name: str) -> str:
    """Classify an embedding stem: none | fusion | backbone.

    Component artifacts count as ``backbone`` (they are a backbone's
    spatial features routed to component models such as ACF).
    """
    if name == "none":
        return "none"
    if name.startswith(FUSION_PREFIX):
        return "fusion"
    return "backbone"


def _backbone_base(name: str) -> str:
    """Strip routing tokens so component/finetuned variants group with
    their base backbone (``resnet50_finetuned_comp`` → ``resnet50``)."""
    return name.removesuffix(COMPONENT_SUFFIX).replace(FINETUNED_MARKER, "")


def _condition_of(name: str) -> str:
    return "finetuned" if is_finetuned_artifact(name) else "frozen"


def enumerate_family_instances(
    cells: pd.DataFrame,
    families: list[str] | tuple[str, ...] = DEFAULT_FAMILIES,
) -> list[FamilyInstance]:
    """Build every :class:`FamilyInstance` present in *cells*.

    *cells* needs ``model_name`` and ``embedding_name`` columns (one row
    per config; duplicates are dropped).  Unknown family names raise.
    """
    unknown = [f for f in families if f not in VALID_FAMILIES]
    if unknown:
        raise ValueError(f"Unknown comparison families {unknown}; valid: {sorted(VALID_FAMILIES)}")

    df = cells[["model_name", "embedding_name"]].drop_duplicates().astype(str)
    instances: list[FamilyInstance] = []
    for family in families:
        instances.extend(_BUILDERS[family](df))
    return [inst for inst in instances if inst.pairs]


def _pairs_of(configs: list[str]) -> tuple[tuple[str, str], ...]:
    return tuple(combinations(sorted(configs), 2))


def _backbone_within_model(df: pd.DataFrame) -> list[FamilyInstance]:
    out: list[FamilyInstance] = []
    mask = df["embedding_name"].map(_embedding_kind) == "backbone"
    sub = df[mask].copy()
    sub["cond"] = sub["embedding_name"].map(_condition_of)
    for (model, cond), grp in sub.groupby(["model_name", "cond"], sort=True):
        configs = [
            _config_key(m, e) for m, e in zip(grp["model_name"], grp["embedding_name"], strict=True)
        ]
        out.append(
            FamilyInstance(
                family="backbone_within_model",
                group=f"model={model},condition={cond}",
                pairs=_pairs_of(configs),
                configs=tuple(sorted(configs)),
            )
        )
    return out


def _model_within_backbone(df: pd.DataFrame) -> list[FamilyInstance]:
    out: list[FamilyInstance] = []
    mask = df["embedding_name"].map(_embedding_kind) == "backbone"
    sub = df[mask].copy()
    sub["base"] = sub["embedding_name"].map(_backbone_base)
    sub["cond"] = sub["embedding_name"].map(_condition_of)
    for (base, cond), grp in sub.groupby(["base", "cond"], sort=True):
        configs = [
            _config_key(m, e) for m, e in zip(grp["model_name"], grp["embedding_name"], strict=True)
        ]
        out.append(
            FamilyInstance(
                family="model_within_backbone",
                group=f"backbone={base},condition={cond}",
                pairs=_pairs_of(configs),
                configs=tuple(sorted(configs)),
            )
        )
    return out


def _fusion_within_model(df: pd.DataFrame) -> list[FamilyInstance]:
    out: list[FamilyInstance] = []
    mask = df["embedding_name"].map(_embedding_kind) == "fusion"
    sub = df[mask].copy()
    sub["cond"] = sub["embedding_name"].map(_condition_of)
    for (model, cond), grp in sub.groupby(["model_name", "cond"], sort=True):
        configs = [
            _config_key(m, e) for m, e in zip(grp["model_name"], grp["embedding_name"], strict=True)
        ]
        out.append(
            FamilyInstance(
                family="fusion_within_model",
                group=f"model={model},condition={cond}",
                pairs=_pairs_of(configs),
                configs=tuple(sorted(configs)),
            )
        )
    return out


def _frozen_vs_finetuned(df: pd.DataFrame) -> list[FamilyInstance]:
    """ONE instance with every (model, base embedding) frozen/finetuned pair.

    "Does fine-tuning help?" is one research question, so Holm must
    correct across ALL its pairs (``m = n_pairs``) — one m=1 instance
    per pair would leave every test uncorrected.  Fine-tuned artifacts
    carry the ``_finetuned`` marker in the embedding name, so both
    conditions coexist in a ``condition="all"`` table.

    ``omnibus_defined=False``: the instance is a bundle of independent
    two-treatment questions (each config's frozen vs finetuned), not one
    K-way "which of these treatments differ?" question, so no Friedman
    over its configs is meaningful (R1).
    """
    pairs: list[tuple[str, str]] = []
    visual = df[df["embedding_name"] != "none"].copy()
    visual["base"] = visual["embedding_name"].map(_backbone_base)
    visual["cond"] = visual["embedding_name"].map(_condition_of)
    for (model, base), grp in visual.groupby(["model_name", "base"], sort=True):
        by_cond: dict[str, str] = {}
        for cond, emb in zip(grp["cond"], grp["embedding_name"], strict=True):
            if cond in by_cond:
                raise ValueError(
                    f"Duplicate {cond!r} embedding for model={model!r}, backbone={base!r}: "
                    f"{by_cond[cond]!r} vs {emb!r} — cannot build an unambiguous "
                    "frozen-vs-finetuned pair."
                )
            by_cond[cond] = emb
        if {"frozen", "finetuned"} <= set(by_cond):
            pairs.append(
                (
                    _config_key(model, by_cond["frozen"]),
                    _config_key(model, by_cond["finetuned"]),
                )
            )
    if not pairs:
        return []
    configs = sorted({c for pair in pairs for c in pair})
    return [
        FamilyInstance(
            family="frozen_vs_finetuned",
            group="all",
            pairs=tuple(pairs),
            configs=tuple(configs),
            omnibus_defined=False,
        )
    ]


def _vs_baseline(df: pd.DataFrame) -> list[FamilyInstance]:
    """ONE instance pairing every config with the pure-BPR baseline.

    The central hypothesis — does the visual signal beat pure
    collaborative BPR? — is one research question, so Holm corrects
    across every (config vs ``bpr_none``) pair (``m = n_configs − 1``,
    the baseline pairs with everyone but itself).  Pairs are ordered
    ``(config, baseline)`` so ``diff_mean`` reads as "config minus
    baseline".  Absent baseline: empty list (logged).

    ``omnibus_defined=False``: the design is a star — every config
    against ONE shared baseline — i.e. a bundle of two-treatment
    questions.  A K-way Friedman over its configs would test "all
    dataset configs are equivalent", which is trivially rejected and
    gates nothing (R1).
    """
    baseline = _config_key(BASELINE_MODEL, BASELINE_EMBEDDING)
    keys = [_config_key(m, e) for m, e in zip(df["model_name"], df["embedding_name"], strict=True)]
    if baseline not in keys:
        logger.warning(
            "vs_baseline family skipped: baseline config %r not present in the results.",
            baseline,
        )
        return []
    pairs = tuple((c, baseline) for c in sorted(keys) if c != baseline)
    if not pairs:
        return []
    return [
        FamilyInstance(
            family="vs_baseline",
            group="all",
            pairs=pairs,
            configs=tuple(sorted(keys)),
            omnibus_defined=False,
        )
    ]


def _all_pairs(df: pd.DataFrame) -> list[FamilyInstance]:
    configs = [
        _config_key(m, e) for m, e in zip(df["model_name"], df["embedding_name"], strict=True)
    ]
    return [
        FamilyInstance(
            family="all_pairs",
            group="all",
            pairs=_pairs_of(configs),
            configs=tuple(sorted(configs)),
        )
    ]


_BUILDERS = {
    "backbone_within_model": _backbone_within_model,
    "model_within_backbone": _model_within_backbone,
    "fusion_within_model": _fusion_within_model,
    "frozen_vs_finetuned": _frozen_vs_finetuned,
    "vs_baseline": _vs_baseline,
    "all_pairs": _all_pairs,
}
