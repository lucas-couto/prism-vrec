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
* ``frozen_vs_finetuned`` — does fine-tuning help?  One pair per
  (recommender, base embedding) present in both conditions (``m = 1``).

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

DEFAULT_FAMILIES = (
    "backbone_within_model",
    "model_within_backbone",
    "fusion_within_model",
    "frozen_vs_finetuned",
)
VALID_FAMILIES = DEFAULT_FAMILIES + ("all_pairs",)


@dataclass(frozen=True)
class FamilyInstance:
    """One independent correction unit: a family applied to one group.

    ``pairs`` lists the ``(config_a, config_b)`` keys to test; Holm runs
    over exactly this set (``m = len(pairs)``).
    """

    family: str
    group: str
    pairs: tuple[tuple[str, str], ...]
    configs: tuple[str, ...] = field(default=())


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
    """One m=1 instance per (model, base embedding) present in BOTH conditions.

    Fine-tuned artifacts carry the ``_finetuned`` marker in the embedding
    name, so both conditions coexist in a ``condition="all"`` table.
    """
    out: list[FamilyInstance] = []
    visual = df[df["embedding_name"] != "none"].copy()
    visual["base"] = visual["embedding_name"].map(_backbone_base)
    visual["cond"] = visual["embedding_name"].map(_condition_of)
    for (model, base), grp in visual.groupby(["model_name", "base"], sort=True):
        by_cond = {c: e for c, e in zip(grp["cond"], grp["embedding_name"], strict=True)}
        if {"frozen", "finetuned"} <= set(by_cond):
            pair = (
                _config_key(model, by_cond["frozen"]),
                _config_key(model, by_cond["finetuned"]),
            )
            out.append(
                FamilyInstance(
                    family="frozen_vs_finetuned",
                    group=f"model={model},backbone={base}",
                    pairs=(pair,),
                    configs=tuple(sorted(pair)),
                )
            )
    return out


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
    "all_pairs": _all_pairs,
}
