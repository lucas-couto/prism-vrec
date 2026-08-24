"""Canonical rules for the artifact-filename routing protocol.

The pipeline encodes routing information in filenames rather than a
sidecar manifest: a fine-tuned backbone carries ``_finetuned``, a 3-D
per-item component artifact ends in ``_comp``, an offline fusion is
prefixed ``hybrid_``, a projection dim is ``_D<dim>`` and the winning
checkpoint ends in ``_best``.  Extract, finetune, fuse, train and
evaluate all depend on these tokens; owning the format/parse rules here
keeps them from drifting apart (previously ``train`` matched
``"_finetuned"`` while ``evaluate`` matched ``"finetuned"`` — an
extractor legitimately named ``finetuned_*`` would have been mis-routed).
"""

from __future__ import annotations

import re

FINETUNED_MARKER = "_finetuned"
COMPONENT_SUFFIX = "_comp"
FUSION_PREFIX = "hybrid_"
BEST_SUFFIX = "_best"

#: A fixed-dim projection artifact carries a ``p<dim>`` segment, written
#: by ``src.extractors.projection`` immediately after the extractor name
#: and before the condition suffix (``resnet50_p128_finetuned``).  Fusion
#: outputs built from projected sources carry it too
#: (``hybrid_mean_p128``), so one rule classifies both.
PROJECTED_SEGMENT = re.compile(r"^p\d+$")


def is_finetuned_artifact(name: str) -> bool:
    """Whether an embedding name comes from a fine-tuned backbone."""
    return FINETUNED_MARKER in name


def is_component_artifact(name: str) -> bool:
    """Whether an embedding stem is a 3-D per-item component artifact.

    Component artifacts (``<extractor>_comp``) feed models that
    declare ``requires_components`` (e.g. ACF); they are routed only to
    those models and excluded from the pooled-embedding pool.
    """
    return name.endswith(COMPONENT_SUFFIX)


def is_projected_artifact(name: str) -> bool:
    """Whether an embedding name is a fixed-dim projection.

    Matches on a whole underscore-separated ``p<dim>`` segment rather
    than a substring, so an extractor legitimately named ``p3d`` or
    ``clip_patch`` is not mistaken for one.  Note the corollary: an
    extractor named with a bare ``p<digits>`` segment *would* be, which
    is part of the filename protocol this module owns.
    """
    return any(PROJECTED_SEGMENT.match(part) for part in name.split("_"))


def projection_dim(name: str) -> int | None:
    """The projected width encoded in *name*, or ``None`` if it is native."""
    for part in name.split("_"):
        if PROJECTED_SEGMENT.match(part):
            return int(part[1:])
    return None


def parse_checkpoint_stem(stem: str, known_models: list[str]) -> tuple[str, str] | None:
    """Split a ``{model_name}_{embedding_name}`` checkpoint stem.

    Recommender names may contain underscores (e.g. ``uniform_noise``),
    so the boundary cannot be inferred positionally.  *known_models* must
    be sorted longest-first so the longest matching recommender name wins
    as the prefix.  Returns ``(model_name, embedding_name)`` (embedding
    ``"none"`` when the stem is exactly a model name), or ``None`` when
    no registered model matches.
    """
    for candidate in known_models:
        if stem == candidate:
            return candidate, "none"
        if stem.startswith(candidate + "_"):
            return candidate, stem[len(candidate) + 1 :]
    return None
