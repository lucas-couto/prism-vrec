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

FINETUNED_MARKER = "_finetuned"
COMPONENT_SUFFIX = "_comp"
FUSION_PREFIX = "hybrid_"
BEST_SUFFIX = "_best"


def is_finetuned_artifact(name: str) -> bool:
    """Whether an embedding name comes from a fine-tuned backbone."""
    return FINETUNED_MARKER in name


def is_component_artifact(name: str) -> bool:
    """Whether an embedding stem is a 3-D per-item component artifact.

    Component artifacts (``<extractor>_D<dim>_comp``) feed models that
    declare ``requires_components`` (e.g. ACF); they are routed only to
    those models and excluded from the pooled-embedding pool.
    """
    return name.endswith(COMPONENT_SUFFIX)


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
