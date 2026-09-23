"""Canonical rules for the artifact-filename routing protocol.

The pipeline encodes routing information in filenames rather than a
sidecar manifest: a fine-tuned backbone carries ``_finetuned``, a 3-D
per-item component artifact ends in ``_comp``, an offline fusion is
prefixed ``hybrid_``, a fixed projection is ``_<method><dim>`` and the winning
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

#: Short token per projection method, written into the artifact name by
#: ``src.extractors.projection``.  The width alone is NOT an identity:
#: ``pca`` and ``pca_whitened`` at the same width produced the same
#: filename under the old ``p<dim>`` token, so they could not coexist in
#: a run and a stale artifact was silently reused (2026-09-10).
PROJECTION_METHOD_TOKENS = {"pca": "pca", "pca_whitened": "pcaw", "random": "rand"}

#: A fixed-dim projection artifact carries a ``<method><dim>`` segment,
#: written immediately after the extractor name and before the condition
#: suffix (``resnet50_pcaw128_finetuned``).  Fusion outputs built from
#: projected sources carry it too (``hybrid_mean_pcaw128``), so one rule
#: classifies both.
#:
#: The bare ``p<dim>`` alternative is the LEGACY token, recognised on
#: read and never written: a leftover ``resnet50_p128.npy`` must still
#: classify as a projection, or the embedding glob would pick it up as a
#: native backbone of its own and it would enter the statistical
#: families beside the real ResNet-50.
PROJECTED_SEGMENT = re.compile(
    r"^(?:p|" + "|".join(sorted(PROJECTION_METHOD_TOKENS.values())) + r")\d+$"
)

#: Method token -> config method name, for parsing a name back.
_TOKEN_TO_METHOD = {token: method for method, token in PROJECTION_METHOD_TOKENS.items()}


def is_finetuned_artifact(name: str) -> bool:
    """Whether an embedding name comes from a fine-tuned backbone."""
    return FINETUNED_MARKER in name


def fusion_strategy_of(stem: str, known_strategies) -> str | None:
    """Strategy name encoded in a ``hybrid_*`` artifact stem, or ``None``.

    Longest-match against *known_strategies*: names prefix each other
    (``pca`` / ``pca_per_model``, ``gated`` / ``adaptive_gated``), so a
    naive prefix test would misattribute ``hybrid_pca_per_model_nc64``
    to ``pca``.  Non-fusion stems and stems whose strategy is not in
    *known_strategies* return ``None``.
    """
    if not stem.startswith(FUSION_PREFIX):
        return None
    rest = stem[len(FUSION_PREFIX) :]
    for name in sorted(known_strategies, key=len, reverse=True):
        if rest == name or rest.startswith(f"{name}_"):
            return name
    return None


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


def _projection_segment(name: str) -> str | None:
    """The whole projection segment of *name*, or ``None`` if it is native."""
    for part in name.split("_"):
        if PROJECTED_SEGMENT.match(part):
            return part
    return None


def projection_dim(name: str) -> int | None:
    """The projected width encoded in *name*, or ``None`` if it is native."""
    part = _projection_segment(name)
    if part is None:
        return None
    return int(part.lstrip("abcdefghijklmnopqrstuvwxyz"))


def projection_method(name: str) -> str | None:
    """The projection method encoded in *name*.

    ``None`` for a native artifact AND for the legacy ``p<dim>`` token,
    which predates the method being part of the name: the width is
    recoverable from it, the recipe is not.
    """
    part = _projection_segment(name)
    if part is None:
        return None
    token = part.rstrip("0123456789")
    return _TOKEN_TO_METHOD.get(token)


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
