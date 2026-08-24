"""Config-vs-disk eligibility predicates for embedding artifacts.

The embeddings and models directories accumulate artifacts across runs
(every backbone ever extracted, every strategy ever fused, every cell
ever trained).  The config — ``recommenders_enabled``,
``extractors_enabled``, ``fusion_strategies_enabled``,
``embedding_variants`` — is the source of truth for what the CURRENT
experiment contains; these predicates answer "would this artifact be a
cell under the current config?" so the evaluate step never resurrects
checkpoints of disabled models/backbones/strategies.

Pure predicates (no logging): callers decide how to report exclusions.
The train step keeps its own list-level filters with per-step logging;
both sides share the parsing helpers in
:mod:`src.utils.artifact_names`.
"""

from __future__ import annotations

from src.utils.artifact_names import (
    FUSION_PREFIX,
    fusion_strategy_of,
    is_projected_artifact,
)


def _variant_matches(embedding_name: str, config: dict) -> bool:
    """Mirror of the train step's ``filter_by_variant`` for one name."""
    variant = str(config.get("embedding_variants", "both"))
    if variant == "both" or embedding_name == "none":
        return True
    return is_projected_artifact(embedding_name) == (variant == "projected")


def _fusion_matches(embedding_name: str, config: dict) -> bool:
    from src.fusions.registry import registered_fusion_strategies

    strategy = fusion_strategy_of(embedding_name, registered_fusion_strategies())
    if strategy is None:
        return False  # unparseable hybrid stem: never eligible
    return strategy in set(config.get("fusion_strategies_enabled") or [])


def _extractor_matches(embedding_name: str, config: dict) -> bool:
    from src.extractors import registered_extractor_names

    enabled = set(config.get("extractors_enabled") or [])
    known = sorted(registered_extractor_names(), key=len, reverse=True)
    base = next(
        (k for k in known if embedding_name == k or embedding_name.startswith(f"{k}_")),
        None,
    )
    return base is not None and base in enabled


def embedding_matches_config(embedding_name: str, config: dict) -> bool:
    """Whether *embedding_name* belongs to the current experiment.

    ``none`` (the non-visual baseline) always matches; ``hybrid_*``
    stems match when their strategy is enabled; every other stem
    matches when its backbone is enabled.  The ``embedding_variants``
    (native/projected/both) gate applies to all of them.
    """
    if not _variant_matches(embedding_name, config):
        return False
    if embedding_name == "none":
        return True
    if embedding_name.startswith(FUSION_PREFIX):
        return _fusion_matches(embedding_name, config)
    return _extractor_matches(embedding_name, config)


def checkpoint_matches_config(model_info: dict, config: dict) -> bool:
    """Whether a ``*_best.pt`` checkpoint is a cell of the current config.

    *model_info* is one entry of
    :func:`src.steps.evaluate.find_best_models` (``model_name`` +
    ``embedding_name``).
    """
    if model_info["model_name"] not in set(config.get("recommenders_enabled") or []):
        return False
    return embedding_matches_config(model_info["embedding_name"], config)
