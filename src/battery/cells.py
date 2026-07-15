"""Declarative battery-cell enumeration (Task I).

Enumerates every cell the full battery must run — datasets x visual
configs x recommenders x seeds — with the fixed rules baked in:

* BPR runs ONCE per (dataset, seed): it is feature-blind, so duplicating
  it per backbone would waste cells.
* AVBPR is excluded (out of the qualification scope).
* DeepStyle runs on Tradesy too — its expected degeneration to VBPR is a
  finding to confirm, not a cell to skip (no special-case here).
* The primary seed of each dataset carries the HP search; the other seeds
  are replay cells (best config re-trained, Task H's ``train_replay``).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.recommenders import get_recommender_spec

#: Recommenders excluded from the qualification battery.
EXCLUDED_RECOMMENDERS = frozenset({"avbpr"})

#: Visual config used by feature-blind recommenders (BPR).
NO_VISUAL = "none"


@dataclass(frozen=True)
class BatteryCell:
    """One unit of battery work."""

    dataset: str
    visual_config: str  # extractor / fusion name, or "none"
    recommender: str
    seed: int
    role: str  # "search" (primary seed) or "replay"

    def key(self) -> str:
        return f"{self.dataset}__{self.visual_config}__{self.recommender}__seed{self.seed}"


def _is_feature_blind(recommender: str) -> bool:
    try:
        return not get_recommender_spec(recommender).requires_visual
    except KeyError:
        return False


def resolve_seeds(config: dict) -> list[int]:
    """Battery seeds: ``seeds:`` if set, else the single ``seed``.

    The first seed is the primary (search) seed.
    """
    seeds = config.get("seeds")
    if seeds:
        return [int(s) for s in seeds]
    return [int(config.get("seed", 42))]


def visual_configs(config: dict) -> list[str]:
    """Every visual configuration: enabled extractors + enabled fusions."""
    return list(config.get("extractors_enabled", [])) + list(
        config.get("fusion_strategies_enabled", [])
    )


def enumerate_cells(config: dict) -> list[BatteryCell]:
    """Every battery cell implied by *config*, rules applied."""
    datasets = config.get("datasets", [])
    seeds = resolve_seeds(config)
    primary = seeds[0] if seeds else None
    recommenders = [
        r for r in config.get("recommenders_enabled", []) if r not in EXCLUDED_RECOMMENDERS
    ]
    visuals = visual_configs(config)

    cells: list[BatteryCell] = []
    for dataset in datasets:
        for seed in seeds:
            role = "search" if seed == primary else "replay"
            for recommender in recommenders:
                if _is_feature_blind(recommender):
                    cells.append(BatteryCell(dataset, NO_VISUAL, recommender, seed, role))
                else:
                    for visual in visuals:
                        cells.append(BatteryCell(dataset, visual, recommender, seed, role))
    return cells
