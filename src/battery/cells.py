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


def resolve_seeds(config: dict) -> list[int]:
    """Battery seeds: ``seeds:`` if set, else the single ``seed``.

    The first seed is the primary (search) seed.
    """
    seeds = config.get("seeds")
    if seeds:
        return [int(s) for s in seeds]
    return [int(config.get("seed", 42))]


def enumerate_cells(
    config: dict,
    *,
    processed_dir: str | None = None,
    embeddings_dir: str | None = None,
    condition: str = "frozen",
) -> list[BatteryCell]:
    """Every battery cell, rules applied, over the ACTUAL feature artifacts.

    Reuses the pipeline's cell discovery (:func:`src.steps.train._iter_cells`)
    so ``visual_config`` is the real embedding stem (``resnet50``,
    ``hybrid_mean_learned_D128``, ``none``), not a strategy name — the same
    identity training, evaluation, checkpoints and the F artifact all use.
    That discovery already bakes in BPR-once (``embedding_name='none'``) and
    the ``_comp`` routing; here we add the AVBPR exclusion and the seed x
    role dimension (primary seed = search, others = replay).  Requires the
    features to be on disk (the battery runs after extract/fuse).
    """
    from src.steps.train import _iter_cells, _resolve_model_names

    processed_dir = processed_dir or config.get("paths", {}).get("data_processed", "data/processed")
    embeddings_dir = embeddings_dir or config.get("paths", {}).get("embeddings", "data/embeddings")
    seeds = resolve_seeds(config)
    primary = seeds[0] if seeds else None
    model_names = [m for m in _resolve_model_names(config) if m not in EXCLUDED_RECOMMENDERS]

    cells: list[BatteryCell] = []
    for pc in _iter_cells(condition, config, processed_dir, embeddings_dir, model_names):
        for seed in seeds:
            role = "search" if seed == primary else "replay"
            cells.append(BatteryCell(pc.dataset_name, pc.embedding_name, pc.model_name, seed, role))
    return cells
