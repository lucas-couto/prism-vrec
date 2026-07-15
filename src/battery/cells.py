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


def resolve_conditions(config: dict) -> list[str]:
    """Conditions to enumerate, from ``pipeline.condition`` (``both`` → both).

    Frozen and finetuned are distinct cells because the finetuned features
    carry a ``_finetuned`` stem, so the visual config (and every downstream
    identity — study, checkpoint, F artifact) differs on its own.  Finetuned
    cells only appear when the finetuned features exist on disk.
    """
    cond = config.get("pipeline", {}).get("condition", "both")
    return ["frozen", "finetuned"] if cond == "both" else [cond]


def enumerate_cells(
    config: dict,
    *,
    processed_dir: str | None = None,
    embeddings_dir: str | None = None,
    conditions: list[str] | None = None,
) -> list[BatteryCell]:
    """Every battery cell, rules applied, over the ACTUAL feature artifacts.

    Reuses the pipeline's cell discovery (:func:`src.steps.train._iter_cells`)
    so ``visual_config`` is the real embedding stem (``resnet50``,
    ``resnet50_finetuned``, ``hybrid_mean_learned_D128``, ``none``), not a
    strategy name — the same identity training, evaluation, checkpoints and
    the F artifact all use.  That discovery already bakes in BPR-once
    (``embedding_name='none'``, frozen only) and the ``_comp`` routing; here
    we add the AVBPR exclusion, the frozen/finetuned condition axis and the
    seed x role dimension (primary seed = search, others = replay).  Requires
    the features to be on disk (the battery runs after extract/fuse).
    """
    from src.steps.train import _iter_cells, _resolve_model_names

    processed_dir = processed_dir or config.get("paths", {}).get("data_processed", "data/processed")
    embeddings_dir = embeddings_dir or config.get("paths", {}).get("embeddings", "data/embeddings")
    seeds = resolve_seeds(config)
    primary = seeds[0] if seeds else None
    model_names = [m for m in _resolve_model_names(config) if m not in EXCLUDED_RECOMMENDERS]
    conditions = conditions if conditions is not None else resolve_conditions(config)

    cells: list[BatteryCell] = []
    seen: set[str] = set()
    for condition in conditions:
        for pc in _iter_cells(condition, config, processed_dir, embeddings_dir, model_names):
            for seed in seeds:
                role = "search" if seed == primary else "replay"
                cell = BatteryCell(pc.dataset_name, pc.embedding_name, pc.model_name, seed, role)
                if cell.key() not in seen:  # frozen/finetuned never collide, but be safe
                    seen.add(cell.key())
                    cells.append(cell)
    return cells
