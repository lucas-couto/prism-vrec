"""Production per-cell executor for the battery runner (Task I).

Runs ONE cell end-to-end, reusing the existing, battle-tested per-cell
functions rather than reimplementing training/evaluation.  The cell's
role and ``hp_search.strategy`` decide the route (R01, Q16):

* ``fixed`` — both roles train the pinned configuration once via
  ``train_replay``; no study is created or read;
* ``grid`` search cell (primary seed) — every declared grid point is
  trained via ``train_replay`` under the primary seed and the winner is
  the best validation metric (``_best.pt`` promotion); Optuna is never
  called;
* ``optuna`` search cell — the configured completed/pruned trial
  workflow (``_optimize_one_cell``), which must complete at least one
  trial;
* replay cell (other seeds) — the primary seed's selected EFFECTIVE
  configuration (:func:`src.recommenders.hp_source.resolve_replay_hyperparams`)
  is retrained under the replay seed in its own results/checkpoint
  namespace, with no study created;
* then a single-cell final evaluation reads THAT seed's ``_best.pt`` and
  writes the per-user artifact (F).

Every returned dict carries ``strategy``, the resolved dataset
``budget`` and, where a configuration was chosen, ``hyperparam_origin``.

The battery runner supplies idempotency/resume/manifest around this.
NOTE: designed to be driven by :func:`src.battery.runner.run_battery`.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import TYPE_CHECKING

from src.battery.cells import NO_VISUAL, BatteryCell, resolve_seeds
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.recommenders.hp_search import CellKey
    from src.recommenders.hp_source import HyperparamOrigin

logger = get_logger(__name__)


class SearchOutcomeError(RuntimeError):
    """A search cell finished without a usable winner (no COMPLETE trial)."""


def _dims(processed_dir: str, dataset: str) -> tuple[int, int]:
    base = Path(processed_dir) / dataset
    with open(base / "user2idx.json") as fh:
        n_users = len(json.load(fh))
    with open(base / "item2idx.json") as fh:
        n_items = len(json.load(fh))
    return n_users, n_items


def _embedding_path(embeddings_dir: str, dataset: str, visual_config: str) -> str | None:
    if visual_config == NO_VISUAL:
        return None
    base = Path(embeddings_dir) / dataset
    npy = base / f"{visual_config}.npy"
    if npy.exists():
        return str(npy)
    sidecar = base / f"{visual_config}.json"
    return str(sidecar) if sidecar.exists() else None


def _seed_config(config: dict, seed: int) -> dict:
    """Deep copy of *config* pinned to *seed* with seed-isolated roots.

    The best-model checkpoint and the resume checkpoints are isolated per
    seed: the study is shared across seeds (the primary seed's search
    supplies the winner), but each seed's TRAINED model must be the one
    evaluated — a shared ``_best.pt`` would let a replay whose validation
    metric is below the search's silently keep the search seed's weights,
    and a shared resume root would let one seed's envelope collide with
    another's.  The F artifact still lands in the shared, seed-keyed
    ``paths.results`` of the caller.
    """
    cfg = copy.deepcopy(config)
    cfg["seed"] = int(seed)
    paths = cfg["paths"]
    cfg["paths"] = {
        **paths,
        "results": f"{paths['results']}_seed{seed}",
        "checkpoints": f"{paths.get('checkpoints', 'checkpoints')}_seed{seed}",
    }
    return cfg


def execute_cell(cell: BatteryCell, config: dict) -> dict:
    """Run one battery cell: search|replay -> final evaluation (F artifact)."""
    from src.recommenders.hp_budget import resolve_hp_budget
    from src.recommenders.hp_search import CellKey, get_strategy
    from src.utils.device import resolve_device

    strategy = get_strategy(config)
    budget = resolve_hp_budget(config, cell.dataset)
    cfg = _seed_config(config, cell.seed)
    base_results = config["paths"]["results"]
    processed_dir = cfg["paths"]["data_processed"]
    device = resolve_device(cfg["device"])
    n_users, n_items = _dims(processed_dir, cell.dataset)
    emb_path = _embedding_path(cfg["paths"]["embeddings"], cell.dataset, cell.visual_config)
    ck = CellKey(cell.dataset, cell.recommender, cell.visual_config)
    train_kwargs = {
        "cell": ck,
        "n_users": n_users,
        "n_items": n_items,
        "embeddings_path": emb_path,
        "processed_dir": processed_dir,
        "device": device,
        "config": cfg,
    }
    result: dict = {"seed": cell.seed, "role": cell.role, "strategy": strategy, "budget": budget}

    if strategy == "fixed":
        origin = _fixed_origin(cfg, ck)
        _train(origin.hyperparams, train_kwargs)
        result["hyperparam_origin"] = origin.to_dict()
    elif cell.role == "search" and strategy == "grid":
        result.update(_search_grid(ck, cfg, train_kwargs))
    elif cell.role == "search":
        result["search"] = _search_optuna(
            ck, cfg, n_users, n_items, emb_path, processed_dir, device
        )
    else:
        origin = _replay_origin(config, ck, base_results)
        _train(origin.hyperparams, train_kwargs)
        result["hyperparam_origin"] = origin.to_dict()

    _evaluate_one_cell(cell, cfg, n_users, n_items, emb_path, device, f_out_dir=base_results)
    return result


def _train(hyperparams: dict, train_kwargs: dict) -> float:
    """One training of *hyperparams* through the shared replay entry point."""
    from src.steps.train import train_replay

    return train_replay(hyperparams=hyperparams, **train_kwargs)


def _search_grid(ck: CellKey, cfg: dict, train_kwargs: dict) -> dict:
    """Grid search of one cell under the primary seed: no study, no Optuna.

    Every declared combination is trained once; selection by validation
    happens in ``_save_best_model`` (strictly better metric replaces the
    on-disk ``_best.pt``, ties keep the earlier point), which is exactly
    the rule the ``max`` below mirrors for the returned summary.
    """
    from src.recommenders.hp_search import get_hyperparam_grid

    _warn_grid_budget_once(cfg)
    grid = get_hyperparam_grid(ck.model_name, cfg)
    logger.info("Grid search cell %s: %d configuration(s)", ck.study_name(), len(grid))
    outcomes = []
    for index, hp in enumerate(grid):
        metric = _train(hp, train_kwargs)
        logger.info("  grid point %d/%d %s: val metric=%.4f", index + 1, len(grid), hp, metric)
        outcomes.append({"hyperparams": hp, "best_metric": float(metric)})
    best = max(outcomes, key=lambda o: o["best_metric"])
    return {"n_configs": len(grid), "grid": outcomes, "best_metric": best["best_metric"]}


def _search_optuna(
    ck: CellKey,
    cfg: dict,
    n_users: int,
    n_items: int,
    emb_path: str | None,
    processed_dir: str,
    device: str,
) -> dict:
    """Optuna search of one cell; a study with no COMPLETE trial is a failure."""
    from src.steps.train import _optimize_one_cell

    summary = _optimize_one_cell(
        ck, n_users, n_items, emb_path, config=cfg, processed_dir=processed_dir, device=device
    )
    if int(summary.get("completed", 0)) < 1:
        raise SearchOutcomeError(
            f"search cell {ck.study_name()!r} has no COMPLETE trial "
            f"({summary.get('pruned', 0)} pruned); no winner exists to evaluate or replay."
        )
    return dict(summary)


def _replay_origin(config: dict, ck: CellKey, base_results: str) -> HyperparamOrigin:
    """The primary seed's selected effective configuration for a replay."""
    from src.recommenders.hp_source import resolve_replay_hyperparams

    primary = resolve_seeds(config)[0]
    return resolve_replay_hyperparams(
        config,
        dataset=ck.dataset_name,
        model_name=ck.model_name,
        embedding_name=ck.embedding_name,
        search_results_root=Path(f"{base_results}_seed{primary}"),
        search_seed=primary,
    )


def _fixed_origin(cfg: dict, ck: CellKey) -> HyperparamOrigin:
    """Pinned hyperparameters of *ck* under ``hp_search.strategy: fixed``."""
    from src.recommenders.hp_source import resolve_cell_hyperparams

    return resolve_cell_hyperparams(
        cfg,
        dataset=ck.dataset_name,
        model_name=ck.model_name,
        embedding_name=ck.embedding_name,
        results_root=Path(cfg["paths"]["results"]),
    )


_GRID_BUDGET_WARNED: set[str] = set()


def _warn_grid_budget_once(cfg: dict) -> None:
    """Log the unequal-grid fairness warning once per distinct message (D1)."""
    from src.recommenders.hp_budget import grid_budget_message
    from src.recommenders.hp_search import get_hyperparam_grid
    from src.steps.train import _resolve_model_names

    sizes = {name: len(get_hyperparam_grid(name, cfg)) for name in _resolve_model_names(cfg)}
    message = grid_budget_message(sizes)
    if message and message not in _GRID_BUDGET_WARNED:
        _GRID_BUDGET_WARNED.add(message)
        logger.warning(message)


def _reset_grid_budget_warning_for_tests() -> None:
    _GRID_BUDGET_WARNED.clear()


def _evaluate_one_cell(
    cell: BatteryCell,
    cfg: dict,
    n_users: int,
    n_items: int,
    emb_path: str | None,
    device: str,
    *,
    f_out_dir: str,
) -> None:
    """Final full-ranking evaluation for one cell -> per-user artifact (F).

    Reads the best checkpoint from the seed-isolated results dir
    (``cfg['paths']['results']``); writes the F artifact to the shared,
    seed-keyed ``f_out_dir`` so all seeds land in one per_user directory.
    """
    from src.steps.evaluate import _evaluate_cell, build_evaluator, find_best_models, load_data

    _, _, seen_inter, test_inter, train_only = load_data(
        cfg["paths"]["data_processed"], cell.dataset
    )
    # Shared construction path with src.steps.evaluate.run: honours the
    # ``evaluation:`` block (protocol / n_negatives / negative_sampling_seed)
    # with identical defaults.  tiebreak_seed = the run's active seed:
    # ``_seed_config`` sets ``cfg['seed'] = cell.seed`` (matching the
    # per-seed results directories), so each seed's cells share one
    # tie-break permutation.
    evaluator = build_evaluator(cfg, seen_inter, test_inter, n_items)
    models = [
        m
        for m in find_best_models(cell.dataset, results_dir=cfg["paths"]["results"])
        if m["model_name"] == cell.recommender and m["embedding_name"] == cell.visual_config
    ]
    if not models:
        raise RuntimeError(f"no best checkpoint found for cell {cell.key()} after training.")
    _evaluate_cell(
        models[0],
        cell.dataset,
        n_users,
        n_items,
        evaluator,
        cfg["paths"]["embeddings"],
        device,
        train_interactions=train_only,
        per_user_out_dir=f_out_dir,
        seed=int(cell.seed),
    )
