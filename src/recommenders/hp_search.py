"""Hyperparameter-search dispatcher: grid vs Optuna.

Two strategies are supported, selected by
``configs/recommenders.yaml -> hp_search.strategy``:

1. ``grid`` (default) — Cartesian product over the lists declared
   under each recommender's YAML block. Identical to the historical
   behaviour of ``src/steps/train.py``.
2. ``optuna`` — Bayesian optimisation via :mod:`optuna`. Each
   recommender declares an ``hp_space:`` block with per-parameter
   ranges (``int``, ``float`` with optional ``log``, or
   ``categorical``); the dispatcher creates an Optuna study per
   ``(dataset, model, embedding)`` cell and runs ``n_trials`` per cell.

The two backends share the same per-trial entry point so the
training loop is agnostic to which strategy chose the
hyperparameters.

Pruning hook
------------
The training loop (``src.utils.training.train_single_run``) accepts
an optional ``optuna_trial`` argument.  When set, it reports the
validation metric every ``eval_every_epochs`` epochs and raises
``optuna.TrialPruned`` whenever the Optuna pruner decides the trial
is not promising.  This is the bulk of the speed-up — typically
40-60% wall-clock saved across a study.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import product
from typing import Any

from src.recommenders.registry import get_recommender_spec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CellKey:
    """Identifies one ``(dataset, model, embedding)`` triplet of the search."""

    dataset_name: str
    model_name: str
    embedding_name: str

    def study_name(self) -> str:
        return f"{self.dataset_name}__{self.model_name}__{self.embedding_name}"


def get_strategy(config: dict) -> str:
    """Return the configured search strategy (``grid`` or ``optuna``)."""
    return config.get("hp_search", {}).get("strategy", "grid")


def has_hp_space(config: dict, model_name: str) -> bool:
    """Whether ``configs/recommenders.yaml -> <model>.hp_space`` is declared."""
    block = config.get(model_name, {})
    return isinstance(block, dict) and isinstance(block.get("hp_space"), dict)


def get_hyperparam_grid(model_name: str, config: dict) -> list[dict]:
    """Cartesian product of grid-search hyperparameters for a recommender.

    Reads the model's :class:`RecommenderSpec` to decide which keys to
    include — there are no model-specific branches here.  Plugin authors
    declare their hyperparameters via ``extra_hyperparam_keys`` /
    ``uses_visual_dim`` when registering.
    """
    spec = get_recommender_spec(model_name)
    common = config.get("common", {})
    model_specific = config.get(model_name, {})

    params: dict = {
        "latent_dim": common.get("latent_dim", [64]),
        "learning_rate": common.get("learning_rate", [0.001]),
        "l2_reg": common.get("l2_reg", [0.0001]),
    }
    if spec.uses_visual_dim:
        params["visual_dim"] = common.get("visual_dim", [64])
    for key in spec.extra_hyperparam_keys:
        if key in model_specific:
            params[key] = model_specific[key]

    keys = list(params.keys())
    values = [params[k] if isinstance(params[k], list) else [params[k]] for k in keys]
    return [dict(zip(keys, combo, strict=False)) for combo in product(*values)]


def _sample_from_space(trial, name: str, entry: dict) -> Any:
    """Materialise one hyperparameter from its ``hp_space`` declaration.

    PyYAML 1.1 (the default in the ``pyyaml`` package) does *not*
    parse no-dot scientific notation such as ``1e-5`` as a float — it
    leaves the value as a string.  Coerce ``low``/``high``/``step``
    explicitly so a YAML written without explicit decimal points
    still works.
    """
    kind = entry["type"]
    if kind == "categorical":
        return trial.suggest_categorical(name, entry["choices"])
    if kind == "int":
        step = entry.get("step")
        return trial.suggest_int(
            name,
            int(entry["low"]),
            int(entry["high"]),
            step=int(step) if step is not None else 1,
            log=bool(entry.get("log", False)),
        )
    if kind == "float":
        step = entry.get("step")
        return trial.suggest_float(
            name,
            float(entry["low"]),
            float(entry["high"]),
            step=float(step) if step is not None else None,
            log=bool(entry.get("log", False)),
        )
    raise ValueError(f"Unknown hp_space type: {kind!r}")


def sample_hyperparams(trial, model_name: str, config: dict) -> dict:
    """Sample a complete hyperparameter dict for *trial* using ``hp_space``.

    Falls back to drawing from the legacy lists (``common``,
    ``<model>.<key>``) when ``hp_space`` is not declared, so plugin
    authors can opt in incrementally.
    """
    spec = get_recommender_spec(model_name)
    block = config.get(model_name, {})
    hp_space = block.get("hp_space")

    sampled: dict = {}

    if isinstance(hp_space, dict) and hp_space:
        for key, entry in hp_space.items():
            sampled[key] = _sample_from_space(trial, key, entry)
    else:
        # Fallback: sample from the grid lists as a random sampler
        # over the declared discrete grid.
        common = config.get("common", {})
        sampled["latent_dim"] = trial.suggest_categorical(
            "latent_dim",
            common.get("latent_dim", [64]),
        )
        sampled["learning_rate"] = trial.suggest_categorical(
            "learning_rate",
            common.get("learning_rate", [0.001]),
        )
        sampled["l2_reg"] = trial.suggest_categorical(
            "l2_reg",
            common.get("l2_reg", [0.0001]),
        )
        if spec.uses_visual_dim and "visual_dim" not in sampled:
            sampled["visual_dim"] = trial.suggest_categorical(
                "visual_dim",
                common.get("visual_dim", [64]),
            )
        for key in spec.extra_hyperparam_keys:
            if key in block and isinstance(block[key], list):
                sampled[key] = trial.suggest_categorical(key, block[key])

    return sampled


def build_sampler(cfg: dict, base_seed: int):
    """Materialise the Optuna sampler from the ``hp_search.optuna`` block."""
    import optuna

    name = cfg.get("sampler", "tpe")
    n_startup = cfg.get("n_startup_trials", 5)

    if name == "tpe":
        return optuna.samplers.TPESampler(
            seed=base_seed,
            n_startup_trials=n_startup,
            multivariate=True,
        )
    if name == "random":
        return optuna.samplers.RandomSampler(seed=base_seed)
    if name == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=base_seed, n_startup_trials=n_startup)
    raise ValueError(f"Unknown optuna sampler: {name!r}")


def build_pruner(cfg: dict):
    """Materialise the Optuna pruner from the ``hp_search.optuna`` block."""
    import optuna

    name = cfg.get("pruner", "median")
    if name == "median":
        return optuna.pruners.MedianPruner(n_startup_trials=cfg.get("n_startup_trials", 5))
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner()
    if name == "none":
        return optuna.pruners.NopPruner()
    raise ValueError(f"Unknown optuna pruner: {name!r}")


def create_study(
    cell: CellKey,
    config: dict,
):
    """Create (or load) the Optuna study for one cell.

    Uses ``storage`` from the YAML when set so studies survive pod
    restarts; otherwise an in-memory study is created.
    """
    import optuna

    optuna_cfg = config.get("hp_search", {}).get("optuna", {})
    base_seed = int(config.get("seed", 42))

    storage = optuna_cfg.get("storage")
    # A persistent SQLite study survives a spot-instance restart (resume).
    # sqlite creates the file but not the directory — make it exist first.
    if isinstance(storage, str) and storage.startswith("sqlite:///"):
        from pathlib import Path

        db_path = Path(storage[len("sqlite:///") :])
        if db_path.parent != Path():
            db_path.parent.mkdir(parents=True, exist_ok=True)

    return optuna.create_study(
        study_name=cell.study_name(),
        direction="maximize",
        sampler=build_sampler(optuna_cfg, base_seed),
        pruner=build_pruner(optuna_cfg),
        storage=storage,
        load_if_exists=True,
    )


def iter_cells(
    cells: list[CellKey],
    config: dict,
    objective: Callable[[CellKey, dict, Any], float],
) -> Iterator[tuple[CellKey, dict, float]]:
    """Drive the search across all cells with the configured strategy.

    Parameters
    ----------
    cells:
        The ``(dataset, model, embedding)`` cells to process.
    config:
        The merged framework config.
    objective:
        Callable invoked once per concrete hyperparameter setting.
        Receives ``(cell, hyperparams, optuna_trial_or_None)`` and
        returns the validation metric.  Receives ``None`` for the
        trial argument in grid mode; receives the live ``Trial``
        object in optuna mode so the training loop can call
        ``trial.report`` / ``trial.should_prune``.

    Yields
    ------
    ``(cell, hyperparams, metric)`` for every concrete trial that
    completed (pruned trials are skipped).
    """
    strategy = get_strategy(config)
    if strategy == "grid":
        yield from _iter_cells_grid(cells, config, objective)
    elif strategy == "optuna":
        yield from _iter_cells_optuna(cells, config, objective)
    else:
        raise ValueError(f"Unknown hp_search.strategy: {strategy!r}")


def _iter_cells_grid(
    cells: list[CellKey],
    config: dict,
    objective: Callable[[CellKey, dict, Any], float],
) -> Iterator[tuple[CellKey, dict, float]]:
    for cell in cells:
        for hp in get_hyperparam_grid(cell.model_name, config):
            metric = objective(cell, hp, None)
            yield cell, hp, metric


def _iter_cells_optuna(
    cells: list[CellKey],
    config: dict,
    objective: Callable[[CellKey, dict, Any], float],
) -> Iterator[tuple[CellKey, dict, float]]:
    import optuna

    optuna_cfg = config.get("hp_search", {}).get("optuna", {})
    n_trials = optuna_cfg.get("n_trials", 30)
    timeout = optuna_cfg.get("timeout_seconds")

    for cell in cells:
        study = create_study(cell, config)

        def _objective(trial, _cell=cell):
            hp = sample_hyperparams(trial, _cell.model_name, config)
            return objective(_cell, hp, trial)

        study.optimize(
            _objective,
            n_trials=n_trials,
            timeout=timeout,
            gc_after_trial=True,
            show_progress_bar=False,
        )

        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue
            # A completed trial always has a value; guard the None case
            # explicitly rather than with ``or`` so a legitimate 0.0
            # metric is not confused with a missing one.
            value = 0.0 if trial.value is None else float(trial.value)
            yield cell, dict(trial.params), value


__all__ = [
    "CellKey",
    "build_pruner",
    "build_sampler",
    "create_study",
    "get_hyperparam_grid",
    "get_strategy",
    "has_hp_space",
    "iter_cells",
    "sample_hyperparams",
]
