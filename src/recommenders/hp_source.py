"""Where a cell's hyperparameters come from: search results or a pinned config.

The K-fold cross-validation runner and the battery's replay seeds must
train every run with *one* effective hyperparameter configuration per
``(dataset, model, embedding)`` cell, and they must be able to say where
that configuration came from.  Two origins exist:

* ``fixed`` — ``hp_search.strategy: fixed``; the values are read straight
  from ``configs/recommenders.yaml`` via
  :func:`src.recommenders.hp_search.get_fixed_hyperparams`.
* ``search`` — any other strategy; the values are the winner of the
  search.  The folds runner reads ``<results_root>/best_hyperparams.json``
  (written by :func:`src.steps.export_best.export_best_hyperparams`,
  generated on the spot from the ``_best.pt`` checkpoints when missing);
  the battery replay reads the primary seed's ``_best.pt`` directly and,
  under Optuna, cross-checks it against the persisted study
  (:func:`resolve_replay_hyperparams`).

Whatever the origin, the returned configuration went through the ONE
canonical expansion (:func:`src.recommenders.hp_search.effective_hyperparams`)
the search itself trained with, so no consumer double-expands or
overrides a per-paper dimension split (R02).  Both origins are returned
as a :class:`HyperparamOrigin`, whose :meth:`to_dict` is what the fold
and battery manifests record.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from src.recommenders.hp_search import (
    CellKey,
    StudyNotFoundError,
    effective_hyperparams,
    get_fixed_hyperparams,
    get_strategy,
    load_existing_study,
)
from src.utils.checkpoint import BestCheckpointError, load_best_checkpoint
from src.utils.logging import get_logger

logger = get_logger(__name__)

BEST_HYPERPARAMS_FILENAME = "best_hyperparams.json"

#: Keys :func:`effective_hyperparams` derives from ``total_dim``.
_DERIVED_KEYS = ("latent_dim", "visual_dim")


class WinnerResolutionError(RuntimeError):
    """A replay cannot establish the search winner it must reproduce.

    Raised for a missing winner artifact, an absent study, a study without
    a COMPLETE trial, or a study whose best trial disagrees with the
    checkpoint the primary seed evaluated.  Never guessed around.
    """


@dataclass(frozen=True)
class HyperparamOrigin:
    """One cell's hyperparameters together with their provenance.

    Attributes
    ----------
    source:
        ``"search"`` when taken from a search run's winners,
        ``"fixed"`` when pinned in the YAML.
    hyperparams:
        The effective configuration to train with (expanded dimensions,
        pinned defaults filled in).
    reference:
        ``"{dataset}__{model}__{embedding}"`` locating the cell inside
        ``best_hyperparams.json`` / the models directory for ``search``;
        ``None`` for ``fixed``.
    best_metric:
        The search's validation metric for that cell; ``None`` for ``fixed``.
    suggestion:
        The raw choice the strategy made (e.g. ``total_dim`` before the
        split), when known; ``None`` for legacy callers.
    provenance:
        Free-form, JSON-serialisable record of where the winner was read
        from (strategy, search seed, study / trial identity, per-key
        sources); ``None`` for legacy callers.
    """

    source: Literal["search", "fixed"]
    hyperparams: dict
    reference: str | None
    best_metric: float | None
    suggestion: dict | None = None
    provenance: dict | None = None

    def to_dict(self) -> dict:
        """Plain, JSON-serialisable view (for manifests and logs)."""
        return asdict(self)


def hyperparam_sources(suggestion: dict, effective: dict) -> dict[str, str]:
    """Label every effective key as ``suggested``, ``derived`` or ``default``."""
    derived = set(_DERIVED_KEYS) if "total_dim" in effective else set()

    def _label(key: str) -> str:
        if key in suggestion:
            return "suggested"
        return "derived" if key in derived else "default"

    return {key: _label(key) for key in effective}


def _suggestion_of(effective: dict) -> dict:
    """The non-derived part of an effective configuration (grid points, winners)."""
    if "total_dim" not in effective:
        return dict(effective)
    return {k: v for k, v in effective.items() if k not in _DERIVED_KEYS}


def resolve_cell_hyperparams(
    config: dict,
    *,
    dataset: str,
    model_name: str,
    embedding_name: str,
    results_root: Path,
) -> HyperparamOrigin:
    """Resolve the hyperparameters one cell must be trained with.

    Parameters
    ----------
    config:
        The merged run configuration (``hp_search`` block + recommender
        blocks).
    dataset, model_name, embedding_name:
        The cell.
    results_root:
        The search run's ``paths.results`` directory, holding
        ``best_hyperparams.json`` (or ``models/`` to generate it from).

    Returns
    -------
    HyperparamOrigin
        ``fixed`` under ``hp_search.strategy: fixed``; ``search`` otherwise.

    Raises
    ------
    FixedHyperparamsError
        Under ``fixed`` when a key still declares several values.
    KeyError
        Under ``search`` when the cell is absent from the winners file.
    """
    if get_strategy(config) == "fixed":
        return _fixed_origin(config, model_name)

    summary = _load_or_export_best(Path(results_root))
    reference = f"{dataset}__{model_name}__{embedding_name}"
    try:
        entry = summary[dataset][model_name][embedding_name]
    except KeyError as exc:
        raise KeyError(
            f"cell {reference!r} not found in "
            f"{Path(results_root) / BEST_HYPERPARAMS_FILENAME}; the search run has no "
            "winner for it (missing _best.pt?) or the cell name differs."
        ) from exc
    recorded = dict(entry["hyperparams"])
    effective = effective_hyperparams(model_name, recorded, config)
    suggestion = _suggestion_of(recorded)
    return HyperparamOrigin(
        source="search",
        hyperparams=effective,
        reference=reference,
        best_metric=float(entry["best_metric"]),
        suggestion=suggestion,
        provenance={
            "strategy": get_strategy(config),
            "winners_file": str(Path(results_root) / BEST_HYPERPARAMS_FILENAME),
            "sources": hyperparam_sources(suggestion, effective),
        },
    )


def _fixed_origin(config: dict, model_name: str) -> HyperparamOrigin:
    effective = get_fixed_hyperparams(model_name, config)
    suggestion = _suggestion_of(effective)
    return HyperparamOrigin(
        source="fixed",
        hyperparams=effective,
        reference=None,
        best_metric=None,
        suggestion=suggestion,
        provenance={"strategy": "fixed", "sources": hyperparam_sources(suggestion, effective)},
    )


def resolve_replay_hyperparams(
    config: dict,
    *,
    dataset: str,
    model_name: str,
    embedding_name: str,
    search_results_root: Path,
    search_seed: int,
) -> HyperparamOrigin:
    """Resolve the effective configuration a replay seed must reproduce.

    The winner is the ``_best.pt`` the primary seed promoted and later
    evaluated (``<search_results_root>/models/<dataset>/<model>_<embedding>_best.pt``);
    its recorded hyperparameters are re-passed through the canonical
    expansion (idempotent) so a legacy winner missing a pinned default is
    completed and a complete one is untouched.  Under ``optuna`` the
    persisted study is loaded (never created) and must hold at least one
    COMPLETE trial whose expanded ``params`` equal the checkpoint's
    configuration — the study and the artifact are two views of one
    selection and may not disagree.

    Raises
    ------
    WinnerResolutionError
        Missing winner artifact, unreadable artifact, absent study, a
        study with no COMPLETE trial, or a study/artifact disagreement.
    """
    if get_strategy(config) == "fixed":
        return _fixed_origin(config, model_name)

    reference = f"{dataset}__{model_name}__{embedding_name}"
    winner_path = (
        Path(search_results_root) / "models" / dataset / f"{model_name}_{embedding_name}_best.pt"
    )
    try:
        payload = load_best_checkpoint(winner_path)
    except BestCheckpointError as exc:
        raise WinnerResolutionError(
            f"replay of {reference!r} (search seed {search_seed}) has no usable winner: {exc}"
        ) from exc
    recorded = dict(payload["hyperparams"])
    effective = effective_hyperparams(model_name, recorded, config)
    best_metric = float(payload["best_metric"])
    provenance: dict = {
        "strategy": get_strategy(config),
        "search_seed": int(search_seed),
        "winner_artifact": str(winner_path),
    }
    suggestion = _suggestion_of(recorded)
    if provenance["strategy"] == "optuna":
        suggestion, study_meta = _optuna_winner(
            config, CellKey(dataset, model_name, embedding_name), effective, best_metric
        )
        provenance.update(study_meta)
    provenance["sources"] = hyperparam_sources(suggestion, effective)
    return HyperparamOrigin(
        source="search",
        hyperparams=effective,
        reference=reference,
        best_metric=best_metric,
        suggestion=suggestion,
        provenance=provenance,
    )


def _optuna_winner(
    config: dict, cell: CellKey, effective: dict, best_metric: float
) -> tuple[dict, dict]:
    """Cross-check the winner artifact against the persisted study."""
    import optuna

    try:
        study = load_existing_study(cell, config)
    except StudyNotFoundError as exc:
        raise WinnerResolutionError(str(exc)) from exc
    states = [t.state for t in study.trials]
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise WinnerResolutionError(
            f"study {cell.study_name()!r} has no COMPLETE trial "
            f"({len(states)} trial(s): {sorted({s.name for s in states})}); nothing to replay."
        )
    best = study.best_trial
    from_study = effective_hyperparams(cell.model_name, dict(best.params), config)
    if from_study != effective:
        if float(best.value) != best_metric:
            raise WinnerResolutionError(
                f"study {cell.study_name()!r} best trial #{best.number} "
                f"(value={best.value!r}, params={best.params}) and the winner artifact "
                f"(best_metric={best_metric!r}, hyperparams={effective}) disagree; the "
                "search must be re-run before it can be replayed."
            )
        logger.warning(
            "study %r best trial #%d ties the winner artifact at %r with a different "
            "configuration; replaying the artifact's (the one the primary seed evaluated).",
            cell.study_name(),
            best.number,
            best_metric,
        )
    meta = {
        "study": cell.study_name(),
        "best_trial": int(best.number),
        "best_value": float(best.value),
        "n_completed": len(completed),
        "n_pruned": sum(1 for s in states if s == optuna.trial.TrialState.PRUNED),
        "n_failed": sum(1 for s in states if s == optuna.trial.TrialState.FAIL),
    }
    return dict(best.params), meta


def _load_or_export_best(results_root: Path) -> dict:
    """Read ``best_hyperparams.json``; build it from ``models/`` when absent."""
    summary_path = results_root / BEST_HYPERPARAMS_FILENAME
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as fh:
            return json.load(fh)
    from src.steps.export_best import export_best_hyperparams

    return export_best_hyperparams(results_root / "models", summary_path)


__all__ = [
    "BEST_HYPERPARAMS_FILENAME",
    "HyperparamOrigin",
    "WinnerResolutionError",
    "hyperparam_sources",
    "resolve_cell_hyperparams",
    "resolve_replay_hyperparams",
]
