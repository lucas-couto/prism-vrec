"""Where a cell's hyperparameters come from: search results or a pinned config.

The K-fold cross-validation runner must train every fold with *one*
hyperparameter configuration per ``(dataset, model, embedding)`` cell,
and it must be able to say where that configuration came from.  Two
origins exist:

* ``fixed`` — ``hp_search.strategy: fixed``; the values are read straight
  from ``configs/recommenders.yaml`` via
  :func:`src.recommenders.hp_search.get_fixed_hyperparams`.
* ``search`` — any other strategy; the values are the winner recorded in
  ``<results_root>/best_hyperparams.json`` (written by
  :func:`src.steps.export_best.export_best_hyperparams`).  When the JSON
  is missing it is generated on the spot from the ``_best.pt``
  checkpoints under ``<results_root>/models``.

Both are returned as a :class:`HyperparamOrigin`, whose :meth:`to_dict`
is what the fold manifest records.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from src.recommenders.hp_search import get_fixed_hyperparams, get_strategy

BEST_HYPERPARAMS_FILENAME = "best_hyperparams.json"


@dataclass(frozen=True)
class HyperparamOrigin:
    """One cell's hyperparameters together with their provenance.

    Attributes
    ----------
    source:
        ``"search"`` when taken from a search run's winners,
        ``"fixed"`` when pinned in the YAML.
    hyperparams:
        The concrete configuration to train with.
    reference:
        ``"{dataset}__{model}__{embedding}"`` locating the cell inside
        ``best_hyperparams.json`` for ``search``; ``None`` for ``fixed``.
    best_metric:
        The search's validation metric for that cell; ``None`` for ``fixed``.
    """

    source: Literal["search", "fixed"]
    hyperparams: dict
    reference: str | None
    best_metric: float | None

    def to_dict(self) -> dict:
        """Plain, JSON-serialisable view (for manifests and logs)."""
        return asdict(self)


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
        return HyperparamOrigin(
            source="fixed",
            hyperparams=get_fixed_hyperparams(model_name, config),
            reference=None,
            best_metric=None,
        )

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
    return HyperparamOrigin(
        source="search",
        hyperparams=dict(entry["hyperparams"]),
        reference=reference,
        best_metric=float(entry["best_metric"]),
    )


def _load_or_export_best(results_root: Path) -> dict:
    """Read ``best_hyperparams.json``; build it from ``models/`` when absent."""
    summary_path = results_root / BEST_HYPERPARAMS_FILENAME
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as fh:
            return json.load(fh)
    from src.steps.export_best import export_best_hyperparams

    return export_best_hyperparams(results_root / "models", summary_path)


__all__ = ["BEST_HYPERPARAMS_FILENAME", "HyperparamOrigin", "resolve_cell_hyperparams"]
