"""HP-search budget: one protocol budget per dataset, shared by all models.

A benchmark is only as fair as its HP search.  An unequal budget
(``n_trials``, patience, ``epochs``, selection metric, validation
subsample) between recommenders is a direct confounder of the central
comparison.  The budget therefore lives in ONE place per dataset and is
identical for every recommender of that dataset; only the search SPACES
stay per-model (legitimate — each model has its own hyperparameters).

Single source:
* ``common:``            → early_stopping_metric/patience, epochs, eval_sample_size
* ``hp_search.optuna:``  → n_trials
* optional ``hp_budget[<dataset>]`` → per-dataset overrides of the above.

:func:`resolve_hp_budget` is the ONLY reader of those keys on the
training side: ``train_single_run`` consumes its output for every entry
path (CLI single cell, grid worker, Optuna trial, battery replay, folds),
so an override reaches the epoch loop, the patience counter, the metric
lookup and the selection evaluator identically everywhere (R03).

The guard-rail refuses any per-model budget key.
"""

from __future__ import annotations

from functools import cache

#: Protocol-budget keys.  These must never appear inside a per-recommender
#: config block — the budget is shared, not per-model.
BUDGET_KEYS = (
    "n_trials",
    "early_stopping_metric",
    "early_stopping_patience",
    "epochs",
    "eval_sample_size",
)

#: Cut-offs the TRAINING-TIME selection evaluator computes.  The selection
#: metric must name one of them: a metric at another K is never produced,
#: so it can only be "read" as a missing key — and a missing key is a
#: failure, never zero (Q06).  Final evaluation uses ``k_values`` instead.
SELECTION_K_VALUES: tuple[int, ...] = (10,)


class BudgetFairnessError(RuntimeError):
    """Raised when a recommender declares its own protocol budget."""


class UnsupportedSelectionMetricError(ValueError):
    """The requested selection metric is not produced by the selection evaluator."""


@cache
def selection_metric_keys() -> tuple[str, ...]:
    """Every ``name@K`` key the selection evaluator produces.

    Derived from the metric implementation itself (an empty ranking at
    the selection cut-offs) so the accepted set can never drift from
    what :class:`~src.evaluation.protocol.Evaluator` actually returns.
    """
    from src.evaluation.metrics import compute_all_metrics

    return tuple(compute_all_metrics([], set(), list(SELECTION_K_VALUES)).keys())


def validate_selection_metric(metric: object) -> str:
    """Return *metric* when the selection evaluator produces it, else raise.

    Raises
    ------
    UnsupportedSelectionMetricError
        For a non-string, a key without ``@K``, an unknown metric name or
        a cut-off outside :data:`SELECTION_K_VALUES`.
    """
    produced = selection_metric_keys()
    if not isinstance(metric, str) or metric not in produced:
        raise UnsupportedSelectionMetricError(
            f"early_stopping_metric {metric!r} is not produced by the selection "
            f"evaluator (cut-offs {list(SELECTION_K_VALUES)}); supported: {list(produced)}. "
            "A metric that is not produced cannot drive selection."
        )
    return metric


def _bounded_int(value: object, key: str, *, minimum: int = 1) -> int:
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"budget key {key!r} must be an integer, got {value!r}.") from exc
    if isinstance(value, bool) or number < minimum:
        raise ValueError(f"budget key {key!r} must be >= {minimum}, got {value!r}.")
    return number


def resolve_hp_budget(config: dict, dataset: str) -> dict:
    """Resolve the shared protocol budget for ``dataset`` (single source).

    Every field is validated and normalised here, so a consumer never
    sees a string epoch count or a metric the evaluator will not produce.

    Raises
    ------
    UnsupportedSelectionMetricError
        When the resolved ``early_stopping_metric`` is not produced at the
        selection cut-offs.
    ValueError
        When ``early_stopping_patience`` / ``n_trials`` / ``eval_sample_size``
        (if set) are not integers >= 1, or ``epochs`` is not an integer >= 0.
    """
    common = config.get("common", {})
    optuna = config.get("hp_search", {}).get("optuna", {})
    budget: dict = {
        "n_trials": optuna.get("n_trials", 30),
        "early_stopping_metric": common.get("early_stopping_metric", "ndcg@10"),
        "early_stopping_patience": common.get("early_stopping_patience", 10),
        "epochs": common.get("epochs", 100),
        "eval_sample_size": common.get("eval_sample_size"),
    }
    override = config.get("hp_budget", {}).get(dataset, {})
    for key in budget:
        if key in override:
            budget[key] = override[key]
    for key in ("n_trials", "early_stopping_patience"):
        budget[key] = _bounded_int(budget[key], key)
    # ``epochs: 0`` is a legal (empty) schedule: the training loop then
    # fails explicitly for producing no validation observation (I01).
    budget["epochs"] = _bounded_int(budget["epochs"], "epochs", minimum=0)
    if budget["eval_sample_size"] is not None:
        budget["eval_sample_size"] = _bounded_int(budget["eval_sample_size"], "eval_sample_size")
    budget["early_stopping_metric"] = validate_selection_metric(budget["early_stopping_metric"])
    return budget


def assert_uniform_budget(config: dict) -> None:
    """Fail loud if any enabled recommender declares a budget key.

    Guarantees that every cell of a dataset runs with the same budget:
    the budget can only come from the shared blocks, never a per-model
    override that would confound the comparison.
    """
    for model in config.get("recommenders_enabled", []):
        block = config.get(model, {})
        if not isinstance(block, dict):
            continue
        offending = [k for k in BUDGET_KEYS if k in block]
        if offending:
            raise BudgetFairnessError(
                f"recommender {model!r} declares protocol-budget key(s) "
                f"{offending}; the HP-search budget is shared per dataset "
                f"(configs common:/hp_search:/hp_budget:), never per model."
            )


def grid_budget_message(grid_sizes: dict[str, int]) -> str | None:
    """Warning text when grid budgets differ across models, else ``None``.

    Under ``hp_search.strategy: grid`` each model gets one selection shot
    per configuration in its declared space, so unequal space sizes are
    unequal selection budgets — a model with a larger grid gets more
    chances to look good on validation (audit D1).  The spaces themselves
    are legitimate per-model choices, so this cannot be equalised
    silently; it is surfaced loudly instead.  The Optuna/battery path is
    the equal-budget protocol (one shared ``n_trials`` per dataset).

    Args:
        grid_sizes: Mapping ``model_name -> number of grid configs``,
            covering every enabled recommender.

    Returns:
        The warning message, or ``None`` when all budgets are equal (or
        fewer than two models are enabled).
    """
    if len(grid_sizes) < 2 or len(set(grid_sizes.values())) <= 1:
        return None
    listing = ", ".join(f"{model}={n}" for model, n in sorted(grid_sizes.items()))
    return (
        "UNEQUAL GRID-SEARCH BUDGETS: configs per model: "
        f"{listing}. Under strategy 'grid' every configuration is one "
        "selection shot, so models with larger search spaces are favoured "
        "in the comparison. For an equal-budget protocol use the "
        "Optuna/battery path (hp_search.strategy: optuna), which gives "
        "every cell the same shared n_trials."
    )
