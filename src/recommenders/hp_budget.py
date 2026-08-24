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

The guard-rail refuses any per-model budget key.
"""

from __future__ import annotations

#: Protocol-budget keys.  These must never appear inside a per-recommender
#: config block — the budget is shared, not per-model.
BUDGET_KEYS = (
    "n_trials",
    "early_stopping_metric",
    "early_stopping_patience",
    "epochs",
    "eval_sample_size",
)


class BudgetFairnessError(RuntimeError):
    """Raised when a recommender declares its own protocol budget."""


def resolve_hp_budget(config: dict, dataset: str) -> dict:
    """Resolve the shared protocol budget for ``dataset`` (single source)."""
    common = config.get("common", {})
    optuna = config.get("hp_search", {}).get("optuna", {})
    budget = {
        "n_trials": int(optuna.get("n_trials", 30)),
        "early_stopping_metric": common.get("early_stopping_metric", "ndcg@10"),
        "early_stopping_patience": int(common.get("early_stopping_patience", 10)),
        "epochs": int(common.get("epochs", 100)),
        "eval_sample_size": common.get("eval_sample_size"),
    }
    override = config.get("hp_budget", {}).get(dataset, {})
    for key in budget:
        if key in override:
            budget[key] = override[key]
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
