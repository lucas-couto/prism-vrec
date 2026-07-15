"""Pydantic schema for the merged configuration dictionary.

The framework reads its YAML configs into a single dict (see
:func:`src.utils.config.load_config`).  This module declares the
expected shape of that dict so a typo like ``extractors_enabld``
fails immediately with a clear message instead of silently disabling
a step.

Validation policy
-----------------
- Top-level structure is strict: every recognised key has a type;
  unknown top-level keys raise ``ValidationError``.
- Plugin-specific blocks (per-extractor config, per-recommender
  hyperparameter grid, per-fusion config) are permissive, plugins
  define their own keys, so the schema only checks they are dicts /
  lists and lets each plugin handle its own validation downstream.
- Step names are validated against the canonical list defined in
  :data:`PIPELINE_STEPS`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Mirrors ``main.STEP_ORDER``, kept in sync manually because importing
# main.py here would create a circular dependency at config-load time.
PIPELINE_STEPS = (
    "download",
    "preprocess",
    "extract",
    "finetune",
    "evaluate_finetuning",
    "fuse",
    "train",
    "evaluate",
    "statistical",
    "export_best",
)

DEVICE_VALUES = ("cuda", "cpu")
CONDITION_VALUES = ("frozen", "finetuned", "both")


class PathsConfig(BaseModel):
    """Filesystem layout used by every step."""

    model_config = ConfigDict(extra="forbid")

    data_raw: str = "data/raw"
    data_processed: str = "data/processed"
    embeddings: str = "data/embeddings"
    checkpoints: str = "checkpoints"
    results: str = "results"
    logs: str = "logs"


class PreprocessingConfig(BaseModel):
    """k-core filtering parameters applied during preprocessing."""

    model_config = ConfigDict(extra="forbid")

    n_min: int = Field(5, ge=1, description="Minimum interactions per user / item.")


class DatasetContract(BaseModel):
    """Per-dataset contract enforced during preprocessing.

    ``expects_categories`` declares whether the dataset is required to
    ship item category labels.  The preprocess step compares this
    declaration against what the provider actually loads and raises when
    they disagree, so a dataset silently gaining or losing categories
    (which flips DeepStyle degeneration and fine-tuning transfer) fails
    loud instead of changing results without a trace.
    """

    model_config = ConfigDict(extra="forbid")

    expects_categories: bool


class DataLoaderConfig(BaseModel):
    """Optional manual override of the DataLoader autotune.

    All three fields are optional. When omitted, the framework picks
    the corresponding value from ``src.utils.dataloader.autotune()``
    based on the host's CPU and cgroup memory budget. Setting any
    field pins it for the whole run.
    """

    model_config = ConfigDict(extra="forbid")

    num_workers: int | None = Field(None, ge=0)
    prefetch_factor: int | None = Field(None, ge=1)
    batch_size: int | None = Field(None, ge=1)


class PipelineConfig(BaseModel):
    """Pipeline orchestration knobs consumed by ``main.py``."""

    model_config = ConfigDict(extra="forbid")

    run_all: bool = True
    start_from: str | None = None
    stop_at: str | None = None
    condition: Literal["frozen", "finetuned", "both"] = "both"

    @field_validator("start_from", "stop_at")
    @classmethod
    def _validate_step_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value not in PIPELINE_STEPS:
            raise ValueError(
                f"unknown step name {value!r}. Valid steps: {', '.join(PIPELINE_STEPS)}"
            )
        return value

    @model_validator(mode="after")
    def _check_step_range(self) -> PipelineConfig:
        if self.start_from is None or self.stop_at is None:
            return self
        start_idx = PIPELINE_STEPS.index(self.start_from)
        stop_idx = PIPELINE_STEPS.index(self.stop_at)
        if start_idx > stop_idx:
            raise ValueError(
                f"pipeline.start_from ({self.start_from}) cannot come after "
                f"pipeline.stop_at ({self.stop_at}) in the canonical step order."
            )
        return self


class EvaluationConfig(BaseModel):
    """Evaluation-protocol knobs consumed by ``src.steps.evaluate``."""

    model_config = ConfigDict(extra="forbid")

    protocol: Literal["full_ranking", "sampled"] = Field(
        "full_ranking",
        description=(
            "Ranking protocol used by the Evaluator.  ``full_ranking`` "
            "(default) scores every catalogue item per user and is the "
            "thesis-grade primary protocol.  ``sampled`` ranks each "
            "user's positives against ``n_negatives`` random unseen "
            "items — much cheaper but statistically inconsistent with "
            "full-ranking (Krichene & Rendle, KDD 2020); use only for "
            "comparability with prior work that used the same protocol."
        ),
    )
    n_negatives: int = Field(
        100,
        ge=1,
        description="Negative pool size per user when ``protocol='sampled'``.",
    )
    negative_sampling_seed: int = Field(
        42,
        ge=0,
        description=(
            "Seed for per-user negative sampling.  Identical seeds across "
            "model runs guarantee identical pools, which paired Wilcoxon "
            "tests rely on."
        ),
    )


class FineTuningConfig(BaseModel):
    """Hyperparameters for the categorical fine-tuning step."""

    model_config = ConfigDict(extra="forbid")

    epochs_max: int = Field(15, ge=1)
    learning_rate: float = Field(0.0001, gt=0.0)
    weight_decay: float = Field(0.0001, ge=0.0)
    batch_size: int = Field(128, ge=1)
    patience: int = Field(5, ge=1)
    extractors: list[str] = Field(default_factory=list)
    tradesy_transfer_from: str = "amazon_fashion"


class OptunaConfig(BaseModel):
    """Tunables for the Bayesian hyperparameter-search backend."""

    model_config = ConfigDict(extra="forbid")

    n_trials: int = Field(30, ge=1, description="Trials run per (dataset, model, embedding) cell.")
    sampler: Literal["tpe", "random", "cmaes"] = "tpe"
    pruner: Literal["median", "hyperband", "none"] = "median"
    n_startup_trials: int = Field(
        5,
        ge=0,
        description="Trials drawn at random before TPE engages its surrogate.",
    )
    warm_start: bool = Field(
        True,
        description=(
            "Re-use the best HPs from earlier cells of the same model "
            "as starting points for subsequent cells."
        ),
    )
    timeout_seconds: int | None = Field(
        None,
        ge=1,
        description="Optional wall-clock cap per cell; None means no cap.",
    )
    storage: str | None = Field(
        None,
        description=(
            "SQLAlchemy URL for trial persistence "
            "(e.g. ``sqlite:///optuna.db``).  None keeps trials in memory."
        ),
    )


class HpSearchConfig(BaseModel):
    """Top-level hyperparameter-search dispatch configuration."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["grid", "optuna"] = "grid"
    optuna: OptunaConfig = Field(default_factory=OptunaConfig)


class HpSpaceEntry(BaseModel):
    """Declaration of a single hyperparameter's search space.

    Plugin authors declare ``hp_space:`` blocks per recommender to
    enable Optuna-driven search.  Falls back gracefully to grid when
    the chosen strategy is ``grid``: each entry materialises into a
    short list (``low / midpoint / high`` for numeric, ``choices``
    for categorical).
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["int", "float", "categorical"]
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    log: bool = False
    choices: list | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> HpSpaceEntry:
        if self.type == "categorical":
            if not self.choices:
                raise ValueError("hp_space entries with type=categorical must declare 'choices'.")
            return self
        # numeric
        if self.low is None or self.high is None:
            raise ValueError(
                f"hp_space entries with type={self.type} must declare 'low' and 'high'.",
            )
        if self.low > self.high:
            raise ValueError(f"hp_space 'low' ({self.low}) is above 'high' ({self.high}).")
        if self.log and (self.low <= 0 or self.high <= 0):
            raise ValueError("hp_space with log=true requires positive low/high.")
        return self


class AlignmentConfig(BaseModel):
    """Dimensionality alignment for equal-dim fusion strategies (v2).

    ``learned`` — per-source projections co-trained with the recommender
    via BPR.  ``pca`` — offline per-source PCA (fit on train items only).
    """

    model_config = ConfigDict(extra="forbid")

    method: Literal["learned", "pca"] = "learned"
    dim: int = Field(128, ge=1)


class CommonTrainingConfig(BaseModel):
    """Shared recommender-training block (``common:`` in recommenders.yaml).

    Declares both the shared hyperparameter grid (list-valued keys
    consumed by ``get_hyperparam_grid``) and the training-loop knobs
    consumed by ``train_single_run``.  Previously this block slipped
    through ``extra='allow'`` untyped, so a typo like
    ``early_stoping_patience`` silently reverted the run to defaults —
    exactly the failure mode this schema exists to prevent.
    """

    model_config = ConfigDict(extra="forbid")

    latent_dim: list[int] | int = Field(default_factory=lambda: [64])
    learning_rate: list[float] | float = Field(default_factory=lambda: [0.001])
    l2_reg: list[float] | float = Field(default_factory=lambda: [0.0001])
    visual_dim: list[int] | int = Field(default_factory=lambda: [64])
    epochs: int = Field(100, ge=1)
    batch_size: int = Field(4096, ge=1)
    early_stopping_patience: int = Field(10, ge=1)
    early_stopping_metric: str = "ndcg@10"
    eval_every_epochs: int = Field(10, ge=1)
    eval_sample_size: int | None = Field(None, ge=1)


class FrameworkConfig(BaseModel):
    """Top-level merged config exposed to every step.

    Plugin-defined blocks (per-extractor, per-fusion, per-recommender)
    are kept loose: ``extra='allow'`` here means new plugins can
    contribute keys without editing this file, but the *known* fields
    are still type-checked and typos in them surface at load time.
    """

    model_config = ConfigDict(extra="allow")

    seed: int = Field(42, ge=0)
    seeds: list[int] | None = Field(
        None,
        description=(
            "Optional list of seeds for multi-seed runs.  When set, the "
            "pipeline executes once per seed under suffixed result/"
            "checkpoint paths (``results_seed<N>``, "
            "``checkpoints_seed<N>``) so paired statistical tests across "
            "seeds become possible.  ``seed`` remains the fallback for "
            "single-run mode and the base for seed-derivation in each "
            "training cell."
        ),
    )
    device: Literal["cuda", "cpu", "auto"] = "auto"
    paths: PathsConfig = Field(default_factory=PathsConfig)
    datasets: list[str] = Field(default_factory=list)
    dataset_contracts: dict[str, DatasetContract] = Field(
        default_factory=dict,
        description=(
            "Per-dataset category contract keyed by dataset name.  Datasets "
            "without an entry skip the check (backwards compatible)."
        ),
    )
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)

    extractors_enabled: list[str] = Field(default_factory=list)
    extractors: dict[str, dict[str, Any]] = Field(default_factory=dict)
    fusion_extractors: list[str] = Field(default_factory=list)
    batch_size: int = Field(256, ge=1)
    checkpoint_every: int = Field(500, ge=1)

    @field_validator("seeds")
    @classmethod
    def _seeds_unique_non_negative(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if not value:
            raise ValueError("seeds must be a non-empty list when provided")
        if any(not isinstance(s, int) or s < 0 for s in value):
            raise ValueError("seeds entries must be non-negative integers")
        if len(set(value)) != len(value):
            raise ValueError("seeds entries must be unique")
        return value

    fusion_strategies_enabled: list[str] = Field(default_factory=list)
    # Per-strategy YAML block, keys depend on the strategy's expand_grid.
    fusion: dict[str, Any] = Field(default_factory=dict)
    strategies: dict[str, Any] | None = None
    normalize_before_fusion: bool = True
    alignment: AlignmentConfig = Field(default_factory=lambda: AlignmentConfig())

    recommenders_enabled: list[str] = Field(default_factory=list)
    common: CommonTrainingConfig = Field(default_factory=CommonTrainingConfig)
    hp_search: HpSearchConfig = Field(default_factory=HpSearchConfig)
    hp_budget: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Optional per-dataset override of the shared HP-search budget "
            "(n_trials, early_stopping_*, epochs, eval_sample_size).  Keyed "
            "by dataset name; all recommenders of a dataset share it.  See "
            "src/recommenders/hp_budget.py."
        ),
    )

    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    k_values: list[int] = Field(
        default_factory=lambda: [5, 10, 20],
        description="Ranking cutoffs used by the evaluate and statistical steps.",
    )

    finetuning: FineTuningConfig = Field(default_factory=FineTuningConfig)


def validate_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate *raw* against :class:`FrameworkConfig`, return the dict.

    Returns the same dict (with defaults filled in) so callers can
    keep the existing ``cfg["paths"]["data_raw"]`` access pattern; the
    validation step only enforces the schema and surfaces typos.
    """
    parsed = FrameworkConfig.model_validate(raw)
    # ``model_dump`` returns plain dict / list / scalar values so no
    # Pydantic-specific objects leak into the rest of the pipeline.
    return parsed.model_dump(exclude_none=False)


__all__ = [
    "CONDITION_VALUES",
    "CommonTrainingConfig",
    "DEVICE_VALUES",
    "DatasetContract",
    "EvaluationConfig",
    "FineTuningConfig",
    "FrameworkConfig",
    "HpSearchConfig",
    "HpSpaceEntry",
    "OptunaConfig",
    "PIPELINE_STEPS",
    "PathsConfig",
    "PipelineConfig",
    "PreprocessingConfig",
    "validate_config",
]
