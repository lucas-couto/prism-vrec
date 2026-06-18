# Hyperparameter search: grid vs Optuna

The recommender training step (`05_train`) supports two backends:

| Strategy | When to use | Where it shines |
| --- | --- | --- |
| `grid` (default) | Small HP grids, parallel-friendly hardware | Predictable, embarrassingly parallel, exhaustive |
| `optuna`         | Larger HP spaces, time-bounded budgets   | Bayesian sampling + median pruning saves 40-60% wall-clock |

Pick the backend in `configs/recommenders.yaml`:

```yaml
hp_search:
  strategy: "optuna"   # or "grid"
```

CLI override (one-off, does not edit the YAML):

```bash
python main.py --hp-search optuna --n-trials 30
python main.py --hp-search grid
```

---

## Grid search (legacy)

Each recommender block in `configs/recommenders.yaml` declares lists of
values; the framework runs the **Cartesian product** of every list.

```yaml
common:
  latent_dim: [64, 128]
  learning_rate: [0.001, 0.01]
  l2_reg: [0.0001, 0.001]

vbpr:
  hp_space: { ... }   # ignored when strategy=grid

avbpr:
  att_hidden: [64, 128]   # extra_hyperparam_keys for AVBPR

acf:
  att_hidden: [64, 128]   # extra_hyperparam_keys for ACF
  max_history: [50]       # H: items per user profile (item-level attention)
```

Jobs are distributed across workers via `TrainingOrchestrator`; every
combination produces one row in the final results table.

---

## Optuna search

Optuna performs **Bayesian optimisation** over the hyperparameter
space and **prunes** trials that look hopeless after a few epochs,
re-using the GPU budget on promising regions.

### Declaring search spaces

Add an `hp_space` block under each recommender you want optimised
with Optuna:

```yaml
vbpr:
  hp_space:
    latent_dim:    { type: int,         low: 8,    high: 256, log: true }
    learning_rate: { type: float,       low: 1e-5, high: 1e-1, log: true }
    l2_reg:        { type: float,       low: 1e-7, high: 1e-2, log: true }
    visual_dim:    { type: categorical, choices: [64, 128, 256] }
```

Supported entry types:

| `type`        | Required keys                          | Optional keys     |
| ------------- | -------------------------------------- | ----------------- |
| `int`         | `low`, `high`                          | `step`, `log`     |
| `float`       | `low`, `high`                          | `step`, `log`     |
| `categorical` | `choices`                              | —                 |

When `log: true`, Optuna samples on a log-uniform scale (recommended
for `learning_rate`, `l2_reg` and any quantity spanning multiple
orders of magnitude).

### Without `hp_space` declaration

Recommenders that have no `hp_space:` block fall back to **sampling
from the legacy lists** (`common.latent_dim`, `common.learning_rate`,
etc.) as if Optuna were a smarter random sampler over the discrete
grid.  Useful as an opt-in migration path.

### Pruning — the actual speed-up

Optuna's `MedianPruner` (default) compares each trial's intermediate
metric against the median of past trials at the same epoch.  Trials
below the median are stopped immediately, freeing the GPU for the next
suggestion.  In practice this kills 40-60% of trials within the first
few epochs — overall wall-clock typically drops by **~50%** vs running
all trials to completion.

Pruners available via `hp_search.optuna.pruner`:

| Value        | Behaviour                                       |
| ------------ | ----------------------------------------------- |
| `median`     | Stop trials below the median at each step.       |
| `hyperband`  | Successive halving across configurations.        |
| `none`       | Disable pruning (full training for every trial). |

### Persistence and resume

By default Optuna keeps studies in memory — they are lost on process
exit.  For long-running pods set `storage` to a SQLAlchemy URL so
trials are persisted:

```yaml
hp_search:
  optuna:
    storage: "sqlite:///optuna.db"
```

Re-running with the same `study_name` (auto-derived from
`(dataset, model, embedding)`) **resumes** the search from the
already-completed trials.  Bonus: install `optuna-dashboard` and run
`optuna-dashboard sqlite:///optuna.db` to inspect convergence and HP
importance interactively — useful figures for theses.

### Parallelism

In the current MVP, Optuna runs **sequentially within each cell**
(within a `(dataset, model, embedding)` triplet).  Different cells
are processed back-to-back rather than in parallel.

This is intentional: Bayesian sampling needs the previous trial's
result to choose the next sample.  Running multiple trials in parallel
inside the same cell weakens the surrogate signal.

If you need parallelism inside a single cell — e.g. you have multiple
GPUs and want concurrent trials — set
`storage: "sqlite:///optuna.db"` and run two pods pointing at the
same DB; Optuna handles concurrent trials safely via the storage
backend.

---

## Choosing a strategy

| Constraint | Recommendation |
| --- | --- |
| Time-bounded run (e.g. cloud credits running out) | `optuna` with low `n_trials` (15-20) — beats grid in same budget. |
| Reproducibility-first benchmark, no hp tuning | `grid` — exhaustive, no surrogate noise. |
| Large HP space (4+ dimensions) | `optuna` — grid blows up combinatorially. |
| Tiny HP space (≤ 2 dimensions, ≤ 6 combos) | `grid` — Optuna overhead not worth it. |
| Uncertain about ranges | `optuna` with broad log-uniform ranges. |

---

## Inspecting a study

After a run with `storage: "sqlite:///optuna.db"`:

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///optuna.db
# opens http://127.0.0.1:8080
```

The dashboard shows convergence curves, HP importance (Sobol-style),
and parallel coordinates — every figure exportable as PNG/SVG for
inclusion in papers / theses.
