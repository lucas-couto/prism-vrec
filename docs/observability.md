# Observability

Every `main.py` invocation produces a self-describing snapshot of the run
under `results/runs/<run_id>/`. Two files live there:

| File | Purpose |
|---|---|
| `manifest.json` | Reproducibility: git SHA, seed, hardware, package versions, config snapshot, per-step wall-time, DataLoader tier. One file per run. |
| `step_timings.json` | Profiling: per-cell wall-time for the expensive steps (extract, finetune, evaluate_finetuning, evaluate). One file per run. |

Both are gitignored. Archive them alongside published results (for
example on Zenodo with a DOI) so a future reader can reconstruct the
run.

## Device resolution

The `device:` field in `configs/default.yaml` accepts:

- `auto` (default): pick `cuda` when a GPU is visible, otherwise `cpu`.
- `cuda`: request a GPU. Falls back to `cpu` with a warning when no GPU
  is detected, so a misconfigured host does not crash.
- `cpu`: force CPU even on a GPU host.

The resolved value is recorded in the manifest:

```json
"device": {
  "requested": "auto",
  "resolved": "cuda"
}
```

## DataLoader autotune

PyTorch's `DataLoader` exposes three knobs (`num_workers`,
`prefetch_factor`, `batch_size`) that interact with the host's CPU
count and cgroup memory budget. A value that maximises throughput on a
128 GB lab box will OOM-kill the worker pool on a 16 GB laptop
container, so a single hardcoded default does not work for every
deployment.

`src/utils/dataloader.py` picks a tier at startup based on what the
container can see:

| Memory budget | num_workers | prefetch_factor | batch_size |
|---|---|---|---|
| `< 8 GB` (laptop / CI) | `min(2, cpu-1)` | 2 | 32 |
| `8-32 GB` (mid-tier) | `min(4, cpu-1)` | 4 | 128 |
| `>= 32 GB` (lab / pod) | `min(12, cpu-1)` | 8 | 256 |

The budget is read in this order: cgroup v2
(`/sys/fs/cgroup/memory.max`), cgroup v1
(`/sys/fs/cgroup/memory/memory.limit_in_bytes`), then total host RAM
via `os.sysconf`. The cgroup "no limit" sentinel (close to `2**63`) is
ignored. One core is reserved for the trainer (`cpu - 1`) and
`num_workers` is floored at 1.

Researchers do not have to set these knobs. When they want to, the
override lives in `configs/default.yaml -> dataloader` (see the
commented-out block at the end of the file). Pinned values win over
the autotune; fields left commented fall through to the tier:

```yaml
# configs/default.yaml
dataloader:
  num_workers: 4
  prefetch_factor: 4
  batch_size: 64
```

Every choice is reproducible from the manifest:

```json
"dataloader_autotune": {
  "cpu_count": 16,
  "memory_budget_gb": 45.0,
  "tier": "loose (>=32 GB)",
  "auto": {"num_workers": 12, "prefetch_factor": 8, "batch_size": 256},
  "resolved": {"num_workers": 4, "prefetch_factor": 4, "batch_size": 64},
  "yaml_overrides": {"num_workers": 4, "prefetch_factor": 4, "batch_size": 64}
}
```

`auto` records the values the autotune would have picked, `resolved`
records the values actually used, and `yaml_overrides` lists the keys
the YAML pinned. There are no environment variable overrides, the YAML
is the single source of truth so reruns reproduce from `git checkout`
alone.

## Per-step wall-time

`manifest.json` carries a `steps` list with one entry per
`_run_step` invocation in `main.py`:

```json
"steps": [
  {"name": "download",             "started_at": "2026-05-14T12:00:00Z", "duration_seconds": 12.3},
  {"name": "preprocess",           "started_at": "2026-05-14T12:00:12Z", "duration_seconds": 82.5},
  {"name": "extract",              "started_at": "2026-05-14T12:01:35Z", "duration_seconds": 2810.0},
  {"name": "finetune",             "started_at": "2026-05-14T12:48:25Z", "duration_seconds": 1820.5},
  {"name": "evaluate_finetuning",  "started_at": "2026-05-14T13:18:46Z", "duration_seconds": 30.2},
  {"name": "fuse (frozen)",        "started_at": "2026-05-14T13:19:16Z", "duration_seconds": 5.1},
  {"name": "fuse (finetuned)",     "started_at": "2026-05-14T13:19:21Z", "duration_seconds": 5.0},
  {"name": "train (frozen)",       "started_at": "2026-05-14T13:19:26Z", "duration_seconds": 7200.0},
  {"name": "train (finetuned)",    "started_at": "2026-05-14T15:19:26Z", "duration_seconds": 7180.0},
  {"name": "evaluate (frozen)",    "started_at": "2026-05-14T17:19:06Z", "duration_seconds": 11.4},
  {"name": "evaluate (finetuned)", "started_at": "2026-05-14T17:19:18Z", "duration_seconds": 11.7},
  {"name": "statistical (all)",    "started_at": "2026-05-14T17:19:30Z", "duration_seconds": 0.04},
  {"name": "export_best",          "started_at": "2026-05-14T17:19:30Z", "duration_seconds": 0.05}
]
```

The condition suffix is part of `name`, so `fuse (frozen)` and
`fuse (finetuned)` are separate entries with their own durations.

## Per-cell wall-time

`step_timings.json` is a flat array of `{step, started_at,
duration_seconds, labels}` entries. The `labels` dict carries the
cell identity, so every line is self-describing. A downstream notebook
plotting "extract time per backbone" can group on `labels.extractor`
directly.

```json
[
  {
    "step": "extract",
    "started_at": "2026-05-14T12:01:35Z",
    "duration_seconds": 187.4,
    "labels": {"dataset": "amazon_fashion", "extractor": "resnet50", "dim": 128}
  },
  {
    "step": "extract",
    "started_at": "2026-05-14T12:04:42Z",
    "duration_seconds": 520.1,
    "labels": {"dataset": "amazon_fashion", "extractor": "vit_b16", "dim": 128}
  },
  {
    "step": "finetune",
    "started_at": "2026-05-14T12:48:25Z",
    "duration_seconds": 245.0,
    "labels": {"dataset": "amazon_fashion", "extractor": "resnet50"}
  }
]
```

The accumulator flushes the full list on every cell append, so an
interrupted run keeps its history up to the failure point.

### Steps that emit per-cell timings

| Step | Cell granularity |
|---|---|
| `extract` | `(dataset, extractor, dim)` |
| `finetune` | `(dataset, extractor)` |
| `evaluate_finetuning` | `(dataset, extractor)` |
| `evaluate` | `(dataset, model_key)` |

### Steps without per-cell timings

`train` and `fuse` distribute their workload across subprocesses
(`TrainingOrchestrator`, `ProcessPoolExecutor`). The per-cell recorder
is a single-process singleton, so workers in another process have
their own empty singleton and the parent does not see their entries.
For these two steps:

- `manifest['steps']` captures the total wall-time per condition.
- `optuna.db` (when `hp_search.strategy: optuna`) captures the
  per-trial durations natively. That file is more authoritative for
  sub-cell breakdowns than anything reconstructed from outside.

## Working with the timings

The framework records every timing structurally during the run — there
is no need to parse `run.log` files after the fact. The per-step list
is at `manifest['steps']`, the per-cell sidecar at
`results/runs/<run_id>/step_timings.json` (see [Timing model](#timing-model)
above). Load both with `pandas.read_json` and plot with whatever charting
library fits your workflow:

```python
import json
import pandas as pd

manifest = json.loads(open("results/runs/<run_id>/manifest.json").read())
steps = pd.DataFrame(manifest["steps"])
cells = pd.read_json("results/runs/<run_id>/step_timings.json")

# Bar chart per step
steps.plot.barh(x="name", y="duration_seconds")

# Mean extract time per backbone
(cells[cells["step"] == "extract"]
    .assign(extractor=lambda d: d["labels"].str["extractor"])
    .groupby("extractor")["duration_seconds"].mean()
    .plot.barh())
```

Cross-run analyses (e.g. determinism checks across two run ids, seed
aggregation across multiple runs) are downstream analysis concerns and
live outside the framework — write a notebook or a one-off script that
reads `manifest.json` and `results/tables/evaluation_aggregated.csv`
(see [§10 of the README](../README.md#10-evaluation)) and tailors the
comparison to your study.

## Carbon footprint (optional)

ML venues (NeurIPS 2022+, EMNLP 2023+, SIGIR) ask authors to declare
the energy and CO2 footprint of trained models. The framework can
record both via [codecarbon](https://codecarbon.io), gated by two
opt-in switches:

1. Install the extra:

   ```
   pip install -e .[carbon]
   ```

2. Set the env var at run time:

   ```
   PRISM_TRACK_CARBON=1 python main.py
   ```

When active, the entire `_run_steps()` block is wrapped in a
`codecarbon.EmissionsTracker` and the result lands in the manifest:

```json
"carbon": {
  "emissions_kg_co2": 0.412,
  "energy_kwh": 1.234,
  "duration_seconds": 53210.1,
  "country_name": "United States",
  "region": "oregon",
  "cpu_model": "...",
  "gpu_model": "NVIDIA GeForce RTX 4090",
  "codecarbon_version": "2.4.1"
}
```

If either gate is missing the helper is a no-op and the pipeline runs
unchanged. Any codecarbon error (start, stop, persistence) is caught
and logged; broken tracking never fails a pipeline that would have
otherwise succeeded.

## Pinning settings instead of auto-tuning

When the host detection picks the wrong tier (for example a 12 GB
container the framework reads as 8 GB because of cgroup quirks), or
when an experiment needs an exact value for reproducibility, uncomment
the relevant fields in `configs/default.yaml -> dataloader`:

```yaml
dataloader:
  num_workers: 4         # overrides the tier value
  prefetch_factor: 4     # leave any field commented to keep the autotune
  batch_size: 64
```

Pinned values are recorded under
`manifest['dataloader_autotune']['yaml_overrides']` and the resolved
settings appear under `resolved`. A reviewer reading the manifest sees
exactly which values the run executed with.
