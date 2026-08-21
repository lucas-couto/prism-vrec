# Observability

Every `main.py` invocation produces a self-describing snapshot of the run
under `results/runs/<run_id>/`. Two files live there:

| File | Purpose |
|---|---|
| `manifest.json` | Reproducibility: git SHA, seed, hardware, package versions, config snapshot, per-step wall-time, DataLoader tier. One file per run. |
| `step_timings.json` | Profiling: per-cell wall-time **and telemetry** for the expensive steps (extract, finetune, evaluate_finetuning, evaluate). One file per run. |
| `telemetry_samples.jsonl` | Raw per-sample utilisation series, for plotting. Only written when `telemetry.save_samples: true`. |

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

## Per-step throughput and cost

Duration alone does not say whether a four-hour `extract` was
GPU-bound or waiting on disk. Every step entry therefore also carries a
`telemetry` block, produced by a background sampler that reads GPU and
CPU counters while the pipeline runs (`src/utils/telemetry.py`):

```json
{
  "name": "extract",
  "started_at": "2026-05-14T12:01:35Z",
  "duration_seconds": 2810.0,
  "telemetry": {
    "sample_interval_seconds": 1.0,
    "samples": 2810,
    "throughput": {
      "flops_per_s":  {"min": 2.1e12, "max": 9.4e12, "mean": 7.9e12, "samples": 2809},
      "total_flops":  2.22e16,
      "total_tflops": 22201.4,
      "items_per_s":  {"min": 210.0, "max": 980.4, "mean": 812.7, "samples": 2809},
      "total_items":  166270
    },
    "cost": {
      "cpu_util_percent":   {"min": 3.1, "max": 610.0, "mean": 228.4, "samples": 2809},
      "cpu_cores_used_mean": 2.284,
      "rss_mb":             {"min": 610.0, "max": 8134.0, "mean": 7420.1, "samples": 2810},
      "gpu_util_percent":   {"min": 0.0, "max": 99.0, "mean": 87.1, "samples": 2810},
      "gpu_mem_mb":         {"min": 1389.0, "max": 9012.0, "mean": 8455.2, "samples": 2810},
      "gpu_power_watts":    {"min": 41.2, "max": 178.9, "mean": 132.4, "samples": 2810},
      "energy_joules":      372044.0,
      "energy_wh":          103.34
    }
  }
}
```

The `download` step reports `network_mb_per_s` and `total_bytes` in the
same `throughput` block instead of compute figures.

**Reading the numbers.**

- `min` / `max` describe the slowest and fastest *sampling window*, not
  the slowest and fastest individual batch. At the default 1 Hz a
  20-minute step is summarised from ~1200 windows — enough to expose a
  stall, not a per-batch profile. A window with fewer than three
  samples reports `mean` only, since two points do not describe a
  spread.
- `cpu_util_percent` is expressed as percent of **one** core, so 228 %
  means 2.28 cores busy on average — DataLoader workers included.
- `energy_joules` is the trapezoidal integral of sampled GPU power over
  the step. It covers the **GPU only**: consumer NVIDIA cards report
  board power through NVML, while CPU package power (RAPL) is not
  readable from inside an unprivileged container. For a whole-system
  figure including CPU and DRAM, use the codecarbon integration below.
- Steps and cells slice the same run-wide sample series, so a cell's
  telemetry nests inside its step's without a second sampler.
- A step that raises still records its partial window: the recording
  happens in a `finally` block, because a failed run is exactly when
  the numbers are wanted.

**Counters vs gauges under multiprocessing.** The two halves of a
telemetry block have different coverage when a step forks workers
(`train` with Optuna inter-cell parallelism, DataLoader workers):

| | Source | Covers subprocesses? |
|---|---|---|
| `cost` (GPU, CPU, RSS) | sampled from the device and the process tree | **Yes** |
| `throughput` (FLOPs, items, bytes) | counters incremented in-process | **No** |

The counter singleton lives in the parent process, so work done inside
a forked worker never reaches it — the same constraint that makes
per-cell timings unavailable for parallel HP search (see [Steps without
per-cell timings](#steps-without-per-cell-timings)). A parallel `train`
step therefore reports its true energy and utilisation but **no**
`items_per_s`. Sequential runs (`--sequential`, or a single worker)
report both. This is a limitation, not a bug to work around: piping
counters back from workers would cost more than the number is worth,
and Optuna's own study database already records per-trial detail.

### FLOP accounting

`flops_per_s` comes from analytic accounting, not from instrumenting
every operation. The first batch through a given `(model, input shape)`
is measured once under torch's `FlopCounterMode`; afterwards the hot
loop multiplies that constant by the number of samples processed. The
alternative — leaving the dispatch counter active — costs 10-30 % of
wall-clock across the pipeline for a number that is knowable in
advance, since every backbone here runs at a fixed resolution.

The per-sample constants are recorded in the manifest so the arithmetic
is auditable:

```json
"telemetry": {
  "flops_per_sample": {"ResNet50Extractor::pooled::(3, 224, 224)": 8178368512.0},
  "training_multiplier": 3.0
}
```

Two consequences worth knowing:

- Training batches are charged `3 x` the forward cost (forward, plus
  backward with respect to inputs and to weights). This is the
  customary approximation, not a measurement — it ignores optimiser
  arithmetic and recomputation under activation checkpointing.
- `FlopCounterMode` attributes matmul-class operators (matmul,
  convolution, attention) and ignores elementwise arithmetic, the same
  convention used when the literature quotes "N GFLOPs" for a backbone.
  A matrix-factorisation recommender (BPR-MF) is embedding lookups and
  elementwise products, so it measures as **zero** and reports no
  `flops_per_s` at all — its cost is memory bandwidth, and its
  throughput is described by `items_per_s`. Attention-based
  recommenders (ACF) dispatch real matmuls and are counted normally.

### Run-level rollup

`manifest.json` also carries a top-level `telemetry` key summing the
whole run:

```json
"telemetry": {
  "probes": {"cpu": "psutil", "gpu": "nvml"},
  "flops_per_sample": { "...": 8178368512.0 },
  "training_multiplier": 3.0,
  "totals": {
    "steps_measured": 13,
    "energy_joules": 1841203.5,
    "energy_wh": 511.45,
    "total_flops": 8.9e16,
    "total_petaflops": 89.4,
    "total_downloaded_gb": 15.68
  }
}
```

`probes` names the backends that produced the readings, so a manifest
is self-describing about its own measurement quality:

| Probe | Source | Notes |
|---|---|---|
| `nvml` | `nvidia-ml-py`, in-process | Preferred. Utilisation, power, memory. |
| `nvidia-smi` | one long-lived subprocess | Fallback. Same fields; the process is started once, not per sample. |
| `torch` | `torch.cuda.memory_allocated` | Last resort. Memory only — **no energy figures**. |
| `none` | — | CPU-only host. |
| `psutil` | in-process | CPU/RSS across the process **and live children**. |
| `getrusage` | stdlib | Fallback. Undercounts DataLoader workers still alive. |

Install the `telemetry` extra for the in-process probes:

```
pip install -e .[telemetry]
```

The Docker image already includes it. Without the extra telemetry still
runs — it degrades to `nvidia-smi` and `getrusage` rather than
switching off.

### Configuration

```yaml
telemetry:
  enabled: true                 # false disables sampling entirely
  sample_interval_seconds: 1.0  # sampler cadence
  save_samples: false           # true also writes telemetry_samples.jsonl
```

Sampling costs one background thread at 1 Hz. Lower the interval for
short steps (0.1 s resolves a 10-second cell); raise it for multi-day
runs where the series would otherwise grow large. With
`save_samples: true` the raw series is written to
`telemetry_samples.jsonl` — one JSON object per sample with a
run-relative timestamp — for plotting utilisation over time.

## Per-cell wall-time

`step_timings.json` is a flat array of `{step, started_at,
duration_seconds, labels, telemetry}` entries. The `labels` dict
carries the cell identity, so every line is self-describing. A
downstream notebook plotting "extract time per backbone" can group on
`labels.extractor` directly.

Each entry also carries the same `telemetry` block documented under
[Per-step throughput and cost](#per-step-throughput-and-cost), scoped
to that cell — so "which backbone drew the most power per image" is a
group-by, not a separate experiment. It is omitted from the examples
below for brevity.

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

# Energy per backbone (Wh), from the same sidecar
(cells[cells["step"] == "extract"]
    .assign(extractor=lambda d: d["labels"].str["extractor"],
            wh=lambda d: d["telemetry"].str["cost"].str["energy_wh"])
    .groupby("extractor")["wh"].sum()
    .plot.barh())

# Where did the run spend its GPU energy?
(steps.assign(wh=lambda d: d["telemetry"].str["cost"].str["energy_wh"])
    .dropna(subset=["wh"])
    .sort_values("wh")
    .plot.barh(x="name", y="wh"))
```

With `telemetry.save_samples: true`, the raw series supports plotting
utilisation over time:

```python
samples = pd.read_json("results/runs/<run_id>/telemetry_samples.jsonl", lines=True)
samples.plot(x="t", y=["gpu_util", "gpu_power"], subplots=True)
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
opt-in switches.

This is complementary to the per-step `cost` block above, not a
duplicate of it: telemetry reports **GPU energy per step**, while
codecarbon reports **whole-system energy and CO2-equivalent for the
whole run**, including CPU and DRAM, weighted by the local grid's
carbon intensity. Quote codecarbon for a footprint declaration; use
telemetry to find which step spent the energy.

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
