# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Dates are UTC.

## [Unreleased]

### Added

- **Multi-seed runs with cross-seed aggregation**.  ``configs/default.yaml``
  now accepts ``seeds: [42, 99, 7]`` (or ``--seeds 42,99,7`` at the
  CLI) to run the pipeline once per seed under
  ``results_seed{N}/`` / ``checkpoints_seed{N}/`` paths.  Inputs
  (data, embeddings) are shared across seeds since they are
  seed-independent; only the recommender training/evaluation output
  is split.  After the last seed finishes the framework writes
  ``<results>/aggregated_across_seeds/evaluation_multi_seed.csv``
  with mean / std / median / min / max / n_seeds per cell — the
  number researchers actually report.  When an SQLite Optuna storage
  is configured, its path is also suffixed per seed so concurrent
  studies do not collide.
- **Sampled evaluation protocol**.  The `Evaluator` now accepts
  `protocol="full_ranking"` (default) or `protocol="sampled"`.  Sampled
  mode draws `n_negatives` unseen items per user, ranks the positives
  against that pool, and reports the same metric set.  Configurable via
  `evaluation.protocol` / `evaluation.n_negatives` /
  `evaluation.negative_sampling_seed` in `configs/evaluation.yaml`, or
  overriden at the CLI with `--eval-protocol {full_ranking,sampled}`.
  A non-zero warning is logged whenever sampled is selected: Krichene &
  Rendle (KDD 2020) showed sampled metrics are statistically
  inconsistent with full-ranking, so the protocol is opt-in and the
  default never changes.
- **Long-format consolidation under `src/reporting/`**. The granular
  per-(dataset, test_type, metric, k) CSVs in `results/tables/`
  (~160 files for the full pipeline) are now reshaped at the end of
  the `statistical` step into three normalised long-format tables:
  `evaluation_aggregated.csv` (one row per cell × metric × k with the
  mean), `bootstrap_ci.csv` (idem + CI bounds) and
  `statistical_tests.csv` (Friedman + pairwise Wilcoxon rows in one
  file with a `test_type` discriminator). Each long table carries
  explicit `dataset`, `recommender`, `extractor`, `fusion`,
  `condition`, `metric`, `k` identifier columns — no more reading
  schema off filenames. Pure pandas, no `torch` dependency, so the
  consolidator can run on a laptop.
- **`src/data/synthetic.py`**: `SyntheticDatasetProvider` generates a
  tiny, deterministic dataset entirely in-process (100 users, 200
  items, 5 categories, 64×64 RGB images). Auto-registered under the
  name `synthetic`, used by the new smoke profile.
- **`configs/smoke/`**: bundled minimal config (synthetic dataset, one
  extractor, one fusion, two recommenders, 1 Optuna trial, 2 epochs)
  for end-to-end smoke validation on any host. Run with
  `python main.py --all --config-dir configs/smoke`.
- **`main.py --config-dir PATH`** flag to point the config loader at
  an alternative directory of YAML files. Used by the smoke profile
  and useful for ablation / experiment-specific configs without
  editing `configs/default.yaml`.

### Changed

- **Automatic category derivation in the DVBPR provider.**
  `DVBPRDataLoader.save_processed` now invokes the McAuley-taxonomy
  helper (extracted into the new `src/data/categories.py` module)
  whenever the `.npy` lacks the canonical one-hot `c` field, writing
  `data/raw/<name>/categories.csv` automatically. The manual
  pre-processing step for `amazon_men` / `amazon_women` / `tradesy`
  is no longer required.

### Removed

- **`scripts/` directory deleted entirely.** The directory mixed
  unrelated concerns (operational scaffolding specific to the
  discontinued RunPod 3-clone setup, retrofit helpers for legacy log
  files, author-side tooling for thesis writing, one piece of
  canonical preprocessing). Each item was either redundant with the
  framework, migrated into the framework, or out of scope for a
  framework repository:
  - `consolidate_tables.py` — redundant: `statistical.run()` already
    calls `write_consolidated()` at the end of the step.
  - `derive_categories.py` — migrated to `src/data/categories.py` +
    auto-invoked by the DVBPR provider; no manual step needed.
  - `extract_timings_from_logs.py` — `src/utils/timing.py` already
    records timings structurally during the run; no need to parse
    logs after the fact.
  - `plot_timings.py`, `aggregate_seeds.py`, `verify_determinism.py`,
    `export_thesis_tables.py` — author-side tooling for thesis
    writing, not part of the framework's public surface.
  - `watchdog.sh`, `setup_watchdog.sh` — operational scaffolding
    specific to the discontinued RunPod multi-clone setup. Users who
    need an external supervisor should provide one (e.g. systemd,
    Kubernetes liveness probes, `tini --restart-on-exit`).
- **`configs/watchdog.example.yaml`** — companion to the removed
  watchdog scripts.
- **§12 "Long-running operational reliability"** section removed from
  `README.md` (reflected the deleted watchdog).

## [1.0.0]

This version covers the contracts the framework exposes to outside users
(researchers, plugin authors, operators):

- Reproducibility: every run writes a manifest with the git SHA, seed,
  hardware, package versions, configuration snapshot, per-step timings
  and DataLoader autotune decisions. The manifest can be archived next
  to a publication and the run reproduced with `git checkout <sha>` plus
  the recorded environment.
- Plugin extensibility: datasets, extractors, fusion strategies and
  recommenders register from `plugins/` without touching `src/`.
- Operational reliability: the watchdog (`scripts/watchdog.sh`), the
  DataLoader autotune (`src/utils/dataloader.py`), the conservative
  recommender training defaults and the single Docker entry point let
  the same pipeline run on a 16 GB Apple Silicon laptop, on a single
  RunPod 4090 and on a multi-pod fleet.

### Added

- Unified Docker setup. Single `Dockerfile` (python:3.11-slim base) and
  single `docker-compose.yml` for every host. `docker compose up -d
  --build` is the only command the researcher needs; GPU is picked up
  automatically when the host's Docker daemon uses
  nvidia-container-runtime by default (RunPod), otherwise the container
  runs on CPU. The `device:` field defaults to `"auto"` and the
  resolved value is recorded under `manifest['device']`.
- VNPR chunk size auto-tunes from the visible GPU's VRAM (500_000 for
  < 12 GB, 2_000_000 for 12-24 GB, 5_000_000 for >= 24 GB).
- Post-run summary. `main.py` prints a block at the end of every run
  with the run id, exit status, total wall-time, the three most
  expensive steps, and the paths to `manifest.json` and
  `step_timings.json` (when present).
- `scripts/plot_timings.py`. Reads `manifest.json` plus
  `step_timings.json` and writes 150-DPI bar charts (per-step total,
  mean extract time per backbone, mean finetune time per backbone,
  mean evaluate time per recommender).
- `scripts/verify_determinism.py`. Compares two run ids and reports drift
  in `git.sha`, `seed`, `package_versions` and `config_snapshot`.
- `scripts/aggregate_seeds.py`. Aggregates evaluation CSVs across seed
  runs into mean / std / median / IQR / n per `(dataset, model,
  embedding)` cell. Only columns matching `precision`, `recall`, `f1`,
  `map`, `ndcg` are aggregated.
- `scripts/extract_timings_from_logs.py`. Reconstructs
  `manifest['steps']` and the per-cell sidecar from existing `run.log`
  files, for pipelines that started before the timing instrumentation
  landed. Output shape matches the live one, so `plot_timings.py`
  consumes it directly.
- Optional `[carbon]` extra. Set `PRISM_TRACK_CARBON=1` and install
  `pip install -e .[carbon]` to record kilograms of CO2-equivalent,
  kWh and grid country in `manifest['carbon']` via codecarbon. The
  pipeline runs unchanged when either gate is missing; codecarbon
  errors never propagate.
- Per-step and per-cell wall-time in the run manifest. The `steps`
  list in `manifest.json` carries one entry per step (with condition
  suffixes like `fuse (frozen)`). The sidecar
  `results/runs/<run_id>/step_timings.json` carries per-cell entries
  for `extract`, `finetune`, `evaluate_finetuning` and `evaluate`. The
  sidecar is flushed on every cell append so an interrupted run keeps
  its history up to the failure point. `train` and `fuse` distribute
  work across subprocesses and only contribute per-step totals; for
  per-trial detail, see `optuna.db` when `hp_search.strategy:
  optuna`.
- DataLoader autotune. `src/utils/dataloader.py` picks `num_workers`,
  `prefetch_factor` and `batch_size` from the CPU count and cgroup
  memory budget (`< 8 GB`, `8-32 GB`, `>= 32 GB` tiers). Researchers
  who need to pin a value can uncomment the matching field in
  `configs/default.yaml -> dataloader`; pinned values appear under
  `manifest['dataloader_autotune']['yaml_overrides']` and the
  resolved values under `resolved`. Replaces hardcoded defaults that
  previously OOM-killed worker pools on small-RAM hosts.
- **Post-hoc fine-tuning evaluation.** New step `evaluate_finetuning` reloads
  every v2 fine-tuning checkpoint and writes a JSON report under
  `results/finetuning/<dataset>_<extractor>.json` with top-1, top-K, macro/
  weighted F1, per-class precision/recall/F1/support, confusion matrix and
  mean cross-entropy loss on the deterministic validation split.
- **Versioned fine-tuning checkpoint format (v2)**, backbone + classification
  head + metadata bundled together so post-hoc evaluation can reproduce the
  exact split the trainer used. Loader is backward-compatible with the
  legacy flat state-dict format (head missing → evaluator skips with a
  warning).
- **Top-level `plugins/` directory**, extractors, fusions, recommenders and
  datasets all live under one obvious extension point. Each subdirectory
  ships a `_example.py` (or `_example/` for datasets) scaffold; the
  underscore prefix is what keeps the auto-discovery from importing it.
- **Plugin contract test suite** under `tests/`, registry round-trip for
  each domain, BaseExtractor `unfreeze_prefixes` declarations, FT checkpoint
  round-trip and legacy compatibility.
- **Functional test suite**, fusion-strategy math (mean/sum/prod/max_pool/
  concat/weighted_mean), BPR-loss numerical correctness against a hand
  reference, and the FineTuner freeze/unfreeze accounting on a toy backbone.
- **GitHub Actions CI**, `ruff check`, `ruff format --check`, `pytest -q`
  on Python 3.11 and 3.12 (matrix), plus an import-validation job that
  exercises every plugin domain's auto-registration on both versions.
- **Pre-commit configuration**, `ruff` + `ruff-format` + standard hygiene
  hooks; identical rules run in CI.
- **Unified per-run session log** at `logs/run_<id>.log`, every module of
  a single run interleaves into one chronological file you can `tail -f`.
- **Recipe gallery** at `docs/recipes.md`, ten common pipeline shapes
  (frozen-only, fusion-only, single recommender, custom-dataset,
  post-hoc FT metrics, etc.) expressed as the smallest YAML edit that
  produces them, plus a quick-reference table of every toggle.
- **Extension guide** at `docs/extending.md` walking through every plugin
  type with a runnable example and a contract checklist.

### Changed

- **`UNFREEZE_MAP` removed from the trainer.** Each extractor now declares
  its own `unfreeze_prefixes` class attribute on `BaseExtractor`, so adding
  a fine-tunable extractor is a one-file change.
- **`condition: frozen` no longer runs the fine-tuning step.** Auto-expanded
  pipeline runs drop steps that are irrelevant to the chosen battery
  (`finetune` + `evaluate_finetuning` for frozen-only, `extract` for
  finetuned-only). Explicit `--step NAME` invocations bypass the filter.
- **Empty `*_enabled` lists now skip the matching step** with a single-line
  info log naming the YAML key to fill in. Researchers who want to extract
  embeddings + train BPR can opt out of fusion and FT by emptying the
  corresponding lists.
- **Plugin folder naming standardised**, `src/fusion/` → `src/fusions/`
  (matching `extractors/` and `recommenders/`); `others/` → `plugins/`
  in every domain.
- **README condensed**, the long extension recipes moved into
  `docs/extending.md`; the project-tree diagram now reflects the new
  `plugins/`, `docs/`, `tests/` directories.
- **Python version support widened to 3.11 + 3.12.** `requires-python` is
  now `>=3.11,<3.13`. The Docker image stays on 3.11 as the canonical
  runtime for bit-identical experiment reproducibility; 3.12 is
  validated in CI for downstream framework users. Python 3.13 is
  deferred until `numpy>=2.0` is adopted.
- **Dockerfile base Python upgraded from 3.10 to 3.11** to match
  `requires-python` (the previous 3.10 base could not install the
  package, latent bug). The image now installs the `deadsnakes` PPA
  for `python3.11` on Ubuntu 22.04 and uses `python -m pip` to route
  through the new interpreter.
- **`pyproject.toml` build backend** floor bumped from `setuptools>=68.0`
  to `setuptools>=70.0` for cleaner 3.12 wheel-build behaviour.
- **PyPI `classifiers` metadata added** for future package publication
  (audience, license, Python 3.11/3.12, topics).
- **Known Dockerfile follow-up:** the system `python3-pip` from
  Ubuntu's apt source remains bound to the system Python 3.10, so
  `pip install …` typed inside an interactive container session goes
  to the wrong site-packages. The build itself routes through
  `python -m pip` and works correctly; a future refactor will bootstrap
  pip into 3.11 via `ensurepip` and drop the system pip package.

### Fixed

- `condition: frozen` previously paid the multi-hour fine-tuning cost even
  when only frozen results were wanted.
- Per-module logger handlers were duplicated when the root logger had its
  own handlers; logging now sets `propagate = False` to prevent the
  double-emission.
- **`AdaptiveGatedFusion` gradient test** (`tests/test_adaptive_gated.py
  ::test_module_parameters_receive_gradient`), added a warm-up SGD
  step so the second backward pass exercises every gate parameter.
  Previously the test asserted on `gate.2.weight` after the first
  backward, which is mathematically zero given the `gate[0]` zero-init
  and Tanh activation (a₁ = Tanh(0) = 0 ⇒ ∂L/∂W₂ = 0). The fusion
  implementation itself is unchanged.
- **Deprecated `datetime.timezone.utc`** in `src/utils/logging.py` and
  `src/utils/manifest.py` replaced with the `datetime.UTC` alias
  (Python 3.11+), addressing ruff `UP017`.

[Unreleased]: https://github.com/lucas-couto/prism-vrec/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/lucas-couto/prism-vrec/releases/tag/v1.0.0
