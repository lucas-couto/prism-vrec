# Battery runbook (Task I)

Operational guide to run the full battery on interruptible (spot)
instances, resume after an interruption, track progress, and handle
failures.

## Before launching

1. **Extraction ready.** The features for every `(dataset, backbone)` and
   the fused matrices must exist under `data/embeddings/<dataset>/`.
2. **Feature sanity gate** (Task G) — fails loud before burning credit:
   ```
   uv run python main.py --validate-features
   ```
   Exits with a non-zero code and a clear message if any matrix is
   corrupted (NaN/Inf, wrong shape/dim/dtype, zeroed row). `train`/`fuse`
   also validate automatically on entry.
3. **Persistent Optuna storage.** Already configured in
   `configs/recommenders.yaml` (`storage: sqlite:///results/optuna/battery.db`)
   — trials survive a restart and the search resumes where it left off.
4. **Conditions (frozen / finetuned).** The runner enumerates both when
   `pipeline.condition: both` (default) — the finetuned cells only appear
   if the finetuned features exist on disk (run the `finetune` step
   first). To run only one, use `pipeline.condition: frozen` or
   `finetuned`. Frozen and finetuned are distinct cells (the stem carries
   the `_finetuned` suffix), with separate artifacts and checkpoints.

## Launch

The run takes half the host by default (`mem_limit` 16g and 8 cores in
`docker-compose.yml`) and, since the batteries run overnight, 0.95 of
the card (`resources.gpu.vram_share` in `configs/resources.yaml`, which
also holds the worker counts, the host headroom, the DataLoader pins
and the feature residency). Above ~0.85 the desktop's own GPU
allocations are no longer guaranteed, so for a daytime launch lower
the share for that launch only:

```bash
PRISM_VRAM_SHARE=0.5 docker compose up -d   # while someone works at the workstation
```

**One dataset per night, without editing tracked files.** The loader
merges every `configs/*.yaml` alphabetically after `default.yaml`; a
later file wins and lists are replaced whole. `configs/zz_local.yaml`
is git-ignored and sorts last, so it is the place for the night's
narrowing:

```yaml
# configs/zz_local.yaml (untracked)
datasets: ["amazon_fashion"]
pipeline:
  run_all: false           # REQUIRED: start_from / stop_at are ignored while run_all is true
  start_from: null
  stop_at: beyond_accuracy # run the statistical / consolidation steps once at the end
```

The easy mistake is leaving `run_all: true`: the range keys are then
dead and the whole pipeline runs (`tests/test_local_override.py` pins
this). `--show-plan` prints the steps the merged YAML resolves to.

```
uv run python main.py --battery
```
The runner:
- **enumerates** the cells (datasets × visual configs × recommenders ×
  seeds) with the built-in rules: BPR runs once per `(dataset, seed)`;
  AVBPR is excluded; DeepStyle runs on Tradesy; the **primary seed carries
  the search** and the others are **replay** of the best config (Task H);
- skips cells already completed (idempotency: valid per-user artifact);
- records the state of each cell in the **manifest**
  `results/battery/manifest.json` (inspectable).

On Docker, follow along with `docker logs -f prism-vrec` (not `docker
compose logs` — see `docs/protocol.md` about the progress bar).

## Resume after an interruption

Just **relaunch the same command**:
```
uv run python main.py --battery
```
`done` cells are skipped; training resumes from the last checkpoint and
the search from the Optuna storage. Nothing completed is redone.

What "done" and "resume" mean since 3.0.0 (task records E04/E06/E07,
I03/I04, S03):

- A manifest cell is `done` only while its per-user artifact validates
  (payload SHA-256, generation id and row count equal to the binding
  recorded in the manifest entry, provenance naming this cell). A
  replaced, torn or missing artifact — or a pre-3.0 entry without a
  binding — sends the cell back to `pending` with a note and re-runs
  it; nothing is deleted, the old generation stays under
  `results/per_user/<dataset>/.generations/<cell_key>/`.
- Grid points, trial winners and evaluate completions are reused only
  when their **scientific identity** digest matches (dataset, mapping,
  split and feature content, effective hyperparameters, budget, seed,
  protocol, condition, fold; `docs/protocol.md` §3c). A changed seed,
  split or feature file re-runs the work instead of reusing it.
- **Legacy resume envelopes are refused, not migrated.** In-flight
  `checkpoints/training/*.pt` and `checkpoints/finetuning/*_ckpt.pt`
  written by 2.12.1 or earlier raise `ResumeStateError` (no
  `envelope_version`, no identity, no reference to the historical
  best). Delete them by hand before relaunching; the trial or
  fine-tuning restarts from scratch. Envelopes written by 3.0.0 resume
  bit-identically to an uninterrupted CPU run (verified with a genuine
  process kill). The GradScaler state is saved and restored, but the
  CUDA/AMP resume path is not certified.
- **Extraction progress v1 restarts.** A `.progress.json` under
  `data/embeddings/` without `schema_version: 2` is recognised and not
  trusted: the cell restarts from row 0 with a warning. Finished
  `.npy` artifacts are unaffected. A v2 progress file resumes from the
  last validated row prefix, also across a batch-size change.
- Replay-seed resume checkpoints live under `checkpoints_seed<N>/`;
  replay winners under `results_seed<N>/models/`.

## Track progress and cost projection

```
uv run python main.py --battery-status
```
Prints the count per state (`pending/running/done/failed`) and the
**estimate of remaining hours** (average duration per cell type ×
pending). Roles without a completed sample yet are reported as "no
estimate", never guessed.

## Failures and retry

- **Fusion runs on one worker** (`resources.workers.fusion: 1` in
  `configs/resources.yaml`, decided 2026-09-06 after two PCA workers were
  OOM-killed in the 16 GB container). A worker that vanishes mid-task
  now raises `FusionWorkerLostError` with the completed count and the
  container cgroup's `oom_kill` counter; finished outputs are reused on
  the next run through their provenance record.


A cell that fails is isolated (the others keep going) and marked `failed`
in the manifest with the error message. To reprocess only the ones that
failed:
```
uv run python main.py --battery --retry-failed
```

Failure semantics since 3.0.0 (task records E01/E02, M05):

- **The process exit code is real.** `main.py` ends with the code
  returned by `run_cli`: `0` only when every required unit succeeded,
  `1` on any failure (with the traceback logged), `130` on Ctrl-C. The
  run manifest records `exit_status: "error"`.
- `--battery` and the `evaluate` step under `folds.enabled: true` raise
  `IncompleteRunError` when any cell of their manifest is not `done`
  (`failed`, `pending` or `running`), so a battery that lost cells
  cannot end with a success marker; the manifest is left as-is for
  `--retry-failed` (or the next `evaluate`, which re-runs exactly the
  cells without a valid fold artifact). `run_battery` / `run_folds` log
  `INCOMPLETE` at error level with the counts.
- Every submitted training job has exactly one terminal outcome
  (`succeeded` / `failed` / `cancelled`, with `attempts` and
  `error_type`). A worker that dies before publishing is reaped from
  its last assignment (`error_type: WorkerExit`); a job the pool never
  started is `cancelled` (`PoolExited`). The train step then raises
  `TrainingJobsFailedError` naming the failed and unaccounted units —
  completed jobs keep their checkpoints and grid progress.
- **OOM retries: 3 attempts.** Only `torch.cuda.OutOfMemoryError` is
  retried (`MAX_OOM_RETRIES = 2`, so at most three attempts, in both
  the sequential and the pool path), each retry halving the ranking
  budget. Any other exception fails the job after one attempt.
- A job whose memory ledger exceeds the resolved host budget is
  recorded as `failed` with `error_type: AdmissionRefused` and a
  reason naming every term; it is never launched, and the run fails
  through the same reconciliation.

## Where the artifacts and metadata live

- **Per-user (F):** `results/per_user/<dataset>/<cell_key>.csv.gz` (held-out
  rank, n_candidates, tie_block_size, top-20) + `<cell_key>.meta.json`
  (dataset, visual config, recommender, seed, d, protocol version).
- **Manifest:** `results/battery/manifest.json` (state + `git_sha`,
  `git_dirty`, per-cell durations and a per-cell `telemetry` block with
  GPU utilisation / power / memory, CPU usage and integrated energy —
  see [`observability.md`](observability.md)).
- **Best-trial checkpoints:** `results/models/<dataset>/`.
- **Search progress:** grid resume records via the checkpoint manager (`hp_search.strategy: grid`); `results/optuna/battery.db` only under `strategy: optuna`.
- **Per-cell generations:** `results/per_user/<dataset>/.generations/<cell_key>/<utc-stamp>-<uuid>/`
  (immutable `records.csv.gz` + `meta.json` + `manifest.json` per
  publication; the canonical `<cell_key>.csv.gz` / `.meta.json` are
  the completion pointer). Never pruned automatically; `list_generations`
  (`src/evaluation/persistence.py`) enumerates them for explicit cleanup.
- **Provenance sidecars:** `<artifact>.provenance.json` next to every
  projected / PCA-aligned / fused feature (`data/embeddings/`), and an
  `item_order` block in every extraction `.meta.json`.
- **Statistical outputs** (`results/tables/`, 3.0.0 names):
  `{dataset}_{condition}[_restricted]_{summary|friedman|pairwise}_{metric}.csv`
  and `{dataset}_{condition}[_restricted]_integrity.json` — the
  latter written BEFORE the tests with the distinct-seed count, the
  shared provenance, expected / completed / missing cells and any cell
  excluded by population, with the reason. `python main.py --report`
  finds the partitioned files by suffix.

Any accuracy metric is **recomputable** from the persisted rank, for any
`k`, without a GPU (`src/evaluation/derive_metrics.py`); the paired
users × systems matrix for the statistical tests comes from
`src/evaluation/paired_loader.py`.

## Knobs added in 3.0.0

- `statistical.population` (`configs/evaluation.yaml`): `strict`
  (default) fails the statistical step when the cells of a comparison
  do not share one user population; `declared_intersection` restricts
  explicitly, writes the `_restricted` partition and reports the
  excluded counts per row. Read the `_integrity.json` first when the
  step fails.
- `resources:` (`configs/resources.yaml`, key names approved
  2026-09-07): `gpu.{vram_share,ranking_vram_share}`,
  `host.{budget_bytes,headroom_bytes,reserved_bytes}`,
  `workers.{training,fusion,dataloader}`,
  `dataloader.{prefetch_factor,batch_size}`,
  `features.{residency,item_block}`. With `host.budget_bytes: null`
  the budget resolves from the cgroup limit (v2, then v1, then host
  RAM, then a 4 GiB fallback — never unlimited) and features stay
  dense. The resolved block is recorded under `manifest['resources']`.
  The ledger is analytic: no peak has been measured on a real
  catalogue.
- `diagnostics:` (`configs/default.yaml`, `enabled: false`): opt-in
  bounded probes per training run under `results/diagnostics/<run_id>.json`
  (feature norms, pre-ReLU branches, score ties, gradients, optimizer
  steps, zero-metric flag). Inert when off; the trajectory is
  bit-identical either way. `scripts/vnpr_collapse_diagnostic.py`
  drives the controlled VNPR experiment (see `docs/protocol.md` §7).

## Running the test suite against the working tree

Tests run inside the container (there is no host virtualenv); bind the
files the image copies at build time so the suite sees the working tree
rather than the image's stale copies (the last four mounts exist
because `tests/test_environment_lock.py` reads `pyproject.toml`,
`uv.lock`, the `Dockerfile` and the CI workflow from `/app`):

```bash
COMPOSE_PROJECT_NAME=prism-vrec docker compose run --rm -T \
  -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$PWD/tests:/app/tests:ro" -v "$PWD/configs:/app/configs:ro" \
  -v "$PWD/pyproject.toml:/app/pyproject.toml:ro" -v "$PWD/uv.lock:/app/uv.lock:ro" \
  -v "$PWD/Dockerfile:/app/Dockerfile:ro" -v "$PWD/.github:/app/.github:ro" \
  --entrypoint python shell -B -m pytest tests/ -m "not slow" -q -p no:cacheprovider
```

Expected on 3.0.0rc1: `1729 passed, 1 xfailed` — the strict `xfail`
documents that `uv.lock` predates the `telemetry` extra and is removed
when the lock is regenerated. CUDA is disabled by the environment
variable; a passing suite certifies the CPU path only.

## K-fold cross-validation over the battery (`folds.enabled`)

The `evaluate` step runs the K-fold protocol whenever `folds.enabled`
is `true` in `configs/default.yaml` — the shipped default, so a plain
`python main.py` (or `docker compose up -d --build`) evaluates by
K-fold after the search step, with the frozen winners from
`results/models` (or the fixed values under `hp_search.strategy:
fixed`). Set `folds.enabled: false` for the single leave-one-out
split; the two never run in one invocation. There is no flag: the
former `--folds` mode was removed (3.0.0), passing it fails naming
`folds.enabled`, and `--show-plan` prints the protocol a run will use:

```bash
python main.py --show-plan     # ... Evaluation protocol: kfold (k=10, ...)
python main.py
```

The runner (`src/folds/runner.py`) is resumable through
`results/folds/manifest.json`: a cell whose concatenated per-user
artifact validates for the current plan is skipped. Per fold it writes
the trained checkpoint under `<results>_fold<i>/models/` and the
partial artifact under `results/folds/fold<i>/per_user/<dataset>/`;
the final artifact lands in the canonical `results/per_user/<dataset>/`
location. The step then builds, from those artifacts, the same files
the single-split evaluator writes — `results/tables/
{dataset}_evaluation_{frozen|finetuned}.csv` (per-user rows, routed by
embedding, tagged `fold_policy: kfold_k<K>`), the completion record
`{dataset}_evaluation_done.csv` the statistical step reconciles
against, and the mean tables — so `beyond_accuracy`, `statistical` and
`--report` consume them unchanged. A cell that failed fails the step
(`IncompleteRunError`) before any table is built. The manifest entry
of every cell records the hyperparameter origin (prior search, with
the source cell reference, or fixed config values), the partition
summary (fold sizes, excluded users by reason), the per-fold seeds and
fold-in reports, the between-fold mean/std of recall@k and ndcg@k, and
the note that this variability is combined (partition + optimisation).
The run manifest records the choice under `evaluation_protocol`
(`mode`, `k`, `seed`), outside the scientific identity. See
`docs/protocol.md` §3b.
