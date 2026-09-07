# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Dates are UTC.

## [Unreleased]

### Fixed

- **The fusion item-order sidecar became a phantom embedding.** #39 made
  offline fusions write `<stem>.meta.json`; the embedding discovery glob
  `hybrid_*.json` (`get_embedding_files`) then listed
  `hybrid_pca_nc128.meta` and `hybrid_pca_per_model_nc64.meta` as
  embeddings (24 phantom jobs on amazon_fashion, failing on load), the
  same defect class as the provenance phantom fixed in 3.0.0rc1.
  Discovery now skips every JSON companion of an artifact
  (`.provenance.json`, `.meta.json`, `_ids.json`).

### Added

- **`configs/resources.yaml` is the single source of computational
  limits** (C06 key names approved by the researcher on 2026-09-07).
  One nested block — `gpu.{vram_share,ranking_vram_share}`,
  `host.{budget_bytes,headroom_bytes,reserved_bytes}`,
  `workers.{training,fusion,dataloader}`,
  `dataloader.{prefetch_factor,batch_size}`,
  `features.{residency,item_block}` — resolved and validated once at
  startup by `src/utils/resources.py::resolve_resources`: negative,
  non-finite, boolean, non-integer or out-of-range values, unknown keys
  at any level and renamed keys fail naming the key; a missing file or
  block resolves to the shipped defaults. Every consumer reads the
  resolved value at its boundary: `cap_process_vram(n, vram_share=)`
  and the worker ranking planner (`gpu.*`), `plan_pool_workers` /
  `detect_max_workers` / `admit_workers` (`host.*`, the reserve and
  headroom are now required arguments), the train step's worker count
  and the fusion pool ceiling (`workers.*`), the DataLoader autotune
  (`workers.dataloader`, `dataloader.*`), the admission ledger and
  `BaseRecommender.configure_item_block` (`features.*`). The resolved
  block is recorded under `manifest['resources']` and stays out of the
  scientific identity (`tests/test_resources_config.py` asserts the
  digest is invariant to it). A `host.budget_bytes` above the cgroup
  limit in effect is logged as a WARNING at startup.
- **`PRISM_VRAM_SHARE`** overrides `resources.gpu.vram_share` for one
  launch, forwarded by `docker-compose.yml` like `PRISM_MEM_LIMIT` /
  `PRISM_CPUS` (empty by default so the YAML governs). Meant for
  unattended windows: the amazon_women VNPR cells with a 2048-2816-d
  dense visual buffer and `latent_dim=128` (20 of 976 jobs) do not fit
  half the card and OOM at construction; `PRISM_VRAM_SHARE=0.95 docker
  compose up -d` overnight runs their retries without a source edit.
  Values outside `(0, 1]` fail at resolution, no longer at import.
- **`configs/zz_local.yaml`** is git-ignored: the per-night narrowing
  (`datasets:`, `pipeline.{run_all,start_from,stop_at}`) goes there and
  is merged last. `docs/battery_runbook.md` documents the pattern and
  the trap that `start_from` / `stop_at` are ignored while `run_all:
  true` (`tests/test_local_override.py`).

### Changed

- **The YAML is the only control surface for run configuration** (by
  the researcher's decision, 2026-09-07). The flags that duplicated
  YAML keys were removed from `main.py`; passing one fails with
  argparse's usage error plus the key that replaces it. The resolved
  plan is printed by `--show-plan` and recorded under
  `manifest['plan']` (`steps`, `condition`); a plan the condition
  filter empties is an error instead of a silent no-op. The
  `bootstrap` compose service writes the step range into the same
  transient override it already used for `datasets`.

  | Removed flag | YAML key |
  |---|---|
  | `--all` | `pipeline.run_all: true` |
  | `--step NAME` | `pipeline.run_all: false`, `start_from: NAME`, `stop_at: NAME` |
  | `--from STEP` / `--to STEP` | `pipeline.run_all: false`, `pipeline.start_from` / `pipeline.stop_at` |
  | `--condition` | `pipeline.condition` |
  | `--hp-search` | `hp_search.strategy` |
  | `--n-trials` | `hp_search.optuna.n_trials` |
  | `--eval-protocol` | `evaluation.protocol` |
  | `--seeds` | `seeds: [...]` |

  Kept: `--battery`, `--folds`, `--battery-status`, `--retry-failed`,
  `--report`, `--report-metric`, `--report-top`, `--show-plan`,
  `--inspect-pending`, `--validate-dataset`, `--validate-features`,
  `--list-*`, `--config-dir`.

### Removed

- The duplicate sources of the limits above, each replaced by one
  `resources` key. A config that still carries a removed key fails at
  load with a message naming the replacement:

  | Removed | Replacement |
  |---|---|
  | `hp_search.workers` (`configs/recommenders.yaml`) | `resources.workers.training` |
  | top-level `dataloader:` block (`configs/default.yaml`, `DataLoaderConfig`) | `resources.workers.dataloader`, `resources.dataloader.{prefetch_factor,batch_size}` |
  | `resources.host_budget_bytes` / `headroom_bytes` / `feature_residency` (M05 proposal) | `resources.host.budget_bytes` / `host.headroom_bytes` / `features.residency` |
  | `resources.max_workers` (M05 proposal) | dropped; `resources.workers.training` |
  | `RUN_RESOURCE_SHARE`, `SOLO_PROCESS_VRAM_FRACTION`, `POOL_VRAM_FRACTION` (`src/utils/device.py`) | `resources.gpu.vram_share` |
  | `_RANKING_VRAM_SHARE` (`src/utils/parallel.py`) | `resources.gpu.ranking_vram_share` |
  | `MAX_FUSION_WORKERS` (`src/steps/fuse.py`) | `resources.workers.fusion` |
  | `RESERVED_BYTES` (`src/utils/memory.py`) | `resources.host.reserved_bytes` (`headroom_bytes` for admission) |
  | `_LAZY_BLOCK_ROWS` (`src/steps/train.py`); `_LAZY_ITEM_BLOCK = 8192` literal | `resources.features.item_block` (the class attribute stays as the injection point) |

  The shipped values are unchanged except the VRAM share: 0.95 of the
  card (the researcher's unattended overnight setting; 0.5 remains the
  desktop-friendly code default when the block is absent), a 0.125
  ranking share, 4 GiB headroom and reserve, one training and one
  fusion worker, DataLoader 10 / 6 / 192, dense features in 8192-row
  blocks.

### Fixed

- **Fresh offline fusions were reported as "legacy, alignment
  unverified".** The fuse step wrote provenance but no `item_order`
  sidecar, so `validate_features` warned on every `hybrid_concat`,
  `hybrid_pca_nc128` and `hybrid_pca_per_model_nc64` it had just built
  (12 warnings per run). A fusion is row-wise over its sources, so
  `_fuse_single` now inherits the block into `<stem>.meta.json` when
  every source carries the same digest — and leaves the artifact
  unverified, truthfully, when they do not. Reused fusions written
  before this change get the sidecar on the next run
  (`tests/test_fusion_item_order_sidecar.py`).

- **`scripts/stamp_item_order.py` wrote the digest where nothing reads
  it.** Its first version merged the `item_order` block's keys
  (`schema_version`, `n_items`, `digest`) into the top level of
  `<stem>.meta.json` instead of nesting them, so `validate_features`
  kept reporting every extractor artifact as "alignment unverified"
  and each sidecar carried three stray keys. The block is now nested,
  the stray keys are removed on re-run, and the run's own validator is
  exercised in `tests/test_stamp_item_order.py`. Feature identity is
  unaffected: `feature_recipe` hashes the `.npy` header and content,
  not the sidecar.

## [3.0.0rc1] - 2026-09-06

**Release candidate.** This version carries the 27 tasks of the
reliability and scientific-integrity specification (2026-09-05); one
task record per task lives under `docs/reliability-sdd/`, indexed by
`docs/reliability-sdd/README.md`.
3.0.0 final — with the Zenodo DOI, the updated `CITATION.cff` and the
"Reproducing the dissertation" section of the README — is cut only after
the real battery completes on this candidate. Until then `CITATION.cff`
and the README's BibTeX stay at 2.12.1, the last archived release.

**Why a major bump.** Three things a 2.x reader relied on are gone:
(1) in-flight resume envelopes and extraction progress files written by
2.12.1 or earlier are refused, never guessed (delete them by hand; the
trial or cell restarts cleanly); (2) the statistical step's output
filenames are partitioned by condition and population policy
(`{dataset}_{condition}[_restricted]_{kind}.csv` instead of
`{dataset}_{kind}.csv`); (3) the recipe identity of every non-learned
online fusion sidecar changed (`recipe_version: 2`), so results trained
from `hybrid_adaptive_gated_*` sidecars under 2.x are not comparable
with results trained under 3.x. Nothing is deleted automatically.

**What is NOT certified by this candidate.** Every task was verified on
CPU in the container with CUDA disabled (`pytest tests/ -m "not slow"`:
1729 passed, 1 xfailed — the strict `xfail` in
`tests/test_environment_lock.py` documents the stale `uv.lock`, see
V01 below). No GPU or AMP path was exercised: the GradScaler
save/restore on resume, the CUDA OOM retry on a real card and the
`optimizer.steps_skipped` diagnostic are implemented but unmeasured.
No resource peak was measured on a real catalogue: the memory ledger
is analytic, the lazy feature path is bit-identical by test but its
page-cache residency and throughput are unmeasured, and no cgroup-
limited run was made (task V02, open). The Docker image was not
rebuilt from the lock-backed Dockerfile. The cause of the historical
VNPR × hybrid collapse is still unresolved (S04 below).

### Fixed

- **First run of the candidate (2026-09-06): four defects found on real
  data, all fixed before the tag.** (1) The E05 provenance sidecar
  `<artifact>.provenance.json` was matched by the embedding discovery
  glob `hybrid_*.json` (`get_embedding_files`, `src/steps/train.py`),
  which created phantom embeddings such as
  `hybrid_adaptive_gated_learned_D128.json.provenance`; 92 battery jobs
  failed on "sidecar lists no components". Discovery now skips the
  `PROVENANCE_SUFFIX` (`tests/test_embedding_discovery.py`). (2) In
  dense mode `_map_visual` and `_resolve_visual`
  (`src/recommenders/base.py`) handed the whole catalogue to the online
  fusion in one call: indexing the resident buffer copies it (3.9 GB
  for amazon_women's learned-fusion concat) before the fusion adds its
  own temporaries, so under the new 8 GB VRAM cap every user batch --
  down to a single user -- hit CUDA OOM and the evaluator fell back to
  per-user item-block scoring on CPU: `eval_s` went from 0.7 s to
  414-482 s per pass. Catalogue-sized requests are now processed in
  `_LAZY_ITEM_BLOCK` rows for the dense buffer as well (the lazy path
  already was); scores are unchanged
  (`test_dense_catalogue_requests_are_blocked_and_unchanged`). (3) The
  fusion pool planner charged 5.6 GB per PCA worker while
  `_assemble_fit_matrix` (`src/fusions/streaming.py`) fancy-indexed a
  whole source with the training index and normalised the copy --
  about 8.3 GB for tradesy -- and two workers were OOM-killed inside the
  16 GB container (`docker events` showed `oom` then `die 1`;
  `BrokenProcessPool` named no cause). The fit matrix and the
  per-model blocks are now gathered in `CHUNK_ROWS` slices
  (`_gather_rows`; tracemalloc peak above the fit matrix 31.4 MB ->
  2.0 MB on the 20K x 256 fixture, `tests/test_fuse_fit_assembly.py`),
  a lost worker raises `FusionWorkerLostError` naming the completed
  count and the cgroup's `oom_kill` counter, and by the researcher's
  decision the pool is pinned to `MAX_FUSION_WORKERS = 1`. (4) Two CI
  failures outside the container: the lazy/dense equivalence tolerance
  was below one float32 ulp (`atol` 1e-7 -> 1e-6, justified in the
  test) and the grid-budget warning test listened on the root logger
  that the project's loggers do not propagate to.

- **A valid all-zero training run vanished from the experiment (F03;
  tasks I01, I02).** `train_single_run` (`src/utils/training.py`)
  started `best_metric` at `0.0`, saved a winner only on strict
  improvement and read the selection metric with `metrics.get(key,
  0.0)`, so a run whose `ndcg@10` stayed at exactly zero — the
  documented VNPR collapse — never wrote `_best.pt`; `find_best_models`
  then did not list the cell and the battery silently lost it. An
  absent key and a NaN metric were indistinguishable from that zero.
  The first finite observation now wins (even `0.0`; ties keep the
  earlier winner and advance patience), a missing, non-scalar or
  non-finite selection metric raises `SelectionMetricError` and leaves
  no success marker, and a schedule that never validates raises
  instead of returning `0.0`. Promotion of the trial-local best
  (`_promote_trial_best`) validates the file through the single
  reader `load_best_checkpoint` (`src/utils/checkpoint.py`: exists,
  deserialises, required keys, finite metric equal to the observed
  best, own selection fingerprint) and raises `BestCheckpointError`
  instead of warning. `_evaluate_cell` (`src/steps/evaluate.py`) uses
  the same reader; the legacy flat-state_dict fallback that fabricated
  `latent_dim=64, l2_reg=1e-4` is identified as a legacy format and
  refused. An all-zero cell is now evaluated end to end with the real
  full-ranking `Evaluator` and recorded in the battery CSVs
  (`tests/test_zero_winner_evaluation.py`).
- **Resume returned the wrong weights and accepted foreign state (F03,
  F07; tasks I03, I04).** `FineTuner.train` (`src/finetuning/trainer.py`)
  restored the current weights and `best_acc` but not the weights that
  produced it; when no later epoch improved, the resumed run reported
  the earlier accuracy while returning later weights (reproduced with
  an injected interruption). The training resume envelope
  (`checkpoints/training/*.pt`) carried no identity, no reference to
  the trial-local best and no GradScaler state, so the same `run_id`
  resumed under another seed or budget silently. Both envelopes are now
  **v2**: the historical best is committed to its own digested file
  before the envelope that references it (`best_ref = {path, digest}`),
  the envelope carries `envelope_version`, an identity digest,
  `has_valid_observation`, `best_metric`, `best_epoch` and the scaler
  state, and the reader validates version, keys, identity and the
  referenced best (existence + digest) before touching the model
  (`validate_resume_envelope`, `validate_best_ref`,
  `src/finetuning/checkpoint.py`). A genuine `os._exit` kill after the
  epoch-two envelope resumes bit-identically to the continuous CPU run
  (`tests/test_training_resume_integrity.py`). Legacy envelopes raise
  `ResumeStateError`. The sampler needs no cursor: `BPRBatchSampler`
  is seeded by `(job_seed, epoch)`.
- **A training job could fail or disappear without failing the run
  (F04; tasks E01, E02).** `_run_sequential` (`src/utils/parallel.py`)
  drained an `mp.Queue` with `queue.empty()` before the feeder thread
  had flushed, returning zero outcomes for one submitted job; the
  parallel loop counted `completed` messages, so a worker that died
  before publishing left its job absent forever; OOM retries differed
  between the two paths. A parent-side `_JobRegistry` now gives every
  submitted job exactly one terminal `JobOutcome` (`succeeded` /
  `failed` / `cancelled`, with `attempt_count` and `error_type`); dead
  workers are reaped from a shared last-assignment table
  (`error_type=WorkerExit`), duplicate or stale messages are ignored,
  only `torch.cuda.OutOfMemoryError` is retried and both paths run a
  job at most `MAX_OOM_RETRIES + 1 = 3` times. `_run_grid` and the
  parallel `_run_optuna` reconcile results against the submitted ids
  and raise `TrainingJobsFailedError` (`src/steps/train.py`) naming the
  failed and unaccounted units; completed jobs keep their artifacts.
  `main.py` gained `run_cli(argv) -> int` — the process now exits with
  the real code (`1` on failure, `130` on interrupt) instead of the
  unconditional `os._exit(0)` — and `--battery` / `--folds` raise
  `IncompleteRunError` when any manifest cell is not `done`; the run
  manifest records `exit_status = "error"`.
- **Spawned workers reloaded the configuration from disk and reused
  another run's work (F05; tasks E03, E04).** `_WorkerContext` called
  `load_config()` in the child, so `--config-dir`, the multi-seed
  override and every CLI override were dropped on a spawned pool, and
  the grid-progress writer hard-coded `checkpoints/grid_search/` while
  the reader used `paths.checkpoints`. The resolved parent snapshot
  now travels with the pool (`TrainingOrchestrator(config=...)`), all
  roots come from it (`checkpoint_root`, `grid_progress_path`), and
  the global `clear_all_training_checkpoints()` call at `train.run`
  startup — which deleted every other run's resume envelope — is
  gone. Reuse is bound to **scientific identity v2**
  (`src/utils/identity.py`, contract proposal C02 implemented as an
  internal module): dataset / item-mapping / split / feature content
  digests, model name and implementation digest, effective
  hyperparameters, selection budget, seed, protocol, condition and
  fold, hashed over canonical JSON. Grid-progress entries carry the
  digest and are reused only on equality (legacy entries are counted,
  warned about and never reused); `_best.pt` carries `identity`,
  `identity_digest` and `selection_scope_digest` (a best from another
  seed, split or feature content is not comparable and is replaced
  with a warning); the evaluate done table stores `checkpoint_digest`
  and `identity_digest` and a cell is skipped only when both match.
  Block sizes, worker counts, devices, paths and wall-clock time are
  execution metadata, not identity.
- **Derived artifacts were reused on file existence alone (F06; task
  E05).** `ensure_projected` (`src/extractors/projection.py`) and the
  fuse step (`src/steps/fuse.py`) reused `<extractor>_p<dim>.npy`,
  `<ext>_pcaD<dim>.npy` and every fused output whenever the file
  existed, without checking the source content, the train-item fit
  set, the method / strategy / normalisation / online recipe, the
  dimensions or the seed. Each artifact now gets a
  `<artifact>.provenance.json` (schema 1) written atomically before the
  artifact; reuse requires an equal provenance digest
  (`check_provenance`), a differing one raises
  `ArtifactProvenanceError` naming the keys, and an artifact without a
  record is reused with an explicit UNVERIFIED warning and never
  relabelled. Recipes carry file names, shapes, dtypes and content
  digests but no directories, so identical content under another root
  is still reused.
- **Per-cell artifacts, the battery manifest and the evaluate tables
  could be torn or duplicated by an interruption (F12; tasks E06,
  E07).** `write_cell_artifact` (`src/evaluation/persistence.py`)
  wrote the records and the metadata in two non-atomic steps with no
  digest linking them, and every reader accepted whatever pair existed;
  `evaluate.run` appended per-user rows and done triples, so a
  re-evaluated cell produced duplicates; `BatteryManifest.save` was a
  plain `write_text`. Publication now writes an immutable generation
  (`<dataset>/.generations/<key>/<stamp>-<uuid>/` with the payload, a
  `meta.json` carrying `row_count`, `expected_user_digest` and a
  `completion` block with the payload SHA-256, and a `manifest.json`),
  fsyncs it, then replaces the canonical `<key>.csv.gz` and the
  completion pointer `<key>.meta.json` by rename; a per-cell lock
  rejects concurrent publishers. Readers validate the pointer against
  the payload digest and row count (`ArtifactIntegrityError` on a torn
  pair); a legacy artifact without a completion block is readable,
  warned about and NOT complete. The battery CSVs are upserted per
  `(model_name, embedding_name)` and the done table rewritten
  atomically with one row per triple. The battery manifest is written
  atomically; a `done` entry now carries an `artifact` binding
  (payload digest, generation, row count, identity digest,
  provenance), and on resume a cell is skipped only when its validated
  artifact equals the binding and names the cell — otherwise it goes
  back to `pending` with a note. Folds: `fold_plan_digest`
  (`src/folds/runner.py`: k, partition seed, `min_profile`, the exact
  user→fold assignment, split and mapping digests) is the concatenated
  artifact's `config_hash`, and a cell is complete only when its
  artifact validates and carries that digest; changing `folds.k` re-runs
  every cell. Generations are never pruned (`list_generations`).
- **Feature rows were written in `item2idx` insertion order, not by
  index, and missing images silently compacted the matrix (F08; task
  S01).** `get_item_ids` (`src/steps/extract.py`) returned
  `list(item2idx.keys())`, so a mapping `{b: 1, a: 0}` put item `b` in
  row 0, and `ImageDataset` dropped items without an image file,
  shifting every later row; `validate_features` checked only the row
  count. `canonical_item_order` (`src/utils/item_order.py`) now derives
  row `i` = the item mapped to `i` (holes, duplicates and non-integer
  indices raise `ItemOrderError`), `ImageDataset` raises
  `MissingImageError` naming the missing items, the extractor asserts
  the written order before saving, and every pooled/component/projected
  sidecar carries an `item_order` block (`schema_version 1`, `n_items`,
  digest) that `validate_features` compares against the current
  `item2idx.json` (`alignment: verified` / `unverified` for legacy
  sidecars / error on mismatch). Built-in providers emit identity
  mappings, so their existing artifacts are content-identical; they
  are reported `unverified` until re-extracted.
- **Extraction resume misread a completed-but-unfinalised part file and
  trusted incompatible progress (F11; task S03).** A part file with
  `rows_done == N` and `last_batch_index == -1` (crash between the
  final progress save and the rename) was resumed from batch zero and
  crashed with a broadcast error; a batch-size change mapped
  `last_batch_index + 1` to a different row offset; a part file from a
  reversed input order or other weights was resumed as compatible.
  `src/extractors/resume.py` writes `.progress.json` **v2** (input
  digest over the ordered `item_ids`, recipe digest over the extractor
  metadata + kind + dtype + `component_grid`, batch size, dtype, shape,
  `complete`), finalises a complete part file without a forward pass,
  rewinds to the last boundary of the current batch size and restarts
  from row 0 with an explicit reason whenever anything is unverifiable.
  Progress files of schema v1 are recognised and not trusted (restart).
- **The non-learned online fusion sidecar ignored `normalize` (F10;
  task S02).** `load_embedding` (`src/fusions/online.py`) stacked the
  equal-dim `adaptive_gated` sources as stored; the sidecar's
  `normalize: true` was read only by the learned-alignment branch, so
  the higher-norm source dominated exactly the additive fusions §10.3
  of `docs/protocol.md` says are normalised (sources `[3,3,3]` /
  `[1,1,1]` loaded with norms `5.196` / `1.732`). Each source is now
  L2-normalised once, at load, before the online module
  (`l2_normalize`, zero rows stay zero), in the eager array
  (`StackedSources`) and in the lazy `StackedFeatureSource` alike;
  `SIDECAR_RECIPE_VERSION = 2` is stamped into every new sidecar and
  exposed by the loaded object (`recipe_version`,
  `sidecar_recipe_version`; a sidecar without the field is legacy).
  **Every historical result trained from a `hybrid_adaptive_gated_*`
  sidecar (`hybrid_adaptive_gated_pca_*.json`,
  `hybrid_adaptive_gated_p*.json`) is non-comparable with results
  produced by this loader** and must be regenerated in its own
  namespace; learned-alignment cells (`*_learned_D*.json`) and every
  offline `.npy` fusion are unaffected. The legacy assertion in
  `tests/test_adaptive_gated.py` that pinned the buggy behaviour was
  corrected.
- **The battery executor sent every non-fixed strategy to Optuna and
  replayed raw suggestions (F09; tasks R01, R02).** `execute_cell`
  (`src/battery/execute.py`) had one route for `fixed` and one for
  everything else, so `hp_search.strategy: grid` created an Optuna
  study, and replay cells trained `study.best_params` — a raw
  `{total_dim, learning_rate}` without the per-paper dimension split
  (`KeyError: 'latent_dim'`). Dispatch is now per strategy: `fixed` →
  one `train_replay`; `grid` search → every grid point through
  `train_replay` under the primary seed, no study; `optuna` search →
  `_optimize_one_cell`, and a study without a COMPLETE trial raises
  `SearchOutcomeError`; replay → `resolve_replay_hyperparams`
  (`src/recommenders/hp_source.py`) of the primary seed's `_best.pt`,
  cross-checked against the existing study under `optuna`
  (`load_existing_study` never creates one; a disagreement raises
  `WinnerResolutionError`). `effective_hyperparams`
  (`src/recommenders/hp_search.py`) is the one canonical, idempotent
  expansion (single-valued defaults filled, `total_dim` split via
  `resolve_dimensions`, contradictory direct dimensions refused) used by
  the grid, fixed and Optuna producers and by every winner consumer;
  `HyperparamOrigin` records the raw `suggestion` and a per-key
  `provenance` (`suggested` / `default` / `derived`). Replay-seed
  resume checkpoints live under `checkpoints_seed<N>`.
- **The per-dataset budget reached only Optuna's trial count (F16;
  task R03).** `resolve_hp_budget` (`src/recommenders/hp_budget.py`)
  supported `hp_budget.<dataset>` overrides of epochs, patience, metric
  and validation sample, but `train_single_run` read `config["common"]`
  directly, so an override changed `n_trials` and nothing else
  (`hp_budget.synthetic.epochs=2` still ran six evaluations). Every
  training path — CLI cell, grid worker, Optuna trial, battery replay,
  folds — now consumes the resolved budget; `train.run()` resolves every
  dataset before any cell trains; `SELECTION_K_VALUES = (10,)` is the
  single declaration of the selection cut-off and a metric the
  selection evaluator does not produce raises
  `UnsupportedSelectionMetricError` before any model or data is built.
  Cadence (`eval_every_epochs`) and `batch_size` stay outside the
  budget. Default patience when `common.early_stopping_patience` is
  absent is now the schema default 10 (the loop used 20; production
  configs always set it).
- **Paired statistics dropped conflicting duplicates and inner-joined
  missing users (F15; task R04).** `_ensure_config`
  (`src/evaluation/statistical.py`) ran `drop_duplicates` on `(user_id,
  config)`, keeping the first of two conflicting rows silently;
  `pairwise_significance` and `friedman_test` `dropna`'d users missing
  from one side; the bootstrap depended on caller row order; non-finite
  values were dropped. `src/evaluation/paired_validation.py` defines the
  observation key (provenance columns + config identity columns +
  `(config, user_id)`), raises `ObservationConflictError` on a
  conflicting duplicate, `DuplicateObservationError` on an identical
  duplicate of a visual cell (the shared `embedding_name == "none"`
  baseline across condition files is the one accepted duplicate),
  `UserPopulationMismatchError` under the default `strict` policy and
  `InvalidMetricValueError` on non-finite values; frames are sorted so
  bootstrap draws are order-invariant. `load_paired`
  (`src/evaluation/paired_loader.py`) validates each cell
  (`validate_cell_records`: unique users, finite ranks, manifest
  `row_count` / `expected_user_digest` when present) and cross-cell
  provenance (`split`, `eval_protocol_version`, `n_items`).
- **Report outputs overwrote each other and counted rows as seeds
  (task R05).** `src/steps/statistical.py` wrote `{dataset}_{kind}.csv`
  regardless of `--condition`; nothing reconciled the tested cells
  against the evaluate step's completion record; `aggregate_seeds`
  counted rows as `n_seeds`; `folds/aggregate.py` located partials by
  filename only. Outputs are now partitioned
  (`{dataset}_{condition}[_restricted]_{summary|friedman|pairwise}.csv`),
  a `{stem}_integrity.json` is written BEFORE any test (seed and
  `n_seeds_distinct`, shared provenance, expected / completed / missing
  cells, `cells_excluded` with reasons, `n_users_per_cell`), missing
  cells always fail, `n_seeds = nunique(seed)` with conflicting rows
  for one seed raising `ObservationConflictError`, and partials must
  match the requested identity with distinct fold seeds
  (`FoldAggregate.n_distinct_fold_seeds`). Legacy unpartitioned files
  still classify as `report_condition = "unpartitioned"`.
- **Ranking kept device-resident catalogue ids and masks outside the
  OOM guard and had no host budget (F01 residual; task M04).**
  `_per_user_frame` allocated `arange(n_items)` on the device and
  `_ensure_train_idx_cache` built every test user's train-index tensor
  there before any guard. Catalogue ids are host-resident and
  transferred inside the guarded `_rank_user_chunk`; train masks are
  assembled per chunk on the host (`_ensure_train_idx_host`,
  `_chunk_train_mask`) and applied with one advanced-index assignment
  (same `-inf` cells as the old `index_fill_`); the CPU score / sort
  workspace is admitted against `host_ranking_bytes(n_items) = 40 B/item`
  (`Evaluator(host_budget_bytes=...)`, default = `available_host_bytes()`,
  `HostMemoryBudgetError` on refusal). Candidate set, masks, seeded
  ties and the retry-same-offset logic are unchanged
  (`tests/test_ranking_golden_fixture.py`: identical records under
  dense, user-batch reduction, offset-0 and mid-pass item OOM).
- **Feature validation materialised the whole catalogue and pools were
  sized against unlimited host memory (task M06).** `validate_matrix`
  (`src/steps/validate_features.py`) `np.load`ed the matrix and built a
  full `(N, D)` finiteness mask; `detect_max_workers(per_worker_bytes=0)`
  sized the GPU pool from VRAM alone. Validation now reduces per block
  of `block_rows` (65 536) over a memmap with identical verdicts and
  stats (`n_blocks`, `block_rows` recorded); an unknown footprint is
  charged `UNKNOWN_WORKER_FOOTPRINT_BYTES` (2 GiB), and
  `TrainingOrchestrator(admission=...)` clamps a pinned worker count
  above the admitted one and raises `AdmissionError` when nothing is
  admitted.
- **The image installed from ranges, CI pinned a torch outside the
  package's range, and the build context leaked `.cache/` and `.env`
  (F14; task V01).** `Dockerfile` copied only `pyproject.toml`; the
  running image (built 2026-08-21) differs from `uv.lock` on 20
  packages, among them **timm 1.0.28 vs 1.0.27**. The dependency layer
  now copies `uv.lock`, runs `uv export --frozen --no-dev
  --no-emit-project` and `pip install --no-deps --require-hashes`, then
  `pip check`; the `telemetry` extra is installed from its ranges in a
  separate `RUN` because the lock predates it. `.github/workflows/ci.yml`
  pins `torch==2.8.0` / `torchvision==0.23.0` from the CPU index;
  `.dockerignore` excludes `.cache/` and `.env`.
  `tests/test_environment_lock.py` guards all of it, with a strict
  `xfail` documenting that `uv.lock` is stale (three-line diff pending
  approval: `prism-vrec → (dynamic)`, `psutil 7.2.2 → 6.1.1`,
  `nvidia-ml-py 13.610.43 → 12.575.51`). `docs/environment.md` records
  the inventory. The image was NOT rebuilt.

### Changed

- **Run resource share 65% -> 50%** (`docker-compose.yml`,
  `src/utils/device.py`, `src/utils/parallel.py`). The 65% share (20g,
  10.4 cores, 0.65 of VRAM) still pushed the desktop into swap once the
  browser, IDE and another project's containers were counted (6 GB of
  swap in use with the machine idle on 2026-09-06). Every service now
  takes `16g` / `8` cores plus a soft `cpu_shares: 512`, so the
  desktop's default weight wins under contention while idle cores stay
  available to the run; `RUN_RESOURCE_SHARE = 0.5` and
  `_RANKING_VRAM_SHARE = 0.125` keep ranking kernels short on the shared
  card. Override per host with `PRISM_MEM_LIMIT` / `PRISM_CPUS`.
- **`scripts/stamp_item_order.py`** adds the `item_order` digest (S01)
  to pre-3.0 extractor sidecars without re-extracting, but only when the
  extraction's own `<stem>_ids.json` record equals the canonical order
  and the matrix has exactly that many rows; anything else is left
  unstamped and reported. Fused and projected artifacts are not
  migrated: regenerate them so they carry `recipe_version` and
  provenance.

- **Raw features can be gathered per forward from a bounded source
  instead of living in a module buffer (F02; tasks M01–M03).**
  `src/data/feature_source.py` introduces the `FeatureSource` Protocol
  (contract proposal C01, implemented as an internal adapter) with
  `NpyFeatureSource` (header at construction, read-only memmap opened
  per reading process, dropped from pickle state so spawned children
  reopen), `ConcatFeatureSource` and `StackedFeatureSource` (per-row
  concatenation / stacking of a sidecar's components, same recipe
  attributes as `RaggedSources`) and `ArrayFeatureSource`.
  `load_embedding(path, *, lazy=False)` is an opt-in keyword; the
  default return forms are unchanged. `BaseRecommender` accepts a
  `FeatureSource` in place of the array, registers no buffer, gathers
  the unique ids of each batch once (`_raw_visual_rows`) and applies
  the online fusion inside the graph; catalogue-sized requests are
  served per `_LAZY_ITEM_BLOCK` (8192) rows (`_map_visual`, VNPR
  `_predict_batch_blocked`). ACF's catalogue projection cache is
  admitted only under `DERIVED_CACHE_MAX_BYTES` (1 GiB dense, 0 lazy)
  and keyed on the projection weight's `_version`, so an optimiser step
  invalidates it without `train()`; VNPR's dense catalogue "cache" is a
  zero-copy alias. Dense vs lazy scores, losses and every gradient
  agree for VBPR, AVBPR, DeepStyle, VNPR and ACF
  (`tests/recommenders/test_lazy_feature_equivalence.py`), and a
  `train_single_run` through both paths promotes bit-identical weights.
  **The default residency stays `dense`**: the lazy path's page-cache
  residency and throughput on production catalogues are unmeasured.
- **Training jobs are admitted against a resolved host budget with a
  per-job ledger (tasks M05, M06).** `_estimate_worker_bytes` charged
  the sidecar's JSON size (a few hundred bytes) for a job that loads
  gigabytes of components, and `plan_pool_workers` always admitted at
  least one worker. `resolve_host_budget` (`src/utils/memory.py`)
  resolves `config → cgroup v2 → cgroup v1 → host RAM → 4 GiB
  fallback` (never unlimited); `estimate_job_bytes`
  (`src/steps/train.py`) sums the worker base (1.5 GiB), the feature
  payload from `.npy` headers (a sidecar = the sum of its components;
  lazy jobs are charged two staging blocks), model + gradient + two
  Adam moments, the interaction dicts and `host_ranking_bytes`;
  `plan_training_admission` refuses a job whose total exceeds
  `budget − headroom` with a reason naming every term — the job is
  recorded as a `failed` outcome (`error_type=AdmissionRefused`, 0
  attempts), never launched, and the run fails through E02's
  reconciliation — and sizes the pool so the aggregate commitment fits.
  The estimates are analytic, not measured.
- **New `resources:` configuration block (contract proposal C06,
  key names PROVISIONAL, awaiting approval).** `host_budget_bytes`
  (null = resolve as above), `headroom_bytes` (default `RESERVED_BYTES`
  = 4 GiB), `max_workers` (null = the CPU/VRAM caps; `0` admits
  nothing) and `feature_residency` (`dense` | `lazy` | `auto`, where
  `auto` goes lazy when the dense payload exceeds half the usable
  budget). Negative, non-finite, boolean or unknown values raise
  `ValueError`. Absent, the block resolves to today's behaviour; it is
  documented as a commented block in `configs/default.yaml`.
- **`statistical.population` (`configs/evaluation.yaml`).** `strict`
  (default) requires one shared user population across the cells of a
  comparison and fails otherwise; `declared_intersection` restricts to
  the intersection explicitly, writes to the `_restricted` partition
  and reports `n_excluded_a` / `n_excluded_b` (`n_users_excluded` for
  Friedman) on every row. Nothing intersects automatically.
- **Selection semantics.** The first finite validation observation is
  the winner even at `0.0` (I01, and the same rule in `FineTuner`;
  patience now resets at that first observation, so an all-zero
  fine-tuning stops one epoch later than before). Ties keep the
  earlier winner.
- **Battery `done` is a validated artifact, not a flag.** A pre-3.0
  battery manifest (none exists after the 2026-09-04 wipe) would have
  every `done` cell re-run once, its artifacts kept and re-published
  as new generations.
- `HyperparamOrigin.to_dict()` (battery and folds manifests) carries
  two additive keys, `suggestion` and `provenance`. An Optuna trial
  sampled from an `hp_space` that omits `l2_reg` now receives the
  pinned default explicitly, so its `hyperparams` dict — hence
  `run_id`, job seed and `_best.pt.hyperparams` — differs from a
  pre-3.0 Optuna run of the same config (none exists; the frozen
  battery uses `grid`).
- The evaluate done CSV gains `checkpoint_digest` / `identity_digest`
  columns; `.meta.json` gains `row_count`, `expected_user_digest` and
  `completion`; `contract_fields(meta)` strips them for callers that
  rebuild a `CellMetadata`. The concatenated fold artifact's
  `config_hash` is the fold-plan digest (was `None`).
- `execute_cell` returns additive keys (`strategy`, `budget`, `search`,
  `n_configs`, `grid`, `best_metric`); `TrainingOrchestrator.run`
  results carry `outcome`, `attempts`, `error_type` and the new status
  value `cancelled`.

### Added

- **Opt-in bounded training diagnostics and the VNPR collapse driver
  (F13; task S04).** `diagnostics:` in `configs/default.yaml` (off by
  default, inert when off) makes `train_single_run` record, on a fixed
  seeded probe at init, at listed optimizer steps and at every
  validation, feature norm quantiles per source, pre-ReLU branch
  statistics (`VNPR.diagnostic_branches`), score spread and tie
  fractions, loss components, gradient and parameter-delta norms,
  optimizer steps applied / skipped, and per-validation `zero_metric`,
  `tie_frequency`, `checkpoint_exists` and `has_valid_observation`
  as separate fields, to `<output_dir>/<run_id>.json`
  (`src/utils/diagnostics.py`). Measurements run under a forked RNG on
  detached copies; the trajectory is bit-identical on or off.
  `scripts/vnpr_collapse_diagnostic.py` runs seven visual-input
  conditions (`native_a`, `native_b`, `native_a_unit`, `learned_mean`,
  `learned_sum`, `learned_mean_nonorm`, `stacked_adaptive_gated`)
  × ≥ 3 seeds × 2 learning rates with the same split, sampler, budget
  and init, and writes `summary.json` and a `report.md` with the
  SPEC's decision table. On synthetic data (42/42 runs) no hypothesis
  was observed — no dead ReLU, no skipped step, no scale-induced
  collapse, no zero metric. **The cause of the historical VNPR × hybrid
  collapse (35.6 % of VNPR × hybrid jobs at exactly 0.0000 in the
  2026-09 battery) is still unresolved**; the discriminating experiment
  is the same driver on amazon_women / tradesy, and no activation,
  regularisation or normalisation change is proposed.
- `src/utils/identity.py`: `canonical_json`, `canonical_digest`,
  `stream_digest`, `mapping_digest`, `split_digest`, `feature_recipe` /
  `feature_digest`, `resolve_data_identity`, `experiment_identity`,
  `selection_scope_digest`, `implementation_digest`, and the provenance
  primitives (`write_provenance`, `read_provenance`,
  `check_provenance`, `ArtifactProvenanceError`).
- `src/evaluation/persistence.py`: `validate_cell_artifact`,
  `read_completion`, `contract_fields`, `list_generations`,
  `user_digest`, `ArtifactIntegrityError`, `ConcurrentPublicationError`;
  `src/battery/manifest.py`: `artifact_binding`, `done_entry_valid`,
  `present_artifact_binding`, `ManifestError`; `src/folds/runner.py`:
  `fold_plan_digest`.
- `src/evaluation/paired_validation.py`, `src/steps/statistical_integrity.py`,
  `src/extractors/resume.py`, `src/utils/item_order.py`,
  `src/data/feature_source.py`, `src/utils/diagnostics.py` (new modules).
- `docs/environment.md` (image inventory, lock mismatches, build
  recipe, CPU-vs-GPU metadata), `docs/reliability-sdd/` (27 task
  records + index).
- Tests (all CPU, container): `test_selection_zero_winner`,
  `test_zero_winner_evaluation`, `test_best_checkpoint_promotion`,
  `test_finetuning_best_state_resume`, `test_training_resume_integrity`,
  `test_parallel_job_outcomes`, `test_train_failure_propagation`,
  `test_identity`, `test_worker_config_snapshot`,
  `test_identity_reuse_binding`, `test_artifact_provenance`,
  `test_cell_publication`, `test_manifest_completion`,
  `test_item_order_alignment`, `test_extraction_resume`,
  `test_online_normalization_parity`, `test_training_diagnostics`,
  `test_vnpr_collapse_diagnostic`, `test_battery_dispatch`,
  `test_battery_strategies_integration`, `test_effective_hyperparams`,
  `test_budget_consumption`, `test_paired_validation`,
  `test_statistical_step_integrity`, `test_feature_source`,
  `recommenders/test_lazy_feature_equivalence`,
  `test_lazy_feature_scaling`, `test_ranking_golden_fixture`,
  `test_training_admission`, `test_blocked_feature_validation`,
  `test_environment_lock`.

### Migration

- **Delete in-flight resume state by hand before resuming with this
  build**: `checkpoints/training/*.pt` and
  `checkpoints/finetuning/*_ckpt.pt` written by 2.12.1 or earlier are
  refused with `ResumeStateError`; the trial / fine-tuning restarts
  cleanly. Nothing is deleted automatically. `.progress.json` files of
  schema v1 under `data/embeddings/` are not resumed: the cell
  restarts from row 0 with a warning; finished `.npy` artifacts are
  unaffected.
- **Statistical outputs moved**: read
  `results/tables/{dataset}_{condition}[_restricted]_{summary|friedman|pairwise}_{metric}.csv`
  and `{stem}_integrity.json`; `main.py --report` finds them by suffix.
- **Non-learned online fusion results are a new lineage**: regenerate
  any model or result trained from a `hybrid_adaptive_gated_*` sidecar
  under 2.x in a separate namespace; do not mix them in a paired table.
  The sidecar bytes on disk are not rewritten (the loader reports
  `sidecar_recipe_version=None` for them and builds recipe 2).
- Artifacts extracted before 3.0 have no `item_order` block
  (`alignment: unverified`); projected / fused artifacts without a
  `.provenance.json` are reused with an UNVERIFIED warning. Re-extract
  or move them aside to obtain verified records. `src/steps/finetune.py`
  does not yet write the `item_order` block for `<extractor>_finetuned.npy`
  (open follow-up), so fine-tuned artifacts validate as `unverified`.
- A `strict` population mismatch now fails the statistical step; set
  `statistical.population: declared_intersection` to restrict
  explicitly (reported, partitioned `_restricted`).
- Pending developer decisions (not applied here): regenerate `uv.lock`
  (three-line diff above; then drop the `xfail` and fold the
  `telemetry` extra into the Dockerfile's `uv export`), rebuild the
  image (expect timm 1.0.27 — read timm's 1.0.27 → 1.0.28 changelog
  before treating pre- and post-rebuild features as identical), approve
  the C06 `resources:` key names.

## [2.12.1] - 2026-09-04

### Fixed

- **The training worker's embedding cache was unbounded and OOM-killed
  the run.** `_worker_fn` (`src/utils/parallel.py`) cached every
  `embeddings_path` it ever loaded in a plain dict, and the worker is
  long-lived (1232 jobs in the frozen battery). Each entry is a whole
  `(n_items, D)` float32 matrix — a learned-alignment sidecar for
  amazon_fashion concatenates to `(166270, 2816)`, i.e. 1.9 GB — so the
  process accumulated every fusion of every dataset until the container
  cgroup killed it at 25 GB anon-rss, without a traceback, one dataset
  into the battery. Both caches are now `SingleSlotCache`, which evicts
  the previous value *before* the loader allocates the next: peak is one
  entry, not two. Jobs are emitted dataset -> model -> embedding, so a
  single slot still serves a cell's whole hyperparameter grid from
  cache; the miss at a cell boundary costs one npy read.
- **A solo CUDA process claimed the whole card, freezing the
  workstation.** The GPU that trains is the same one that draws the
  desktop. `set_per_process_memory_fraction` was called only when
  `n_workers > 1`, so the single-worker battery (the pinned
  configuration since 2026-08-29) ran uncapped, and `extract` and
  `evaluate` — each its own process, the longest and the heaviest GPU
  steps respectively — never capped at all. The cap now lives in one
  place, `cap_process_vram` (`src/utils/device.py`, with
  `SOLO_PROCESS_VRAM_FRACTION = 0.85` and `POOL_VRAM_FRACTION = 0.90`),
  and every long-lived CUDA entry point calls it. A regression test
  walks those entry points' ASTs, and a second fails if any module
  outside `device.py` sets the fraction directly.

### Changed

- **`_RANKING_VRAM_SHARE` 0.5 -> 0.25** (`src/utils/parallel.py`). A
  ranking batch sized to fill half of 16 GB issues kernels long enough
  that the compositor never gets a slice; on 2026-09-01 that tripped the
  display driver's watchdog. Combined with the 0.85 cap the validation
  user-batch on amazon_women falls from ~822 users to ~373, which also
  puts VNPR's eval back inside the card: it allocates four `(B, N)`
  buffers in `predict_batch`, against the model-agnostic 28 B/element
  the budget assumes. Metrics are unaffected — `plan_ranking_batch`
  documents batching as a pure execution detail, every row is scored,
  masked and sorted independently.
- **Container limits sized for the workstation, not for the run**
  (`docker-compose.yml`, all three services). `mem_limit` /
  `memswap_limit` 24g -> 16g and a new `cpus: 12`, both overridable
  (`PRISM_MEM_LIMIT`, `PRISM_CPUS`). At 24g of 32 GB the host was left
  with ~8 GB and swapped; uncapped, the sampler and dataloader workers
  took all 16 cores. With the cache fixed, a worker peaks at roughly one
  embedding matrix plus one dataset's interaction index, so 16g is
  ample.

## [2.12.0] - 2026-09-04

### Added

- **`component_grid` (configs/extractors.yaml): ACF component maps are
  pooled to a `g × g` grid at extraction.** `src/extractors/components.py`
  adaptive-average-pools each backbone's native `√M × √M` region map
  before the fp16 `*_comp.npy` is written (streaming and in-memory
  paths); the default `2` keeps four quadrant descriptors per item
  (ResNet-50 on Amazon Women: 5.7 GB instead of ≈ 70 GB; ≈ 20 GB for the
  whole battery instead of ≈ 1.1 TB). Recorded in the artifact
  `.meta.json` (`component_grid`, `pooling`); declared divergence from
  the paper's 49 regions in `docs/protocol.md` §7. Unset = native
  regions.

### Changed

- **Component artifacts are memmapped and kept in fp16 inside the
  model.** `load_embedding` opens `*_comp.npy` with `mmap_mode="r"`,
  `BaseRecommender` keeps the on-disk dtype for raw component buffers
  (fp32 upcast remains for every other path), and ACF casts the gathered
  rows on use and builds its eval-mode catalogue projection in chunks of
  16 384 items. Together with the grid this is what let ACF back into the
  validation profile after the 24 GB OOM (Phase C).

### Removed

- **`configs/validation/` and `configs/validation_tradesy/`.** The
  pre-battery validation profiles (Phases A–D, 2026-09-03/04) served
  their purpose — K-fold machinery verified, VNPR regulariser and the
  `--folds` skip bug found and fixed in 2.11.0, search grid decided —
  and were removed together with every run output when the user
  restarted the pipeline from scratch (`data/raw` and the download
  timings in `results/pre_training_costs.*` kept). Their reports are
  summarised in 2.11.0's entries and in `docs/protocol.md`.

## [2.11.0] - 2026-09-04

### Added

- **Fold-tagged training logs under `--folds`.** `train_single_run`
  accepts a keyword-only `log_context` appended to every per-epoch
  `timing` line (`... eval_s=0.36 fold=2/5`), and the K-fold runner
  logs a `Fold i/k of <cell>: training on N users, folding in M
  held-out users` header before each fold. Other callers are unchanged.
- **User-level K-fold cross-validation** (`python main.py --folds`,
  `configs/default.yaml -> folds:`; off by default). Users are
  partitioned into `k` seeded, balanced folds (`src/folds/partition.py`;
  ineligible users counted as `no_target` / `profile_too_small`). Per
  cell and fold the model trains on the other folds with the cell's
  frozen hyperparameters under seed `folds.seed + i`, the held-out users
  are folded in (`src/folds/foldin.py`: only their rows in
  `_USER_TABLES` are optimised, everything else frozen, same BPR loss
  and sampler over the profile; `rebuild_user_state` for
  history-consuming models) and ranked on their single target; the `k`
  partial per-user artifacts are concatenated into the canonical cell
  artifact (`src/folds/aggregate.py`) so the paired statistics keep the
  user as the unit, with between-fold mean/std as descriptive
  variability. Manifest at `results/folds/manifest.json` (resumable)
  records the hyperparameter origin, the partition summary, per-fold
  seeds and the combined-variability note. Precedent: Rendle et al.
  (2009, §6.2, §2). Docs: `docs/protocol.md` §3b / §10.7.
- `hp_search.strategy: fixed` — no search: every hyperparameter must be
  a scalar (or one-element list); each cell trains once, no Optuna study
  is created or read; a multi-valued list fails loud
  (`FixedHyperparamsError`). Applies to the battery and to the folds.
- `src/recommenders/hp_source.py::resolve_cell_hyperparams` — one
  resolver for the two hyperparameter origins (prior search via
  `results/best_hyperparams.json`, generated on demand from the
  `_best.pt` checkpoints; or fixed config values), recorded per cell.
- `BaseRecommender._USER_TABLES` (fold-in surface, distinct from the
  regularisation tables) and `rebuild_user_state`; `bpr_step` extracted
  from the training loop.
- Tests: `tests/test_fold_partition.py`, `tests/test_fold_in.py`,
  `tests/test_fold_aggregate.py`, `tests/test_fixed_strategy.py`,
  `tests/test_hp_source.py`, `tests/test_folds_runner.py`;
  `src.folds.foldin` joins the no-`test.csv` structural guard.

### Changed

- **Hyperparameter search is an exhaustive grid over `learning_rate ×
  total_dim` (4 points per cell); `l2_reg` pinned to 1e-4;
  `hp_search.strategy: grid`.** Validation Phase D
  (`results/validation/phase_d_search_cost.md`): of the 182 Optuna
  studies of the 2026-08 battery only lr (0.001 in 153) and total_dim
  (128 in 138) carried signal, l2_reg and every per-model axis split
  ~50/50, and TPE re-sampled winners on the 16-point categorical space
  (median 17 trials, 6 distinct combos). Estimated search cost drops from
  ≈ 80 h to ≈ 19 h; the K-fold run then trains the per-cell winners
  (`best_hyperparams.json`, `hp_source` = search) under frozen values.
- **Paired effect size reported as the win / loss / tie triplet**
  (metrics audit #1, #2). `pairwise_significance` adds `n_wins`,
  `n_losses`, `n_ties`, `pct_wins`, `pct_losses`, `pct_ties` next to
  `cliffs_delta` and DROPS `cliffs_magnitude`: the Romano et al. cut-offs
  were calibrated for the between-groups delta, and under leave-one-out
  the ties that dominate the denominator would label every comparison
  "negligible". `paired_outcomes` / `PairedOutcomes` expose the counts;
  `cliffs_delta_magnitude` is removed. No metric value changes.
- **EFD exclusion share is instrumented** (metrics audit #5):
  `efd_excluded_frac@k` per user (share of the top-k with zero train
  popularity, hence excluded from the EFD mean) and
  `efd_excluded_frac_mean` per cell in the coverage table, with the EFD
  docstring stating the `1/|valid|` denominator.
- `assert_dimension_parity` ignores `latent_dim` / `visual_dim` keys the
  config schema materialises as `None` (found by the pre-battery validation run).
- K-fold training users keep `train` only (no user's `test.csv` item
  enters training); the split reader lives in `src/folds/splits_io.py`
  and `src/folds/partition.py` is under the no-`test.csv` guard.
- `CellMetadata.fold` carries the fold provenance (partial:
  `{index, k, seed, n_users}`; concatenated: `{k, seeds, n_users_per_fold}`);
  the `.fold.json` sidecar is gone.
- **Weighted-family strategies renamed to say what they compute**
  (fusion audit #1): `gated` -> `sigmoid_gated`, `attention_weighted` ->
  `softmax_weighted`, matching the qualification's "combinações
  ponderadas" (fixed weights / softmax over configurable values summing
  to one / independent sigmoids then normalised). Mechanism, canonical
  values and the FIXED-weight contract are unchanged; only the registry
  keys, `fuse_*` function names, config keys and artifact stems
  (`hybrid_sigmoid_gated_*`, `hybrid_softmax_weighted_*`) change. No
  artifact with the old stems exists on disk. `sum` is kept: with
  pre-fusion L2 and no post-fusion normalisation it equals `2 x mean`
  and is absorbed by the recommender's projection up to L2
  regularisation, which `docs/protocol.md` §10.3 now states.
- **`weighted_mean` sweeps both sides of the convex family** (fusion
  audit #3). `configs/fusion.yaml` sets `w_cnn: [0.3, 0.7]` (was
  `[0.7]`): every canonical point of the fixed-weight family (mean 0.50,
  sigmoid_gated 0.59, weighted_mean 0.70, softmax_weighted 0.73) sat in the
  CNN-dominant half, so the sweep could not tell an optimum near 0.7
  from a monotone "more CNN is better" curve whose end point is plain
  ResNet-50. One extra cell per dataset per recommender
  (`hybrid_weighted_mean_w0.3_learned_D128`). Requires re-running the
  `fuse` step to write the new sidecar; existing checkpoints are
  unaffected.

### Removed

- **`configs/smoke/` and its README/recipes sections.** The synthetic
  end-to-end profile no longer ships; the `synthetic` dataset provider
  stays (it backs the unit tests) and `--config-dir` is documented
  through the validation profiles instead.
- **Dead `metrics:` list in `configs/evaluation.yaml`.** No step read it
  (the top-level schema is `extra="allow"` for plugin blocks, so the
  key passed silently); the set of metrics each step writes is fixed
  and now documented in place, and `statistical.primary_metrics`
  remains the only selector. Step labels "06/06b/07" in the comments
  replaced by the real step names (`evaluate`, `beyond_accuracy`,
  `statistical`).

### Fixed

- **VNPR trained to a visual-only model: whole-matrix L2 under Adam
  zeroed its collaborative tables.** The paper-faithful regulariser
  (`_L2_USER_TABLES = _L2_ITEM_TABLES = ()`, whole `W_u`/`W_i`/`W_i'`/`W_v`
  penalised every step) drove every rarely-gathered row to exactly 0.0
  under Adam (validation profile, all three VNPR cells), and on the
  unit-norm learned-fusion input the surviving visual term was flat
  (val 0.0002). VNPR now uses the BPR-Opt gathered rows like every other
  recommender (`W_i` by positives, `W_i'` by negatives, `W_u`/`W_v` by
  users; `dense` still unpenalised). Declared divergence in
  `src/recommenders/vnpr.py` and `docs/protocol.md` ("VNPR
  regularisation"); regression test in `tests/recommenders/test_vnpr_paper.py`.
  VNPR checkpoints trained before this change are invalid.
- **`--folds` no longer skips cells whose leave-one-out artifact already
  exists.** The concatenated fold artifact is written to the cell's
  canonical `per_user/` path on purpose (the paired loader consumes it
  unchanged), so `_cell_done` in `src/folds/runner.py` could not tell it
  from the artifact the `evaluate` step writes under the same seed and
  marked every cell "fold artifact already present" without training a
  single fold. It now requires the concatenated K-fold provenance
  (`fold = {"k", "seeds", "n_users_per_fold"}`) in the `.meta.json`.
  Regression test in `tests/test_folds_runner.py`.

## [2.10.0] - 2026-09-02

Every built-in recommender now reproduces its paper's score and
regularisation, and all recommenders draw their dimensions from one
shared budget. Checkpoints and results from earlier versions are not
comparable with runs after this release; the battery must be retrained.

### Changed

- **Dimension parity.** `common.total_dim` replaces `common.latent_dim`
  / `common.visual_dim`. Each recommender derives its own dimensions
  through `RecommenderSpec.dim_split` (`latent`: `latent_dim = T`;
  `half`: `latent_dim = visual_dim = T/2`, the VBPR 50/50 split), so the
  BPR-vs-visual comparison (H1) controls capacity as in He & McAuley's
  baseline protocol. `assert_dimension_parity` refuses a direct
  `latent_dim` / `visual_dim` in `common:`, a model block or an
  `hp_space` before training starts. `get_hyperparam_grid` now lives
  only in `src/recommenders/hp_search.py` (the `train.py` duplicate is
  gone) and expands `total_dim` for both grid and Optuna.
- **Per-group L2 regularisation** (`src/recommenders/base.py`).
  `_L2_LAMBDA_KEYS` maps a table or `(table, role)` to a config key,
  `_L2_LAMBDA_DEFAULTS` supplies paper defaults, `_L2_UNREGULARIZED`
  excludes modules; `_l2_gathered_terms` / `_l2_shared_terms` return
  `(key, tensor)` pairs and `l2_reg()` is the already-weighted penalty
  (`bpr_loss = bpr + l2_reg()`). Unlisted parameters keep the single
  `l2_reg`.
- **BPR** is BPR-MF (Rendle et al. 2009, §4.3.1): `γ_u^T γ_i`, no item
  bias; `λ_W` (`l2_reg`), `λ_H+` (`l2_reg_item_pos`), `λ_H−`
  (`l2_reg_item_neg`).
- **VBPR / AVBPR** carry the visual bias `β'^T f_i` of Eq. 4
  (`visual_bias`, zero-initialised, dimension of the native feature),
  with `λ_E = 0` by default (`l2_reg_projection`) and `λ_β`
  (`l2_reg_visual_bias`). AVBPR mirrors VBPR so attention stays the only
  difference between them.
- **DeepStyle** follows Eqs. 2–3 and 6 (Liu et al. 2017): one user vector
  `p_u`, one `d = latent_dim` (`style_dim` is refused when it differs),
  no item bias, single `λ`. On Tradesy it degenerates into a restricted
  VBPR (`γ_u ≡ θ_u`, no `β_i`, no `β'`), which the paired test now
  states explicitly.
- **ACF** follows Eq. 6 (Chen et al. 2017): `(γ_u + Σ α(u,l) p_l)^T γ_j`,
  no visual term in the score, no item bias; the attention nets and
  projections are unpenalised (Eq. 5); `predict_batch` is a single GEMM
  (tiling and `_comp_hidden_cache` removed). With uniform attention the
  model degenerates into SVD++ (tested). `max_history` accepts `null`
  (full history) and, when the history exceeds it, subsamples
  **uniformly with a seed** (`history_seed` = the run seed, injected by
  train and evaluate) instead of keeping the lowest item indices, which
  correlate with popularity in the DVBPR splits.
- **VNPR** is the NPR/VNPR architecture (Niu, Caverlee & Lu 2018):
  user/item embeddings merged by Hadamard product, the user's visual
  vector `v_u ∈ R^{D_v}` multiplied element-wise by the raw image
  feature, concatenation of the two products into a single-neuron ReLU
  dense layer, mirrored item tables `W_i` / `W_i'` for the two branches,
  `½(r + r')` at inference, optional embedding dropout (`vnpr.dropout`),
  L2 over the whole embedding matrices. The MLP-over-concatenation
  model, `hidden_layers`, the VRAM chunk helpers and the Optuna warning
  filter in `main.py` are gone. Learning-rate decay and a per-model
  patience are deliberately not reproduced: the early-stopping budget
  is shared per dataset (`hp_budget`).
- `LinearVisualScoreMixin` gained neutral hooks `_item_bias_term` and
  `_item_visual_bias`, and its full-catalogue cache
  (`_full_catalog_lookup`) only serves eval lookups over
  `arange(n_items)` — a training batch or an eval `forward` of that
  length can no longer poison it.
- Registry: `register_recommender(..., dim_split=...)`; the built-ins
  declare their `λ` keys as `extra_hyperparam_keys`. Grid sizes: BPR,
  VBPR, VNPR, DeepStyle 8; AVBPR, ACF 16.

### Added

- `tests/test_dimension_parity.py`, `tests/recommenders/test_vbpr_visual_bias.py`,
  `tests/recommenders/test_vnpr_paper.py` (replaces
  `test_vnpr_factorized_eval.py`), and the rewritten
  `test_recommender_bpr_loss.py`, `test_deepstyle_paper.py`,
  `test_acf_model.py`, `test_acf_attention.py`,
  `test_acf_vectorized_eval.py`.

## [2.9.3] - 2026-09-02

### Fixed

- **DVBPR split parser dropped item id 0.** `_parse_split` in
  `src/data/dvbpr.py` chained the `productid` lookups with `or`, so a
  product id of `0` was falsy and silently discarded from every split.
  Users whose held-out was item 0 vanished from `val.csv`/`test.csv`
  (amazon_men: 55 users lost the item, 5 disappeared from test;
  amazon_fashion: 1 user gone from test). The lookup now chains on
  `is None` via `_product_id`. `data/processed/` must be regenerated;
  every battery run before this fix used the truncated splits.
- **Leave-one-out disjointness enforced on raw provider splits.**
  `deduplicate_leave_one_out` (`src/utils/splits.py`) removes held-out
  items from the user's training history, resolves an item held out for
  both validation and test in favour of test, and drops users left with
  no training history. Tradesy's DVBPR partition merges purchase and
  "want" lists, so ~1.6k held-out pairs also sat in training and were
  unhittable under the masked ranking (the user silently scored 0). The
  clean Amazon partitions pass through untouched. Tradesy loses 5 users
  that only had 2 distinct items.
- **Pooled extraction streams to disk** (`src/extractors/base.py`). The
  accumulate-and-pickle checkpoint OOM-killed the container on
  amazon_women x resnet50 and left a 0-byte checkpoint that crashed the
  next resume. With a `checkpoint_path` the pooled path now writes an
  fp32 `<base>.part.npy` memmap with a progress sidecar, exactly like the
  component path, and ignores stale or corrupt legacy checkpoints.
- **VRAM sizing uses free memory, not total.** `detect_max_workers`
  (`src/utils/parallel.py`) reads `torch.cuda.mem_get_info`, and
  `main.py` releases the orchestrator's cached CUDA blocks between steps,
  so training workers are budgeted from what is actually available.
  `hp_search.workers` is pinned to 1 in `configs/recommenders.yaml`
  after the 2026-08-29 OOM.
- **VNPR evaluation chunk sized off free VRAM.** `_free_vram_chunk_pairs`
  (`src/recommenders/vnpr.py`) budgets half of the free allowance per
  eval call, capped by the static tier; online-fusion cells on
  amazon_women OOMed with the static chunk. Scores are unchanged.
- `docker-compose.yml` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  on both services to keep allocator fragmentation from turning the
  full-ranking eval peaks on the large datasets into CUDA OOMs.

### Changed

- `common.early_stopping_patience` 20 -> 5 epochs (`configs/recommenders.yaml`):
  with `eval_every_epochs: 10` the counter advances by 10 per validation,
  so any non-improving validation now stops the run instead of two.

### Added

- `tests/test_dvbpr_parse_item_zero.py`, `tests/test_split_deduplication.py`,
  `tests/test_pooled_streaming.py`, VNPR chunk-sizing cases in
  `tests/recommenders/test_vnpr_factorized_eval.py`, and the free-VRAM
  worker case in `tests/test_hp_search.py`.

## [2.9.2] - 2026-08-27

### Changed

- **Fusion alignment is the learned linear projection, on native
  features.** `configs/fusion.yaml` sets `extractor_variants: native`
  (was `projected`), so the equal-dim fusion family (mean, sum, prod,
  max_pool, weighted_mean, attention_weighted, gated, adaptive_gated)
  routes through `alignment: {method: learned, dim: 128}` instead of
  bypassing it: each extractor gets its own `Linear(D_i -> d)`
  (ResNet-50 2048 -> 128, ViT-B/16 768 -> 128) trained end-to-end by the
  recommender's BPR loss, and fusion happens after projection. The
  concatenation family (concat, pca, pca_per_model) is unchanged and
  keeps native dims. `configs/recommenders.yaml` sets
  `embedding_variants: native` and `configs/extractors.yaml` sets
  `projection.method: none`, so no fixed `_p<dim>` artifacts are
  produced. Runs from 2.9.0/2.9.1 (pca_whitened p128 sources) are not
  comparable with runs after this change.

### Added

- Methodological guarantee, made structural and documented: every
  equal-dim fusion instantiates the same projection architecture via
  the strategy-agnostic `LearnedAlignmentFusion.build_projections`,
  sized by the single `alignment.dim` key, so a comparison between two
  fusions isolates the fusion operation. Documented in
  `src/fusions/online.py`, `configs/fusion.yaml` and
  `docs/learned_fusion.md`.
- `tests/test_learned_alignment_guarantee.py`: identical projection
  signature and parameter count across the eight strategies; projection
  parameters registered in the optimiser, receiving finite gradients and
  updating after one BPR step through VBPR; `alignment.dim` propagated
  from the schema through the fuse sidecar to the module; concat/PCA
  strategies offline, dim-agnostic and free of the projection.


## [2.9.1] - 2026-08-26

### Fixed

- **The container no longer writes root-owned files onto the host.** The
  image declares no `USER`, so the pipeline ran as uid 0; a bind mount,
  unlike a named volume, writes to the host filesystem with the
  process's uid. Every dataset, embedding, checkpoint and result was
  therefore born `root:root` and could not be deleted by the researcher
  without borrowing Docker's root back through a throwaway container.

  The fix was blocked by `model_cache:/root/.cache` — as uid 1000 the
  pretrained-weight cache would be unwritable — so the cache moved as
  part of it:

  - `user: "${PRISM_UID:-1000}:${PRISM_GID:-1000}"` on `pipeline`,
    `bootstrap` and `shell`, overridable per host.
  - The weight cache became a bind mount at `/app/.cache` and the named
    volume was dropped. uid 1000 has no `/etc/passwd` entry inside the
    image, so `HOME` would resolve to `/` and timm / open_clip /
    torchvision would fail to cache; `HOME`, `XDG_CACHE_HOME`,
    `TORCH_HOME` and `HF_HOME` now all point at the writable path.
  - `results/`, `checkpoints/` and `logs/` are kept in the repo with
    their own `.gitignore`, as `data/` already was. A bind-mount target
    missing on the host is created by the daemon **as root**, so without
    this a fresh clone reproduces the bug.

  No `Dockerfile` change and no rebuild required: every `mkdir` in the
  image sits under a bind mount that replaces it, and the code creates
  its subdirectories at runtime.

- **The run manifest records the version of the framework itself**, not
  only of its dependencies. Previously `package_versions` tracked torch,
  timm, open_clip and friends but not `prism-vrec`, leaving an install
  without a git checkout (`pip install prism-vrec`, where the `git`
  block is null) with no identifier of the code that produced the run.

  Simply adding the distribution to the tracked list would have been
  worse than the omission. `importlib.metadata` reads the `dist-info`
  written at *install* time, and the Docker setup bind-mounts `./src`
  over an editable install — so that metadata is frozen at the version
  the image was built with while the executing code is whatever the
  mount provides. On the image in use it reported `2.5.0` for code that
  was `2.9.0`. A wrong version in a manifest is more dangerous than a
  missing one: a null is noticed, a plausible number gets cited.

  The version now lives in `src.__init__.__version__` — the one place
  that travels with the mounted code — and `pyproject.toml` consumes it
  through `[tool.setuptools.dynamic]` instead of declaring its own, so
  there is a single place to bump and the packaged metadata cannot
  disagree with the running code. Verified in both directions: against
  the stale image and with no rebuild the manifest reports `2.9.1`, and
  setuptools still resolves the version statically to build a real
  wheel.

### Changed

- `configs/default.yaml` pins the DataLoader sizing
  (`num_workers: 10`, `prefetch_factor: 6`, `batch_size: 192`) instead
  of taking the autotune's "balanced" tier. The override is recorded in
  the manifest under `dataloader_autotune.yaml_overrides`, alongside the
  values the autotune would have picked.

## [2.9.0] - 2026-08-24

### Added

- **Beyond-accuracy metrics as a post-hoc step (`beyond_accuracy`,
  06b)**, between `evaluate` and `statistical`. It consumes the
  per-user top-20 artifacts persisted at final evaluation
  (`results/per_user/`) — it never re-ranks and never touches the
  Evaluator hot path — and merges the new columns into the same
  per-user tables as the accuracy metrics, for every `k_values` ≤ 20:

  - `efd@k` — novelty as Expected Free Discovery in its Mean
    Self-Information reduction (Vargas & Castells, RecSys 2011,
    eq. 14). Item popularity is estimated on the TRAIN split only;
    items unseen in train are excluded from the average (logged), and
    an optional `use_rank_relevance` flag (off by default) applies the
    paper's exponential rank discount (base 0.85) for future
    comparability with Deldjoo et al. (2021).
  - `ild@k` — intra-list diversity (Vargas & Castells 2011, eq. 16)
    over visual embeddings, with the cosine similarity normalised to
    [0, 1] before the complement so negative cosines cannot produce
    distances above 1. **Pre-registered decision:** ILD is computed in
    ONE fixed reference space — the native ResNet50 artifact
    (`beyond_accuracy.reference_embedding`) — for every system,
    never in the evaluated extractor's own space: self-model diversity
    is not comparable across systems. Missing artifact fails loud.
    Single-item lists report NaN, never a forced 0.
  - `cat_entropy@k` — Shannon entropy (base 2) of the top-k category
    distribution, using the same `item_category_array` source as
    DeepStyle/fine-tuning. Only for datasets with
    `expects_categories: true`; Tradesy is explicitly N/A (no column
    written), honouring the category contract.
  - `icov@k` — item coverage over the TRAIN catalogue. AGGREGATE
    (one value per cell): the column is replicated across per-user
    rows for convenience and also written to
    `{dataset}_beyond_accuracy_coverage.csv`; the statistical step
    refuses to include `icov` in the per-user Wilcoxon/Friedman
    families (no per-user distribution — distinct treatment required).

  Adding `efd` / `ild` / `cat_entropy` to
  `statistical.primary_metrics` runs them through the same
  Wilcoxon/Holm comparison families as `recall`/`ndcg`. Pure metric
  functions live in `src/evaluation/beyond_accuracy.py` in the style
  of `metrics.py`; `compute_all_metrics` is untouched.

### Changed

- The statistical step writes one CSV per (dataset, kind) —
  `summary` / `friedman` / `pairwise` / `aggregated` — with `metric`
  and `k` identity columns, instead of one file per `metric@k`; the
  long-format consolidators split them back per metric group.
- The evaluate step writes `{dataset}_evaluation_mean_{target}.csv`
  (one mean row per config plus `n_users`) alongside the per-user
  table.

## [2.8.3] - 2026-08-24

### Fixed

- **The config is the source of truth everywhere** (b92cf2c). `evaluate`
  only runs `*_best.pt` checkpoints whose model / backbone / strategy /
  variant is enabled in the CURRENT config (stale cells accumulated in
  the models dir were being evaluated into the tables); the
  `validate_features` gate scans only artifacts the current config
  trains (shared predicates in `src/utils/variant_filters.py`); dead
  keys `preprocessing.n_min` and `paths.logs` removed;
  `extractors.<name>.weights` promoted to a validated contract.

## [2.8.2] - 2026-08-24

### Fixed

- **Audit code block** (#28): report aggregation (A1/S8), paired Cliff's
  delta (S2), log-sigmoid BPR loss (F9), early-stopping patience and
  warm start (F10/F11), config-driven fusion + extractor cell filters,
  `hp_search.workers` honoured.

## [2.8.1] - 2026-08-24

### Fixed

- **Runtime hardening after the first 2.8.0 runs** (#27): worker VRAM
  caps sum below the card with allowance-aware budgets (VNPR eval
  chunk, ranking budget), OOM retry per Optuna cell, parent-side sqlite
  schema init, fp16 streaming component extraction with resume, float32
  fusion outputs, `pca_whitened` fit set, per-cell ETA logging.

## [2.8.0] - 2026-08-24

### Changed

- **2026-08-22 adversarial methodology audit cycle** (#26): BPR L2
  semantics (BPR-Opt gathered rows), the `adaptive_gated` architecture
  and the weighted-fusion canonical values (w_cnn 0.7, fixed logits)
  changed. Results and checkpoints from <= 2.7.0 are NOT comparable
  with 2.8.0. Shipped in the same PR as 2.7.0 (fixed-dim projection),
  tagged separately (v2.7.0 at fc3a54c, v2.8.0 at 27a344f).

## [2.7.0] - 2026-08-22

### Added

- **Optional fixed linear projection of every extractor to one common
  dimension**, configured under `projection:` in
  `configs/extractors.yaml`:

  ```yaml
  projection:
    method: pca # none (default) | random | pca
    dim: 128
    seed: 42 # method: random only
  ```

  Writes `<extractor>_p<dim>.npy` **alongside** the native artifact —
  the v2 native-dim contract is untouched and both can be trained and
  compared in the same battery. `random` is a seeded semi-orthogonal
  matrix (QR of a Gaussian; data-independent, so it cannot leak
  validation or test items and reproduces on any machine); `pca` is fit
  on train items only, mirroring `alignment.method: pca`. Any extractor
  overrides the block under `extractors.<name>.projection`, including
  `method: none` to stay native-only.

  This gives the element-wise fusion family (`mean`, `sum`, `prod`,
  `max_pool`, ...) equal-dim sources with nothing learned online: point
  `fusion_extractors` at the projected names and the fuse step consumes
  them directly. The `_p<dim>` token sits before the condition suffix
  (`resnet50_p128.npy`, `resnet50_p128_finetuned.npy`), so one
  `fusion_extractors` entry resolves in both conditions, and `train` /
  `evaluate` discover the projected stems by the same glob that finds
  every other embedding — no registration.

  Projecting an already-extracted catalogue never reloads a backbone:
  the hook reads the source's own `.meta.json` for provenance and
  streams the matrix in 8192-row chunks, so peak memory is a function of
  the chunk, not of the catalogue. The projector itself is persisted as
  `<extractor>_p<dim>.proj.npz` plus a `.proj.json` (method, dim, seed,
  fit set), and the artifact's sidecar records `source_native_dim`
  alongside the projected `native_dim` — which is also what keeps the
  loader's metadata cross-check satisfied.

  **`docs/protocol.md` §1b states the methodological caveat**: a fixed
  projection is an experimental variable, not a free normalisation.
  `method: random` is exactly the seeded random projection §1 rejects as
  a *default*, so a comparison run only on projected artifacts compares
  "backbone × compression". Report it next to the native-dim result from
  the same run.

- **Two flags selecting which variant is consumed**, separate from the
  decision to write it:

  ```yaml
  # configs/recommenders.yaml
  embedding_variants: both # native | projected | both

  # configs/fusion.yaml
  extractor_variants: native # native | projected | both
  ```

  `embedding_variants` filters the training cells at their single source
  of truth (`train._iter_cells`), so `evaluate` — which enumerates from
  the checkpoints `train` produced — follows automatically. The
  non-visual `"none"` pseudo-embedding is never filtered out, so a
  projected-only battery keeps plain BPR. Fusion outputs are classified
  with the sources they were built from, since they carry the same
  token.

  `extractor_variants` expands `fusion_extractors` into the chosen
  family, keeping the logical names in the config. With `projected` the
  sources already share a width, so the entire `alignment:` block is
  bypassed: the equal-dim strategies fuse them directly, with no
  `Linear(D_i -> D)` co-trained by BPR and no PCA fit inside the fuse
  step — the point of projecting at extraction time. Outputs carry the
  variant token (`hybrid_mean` vs `hybrid_mean_p128`) so both families
  coexist in one dataset directory, and `both` runs one pass each. A
  projected variant whose extractors have no projection, or are
  projected to *different* widths, fails loudly rather than fusing
  matrices that do not share a space.

### Changed

- `train_item_indices` — the train-only fit set that anything learned
  from item features must be fit on — moved to `src/utils/splits.py`,
  shared by the fusion PCA alignment and the new projection instead of
  being private to `src/steps/fuse.py`. `fit_pca_on_rows` is now
  exported from the `src.fusions` package rather than reached through
  `src.fusions.strategies`.

## [2.6.4] - 2026-08-22

### Added

- **Per-dataset download timings, with each dataset's weight on disk.**
  `download` was one opaque window covering every configured dataset,
  so "which dataset cost the download hour" was unanswerable. Its loop
  now opens one cell per dataset, labelled `size_mb` (total weight of
  the dataset's raw dir) and `downloaded_mb` (how much of that arrived
  over the network this run — `0.0` when the archive was already
  there). Each cell is flushed as it closes, so a run interrupted
  halfway through the downloads still documents the datasets it did
  fetch.

  Unlike every other step, a `download` cell is recorded even when
  nothing new was fetched: re-validating a multi-gigabyte archive by
  size / checksum is real wall-time, and it is exactly the window a
  reader wants when a "no-op" re-run takes twenty minutes.

  `Cell.label(**labels)` is new, for the values that only exist once
  the work has run (you cannot weigh a dataset before downloading it).
  Existing `time_cell` call sites are untouched.

### Fixed

- **The "saved" log line at the end of `_finetune_and_extract` is
  reachable again.** 2.6.3 inserted the `return True` above it, leaving
  it as dead code, so a successful re-extraction never logged its
  output shape (`ruff` RET503). The log now precedes the return.

- **Per-step timings survive an interrupted run.** The step-level list
  — the only place `download`, `preprocess` and `export_best` are timed
  at all, since they open no cells — was written to `manifest.json`
  exclusively by `finish_run`, i.e. once, at the very end. A run killed
  or crashed before that point left the manifest's `steps` key absent
  and `step_timings.json` holding only the per-cell entries, so the
  download duration was nowhere on disk.

  `record_step` now flushes the accumulated list to a sidecar
  `results/runs/<run_id>/steps.json` after every step, the same way
  `time_cell` already flushed the per-cell list. `finish_run` still
  writes `manifest['steps']` unchanged, and the per-cell
  `step_timings.json` keeps its flat-array format — no consumer of
  either file has to change.

## [2.6.3] - 2026-08-21

### Fixed

- **Full-ranking evaluation no longer OOMs on large catalogues.** A
  hyperparameter grid failed on essentially every `amazon_women` and
  `tradesy` cell — hundreds of `OOM on <job_id>` warnings — while
  `amazon_fashion` passed. The model was never the cause: the same job
  OOM-ed at `latent_dim=64` and `128`, and always immediately after
  `Evaluator initialised`.

  `_evaluate_batched` allocated the `(B, N)` matrix four times over: the
  scores, their tie-break reordering (fp32), the sort permutation and
  the permutation mapped back to item ids (int64). At 24 B per
  `(user, item)` pair plus `torch.sort`'s stable-sort workspace, a fixed
  batch of 512 users needs ~5.7 GB on `amazon_women` (347,591 items) —
  against the ~6.25 GB each of three workers gets from
  `set_per_process_memory_fraction(1/3 + 0.05)` on a 16 GB card.
  `amazon_fashion` (166,270 items) needs ~2.7 GB and fit, which is
  exactly the split the log shows.

  Three changes, none of which alter a single metric:

  - The full-width `order[sorted_perm]` is gone. Only the first
    `max(max_k, 20)` columns of each ranking are ever read, so the
    permutation is sliced *before* being mapped back to item ids — the
    gather drops from `(B, N)` to `(B, 20)`, saving 1.4 GB per batch.
    Taking the head commutes with the element-wise lookup, so the values
    are identical.
  - The `(B, N)` buffers are released as soon as they are dead, instead
    of staying resident until rebound on the next iteration.
  - The user batch is now derived from the catalogue size and the
    process's GPU allowance (`plan_ranking_batch`) rather than fixed at
    512, which the caller's value now merely bounds. Every row is
    ranked, masked and scored independently, so batching is a pure
    execution detail: `tests/test_ranking_batch_budget.py` pins that a
    one-user batch produces the same per-user frame as a single big one.

- **OOM retries now actually reduce something.** `TrainingOrchestrator`
  incremented `retry_count` and put the *same* job back on the queue, so
  a cell that overflowed once overflowed again on every attempt before
  being declared unrecoverable. The worker now derives a ranking budget
  from its own GPU allowance — a cap `torch.cuda.get_device_properties`
  cannot report, so it is passed explicitly through `train_single_run`
  to the `Evaluator` — and halves it per retry, which halves the user
  batch that overflowed. Retries are still deferred to a sequential pass
  at the end of the grid; that pass runs without the per-process cap, so
  it also gets the whole device.

## [2.6.2] - 2026-08-21

_No separate git tag: shipped inside `v2.6.3` (2026-08-21)._

### Changed

- **Chunked, memory-mapped execution of the offline fusion strategies**
  (`src/fusions/streaming.py`). The in-memory strategies hold every
  source matrix, an L2-normalised copy of each, and the fused result at
  once — ~11.5 GB for a single `concat` on `amazon_women` (resnet50
  2.85 GB + vit_b16 0.99 GB, their normalised copies, and a 3.84 GB
  output). `concat`, `pca` and `pca_per_model` now open their sources
  with `mmap_mode="r"`, process 8192 rows at a time and write straight
  into a memory-mapped `.npy` via the new
  `atomic_io.atomic_np_memmap_save`. Peak memory becomes a function of
  the chunk size rather than the catalogue size: ~200 MB per worker for
  `concat`, down from ~11.5 GB. The `alignment.method: pca` route gets
  the same treatment through `stream_pca_align`, which previously loaded
  every native matrix into the *parent* process at once.

  **The numbers do not change.** L2 normalisation and concatenation are
  row-wise, so `concat` is bit-identical. The PCA strategies fit on an
  identically-assembled matrix through the shared
  `strategies.fit_pca_on_rows` (extracted from `_fit_pca_train_only`,
  which keeps its behaviour), so the components are identical; only the
  `transform` is chunked, and each output row depends solely on its own
  input row. `tests/test_fusion_streaming.py` pins the equivalence
  against the in-memory functions. Strategies without a row-wise
  decomposition — including anything from `plugins/fusions/` — keep
  running through the registry's in-memory path, which remains the
  contract every strategy is written against.

  The pool sizing added in 2.6.1 is now regime-aware: streamed tasks are
  charged for one chunk (plus the PCA fit matrix, the one allocation an
  exact fit cannot avoid), so a catalogue-sized `concat` no longer forces
  the pool down to a single worker.

### Fixed

- **`pca_per_model` now honours its configured component count.**
  `_expand_pca_per_model` emitted the kwarg under the YAML spelling
  (`n_components_per_model`), but `fuse_pca_per_model` consumes
  `n_components`. The mismatch left the value in `**kwargs`, where
  `_warn_ignored_kwargs` discarded it, and every task fell back to the
  signature default of 64. The documented sweep
  `n_components_per_model: [32, 64, 128]` therefore wrote three
  *identical* 64-dim matrices under `_nc32`, `_nc64` and `_nc128` — a
  variable that appeared to be swept and never varied. The YAML key is
  unchanged; only the kwarg the grid emits was corrected.

  Outputs are unaffected under the shipped `configs/fusion.yaml`, whose
  single value (64) already matched the default that was being used.
  Any run that configured a *different* value produced embeddings that
  do not match their filename and must be re-fused.

## [2.6.1] - 2026-08-21

_No separate git tag: shipped inside `v2.6.3` (2026-08-21)._

### Fixed

- **Worker pools are sized against host memory, not just CPU count.** A
  full run could exhaust system RAM and take the host down with it: the
  fusion step sized its `ProcessPoolExecutor` at `min(len(pending),
  os.cpu_count())`, and each worker loads every source `.npy` fully into
  memory plus a fused output of comparable size. On a 16-core host with
  the reference catalogues that is 12 workers holding ~14 GB each. The
  kernel log from the incident shows `constraint=CONSTRAINT_NONE,
  global_oom` — because the container ran without a memory limit, the
  OOM killer was not confined to the run and reaped the host's desktop
  session instead.

  New `src/utils/memory.py` centralises the memory budget (cgroup v2 ->
  cgroup v1 -> host RAM -> 4 GB fallback, previously private to
  `src/utils/dataloader.py`) and adds `plan_pool_workers()`, which
  clamps a caller's CPU/task cap to what actually fits after a 4 GB
  reserve for the parent process and the host. `src/steps/fuse.py`
  estimates each task's peak from its source file sizes (sidecar-only
  tasks cost nothing) and sizes the pool accordingly;
  `detect_max_workers()` and `TrainingOrchestrator` gained an optional
  `per_worker_bytes`, which `src/steps/train.py` fills from the largest
  embedding matrix and interaction file across the pending jobs. The
  DataLoader autotune is unchanged, it now shares the budget helper.

- **The container can no longer take the host's memory.** All three
  compose services declare `mem_limit` / `memswap_limit`
  (`${PRISM_MEM_LIMIT:-24g}`), so an overshoot kills a worker inside the
  container instead of triggering a global OOM. The limit is also what
  lets the in-container sizing heuristics read a real cgroup budget
  rather than the host's total RAM.

### Changed

- **Work that was already done is no longer timed or costed.** Re-running
  a finished pipeline appended a cell to `step_timings.json` and a full
  telemetry window to the manifest for every `(dataset, extractor)` that
  merely found its `.npy` on disk, crediting the run with a fraction of a
  second and zero energy for an extraction that had cost an hour
  earlier. `time_cell` now yields a `Cell` handle whose `skip()` leaves
  no entry behind; `extract` and `finetune` call it when their outputs
  already exist, and `evaluate` / `evaluate_finetuning`, which
  short-circuit before the timer starts, call `timing.note_skipped_cell()`
  instead. A step whose cells were *all* skipped is recorded as
  `"skipped": true` with no telemetry block. Steps that emit no cells at
  all (`download`, `preprocess`, `report`) are timed exactly as before.

## [2.6.0] - 2026-08-21

### Added

- **Per-step throughput and cost telemetry.** Every pipeline step and
  every profiled cell now records, alongside its wall-time, how fast it
  ran and what it cost. A background sampler (`src/utils/telemetry.py`,
  1 Hz by default) reads GPU utilisation / power / memory and
  process-tree CPU / RSS; each step reports the min, max and mean of
  those series over its own window plus the trapezoidal integral of GPU
  power as `energy_joules`. Download steps additionally report
  `network_mb_per_s`; model steps report `flops_per_s` and
  `items_per_s`. The manifest gains a run-level `telemetry` rollup
  (total energy, total FLOPs, bytes downloaded) and names the probe
  backends that produced the readings, so a manifest is self-describing
  about its own measurement quality.

  FLOPs are accounted analytically: the first batch through a given
  `(model, input shape)` is measured once under torch's
  `FlopCounterMode` and the hot loop then multiplies that per-sample
  constant by the number of samples (`src/utils/flops.py`). Leaving the
  dispatch counter active would cost 10-30 % of pipeline wall-clock for
  a number that is fixed in advance, since every backbone here runs at
  a fixed resolution. The measured constants are recorded in the
  manifest so the arithmetic is auditable. Training batches are charged
  the customary `3 x` forward approximation, flagged as such rather
  than presented as measured.

  Coverage includes the battery runner, which does not go through
  `main._run_single`: it starts its own sampler and records a
  `telemetry` block per cell in `results/battery/manifest.json`,
  alongside the per-cell duration already tracked there.

  One documented limitation: throughput *counters* (FLOPs, items,
  bytes) are incremented in-process, so work done inside forked workers
  — Optuna inter-cell parallelism, DataLoader workers — does not reach
  them; a parallel `train` step reports accurate energy and utilisation
  but no `items_per_s`. The sampled *gauges* (GPU, CPU, RSS) do cover
  the whole process tree. This mirrors the existing constraint on
  per-cell timings for parallel HP search.

  Enabled by default and configurable under `telemetry:` in
  `configs/default.yaml`. A new `telemetry` extra
  (`pip install -e .[telemetry]`, already in the Docker image) installs
  NVML and psutil for in-process probing; without it telemetry degrades
  to a long-lived `nvidia-smi` reader and `getrusage` rather than
  switching off. Complements — does not replace — the optional
  codecarbon integration, which reports whole-system energy and CO2 for
  the whole run rather than GPU energy per step. See
  `docs/observability.md`.

- **GPU reservation in `docker-compose.yml`.** The `pipeline` and
  `shell` services now declare a `deploy.resources.reservations.devices`
  entry for the NVIDIA driver. The file previously assumed a host whose
  Docker daemon uses `nvidia-container-runtime` *as its default runtime*
  (RunPod, lab servers); on a workstation with the toolkit installed the
  normal way, no GPU was passed through and `device: "auto"` silently
  resolved to `cpu`.

## [2.5.0] - 2026-08-05

### Changed

- **PyTorch pinned to the 2.8 series (Blackwell support).** ``torch`` moved
  from ``>=2.1.0,<2.6`` to ``>=2.8,<2.9`` and ``torchvision`` from
  ``>=0.16.0,<0.21`` to ``>=0.23,<0.24``. The previous ceiling predates
  ``sm_120``, so the pipeline aborted with "no kernel image is available
  for execution on the device" on Blackwell cards (RTX 5090, RTX PRO
  6000) — the current price/throughput sweet spot on interruptible cloud
  marketplaces. 2.8 is the earliest range that ships those kernels; the
  window is deliberately narrow because it is exactly what was
  re-validated. Cloud image guidance in ``README.md`` updated to require
  CUDA 12.8+.

  Re-validation performed on macOS/arm64 (CPU + MPS), comparing a clean
  end-to-end ``--all --config-dir configs/smoke`` run on each stack:

  - full suite green on both (519 passed, same single pre-existing
    ``scipy`` warning, no new deprecations);
  - each stack is reproducible run-to-run (identical embedding digest
    across repeated clean runs);
  - frozen ``resnet50`` embeddings differ between stacks by float32
    rounding noise only — ``max |Δ| = 1.43e-06`` against ``max |x| =
    8.02`` (relative ``1.78e-07``, ≈1 float32 ULP), ``mean |Δ| =
    1.9e-09``, per-item ``cosine ≥ 0.9999999``, ``allclose(rtol=1e-5)``
    true;
  - every downstream table under ``results/smoke/tables`` (evaluation,
    bootstrap CIs, Friedman, pairwise) is **bit-identical** across the
    two stacks.

  Only ``torch``, ``torchvision`` and ``sympy`` changed in the lock file —
  ``timm``, ``transformers``, ``numpy`` and the rest resolve unchanged.
  No battery has been run under the 2.x protocol yet, so no existing
  result artifact is invalidated by this change.

  Not covered by this validation: ``sm_120`` kernels themselves, which
  have no macOS wheel. First run on Blackwell hardware must be smoke-
  tested before committing to a full battery.

## [2.4.4] - 2026-07-15

### Fixed

- **Battery runbook translated to English.** ``docs/battery_runbook.md``
  had been written in Portuguese, violating the docs-in-English rule; its
  content is now fully in English (commands, paths, and structure
  unchanged).

## [2.4.3] - 2026-07-15

### Added

- **Battery covers the finetuned condition.** The cell enumerator only
  produced frozen cells; it now enumerates both frozen and finetuned per
  ``pipeline.condition`` (``both`` → both) via ``_iter_cells``. Finetuned
  cells carry a distinct ``_finetuned`` embedding stem, so their study /
  checkpoint / F artifact are separate from the frozen twin, and BPR
  (feature-blind) is not duplicated. Finetuned cells appear only when the
  finetuned features exist on disk (run the ``finetune`` step first).
  Runbook updated.

## [2.4.2] - 2026-07-15

### Fixed

Three gaps in the production battery ``execute_cell`` path, surfaced by a
CPU pilot on synthetic data.

- **Replay checkpoint isolation.** The Optuna study is shared across
  seeds (D2), but ``_best.pt`` was not seed-isolated, so a replay whose
  validation metric fell below the search seed's would silently keep the
  search seed's checkpoint (``_save_best_model`` only overwrites on
  ``>=``). ``execute_cell`` now trains and reads under a seed-suffixed
  results dir; the F artifact still lands in the shared, seed-keyed
  ``per_user`` directory.
- **Fusion cell resolution.** The battery enumerator used the fusion
  strategy name (``mean``) as the visual config, but the fused artifact
  is ``hybrid_mean_*`` — so every fusion cell failed to resolve its
  embedding/checkpoint. ``enumerate_cells`` now reuses the pipeline's
  cell discovery (``_iter_cells``), so the visual config is the real
  embedding stem used everywhere (training, eval, checkpoint, artifact).
- **Extraction checkpoint path.** It was hardcoded to
  ``checkpoints/extraction/``, ignoring ``paths.checkpoints`` — a run
  under one profile could resume from another profile's stale extraction
  state. It now honours the configured checkpoints path.

## [2.4.1] - 2026-07-15

Follow-up corrections. No behaviour or artifact change.

### Changed

- **Per-user records now come from a single scoring pass.** The Task F
  per-user sufficient statistic (rank / n_candidates / tie_block_size /
  top-20) was recomputed in a second scoring pass at final evaluation.
  The three ranking paths now emit these as ``_``-prefixed columns in the
  one metric pass; ``evaluate_with_records`` returns
  ``(metrics, records)`` from that single pass (metrics still drop the
  diagnostics). ``tie_block_size`` is reused, not recomputed.
- **``train_single_run`` selection kwarg renamed** ``test_interactions``
  → ``selection_interactions`` (it carries the VALIDATION held-outs used
  for model selection). The Evaluator's own ``test_interactions`` param
  is unchanged (accurate on the final-eval path).
- **CHANGELOG 2.3.0 comparability note** amended: results from ≤ 2.2.7 are
  not comparable for two independent reasons — model selection moved from
  test to validation (HP search was previously selected on the test set),
  and the tie-break changed.

## [2.4.0] - 2026-07-15

Battery-readiness build: four new capabilities to run the full battery
on interruptible cloud instances. No change to how models train/evaluate
beyond persistence and gating.

### Added

- **Feature sanity gate (G).** ``src/steps/validate_features.py`` +
  ``--validate-features`` command: validate every feature matrix (shape,
  native dim from the extractor registry, dtype, NaN/Inf, zero-norm
  rows) with stats logging; fail loud, never warn. An automatic gate
  runs at the entry of ``fuse`` (native inputs) and ``train`` (all
  consumed backbone + fused ``.npy``).
- **Per-user persistence (F).** Final evaluation writes a per-cell
  artifact (``results/per_user/<dataset>/<key>.csv.gz`` + metadata JSON)
  holding the held-out rank (a sufficient statistic under LOO),
  n_candidates, tie_block_size and top-20 items. ``derive_metrics``
  recomputes any metric at any k without a GPU (equivalence to the
  online Evaluator is tested); ``paired_loader`` assembles the
  users×systems matrix and refuses to intersect mismatched user sets.
  Format is csv.gz (avoids the pyarrow dependency).
- **HP-search budget fairness (H).** ``src/recommenders/hp_budget.py``:
  one shared protocol budget per dataset (n_trials/selection metric/
  patience/epochs/eval_sample_size), a guard-rail that rejects any
  per-model budget key, and ``train_replay`` — the D2 "train one fixed
  config, no search" entry point. D1 (final eval consumes the best-trial
  checkpoint) documented; no post-search retrain.
- **Battery runner (I).** ``src/battery/``: declarative cell enumerator
  (BPR once per dataset/seed; AVBPR excluded; DeepStyle on Tradesy;
  primary seed searches, others replay), an inspectable JSON state
  manifest with idempotency + resume + failure isolation + retry, cost
  projection, git/duration metadata, and ``--battery`` / ``--battery-status``
  commands. Optuna storage migrated to persistent SQLite for resume.
  Smoke + resume + failure/retry tests; ``docs/battery_runbook.md``.

### Changed

- ``docs/protocol.md`` §10.6: uniform HP-search budget per dataset,
  selection in validation, search on the primary seed + best config
  replayed on the others, best-trial checkpoint, test untouched.

## [2.3.0] - 2026-07-15

Evaluation-protocol changes decided by the two diagnostic audits. They
change model selection and tie-breaking; no battery had run yet, so
nothing is invalidated retroactively — but see the tie-break note below
on cross-version comparability.

### Changed

- **Model selection runs on validation, never on the test set.** The
  Optuna trial path (`src/steps/train.py`) loaded `test.csv` and chose
  hyperparameters + the early-stopping epoch by maximising ndcg@10 on a
  test-user subsample — an optimistic bias. It now loads `val.csv` and
  scores validation held-outs (train items as the mask), matching the
  grid worker (`src/utils/parallel.py`) which already did so. The test
  set is read only by the final evaluate step. The ~2000-user selection
  subsample keeps its dedicated-RNG mechanics, now over the validation
  population.
- **Exact-score ties broken by a seed-fixed random permutation.**
  Previously ties were broken by lower item id; a pod check found
  item_idx correlates with popularity (Spearman -0.34 to -0.45 across
  the 4 datasets), so an id tie-break systematically favoured popular
  items inside a tie block — hurting pure BPR (mass cold-item ties) more
  than visual models. All three ranking paths
  (`src/evaluation/protocol.py`) now break ties by a permutation drawn
  once per run from the global seed, shared by every model/trial of a
  (dataset, seed) run. When the held-out is not tied the rank is
  unchanged.

  **Comparability note (two independent reasons results from ≤ 2.2.7 are
  not comparable):** (1) model selection moved from the test set to
  validation — any number produced with an HP search on ≤ 2.2.7 selected
  hyperparameters/early-stopping on the test set and is optimistically
  biased; and (2) evaluations that contain exact-score ties now break
  them by a seeded permutation instead of by item id.

### Added

- **Exact-tie instrumentation.** Each evaluation logs the fraction of
  held-outs sitting in an exact-score tie plus the mean/max tie-block
  size (`src/evaluation/protocol.py`), turning the audit's unmeasured
  tie frequency into a number recorded during the battery.
- Guards/tests: model selection reads `val.csv` only (structural +
  behavioural), validation-population subsample determinism (updated),
  tie-break permutation determinism, non-tied-rank invariance,
  tied-rank follows the key and varies by seed, and tie instrumentation.

## [2.2.7] - 2026-07-15

### Changed

- **Live in-place progress bars across the pipeline.** The compose
  ``pipeline`` service now allocates a pseudo-TTY (``tty: true`` +
  ``stdin_open: true``), so tqdm renders live redrawing bars for every
  long step — download, image extraction, feature extraction,
  evaluation, and training. Follow them with a raw stream (``docker
  logs -f prism-vrec`` or ``docker attach prism-vrec``), **not** ``docker
  compose logs``, whose per-line ``service |`` prefix buffers the ``\r``
  the bar uses to redraw.
- **Training shows an Optuna-cell bar.** ``_run_optuna`` now drives a
  ``tqdm`` bar (``Training (Optuna cells): 145/580 [1.2h<3.6h,
  1.9cell/h]``) instead of periodic log lines, giving %, elapsed, ETA
  and rate on a single redrawing line. Parent-side observability only —
  worker computation, RNG and cell ordering are untouched.
- **Download reverts to the tqdm bar.** The per-line percentage logging
  added in 2.2.5/2.2.6 (a non-TTY workaround that appended a line every
  15 s) is removed; under the new TTY the native bar redraws in place.
  Off a TTY the bar auto-disables (``disable=None``), keeping CI quiet.

## [2.2.6] - 2026-07-15

### Fixed

- **Download progress frozen at the resume offset (regression from
  2.2.5).** 2.2.5 read ``pbar.n`` for the throttled log line, but the
  tqdm bar auto-disables off a TTY (``disable=None``) and a disabled
  bar freezes ``pbar.n`` at ``initial`` — so the log showed ``0.0% (0 /
  N MB)`` on a fresh download and ``X% (resume_offset MB)`` on a resume
  for the entire transfer, even though bytes were flowing to disk. The
  progress line now tracks its own byte counter
  (`src/data/dvbpr.py`), so the percentage advances in non-TTY logs
  regardless of tqdm's enabled/disabled state.

## [2.2.5] - 2026-07-15

### Fixed

- **Download progress invisible in non-TTY logs.** The dataset download
  (`src/data/dvbpr.py`) rendered progress only through a tqdm bar, whose
  ``\r`` in-place updates do not show in Docker Compose / pod / nohup
  logs — users saw the "Resuming download…" line but no progress. The
  bar now auto-disables off a TTY (``disable=None``) and a throttled
  ``logger`` line (every 15 s: ``amazon_fashion: 47.8% (2102 / 4400
  MB)``) is emitted through the normal logging handlers, so progress is
  visible in both interactive terminals and captured logs.

## [2.2.4] - 2026-07-15

### Added

- **Battery progress + ETA for the Optuna backend.** The parallel
  Optuna search (`src/steps/train.py::_run_optuna`) now logs a
  cell-level forecast — ``Progress: 145/580 cells (25.0%) | 3 workers |
  ETA: ~3.6 h`` — every 30 s and on each cell completion, bringing it to
  parity with the grid orchestrator's existing progress line
  (`src/utils/parallel.py`). Parent-side observability only: it never
  touches worker computation, RNG, or cell ordering, so runs stay
  bit-identical. The grid backend already reported this.

## [2.2.3] - 2026-07-15

Documentation only; no code change.

### Fixed

- **Citation version.** The README BibTeX still pinned ``version =
  {2.1.0}``, telling users to cite an outdated release. Bumped to the
  current version and documented the ``dataset_contracts`` config block
  (added in 2.2.0) in the ``configs/default.yaml`` section. Version
  reconciled across ``pyproject.toml`` / ``CITATION.cff`` / ``uv.lock``
  / README so the declared version matches the tag.

## [2.2.2] - 2026-07-15

Housekeeping; no runtime behaviour change.

### Fixed

- **CI: ``ruff format``.** ``src/fusions/strategies.py`` was left
  unformatted (the multi-arg ``_fit_pca_train_only`` call in
  ``fuse_pca_per_model`` was hand-wrapped), failing ``ruff format
  --check .`` on ``main``. Reformatted (ruff 0.15.17): one argument per
  line. No logic change.

### Changed

- **``data/`` kept in the repo, contents ignored.** Added
  ``data/.gitignore`` that ignores everything under ``data/`` except
  itself, and simplified the root ``.gitignore`` data section
  accordingly, so raw datasets / processed splits / embeddings / smoke
  artifacts stay out of Git while the directory remains tracked.

## [2.2.1] - 2026-07-15

Documentation only; no code change.

### Changed

- **Protocol doc de-versioned.** Renamed ``docs/protocol_v2.md`` ->
  ``docs/protocol.md`` and dropped ``v2`` from the title, since
  versioning belongs to releases, not the methodology. Reworded the
  ``v1.x``-vs-``v2`` historical contrast as "an earlier version of the
  framework" vs "this protocol"; updated the path references in README
  and CHANGELOG. Model names (DINOv2) and the versioned release history
  are untouched.

## [2.2.0] - 2026-07-15

Audit follow-up: hardening guards, a regression test, and doc fixes from
the 4-point diagnostic audit. No experimental protocol changes and no
training artifact is invalidated — the two behaviour changes only turn
previously silent decisions into loud, declared ones.

### Added

- **Explicit per-dataset category contract.** A ``dataset_contracts``
  block in ``configs/default.yaml`` declares ``expects_categories`` per
  dataset (amazon_* = true, tradesy = false), validated by the new
  ``DatasetContract`` schema. ``src.steps.preprocess`` enforces it via
  ``src.data.categories.enforce_category_contract``: a mismatch between
  the declaration and what the provider's ``load_categories()`` returns
  now raises ``CategoryContractError`` instead of silently flipping
  DeepStyle degeneration and fine-tuning transfer. Datasets without an
  entry skip the check (backwards compatible).
- **Regression test for validation-subsample determinism.** Locks the
  invariant that the ~2000-user early-stopping subsample is a pure
  function of ``sample_seed`` (dedicated ``np.random.default_rng`` over a
  sorted population), independent of global RNG state mutated by model
  init / negative sampling — so a future refactor to a shared RNG fails
  CI instead of silently desynchronising validation across trials.

### Changed

- **PCA transductive fallback is now an opt-in error.**
  ``_fit_pca_train_only`` (and ``fuse_pca`` / ``fuse_pca_per_model`` /
  ``pca_align``) raise when ``train_items=None`` unless
  ``allow_transductive=True`` is passed explicitly. A fit over all rows
  is the test→fit leak the native-dim protocol eliminated; it was previously a
  warning (which does not fail CI). No production path passes ``None``
  (``src.steps.fuse`` always supplies train item indices), so behaviour
  there is unchanged; only synthetic tests opt in.

### Fixed

- **DeepStyle documentation.** ``docs/protocol.md`` described the
  removed MLP-projector DeepStyle (no category subtraction, "does not
  degenerate on Tradesy"). Rewritten to the paper-faithful formulation
  ``θ_i = E·f_i − c_cat(i)`` with the analytic degeneration
  to VBPR on category-less Tradesy. Also corrected the contradictory
  ``_ensure_categories_sidecar`` docstring that listed tradesy among
  taxonomy-bearing datasets.

## [2.1.0] - 2026-07-12

Statistical-validity pass (C1-C4): changes what the analysis can claim,
not how models are trained. No training artifact is invalidated.

### Changed

- **Comparison families (C1).** Holm and the Friedman omnibus now run
  WITHIN the family of comparisons a research question defines
  (``src/evaluation/comparison_families.py``), never over the Cartesian
  product of every config — all-pairs Holm over ~77 configs ran with
  ``m ≈ 2900`` and rejected everything artificially. Families:
  ``backbone_within_model``, ``model_within_backbone``,
  ``fusion_within_model``, ``frozen_vs_finetuned``; each output row
  carries ``family``, ``group`` and ``n_comparisons_in_family`` so the
  correction is auditable. ``all_pairs`` remains as an exploratory
  option (and is the smoke profile's setting, whose grid is too small
  for the question-aligned families).
- **Primary metrics under LOO (C2).** Step 07 analyses ``recall`` (≡
  HitRate) and ``ndcg`` by default; ``precision@k = recall@k / k`` and
  ``map@k = 1/rank`` are deterministic transforms under leave-one-out
  and are only analysed with ``include_derived_metrics: true`` — never
  as independent evidence. The derivation is documented in the module
  and in ``docs/protocol.md`` §5.
- **Cliff's delta promoted to primary effect size (C3).** Cohen's d is
  parametric and inflates on zero-dominated paired differences (the
  same property that motivated Wilcoxon ``pratt``); it is now off by
  default (``include_cohens_d``) and documented as diagnostic-only.

### Added

- **Paired-difference bootstrap CI (C4).** Every pairwise row now
  reports ``diff_mean`` / ``diff_ci_lower`` / ``diff_ci_upper``
  (resampling users, seed-fixed) — the CI that must agree with the
  Wilcoxon verdict. Per-config CIs remain as descriptive statistics;
  their overlap does not contradict a significant paired test.
- ``tests/test_comparison_families.py``: family enumeration (no pair
  varies two dimensions; C(n,2) sizes; frozen never paired with
  finetuned within a backbone family), Holm ``m`` = family size,
  diff-CI ↔ Wilcoxon consistency, effect-size policy.

## [2.0.1] - 2026-07-12

Documentation-only release so the Zenodo archive carries README/docs
that match the v2 protocol. No code changes.

### Changed

- README synced with the v2 protocol: native-dim extraction
  (``raw_dim`` table including ConvNeXt-Base), ``projection_dims`` /
  ``embedding_dims`` removed from config examples, fusion ``alignment``
  block documented, DeepStyle paper formula, ACF component artifact
  naming (``<extractor>_comp.npy``), ``eval_sample_size`` example.
- ``docs/hp_search.md`` describes inter-cell Optuna parallelism (B7);
  ``docs/learned_fusion.md`` uses v2 sidecar naming and mentions
  ``RaggedSources``.
- Citation policy: ``CITATION.cff`` and the README bibtex cite the
  concept DOI (always the latest version); per-version DOIs remain
  available on Zenodo (v2.0.0: 10.5281/zenodo.21325967).

## [2.0.0] - 2026-07-12

**Breaking: new experimental protocol.** Every 1.x embedding, checkpoint
and result table is incompatible and must be regenerated. 1.x results
remain traceable via the ``v1.1.2`` tag and the Zenodo archive.

### Changed (protocol)

- **Native dimensionality at extraction (Mudanca 1).** Extractors now
  save the backbone's native pooled feature (ResNet-50 2048, ConvNeXt
  1024, ViT-B/16 / CoAtNet-0 / DINOv2 768, LeViT-256 / CLIP 512, CvT-13
  384). The v1.x shared ``Linear+ReLU`` projection - which in the frozen
  condition was an **untrained seeded random projection** - is gone; the
  learned projection ``E`` inside each recommender (which already
  existed) maps native -> ``d`` (``common.visual_dim``), trained by BPR
  with the backbone frozen. Native dims are read from the model via a
  probe forward, validated against ``configs/extractors.yaml``
  (``raw_dim``, which had LeViT-256 wrong: 384 -> **512**). Artifacts are
  ``<extractor>.npy`` + ``<extractor>.meta.json`` sidecar (backbone,
  native dim, extraction point, exact weights id, transform recipe);
  the loader cross-checks features against the sidecar and fails loudly
  on mismatch. ``projection_dims`` is removed from config/schema.
- **Canonical per-backbone preprocessing (Mudanca 1b)** - fixes the
  worst silent bug in 1.x: transforms are resolved from the library
  that ships the weights. ViT-B/16 (augreg2) and CoAtNet-0 (sw_in1k)
  normalise with 0.5/0.5/0.5, **not ImageNet** as 1.x applied; all timm
  recipes use bicubic resize+crop_pct, not direct bilinear resize.
  Declared extraction points: CLIP = 512 projected (``encode_image``
  space), CvT-13 pooled = CLS token, ViT/DINOv2 = CLS token. Pinned by
  ``tests/test_canonical_transforms.py``.
- **Fusion with native sources (Mudanca 4).** The equal-dim strategy
  family gets a configurable alignment (``alignment.method``):
  ``learned`` (default) - per-source ``Linear(D_i->D)`` co-trained via
  BPR (``LearnedAlignmentFusion``, ragged concat buffer + JSON sidecar);
  or ``pca`` - offline per-source PCA. The concat family operates on
  native dims (``concat`` -> 2816). **Every PCA (joint, per-model and
  alignment) now fits exclusively on items with a training interaction**
  - fitting on the full catalogue leaked test-item structure - with
  fixed seed and logged cumulative explained variance.
  ``pca_per_model`` is documented as concatenation (-> ``M*k``).
- **Deterministic tie-breaking.** The three ranking paths disagreed
  under score ties (backend-dependent ``topk``, arbitrary
  ``argpartition`` boundaries, pool-order bias in the sampled path that
  favoured positives). All paths now share one rule: stable sort, ties
  broken by lower item id.
- **Wilcoxon ``zero_method="pratt"``.** Per-user LOO metrics are
  0/1-heavy; scipy's default dropped all zero differences, shrinking
  the effective sample far below ``n_users``. Pairwise tables now
  report ``n_pairs`` and ``n_nonzero_pairs``.

### Changed (models — Parte A)

- **DeepStyle per the paper (Liu et al., SIGIR 2017):** linear embedding
  ``E`` (no MLP) and a **learned category embedding subtracted** from the
  item's visual style vector. Models declare ``wants_categories``; the
  pipeline wires an item→category index array (``src/data/categories.py``)
  through grid workers, Optuna trials and evaluation. Datasets without
  category labels (Tradesy) degenerate to VBPR **by design** and this is
  logged. The 1.x MLP variant — responsible for the 9-17x training-time
  outlier in the efficiency audit — is gone.

### Performance (Parte B — no protocol change unless stated)

- **B1** ``ACF.predict_batch`` vectorized (user×item tiles + component
  hidden-state cache); rankings identical (allclose 1e-5) to the per-user
  Python loop it replaces. Was the audit's #1 bottleneck.
- **B2** Immutable buffers (visual embeddings, interaction history)
  excluded from checkpoints (``persistent=False``): 5.7x smaller
  checkpoints on the synthetic dataset.
- **B3** Per-model ``train_s``/``eval_s`` timing instrumentation.
- **B4** VNPR's first MLP layer factored in ``predict_batch`` (declared
  not bit-identical; metric-affecting swaps only in <1e-6 score gaps,
  metrics verified exactly equal).
- **B5** BPR negative sampling vectorized per epoch (``BPRBatchSampler``:
  one shuffle + bulk draw + vectorized collision redraw via
  ``torch.isin``): 7.1x faster than the per-sample rejection loop at
  amazon_fashion scale. Negative sequence differs from 1.x (accepted:
  v2 re-runs every battery).
- **B6** Training-time validation on a fixed, deterministic 2000-user
  subsample shared by every model/trial of a dataset
  (``common.eval_sample_size``). Final reported metrics still rank the
  full test set.
- **B7** Optuna search parallelized across cells (spawn worker pool,
  cap 3, per-process CUDA memory fraction, completed-cell skip);
  trials within a cell remain sequential for TPE.

### Fixed

- **Frozen-BatchNorm corruption during fine-tuning (Parte C).**
  ``FineTuner.train`` used a bare ``model.train()``, so BatchNorm layers
  in the FROZEN stages kept re-estimating ``running_mean/var`` on
  fine-tuning data while their weights stayed frozen (measured drift up
  to 12.65 on LeViT-256's stem BN after one epoch) — re-extraction then
  ran with corrupted stats. LeViT-256, the only BN-everywhere backbone
  of the eight, degraded hardest; LayerNorm backbones were unaffected.
  Frozen-stage norms are now pinned to eval mode every epoch (generic
  prefix rule, no per-backbone branch), and the train loader drops a
  degenerate size-1 tail batch. Hypotheses H1-H5 (head replacement,
  distillation forward, BN crash at batch 1) were empirically refuted.
- Single-component online-fusion sidecars (smoke profile) no longer
  crash ``load_embedding``; fusion degenerates to a passthrough with a
  warning. Empty sidecars still fail loudly.

### Added

- ``docs/protocol.md`` - every methodological declaration (CLIP 512,
  CvT CLS token, resolution posture, PCA protocol, DeepStyle variant
  without category subtraction, ACF fed with real components, the
  architecture-vs-pretraining confounder) with code pointers.
- Provenance columns in every recorded result: ``protocol``,
  ``visual_input_dim``, ``n_trainable_params``.
- ``alignment`` config block (schema-validated), ``RaggedSources`` /
  ``LearnedAlignmentFusion``, per-artifact ``.meta.json`` sidecars.

### Removed

- ``projection_dims`` and ``embedding_dims`` configuration (extraction
  is single-pass native; the dim filter survives only for fusion
  artifacts, which carry an explicit alignment-dim token).

## [1.1.2] - 2026-07-11

Quality pass across the whole codebase (driven by a full multi-agent
audit). No change to the experimental protocol: every model refactor was
verified byte-identical (state_dict keys, seeded weights, forward /
predict / component outputs) before landing.

### Fixed

- **Parallel OOM retries were silently dropped.** ``TrainingJob.job_id``
  used Python's per-process-salted ``hash()``, so the id computed inside
  a spawned worker never matched the parent's copy and OOM'd jobs were
  never requeued. Now derived from a deterministic ``hashlib`` digest.
- **Empty image datasets no longer "succeed" silently.** A wrong or
  unmounted ``image_dir`` was swallowed by a bare ``except``; extraction
  then wrote a degenerate ``.npy`` (skipped forever as "already exists")
  and fine-tuning ran 0 batches, early-stopped at ``val_acc=0``, and
  saved untouched weights labelled as fine-tuned. Both paths now log and
  raise.
- **DINOv2 ``torch.hub`` load is pinned to a commit** instead of tracking
  the remote default branch, closing a reproducibility hole (verified
  byte-identical to the previously cached checkout).
- **Fine-tuning resume is now bit-identical.** The resume checkpoint
  persists and restores RNG + GradScaler state, so an interrupted-then-
  resumed run draws the same shuffle/augmentation sequence as an
  uninterrupted one. *(Behaviour change for resumed fine-tuning runs.)*
- Fusion strategies warn on unknown ``**kwargs`` (a typo'd hyperparameter
  was silently discarded); ``asserts`` guarding required visual
  embeddings became ``raise`` (asserts are stripped under ``python -O``).
- All durable writes (run manifest, carbon block, fine-tuning
  checkpoints, best-model, grid progress, timing sidecar, reports,
  category CSVs) now go through the fsync+retry ``atomic_io.atomic_write``
  instead of hand-rolled tmp+rename; several silently-swallowed
  exceptions now log.
- ``evaluate._route_targets`` matched ``"finetuned"`` while ``train``
  matched ``"_finetuned"``; both now use one shared rule, so an extractor
  named ``finetuned_*`` can't be mis-routed.
- Plugin data downloads verify ``Content-Length`` before promoting the
  ``.partial`` file; tar extraction falls back gracefully when the
  ``filter="data"`` kwarg is absent (3.11.0–3.11.3).

### Added

- Typed ``common`` recommender-training block and ``k_values`` in the
  config schema (previously untyped via ``extra="allow"``, so a typo
  reverted runs to hidden defaults).
- ``PRISM_RUN_ID`` / ``PRISM_SKIP_CONFIG_VALIDATION`` env vars (legacy
  ``HVR_`` names still honoured).
- Direct unit tests for ``metrics.py`` (hand-computed values) and model
  contract tests for bpr/vbpr/avbpr/deepstyle/vnpr; a ``slow`` pytest
  marker isolates backbone-downloading tests, with a dedicated CI job
  that caches the weights.

### Changed

- **Deduplicated the model layer** with no behavioural change: the eight
  extractors now share ``BaseExtractor`` boilerplate via a ``backbone_cls``
  hook (−137 lines), and vbpr/avbpr/deepstyle share a
  ``LinearVisualScoreMixin`` (−138 lines). Both verified byte-identical.
- The three plugin ``auto_register`` scanners collapse into one
  ``utils.plugin_scan``; the filename-routing tokens
  (``_finetuned``/``_comp``/``hybrid_``/``_best``) and the checkpoint-stem
  parser are centralised in ``utils.artifact_names``; ``build_job_list``
  and ``_list_cells`` share one cell-enumeration generator (verified
  identical job ordering).
- Config comments translated to English; CI ``ruff`` pinned to the
  pre-commit version.

### Removed

- Dead ``src/data/preprocessing.py`` (``kcore_filter`` /
  ``leave_one_out_split`` / ``build_mappings``): exported but unused, and
  ``leave_one_out_split`` predated the 3-way split protocol.

## [1.1.1] - 2026-06-18

### Changed

- Simplified the ACF comment in ``configs/recommenders.yaml``: ``acf``
  now just appears in the "valid names" list like the other
  recommenders, with no inline explanation (and in English).

## [1.1.0] - 2026-06-18

### Added

- **ACF recommender (Attentive Collaborative Filtering, Chen et al.,
  SIGIR 2017)** with both attention levels.  *Component-level* attention
  weights an item's ``M`` pre-pool components (spatial cells / patch
  tokens); *item-level* attention weights the user's training history to
  build the augmented user profile.  Scored in the framework's
  BPR-pairwise form ``p̂_u·(γ_l+v_l)+β_l``.  Registered as built-in
  ``acf`` (``src/recommenders/acf.py``).  Three additive, defaulted
  contract extensions keep every existing model bit-identically
  reproducible: ``BaseRecommender`` gains an optional
  ``train_interactions`` constructor arg plus ``wants_history`` /
  ``consumes_raw_components`` class flags, and ``RecommenderSpec`` gains
  ``requires_components``.  Component artifacts
  (``<extractor>_D<dim>_comp.npy``, shape ``(n_items, M, D)``) are routed
  only to ``acf`` and excluded from the pooled pool used by every other
  recommender.  The user history is built train-only, so validation/test
  never leak into the profile.
- **Component feature extraction**.  ``BaseExtractor`` gains
  ``supports_components`` / ``_forward_components`` /
  ``extract_components_batch`` / ``save_components``.  All eight
  extractors expose their pre-pool components (``M`` per backbone):
  ResNet-50 / ConvNeXt-Base / CoAtNet-0 / CLIP ViT-B/32 = 49,
  ViT-B/16 / CvT-13 = 196, LeViT-256 = 16, DINOv2 ViT-B/14 = 256 — each
  projected through the same trainable ``projection`` as the pooled
  path.  Opt-in via ``extract_components: true`` in
  ``configs/default.yaml`` — the pooled extraction path is unchanged and
  byte-identical when the flag is off.
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

[Unreleased]: https://github.com/lucas-couto/prism-vrec/compare/v2.12.0...HEAD
[2.12.1]: https://github.com/lucas-couto/prism-vrec/compare/v2.12.0...v2.12.1
[2.12.0]: https://github.com/lucas-couto/prism-vrec/compare/v2.11.0...v2.12.0
[2.11.0]: https://github.com/lucas-couto/prism-vrec/compare/v2.10.0...v2.11.0
[2.10.0]: https://github.com/lucas-couto/prism-vrec/compare/v2.9.3...v2.10.0
[2.9.3]: https://github.com/lucas-couto/prism-vrec/compare/v2.9.2...v2.9.3
[2.9.2]: https://github.com/lucas-couto/prism-vrec/compare/v2.9.1...v2.9.2
[2.9.1]: https://github.com/lucas-couto/prism-vrec/compare/v2.9.0...v2.9.1
[2.9.0]: https://github.com/lucas-couto/prism-vrec/compare/v2.8.3...v2.9.0
[2.8.3]: https://github.com/lucas-couto/prism-vrec/compare/v2.8.2...v2.8.3
[2.8.2]: https://github.com/lucas-couto/prism-vrec/compare/v2.8.1...v2.8.2
[2.8.1]: https://github.com/lucas-couto/prism-vrec/compare/v2.8.0...v2.8.1
[2.8.0]: https://github.com/lucas-couto/prism-vrec/compare/v2.7.0...v2.8.0
[2.7.0]: https://github.com/lucas-couto/prism-vrec/compare/v2.6.4...v2.7.0
[2.6.4]: https://github.com/lucas-couto/prism-vrec/compare/v2.6.3...v2.6.4
[2.6.3]: https://github.com/lucas-couto/prism-vrec/compare/v2.6.0...v2.6.3
[2.6.2]: https://github.com/lucas-couto/prism-vrec/releases/tag/v2.6.3
[2.6.1]: https://github.com/lucas-couto/prism-vrec/releases/tag/v2.6.3
[2.6.0]: https://github.com/lucas-couto/prism-vrec/compare/v2.5.0...v2.6.0
[2.5.0]: https://github.com/lucas-couto/prism-vrec/compare/v2.4.4...v2.5.0
[2.4.4]: https://github.com/lucas-couto/prism-vrec/compare/v2.4.3...v2.4.4
[2.4.3]: https://github.com/lucas-couto/prism-vrec/compare/v2.4.2...v2.4.3
[2.4.2]: https://github.com/lucas-couto/prism-vrec/compare/v2.4.1...v2.4.2
[2.4.1]: https://github.com/lucas-couto/prism-vrec/compare/v2.4.0...v2.4.1
[2.4.0]: https://github.com/lucas-couto/prism-vrec/compare/v2.3.0...v2.4.0
[2.3.0]: https://github.com/lucas-couto/prism-vrec/compare/v2.2.7...v2.3.0
[2.2.7]: https://github.com/lucas-couto/prism-vrec/compare/v2.2.6...v2.2.7
[2.2.6]: https://github.com/lucas-couto/prism-vrec/compare/v2.2.5...v2.2.6
[2.2.5]: https://github.com/lucas-couto/prism-vrec/compare/v2.2.4...v2.2.5
[2.2.4]: https://github.com/lucas-couto/prism-vrec/compare/v2.2.3...v2.2.4
[2.2.3]: https://github.com/lucas-couto/prism-vrec/compare/v2.2.2...v2.2.3
[2.2.2]: https://github.com/lucas-couto/prism-vrec/compare/v2.2.1...v2.2.2
[2.2.1]: https://github.com/lucas-couto/prism-vrec/compare/v2.2.0...v2.2.1
[2.2.0]: https://github.com/lucas-couto/prism-vrec/compare/v2.1.0...v2.2.0
[2.1.0]: https://github.com/lucas-couto/prism-vrec/compare/v2.0.1...v2.1.0
[2.0.1]: https://github.com/lucas-couto/prism-vrec/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/lucas-couto/prism-vrec/compare/v1.1.2...v2.0.0
[1.1.2]: https://github.com/lucas-couto/prism-vrec/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/lucas-couto/prism-vrec/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/lucas-couto/prism-vrec/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/lucas-couto/prism-vrec/releases/tag/v1.0.0
