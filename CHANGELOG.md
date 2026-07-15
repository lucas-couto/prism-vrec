# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Dates are UTC.

## [Unreleased]

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

[Unreleased]: https://github.com/lucas-couto/prism-vrec/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/lucas-couto/prism-vrec/compare/v1.1.2...v2.0.0
[1.1.2]: https://github.com/lucas-couto/prism-vrec/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/lucas-couto/prism-vrec/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/lucas-couto/prism-vrec/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/lucas-couto/prism-vrec/releases/tag/v1.0.0
