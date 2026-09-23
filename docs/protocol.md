# Experimental Protocol — Declarations

This document records every methodological decision the protocol fixes
explicitly, in the order a reviewer would ask about them. Each item is
implemented in code (pointers included) and must be restated in the
dissertation's methodology chapter.

Evidence index: every reliability change made for 3.0.0 has a task
record under `docs/reliability-sdd/` (defect, reproduction, files,
verification table, limits). Where this document says "verified", the
record is the evidence; where it says "unmeasured" or "unresolved", no
record claims otherwise. All of that verification ran on CPU in the
container; GPU/AMP paths and real-catalogue resource peaks are not
certified by 3.0.0rc1 (see `CHANGELOG.md`).

## 1. Native dimensionality at extraction; learned projection `E` in the recommender

Comparing backbones is only valid if the backbone is the sole variable.
An earlier version of the framework forced every extractor through a
shared `Linear+ReLU` projection to a common dim — and in the frozen
condition that projection was **never trained** (a seeded random
projection), so the benchmark compared "backbone × random compression",
not backbones.

This protocol saves the **native** feature of each backbone at
extraction; the learned projection `E` inside each recommender (VBPR's `E`,
DeepStyle's linear style projection, ACF's component projection) maps
`D_backbone → d`, trained jointly by the BPR loss with the backbone frozen
(fine-tuning end-to-end would be DVBPR, out of scope). `d` is derived
from the shared dimension budget `common.total_dim` (see §7) and is
identical across all backbones of a comparison. VNPR is the exception,
by construction of its paper: it maps the *user* into the image-feature
space (`v_u ∈ R^{D_backbone}`) and consumes `f_i` as is.

| Backbone | Weights (exact) | Extraction point | Native dim |
|---|---|---|---|
| ResNet-50 | torchvision `IMAGENET1K_V2` | global avg pool (after layer4) | 2048 |
| ConvNeXt-Base | timm `convnext_base.fb_in22k_ft_in1k` | global avg pool | 1024 |
| ViT-B/16 | timm `vit_base_patch16_224.augreg2_in21k_ft_in1k` | **CLS token** | 768 |
| CoAtNet-0 | timm `coatnet_0_rw_224.sw_in1k` | global avg pool | 768 |
| DINOv2 ViT-B/14 | torch.hub `facebookresearch/dinov2` (pinned commit) | **CLS token** | 768 |
| LeViT-256 | timm `levit_256.fb_dist_in1k` | pooled final-stage tokens | **512** (the "256" in the name is the stage-1 width) |
| CLIP ViT-B/32 | open_clip `laion2b_s34b_b79k` | **projected output (512, the `encode_image` space)** — the canonical practical use of CLIP as an extractor; the 768-d pre-projection width is NOT used | 512 |
| CvT-13 | HF `microsoft/cvt-13` (224px) | **CLS token** (the `[B, 384, 14, 14]` spatial map is used only as ACF components, never flattened into the pooled feature) | 384 |

Native dims are **read from the model** by a probe forward
(`BaseExtractor._probe_native_dim`), never hardcoded, and validated
against `configs/extractors.yaml` (`raw_dim`) — a mismatch fails the
extraction loudly. Every artifact ships a `.meta.json` sidecar
(backbone, native dim, extraction point, exact weights id, transform
recipe); the loader cross-checks features against it.

### 1b. Optional fixed projection to a common dimension (opt-in, off by default)

`projection:` in `configs/extractors.yaml` writes an *additional*
artifact per extractor, `<extractor>_<method><dim>.npy`, carrying a fixed
linear map of the native feature to one shared width. It exists so the
element-wise fusion family can consume equal-dim sources without an
alignment learned online, and so a reviewer who asks for "every
backbone at 128-d" gets exactly that. The native artifact is untouched
and both are trainable side by side.

**This re-enables, as an explicit variable, the very thing §1 rejected
as a default.** `method: random` is the seeded random projection the v1
protocol was criticised for: a comparison run *only* on
`<extractor>_pcaw128` artifacts compares "backbone × fixed compression",
not backbones, and the narrower `dim` is, the more of the difference
between backbones the compression can absorb. That is a legitimate
experiment — it is not the headline benchmark. The defensible uses are:

- **As a controlled ablation**, reported next to the native-dim result
  from the same run, so the cost of the compression is visible rather
  than assumed away.
- **As a fusion input**, where the alternative (`alignment: learned`)
  is itself a projection — a fixed one merely moves it earlier and
  removes it from the recommender's gradient.

`method: pca` (fit on train items only, mirroring `alignment.method:
pca`) preserves more variance than `random` and is the better default
of the two when the projection feeds a comparison; it is
dataset-dependent, so its basis does not transfer across datasets.
`method: pca_whitened` additionally equalises the per-component
train-set variance (Jégou & Chum, ECCV 2012), so no single principal
direction dominates the inner products the recommenders compute — same
fit set and leakage guarantees as `pca`.

What is *written* (`projection:` in extractors.yaml) and what is
*consumed* are separate decisions: `embedding_variants` in
recommenders.yaml (`native` / `projected` / `both`) selects the training
cells, and `extractor_variants` in fusion.yaml selects the fusion
sources. `both` is what produces the paired report this section asks
for. With `extractor_variants: projected` the `alignment:` block is
bypassed — the sources already share a width — which removes the
learned alignment from the recommender's gradient and makes the
projection the only compression in the pipeline.

Provenance is on disk: the projector is saved as
`<extractor>_<method><dim>.proj.npz` with a `.proj.json` describing method,
dim, seed and fit set, and the artifact's own `.meta.json` records
`source_native_dim` alongside the projected `native_dim`.

## 2. Canonical per-backbone preprocessing

The preprocessing recipe is part of the model. Three **distinct
normalisations** coexist across the 8 backbones:

- ImageNet (`0.485/0.456/0.406`): ResNet-50, ConvNeXt, LeViT, CvT, DINOv2
- Inception-style (`0.5/0.5/0.5`): **ViT-B/16 (augreg2)**, **CoAtNet-0 (sw_in1k)**
- CLIP (`0.48145466/…`): CLIP ViT-B/32

That earlier version applied ImageNet normalisation + direct bilinear
224 resize to all timm backbones — ViT-B/16 and CoAtNet-0 ran silently
degraded, and no backbone used its canonical bicubic resize+crop. This
protocol resolves each transform from the library that ships the weights (torchvision
`weights.transforms()`, `timm.data.resolve_model_data_config`,
`AutoImageProcessor`, open_clip's `preprocess`; DINOv2 is the one
hand-built recipe, matching its reference eval transform: resize 256
bicubic → crop 224, ImageNet norm). Pinned by
`tests/test_canonical_transforms.py`.

**Resolution posture (declared): all backbones consume 224×224 crops**,
their canonical eval resolution — the resize path (resize size,
interpolation, crop_pct) differs per recipe and is recorded in each
artifact's metadata. No hidden resolution variable.

### 2b. Interaction filtering is inherited, not applied

No k-core filtering runs inside this framework. The DVBPR datasets
arrive **pre-partitioned upstream** (their published splits already
carry the original papers' filtering); re-filtering here would silently
change the benchmark population. Plugin CSV datasets have their own
knob (`CSVDatasetProvider(min_user_interactions=...)`, default 3,
applied before the leave-one-out split). A `preprocessing.n_min` key
used to exist in `configs/default.yaml` and was read by nothing — it
has been removed so the config cannot claim a filtering step that
never happens.

## 3. Evaluation protocol: full ranking default, sampled opt-in

`full_ranking` is the default and the only protocol for reported
numbers (Krichene & Rendle, KDD 2020: sampled metrics can invert model
rankings). `sampled` exists for fast iteration only and is locked when
used: `n_negatives`, `negative_sampling_seed` (per-user seeded pools →
identical across models, required by the paired tests), sampling from
items unseen by the user. **Every recorded result row carries a
`protocol` column**; train-time BPR negative sampling is a different
thing entirely and is not configurable here.

**Model selection on validation.** Early stopping and the search
objective (`ndcg@10`; exhaustive grid over `learning_rate × total_dim`
since validation Phase D, `results/validation/phase_d_search_cost.md`)
score the **validation** held-outs, never the test set: the training path loads `val.csv` and masks each user's train
items (`src/steps/train.py`, `src/utils/parallel.py`;
`src/utils/training.py` builds the selection `Evaluator`). The test set
is read only by the final evaluate step (`src/steps/evaluate.py`), so
hyperparameters and the stopping epoch are never chosen by looking at
test performance — the reported test numbers are an out-of-sample
estimate, not an optimistically-biased one. During validation the
user's own test item stays in the candidate set and competes as an
ordinary item; this is neutral across models and leaks nothing to the
model (the model never sees which items are held out).

**Selection rule (3.0.0, task records I01/R03).** The selection
cut-off is the single declaration `SELECTION_K_VALUES = (10,)`
(`src/recommenders/hp_budget.py`); a selection metric the training
evaluator does not produce (`UnsupportedSelectionMetricError`) fails
before any model or data is built. The **first finite validation
observation is the winner even when it is exactly 0.0**; afterwards
only strict improvement replaces it, ties keep the earlier winner and
advance patience. A missing, non-scalar or non-finite selection metric
raises `SelectionMetricError` and leaves no winner and no success
marker — a legitimate zero result and an invalid one are never
confused, and a zero cell is evaluated and reported like any other
(before 3.0.0 an all-zero run wrote no `_best.pt` and vanished from
the battery). The per-dataset budget (`hp_budget.<dataset>`: epochs,
patience, metric, validation sample, trial count) is consumed
identically by every training path — CLI cell, grid worker, Optuna
trial, battery replay and folds — while `eval_every_epochs` and
`batch_size` stay shared.

**Training-time validation subsample (`common.eval_sample_size = 2000`).**
Selection scores a fixed subset of 2000 **validation** users instead of
all of them. The subset is drawn once per dataset, deterministically
(dedicated `np.random.default_rng`, `sample_seed` = global run seed —
not the per-trial job seed; `src/evaluation/protocol.py`), and is
identical for every model/embedding/trial, so selection remains a
paired comparison on a common validation-user set; only its variance
changes (standard error on ndcg@10 stays well below between-config
gaps). The validation metric is still full-ranking over all items for
those users. The final evaluate step constructs its `Evaluator` without
`sample_size` and ranks the entire test set.

**Item-side transduction in fine-tuning (declared).** The CNN/ViT
fine-tuning trains a category classifier on the **full catalogue**
(`src/steps/finetune.py` builds `CategoryDataset` splits from every
item's image and category label; `src/finetuning/dataset.py`), so the
images and category labels of items that later appear as test held-outs
are seen by the backbone. No interaction data is used at this stage and
no test **interactions** leak — the backbone never observes which user
held out which item. This is a deliberate protocol decision, standard
in visual recommendation (item content is catalogue metadata, available
before any interaction), and it slightly favours the finetuned
condition relative to a strictly inductive protocol in which held-out
items' images would be excluded from fine-tuning.

**Held-out items as BPR training negatives.** During BPR training a
user's own val/test positives are eligible to be drawn as negatives
(`src/utils/training.py` excludes only the user's train items) — the
standard protocol, with negligible metric deflation given catalogue
sizes.

## 3b. User-level K-fold cross-validation (the `folds` pipeline step)

Precedent: Rendle et al. (UAI 2009, §6.2) evaluate BPR with
leave-one-out, repeat the experiment 10 times over freshly drawn splits,
and run the hyperparameter grid search **once, on the first round**,
keeping the winners constant afterwards. Section 2 of the same paper
notes that the fold-in strategy known for MF applies to BPR. The
`folds:` block of `configs/default.yaml` reproduces that procedure with
users as the partition unit:

- Users are split into `k` mutually exclusive, balanced folds
  (`src/folds/partition.py`, seeded). A user is eligible when it has
  exactly one `test.csv` item (the target) and a profile
  `train ∪ val` of at least `min_profile` items; ineligible users are
  counted in the manifest (`no_target`, `profile_too_small`) and stay in
  the training pool.
- In fold *i* the fold's users leave the training set. The model trains
  on the other folds (history `train` only, early stopping on `val`; no
  user's `test.csv` item ever enters training, as in the sequential
  protocol)
  with the cell's **frozen** hyperparameters — the prior search's winner
  from `results/best_hyperparams.json`, or the fixed values of the config
  under `hp_search.strategy: fixed` (`src/recommenders/hp_source.py`
  records which). Fold *i* runs under seed `folds.seed + i`.
- The held-out users are **folded in** (`src/folds/foldin.py`): their
  rows in every user table are re-initialised, every other parameter is
  frozen, and only those rows are optimised with the same BPR loss and
  negative sampler over the profile alone. History-consuming models
  rebuild the user's non-parametric state from the profile.
- Each held-out user is then ranked on its single target with the
  profile masked, which keeps `_require_leave_one_out` satisfied and the
  per-user artifact identical in format to a normal cell.
- The `k` partial artifacts are **concatenated** (`src/folds/aggregate.py`)
  into the cell's canonical artifact with `n` = every evaluated user, so
  Wilcoxon, Holm and Cliff's δ stay paired by user without change.
  Between-fold mean and standard deviation are recorded in the manifest
  as descriptive variability only.
- Folds and seeds are distinct variance sources: the manifest states that
  the reported between-fold variability is combined (partition +
  optimisation). The multi-seed robustness experiment stays separate.
- **Fold plan digest (3.0.0, task record E07).** `fold_plan_digest`
  (`src/folds/runner.py`) hashes `k`, the partition seed, `min_profile`,
  the exact user→fold assignment and the train/val/test split and
  item-mapping digests; it is the concatenated artifact's
  `config_hash` (previously `None`) and flows into the partial
  artifacts. A fold cell is complete only when its artifact validates
  (payload digest, row count) and carries this digest with `fold.k`
  and `fold.seeds == [seed + i]`; changing `folds.k`, the partition
  seed or the split re-runs every cell. Each fold's training carries
  `fold={index, k, partition_seed, min_profile}` in its scientific
  identity, so its resume envelope and winner are bound to the fold.

## 3c. Scientific identity (v2) versus execution metadata

Every reuse decision of 3.0.0 — skipping a completed grid point,
comparing two trial winners, accepting an evaluate completion, resuming
a training envelope — is bound to a canonical **scientific identity**
(`src/utils/identity.py`, schema version 2; task records E03/E04):

| Field | Content |
|---|---|
| `dataset_digest`, `item_mapping_digest`, `split_digest` | SHA-256 of the interaction files, of `item2idx.json` by numeric index (insertion order irrelevant) and of the train/val(/test) splits in use |
| `feature_digest` | content digest of the visual artifact plus, for a sidecar, the recipe (strategy, alignment, dim, normalisation, `recipe_version`) and the recipe of every component, in order |
| `model_name`, `implementation_digest` | registered name and a digest of the recommender's source |
| `effective_hyperparams` | the canonical expanded configuration (`effective_hyperparams`, §10.6): dimensions split from `total_dim`, single-valued defaults filled |
| `selection_budget` | epochs, batch size, patience, cadence, metric, validation sample size and sample seed |
| `seed`, `protocol`, `condition`, `fold` | training seed; full-ranking, K, mask policy and tie-break seed; frozen/finetuned; fold identity when applicable |

The identity is the SHA-256 of the canonical JSON of that payload
(sorted keys, compact separators, finite numbers only, strict
bool/int/float distinction). Selection identity uses train+val splits;
evaluation identity adds the test split and the streamed digest of the
exact `_best.pt` evaluated. **Execution metadata is deliberately
outside the identity**: block and batch sizes used for ranking, worker
count, device, filesystem roots, the lazy/dense feature residency, the
number of gradient-accumulation micro-batches of an OOM retry and
wall-clock time change how a result is computed, not what it is (the
lazy path is verified numerically equivalent; the accumulated step is
verified to take the same optimiser step as the full batch for every
recommender; ranking layouts are verified against a golden fixture).
Because they are outside the identity, every OOM retry that changes
them is listed in `results/runs/<run_id>/oom_recoveries.csv` (§10.8). Identical content under another
root therefore has the same identity; another seed, split, feature
content, budget or protocol does not, and legacy artifacts without an
identity are identified as such and never reused as if they matched.

## 4. Deterministic tie-breaking

All three ranking paths (batched torch, sampled numpy, single-user
numpy) implement one rule: exact-score ties are broken by a **fixed
random permutation of the item ids**, drawn once per run from a
dedicated `np.random.default_rng(seed)` (global run seed, not the
per-trial job seed) and shared by every model/trial of a `(dataset,
seed)` run. Item id is NOT used as the tie-break: `item_idx` correlates
with popularity in the DVBPR splits (Spearman −0.34 to −0.45), so an id
tie-break would systematically favour popular items inside a tie block —
penalising models with mass exact-ties (pure BPR over cold items) more
than models with distinct visual scores. In the sampled path ties are
likewise NOT broken by pool position (positives come first — that would
inflate metrics). When the held-out item is not tied, the returned rank
is identical to a plain descending sort. Each evaluation logs the
fraction of held-outs in an exact-score tie and the mean/max tie-block
size, so the real exact-tie frequency is measured during the battery.

## 5. Statistics

- **Wilcoxon signed-rank, `zero_method="pratt"`**: per-user LOO metrics
  are 0/1-heavy; the scipy default drops all zero differences,
  shrinking the effective sample far below `n_users`. Pratt keeps them.
  Every pairwise table reports `n_pairs` and `n_nonzero_pairs`.
- **Comparison families** (`src/evaluation/comparison_families.py`):
  the Holm correction and the Friedman omnibus are applied WITHIN the
  family of comparisons one research question defines — never over the
  Cartesian product of every config (all-pairs Holm over ~77 configs
  runs with `m ≈ 2900` and rejects everything artificially). Each
  family varies exactly one dimension: `backbone_within_model`
  (`m = C(n_backbones, 2)` per recommender), `model_within_backbone`,
  `fusion_within_model`, `frozen_vs_finetuned` (ONE instance per
  dataset containing every frozen/finetuned pair, `m = n_pairs` — one
  research question, one correction unit), and `vs_baseline` (ONE
  instance per dataset pairing every config with the pure-BPR
  baseline `bpr_none`, `m = n_configs − 1` since the baseline pairs
  with everyone but itself — the central visual-signal hypothesis).
  Every result row carries `family`, `group` and
  `n_comparisons_in_family` so the correction is auditable; `all_pairs`
  exists as an exploratory option only.
- **Friedman gate (annotate, don't suppress)**: the family omnibus is
  computed alongside the pairwise tests and its verdict is joined onto
  every pairwise row as `omnibus_significant` (NaN where Friedman is
  undefined, i.e. fewer than 3 configs, or disabled). Pairwise rows
  are never suppressed, but a pairwise effect must not be claimed at
  the family level when its omnibus is not significant. The gate only
  applies to the one-dimension families (`backbone_within_model`,
  `model_within_backbone`, `fusion_within_model`), whose configs form a
  homogeneous K-way design. `vs_baseline` and `frozen_vs_finetuned`
  are bundles of two-treatment questions (a star against one baseline;
  per-config frozen/finetuned pairs): the only K-way omnibus available
  there — "all dataset configs are equivalent" — is trivially rejected
  and gates nothing, so those families report
  `omnibus_significant = NaN` by construction and their evidence is
  the Holm-corrected pairwise tests alone.
- **Primary metrics under LOO**: with one relevant item per user only
  two independent signals exist — hit-or-not (recall@k ≡ HitRate@k) and
  hit rank (ndcg@k). precision@k = recall@k / k and map@k = 1/rank are
  deterministic transforms; they stay in the raw evaluation CSVs, are
  excluded from the reported tests by default
  (`statistical.include_derived_metrics`), and must never be read as
  independent evidence.
- Friedman as the non-parametric omnibus (no normality assumption over
  per-user metric distributions), Holm–Bonferroni for multiple
  comparisons (uniformly more powerful than Bonferroni at the same
  FWER), percentile bootstrap CIs.
- **Effect size: PAIRED Cliff's delta is primary** — ``(wins −
  losses) / n`` over per-user differences, the same pairing the
  Wilcoxon uses (ties count in the denominator, consistent with
  ``pratt``), reported **together with the win / loss / tie triplet**
  (`n_wins`, `n_losses`, `n_ties`, `pct_wins`, `pct_losses`,
  `pct_ties`). **No magnitude label is attached**: the Romano et al.
  (2006) cut-offs 0.147 / 0.33 / 0.474 were calibrated for the
  between-groups delta against Cohen's d; the paired delta is a
  different quantity, and under leave-one-out the ties that dominate
  the denominator (both models miss the single held-out for most
  users) bound it to tiny values, so every comparison would read
  "negligible" — including consistent ones. The net delta is also
  ambiguous on its own (1% wins / 0% losses / 99% ties and 50.5% /
  49.5% / 0% both give δ = 0.01), which is why the triplet is the
  reported interpretation: "A beats B in X% of users, loses in Y%,
  ties in Z%, δ = X − Y", read alongside `diff_mean` and its CI. The
  between-groups form collapses to ``p_a − p_b`` on the 0/1-heavy LOO
  metrics and is not reported. Cohen's d is parametric and inflates on
  such vectors (the std shrinks); it is off by default and available
  for diagnostics only.
- **Paired-difference bootstrap CI** on every pairwise row
  (`diff_mean`, `diff_ci_lower/upper`, resampling USERS): a RAW 95% CI
  whose agreement is with the raw Wilcoxon p-value at alpha, NOT with
  the Holm-corrected `significant` verdict — a CI excluding zero under
  a Holm-non-significant verdict is the correction working, not a
  contradiction. Per-config CIs are descriptive — under paired
  inference, overlapping individual CIs do NOT imply absence of a
  significant difference.
- **Cross-seed scope of the p-values**: every p-value in a per-seed
  `statistical_tests.csv` is a statement about ONE training
  realisation (the models trained under that seed), not about the
  methods in general. Multi-seed runs write
  `statistical_tests_across_seeds.csv` — per pair: `n_seeds`,
  `n_seeds_significant` (Holm-corrected verdict per seed), the median
  paired difference and its sign agreement across seeds, and
  min/median/max of the Holm-corrected p-value. This reconciliation is
  deliberately descriptive (no Fisher-style p-value combination);
  method-level claims require the verdict AND the sign of the effect
  to agree across seeds. `n_seeds` is the number of DISTINCT seeds
  (`nunique`), not of rows: a duplicated source file is dropped with a
  warning and two conflicting rows for one seed raise
  `ObservationConflictError` (3.0.0, task record R05).
- **Paired observation validity (3.0.0, task records R04/R05).**
  Before any test, every per-user frame is validated
  (`src/evaluation/paired_validation.py`). The observation key is the
  provenance columns (`dataset, seed, split, protocol,
  eval_protocol_version, fold_policy, split_digest, generation_id` —
  each single-valued in a frame), the config identity columns
  (`visual_input_dim, n_trainable_params, d, checkpoint_digest` —
  constant within a config) and `(config, user_id)`. Two rows with
  the same key and different values are a conflict (error); an
  identical duplicate of a visual cell is a torn append (error); the
  one accepted duplicate is the shared `bpr/none` baseline appearing in
  both condition files. Non-finite metric values fail instead of being
  dropped. **Population policy** (`statistical.population`): `strict`
  (default) requires every cell of a comparison to cover one shared
  user population and fails naming the users absent from each side;
  `declared_intersection` restricts to the intersection explicitly,
  writes to a separate `_restricted` partition and reports
  `n_excluded_a` / `n_excluded_b` (Friedman: `n_users_excluded`) on
  every row. Nothing intersects silently. Outputs are partitioned by
  condition and policy —
  `results/tables/{dataset}_{condition}[_restricted]_{summary|friedman|pairwise}_{metric}.csv`
  — and a `{dataset}_{condition}[_restricted]_integrity.json` is
  written BEFORE the tests with the seed and distinct-seed count, the
  shared provenance, the expected / completed / missing cells
  (reconciled against the evaluate step's completion record) and the
  excluded cells with their reason; a missing expected cell always
  fails, so a table over only the successful models is never published
  as a complete battery.

## 6. Fusion pipeline (Pipeline B — separate from the 8-extractor Pipeline A)

Sources: ResNet-50 (2048) + ViT-B/16 (768), native.

- **Element-wise family (8 of 11 strategies)** requires alignment, and
  the alignment method is an experimental variable
  (`alignment.method`): `learned` (default) — per-source
  `Linear(D_i→D)` co-trained via BPR (`LearnedAlignmentFusion`), the
  analogue of `E`; or `pca` — per-source PCA to `D`.
- **Concat family** operates on native dims: `concat` → 2816-d;
  `pca` (joint) reduces the 2816-d concat; **`pca_per_model`
  CONCATENATES after per-source PCA (→ `M·k`)** — declared, it is a
  concatenation-family strategy.
- **PCA protocol**: every PCA (`pca`, `pca_per_model`, `pca` alignment)
  is **fit exclusively on items with ≥1 training interaction** and
  applied to all items; seed fixed; cumulative explained variance
  logged per fit. The `k` of the PCA is itself a confounder vs the
  2816-d concat — report explained variance and/or sweep `k`.
- The fused `h_i` enters the recommender as the item's visual feature,
  through the same `E` as any single extractor.

## 7. Model-specific declarations

- **ACF is NOT degenerate**: it consumes `(n_items, M, D_native)`
  component artifacts (`*_comp.npy`), so component-level attention has
  real components to attend. Its user-history side is built from train
  interactions only.
- **ACF components are pooled to a 2×2 grid (`component_grid: 2`,
  `configs/extractors.yaml`; declared divergence).** The paper attends
  over the 7×7 = 49 regions of a ResNet-152 map. Saving the native
  region maps (49 for ResNet-50 / ConvNeXt / CoAtNet / CLIP, 196 for
  ViT-B/16, 256 for DINOv2) costs 50–500× the pooled features — ≈ 1.1 TB
  for the four datasets and eight backbones, and one artifact alone
  (ResNet-50 on Amazon Men, 22 GB fp16 → 44 GB fp32) exceeded the
  container's RAM cap during validation Phase C (2026-09-04). The extract
  step therefore adaptive-average-pools each `√M × √M` map to `2 × 2`
  before saving (`src/extractors/components.py`): four quadrant
  descriptors per item, M = 4 for every backbone, ≈ 20 GB in total. The
  mechanism — one attention weight per spatial region, Eq. 10–12 — is
  unchanged; the spatial granularity is coarser, which must be stated
  next to any ACF result. The artifacts stay fp16 on disk and in the
  model buffer (`BaseRecommender` keeps the on-disk dtype for raw
  components; ACF casts the gathered rows and builds its eval cache in
  chunks), so the catalogue costs half the VRAM of an fp32 copy. The
  grid is configurable (`3` = nine regions) and recorded in the
  artifact's `.meta.json` (`component_grid`, `pooling`).
- **ACF component vectors are L2-normalised on the way into the model
  (declared divergence from the raw artifact; 2026-09-16).** Each of the
  `M` component vectors of a native `*_comp.npy` is scaled to unit norm
  as it reaches the recommender — once at buffer construction on the
  dense path, per gathered row on the lazy one, so residency stays an
  execution detail (`l2_normalize_components`, the same rule
  `l2_normalize` applies to a pooled row). Without it ACF was the only
  model reading unnormalised features: the pooled embeddings the other
  recommenders consume are L2-normalised offline by the fusion step,
  while a single-extractor component artifact carries no sidecar and so
  was never normalised. ACF's attention logits therefore scaled with the
  backbone's native magnitude — mean component norm 11.9 for CLIP
  ViT-B/32 but 154.5 for ConvNeXt-B and 659.9 for CvT-13 on Amazon Men —
  and overflowed to `inf` under autocast, then to `NaN` through the
  softmax: five cells of the frozen grid died with `NonFiniteScoresError`
  at `learning_rate = 0.01` (CoAtNet-0 and ConvNeXt-B, 2026-09-16), while
  no CLIP cell ever failed. The normalisation makes the comparison
  between backbones a comparison of *direction*, which is what the
  attention is meant to weigh, instead of one confounded by each
  backbone's activation scale. The artifacts on disk are unchanged; ACF
  results produced before this date are not comparable with later ones
  and were discarded.
- **Dimension parity**: every recommender draws its dimensions from one
  budget `common.total_dim` (`RecommenderSpec.dim_split`), and that
  budget is the **collaborative** capacity: `latent_dim = T` for BPR-MF,
  VBPR/AVBPR, VNPR, DeepStyle (`d = T`) and ACF (`k = T`). A visual
  model's own dimensions sit **beside** the budget rather than inside
  it: VBPR/AVBPR get `visual_dim = T` alongside their `T` latent
  factors, and VNPR's visual user vector is `D_backbone`-wide by
  construction. A comparison at a fixed `T` is therefore a comparison of
  the visual mechanism, not of how many collaborative factors each model
  was left with. `assert_dimension_parity` refuses a direct `latent_dim`
  / `visual_dim` anywhere.
  **Changed 2026-09-09 (researcher's decision).** Until then VBPR/AVBPR
  used `dim_split: half` — `latent_dim = visual_dim = T/2` — so at
  `T = 128` VBPR competed against BPR-MF holding 64 collaborative
  factors against BPR's 128. That accounting, not the visual term, was
  the leading explanation for VBPR trailing BPR on amazon_fashion
  (hypothesis H1 of the 2026-09-07 fidelity audit: the logs already
  showed BPR-64 0.0033 < VBPR-64+64 0.0069 < BPR-128 0.0086, i.e. VBPR
  beating BPR at equal collaborative capacity and losing at equal
  total). Consequences, stated rather than discovered: a visual model
  now holds more parameters than BPR-MF at the same `T`, by exactly its
  visual side — that asymmetry is the deliberate choice, since charging
  the visual dimensions against the collaborative budget is what made
  the earlier comparison unfair; and **every VBPR/AVBPR result produced
  before this change is not comparable to results produced after it**,
  because the same `T` now resolves to different dimensions. `"half"`
  remains a supported value of `dim_split`; no model registers it.
  Guarded by
  `tests/test_dimension_parity.py::TestRegisteredModelsShareTheCollaborativeBudget`.
- **Per-paper formulations and regularisation** (2.10.0): each built-in
  reproduces its paper's score and L2 scheme rather than a shared
  convention. BPR-MF: `γ_u^T γ_i`, no item bias, `λ_W / λ_H+ / λ_H−`
  (`l2_reg`, `l2_reg_item_pos`, `l2_reg_item_neg`). VBPR: Eq. 4 with the
  visual bias `β'^T f_i`, `λ_E = 0` by default (`l2_reg_projection`),
  `λ_β` (`l2_reg_visual_bias`); AVBPR mirrors it so attention is the
  only difference. DeepStyle: one user vector `p_u`, one `d`, no item
  bias, single `λ` (Eq. 6). ACF: Eq. 6 score (no visual term, no item
  bias), attention nets and projections unpenalised (Eq. 5). VNPR:
  Hadamard merge + single-neuron ReLU dense, mirrored item tables,
  ½-averaged branches at inference; its L2 is the BPR-Opt gathered-row
  reading, NOT the paper's whole matrices (declared divergence, see the
  next item). The BPR-Opt gathered-row reading of the L2 term is the
  framework default for every built-in; the constants are per group.
- **VNPR regularisation: whole-matrix L2 under Adam collapses the
  model (found 2026-09-04, validation profile).** Reproducing the
  paper's objective literally — `λ(‖W_u‖² + ‖W_i‖² + ‖W_i'‖² + ‖W_v‖²)`
  added to every step — trained the three collaborative tables to
  row norms of EXACTLY 0.0 in every VNPR cell (Amazon Men, lr 1e-3,
  λ 1e-4, batch 4096), while VBPR/DeepStyle/BPR under BPR-Opt kept
  norms 0.4–1.9. Mechanism: Adam normalises gradient magnitude, so a
  row that receives only the L2 gradient moves ≈ lr towards zero at
  every step regardless of λ; with 46 steps per epoch a Xavier row
  (~0.05) is dead within the first epoch, and only `W_v`, which gets a
  dense gradient from the image feature every step, survives. The
  model degenerates to `ReLU(v_h·f_i + b)`, a visual-only scorer. On
  the learned-fusion input (`hybrid_mean_learned_D128`, ‖f‖ ≈ 1 vs
  10.7 for ResNet-50 and 31.9 for ViT) even that term is flat (per-user
  score spread over items 0.024 vs 0.38/0.58): val metric 0.0002,
  recall@10 0.0007, reproduced across three trainings. Workaround
  adopted: VNPR regularises the rows gathered by the batch like every
  other recommender (`W_i` by the positives, `W_i'` by the negatives,
  `W_u`/`W_v` by the users; dense layer unpenalised as in the paper).
  Comparability across recommenders is preferred over this fidelity
  detail; the paper does not state an optimiser that would make the
  literal objective usable. Guarded by
  `tests/recommenders/test_vnpr_paper.py::test_adam_training_keeps_rows_outside_the_batch_at_their_initial_norm`.
  Every VNPR checkpoint trained before this change is visual-only and
  must be discarded.
  **Status: RESOLVED 2026-09-09 — dead ReLU at initialisation.** The
  BPR-Opt reading removed the whole-matrix mechanism described above,
  but the 2026-09 battery still finished cells at `best_metric =
  0.0000` with every catalogue item tied. Over the complete 3.0.0-rc.1
  grid (320 VNPR jobs, four datasets) that is 79 cells: 76 of 192
  fused/hybrid (39.6 %) and 3 of 128 native (2.3 %, all amazon_women
  with `lr = 0.01`); VBPR, DeepStyle and BPR never collapse. The cause
  is the interaction between the initialisation and the single ReLU
  neuron of Eq. 3, and it is attributable and reproducible per cell:
  * the branch pre-activation at initialisation is a sum of products of
    two Xavier-uniform rows, whose bound `sqrt(6/(n+k))` shrinks with
    the vocabulary; on the real catalogues its spread is ~3e-4 to 1e-3
    on fused (unit-norm) features and ~5e-3 on native ones, while one
    Adam step moves the unpenalised bias `b` by ≈ `learning_rate`;
  * a single adverse step therefore drives EVERY item's pre-activation
    below zero, `ReLU` returns a constant 0, and the data gradient
    vanishes for every parameter — the ranking is exactly flat and the
    loss sits at `ln 2` plus the L2 term. The model cannot recover: the
    surviving L2 gradient then decays the online-fusion projections to
    exactly 0.0 (the same Adam-plus-L2 mechanism as above);
  * the outcome is decided by the sign of the first applied bias step,
    so it is deterministic per cell and invisible to a seed sweep. The
    aggravation by `lr = 0.01` (50 of the 79) and by the fused features
    is exactly the ratio `learning_rate / pre-activation spread`.
  Fix adopted: `VNPR.DENSE_BIAS_INIT = 1.0` — the dense bias starts
  positive instead of at the customary zero, so every unit fires before
  the first step and one adverse step cannot silence the neuron. The
  value is where healthy cells converge on their own (`+0.4` to `+2.2`).
  Initialisation is declared framework-side, not a property of the
  paper, so this changes no architecture, score function or
  regularisation term, and touches no other recommender (VNPR is the
  only built-in with a non-linearity over the merged vector). Guarded
  by `tests/recommenders/test_vnpr_paper.py::test_dense_bias_starts_positive_so_every_branch_is_active_at_initialisation`
  and `::test_a_bias_step_larger_than_the_preactivation_range_is_unrecoverable`.
  Verified end-to-end before adoption on tradesy
  `hybrid_sigmoid_gated_l1_0_learned_D128` (`lr` 1e-3, `T` 64, 30
  epochs, selection Evaluator): `0.0000` with the zero bias — including
  the same final `b = -0.006` as the battery checkpoint — against
  `0.0026` with the positive one, in the band of that dataset's healthy
  hybrid cells. **Every VNPR result produced before this change is
  invalid and was deleted**; the earlier "‖f‖ ≈ 1 flattens the visual
  term" reading was a hypothesis and is superseded by the account above.
  The opt-in training diagnostics (`diagnostics:` in
  `configs/default.yaml`) and `scripts/vnpr_collapse_diagnostic.py`
  remain available; note that the driver's synthetic default (300
  users) has a Xavier bound an order of magnitude larger than a real
  catalogue and therefore cannot reproduce this failure — use
  `--processed-dir` with real features.
- **One sampling cap for every PCA fit (2026-09-10).**
  `PCA_FIT_MAX_ROWS = 750_000` (`src/fusions/strategies.py`) bounds the
  fit set of EVERY PCA in the framework — fusion strategies, the
  per-region component pass, the fusion alignment and the extractors'
  fixed projection — because an estimator that saw 1.17 M rows and one
  that saw 300 k are not the same estimator, and a battery that mixes
  them is not a fair comparison. Larger fit sets are sampled without
  replacement, indices sorted, seeded by the caller's `random_state`
  (42 when unset), and the sampling is logged with both counts.

  The value is a byte budget in rows: 8 GB of host RAM over the widest
  fit matrix in the battery, the per-region `concat` of the two fusion
  extractors at 2048 + 768 = 2816 float32 columns (750,000 × 2816 × 4 B
  = 7.87 GiB). The fit is scikit-learn on CPU and never touches the GPU.
  Measured process peak on amazon_women: 9.97 GiB, the extra ≈2 GiB
  being the randomized-SVD workspace, the gather chunk and the output
  memmap pages.

  It was forced by the per-region pass, which multiplies the fit set by
  the region count: amazon_women is 291,812 train items × 4 regions =
  1,167,248 rows, a 13.1 GiB matrix against a 16 GiB container.
  Statistically the cap is slack rather than a compromise — 128
  components over 2,816 dimensions are estimated from a sample two
  orders of magnitude larger than the dimension either way — and **every
  pooled fit in the battery is already below it** (one row per train
  item), so no pooled fusion, alignment or projection artifact changes.

  The cap is applied to the INDICES, before the fit matrix is assembled.
  Capping the assembled rows is worse than useless: the matrix is
  already allocated and the sample adds a copy on top (a 21 GiB peak,
  measured, against 13.1 uncapped). `fit_pca_on_rows` therefore WARNS on
  an oversized fit set instead of sampling it, so a caller that forgets
  is loud rather than silently unbounded.

  **Known limitation:** the artifact provenance does NOT cover the cap.
  `fit_set_digest` records `train_items`, which does not change when
  `PCA_FIT_MAX_ROWS` does. Recording it would change the digest of every
  PCA artifact already on disk and refuse to reuse ones that are in fact
  identical (the same argument the `component` flag is excluded for). If
  the cap is ever changed, **delete the PCA artifacts by hand** — the
  pipeline will not detect that they are stale.

- **VNPR visual input: the paper's offline reduction (2026-09-10).**
  The dead-ReLU fix above removed the *global* collapse; a per-user tail
  survived it on unfused backbones. On amazon_women, 4-6% of users got
  an all-tied score list — for them the single ReLU is negative over the
  whole catalogue, so every item ranks by the tie-break — while the
  fused cells were clean (0.00%). Fraction of users with a tie block
  above 10, single split, 97,678 users: `levit_256` 6.10%, `cvt_13`
  5.13%, `dinov2_vitb14` 4.47%, `vit_b16` 0.92%, `resnet50` 0.65%,
  `convnext_base` 0.42%, `coatnet_0` 0.02%, `clip_vitb32` and every
  hybrid 0.00%. **Report this as the FRACTION of affected users**: the
  mean tie block is driven by the tail and overstates it by orders of
  magnitude.

  VNPR is the only recommender exposed. VBPR (`E`) and DeepStyle
  (`visual_projection`) learn a projection that absorbs the input scale
  and score 1.00-1.01 mean tie blocks on the SAME embeddings; VNPR has
  none by construction of its paper. Niu et al. (WSDM 2018) §5 and
  footnote 2 are explicit that the reduction happens *before* the model:
  fc6 is 4,096-d, "dimension reduction is further applied ... separately
  from model training", by PCA (compared against a stacked
  auto-encoder), and the learned kernel of visual BPR is rejected as
  "less efficient". The framework implemented the architecture and
  omitted that data step, feeding the raw native feature.

  Adopted: `vnpr.visual_input` (`configs/recommenders.yaml`) routes VNPR
  to `<source>_pcaw128` artifacts — the extract step writes them per
  backbone, the fuse step per offline fusion, after fusion, over the
  vector the model receives. **`pca_whitened` instead of the paper's
  plain PCA is a declared divergence**, measured rather than assumed:
  on `cvt_13`, plain PCA takes the tail from 5.13% only to 0.89% and
  *raises* the per-component magnitude (‖f‖/√dim 3.709 → 3.812), while
  dropping the width to 64 leaves the tail flat (0.96%) and costs 38% of
  recall@10 — the dimension is not the lever, the per-component variance
  is. Whitening takes the tail to 0.01% on all three worst backbones and
  lifts recall@10 by 10-61% (`cvt_13` 0.00648 → 0.01046, `levit_256`
  0.00738 → 0.01107, `dinov2_vitb14` 0.01266 → 0.01396). The width is
  the paper's 128.

  Scope and consequences: the routing is per recommender, so VBPR,
  DeepStyle and ACF keep reading the native feature their own papers
  prescribe — the projection axis was deliberately NOT extended to them.
  A projected artifact is a routing variant of its backbone, not a
  backbone of its own, so `_backbone_base` strips the token and
  `model_within_backbone` still asks "which model wins on ResNet-50"
  with each model on its own input. **Every VNPR result on a native
  artifact produced before this change is invalid**; the hybrid cells
  are unaffected.

  **The rule is SUPERSEDE, not require (corrected 2026-09-11).** An
  input is dropped only when a projected counterpart of it exists.
  Online fusions (`alignment: learned`) are JSON sidecars whose mixing
  happens inside the recommender at train time, so no array of theirs
  can be projected — and they never needed it: their degenerate-user
  tail is 0.00%, because the alignment already delivers a normalised
  vector at the paper's width. Requiring the projection outright left
  VNPR with 3 fusion strategies where every other model has 12, gutting
  the `fusion_within_model` family for it and removing VNPR entirely
  from `model_within_backbone` on the other nine. VNPR therefore runs
  on 8 projected backbones, 3 projected offline fusions and 9 online
  sidecars — 20 cells, the same count as every other visual model.

  The two routes answer different questions and both are in the grid:
  an online fusion learns a per-source projection and mixes on top of
  it, while an offline one mixes first and reduces afterwards. On the
  amazon_women smoke grid (one HP point, selection metric, no folds)
  the offline `hybrid_concat_pcaw128` leads at 0.00861 against 0.00662
  for the best online fusion and 0.00679 for the best single backbone,
  with the nine online strategies clustered between 0.00474 and
  0.00662 — the order of operations separates the routes more than the
  choice of strategy does. Indicative only; the folds decide.

- **DeepStyle (paper formulation)**: the item style term is
  `s_i = E·f_i − l_cat(i)` — a linear projection `E` (`D_backbone → d`)
  minus a **learned category embedding** subtracted in the style space,
  and the score is `p_u^T (s_i + q_i)` with a single user vector. On
  the Amazon datasets, whose per-item category varies (declared
  `expects_categories: true`), this makes DeepStyle differ from VBPR.
  On Tradesy, which has no category (`expects_categories: false`,
  enforced at preprocess), every item maps to a single null category,
  so `l_cat(i)` is constant across items; the `p_u·l₀` term is
  item-independent and cancels in every BPR pairwise comparison, so
  DeepStyle **analytically degenerates into a restricted VBPR**
  (`γ_u ≡ θ_u`, `k_v = k`, no `β_i`, no `β'`) — not into the production
  VBPR. This is the expected, verified behaviour (see
  `tests/recommenders/test_deepstyle_paper.py::TestTradesyDegeneration`),
  not a bug. An earlier MLP-style variant (which did not subtract a
  category vector) was removed in commit `60c7436`.
- **ACF per-region fusion (3.0.0-rc.2)**: ACF consumes per-item
  component maps, so it could not read the pooled `hybrid_*` artifacts
  and was absent from the 3.0.0-rc.1 battery. It now joins the fusion
  family through **early per-region fusion**: each component source
  `(n_items, R, D_i)` is flattened to `(n_items · R, D_i)`, the same
  strategy that the pooled recommenders use runs on those rows, and the
  result is folded back to `(n_items, R, D_fused)`. The artifacts are
  named `hybrid_<strategy>…_comp.npy` (offline) and
  `hybrid_<strategy>_learned_D<dim>_comp.json` (online), with `_comp`
  last so they stay routed to component models only, and the pass is
  gated by the recommender roster — no separate configuration key.
  Three properties make the axis comparable with the pooled models and
  are pinned by tests
  (`tests/test_fusion_per_region_components.py`): every region is fused
  independently of the others; any fitted parameter (a PCA basis, the
  learned projections `D_backbone → D`) is ONE, shared by every region,
  so the regions stay in a common space and the component attention
  keeps comparing like with like; and a PCA fit sees only the rows owned
  by training items (`i · R + r`), never a validation or test item. The
  `alignment: pca` route is not built for components and is skipped with
  a log rather than emitting a wrong artifact. Both `sum` and `mean` are
  kept for symmetry with the other recommenders even though they are the
  same model here (`mean = ½·sum`, absorbed by the unpenalised `W_c`);
  `concat` and the aligned additive strategies are NOT equivalent,
  because each aligned source is L2-normalised before the operation.
  Full record: `docs/reliability-sdd/S05.md`.
- **ACF history**: the paper sums over the full `R(u)`; `max_history`
  (H = 50) is an implementation bound. When `|R(u)| > H` the profile is a
  **seeded uniform subsample** (`history_seed` = the run seed), never the
  lowest item indices, which correlate with popularity in the DVBPR
  splits. `max_history: null` keeps the full history.
- **Trainable-parameter counts differ across backbones** because `E`'s
  input is the native dim — an expected second-order effect, reported
  per cell (`n_trainable_params` column), never hidden.

## 8. Known confounder to acknowledge (defense question 13)

CLIP and DINOv2 are both ViT-B under the hood; if CLIP wins, the design
cannot separate architecture from pre-training data (2B image-text
pairs). The honest claim: this benchmark compares **extractors as
available in practice** (architecture + weights + canonical recipe),
not pure architectures.

## 9. Recommended robustness checks (before the defense)

Run the comparison at ≥2 values of `d` (64, 128), under both protocols,
and with multiple seeds (`seeds: [...]` is supported), verifying the
backbone ranking is stable. Each of these preempts a standard committee
question.

## 10. Declarações de protocolo (pt-BR — para a dissertação)

Declarações fechadas pelas auditorias de diagnóstico, em tom de
protocolo, prontas para migrar ao capítulo de metodologia.

### 10.1. Escopo do ajuste fino (fine-tuning) dos backbones

O ajuste fino dos backbones visuais é supervisionado exclusivamente pela
categoria dos itens e utiliza, de forma transdutiva, as imagens e os
rótulos de categoria de todo o catálogo — incluindo itens posteriormente
reservados para validação e teste da tarefa de recomendação —, sem em
nenhum momento acessar as interações usuário-item nem quais itens compõem
os conjuntos de validação/teste. Trata-se de uso a priori de metadados de
catálogo (equivalente ao emprego de representações pré-treinadas sobre
todos os itens), e não de vazamento de sinal de interação; a partição de
recomendação (treino/validação/teste do DVBPR) permanece fixa e é
consumida apenas nas etapas posteriores.

### 10.2. Seleção de modelo em validação

A seleção de hiperparâmetros e a parada antecipada (early stopping,
ndcg@10) são conduzidas sobre os usuários de **validação**, mascarando os
itens de treino de cada usuário. O conjunto de teste não é acessado em
nenhum momento do treinamento ou da seleção — é consumido apenas na
avaliação final —, de modo que as métricas reportadas são uma estimativa
fora da amostra, não enviesada pela seleção. Durante a validação, o item
de teste do usuário permanece no conjunto de candidatos e compete como um
item qualquer; isso é neutro entre os modelos e nada revela ao modelo.

Regra de seleção (3.0.0): o corte de seleção é K = 10, declarado uma
única vez no código (`SELECTION_K_VALUES`); uma métrica de seleção que o
avaliador de treino não produz interrompe a execução antes de qualquer
treinamento. A **primeira observação finita em validação é a vencedora,
mesmo quando vale exatamente 0,0**; depois disso, apenas melhora estrita
a substitui, e empates mantêm a vencedora anterior. Uma métrica ausente,
não escalar ou não finita é uma falha explícita — não gera vencedora nem
marcador de sucesso —, de modo que um resultado legitimamente nulo e um
resultado inválido nunca se confundem, e uma célula com métrica zero é
avaliada e reportada como qualquer outra. O orçamento por dataset
(épocas, paciência, métrica, subamostra de validação, número de trials)
é consumido de forma idêntica por todos os caminhos de treinamento.

### 10.3. Normalização pré-fusão

Antes de qualquer fusão element-wise, cada fonte é L2-normalizada por
vetor (`normalize_before_fusion: true`, padrão). Justificativa: a razão
entre as normas médias das fontes medida é de ~16× nas features brutas e
~9× após o alinhamento por PCA — sem a normalização, a fonte de maior
norma dominaria as fusões aditivas, tornando a comparação entre
estratégias um artefato de escala.

O ponto em que a normalização é aplicada depende do caminho (auditoria
de fusão, 2026-09-04):

| Caminho | Estratégias | Onde normaliza |
|---|---|---|
| Offline nativo | `concat`, `pca`, `pca_per_model` | nas features nativas, antes da operação; a matriz de fit da PCA conjunta também é normalizada |
| Online aprendido (`alignment.method: learned`) | família equal-dim | depois do `Linear(D_i → dim)` por fonte, antes da operação |
| Alinhamento por PCA (`alignment.method: pca`) | família equal-dim | a PCA por fonte é ajustada nas features brutas; a normalização é aplicada às fontes já reduzidas |
| Online não aprendido (`adaptive_gated` sobre fontes de mesma largura — `alignment: pca` ou `alignment: none` com fontes projetadas) | família equal-dim | **por fonte, uma única vez, no carregamento, antes da operação** (`load_embedding`, receita do sidecar versão 2) |

Correção declarada (3.0.0, registro S02): até a 2.12.1 o caminho online
não aprendido empilhava as fontes como armazenadas e ignorava a flag
`normalize` do sidecar — era o único caminho desta seção alimentado por
fontes não normalizadas, exatamente na família aditiva em que a fonte de
maior norma domina. A partir da 3.0.0 cada fonte é L2-normalizada por
vetor no carregamento (linhas nulas permanecem nulas), nos caminhos denso
e preguiçoso, e o sidecar recebe `recipe_version: 2`; um sidecar sem o
campo é identificado como legado. **Aviso de comparabilidade histórica:
todo modelo ou resultado treinado a partir de sidecars
`hybrid_adaptive_gated_*` sob a 2.x (`hybrid_adaptive_gated_pca_*.json`,
`hybrid_adaptive_gated_p*.json`) não é comparável com resultados
produzidos pela 3.x e deve ser regenerado em namespace próprio.** As
células de alinhamento aprendido (`*_learned_D*.json`) e todas as fusões
offline (`.npy`) não são afetadas. Nenhum ajuste de PCA mudou; apenas o
consumo das fontes no caminho online.

Consequências declaradas:

- **`sum` = 2 × `mean`**, exatamente, nos dois caminhos: as fontes são
  unitárias antes da operação e nada normaliza o vetor fundido depois.
  O fator é absorvido pela projeção linear do recomendador (`E`), de modo
  que a diferença residual entre as duas células é apenas a penalidade L2
  sobre `E` (treinar com `sum` equivale a treinar `mean` com `l2_reg`
  quatro vezes menor) e a trajetória de otimização. `sum` permanece na
  bateria como controle de sensibilidade à escala, não como mecanismo de
  fusão distinto.
- **`mean`, `sigmoid_gated`, `weighted_mean` e `softmax_weighted`** são a
  mesma combinação convexa `w·e_cnn + (1−w)·e_vit` com `w` fixo
  (0.500, 0.594, 0.300/0.700 e 0.731 respectivamente). O que distingue
  os três membros ponderados é como `w` é derivado do valor configurado
  (direto, sigmoides normalizadas, softmax), conforme a Seção 3.4 da
  qualificação. `weighted_mean` cobre os dois lados (`w_cnn` 0.3 e 0.7)
  para que a varredura não fique restrita ao semiplano CNN-dominante.
- **`concat`** opera em 2816 dimensões nativas, contra 128 da família
  equal-dim. A capacidade da camada visual aprendida é comparável: a
  família equal-dim aprende `Linear(2048→128)` + `Linear(768→128)` mais
  `E(128→k_v)`, e `concat` aprende `E(2816→k_v)`; com `k_v = 128` os
  totais coincidem. A diferença estrutural é profundidade e a
  normalização intermediária, não contagem de parâmetros, e deve ser
  reportada junto do resultado de `concat`.

### 10.4. Desempate no ranking

Empates exatos de score são resolvidos por uma permutação aleatória fixa
dos itens, seedada a partir do seed global do run e compartilhada por
todos os modelos e trials de uma execução `(dataset, seed)`.
Justificativa: a correlação de Spearman entre `item_idx` e popularidade é
de −0,34 a −0,45 nos quatro datasets, o que tornaria o desempate por id
equivalente a um desempate por popularidade. A frequência real de empates
exatos é medida e registrada durante a própria bateria.

### 10.5. Conjunto de candidatos

A avaliação é full ranking sobre o catálogo inteiro, incluindo itens sem
qualquer interação de treino — itens frios com representação visual real
fazem parte do objeto de estudo do benchmark. Na validação, o item de
teste do usuário permanece como candidato (ver 10.2).

### 10.6. Orçamento e fluxo da busca de hiperparâmetros

O orçamento da busca de hiperparâmetros (número de trials, métrica de
seleção ndcg@10 em validação, paciência, épocas máximas, tamanho do
subsample de validação) é **uniforme para todos os recomendadores de um
mesmo dataset** — configurado numa fonte compartilhada única, nunca por
modelo; apenas os espaços de busca são por modelo, pois cada um tem seus
próprios hiperparâmetros. A busca completa é executada **apenas na seed
primária** de cada dataset; nas demais seeds, a melhor configuração
encontrada é re-treinada (replay), com parada antecipada em validação
ativa por seed. A avaliação final de cada célula consome o checkpoint do
melhor trial (cujo early stopping já rodou em validação) — não há
re-treino pós-busca, e o procedimento é idêntico para todos os modelos.
O conjunto de teste permanece intocado até a avaliação final.

### 10.7. Validação cruzada K-fold por usuário

Seguindo Rendle et al. (2009, §6.2), que repetem o experimento de
leave-one-out dez vezes sobre novas partições e otimizam os
hiperparâmetros por busca em grade apenas na primeira rodada, mantendo-os
constantes nas demais, a validação cruzada particiona **usuários** em K
folds mutuamente exclusivos. No fold *i*, os usuários do fold saem do
treino; o modelo é treinado nos demais folds — apenas com o histórico de
treino desses usuários (o item de teste fica fora do treino para todos os
usuários, como no protocolo sequencial) — com os hiperparâmetros
**congelados** da busca prévia (ou fixados na configuração), sob a seed
`seed + i`; os usuários retidos são incorporados por *fold-in* — apenas
as linhas de usuário são otimizadas, com todos os demais parâmetros
congelados, a partir do perfil (train ∪ val) — estratégia que o próprio
paper do BPR (§2) indica como aplicável ao método; e cada um é avaliado
no seu único item-alvo (o held-out de teste). Como cada usuário é
avaliado exatamente uma vez ao longo dos K folds, os registros por usuário
são concatenados num único conjunto por célula, preservando o usuário como
unidade dos testes pareados; a média e o desvio-padrão entre folds são
reportados apenas como variabilidade descritiva, combinada (partição e
otimização). A busca de hiperparâmetros não é aninhada nos folds, por
reprodução do procedimento original.

### 10.8. Recuperação de falta de memória de GPU (OOM)

Quando um treinamento esgota a memória da GPU
(`torch.cuda.OutOfMemoryError`), o job é repetido até duas vezes, e cada
repetição altera apenas a forma de execução, nunca o que é calculado.
Primeiro, as features visuais passam a ser lidas sob demanda, em vez de
residirem inteiras na memória (caminho verificado como numericamente
equivalente). Se a leitura sob demanda já estava ativa, cada passo de
otimização passa a ser dividido em micro-batches com acumulação de
gradiente: o batch de 4096 triplas vira 2 × 2048 e depois 4 × 1024, com
um único passo do otimizador sobre o batch inteiro. Como a perda BPR e o
termo L2 das linhas amostradas são médias sobre as triplas e o termo L2
compartilhado é somado uma única vez, a ponderação de cada micro-batch
por `n_k / N` reproduz exatamente o gradiente do batch completo
(verificado por teste para todos os recomendadores). A única diferença
estocástica é que as máscaras de dropout (VNPR) são sorteadas por
micro-batch. O orçamento de memória do ranking de validação também cai
pela metade a cada repetição. Toda ativação é registrada em
`results/runs/<run_id>/oom_recoveries.csv`, com a identidade do job, a
tentativa, a ação tomada e o desfecho (`retrying`, `recovered` ou
`failed`), para que os resultados obtidos sob recuperação possam ser
identificados.

### Expanded frozen grid (2026-09-12)

The full frozen run searches `total_dim = [64, 128, 256]`,
`learning_rate = [0.0003, 0.001, 0.01]`, and
`l2_reg = [0.00001, 0.0001, 0.001]`. BPR, VBPR and DeepStyle receive
27 selection trials per dataset/embedding cell. VNPR additionally searches
`dropout = [0.0, 0.5]`; ACF searches `att_hidden = [64, 128]` with
`max_history = 50`, giving each 54 trials. AVBPR, when enabled, also has
54 trials through its attention-width axis; it is excluded from this run.
Model-specific regularization coefficients remain fixed as declared in
`configs/recommenders.yaml`. Thus the common L2 axis does not vary every
regularization group in BPR or VBPR.

These are unequal search budgets: VNPR and ACF receive twice as many
validation selection opportunities. Comparisons must disclose this budget
alongside model quality and measured search cost. Counts exclude fusion
parameter variants and subsequent fold retraining. The local full-run
configuration covers all four datasets, frozen features only, and two
folds. Optuna has no separate ACF search-space override and derives its
categorical axes from the same declarations.
