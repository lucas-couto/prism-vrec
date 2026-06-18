# Recipes

Common shapes the pipeline can take, expressed as the smallest set of
config edits that produces them. Every recipe assumes a clean checkout
and runs from the repo root with `python main.py` (or
`docker compose up -d --build`).

The pipeline is **opt-in** by configuration: empty `*_enabled` lists
auto-skip the matching step, `pipeline.condition` toggles the
fine-tuning battery, and `--step` / `--from` / `--to` give per-run
overrides. None of these knobs is mutually exclusive.

> See [`docs/extending.md`](extending.md) for how to add new
> extractors, fusions, recommenders or datasets to the registry.
> Recipes here only configure existing components.

---

## 0. "Smoke test — validate my install in under 5 minutes"

No editing required, a `configs/smoke/` bundle ships with the framework.
Synthetic dataset (100 users, 200 items, in-process images), one
extractor, one fusion, two recommenders, 1 Optuna trial, 2 epochs.
Numerical results are meaningless by design; the goal is to confirm
that `preprocess → extract → fuse → train → evaluate → statistical →
export_best` completes without errors on your host.

Run:

```bash
python main.py --all --config-dir configs/smoke
```

Outputs land in `data/smoke_*`, `results/smoke`, `logs/smoke` (so they
do not collide with a real run). Delete those directories to repeat
from scratch.

`--config-dir` is the canonical way to switch config bundles without
touching `configs/default.yaml`. Useful for ablation profiles, CI smoke
jobs, or experiment-specific configs.

---

## 0b. "Match the sampled-evaluation protocol used by prior work"

Some recsys baselines (NCF-style papers, early DVBPR variants)
report metrics over a small pool of negative samples instead of a
full-ranking against the catalogue.  The two protocols are
**statistically inconsistent** (Krichene & Rendle 2020) — model
orderings can flip between them — so use this only for direct
comparability with that prior work, never as your primary number.

CLI override (no YAML edit needed):

```bash
python main.py --all --eval-protocol sampled
```

Or pin it permanently in `configs/evaluation.yaml`:

```yaml
evaluation:
  protocol: "sampled"
  n_negatives: 100
  negative_sampling_seed: 42
```

The seed must stay constant across model runs you compare — paired
Wilcoxon tests need identical per-user pools.

---

## 0c. "Run the pipeline with multiple seeds for variance estimates"

A single seed gives a point estimate per cell.  For thesis-grade
numbers reviewers will trust, run the pipeline 3-5 times with
different seeds and report mean ± std.  The framework supports this
natively without manual config duplication.

YAML:

```yaml
# configs/default.yaml
seeds: [42, 99, 7]
```

Or CLI:

```bash
python main.py --all --seeds 42,99,7
```

What happens:

- The pipeline runs once per seed.
- ``paths.results`` and ``paths.checkpoints`` are automatically
  suffixed with ``_seed{N}`` so the runs do not overwrite each other.
- Shared inputs (``data/raw``, ``data/processed``, ``data/embeddings``)
  are reused — extract / preprocess only run for the first seed and
  are then idempotently skipped.
- After the last seed finishes, the framework writes
  ``results/aggregated_across_seeds/evaluation_multi_seed.csv`` with
  ``mean_across_seeds``, ``std_across_seeds``, ``median``, ``min``,
  ``max`` and ``n_seeds`` per (dataset, recommender, extractor,
  fusion, condition, metric, k) cell.

Cost scales linearly with the number of seeds.  Estimate the single-
seed cost first (see ``configs/smoke/`` for a quick check), then
multiply.  Three seeds on the full DVBPR pipeline ≈ 3× the wall-clock
and 3× the EC2 cost — budget accordingly.

The seed configured in each iteration also drives every cell-level
seed derivation (``_derive_job_seed`` in ``src/utils/training.py``),
so model initialisation, batch shuffling, and BPR negative sampling
all change between seeds.

---

## 1. "I just want frozen embeddings + plain BPR"

No fine-tuning, no fusion. The cheapest viable run.

```yaml
# configs/default.yaml
pipeline:
  condition: frozen           # skips finetune + evaluate_finetuning

# configs/extractors.yaml
extractors_enabled:
  - resnet50

# configs/fusion.yaml
fusion_strategies_enabled: []  # skips fuse step entirely

# configs/recommenders.yaml
recommenders_enabled:
  - bpr
```

Run:

```bash
python main.py
```

Pipeline executes: download → preprocess → extract → train → evaluate
→ statistical → export_best.

---

## 2. "Compare two extractors with frozen embeddings, no fusion"

Same shape as recipe 1 but with multiple backbones competing.

```yaml
# configs/extractors.yaml
extractors_enabled:
  - resnet50
  - vit_b16

# configs/fusion.yaml
fusion_strategies_enabled: []

# configs/recommenders.yaml
recommenders_enabled:
  - bpr
  - vbpr
```

`pipeline.condition: frozen` keeps the run finetune-free.

---

## 3. "Compare two fusion strategies on a single recommender"

```yaml
# configs/extractors.yaml
extractors_enabled: [resnet50, vit_b16]
fusion_extractors:  [resnet50, vit_b16]   # the pair fused together

# configs/fusion.yaml
fusion_strategies_enabled:
  - mean
  - concat

# configs/recommenders.yaml
recommenders_enabled:
  - vbpr

# configs/default.yaml
pipeline:
  condition: frozen
```

---

## 4. "Full battery, frozen + fine-tuned, all extractors, all fusions"

The default `configs/` ship this configuration.

```yaml
# configs/default.yaml
pipeline:
  condition: both
```

```bash
python main.py            # or docker compose up -d --build
```

---

## 5. "Re-evaluate after adding one new extractor"

You finished a long run last week and want to add `dinov2_vitb14` to
the comparison **without** rerunning the existing extractors.

```yaml
# configs/extractors.yaml
extractors_enabled:
  - resnet50
  - vit_b16
  - dinov2_vitb14    # new
```

```bash
# Extract + train + evaluate only the new extractor's slice; the
# existing .npy / checkpoint files for resnet50 and vit_b16 are
# detected and skipped automatically (idempotent steps).
python main.py --from extract
```

---

## 6. "Quick run on a custom dataset before launching a long one"

Drop your data under `plugins/datasets/<name>/` (see
[`docs/extending.md` § 4](extending.md#4-add-a-dataset)).

```yaml
# configs/default.yaml
datasets:
  - my_dataset             # only this one, DVBPR won't run
pipeline:
  condition: frozen
```

```bash
python main.py --step preprocess
python main.py --step extract
python main.py --step train
```

The fixed step ordering keeps `download` at the top of the list, but
because our directory already has the files on disk, the download
step is a no-op.

---

## 7. "Just compute the FT classification metrics on existing checkpoints"

Existing v2 fine-tuning checkpoints under `checkpoints/finetuning/`
are scanned and post-hoc metrics (top-K, F1, confusion matrix) are
written to `results/finetuning/`.

```bash
python main.py --step evaluate_finetuning
```

Idempotent, combinations whose JSON already exists are skipped.

---

## 8. "Stop after extraction and ship the embeddings somewhere else"

```bash
python main.py --from download --to extract --condition frozen
```

The output `.npy` files land under `data/embeddings/<dataset>/`. From
there a downstream system can train any recommender it likes, this
framework no longer needs to be in the loop.

---

## 9. "I have a custom recommender; everything else stays default"

After adding the plugin under `plugins/recommenders/<name>.py`:

```yaml
# configs/recommenders.yaml
recommenders_enabled:
  - bpr
  - vbpr
  - my_model           # new

# (optional) hyperparameter grid for my_model
my_model:
  custom_param: [16, 32, 64]
```

```bash
python main.py --from train     # train + evaluate + statistical
```

The earlier steps' outputs (extracts, fusions) are reused.

---

## 9b. "Run ACF (component-level + item-level attention)"

ACF is the only recommender that consumes per-item *component*
embeddings (`<extractor>_D<dim>_comp.npy`, shape `(n_items, M, D)`)
instead of the pooled `(n_items, D)` ones. Two edits enable it: turn on
component extraction, and add `acf` to the recommender list.

```yaml
# configs/default.yaml
extract_components: true        # extractors also emit the 3-D *_comp artifacts

# configs/recommenders.yaml
recommenders_enabled:
  - bpr
  - vbpr
  - acf                         # routed only to the *_comp artifacts

acf:
  att_hidden: [64, 128]
  max_history: [50]             # H: items per user profile (item-level attention)
```

```bash
python main.py --from extract   # re-extract to add the *_comp files, then train+eval
```

Notes:

- The pooled `.npy` files are **not** recomputed — only the new `_comp`
  files are added (idempotent extraction). Existing non-ACF results are
  untouched.
- Component artifacts are `M`× larger on disk and re-run each backbone
  forward, so `extract` with `extract_components: true` costs more than
  a pooled-only run — budget accordingly.
- The user history is built from **train** interactions only, so
  validation/test items never enter the profile.

---

## 10. "Dry-run, what is still pending in Battery 1?"

```bash
python main.py --inspect-pending frozen
```

Prints a per-`(dataset, model)` count of training jobs that have not
completed yet. Useful before launching a multi-day rerun on a fresh
GPU.

---

## What each toggle controls (quick reference)

| Knob | Lives in | Effect when empty / off |
| --- | --- | --- |
| `datasets:` | `configs/default.yaml` | Every step that loops datasets becomes a no-op. |
| `extractors_enabled` | `configs/extractors.yaml` | `extract` and `finetune` skip. |
| `extract_components` | `configs/default.yaml` | Off → no `*_comp` artifacts written (ACF cannot run); pooled output unchanged. |
| `finetuning.extractors` | `configs/finetuning.yaml` | `finetune` and `evaluate_finetuning` skip. |
| `fusion_strategies_enabled` | `configs/fusion.yaml` | `fuse` step skips entirely. |
| `recommenders_enabled` | `configs/recommenders.yaml` | `train` step skips entirely. |
| `pipeline.condition: frozen` | `configs/default.yaml` | Drops `finetune` + `evaluate_finetuning`. |
| `pipeline.condition: finetuned` | `configs/default.yaml` | Drops `extract`. |
| `--step NAME` | CLI | Bypasses the condition filter, runs `NAME` regardless. |
| `--from STEP --to STEP` | CLI | Slice of `STEP_ORDER`; condition filter still applies. |
