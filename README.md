<p align="center">
  <img src="assets/prism_vrec_readme_banner.svg" alt="Prism VRec — a reproducible framework for visual recommendation research">
</p>

# Prism VRec

**A reproducible framework for evaluating visual feature extractors in recommender systems, including pure architectures, CNN-Transformer hybrids, foundation models, and late-fusion strategies.**

Comes with a fashion starter pack (4 pre-configured DVBPR datasets) and supports any visual recommendation domain via CSV.

Authored by **Lucas Silva Couto** ([ORCID 0009-0000-0641-8166](https://orcid.org/0009-0000-0641-8166)) with **Prof. Dr. Marcos Aurelio Domingues** ([ORCID 0000-0001-7195-0714](https://orcid.org/0000-0001-7195-0714)), at the Graduate Program in Computer Science, Universidade Estadual de Maringá (UEM). One reported use of the framework is the author's M.Sc. dissertation (see [§16](#16-citation)).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20357510.svg)](https://doi.org/10.5281/zenodo.20357510)

---

## Table of contents

1. [Overview](#1-overview)
2. [Quick start](#2-quick-start)
3. [Pipeline](#3-pipeline)
4. [Configuration](#4-configuration)
5. [Datasets](#5-datasets)
6. [Extractors](#6-extractors)
7. [Fusion strategies](#7-fusion-strategies)
8. [Recommender models](#8-recommender-models)
9. [Extending the framework](#9-extending-the-framework)
10. [Evaluation](#10-evaluation)
11. [Performance optimisations](#11-performance-optimisations)
12. [Reproducibility](#12-reproducibility)
13. [Project structure](#13-project-structure)
14. [Hardware requirements](#14-hardware-requirements)
15. [Citation](#15-citation)
16. [License](#16-license)

---

## 1. Overview

The framework lets you compare visual feature extractors as drop-in front-ends for recommender systems under a single, reproducible protocol. It supports four families of extractors, eleven late-fusion strategies (ten offline plus one learned online), and six recommender models, all wired through a configurable, idempotent pipeline.

**Extractor families covered out of the box:**

| Family                                                | Extractors                                                                |
| ----------------------------------------------------- | ------------------------------------------------------------------------- |
| Pure architectures                                    | ResNet50 (CNN), ViT-B/16 (Transformer)                                    |
| CNN+Transformer architectural hybrids                 | CvT-13, CoAtNet-0, LeViT-256                                              |
| Foundation models (ViTs with specialised pretraining) | CLIP ViT-B/32 (contrastive multimodal), DINOv2 ViT-B/14 (self-supervised) |
| Late-fusion strategies                                | 11 strategies (10 offline + 1 learned online) over ResNet50 + ViT-B/16 embeddings |

**Research question the framework lets you answer:** _In visual recommendation, does CNN+ViT hybridisation, whether by late fusion or architectural design, compete with (a) pure architectures and (b) foundation models, as state-of-the-art baselines?_

The repository ships a **fashion starter pack** (the four DVBPR datasets) for out-of-the-box reproduction. The framework itself is domain-agnostic; any other visual recommendation domain can be plugged in via `CSVDatasetProvider` (see [§5](#5-datasets) and [§9](#9-extending-the-framework)).

The default pipeline runs two batteries:

- **Battery 1, frozen extractors**: full evaluation with pretrained extractors used as-is.
- **Battery 2, fine-tuned extractors**: domain adaptation via category classification on the extractors listed in `configs/finetuning.yaml`.

Each `*_enabled` list in the YAML configs (extractors, fusions, recommenders) controls the experiment grid; the total number of training jobs is the Cartesian product of those lists times the dataset and hyperparameter grids.

> **A note on terminology.** Throughout this README, _foundation models_ refers to ViTs with specialised pretraining paradigms (contrastive multimodal for CLIP, self-supervised for DINOv2). They are **not** architectural CNN+ViT hybrids, their image encoder is a pure ViT.

---

## 2. Quick start

### Prerequisites

- Docker (and Docker Compose v2).
- Optional, for GPU acceleration: an NVIDIA GPU plus the NVIDIA Container Toolkit. When absent, the same image runs on CPU.

### Local, one command

```bash
git clone https://github.com/lucas-couto/prism-vrec.git
cd prism-vrec

docker compose up -d --build    # builds the image and starts the pipeline
docker compose logs -f          # follow progress
```

The container runs `python main.py` with the run plan declared in `configs/default.yaml`. The defaults run Battery 1 + Battery 2 end-to-end.

(Optional) drop a HuggingFace token in `.env` for faster pretrained weight downloads:

```bash
echo "HF_TOKEN=hf_your_token_here" > .env
```

### Cloud, one block, copy-paste

Works on any pod with a CUDA + PyTorch image (e.g. `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-devel-ubuntu22.04`) with a persistent volume mounted at `/workspace`:

The image must ship **CUDA 12.8 or newer**: `torch` is pinned to the 2.8
series, the earliest that provides `sm_120` kernels for Blackwell cards
(RTX 5090, RTX PRO 6000). Older CUDA images abort with "no kernel image
is available for execution on the device" on that hardware. Only the
host **driver** has to satisfy this — the `torch` wheel bundles its own
CUDA runtime, so a plain base image plus `pip install -e .` works too.

```bash
cd /workspace
git clone https://github.com/lucas-couto/prism-vrec.git
cd prism-vrec
pip install -e . && python main.py
```

The DataLoader workers, prefetch factor, training batch size and VNPR chunk size are picked at startup from the host's CPU count, cgroup memory budget and visible GPU VRAM (see [§12 Reproducibility](#12-reproducibility) and [`docs/observability.md`](docs/observability.md)). No manual tuning needed.

### Inspect remaining work

```bash
docker compose --profile tools run --rm runner --inspect-pending frozen
docker compose --profile tools run --rm runner --inspect-pending finetuned
```

Lists pending `(dataset, model)` combinations after accounting for completed checkpoints.

### Reshaping the run

The pipeline is opt-in by configuration: empty `*_enabled` lists auto-skip the matching step, and `pipeline.condition` toggles the fine-tuning battery. Common shapes, frozen-only, fusion-only, single-recommender, custom-dataset, are documented in [`docs/recipes.md`](docs/recipes.md).

### Results

```
results/tables/
├── evaluation_aggregated.csv              # Long-format mean per (recommender, extractor, fusion, condition, metric, k)
├── bootstrap_ci.csv                       # Idem + ci_lower / ci_upper / ci_width
├── statistical_tests.csv                  # Friedman + pairwise Wilcoxon rows (test_type discriminator)
│
├── {dataset}_evaluation_frozen.csv        # Battery 1 per-user metrics (granular, kept for back-compat)
├── {dataset}_evaluation_finetuned.csv     # Battery 2 per-user metrics
├── {dataset}_evaluation_combined.csv      # Both batteries merged
├── {dataset}_summary_{metric}.csv         # Per-model mean + bootstrap CI (granular)
├── {dataset}_friedman_{metric}.csv        # Friedman omnibus test (granular)
└── {dataset}_pairwise_{metric}.csv        # Wilcoxon + correction + effect sizes (granular)

results/best_hyperparams.json              # Winning hyperparams per (dataset, model, embedding)
results/models/<dataset>/*_best.pt         # Best checkpoint per cell
```

The three long-format files at the top consolidate the ~160 granular per-(dataset, test_type, metric, k) CSVs into one row per observation with explicit identifier columns. Use them for thesis-time analysis (`pandas.read_csv` + `df.query`); the granular CSVs are kept for backwards compatibility.

---

## 3. Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE OVERVIEW                                 │
│                                                                          │
│  01 Download ─→ 02 Preprocess ─→ 03 Extract ─→ 04 Fuse ─→ 05 Train     │
│     (DVBPR)      (splits+imgs)    (8 models)   (11 strat)  (parallel)   │
│                                                                          │
│                        ─→ 06 Evaluate ─→ 07 Stats ─→ 08 Best Hyperparams │
└──────────────────────────────────────────────────────────────────────────┘
```

| Step | Module                             | Description                                                                      |
| ---- | ---------------------------------- | -------------------------------------------------------------------------------- |
| 01   | `src/steps/download.py`            | Download every dataset registered for the names in `datasets:`                   |
| 02   | `src/steps/preprocess.py`          | Build `train.csv` / `val.csv` / `test.csv` and extract per-item images           |
| 03   | `src/steps/extract.py`             | Extract frozen embeddings for every `(extractor, dataset, dim)`                  |
| 03b  | `src/steps/finetune.py`            | Fine-tune 5 extractors via category classification, then re-extract              |
| 03c  | `src/steps/evaluate_finetuning.py` | Post-hoc top-K / F1 / confusion matrix on the deterministic FT val split         |
| 04   | `src/steps/fuse.py`                | Apply 11 fusion strategies (parallelised across CPU cores)                       |
| 05   | `src/steps/train.py`               | Recommender grid search (one job per `(dataset, model, embedding, hyperparams)`) |
| 06   | `src/steps/evaluate.py`            | Full-ranking evaluation on the test set                                          |
| 07   | `src/steps/statistical.py`         | Bootstrap CIs, Friedman omnibus, pairwise Wilcoxon with effect sizes             |
| 08   | `src/steps/export_best.py`         | Export the winning hyperparameters as a single JSON                              |

Every step is idempotent, re-running after an interruption resumes from the last checkpoint. `main.py` orchestrates them by reading `configs/default.yaml`.

---

## 4. Configuration

Every aspect of a run is controlled by YAML files under `configs/`. Edit them and re-run `docker compose up -d --build` (or `python main.py`).

### `configs/default.yaml`

```yaml
seed: 42
device: "cuda"

# Names listed here must resolve to a registered DatasetProvider.  The
# four DVBPR datasets ship pre-registered.  Anything dropped under
# plugins/datasets/<name>/ becomes valid here automatically.
datasets:
  - "amazon_fashion"
  - "amazon_women"
  - "amazon_men"
  - "tradesy"

# Per-dataset contract: whether the dataset is EXPECTED to ship item
# category labels.  Enforced at preprocess time — if the raw data
# disagrees with the declaration the run fails loudly instead of
# silently flipping DeepStyle degeneration and fine-tuning transfer.
# Datasets without an entry skip the check.
dataset_contracts:
  amazon_fashion: { expects_categories: true }
  amazon_women:   { expects_categories: true }
  amazon_men:     { expects_categories: true }
  tradesy:        { expects_categories: false }

pipeline:
  run_all: true # set to false to use start_from / stop_at
  start_from: null # e.g. "train"
  stop_at: null # e.g. "fuse"
  condition: "both" # "frozen", "finetuned" or "both"
```

CLI flags (`--all` / `--step` / `--from` / `--to` / `--condition`) override the YAML when present.

### `configs/extractors.yaml`

```yaml
# Which extractors to actually run. Empty / missing = none run.
extractors_enabled:
  - resnet50
  - vit_b16
  - cvt_13
  - coatnet_0
  - levit_256
  - clip_vitb32
  - dinov2_vitb14
  - convnext_base

extractors:
  resnet50: { ..., raw_dim: 2048, role: primary }
  vit_b16: { ..., raw_dim: 768, role: primary }
  cvt_13: { ..., raw_dim: 384, role: hybrid }
  coatnet_0: { ..., raw_dim: 768, role: hybrid }
  levit_256: { ..., raw_dim: 512, role: hybrid }
  clip_vitb32: { ..., raw_dim: 512, role: foundation }
  dinov2_vitb14: { ..., raw_dim: 768, role: foundation }
  convnext_base: { ..., raw_dim: 1024, role: hybrid }

fusion_extractors: ["resnet50", "vit_b16"] # primary pair sent into fusion
batch_size: 256 # extraction batch size
checkpoint_every: 500 # save partial extraction every N batches
```

Extraction is **native-dim** (v2 protocol): each extractor saves the backbone's own pooled feature. `raw_dim` is a declaration validated against a probe forward through the real model — a mismatch fails extraction loudly. The learned projection `E` inside each recommender maps the native feature to the model's visual dim, derived from the shared dimension budget (`common.total_dim` in `recommenders.yaml`).

Optionally, a **fixed linear projection** brings every extractor into one shared width — for a reviewer who asks that all backbones emit, say, 128-d, or to feed the element-wise fusion family equal-dim sources with nothing learned online:

```yaml
projection:
  method: pca # none (default) | random | pca
  dim: 128
  seed: 42 # method: random only
```

This writes `<extractor>_p128.npy` **alongside** the native artifact, never in place of it, so native and projected can be trained and compared in the same battery. Two flags then decide what *consumes* them:

```yaml
# configs/recommenders.yaml — which variants the recommenders train on
embedding_variants: both # native | projected | both

# configs/fusion.yaml — which variants feed the fusion
extractor_variants: native # native | projected | both
```

`extractor_variants: projected` is the shortcut this feature exists for: the sources already share a width, so the whole `alignment:` block is bypassed — nothing learned online, no PCA fit inside the fuse step. Outputs stay apart by the same token (`hybrid_mean` vs `hybrid_mean_p128`), and a hybrid built from projected sources is classified with them by `embedding_variants`. `random` is a seeded semi-orthogonal matrix (data-independent, so it cannot leak val/test items); `pca` is fit on train items only, like the fusion alignment; `pca_whitened` adds per-component variance equalisation on the same fit set. Any extractor overrides the block under `extractors.<name>.projection`. Already-extracted embeddings are projected in a linear pass — the backbone is not loaded again. Read [`docs/protocol.md` §1b](docs/protocol.md) before reporting a comparison run on projected artifacts: a fixed projection is a variable, not a free normalisation.

To ablate an extractor, just remove its name from `extractors_enabled`, its block under `extractors:` stays as catalogue but the pipeline skips it (and the fine-tuning step skips it too).

### `configs/fusion.yaml`

```yaml
# Which fusion strategies to actually run. Empty / missing = none run.
fusion_strategies_enabled:
  - mean
  - sum
  - prod
  - max_pool
  - weighted_mean
  - softmax_weighted
  - sigmoid_gated
  - concat
  - pca
  - pca_per_model
  - adaptive_gated

normalize_before_fusion: true

# Equal-dim strategies fuse sources with DIFFERENT native dims, so an
# alignment brings them to a common dimension first (v2 protocol):
#   learned - per-source Linear(D_i -> dim) co-trained with the
#             recommender via BPR (analogous to the projection E).
#   pca     - per-source PCA to dim, fit ONLY on train items.
# The concatenation family (concat/pca/pca_per_model) ignores this
# block and operates on native dims.
alignment:
  method: learned
  dim: 128

strategies:
  mean: {}
  sum: {}
  prod: {}
  max_pool: {}
  # Weighted strategies are FIXED convex combinations (thesis spec);
  # distinct canonical values keep them from collapsing into `mean`.
  weighted_mean:
    w_cnn: [0.3, 0.7]     # weights (0.300, 0.700) and (0.700, 0.300)
  softmax_weighted:
    logits: [[1.0, 0.0]]  # softmax -> (0.731, 0.269)
  sigmoid_gated:
    logits: [[1.0, 0.0]]  # normalised sigmoids -> (0.594, 0.406)
  concat: {}
  pca:
    n_components: [128]
  pca_per_model:
    n_components_per_model: [64]
  adaptive_gated: {}      # the only LEARNED fusion (online, no .npy)
```

### `configs/recommenders.yaml`

```yaml
# Which recommenders to actually run. Empty / missing = none run.
recommenders_enabled: ["bpr", "vbpr", "vnpr", "deepstyle", "acf"]

common: # shared by every recommender
  total_dim: [64, 128]  # shared dimension budget T (parity guard; see below)
  learning_rate: [0.001, 0.01]
  l2_reg: [0.0001]
  epochs: 50
  batch_size: 4096
  early_stopping_patience: 5
  early_stopping_metric: "ndcg@10"
  eval_every_epochs: 10
  eval_sample_size: 2000 # training-time validation user subsample (final eval is always full ranking)

# Per-paper regularisation constants; absent keys fall back to common.l2_reg.
bpr:
  l2_reg_item_pos: 0.0001 # λ_H+ (Rendle et al. 2009)
  l2_reg_item_neg: 0.0001 # λ_H−

vbpr:
  l2_reg_projection: 0.0 # λ_E = 0 (He & McAuley 2016)
  l2_reg_visual_bias: 0.0001 # λ_β

vnpr:
  dropout: 0.0 # embedding dropout (Niu et al. 2018)

avbpr:
  att_hidden: [64, 128]
  l2_reg_projection: 0.0
  l2_reg_visual_bias: 0.0001

# ACF needs per-item component embeddings (*_comp artifacts) and the
# user history. With `acf` in recommenders_enabled the extract step
# auto-enables component extraction (no flag needed); the train step
# fails loud if the *_comp artifacts are still missing.
acf:
  att_hidden: [64, 128]
  max_history: [50] # seeded uniform subsample when |R(u)| > H; null = full history
```

**Dimension parity.** `common.total_dim` is the one dimension budget every recommender draws from, so the central comparison controls capacity (the VBPR baseline protocol: all MF methods use the same total number of factors, VBPR splits it 50/50). Each model's `RecommenderSpec.dim_split` derives its own dimensions: BPR-MF `latent_dim = T`; VBPR/AVBPR `latent_dim = visual_dim = T/2`; DeepStyle `d = T`; ACF `k = T`; VNPR latent `k = T` (its visual user vector lives in the image-feature space by construction). Declaring `latent_dim` / `visual_dim` directly — in `common:`, a model block or an `hp_space` — is refused by the parity guard before training.

Each list becomes a Cartesian dimension; scalars stay constant. Combination counts per recommender:

| Recommender | Combinations | Dimensions                         |
| ----------- | -----------: | ---------------------------------- |
| BPR         |            8 | total_dim × LR × L2                |
| VBPR        |            8 | same (T split 50/50)               |
| VNPR        |            8 | same                               |
| DeepStyle   |            8 | same                               |
| AVBPR       |           16 | + att_hidden                       |
| ACF         |           16 | + att_hidden (+ max_history)       |

### `configs/finetuning.yaml`

```yaml
finetuning:
  epochs_max: 15
  learning_rate: 0.0001
  weight_decay: 0.0001
  batch_size: 128
  patience: 5
  extractors: ["resnet50", "vit_b16", "cvt_13", "coatnet_0", "levit_256"]
  tradesy_transfer_from: "amazon_fashion" # source for datasets without categories
```

### `configs/evaluation.yaml`

```yaml
k_values: [5, 10, 20]
metrics: ["precision", "recall", "f1", "map", "ndcg"]

statistical:
  alpha: 0.05
  correction: "holm" # "holm", "bonferroni" or "none"

  bootstrap:
    enabled: true
    n_iterations: 1000

  friedman:
    enabled: true

  effect_size: true
```

### Runtime sizing knobs

The framework reads two sources for DataLoader sizing and VNPR chunking:

1. The `dataloader:` block in `configs/default.yaml` (commented out by default). Uncomment any field to pin it.
2. The autotune in `src/utils/dataloader.py` (CPU + cgroup memory for DataLoader, GPU VRAM for VNPR chunk).

When the YAML pins a value, it wins; otherwise the autotune picks the tier value. The resolved values plus any active YAML overrides are recorded under `manifest['dataloader_autotune']`. There are no environment variables for these knobs, the YAML is the only override path so reruns stay reproducible from `git checkout` alone.

The fine-tuning training batch size lives in `configs/finetuning.yaml -> finetuning.batch_size`; the frozen-extract batch size lives in `configs/extractors.yaml -> batch_size`. Both are already YAML-only.

| Knob                                  | Source                                                                 |
| ------------------------------------- | ---------------------------------------------------------------------- |
| `num_workers` / `prefetch_factor`     | `configs/default.yaml -> dataloader.*` or autotune (2 / 4 / 12 by memory tier) |
| Re-extract batch size (inside finetune and evaluate_finetuning)  | `configs/default.yaml -> dataloader.batch_size` or autotune (32 / 128 / 256 by memory tier) |
| Fine-tuning training batch size       | `configs/finetuning.yaml -> finetuning.batch_size`                     |
| Frozen-extract batch size             | `configs/extractors.yaml -> batch_size`                                |
| VNPR `(user, item)` pairs per forward | Autotune from GPU VRAM (500_000 / 2_000_000 / 5_000_000 by tier)       |

---

## 5. Datasets

### Fashion starter pack

The repository ships with the four DVBPR datasets pre-registered (Kang et al., ICDM 2017) for out-of-the-box reproduction. They are bundled as a convenient starting point, not as the framework's scope — the framework is domain-agnostic (see [Beyond fashion](#beyond-fashion-domain-extensibility)).

| Dataset        |  Users |   Items | Interactions | Categories |
| -------------- | -----: | ------: | -----------: | :--------: |
| Amazon Fashion | 45,184 | 166,270 |      267,635 |    yes     |
| Amazon Women   | 97,678 | 347,591 |      632,321 |    yes     |
| Amazon Men     | 34,244 | 110,636 |      186,339 |    yes     |
| Tradesy        | 33,864 | 326,393 |      594,680 |     no     |

Tradesy has no category labels; its fine-tuning step transfers the weights produced for Amazon Fashion (see `finetuning.tradesy_transfer_from`).

### On-disk schema

Once a dataset is materialised, it lives in this canonical layout, every step from 03 onward reads only this:

```
data/raw/<name>/                          ← provider-specific raw data
data/raw/<name>/images/<item_idx>.jpg     ← one image per item
data/processed/<name>/train.csv           ← columns: user_idx,item_idx
data/processed/<name>/val.csv             ← idem (1 row per user, leave-one-out)
data/processed/<name>/test.csv            ← idem
data/processed/<name>/user2idx.json       ← {"<external user id>": <user_idx>}
data/processed/<name>/item2idx.json       ← idem
```

**Schema invariants** (enforced by the helpers):

- `user_idx` ranges over `[0, n_users)` _contiguously_.
- `item_idx` ranges over `[0, n_items)` _contiguously_.
- CSVs have exactly two columns: `user_idx` and `item_idx` (header included).
- `val` / `test` follow leave-one-out (1 held-out item per user).
- Image files use the integer `item_idx` as the stem (`.jpg`/`.jpeg`/`.png`/`.webp`).

### Beyond fashion: domain extensibility

The framework is agnostic to the recommendation domain. The fashion starter pack is one configured instance; any domain with `(user, item, image)` interactions can be plugged in. Three zero-Python paths exist:

- **Drop a directory** under `plugins/datasets/<name>/` with `interactions.csv` (`user_id, item_id, image_path`) plus an `images/` folder. The auto-discovery (`src/data/auto_register.py`) registers it at startup and the pipeline picks it up.
- **Use `CSVDatasetProvider`** (`src/data/example_csv.py`) programmatically when you need finer control over splits, categories, or image sources.
- **Use `SyntheticDatasetProvider`** (`src/data/synthetic.py`) for plumbing checks and tests — it generates a tiny, deterministic dataset entirely in-process with no external download. Used by the unit tests (`tests/test_synthetic_dataset.py`).

The full contract for custom providers lives in `src/data/base.py`.

### Adding your own dataset

The simplest path is **zero Python**: drop a directory under `plugins/datasets/<name>/` and the pipeline auto-registers it. Then add `"<name>"` to the `datasets:` list in `configs/default.yaml`.

#### Path 1, files already on disk

```
plugins/datasets/my_dataset/
├── interactions.csv         # columns: user_id, item_id, image_path
└── images/                  # one image per item (any extension)
    ├── it_001.jpg
    ├── it_002.jpg
    └── ...
```

- `interactions.csv` is the user-item interaction log, one row per `(user, item)`.
- The default split is leave-one-out: 1 test + 1 val per user, the rest in train.
- Users with fewer than 3 interactions are dropped.

#### Path 2, describe URLs to download

```
plugins/datasets/my_dataset/
└── source.yaml
```

```yaml
# plugins/datasets/my_dataset/source.yaml
interactions:
  url: https://example.com/interactions.csv
  # or:
  # path: /absolute/path/to/interactions.csv

images:
  url: https://example.com/images.tar.gz # tar/zip extracted in place
  # or:
  # path: /absolute/path/to/images-folder/
```

The first run downloads / copies into `data/raw/<name>/` and behaves identically to Path 1 from there.

#### Path 3, full programmatic control

```python
from src.data.base import (
    DatasetProvider, register_dataset_provider, write_processed_splits,
)

class MovieLensVisProvider(DatasetProvider):
    name = "movielens_vis"
    def download(self) -> None: ...
    def save_processed(self, processed_dir) -> None: ...
    def extract_images(self, image_dir) -> None: ...
    def load_categories(self) -> dict[int, int] | None: ...   # optional
    def num_categories(self) -> int: ...                       # optional

register_dataset_provider("movielens_vis", MovieLensVisProvider)
```

See `src/data/base.py` for the full contract and `src/data/example_csv.py` for a worked-example provider.

#### Running with your dataset only (skipping DVBPR)

Edit the `datasets:` list in `configs/default.yaml` so it contains only the names you want. The pipeline runs whatever is on the list, nothing more, nothing less:

```yaml
# configs/default.yaml
datasets:
  - "my_dataset" # only this one runs
```

Any DVBPR files already on disk under `data/raw/` and `data/processed/` are left untouched.

#### Sanity-check the layout before launching a long run

```python
from src.data.base import validate_layout

problems = validate_layout("my_dataset")
print(problems or "OK")
```

A non-empty list reports schema violations.

---

## 6. Extractors

| #   | Extractor           | Architecture                                 | Role       | Native Dim |
| --- | ------------------- | -------------------------------------------- | ---------- | ---------: |
| 1   | **ResNet50**        | CNN                                          | Primary    |       2048 |
| 2   | **ViT-B/16**        | Transformer                                  | Primary    |        768 |
| 3   | **CvT-13**          | Convolutional token embeddings + Transformer | Hybrid     |        384 |
| 4   | **CoAtNet-0**       | Depthwise conv + self-attention              | Hybrid     |        768 |
| 5   | **LeViT-256**       | Conv → Transformer (sequential)              | Hybrid     |        512 |
| 6   | **CLIP ViT-B/32**   | ViT (LAION-2B pretrained)                    | Foundation |        512 |
| 7   | **DINOv2 ViT-B/14** | ViT (self-supervised)                        | Foundation |        768 |
| 8   | **ConvNeXt-Base**   | Modernised CNN (ImageNet-22k → 1k)           | Hybrid     |       1024 |

Extraction saves each backbone's **native** pooled feature with the backbone's canonical preprocessing, resolved from the library that ships the weights. There is no shared projection head at extraction time: the learned projection `E` inside each recommender maps the native dim to the model's visual dim (derived from the shared budget `common.total_dim`), trained jointly by the BPR loss. Every artifact carries a `.meta.json` sidecar (backbone, native dim, extraction point, exact weights id, transform recipe) that the loader cross-checks on read. See `docs/protocol.md` for every methodological declaration.

---

## 7. Fusion strategies

Applied to the primary pair declared in `fusion_extractors` (default: ResNet50 + ViT-B/16):

| #   | Strategy           | Notes                                         |
| --- | ------------------ | --------------------------------------------- |
| 1   | Mean               | Element-wise average                          |
| 2   | Sum                | Element-wise sum                              |
| 3   | Product            | Hadamard product                              |
| 4   | Max Pool           | Element-wise maximum                          |
| 5   | Weighted Mean      | Fixed configurable weights (canonical `w_cnn = 0.7`) |
| 6   | Attention Weighted | Softmax over fixed configurable logits (canonical `[1.0, 0.0]`) |
| 7   | Gated              | Normalised sigmoids over fixed configurable logits (canonical `[1.0, 0.0]`) |
| 8   | Concat             | Feature concatenation                         |
| 9   | PCA                | PCA on concatenated features                  |
| 10  | PCA per Model      | Per-model PCA then concat                     |
| 11  | Adaptive Gated     | Per-item gate MLP, learned online (co-trained via BPR) |

Strategies 1–10 are offline convex/algebraic combinations with no trainable fusion parameters — 5–7 differ only in how their fixed configurable values map to convex weights. Strategy 11 (`adaptive_gated`) is the only learned fusion.

Optional L2 normalisation before fusion (`normalize_before_fusion` in `configs/fusion.yaml`).

Because sources keep their native dims in v2, the equal-dim family (1-7 plus 11, `adaptive_gated`) first brings them to a common dimension via the configured `alignment`: `learned` (per-source `Linear(D_i → D)` co-trained with the recommender by BPR) or `pca` (offline per-source PCA, fit only on items with a training interaction). The concatenation family (8-10) operates directly on native dims.

---

## 8. Recommender models

| Model         | Score                                                        | Visual features     |
| ------------- | ------------------------------------------------------------ | ------------------- |
| **BPR**       | γ_u^T γ_i (BPR-MF, no item bias; Rendle et al., UAI 2009)    | None (CF baseline)  |
| **VBPR**      | γ_u^T γ_i + θ_u^T (E f_i) + β_i + β'^T f_i (Eq. 4, He & McAuley, AAAI 2016; α, β_u cancel pairwise) | Linear projection E, λ_E = 0 |
| **VNPR**      | ½ Σ_branches ReLU(w^T [p_u ∘ q_i, v_u ∘ f_i] + b), mirrored item tables W_i / W_i' (Niu et al., WSDM 2018) | User visual vector v_u in the image-feature space |
| **DeepStyle** | p_u^T ((E f_i − l_cat(i)) + q_i), one user vector, no item bias (Eqs. 2–3, Liu et al., SIGIR 2017) | Style = linear E minus learned category embedding |
| **AVBPR**     | VBPR with θ_i = (E f_i) ⊙ softmax(g(E f_i)) — own attentional adaptation, not a published model | Attention-weighted  |
| **ACF**       | (γ_u + Σ_{l∈R(u)} α(u,l) p_l)^T γ_j (Eq. 6, Chen et al., SIGIR 2017); visual enters the item-level attention only | Component + history attention |

All trained with BPR loss, Adam optimiser, mixed-precision (FP16 via `torch.amp`), and early stopping on validation NDCG@10.

**ACF** (Attentive Collaborative Filtering) is the only recommender that consumes per-item *component* embeddings — the pre-pool spatial cells / patch tokens of shape `(n_items, M, native_dim)` written as `<extractor>_comp.npy` (auto-extracted whenever a component-consuming recommender such as ACF is enabled; `extract_components: true` forces them without one). All eight extractors expose components (`M`: ResNet-50 / ConvNeXt / CoAtNet / CLIP = 49, ViT-B/16 / CvT = 196, LeViT = 16, DINOv2 = 256). ACF's two attention levels weight (a) an item's `M` components and (b) the items in the user's training history (built train-only, so validation/test never leak into the profile). Faithful to the paper, the sampled BPR positive stays in the history at train time. See `src/recommenders/acf.py`.

---

## 9. Extending the framework

Every component type is pluggable. Drop a Python module under the matching `plugins/` directory and the pipeline auto-registers it at startup, no edits to existing files.

> **Read the full guide:** [`docs/extending.md`](docs/extending.md) walks through every plugin type with a runnable example and a contract checklist. Each `plugins/<domain>/` directory ships a ready-to-copy `_example.py` (or `_example/` for datasets), copy it under any name without the leading underscore to activate. Run `pytest -q` before submitting to verify your plugin still satisfies the registration contract.

| Component       | Drop module under          | Registry helper                                  | Catalogue         |
| --------------- | -------------------------- | ------------------------------------------------ | ----------------- |
| Dataset         | `plugins/datasets/<name>/` | `register_dataset_provider` (or the auto-layout) | `DatasetProvider` |
| Extractor       | `plugins/extractors/`      | `register_extractor`                             | `BaseExtractor`   |
| Fusion strategy | `plugins/fusions/`         | `register_fusion_strategy`                       | `FusionSpec`      |
| Recommender     | `plugins/recommenders/`    | `register_recommender`                           | `RecommenderSpec` |

After registering, add the new name to the matching toggle list (`extractors_enabled`, `fusion_strategies_enabled`, `recommenders_enabled`) and run the pipeline.

Plugins are zero-edit by design: the four package `__init__.py` files never need to be touched. The auto-discovery scans the `plugins/` directories at import time, skipping anything whose name starts with `_` or `.` (this is how the `_example.py` scaffolds stay inert until you copy them). See [`docs/extending.md`](docs/extending.md) for the full contract of every plugin type.

---

## 10. Evaluation

### Protocol

- **Ranking** (default = `full_ranking`): score every item, mask the user's training and validation history, compute metrics on the test set. A `sampled` alternative is selectable via `evaluation.protocol` in `configs/evaluation.yaml` — see [Ranking protocol](#ranking-protocol-full-ranking-vs-sampled) below.
- **Metrics**: precision, recall, F1, MAP, NDCG @ {5, 10, 20}.
- **Split**: validation drives early stopping and hyperparameter selection; test is reported as the final metric.
- **Per-user output**: step 06 saves per-user metrics so step 07 can run paired tests without re-running training.

### Ranking protocol: full-ranking vs sampled

Two protocols are available, selected by `evaluation.protocol`:

| | `full_ranking` (default, recommended) | `sampled` |
|---|---|---|
| Pool per user | every catalogue item, minus seen | `n_negatives` random unseen items + positives |
| Cost | O(n_items) per user | O(n_negatives) per user — much cheaper |
| Preserves model ordering vs ground-truth ranking? | Yes (it *is* the ground-truth ranking) | **No** — Krichene & Rendle (KDD 2020) showed sampled metrics can flip relative model orderings |
| When to use | Primary thesis / paper number | Only for comparability with prior work that adopted the same protocol |

CLI override:

```bash
python main.py --all --eval-protocol sampled
```

YAML knobs (`configs/evaluation.yaml`):

```yaml
evaluation:
  protocol: "full_ranking"      # or "sampled"
  n_negatives: 100              # pool size when protocol="sampled"
  negative_sampling_seed: 42    # deterministic, identical across models
```

The `negative_sampling_seed` is intentional: paired statistical tests (Wilcoxon) require that two models being compared see the **same** sampled pool per user, otherwise the per-user differences are confounded by sampling noise. Keep it constant across model runs you intend to compare.

### Statistical methods

Step 07 produces three granular CSVs per dataset and metric, plus three long-format consolidated CSVs spanning every dataset/metric/k combination:

| Output                            | Method                                           |
| --------------------------------- | ------------------------------------------------ |
| `{dataset}_summary_{metric}.csv`  | Bootstrap CI on the per-config mean (descriptive) |
| `{dataset}_friedman_{metric}.csv` | Friedman omnibus test, one row per comparison-family instance |
| `{dataset}_pairwise_{metric}.csv` | Wilcoxon signed-rank + within-family correction + Cliff's delta + paired-difference bootstrap CI |
| `evaluation_aggregated.csv`       | Long-format mean per cell (all datasets/metrics/k) |
| `bootstrap_ci.csv`                | Long-format bootstrap CI rows                     |
| `statistical_tests.csv`           | Long-format Friedman + Wilcoxon rows              |

The three long-format files are auto-generated at the end of `statistical.run()` via `src.reporting.write_consolidated` — they collapse the ~160 granular files into one row per observation with explicit `dataset` / `recommender` / `extractor` / `fusion` / `condition` / `metric` / `k` columns, making thesis-time analysis a single `pandas.read_csv` + `df.query`.

#### Comparison families

The correction and the Friedman omnibus are applied **within the family of comparisons a research question defines** (`src/evaluation/comparison_families.py`), never over the Cartesian product of every config (all-pairs Holm over ~77 configs would run with `m ≈ 2900` and reject everything artificially). Each family varies exactly one experimental dimension — `backbone_within_model`, `model_within_backbone`, `fusion_within_model`, `frozen_vs_finetuned` — and every result row carries `family`, `group` and `n_comparisons_in_family` so the correction is auditable. `all_pairs` is available as an exploratory option only (`statistical.families`).

#### Primary metrics under leave-one-out

With exactly one relevant item per user, only two independent signals exist: hit-or-not (`recall@k` ≡ HitRate@k) and hit rank (`ndcg@k`). `precision@k = recall@k / k` and `map@k = 1/rank` are deterministic transforms of them; they stay in the raw evaluation CSVs but are only analysed with `include_derived_metrics: true`, and must not be read as independent evidence.

#### Multiple comparison corrections

| Setting                  | Method                    |
| ------------------------ | ------------------------- |
| `correction: holm`       | Holm-Bonferroni (default) |
| `correction: bonferroni` | Vanilla Bonferroni        |
| `correction: none`       | Raw p-values              |

#### Effect sizes and paired-difference CI

- **Paired Cliff's delta (primary)**, `(wins − losses) / n` over per-user differences, in `[-1, 1]`, reported together with the win / loss / tie triplet (`n_wins`, `n_losses`, `n_ties` and their fractions). No magnitude label: the Romano et al. cut-offs were calibrated for the between-groups delta, and under leave-one-out the ties that dominate the denominator would make every comparison read "negligible". Read the effect as "A beats B in X% of users, loses in Y%, ties in Z%" alongside `diff_mean` and its CI. Consistent with Wilcoxon+pratt on 0/1-heavy per-user LOO metrics.
- **Cohen's d (diagnostic only, off by default)** — parametric; on zero-dominated paired differences the `std` shrinks and `d` inflates without a sensible interpretation. Enable with `include_cohens_d: true` if needed.
- **Paired-difference bootstrap CI** (`diff_mean`, `diff_ci_lower/upper`) on every pairwise row, resampling users: this CI must agree with the RAW (uncorrected) Wilcoxon p-value — the `significant` column is Holm-corrected, so a corrected non-significant verdict can coexist with a CI excluding zero. Per-config CIs in the summary table are descriptive — because the inference is paired, overlapping individual CIs do **not** imply absence of a significant difference.

Each method is toggled individually in `configs/evaluation.yaml` -> `statistical:`.

---

## 11. Performance optimisations

### Recommender training and evaluation

| Optimisation                    | Where                                        | Effect                                                                                                                           |
| ------------------------------- | -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Vectorised `VNPR.predict_batch` | `src/recommenders/vnpr.py`                   | Replaces a per-user Python loop with a chunked MLP forward over `(b × N)` pairs. Chunk size is picked from the GPU's VRAM at startup. |
| Item-feature cache for VNPR     | `src/recommenders/vnpr.py`                   | Caches `[q_i, v_i]` for the full catalogue during evaluation, reused across `predict_batch` calls.                               |
| GPU top-K in the Evaluator      | `src/evaluation/protocol.py`                 | Masks training items with `index_fill_` on GPU and runs `torch.topk` directly; only `(B, K)` indices cross to CPU.               |
| Per-user train-mask cache       | `src/evaluation/protocol.py`                 | Builds per-user `LongTensor`s of training-item indices once and reuses across epochs.                                            |
| Combined pos+neg forward        | `src/recommenders/{vbpr,avbpr,deepstyle}.py` | Single `(2B,)` batched forward through the visual / style / attention path instead of two separate B-sized passes.               |
| Per-epoch GPU loss accumulation | `src/utils/training.py`                      | Accumulates loss as a GPU tensor; calls `.item()` once per epoch instead of per batch.                                           |
| Deterministic per-job seed      | `src/utils/training.py`                      | SHA-256 of `(dataset, model, embedding, hyperparams)` produces a reproducible seed per job.                                      |
| AMP compatibility shim          | `src/utils/amp_compat.py`                    | Routes through `torch.amp.{GradScaler,autocast}('cuda', ...)` on PyTorch 2.3+ and falls back to `torch.cuda.amp` on PyTorch 2.1. |

### Fine-tuning

| Optimisation                                                | Effect                                                                                                                         |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Single directory scan in `CategoryDataset` / `ImageDataset` | One `os.listdir()` instead of N `path.exists()` calls.                                                                         |
| Persistent DataLoader workers                               | `persistent_workers=True` removes per-epoch fork/warm-up.                                                                      |
| Tunable batch / workers / prefetch                          | YAML-only: `configs/finetuning.yaml -> finetuning.batch_size`; workers/prefetch via `configs/default.yaml -> dataloader.*` or autotune. |
| Per-job try/except                                          | Failures are isolated per `(extractor × dataset)` so the queue continues after an OOM, layer-name mismatch, or HF Hub timeout. |

---

### Need a process supervisor?

The previous `scripts/watchdog.sh` supervisor (RunPod-era operational scaffolding) has been removed in favour of platform-native supervisors. For long runs on cloud pods, use the host's own facilities (systemd unit, Kubernetes liveness probe, `tini --restart-on-exit`, Docker `restart: unless-stopped`). The pipeline is idempotent: a hard kill followed by `python main.py` resumes from the last checkpoint regardless of which supervisor restarted it.

---

## 12. Reproducibility

Every grid-search run uses a deterministic per-job seed derived from the job identity (`dataset`, `model`, `embedding`, `hyperparams`) XOR-ed with the global base seed (42 by default; see `_derive_job_seed` in `src/utils/training.py`). CUDA matmul is non-deterministic by default, so numerical drift between runs is about 1e-5 in FP32.

### Run manifest

Every `main.py` invocation writes a manifest to `results/runs/<run_id>/manifest.json` (gitignored, since it is an execution artefact). The manifest captures:

- `git.sha`, `git.dirty`, `git.branch`: exact code state.
- `seed`: global RNG seed.
- `hardware`: GPU name, VRAM, CUDA version, RAM, CPU count.
- `device`: requested value (`auto` / `cuda` / `cpu`) and the value actually used.
- `dataloader_autotune`: tier picked from the cgroup memory budget plus the resolved `num_workers`, `prefetch_factor`, `batch_size`.
- `package_versions`: versions of `torch`, `transformers`, `numpy`, and the other pinned dependencies.
- `config_snapshot`: every YAML merged into a single dict.
- `steps`: list of `{name, started_at, duration_seconds, telemetry}`, one entry per step. Condition-aware, so `fuse (frozen)` and `fuse (finetuned)` are separate entries. Written when the run closes; the sidecar `results/runs/<run_id>/steps.json` carries the same array flushed after every step, so an interrupted run still documents the steps it did run.
- `telemetry`: run-level rollup — total GPU energy, total FLOPs, bytes downloaded, plus the probe backends and FLOP calibrations that produced them.
- `started_at`, `finished_at`, `duration_seconds`, `exit_status`.

Each step's `telemetry` block answers how fast it ran and what it cost, not just how long it took: min / max / mean of GPU utilisation, power and memory, CPU usage and the relevant throughput series (`network_mb_per_s` for `download`, `flops_per_s` and `items_per_s` for the model steps), plus `energy_joules` integrated from sampled GPU power. Sampling is on by default at 1 Hz and configurable under `telemetry:` in `configs/default.yaml`.

A sidecar `results/runs/<run_id>/step_timings.json` records per-cell wall-time and the same telemetry block, scoped to the cell: one entry per `dataset` for `download` — labelled with the dataset's weight on disk (`size_mb`) and how much of it came over the network this run (`downloaded_mb`) — one entry per `(dataset, extractor, dim)` for `extract`, per `(dataset, extractor)` for `finetune` and `evaluate_finetuning`, per `(dataset, model_key)` for `evaluate`. Each entry carries a `labels` dict with the cell identity for direct use in downstream notebooks.

To cite specific results, archive the manifest alongside the publication (for example as a Zenodo release with a DOI). Anyone can then reproduce the run with `git checkout <sha>` plus the recorded package versions.

Manifests where `exit_status` is not `"ok"` document an aborted run and should not be cited as authoritative results.

See [`docs/observability.md`](docs/observability.md) for the full schema and recipes for plotting the timings and telemetry in `pandas`.

---

## 13. Project structure

```
prism-vrec/
├── main.py                       # Pipeline entrypoint
│
├── configs/
│   ├── default.yaml              # Orchestration: seed, device, datasets, paths, pipeline:
│   ├── extractors.yaml           # 8 extractors + projection dims + fusion pair
│   ├── fusion.yaml               # 11 fusion strategies + normalisation
│   ├── recommenders.yaml         # Grid-search hyperparameters (6 models)
│   ├── evaluation.yaml           # Metrics, k-values, statistical tests
│   ├── finetuning.yaml           # Fine-tuning hyperparameters
│
├── src/
│   ├── data/
│   │   ├── base.py               # DatasetProvider ABC + registry + helpers
│   │   ├── dvbpr.py              # DVBPR provider (4 fashion datasets, auto-derives categories)
│   │   ├── categories.py         # McAuley-taxonomy → categories.csv helper
│   │   ├── example_csv.py        # Worked-example provider for CSV+images datasets
│   │   ├── synthetic.py          # In-process synthetic dataset (backs the unit tests)
│   │   └── auto_register.py      # Auto-discovery for plugins/datasets/<name>/
│   │
│   ├── extractors/
│   │   ├── base.py               # BaseExtractor ABC (FP16 inference, checkpointing)
│   │   ├── resnet.py, vit.py, cvt.py, coatnet.py, levit.py, clip.py, dinov2.py, convnext.py
│   │
│   ├── fusions/
│   │   └── strategies.py         # 11 fusion strategies + factory
│   │
│   ├── recommenders/
│   │   ├── base.py               # BaseRecommender (BPR loss, L2 reg)
│   │   ├── bpr.py, vbpr.py, vnpr.py, deepstyle.py, avbpr.py, acf.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py            # Precision, Recall, F1, MAP, NDCG @ K
│   │   ├── protocol.py           # Full-ranking evaluator (GPU top-k, train-mask cache)
│   │   └── statistical.py        # Bootstrap CI, Friedman, Wilcoxon, effect sizes
│   │
│   ├── reporting/
│   │   ├── long_format.py        # Reshape granular CSVs into tidy long-format
│   │   └── consolidate.py        # Orchestrates write of the 3 long-format files
│   │
│   ├── finetuning/
│   │   ├── dataset.py            # Category-classification dataset
│   │   ├── trainer.py            # Generic fine-tuner (unfreezing, AMP, early stopping)
│   │   ├── checkpoint.py         # FT checkpoint format (backbone + head + metadata)
│   │   └── evaluator.py          # Post-hoc top-K / F1 / confusion matrix on FT val split
│   │
│   ├── steps/                    # Pipeline steps imported by main.py
│   │   ├── download.py
│   │   ├── preprocess.py
│   │   ├── extract.py
│   │   ├── finetune.py
│   │   ├── evaluate_finetuning.py
│   │   ├── fuse.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── statistical.py
│   │   └── export_best.py
│   │
│   └── utils/
│       ├── amp_compat.py         # AMP shim (PyTorch 2.1 / 2.3+)
│       ├── config.py             # YAML config loader (singleton)
│       ├── logging.py            # Console + file logger factory
│       ├── seed.py               # Global RNG seeding
│       ├── checkpoint.py         # Atomic checkpoint manager
│       ├── training.py           # Single training run
│       └── parallel.py           # Parallel training orchestrator
│
├── plugins/                      # User-supplied extension points (auto-discovered)
│   ├── extractors/_example.py    # Drop register_extractor() modules here
│   ├── fusions/_example.py       # Drop register_fusion_strategy() modules here
│   ├── recommenders/_example.py  # Drop register_recommender() modules here
│   └── datasets/_example/        # Dataset scaffold (interactions.csv + images/)
│                                 # Names starting with _ are never auto-registered
├── docs/
│   └── extending.md              # Full guide for plugin authors
│
├── tests/                        # pytest contract suite for plugins + FT checkpoint
│
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## 14. Hardware requirements

The same image runs on all three tiers below. `docker compose up -d --build` is the only command the user needs, and the framework auto-detects the host (GPU presence, CPU count, cgroup memory, VRAM) at startup. The resolved values are recorded under `manifest['device']` and `manifest['dataloader_autotune']`.

| Tier             | CPU      | RAM     | GPU                          | Disk   | What it is good for                                                                  |
| ---------------- | -------- | ------- | ---------------------------- | ------ | ------------------------------------------------------------------------------------ |
| CPU only         | 4 cores  | 8 GB    | none                         | 50 GB  | Data-only steps (download, preprocess), sanity checks, plot generation               |
| Mid              | 8 cores  | 32 GB   | 8-16 GB VRAM                 | 100 GB | End-to-end pipeline on one DVBPR dataset; longer wall-time on the full four          |
| Full             | 16 cores | 64 GB   | 24 GB+ VRAM (RTX 4090, A100) | 200 GB | Full four DVBPR datasets, both batteries, all extractors and fusions, in 1-3 days    |


---

## 15. Citation

### Citing the framework (software)

If you use this framework in your work, please cite the software:

```bibtex
@software{couto_prism_vrec,
  title   = {prism-vrec: A reproducible framework for evaluating visual feature extractors in recommender systems},
  author  = {Couto, Lucas Silva and Domingues, Marcos Aurelio},
  year    = {2026},
  version = {2.11.0},
  doi     = {10.5281/zenodo.20357510},
  url     = {https://doi.org/10.5281/zenodo.20357510}
}
```

Every release is archived on Zenodo with its own version DOI (v2.0.0: [10.5281/zenodo.21325967](https://doi.org/10.5281/zenodo.21325967)); the concept DOI [10.5281/zenodo.20357510](https://doi.org/10.5281/zenodo.20357510) always resolves to the latest archived version — cite it unless you need to pin an exact version. A companion tool paper is planned and this section will be updated when it is available. See `CITATION.cff` for the canonical software citation metadata.

### Citing a reported use of the framework

The author's M.Sc. dissertation is one reported use of this framework:

```bibtex
@mastersthesis{couto2026hybrid,
  title   = {Hybrid Visual Feature Extraction for Fashion Recommendation Based on Deep Learning},
  author  = {Couto, Lucas Silva},
  advisor = {Domingues, Marcos Aurelio},
  year    = {2026},
  school  = {Universidade Estadual de Maringá},
  type    = {M.Sc. Dissertation},
  address = {Maringá, PR, Brazil},
  note    = {Graduate Program in Computer Science}
}
```

Rendered in IEEE format:

> [1] L. S. Couto, "Hybrid Visual Feature Extraction for Fashion Recommendation Based on Deep Learning," M.Sc. dissertation, Graduate Program in Computer Science, Universidade Estadual de Maringá, Maringá, PR, Brazil, 2026.

---

## 16. License

The framework code is released under the [MIT License](LICENSE), Copyright (c) 2026 Lucas Silva Couto. Datasets bundled or referenced by the fashion starter pack remain subject to their original licenses (see [DVBPR](https://github.com/kang205/DVBPR)); review and comply with each dataset's terms before redistribution.
