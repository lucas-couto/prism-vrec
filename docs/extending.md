# Extending the framework

This guide is the single source of truth for contributors who want to
plug a new component into the pipeline. Five plugin types are
supported — pick the one that matches what you are adding and follow
the recipe.

| Want to add… | Drop your file at | Register via | Subclass / API |
| --- | --- | --- | --- |
| Visual extractor | `plugins/extractors/<name>.py` | `register_extractor` | `BaseExtractor` |
| Fusion strategy | `plugins/fusions/<name>.py` | `register_fusion_strategy` | callable |
| Recommender | `plugins/recommenders/<name>.py` | `register_recommender` | `BaseRecommender` |
| Dataset | `plugins/datasets/<name>/` | (auto, from layout) | `DatasetProvider` *(optional)* |
| Pipeline step | `src/steps/<name>.py` + `main.py:STEP_ORDER` | (manual) | function `run()` |

> **One rule to remember:** never edit the package `__init__.py` files
> of the four plugin domains. The auto-discovery scans `plugins/`
> directories at import time, so dropping a file is enough — no other
> file in the framework needs to change.

Each `plugins/<domain>/` directory ships with a runnable
`_example.py` (or `_example/` directory for datasets). The leading
underscore is what tells the auto-discovery to skip the file; copy
it under any name **without** the underscore to activate. The same
templates serve as worked references throughout this guide.

---

## 1. Add a visual extractor

Use this when you want to compare an existing or novel image-encoding
backbone (CNN, ViT, hybrid, foundation model) against the ones already
shipped.

### 1.1 Contract

A custom extractor is any subclass of
[`BaseExtractor`](../src/extractors/base.py) that implements two
methods:

- `_build_model() -> nn.Module` — returns the trainable backbone. Its
  last submodule **must** be named `projection` and end in a layer whose
  `in_features` matches the backbone's pooled-feature size. The
  fine-tuner replaces this layer with a classification head; if the
  contract is broken `FineTuner.__init__` raises.
- `_build_transform()` — returns the image transform pipeline used both
  at extraction and at fine-tuning time.

Optional:

- `unfreeze_prefixes: list[str]` — module-name prefixes that stay
  trainable during fine-tuning. Defaults to the empty list (only the
  classification head is trained — fine equivalent of "linear probe").

### 1.2 Minimal example

Copy [`plugins/extractors/_example.py`](../plugins/extractors/_example.py)
to a new file in the same directory (any name **without** a leading
underscore) and edit:

```python
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

from src.extractors.base import BaseExtractor
from src.extractors.registry import register_extractor


class _ResNet18Backbone(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.projection = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.projection(x)


class ResNet18Extractor(BaseExtractor):
    # Unfreeze the last block during fine-tuning.
    unfreeze_prefixes = ["features.7"]

    def __init__(self, device: str = "cuda", output_dim: int = 128) -> None:
        super().__init__(device=device, output_dim=output_dim)
        self.model = _ResNet18Backbone(output_dim).to(self.device).eval()
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()

    def _build_model(self):
        return self.model

    def _build_transform(self):
        return self.transform


register_extractor("my_resnet18", ResNet18Extractor)
```

### 1.3 Wire it into the run

Add the name to `configs/extractors.yaml`:

```yaml
extractors_enabled:
  - resnet50
  - my_resnet18      # <-- new
```

That is the only edit. The pipeline picks the new extractor up at
the next run; `extract`, `finetune` and `evaluate_finetuning`
automatically include it.

### 1.4 Expose component features (optional — for ACF)

By default an extractor emits one pooled `(N, output_dim)` vector per
item. Models with **component-level attention** (e.g. ACF) instead need
the *pre-pool* representation: the spatial feature-map cells (CNNs) or
patch tokens (ViTs) of shape `(N, M, output_dim)`. To make a backbone
expose them:

- Set the class attribute `supports_components = True`.
- Override `_forward_components(images) -> (B, M, output_dim)`, running
  the pre-pool tensor through the **same** `projection` as the pooled
  path so components live in the same `output_dim` space (see
  `src/extractors/resnet.py` for the conv5 → `M=49` reference).

The base class already provides `extract_components_batch` and
`save_components`; the `extract` step writes
`<extractor>_D<dim>_comp.npy` (3-D) **only** when
`extract_components: true` is set in `configs/default.yaml`. The pooled
path is unchanged and byte-identical when the flag is off. All eight
built-in extractors implement this (`M`: ResNet-50 / ConvNeXt / CoAtNet
/ CLIP = 49, ViT-B/16 / CvT = 196, LeViT = 16, DINOv2 = 256).

---

## 2. Add a fusion strategy

Use this when you want to compare a different recipe for combining
multiple visual embeddings into one representation.

### 2.1 Contract

A fusion strategy is a plain callable:

```python
fn(embeddings: list[np.ndarray], normalize: bool = True, **kwargs) -> np.ndarray
```

It receives a list of `(N, dm)` matrices (one per extractor in
`fusion_extractors`) and returns a single `(N, Dh)` matrix. The
``register_fusion_strategy`` helper takes two metadata flags:

- `equal_dim_required` — `True` when every input matrix must share the
  same embedding dimension. The orchestrator skips invalid combinations.
- `expand_grid(cfg) -> list[(suffix, kwargs)]` — turns the strategy's
  YAML block (under `configs/fusion.yaml`) into a list of concrete
  tasks. The default produces a single task with no kwargs and an
  empty filename suffix; override it when your strategy has its own
  hyperparameters (see `weighted_mean` in
  `src/fusions/strategies.py`).

### 2.2 Minimal example

Copy [`plugins/fusions/_example.py`](../plugins/fusions/_example.py)
to a new file in the same directory (any name **without** a leading
underscore) and edit:

```python
import numpy as np

from src.fusions.registry import register_fusion_strategy


def my_max_pool(embeddings, normalize=True, **kwargs):
    """Element-wise maximum across the M input embeddings."""
    stacked = np.stack(embeddings, axis=0)        # (M, N, D)
    fused = stacked.max(axis=0)                   # (N, D)
    if normalize:
        norms = np.linalg.norm(fused, axis=1, keepdims=True) + 1e-12
        fused = fused / norms
    return fused


register_fusion_strategy(
    "my_max_pool",
    my_max_pool,
    equal_dim_required=True,
)
```

### 2.3 Wire it into the run

Add the name to `configs/fusion.yaml -> fusion_strategies_enabled`.

---

## 3. Add a recommender

Use this when you want to evaluate a new visual-aware (or visual-free)
recommender on the same pre-computed embeddings as the built-ins.

### 3.1 Contract

A custom recommender is any subclass of
[`BaseRecommender`](../src/recommenders/base.py) that implements:

- `forward(user_ids, pos_item_ids, neg_item_ids) -> (score_pos, score_neg)`
- `predict(user_id, item_ids) -> scores`

Plus the metadata supplied at registration:

| Metadata field | What it controls |
| --- | --- |
| `priority` | Schedule order (lower = earlier). Cheap models first. |
| `requires_visual` | `False` for plain BPR; runs only in the `frozen` condition with `embedding_name="none"`. |
| `uses_visual_dim` | Adds `common.visual_dim` to the hyperparameter grid. |
| `extra_hyperparam_keys` | Tuple of keys read from `configs/recommenders.yaml -> <name>:`. Each value may be a scalar or a list (becomes a Cartesian dimension in the grid). |
| `requires_components` | `True` to consume 3-D per-item component embeddings (`*_comp` artifacts) instead of pooled 2-D ones. The train/eval enumeration routes `_comp` artifacts only to such models. See § 3.4. |

### 3.2 Minimal example

Copy [`plugins/recommenders/_example.py`](../plugins/recommenders/_example.py)
to a new file in the same directory (any name **without** a leading
underscore) and edit. The example below is a deterministic
uniform-noise baseline — useful as a sanity floor in benchmarks:

```python
import torch
import torch.nn as nn

from src.recommenders.base import BaseRecommender
from src.recommenders.registry import register_recommender


class UniformNoiseRecommender(BaseRecommender):
    """Uniform-noise scorer; serves as a ranking sanity floor."""

    def __init__(self, n_users, n_items, visual_embeddings, config):
        super().__init__(n_users, n_items, visual_embeddings, config)
        # A trainable parameter so the optimiser does not complain.
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, user_ids, pos_item_ids, neg_item_ids):
        pos = torch.rand_like(user_ids, dtype=torch.float32) + self.dummy
        neg = torch.rand_like(user_ids, dtype=torch.float32) + self.dummy
        return pos, neg

    def predict(self, user_id, item_ids):
        return torch.rand(item_ids.shape[0], device=self.dummy.device) + self.dummy


register_recommender(
    "uniform_noise",
    UniformNoiseRecommender,
    priority=0,
    requires_visual=False,
    uses_visual_dim=False,
)
```

### 3.3 Wire it into the run

Add the name to `configs/recommenders.yaml -> recommenders_enabled` and,
if your model has extra hyperparameters, also a block under the same
file with the value lists.

### 3.4 Component- and history-consuming recommenders (the ACF pattern)

The built-in `acf` (`src/recommenders/acf.py`) is the reference for a
recommender that needs more than a pooled embedding. Two additive,
defaulted hooks on `BaseRecommender` make this possible without
touching the other models:

- **Component buffer.** Set the class attribute
  `consumes_raw_components = True`. A 3-D `(n_items, M, D)` visual buffer
  is then kept raw — the base class does **not** instantiate an online
  fusion module (which assumes exactly two stacked sources). The model
  indexes `self.visual_features[item_ids]` itself. Pair this with
  `requires_components=True` at registration so the enumeration feeds it
  the `*_comp` artifacts (produced per § 1.4).
- **User history.** Set the class attribute `wants_history = True`. The
  train and eval steps then pass `train_interactions` (a
  `{user_idx: set(item_idx)}` dict, **train-only** so val/test never
  leak) to the constructor — accept it as a keyword-only
  `train_interactions=None` argument. Models that do not set the flag
  never receive the keyword, so their constructor is untouched.

All four flags (`requires_components`, `consumes_raw_components`,
`wants_history`, plus `supports_components` on the extractor side)
default off, so every pre-existing recommender, fusion and embedding
stays bit-identically reproducible.

---

## 4. Add a dataset

Two layouts are supported. Pick the simpler one when possible. A
ready-to-copy scaffold lives at
[`plugins/datasets/_example/`](../plugins/datasets/_example/) — the
leading underscore is what keeps it from being auto-registered, so
you can browse the layout safely. Copy the directory under any name
**without** the underscore (e.g. `plugins/datasets/my_dataset/`) to
activate it.

### 4.1 Layout A — files already on disk (no Python required)

```text
plugins/datasets/<name>/
├── interactions.csv     # columns: user_id, item_id, image_path
├── categories.csv       # OPTIONAL — columns: item_id, category_label
└── images/              # one file per item (any extension)
```

Then add `"<name>"` to `configs/default.yaml -> datasets:`. That is it.
The auto-registration uses
[`CSVDatasetProvider`](../src/data/example_csv.py) — a generic provider
that reads the CSV and discovers the per-item images.

### 4.2 Layout B — describe URLs to download

```text
plugins/datasets/<name>/
└── source.yaml
```

`source.yaml`:

```yaml
interactions:
  url: https://example.com/interactions.csv
images:
  url: https://example.com/images.tar.gz
```

The first run downloads everything into `data/raw/<name>/`. Subsequent
runs skip the download. Local paths (`path: /abs/path`) work too.

### 4.3 Custom Python provider

Drop in when neither layout fits — for example, when interactions live
inside a binary archive, or when categories need bespoke decoding. See
[`src/data/dvbpr.py`](../src/data/dvbpr.py) for the canonical example
of a `DatasetProvider` subclass with a hand-written
`load_categories()` method.

### 4.4 Why `categories.csv` matters (and how to get it)

Without `categories.csv`, the dataset is fully usable but **only
through Battery 1 (frozen embeddings)**.  Battery 2 (fine-tuning of
the visual extractors on a per-dataset classification task) needs
labelled items.  Datasets without categories will either:

* Skip Battery 2 entirely, or
* Fall back to **transfer learning** from a labelled dataset — the
  ``finetuning.tradesy_transfer_from`` knob in
  ``configs/finetuning.yaml`` controls the source.  The framework
  never trains a fine-tuner from scratch on a labelled dataset to
  use it on an unlabelled one without you opting in.

#### Format

```csv
item_id,category_label
i_001,0
i_002,1
i_003,0
i_004,2
```

* ``item_id`` matches the same external id used in
  ``interactions.csv``.  The provider remaps labels to a contiguous
  ``[0, n_classes)`` range automatically — your labels do not need to
  be 0-indexed or contiguous, only consistent.
* Items missing from ``categories.csv`` are simply not used during
  the fine-tuning step (still get embeddings extracted).
* The label space across rows defines ``n_classes``; the fine-tuner
  uses this to size its classification head.

#### Deriving `categories.csv` from a textual taxonomy

Many real-world datasets ship an item taxonomy as text (Amazon's
McAuley dump, Polyvore, etc.).  The framework's DVBPR provider
handles this automatically: when the ``.npy`` lacks the canonical
one-hot ``c`` field (as in ``amazon_men`` / ``amazon_women`` /
``tradesy``), ``DVBPRDataLoader.save_processed`` invokes
``src.data.categories.derive_categories`` on the embedded McAuley
taxonomy and writes ``data/raw/<name>/categories.csv`` for the
fine-tuning step. No manual pre-processing required.

To derive categories from a different layout (custom provider, non-
DVBPR ``.npy``, or in-memory iterable), call the helper directly:

```python
from src.data.categories import derive_categories, write_categories_csv

# items: iterable of (item_id, item_dict) — item_dict has a 'categories' field
mapping = derive_categories(items, level=3, min_samples=50)
if mapping is not None:
    write_categories_csv(mapping, "data/raw/my_dataset/categories.csv")
```

The helper returns ``{item_id: contiguous_label}`` with labels
remapped to ``[0, n_classes)``.  See ``src/data/categories.py`` for
the full contract (level fallback, minimum-samples filter, bytes-key
support).

#### Validating before you launch a multi-day run

```bash
python main.py --validate-dataset <name>
```

The check fails with a non-zero exit code when the layout is broken
(missing files, label space inconsistent, etc.).  Wire it into your
CI / shell aliases.

---

## 5. Add a pipeline step

Pipeline steps are not auto-discovered — they are part of the pipeline
ordering, which is intentional. Edit `main.py:STEP_ORDER` (and
`STEP_FUNCTIONS`) explicitly.

A step module under `src/steps/` exposes one callable:

```python
def run() -> None:
    """Idempotent. Skip work whose output already exists. Log progress."""
```

Every step reads the merged config via `src.utils.config.load_config()`
and writes its outputs under the relative paths declared in
`configs/default.yaml -> paths:`. Failures should be isolated per
sub-job (per `(extractor, dataset)`, per recommender, etc.) rather than
aborting the whole queue.

---

## 6. Validating a plugin before running the full pipeline

```bash
# Install dev deps once:
pip install -e ".[dev]"

# Run the focused contract suite:
pytest -q

# Run only your plugin's tests (e.g. by file pattern):
pytest tests/test_extractor_registry.py -q
```

The suite under [`tests/`](../tests/) checks:

- Plugin registration and lookup contracts.
- BaseExtractor `unfreeze_prefixes` declarations on every fine-tunable
  built-in.
- Fine-tuning checkpoint round-trip and legacy compatibility.

Add a test there for any non-trivial plugin you submit so the next
contributor inherits the safety net you used.

---

## 7. Common mistakes

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `KeyError: No extractor registered for 'my_x'` | The plugin module never got imported. | Make sure the file lives under `plugins/` and the file does **not** start with `_`. |
| FT loss never decreases | `unfreeze_prefixes` matches no submodule names. | Print `model.named_parameters()` and adjust the prefix. |
| Re-extraction fails with "missing key projection" | Backbone shape changed between runs. | Delete the embedding file under `data/embeddings/<dataset>/` so the step recomputes. |
| `RecommenderSpec` lookup error | Forgot to add the new name to `recommenders_enabled`. | Edit `configs/recommenders.yaml`. |
