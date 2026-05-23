# User datasets

Drop a directory under ``plugins/datasets/<name>/`` and the pipeline
will pick it up automatically, no Python code, no edits to
``src/``.  After creating the directory, add ``"<name>"`` to the
``datasets:`` list in ``configs/default.yaml`` and run the pipeline.

The auto-registration is opt-in by name: dropping a directory is not
enough, the YAML must explicitly mention it.  This keeps reruns
predictable and avoids accidentally pulling in scratch data.

> A ready-to-copy scaffold lives in [`_example/`](_example/), a
> directory with the leading underscore is **never** auto-registered,
> so it is safe to keep around as a reference.  Copy it to a name
> without the underscore (e.g. `my_dataset/`), edit the CSVs, and you
> are done.

## Two supported layouts

### Layout A, files already on disk

```
plugins/datasets/<name>/
├── interactions.csv      # columns: user_id, item_id, image_path
└── images/               # one image per item (any extension)
    ├── it_001.jpg
    ├── it_002.jpg
    └── ...
```

* ``interactions.csv`` is the user-item interaction log.  One row per
  observed ``(user, item)`` pair.  ``image_path`` is interpreted
  relative to the directory of ``interactions.csv``.
* The default split is leave-one-out: each user contributes 1 random
  interaction to ``test``, 1 to ``val``, the rest to ``train``.  Users
  with fewer than 3 interactions are dropped.  Reproducible via the
  global ``seed`` in ``configs/default.yaml``.

### Layout B, describe URLs to download

```
plugins/datasets/<name>/
└── source.yaml
```

with::

    interactions:
      url: https://example.com/interactions.csv
      # or, if the file already lives somewhere on disk:
      # path: /absolute/path/to/interactions.csv
    images:
      url: https://example.com/images.tar.gz
      # path: /absolute/path/to/images-folder/

The first run downloads the URLs into ``data/raw/<name>/`` and from
there behaves exactly like Layout A.  Tar and zip archives are
extracted in place.

## Sanity-check the layout before launching a multi-day grid search

```python
from src.data.base import validate_layout

# After running the preprocess step at least once
problems = validate_layout("<name>")
print(problems or "OK")
```

A non-empty list is the function's way of telling you the on-disk
layout is broken, fix it before launching anything expensive.

## When auto-registration is not enough

Auto-registration covers the vast majority of "drop a CSV + a folder
of images" cases.  When you need full control (custom split logic,
streamed images, weird metadata), write a small
``DatasetProvider`` subclass and register it explicitly, see
``src/data/base.py`` for the contract and ``src/data/example_csv.py``
for a worked example.

For installation-validation scenarios (CI, post-install smoke check,
"does my plugin still register?"), the framework ships a built-in
``SyntheticDatasetProvider`` (``src/data/synthetic.py``) that
generates a tiny deterministic dataset entirely in-process — no
download, no external files.  It is auto-registered under the name
``synthetic`` and powers the bundled smoke profile
(``python main.py --all --config-dir configs/smoke``, see
[the README](../../README.md#12-smoke-profile-installation-validation)).
