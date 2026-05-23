# Example dataset layout

This directory is a scaffold showing the two layouts the auto-discovery
accepts. The leading underscore on `_example/` is what keeps the
pipeline from registering it — `src/data/auto_register.py` skips
directories whose names start with `.` or `_`.

To add your own dataset, copy this directory to a name **without** a
leading underscore (e.g. `plugins/datasets/my_dataset/`), fill the
files in, and add `"my_dataset"` to the `datasets:` list in
`configs/default.yaml`.

## Layout A — files already on disk

```
plugins/datasets/<name>/
├── interactions.csv         # columns: user_id, item_id, image_path
├── categories.csv           # OPTIONAL — columns: item_id, category_label
└── images/                  # one image per item (any extension)
    ├── it_001.jpg
    ├── it_002.jpg
    └── ...
```

- `interactions.csv` is the user-item interaction log — one row per
  `(user, item)`. `image_path` is interpreted relative to the
  dataset directory.
- `categories.csv` enables the fine-tuning step. Without it, the FT
  step skips this dataset (or transfers from a labelled dataset, see
  `configs/finetuning.yaml -> tradesy_transfer_from`).
- The default split is leave-one-out: 1 test + 1 val per user, the
  rest in train. Users with fewer than 3 interactions are dropped.

## Layout B — describe URLs to download

```
plugins/datasets/<name>/
└── source.yaml
```

```yaml
interactions:
  url: https://example.com/interactions.csv
  # or: path: /absolute/path/to/interactions.csv

images:
  url: https://example.com/images.tar.gz   # tar/zip extracted in place
  # or: path: /absolute/path/to/images-folder/
```

The first run downloads/copies into `data/raw/<name>/` and behaves
identically to Layout A from there.

## Layout C — full programmatic provider

For datasets that do not fit either layout (binary archives, database
queries, bespoke category logic), subclass `DatasetProvider` and
register it from a Python module imported at startup. See
[`docs/extending.md`](../../../docs/extending.md) § 4.3 for the recipe
and `src/data/dvbpr.py` for the canonical example.
