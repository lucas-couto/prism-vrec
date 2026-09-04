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

## Track progress and cost projection

```
uv run python main.py --battery-status
```
Prints the count per state (`pending/running/done/failed`) and the
**estimate of remaining hours** (average duration per cell type ×
pending). Roles without a completed sample yet are reported as "no
estimate", never guessed.

## Failures and retry

A cell that fails is isolated (the others keep going) and marked `failed`
in the manifest with the error message. To reprocess only the ones that
failed:
```
uv run python main.py --battery --retry-failed
```

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

Any accuracy metric is **recomputable** from the persisted rank, for any
`k`, without a GPU (`src/evaluation/derive_metrics.py`); the paired
users × systems matrix for the statistical tests comes from
`src/evaluation/paired_loader.py`.

## K-fold cross-validation over the battery (`--folds`)

After the hyperparameter search has finished (or with
`hp_search.strategy: fixed`), enable `folds:` in `configs/default.yaml`
and run:

```bash
python main.py --folds
```

The runner (`src/folds/runner.py`) is resumable through
`results/folds/manifest.json`: a cell whose concatenated per-user
artifact already exists is skipped. Per fold it writes the trained
checkpoint under `<results>_fold<i>/models/` and the partial artifact
under `results/per_user/<dataset>/folds/fold<i>/`; the final artifact
lands in the canonical `results/per_user/<dataset>/` location, so
`--report` and the paired statistics consume it unchanged. The manifest
entry of every cell records the hyperparameter origin (prior search,
with the source cell reference, or fixed config values), the partition
summary (fold sizes, excluded users by reason), the per-fold seeds and
fold-in reports, the between-fold mean/std of recall@k and ndcg@k, and
the note that this variability is combined (partition + optimisation).
See `docs/protocol.md` §3b.
