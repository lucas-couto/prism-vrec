# Reliability task records — index

One record per task of the reliability and scientific-integrity
specification (2026-09-05), shipped in 3.0.0rc1. Each record states the
defect and its reproduction, the files changed, the verification table
(all on CPU in the container, CUDA disabled), the acceptance checklist
and the limits. This directory is the evidence index referenced by
`docs/protocol.md`, `docs/battery_runbook.md` and `CHANGELOG.md`.

Status legend: **verified** = regression reproduced before the patch and
green after, focused suites green; **not resource-certified** = no GPU,
no real-catalogue peak measured; **unresolved** = the scientific question
stays open by design.

| Record | Finding | Scope | Status |
|---|---|---|---|
| [I01](I01.md) | F03 | First finite selection observation wins (zero included); invalid metric fails; promotion validated | verified |
| [I02](I02.md) | F03 | Evaluate requires a loadable, complete winner; legacy flat checkpoint refused | verified |
| [I03](I03.md) | F07 | Fine-tuning resume envelope v2 with the historical best in its own digested file | verified; CUDA/AMP scaler resume not certified |
| [I04](I04.md) | F03/F07 | Training resume envelope v2 (identity, best ref, scaler); genuine-kill equivalence | verified; CUDA/AMP not certified |
| [E01](E01.md) | F04 | One terminal outcome per submitted job; dead workers reaped; bounded OOM retry (3 attempts) | verified; not measured on a GPU pool |
| [E02](E02.md) | F04 | Failed / unaccounted work fails the step and the run; real CLI exit code; `IncompleteRunError` | verified |
| [E03](E03.md) | F05 | Workers receive the resolved config snapshot; roots from the snapshot; scientific identity v2 module | verified |
| [E04](E04.md) | F05/F06 | Grid, winner, resume and evaluate reuse bound to the identity; global checkpoint wipe removed | verified |
| [E05](E05.md) | F06 | `.provenance.json` for projected / PCA-aligned / fused artifacts; mismatch refused, legacy unverified | verified |
| [E06](E06.md) | F12 | Atomic per-cell generations behind a validated completion pointer; upserted summary rows | verified (injected interruptions; local POSIX fs) |
| [E07](E07.md) | F12/F06 | Atomic manifest; `done` bound to validated artifacts; fold-plan digest | verified |
| [S01](S01.md) | F08 | Feature row order from `item2idx` values; missing images fail; item-order digest in sidecars | verified; no production re-extraction |
| [S02](S02.md) | F10 | Non-learned online sidecar normalises each source at load; `SIDECAR_RECIPE_VERSION = 2` | verified; historical `hybrid_adaptive_gated_*` results non-comparable |
| [S03](S03.md) | F11 | Extraction resume v2 from the validated durable prefix; v1 progress restarts | verified |
| [S04](S04.md) | F13 | Opt-in training diagnostics + controlled VNPR collapse driver | probes verified; **cause still unresolved on synthetic data** (real-data run pending) |
| [R01](R01.md) | F09 | Battery executor dispatches per declared strategy; replay from the primary winner | verified |
| [R02](R02.md) | F09 | One canonical `effective_hyperparams`; replay resolution with provenance | verified |
| [R03](R03.md) | F16 | Per-dataset budget consumed by every training path; `SELECTION_K_VALUES`; unsupported metric fails early | verified |
| [R04](R04.md) | F15 | Paired observation validation: key, duplicates, population policy, non-finite values | verified; not profiled on real tables |
| [R05](R05.md) | F15 | Partitioned, reconciled reports; `_integrity.json`; distinct-seed counts; fold partial identity | verified |
| [M01](M01.md) | F02 | `FeatureSource` adapters; `load_embedding(lazy=True)` opt-in | verified; not resource-certified |
| [M02](M02.md) | F02 | Recommenders gather raw rows per forward from lazy sources; blocked catalogue requests | verified numerically for every visual recommender |
| [M03](M03.md) | F02 | Bounded, version-keyed derived caches (ACF projection, VNPR catalogue alias) | verified |
| [M04](M04.md) | F01 | Host-resident catalogue ids and chunk masks; checked host ranking budget; golden fixture | verified; CUDA OOM simulated |
| [M05](M05.md) | F02 | Host budget resolution, per-job memory ledger, admission; `resources:` block (C06, provisional) | verified with synthetic budgets; estimates analytic |
| [M06](M06.md) | — | Block-wise feature validation; unknown worker footprint charged; pools enforce admission | verified (`tracemalloc` on a synthetic memmap) |
| [V01](V01.md) | F14 | Lock-backed Dockerfile with hash verification; CI torch pins; `.dockerignore`; `docs/environment.md` | verified by resolution only; **image not rebuilt**, `uv.lock` regeneration pending |

Not in this directory: M00 (the adaptive item-block ranking fallback,
implemented before the specification and retained by M04's tests), V02
(resource certification on real data — open) and V03 (this
documentation pass, recorded in `CHANGELOG.md` 3.0.0rc1).

Finding map (specification audit): F01 → M00/M04; F02 → M01–M06;
F03 → I01/I02/I04; F04 → E01/E02; F05 → E03/E04; F06 → E04/E05/E07;
F07 → I03/I04; F08 → S01; F09 → R01/R02; F10 → S02; F11 → S03;
F12 → E06/E07; F13 → S04; F14 → V01 (V02 open); F15 → R04/R05;
F16 → R03.

Open follow-ups noted by the records: `src/steps/finetune.py` does not
write the `item_order` block for `<extractor>_finetuned.npy` (S01);
`src/recommenders/_scoring.py` caches rely on `train()` for
invalidation (M03 recommends a version-keyed guard); the battery
executor's `_evaluate_one_cell` does not pass `lazy_features` (M05);
`_run_optuna`'s cell pool is still sized by the VRAM heuristic (M05);
`CheckpointManager.clear_all_training_checkpoints` is defined and
never called (E04); `evaluate._evaluate_cell` still reads
`load_config()["paths"]` for DeepStyle categories (E03).
