# Task execution record: RC1-FOLLOWUP (first real run of the candidate)

- Status: implemented / verified on CPU; the real-data effect (eval time, fusion peak) is to be confirmed on the next run
- Date (UTC): 2026-09-06
- Executor: integration branch `feat/3.0.0-rc.1-reliability-sdd`
- Source HEAD and dirty diff reference: 9a50f8c + this change set
- Specification version: reliability and scientific-integrity specification (2026-09-05)
- Dependencies verified: E05 (provenance sidecars), M02/M03 (visual mapping), M05 (admission), M00 (per-user fallback)
- Contract proposals and approval references: contracts preserved; `MAX_FUSION_WORKERS = 1` is a researcher decision (2026-09-06)

## Defect and reproduction

Observed on the first battery run of `3.0.0rc1` (run `2026-09-06T14-34-15Z` and its restart), all confirmed:

1. **Phantom embeddings.** `get_embedding_files` globbed `hybrid_*.json` and picked up the E05 sidecars `hybrid_*.json.provenance.json`; 92 jobs (44 amazon_fashion, 44 amazon_men, 4 amazon_women) failed with "online sidecar ... lists no components; cannot stack", and `hybrid_concat.npy.provenance` was warned about as an unregistered strategy. With E02 the step would have ended in `TrainingJobsFailedError` after all 976 jobs.
2. **Per-user fallback on every user.** For `vbpr x hybrid_*_learned_D128 x amazon_women` the ranking planner's 106-user batch OOMed, halved down to 1 user, still OOMed, and the evaluator fell back to per-user item-block scoring on CPU: 37,015 warnings, `eval_s` 414-482 s per pass instead of 0.7 s. Mechanism: in dense mode `_map_visual`/`_resolve_visual` passed the whole catalogue to the online fusion in one call; `self.visual_features[item_ids]` copies the 3.9 GB buffer before the fusion's temporaries, independent of the user batch, and the VRAM cap is 8 GB since the 50% share.
3. **Fusion worker OOM-killed.** `docker events` recorded `oom` then `die 1` for the container during `fuse`; the planner had admitted 2 workers at 5.6 GB each while the fit-matrix assembly held a whole gathered source plus its normalised copy on top of the fit matrix (about 8.3 GB for tradesy). `BrokenProcessPool` carried no cause.
4. **CI-only test failures.** `test_eval_scores_match_dense[DeepStyle-stacked]` differed by 2.4e-7 absolute on the GitHub runner (OpenBLAS) with `atol=1e-7`; `test_unequal_grid_sizes_are_reported_not_equalised` asserted on `caplog`, which listens on the root logger the project's loggers do not propagate to.

Before/after for (3): tracemalloc peak above the fit matrix on a 20K x 256 x 2-source fixture, 16K fit rows, chunk 1024: **31.4 MB before, 2.0 MB after** (bound 4.0 MB).

## Implementation

- `src/steps/train.py`: discovery skips `PROVENANCE_SUFFIX` (new constant in `src/utils/identity.py`, used by `provenance_path`).
- `src/recommenders/base.py`: `_map_visual` blocks every request larger than `_LAZY_ITEM_BLOCK` (dense included); `_resolve_visual` fuses 1-D requests block by block through the new `_fuse_rows`. Row-wise functions only, so the concatenation equals one call; autograd preserved (`torch.cat`).
- `src/fusions/streaming.py`: `_gather_rows` (chunked gather + normalise into a preallocated block); `_assemble_fit_matrix` and `_stream_pca_per_model` use it; `_stream_pca` passes its chunk size.
- `src/steps/fuse.py`: `MAX_FUSION_WORKERS = 1` applied after the memory plan (the plan is still logged); `_run_fusion_pool` re-raises `BrokenProcessPool` as `FusionWorkerLostError` with the completed count and `_cgroup_oom_kills()` (reads `/sys/fs/cgroup/memory.events`); `_STREAM_FIT_FACTOR` comment corrected.
- Tests: `tests/test_embedding_discovery.py`, `tests/test_fuse_fit_assembly.py` (assembly equality + bounded peak, per-model gather, lost worker with a real `os._exit` worker, healthy pool, optional cgroup counter), `tests/test_fuse_worker_sizing.py` (planner tests lift the pin; one test asserts the pin), `tests/recommenders/test_lazy_feature_equivalence.py` (`ATOL` 1e-7 -> 1e-6 with justification; `test_dense_catalogue_requests_are_blocked_and_unchanged` for VBPR pooled/learned, VNPR learned gated, DeepStyle stacked with a raw-row spy), `tests/test_battery_dispatch.py` (logger propagation enabled for the assertion).

Scientific invariants: none touched. Scores, candidates, masks, ties and selection are unchanged; the blocking is an execution layout.

## Verification

| Command / experiment | Environment | Result | Evidence artifact |
|---|---|---|---|
| focused: fuse assembly, sizing, streaming, discovery, dispatch, lazy equivalence, golden ranking, provenance, projection | container, CPU | 174 passed | this record |
| dense blocking equivalence (`-k dense_catalogue`) | container, CPU | 5 passed | this record |
| before/after assembly peak script | container, CPU | 31.4 MB -> 2.0 MB above fit matrix | this record |
| full non-slow suite | container, CPU | see CHANGELOG / HANDOFF | CI run on PR #37 |

## Acceptance

- [x] Task-specific acceptance oracle passed (phantom stems gone; catalogue requests bounded; assembly peak bounded; lost worker named).
- [x] Relevant Q requirements passed (Q01/Q02 unchanged scores; Q10/Q11 admission honest, failure explicit).
- [x] No unrelated source modifications.
- [x] No silent protocol/data/population changes.
- [x] Failure and migration behavior verified where relevant.

## Limits and next handoff

Not measured on the GPU: the next run must show `eval_s` back near 1 s for `hybrid_*_learned_D128 x amazon_women` and no "Ranking OOM at a single user" bursts. The one-worker fusion pin trades wall-clock for memory safety; the planner's would-be size is logged for a later review. Completed jobs of the failed run are reused through identity v2; the 92 phantom cells simply no longer exist.
