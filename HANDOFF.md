# HANDOFF — PRISM VREC

Last updated: 2026-09-05

## Current task and constraints

The user authorized incremental fixes after a repository audit, then requested a detailed executable spec-driven development package before continuing implementation. Full-ranking evaluation must remain in place: every catalogue item participates; no sampled-negative replacement. Existing working-tree changes must be preserved.

## Latest deliverable: executable specifications

- Published 24 files, including six module specifications and 30 task cards, at `/home/lucas-couto/.Codex/projects/prism-vrec/reliability-sdd-2026-09-05/README.md`.
- Package includes requirements, evidence-ranked audit findings, proposed contracts, implementation order, test/fault/resource matrices, migration/release gates and a task execution template.
- Start implementation with I01 (save the first valid selection winner even when zero), then I02 and failure propagation E01/E02. Follow MODULES.md for the complete order.
- M00 identifies the prior item-block fallback below; it is implemented locally but not GPU/resource-certified. All other tasks are specified/open.
- New public interface/schema proposals C01–C06 require explicit approval before implementation. Documentation approval does not imply contract approval.
- Documentation links and code-fence structure were checked; all 30 task IDs are unique. Source hashes and the dirty-file inventory are recorded in snapshot.json.
- No further source implementation or test runs were performed while writing the package. Prior test results are explicitly historical evidence, not fresh certification.

## Previously completed implementation

- Replaced the single-user full-catalogue scoring fallback with adaptive item blocks, initially at most 1024 items.
- On CUDA allocation failure, shrink the item block and retry the same offset after leaving the exception handler, so the traceback no longer retains failed tensors.
- Copy successful scores into a host vector and use the existing full-catalogue CPU ranking, masks, metrics and seeded tie-break.
- Move batched evaluation throughput accounting after successful ranking to avoid counting failed retries.
- Added five regression tests: exact catalogue coverage and records, mid-pass retry, traceback tensor release, explicit one-item failure, and VNPR score equivalence within floating-point tolerance.

## Validation

234 relevant tests passed in the Docker environment with CUDA disabled, covering recommenders, ranking budgets, deterministic ranking, non-finite guards, sampled-path compatibility, records and paper properties. Ruff check/format and git diff --check passed. CUDA failures were simulated; no production GPU memory or dataset battery validation was performed.

## Scope and remaining work

This is NOT a universal no-OOM guarantee. Model parameters, resident features, initial GPU index caches, CPU score vectors and sorting buffers still require memory. An item prediction that cannot fit even at block size one fails explicitly. The training model and scientific protocol were not changed.

Further audit findings remain open: resident feature memory, failure propagation, zero-valued runs without a best checkpoint, seed/checkpoint isolation, stale artifact reuse, fine-tuning best-state resume, item-index ordering, battery replay hyperparameter expansion, non-learned online normalization, extraction completion recovery and transactional persistence. The cause of the observed VNPR collapse on learned fusion is still unproven; learned alignment does apply pre-fusion normalization.

## Files changed in this step

- src/evaluation/protocol.py
- tests/test_ranking_batch_budget.py
- HANDOFF.md

No commit was created. Do not interpret earlier conversation promises about universal OOM safety as verified behavior.
