---
name: Bug report
about: Something in the framework misbehaves — wrong math, broken pipeline step, regression.
labels: bug
---

## What happened

A clear, factual description of the misbehaviour.

## What you expected to happen

What the documented contract promises (cite the relevant doc/section
when possible — README, `docs/extending.md`, `docs/recipes.md`).

## Reproduction

```bash
# Smallest set of commands that produces the bug, ideally from a fresh
# checkout.  If a config edit is required, paste the exact YAML diff.
```

## Environment

- Framework version / commit SHA: (e.g. `8041e4b`)
- Python: (e.g. `3.10.13`)
- PyTorch / CUDA: (e.g. `torch 2.3.1+cu121` on `RTX 4090`)
- OS / runtime: (e.g. `Ubuntu 22.04 inside Docker on RunPod`)

## Logs

Relevant excerpt from `logs/run_<id>.log` or stdout.  Trim to the
20–50 lines around the error so the report stays scannable.

```
[paste here]
```

## Possible cause / what you have already tried

Optional, but helps maintainers triage faster.
