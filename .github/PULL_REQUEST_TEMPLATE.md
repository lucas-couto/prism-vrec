## What this PR does

One paragraph, in English.  Cite the matching issue when applicable
(e.g. "fixes #42").

## Why

The motivation that justifies the change.  Skip for trivial fixes
(typos, dead links).

## How to verify

Smallest reproduction the reviewer can run locally.

```bash
# example
ruff check . && ruff format --check .
pytest -q
```

## Checklist

- [ ] `ruff check .` passes
- [ ] `ruff format --check .` passes
- [ ] `pytest -q` passes
- [ ] If a plugin contract or the FT checkpoint format changed, the
      matching test was updated
- [ ] If a feature was added, `CHANGELOG.md`'s `[Unreleased]` section
      mentions it
- [ ] Commit subjects follow Conventional Commits and stay under 70
      characters
