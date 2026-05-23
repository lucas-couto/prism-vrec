# Contributing

Thanks for considering a contribution. The framework is a research
codebase — most welcome contributions fall into one of three buckets:

1. **New plugin** — extractor, fusion strategy, recommender or dataset.
2. **Bug fix** — incorrect math, wrong path, broken auto-discovery,
   regression in one of the existing pipeline steps.
3. **Documentation** — recipes, clarifications, typos.

## Quick start

```bash
git clone https://github.com/lucas-couto/prism-vrec.git
cd prism-vrec

# Install package + dev tools (ruff, pre-commit, pytest)
pip install -e ".[dev]"

# Wire pre-commit so every commit runs ruff + the standard hygiene hooks
pre-commit install
```

## Adding a plugin

This is the most common contribution path and the framework is designed
to make it small. Read [`docs/extending.md`](docs/extending.md) — it
walks through the contract of every plugin type with a runnable
example. Each `plugins/<domain>/` directory ships an `_example.py`
(or `_example/` for datasets) you can copy as a starting point.

Smallest acceptable PR for a new plugin:

1. The plugin file under `plugins/<domain>/<name>.py`.
2. The new name added to the relevant `*_enabled` list in `configs/`
   (or a note in the PR explaining why it is opt-in).
3. A focused test under `tests/` exercising the new contract — at
   minimum that the plugin registers and instantiates.

## Bug fixes

Open an issue first (see the templates in `.github/ISSUE_TEMPLATE/`)
when the bug is non-obvious or touches the algorithmic core (training,
evaluation, fusion math, FT checkpoint format). Algorithmic changes
need a regression test that pins the corrected behaviour.

For trivial fixes (typos, dead links, broken paths) skip the issue and
go straight to a PR.

## Coding standards

- **Python 3.11 or 3.12**, fully type-hinted. CI validates both; the Docker image is pinned to 3.11 for canonical-runtime reproducibility.
- **Ruff** is the single source of truth for lint and formatting.
  Configuration lives under `[tool.ruff]` in `pyproject.toml`. Run
  `ruff check . && ruff format .` before pushing — CI runs the same
  commands.
- **English everywhere in code** (variables, comments, docstrings, log
  messages, commit subjects). User-facing strings, when present, may
  be in Portuguese but should be isolated.
- **No `print()` in library code** — use a logger obtained via
  `src.utils.logging.get_logger(__name__)` so output lands in both the
  per-module file and the unified session log.
- **Commit messages** follow Conventional Commits
  (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `ci:`, `chore:`).
  The subject line is in the imperative ("add X" / "fix Y"), under 70
  characters, in English.

## Tests

```bash
pytest -q                           # full suite
pytest tests/test_extractor_*.py    # one file
pytest -k bpr                       # by keyword
```

The suite is structured into:

- **Contract tests** — registry round-trip, plugin registration,
  checkpoint format. These run in seconds, no GPU, no model load.
- **Functional tests** — fusion math, BPR loss, FT freeze/unfreeze
  accounting on toy inputs. CPU-only, no real backbone.

Adding a plugin? Add a contract test. Touching algorithmic code? Add
or update a functional test that pins the corrected behaviour against
a hand-computed reference.

## Pull request checklist

Before opening a PR:

- [ ] `ruff check .` passes.
- [ ] `ruff format --check .` passes.
- [ ] `pytest -q` passes.
- [ ] If you touched a plugin contract or the FT checkpoint format,
      the matching test was updated to cover the new behaviour.
- [ ] If you added a feature, the `[Unreleased]` section in
      `CHANGELOG.md` mentions it.
- [ ] Commit message follows Conventional Commits and the subject is
      under 70 chars.

PRs that match this checklist usually merge in one review pass.

## Reporting security issues

Please do not open a public issue for security-sensitive findings.
Email the project author at the address in `CITATION.cff` instead.

## Code of conduct

Be civil; assume good faith; favour direct technical feedback over
indirect critique. The maintainer reserves the right to close threads
that derail into off-topic disputes.
