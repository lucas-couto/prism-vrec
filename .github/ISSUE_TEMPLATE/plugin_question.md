---
name: Plugin authoring question
about: You are writing a plugin (extractor / fusion / recommender / dataset) and the contract is unclear.
labels: question, plugin
---

## What plugin type are you adding

- [ ] Visual extractor (`plugins/extractors/`)
- [ ] Fusion strategy (`plugins/fusions/`)
- [ ] Recommender model (`plugins/recommenders/`)
- [ ] Dataset (`plugins/datasets/`)

## What the docs already cover

Have you read [`docs/extending.md`](../../docs/extending.md) §
matching your plugin type?  Quote the specific paragraph that did not
answer your question, or note that the section is missing the case
you ran into.

## What you tried

Show the smallest version of your plugin that exhibits the question.
A 30-line snippet with the relevant `register_*()` call is usually
enough.

```python
# plugins/<domain>/<your_file>.py
...
```

## Behaviour observed

Stack trace, log line, or "the registry never picked it up" — be
explicit so the maintainer can pinpoint whether the issue is in your
plugin, the contract, or the auto-discovery.

## Behaviour expected

What the docs led you to expect.

## Environment

- Framework version / commit SHA:
- Python:
