"""User-supplied extension points for the framework.

Subpackages
-----------
* ``plugins.extractors`` — visual feature extractors registered via
  :func:`src.extractors.registry.register_extractor`.
* ``plugins.fusions`` — fusion strategies registered via
  :func:`src.fusions.registry.register_fusion_strategy`.
* ``plugins.recommenders`` — recommender models registered via
  :func:`src.recommenders.registry.register_recommender`.
* ``plugins.datasets`` — user-supplied dataset directories
  (interactions.csv + images/, or source.yaml).  Auto-discovered by
  :mod:`src.data.auto_register`.

Drop a Python module into the matching subpackage and the pipeline
auto-registers it at startup — no edits to ``src/`` are required.  See
``docs/extending.md`` for the full contract of every plugin type.
"""
