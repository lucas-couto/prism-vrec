"""Registry for recommendation models.

A custom recommender is any subclass of :class:`BaseRecommender`
registered under a name via :func:`register_recommender`.  Once
registered, the name can appear in ``recommenders_enabled`` inside
``configs/recommenders.yaml`` and the pipeline picks it up like a
built-in: grid search, parallel orchestration, evaluation.

The :class:`RecommenderSpec` holds everything the pipeline needs to know
about a model so that ``src/steps/train.py`` and ``src/utils/parallel.py``
remain model-agnostic — no hardcoded ``if model_name == ...`` branches.

Adding a custom recommender
---------------------------

::

    # plugins/recommenders/my_model.py
    from src.recommenders.base import BaseRecommender
    from src.recommenders.registry import register_recommender

    class MyRecommender(BaseRecommender):
        def forward(self, user, pos_item, neg_item): ...
        def predict(self, user_idx): ...

    register_recommender(
        "my_model",
        MyRecommender,
        priority=5,
        requires_visual=True,
        uses_visual_dim=False,
        extra_hyperparam_keys=("custom_param",),
    )

Then in ``configs/recommenders.yaml``::

    recommenders_enabled:
      - my_model

    my_model:
      custom_param: [16, 32]   # becomes a Cartesian dimension
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RecommenderSpec:
    """Everything the pipeline needs to drive one recommender.

    Attributes
    ----------
    name:
        Registry key; appears in ``configs/recommenders.yaml``.
    cls:
        Subclass of :class:`BaseRecommender`.
    priority:
        Lower = scheduled first.  Used by the parallel orchestrator to
        run cheap models before expensive ones (so partial results
        appear faster).
    requires_visual:
        ``True`` if the model consumes a visual-embedding matrix.
        Models with ``requires_visual=False`` (e.g. plain BPR) only run
        in the ``frozen`` condition with ``embedding_name="none"``.
    uses_visual_dim:
        ``True`` when the model has a ``visual_dim`` hyperparameter.
        Adds ``common.visual_dim`` to the grid.
    extra_hyperparam_keys:
        Tuple of keys to read from
        ``configs/recommenders.yaml -> <name>:``.  Each value may be a
        scalar or a list (lists become Cartesian dimensions in the
        grid).
    """

    name: str
    cls: type
    priority: int = 5
    requires_visual: bool = True
    uses_visual_dim: bool = False
    extra_hyperparam_keys: tuple[str, ...] = field(default_factory=tuple)


_REGISTRY: dict[str, RecommenderSpec] = {}


def register_recommender(
    name: str,
    cls: type,
    *,
    priority: int = 5,
    requires_visual: bool = True,
    uses_visual_dim: bool = False,
    extra_hyperparam_keys: tuple[str, ...] | list[str] = (),
) -> None:
    """Register a recommender class under ``name``.

    Re-registering an existing name overwrites the previous binding.
    The class must be a subclass of :class:`BaseRecommender` (validated
    lazily to avoid an import cycle at module-load time).
    """
    if not isinstance(cls, type):
        raise TypeError(
            f"register_recommender({name!r}): cls must be a class, got {type(cls).__name__}"
        )

    # Local import avoids circular dependency (base.py -> registry).
    from src.recommenders.base import BaseRecommender

    if not issubclass(cls, BaseRecommender):
        raise TypeError(
            f"register_recommender({name!r}): {cls.__name__} must subclass BaseRecommender"
        )

    _REGISTRY[name] = RecommenderSpec(
        name=name,
        cls=cls,
        priority=priority,
        requires_visual=requires_visual,
        uses_visual_dim=uses_visual_dim,
        extra_hyperparam_keys=tuple(extra_hyperparam_keys),
    )


def get_recommender_spec(name: str) -> RecommenderSpec:
    """Return the :class:`RecommenderSpec` registered under ``name``."""
    spec = _REGISTRY.get(name)
    if spec is None:
        raise KeyError(
            f"No recommender registered for {name!r}.  "
            f"Available recommenders: {registered_recommender_names()}.  "
            f"Register a custom recommender via "
            f"src.recommenders.registry.register_recommender(name, cls, ...)."
        )
    return spec


def get_recommender_class(name: str) -> type:
    """Return the recommender class registered under ``name``."""
    return get_recommender_spec(name).cls


def registered_recommender_names() -> list[str]:
    """Return the sorted list of currently-registered recommender names."""
    return sorted(_REGISTRY)


def is_registered(name: str) -> bool:
    """Return True iff ``name`` is currently registered."""
    return name in _REGISTRY


def iter_specs() -> list[RecommenderSpec]:
    """Return the registered specs in priority then name order."""
    return sorted(_REGISTRY.values(), key=lambda s: (s.priority, s.name))
