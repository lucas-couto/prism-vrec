"""Registry for visual feature extractors.

A custom extractor is any subclass of :class:`BaseExtractor` registered
under a name via :func:`register_extractor`.  Once registered, the name
can appear in ``extractors_enabled`` (and optionally
``fusion_extractors``) inside ``configs/extractors.yaml`` and the
pipeline picks it up exactly like a built-in.

Built-in extractors are registered in :mod:`src.extractors`'s package
init.  Custom extractors dropped under ``plugins/extractors/`` are
auto-discovered at import time (see :mod:`src.extractors.auto_register`).
"""

from __future__ import annotations

from collections.abc import Callable

from src.extractors.base import BaseExtractor

# A factory (rather than the bare class) keeps lazy imports possible:
# a plugin module that needs heavy deps only imports them when its
# factory is invoked.
_REGISTRY: dict[str, Callable[[], type[BaseExtractor]]] = {}


def register_extractor(
    name: str,
    cls_or_factory: type[BaseExtractor] | Callable[[], type[BaseExtractor]],
) -> None:
    """Register an extractor class (or factory) under ``name``.

    ``cls_or_factory`` may be either:

    * a :class:`BaseExtractor` subclass — registered directly;
    * a zero-argument callable returning such a subclass — useful when
      the import cost is high and should be deferred.

    Re-registering an existing name overwrites the previous binding.
    """
    if isinstance(cls_or_factory, type):
        if not issubclass(cls_or_factory, BaseExtractor):
            raise TypeError(f"register_extractor({name!r}): class must subclass BaseExtractor")
        cls = cls_or_factory
        _REGISTRY[name] = lambda: cls
        return

    if not callable(cls_or_factory):
        raise TypeError(
            f"register_extractor({name!r}): expected a class or callable, "
            f"got {type(cls_or_factory).__name__}"
        )
    _REGISTRY[name] = cls_or_factory


def get_extractor_class(name: str) -> type[BaseExtractor]:
    """Return the registered extractor class for ``name``.

    Raises :class:`KeyError` (with a helpful list of available names)
    when ``name`` is unknown.
    """
    factory = _REGISTRY.get(name)
    if factory is None:
        raise KeyError(
            f"No extractor registered for {name!r}.  "
            f"Available extractors: {registered_extractor_names()}.  "
            f"Register a custom extractor via "
            f"src.extractors.registry.register_extractor(name, cls)."
        )
    cls = factory()
    if not (isinstance(cls, type) and issubclass(cls, BaseExtractor)):
        raise TypeError(
            f"Factory for {name!r} returned {cls!r}, expected a BaseExtractor subclass."
        )
    return cls


def registered_extractor_names() -> list[str]:
    """Return the sorted list of currently-registered extractor names."""
    return sorted(_REGISTRY)


def is_registered(name: str) -> bool:
    """Return True iff ``name`` is currently registered."""
    return name in _REGISTRY
