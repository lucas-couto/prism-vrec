"""The training worker's caches must hold one entry, not the whole run.

An unbounded cache here accumulated every dataset and every fusion
matrix of a 1232-job battery until the container cgroup OOM-killed the
worker mid-run.
"""

from __future__ import annotations

import gc
import weakref

from src.utils.parallel import SingleSlotCache


def test_should_return_cached_value_without_calling_loader_on_a_hit() -> None:
    cache = SingleSlotCache()
    calls: list[str] = []

    first = cache.get_or_load("a", lambda: calls.append("a") or ["value-a"])
    second = cache.get_or_load("a", lambda: calls.append("a") or ["value-a"])

    assert first is second
    assert calls == ["a"]


def test_should_reload_after_a_different_key_evicted_the_entry() -> None:
    cache = SingleSlotCache()
    calls: list[str] = []

    def loader(key: str):
        calls.append(key)
        return [key]

    cache.get_or_load("a", lambda: loader("a"))
    cache.get_or_load("b", lambda: loader("b"))
    cache.get_or_load("a", lambda: loader("a"))

    assert calls == ["a", "b", "a"]


class _Matrix:
    """Weak-referenceable stand-in for a loaded embedding matrix."""


def test_should_release_the_previous_value_when_a_new_key_arrives() -> None:
    cache = SingleSlotCache()
    evicted = weakref.ref(cache.get_or_load("a", lambda: _Matrix()))

    cache.get_or_load("b", lambda: _Matrix())
    gc.collect()

    assert evicted() is None


def test_should_drop_the_previous_value_before_the_loader_allocates() -> None:
    cache = SingleSlotCache()
    evicted = weakref.ref(cache.get_or_load("a", lambda: _Matrix()))
    alive_during_load: list[bool] = []

    def loader():
        gc.collect()
        alive_during_load.append(evicted() is not None)
        return _Matrix()

    cache.get_or_load("b", loader)

    assert alive_during_load == [False]
