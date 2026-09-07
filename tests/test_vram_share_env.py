"""``PRISM_VRAM_SHARE`` sets the run's VRAM share; bad values fail loud."""

from __future__ import annotations

import importlib

import pytest

import src.utils.device as device


def _reload(monkeypatch, value: str | None):
    if value is None:
        monkeypatch.delenv(device.RUN_RESOURCE_SHARE_ENV, raising=False)
    else:
        monkeypatch.setenv(device.RUN_RESOURCE_SHARE_ENV, value)
    return importlib.reload(device)


@pytest.fixture(autouse=True)
def _restore(monkeypatch):
    yield
    monkeypatch.delenv(device.RUN_RESOURCE_SHARE_ENV, raising=False)
    importlib.reload(device)


def test_default_is_half_the_card(monkeypatch):
    mod = _reload(monkeypatch, None)

    assert mod.RUN_RESOURCE_SHARE == mod.DEFAULT_RUN_RESOURCE_SHARE == 0.5
    assert mod.SOLO_PROCESS_VRAM_FRACTION == 0.5


def test_env_override_applies_to_every_cap(monkeypatch):
    mod = _reload(monkeypatch, "0.95")

    assert mod.RUN_RESOURCE_SHARE == 0.95
    assert mod.SOLO_PROCESS_VRAM_FRACTION == 0.95
    assert mod.POOL_VRAM_FRACTION == 0.95


@pytest.mark.parametrize("bad", ["0", "1.5", "-0.2", "abc", "nan"])
def test_out_of_range_or_unparsable_values_fail_at_import(monkeypatch, bad):
    with pytest.raises(ValueError, match="PRISM_VRAM_SHARE"):
        _reload(monkeypatch, bad)


def test_blank_means_default(monkeypatch):
    mod = _reload(monkeypatch, "  ")

    assert mod.RUN_RESOURCE_SHARE == 0.5
