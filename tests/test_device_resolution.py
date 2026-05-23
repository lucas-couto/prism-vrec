"""Tests for the device resolver.

``resolve_device`` is what turns the ``device:`` field in
``configs/default.yaml`` into the concrete string the steps pass to
PyTorch.  Researchers never set ``device`` to ``"cpu"`` or ``"cuda"``
explicitly: they keep the default ``"auto"``, and the framework
follows the host.  These tests pin that behaviour.
"""

from __future__ import annotations

import sys
import types

import pytest

from src.utils import device as device_mod


@pytest.fixture
def fake_torch(monkeypatch):
    """Install a fake ``torch`` module exposing only what we need.

    Useful so the tests can exercise both the GPU-visible and the
    GPU-absent branches without depending on the host hardware.
    """

    def _install(cuda_available: bool) -> None:
        fake = types.ModuleType("torch")
        fake.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
        monkeypatch.setitem(sys.modules, "torch", fake)

    return _install


def test_cpu_is_always_honoured_without_touching_torch(monkeypatch):
    """``cpu`` is the one value that must not require torch at all."""
    monkeypatch.setitem(sys.modules, "torch", None)
    assert device_mod.resolve_device("cpu") == "cpu"


def test_auto_picks_cuda_when_available(fake_torch):
    fake_torch(cuda_available=True)
    assert device_mod.resolve_device("auto") == "cuda"


def test_auto_falls_back_to_cpu_when_no_gpu(fake_torch):
    fake_torch(cuda_available=False)
    assert device_mod.resolve_device("auto") == "cpu"


def test_explicit_cuda_falls_back_to_cpu_when_no_gpu(fake_torch):
    """Explicit ``cuda`` must never crash on a host without a GPU."""
    fake_torch(cuda_available=False)
    assert device_mod.resolve_device("cuda") == "cpu"


def test_explicit_cuda_is_honoured_when_gpu_visible(fake_torch):
    fake_torch(cuda_available=True)
    assert device_mod.resolve_device("cuda") == "cuda"


def test_unknown_value_falls_back_to_cpu(fake_torch):
    """A typo in the YAML should not crash the pipeline at runtime."""
    fake_torch(cuda_available=True)
    assert device_mod.resolve_device("mps") == "cpu"


def test_missing_torch_falls_back_to_cpu(monkeypatch):
    """Importing torch can fail in odd test environments (no GPU stack
    installed).  The resolver must degrade gracefully."""

    real_import = (
        __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    )

    def _fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("torch is intentionally unavailable in this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    assert device_mod.resolve_device("auto") == "cpu"
