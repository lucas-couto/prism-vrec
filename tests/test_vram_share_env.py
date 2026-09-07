"""``PRISM_VRAM_SHARE`` overrides ``resources.gpu.vram_share`` for one launch; bad values fail loud.

Resolved by :func:`resolve_resources` at startup, never at import time,
and applied through :func:`cap_process_vram` to every CUDA entry point.
"""

from __future__ import annotations

import pytest

from src.utils import device
from src.utils.resources import VRAM_SHARE_ENV, resolve_resources

YAML = {"resources": {"gpu": {"vram_share": 0.4}}}


def test_default_is_half_the_card():
    assert resolve_resources({}, env={}).gpu.vram_share == 0.5


def test_yaml_value_applies_when_the_env_is_absent():
    assert resolve_resources(YAML, env={}).gpu.vram_share == 0.4


def test_env_override_wins_over_the_yaml_for_one_launch():
    resolved = resolve_resources(YAML, env={VRAM_SHARE_ENV: "0.95"})

    assert resolved.gpu.vram_share == 0.95
    # Only the run share is overridden; the ranking share stays the YAML's.
    assert resolved.gpu.ranking_vram_share == 0.125


def test_env_is_read_from_the_process_environment_by_default(monkeypatch):
    monkeypatch.setenv(VRAM_SHARE_ENV, "0.7")

    assert resolve_resources(YAML).gpu.vram_share == 0.7


@pytest.mark.parametrize("bad", ["0", "1.5", "-0.2", "abc", "nan", "inf"])
def test_out_of_range_or_unparsable_values_fail_at_resolution(bad):
    with pytest.raises(ValueError, match="PRISM_VRAM_SHARE"):
        resolve_resources(YAML, env={VRAM_SHARE_ENV: bad})


def test_blank_means_the_yaml_value():
    assert resolve_resources(YAML, env={VRAM_SHARE_ENV: "  "}).gpu.vram_share == 0.4


def test_cap_process_vram_splits_the_resolved_share_across_workers(monkeypatch):
    import torch

    applied: list[float] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", applied.append)

    assert device.cap_process_vram(1, vram_share=0.95) == 0.95
    assert device.cap_process_vram(4, vram_share=0.8) == 0.2
    assert applied == [0.95, 0.2]


def test_cap_process_vram_is_a_no_op_without_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert device.cap_process_vram(vram_share=0.5) == 0.0
