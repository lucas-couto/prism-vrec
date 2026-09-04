"""The training GPU is also the display GPU: it must never be claimed whole.

A process that caps itself at 1.0 starves the compositor, which freezes
the workstation while a battery runs (and tripped the display driver's
watchdog on 2026-09-01).
"""

from __future__ import annotations

import ast
from pathlib import Path

from src.evaluation.protocol import _DEFAULT_RANKING_VRAM_FRACTION
from src.utils.device import POOL_VRAM_FRACTION, SOLO_PROCESS_VRAM_FRACTION
from src.utils.parallel import _RANKING_VRAM_SHARE

#: Every entry point that owns a CUDA process for a long stretch and so
#: must cap itself.  Trained workers aside, these run standalone.
_CAPPED_ENTRY_POINTS = (
    "src/steps/extract.py",
    "src/steps/evaluate.py",
    "src/steps/train.py",
    "src/utils/parallel.py",
)


def _planned_fraction(n_workers: int) -> float:
    """Mirror of the fraction :func:`cap_process_vram` applies."""
    return POOL_VRAM_FRACTION / n_workers if n_workers > 1 else SOLO_PROCESS_VRAM_FRACTION


def test_should_leave_vram_headroom_for_a_solo_process() -> None:
    assert _planned_fraction(1) < 1.0


def test_should_keep_the_sum_of_worker_caps_below_the_card() -> None:
    for n_workers in (1, 2, 3, 4, 8):
        assert _planned_fraction(n_workers) * n_workers <= POOL_VRAM_FRACTION


def test_should_size_the_ranking_budget_below_half_the_process_allowance() -> None:
    assert 0 < _RANKING_VRAM_SHARE <= 0.25
    assert 0 < _DEFAULT_RANKING_VRAM_FRACTION < 0.5


def test_should_cap_every_long_lived_cuda_entry_point() -> None:
    for module_path in _CAPPED_ENTRY_POINTS:
        source = Path(module_path).read_text(encoding="utf-8")
        calls = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "cap_process_vram"
        ]

        assert calls, f"{module_path} never caps its CUDA process"


def test_should_not_set_the_memory_fraction_outside_the_shared_helper() -> None:
    offenders = [
        path
        for path in Path("src").rglob("*.py")
        if path.name != "device.py"
        and "set_per_process_memory_fraction(" in path.read_text(encoding="utf-8")
    ]

    assert offenders == [], f"cap VRAM through cap_process_vram(), not directly: {offenders}"
