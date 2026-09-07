"""The training GPU is also the display GPU: it must never be claimed whole.

A process that caps itself at 1.0 starves the compositor, which freezes
the workstation while a battery runs (and tripped the display driver's
watchdog on 2026-09-01).  The shares live in ``configs/resources.yaml``;
these tests read the shipped file so a careless edit there fails here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import yaml

from src.evaluation.protocol import _DEFAULT_RANKING_VRAM_FRACTION
from src.utils.resources import resolve_resources

#: Every entry point that owns a CUDA process for a long stretch and so
#: must cap itself.  Trained workers aside, these run standalone.
_CAPPED_ENTRY_POINTS = (
    "src/steps/extract.py",
    "src/steps/evaluate.py",
    "src/steps/train.py",
    "src/utils/parallel.py",
)


def _shipped():
    raw = yaml.safe_load(Path("configs/resources.yaml").read_text(encoding="utf-8"))
    return resolve_resources(raw, env={})


def _planned_fraction(n_workers: int) -> float:
    """Mirror of the fraction :func:`cap_process_vram` applies."""
    return _shipped().gpu.vram_share / max(1, n_workers)


def test_should_leave_vram_headroom_for_a_solo_process() -> None:
    # Never the whole card: a CUDA context and the compositor live outside the cap.
    assert _planned_fraction(1) < 1.0
    assert resolve_resources(None, env={}).gpu.vram_share == 0.5  # desktop-friendly default


def test_should_hold_every_worker_count_to_the_run_resource_share() -> None:
    share = _shipped().gpu.vram_share
    for n_workers in (1, 2, 3, 4, 8):
        assert _planned_fraction(n_workers) * n_workers <= share


def test_should_size_the_ranking_budget_below_half_the_process_allowance() -> None:
    assert 0 < _shipped().gpu.ranking_vram_share <= 0.25
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
        for call in calls:
            assert any(kw.arg == "vram_share" for kw in call.keywords), (
                f"{module_path}: cap_process_vram must receive resources.gpu.vram_share"
            )


def test_should_not_set_the_memory_fraction_outside_the_shared_helper() -> None:
    offenders = [
        path
        for path in Path("src").rglob("*.py")
        if path.name != "device.py"
        and "set_per_process_memory_fraction(" in path.read_text(encoding="utf-8")
    ]

    assert offenders == [], f"cap VRAM through cap_process_vram(), not directly: {offenders}"


def test_should_not_read_the_vram_share_env_at_import_time() -> None:
    offenders = [
        path
        for path in Path("src").rglob("*.py")
        if path.name != "resources.py"
        and "PRISM_VRAM_SHARE" in path.read_text(encoding="utf-8")
        and "os.environ" in path.read_text(encoding="utf-8")
    ]

    assert offenders == [], f"PRISM_VRAM_SHARE is resolved only by resolve_resources: {offenders}"
