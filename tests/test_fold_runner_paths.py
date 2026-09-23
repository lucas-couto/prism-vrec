"""Fold roots must stay inside the configured results / checkpoints trees.

Found 2026-09-09 on the first real fold run: the roots were derived as
SIBLINGS (``results_fold0`` next to ``results``), which lands outside
every bind mount a container is given, so every one of the 75 cells died
with ``Permission denied`` on the image's read-only working directory.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from src.folds.runner import _fold_paths


def _config() -> dict:
    return {"paths": {"results": "results", "checkpoints": "checkpoints", "embeddings": "emb"}}


def test_fold_roots_are_nested_under_the_configured_roots() -> None:
    paths = _fold_paths(_config(), 0)

    assert PurePosixPath(paths["results"]).parts[0] == "results"
    assert PurePosixPath(paths["checkpoints"]).parts[0] == "checkpoints"


def test_each_fold_gets_its_own_root() -> None:
    first, second = _fold_paths(_config(), 0), _fold_paths(_config(), 1)

    assert first["results"] != second["results"]
    assert first["checkpoints"] != second["checkpoints"]


def test_the_other_paths_are_carried_through_untouched() -> None:
    paths = _fold_paths(_config(), 3)

    assert paths["embeddings"] == "emb"


class TestFoldCellOomRecovery:
    """A fold cell that dies allocating must be retried with lazy reads.

    Found 2026-09-09: the training pool escalates such a job, the fold
    runner did not, so the ACF cells the pool had rescued died here and
    failed the whole K-fold run.
    """

    def _cell(self):
        from src.battery.cells import BatteryCell

        return BatteryCell(
            dataset="amazon_men",
            visual_config="hybrid_concat_comp",
            recommender="acf",
            seed=42,
            role="search",
        )

    def test_the_retry_forces_lazy_and_keeps_everything_else(self) -> None:
        import torch

        from src.folds.runner import _run_cell_with_oom_recovery

        seen: list[dict] = []

        def runner(cell, config, plan, frames, *, results_dir, device):
            seen.append(config)
            if len(seen) == 1:
                raise torch.cuda.OutOfMemoryError("tried to allocate 8.59 GiB")
            return {"ok": True}

        config = {"resources": {"features": {"residency": "auto"}}, "seed": 7}

        out = _run_cell_with_oom_recovery(
            runner, self._cell(), config, None, None, results_dir="r", device="cuda"
        )

        assert out == {"ok": True}
        assert seen[0]["resources"]["features"]["residency"] == "auto"
        assert seen[1]["resources"]["features"]["residency"] == "lazy"
        assert seen[1]["seed"] == 7
        assert config["resources"]["features"]["residency"] == "auto", "caller's config mutated"

    def test_a_cell_that_succeeds_is_not_retried(self) -> None:
        from src.folds.runner import _run_cell_with_oom_recovery

        calls = []

        def runner(cell, config, plan, frames, *, results_dir, device):
            calls.append(config)
            return {"ok": True}

        _run_cell_with_oom_recovery(
            runner, self._cell(), {}, None, None, results_dir="r", device="cpu"
        )

        assert len(calls) == 1
