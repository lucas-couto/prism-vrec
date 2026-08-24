"""D4: battery evaluation honours the ``evaluation:`` config block.

The battery executor and the sequential evaluate step must construct
their final-evaluation ``Evaluator`` through the same path
(``src.steps.evaluate.build_evaluator``): same protocol / n_negatives /
negative_sampling_seed handling with identical defaults, and
``tiebreak_seed`` = the run's ACTIVE seed (``config['seed']``, which the
battery sets to the cell's seed).
"""

from __future__ import annotations

import numpy as np

import src.battery.execute as bx
import src.steps.evaluate as ev
from src.battery.cells import BatteryCell
from src.evaluation.protocol import Evaluator
from src.steps.evaluate import build_evaluator

_SEEN = {0: {0}}
_TEST = {0: {1}}
_N_ITEMS = 30


def _config(seed: int = 7, evaluation: dict | None = None) -> dict:
    cfg: dict = {
        "seed": seed,
        "k_values": [5],
        "paths": {"data_processed": "p", "embeddings": "e", "results": "r"},
    }
    if evaluation is not None:
        cfg["evaluation"] = evaluation
    return cfg


class TestBuildEvaluator:
    def test_defaults_match_the_sequential_step(self) -> None:
        e = build_evaluator(_config(), _SEEN, _TEST, _N_ITEMS)

        assert e.protocol == "full_ranking"
        assert e.n_negatives == 100
        assert e.negative_sampling_seed == 42
        assert e.k_values == [5]

    def test_reads_the_evaluation_block(self) -> None:
        cfg = _config(
            evaluation={"protocol": "sampled", "n_negatives": 33, "negative_sampling_seed": 5}
        )

        e = build_evaluator(cfg, _SEEN, _TEST, _N_ITEMS)

        assert e.protocol == "sampled"
        assert e.n_negatives == 33
        assert e.negative_sampling_seed == 5

    def test_tiebreak_seed_is_the_active_seed(self) -> None:
        e = build_evaluator(_config(seed=13), _SEEN, _TEST, _N_ITEMS)
        ref = Evaluator(_SEEN, _TEST, _N_ITEMS, k_values=[5], tiebreak_seed=13)

        assert np.array_equal(e._tiebreak_key, ref._tiebreak_key)


class TestBatteryUsesSharedEvaluator:
    def test_battery_evaluator_honours_evaluation_block(self, monkeypatch) -> None:
        captured: dict = {}
        monkeypatch.setattr(ev, "load_data", lambda p, d: (2, _N_ITEMS, _SEEN, _TEST, {0: {0}}))
        monkeypatch.setattr(
            ev,
            "find_best_models",
            lambda d, results_dir: [
                {"model_name": "bpr", "embedding_name": "none", "path": "x.pt"}
            ],
        )

        def _capture(model_info, dataset, n_users, n_items, evaluator, *a, **k):
            captured["evaluator"] = evaluator
            captured["seed"] = k.get("seed")

        monkeypatch.setattr(ev, "_evaluate_cell", _capture)

        cell = BatteryCell("amazon_x", "none", "bpr", seed=13, role="replay")
        # execute_cell sets cfg['seed'] = cell.seed before evaluating;
        # _evaluate_one_cell receives that per-cell config.
        cfg = _config(
            seed=cell.seed,
            evaluation={"protocol": "sampled", "n_negatives": 33, "negative_sampling_seed": 5},
        )

        bx._evaluate_one_cell(cell, cfg, 2, _N_ITEMS, None, "cpu", f_out_dir="out")

        e = captured["evaluator"]
        assert e.protocol == "sampled"
        assert e.n_negatives == 33
        assert e.negative_sampling_seed == 5
        assert captured["seed"] == 13

    def test_battery_tiebreak_matches_the_cell_seed(self, monkeypatch) -> None:
        captured: dict = {}
        monkeypatch.setattr(ev, "load_data", lambda p, d: (2, _N_ITEMS, _SEEN, _TEST, {0: {0}}))
        monkeypatch.setattr(
            ev,
            "find_best_models",
            lambda d, results_dir: [
                {"model_name": "bpr", "embedding_name": "none", "path": "x.pt"}
            ],
        )
        monkeypatch.setattr(
            ev,
            "_evaluate_cell",
            lambda mi, ds, nu, ni, evaluator, *a, **k: captured.setdefault("evaluator", evaluator),
        )

        cell = BatteryCell("amazon_x", "none", "bpr", seed=21, role="replay")
        bx._evaluate_one_cell(
            cell, _config(seed=cell.seed), 2, _N_ITEMS, None, "cpu", f_out_dir="out"
        )

        ref = Evaluator(_SEEN, _TEST, _N_ITEMS, k_values=[5], tiebreak_seed=21)
        assert np.array_equal(captured["evaluator"]._tiebreak_key, ref._tiebreak_key)
