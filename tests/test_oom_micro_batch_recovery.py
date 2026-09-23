"""OOM recovery by gradient accumulation, and its audit trail.

Three guarantees:

* ``bpr_accumulated_step`` takes the SAME optimiser step as ``bpr_step``
  for every built-in recommender (the full-batch objective is a mean
  over triples plus a shared term, so ``n_k / N`` weights reproduce it);
* the escalation is lazy reads first, then twice the micro-batches per
  retry, on BOTH execution paths (grid pool and fold runner);
* every retry, recovery and final failure is a row of
  ``results/runs/<run>/oom_recoveries.csv``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from src.recommenders.bpr import BPR
from src.utils import oom_recoveries
from src.utils.amp_compat import get_grad_scaler
from src.utils.oom_recoveries import escalate
from src.utils.training import bpr_accumulated_step, bpr_step
from tests.recommenders.test_lazy_feature_equivalence import (
    CASES,
    N_ITEMS,
    N_USERS,
    _build,
    _variant,
)

USERS = torch.tensor([0, 3, 3, 6, 1, 2, 5, 0, 4, 6, 1])
POS = torch.tensor([2, 9, 9, 22, 4, 7, 1, 5, 3, 0, 4])
NEG = torch.tensor([9, 0, 15, 4, 11, 8, 17, 6, 20, 13, 2])


@pytest.fixture(autouse=True)
def _clean_recorder():
    oom_recoveries.reset_for_tests()
    yield
    oom_recoveries.reset_for_tests()


def _twins(model_cls, kind, tmp_path: Path):
    if model_cls is BPR:
        torch.manual_seed(0)
        a = BPR(N_USERS, N_ITEMS, None, {"latent_dim": 4, "l2_reg": 1e-3})
        b = BPR(N_USERS, N_ITEMS, None, {"latent_dim": 4, "l2_reg": 1e-3})
    else:
        dense, _ = _variant(kind, tmp_path)
        a, b = _build(model_cls, dense), _build(model_cls, dense)
    b.load_state_dict(a.state_dict())
    return a, b


STEP_CASES = [(BPR, "none"), *CASES]
STEP_IDS = [f"{cls.__name__}-{kind}" for cls, kind in STEP_CASES]


@pytest.mark.parametrize("micro_batches", [2, 3, 4])
@pytest.mark.parametrize(("model_cls", "kind"), STEP_CASES, ids=STEP_IDS)
def test_accumulated_step_matches_the_full_batch_step(model_cls, kind, micro_batches, tmp_path):
    full, split = _twins(model_cls, kind, tmp_path)
    full.train()
    split.train()
    opt_full = torch.optim.SGD(full.parameters(), lr=0.5)
    opt_split = torch.optim.SGD(split.parameters(), lr=0.5)
    common = {"device": "cpu", "use_cuda": False}

    loss_full = bpr_step(full, opt_full, get_grad_scaler(enabled=False), USERS, POS, NEG, **common)
    loss_split = bpr_accumulated_step(
        split,
        opt_split,
        get_grad_scaler(enabled=False),
        USERS,
        POS,
        NEG,
        micro_batches=micro_batches,
        **common,
    )

    torch.testing.assert_close(loss_split, loss_full, rtol=1e-5, atol=1e-6)
    for (name, p_full), (_, p_split) in zip(
        full.named_parameters(), split.named_parameters(), strict=True
    ):
        torch.testing.assert_close(p_split, p_full, rtol=1e-5, atol=1e-6, msg=name)


def test_one_micro_batch_is_the_plain_step(tmp_path):
    full, split = _twins(BPR, "none", tmp_path)
    common = {"device": "cpu", "use_cuda": False}

    bpr_step(full, torch.optim.SGD(full.parameters(), lr=0.5), get_grad_scaler(enabled=False),
             USERS, POS, NEG, **common)  # fmt: skip
    bpr_accumulated_step(split, torch.optim.SGD(split.parameters(), lr=0.5),
                         get_grad_scaler(enabled=False), USERS, POS, NEG,
                         micro_batches=1, **common)  # fmt: skip

    for p_full, p_split in zip(full.parameters(), split.parameters(), strict=True):
        assert torch.equal(p_full, p_split)


def test_escalation_goes_lazy_first_then_halves_the_micro_batch_size():
    first = escalate(lazy_features=False, micro_batches=1)
    second = escalate(lazy_features=first.lazy_features, micro_batches=first.micro_batches)
    third = escalate(lazy_features=second.lazy_features, micro_batches=second.micro_batches)

    assert (first.lazy_features, first.micro_batches, first.action) == (True, 1, "lazy_features")
    assert (second.micro_batches, second.action) == (2, "micro_batches")
    assert third.micro_batches == 4


# ------------------------------------------------------------ grid pool path
def _job(**kwargs):
    from src.utils.parallel import TrainingJob

    return TrainingJob(
        dataset_name="ds",
        model_name="acf",
        embedding_name="resnet50_comp",
        hyperparams={"learning_rate": 0.001},
        n_users=4,
        n_items=8,
        embeddings_path=None,
        processed_dir="p",
        device="cuda",
        **kwargs,
    )


def _read_csv(run_dir: Path) -> list[dict]:
    with (run_dir / oom_recoveries.FILENAME).open() as handle:
        return list(csv.DictReader(handle))


def test_an_already_lazy_job_retries_with_more_micro_batches_and_is_recorded(tmp_path):
    from src.utils.parallel import _JobRegistry

    oom_recoveries.bind_run_dir(tmp_path)
    job = _job(lazy_features=True)
    registry = _JobRegistry([job])

    registry._record_oom(job, 1, {"error": "CUDA out of memory. Tried to allocate 9 GiB\nmore"})
    assert job.micro_batches == 2
    registry.take_retries()
    registry._record_oom(job, 2, {"error": "CUDA out of memory"})
    assert job.micro_batches == 4
    registry.take_retries()
    registry.record({"job_id": job.job_id, "status": "ok", "attempt": 3, "best_metric": 0.1})

    rows = _read_csv(tmp_path)
    assert [r["outcome"] for r in rows] == ["retrying", "retrying", "recovered"]
    assert [r["micro_batches"] for r in rows] == ["2", "4", "4"]
    assert [r["attempt"] for r in rows] == ["1", "2", "3"]
    assert rows[0]["action"] == "micro_batches"
    assert rows[0]["error"] == "CUDA out of memory. Tried to allocate 9 GiB"
    assert rows[1]["ranking_budget_factor"] == "0.25"
    assert rows[0]["step"] == "train" and rows[0]["model"] == "acf"


def test_a_job_that_exhausts_its_retries_records_the_failure(tmp_path):
    from src.utils.parallel import MAX_OOM_RETRIES, _JobRegistry

    oom_recoveries.bind_run_dir(tmp_path)
    job = _job(lazy_features=True)
    registry = _JobRegistry([job])

    for attempt in range(1, MAX_OOM_RETRIES + 2):
        registry._record_oom(job, attempt, {"error": "CUDA out of memory"})
        registry.take_retries()

    outcomes = [r["outcome"] for r in _read_csv(tmp_path)]
    assert outcomes == ["retrying"] * MAX_OOM_RETRIES + ["failed"]


def test_a_job_that_never_ooms_writes_no_recovery_file(tmp_path):
    from src.utils.parallel import _JobRegistry

    oom_recoveries.bind_run_dir(tmp_path)
    job = _job()
    registry = _JobRegistry([job])

    registry.record({"job_id": job.job_id, "status": "ok", "attempt": 1, "best_metric": 0.1})

    assert not (tmp_path / oom_recoveries.FILENAME).exists()


# ------------------------------------------------------------ fold runner path
def _cell():
    from src.battery.cells import BatteryCell

    return BatteryCell(
        dataset="amazon_men",
        visual_config="hybrid_concat_comp",
        recommender="acf",
        seed=42,
        role="search",
    )


def test_the_fold_runner_escalates_to_micro_batches_and_records_it(tmp_path):
    from src.folds.runner import _micro_batches_of, _run_cell_with_oom_recovery

    oom_recoveries.bind_run_dir(tmp_path)
    seen: list[dict] = []

    def runner(cell, config, plan, frames, *, results_dir, device):
        seen.append(config)
        if len(seen) < 3:
            raise torch.cuda.OutOfMemoryError("tried to allocate 8.59 GiB")
        return {"ok": True}

    config = {"resources": {"features": {"residency": "lazy"}}, "seed": 7}

    out = _run_cell_with_oom_recovery(
        runner, _cell(), config, None, None, results_dir="r", device="cuda"
    )

    assert out == {"ok": True}
    assert [_micro_batches_of(c) for c in seen] == [1, 2, 4]
    rows = _read_csv(tmp_path)
    assert [r["outcome"] for r in rows] == ["retrying", "retrying", "recovered"]
    assert rows[0]["step"] == "folds" and rows[0]["job"] == _cell().key()
    assert "_oom_recovery" not in config, "caller's config mutated"


def test_the_fold_runner_fails_after_its_retries_and_records_it(tmp_path):
    from src.folds.runner import _run_cell_with_oom_recovery

    oom_recoveries.bind_run_dir(tmp_path)

    def runner(cell, config, plan, frames, *, results_dir, device):
        raise torch.cuda.OutOfMemoryError("tried to allocate 8.59 GiB")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        _run_cell_with_oom_recovery(
            runner, _cell(), {"resources": {"features": {"residency": "auto"}}}, None, None,
            results_dir="r", device="cuda",
        )  # fmt: skip

    rows = _read_csv(tmp_path)
    assert [r["action"] for r in rows[:2]] == ["lazy_features", "micro_batches"]
    assert rows[-1]["outcome"] == "failed"


def test_events_recorded_before_the_bind_are_written_at_bind(tmp_path):
    oom_recoveries.record_recovery(step="train", job="x", attempt=1, outcome="retrying")

    oom_recoveries.bind_run_dir(tmp_path)

    assert [r["job"] for r in _read_csv(tmp_path)] == ["x"]
