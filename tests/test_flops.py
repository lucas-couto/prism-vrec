"""Tests for analytic FLOP accounting."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.utils import flops, telemetry  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_state():
    flops.reset_for_tests()
    telemetry.reset_for_tests()
    yield
    flops.reset_for_tests()
    telemetry.reset_for_tests()


class _Linear(torch.nn.Module):
    """One 100x200 matmul: 2 * 100 * 200 = 40000 FLOPs per sample."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(100, 200, bias=False)

    def forward(self, x):
        return self.fc(x)


EXPECTED_FLOPS_PER_SAMPLE = 2 * 100 * 200


def test_should_measure_flops_per_sample_for_a_known_matmul():
    model = _Linear()

    measured = flops.calibrate("linear", model, torch.randn(8, 100))

    assert measured == pytest.approx(EXPECTED_FLOPS_PER_SAMPLE)


def test_should_measure_only_once_per_key():
    model = _Linear()
    flops.calibrate("linear", model, torch.randn(4, 100))

    # A model that would raise if called again proves the cache is used.
    def _exploding(*_args):
        raise AssertionError("calibration ran twice for the same key")

    second = flops.calibrate("linear", _exploding, torch.randn(4, 100))

    assert second == pytest.approx(EXPECTED_FLOPS_PER_SAMPLE)


def test_should_scale_recorded_flops_by_item_count():
    flops.calibrate("linear", _Linear(), torch.randn(4, 100))
    telemetry.start({"telemetry": {"enabled": True, "sample_interval_seconds": 0.05}})
    marker = telemetry.mark()

    flops.record("linear", 50)
    summary = telemetry.summarise_since(marker)

    assert summary["throughput"]["total_flops"] == pytest.approx(EXPECTED_FLOPS_PER_SAMPLE * 50)


def test_should_charge_training_batches_the_backward_multiplier():
    flops.calibrate("linear", _Linear(), torch.randn(4, 100))
    telemetry.start({"telemetry": {"enabled": True, "sample_interval_seconds": 0.05}})
    marker = telemetry.mark()

    flops.record("linear", 10, training=True)
    summary = telemetry.summarise_since(marker)

    expected = EXPECTED_FLOPS_PER_SAMPLE * 10 * flops.TRAINING_MULTIPLIER
    assert summary["throughput"]["total_flops"] == pytest.approx(expected)


def test_should_record_nothing_for_an_uncalibrated_key():
    telemetry.start({"telemetry": {"enabled": True, "sample_interval_seconds": 0.05}})
    marker = telemetry.mark()

    flops.record("never-calibrated", 1000)
    summary = telemetry.summarise_since(marker)

    assert "flops_per_s" not in summary.get("throughput", {})


def test_should_return_none_when_the_model_cannot_be_probed():
    def _broken(_x):
        raise RuntimeError("no forward here")

    assert flops.calibrate("broken", _broken, torch.randn(2, 100)) is None


def test_should_return_none_for_a_model_that_dispatches_no_counted_ops():
    # Elementwise-only work is deliberately not attributed; see the module
    # docstring on why a factorisation recommender reports no FLOPs.
    class _Elementwise(torch.nn.Module):
        def forward(self, x):
            return x * 2 + 1

    assert flops.calibrate("elementwise", _Elementwise(), torch.randn(4, 100)) is None


def test_should_splat_tuple_inputs_for_multi_argument_forwards():
    class _TwoArg(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(100, 200, bias=False)

        def forward(self, a, b):
            return self.fc(a) + self.fc(b)

    measured = flops.calibrate("two-arg", _TwoArg(), (torch.randn(4, 100), torch.randn(4, 100)))

    assert measured == pytest.approx(EXPECTED_FLOPS_PER_SAMPLE * 2)


def test_should_expose_calibrations_for_the_manifest():
    flops.calibrate("linear", _Linear(), torch.randn(2, 100))

    assert flops.calibrated()["linear"] == pytest.approx(EXPECTED_FLOPS_PER_SAMPLE)


def test_should_probe_an_uncountable_model_only_once():
    # Regression: a model the counter cannot attribute (elementwise-only,
    # e.g. matrix factorisation) must not re-enter FlopCounterMode on every
    # batch — that is the per-batch overhead analytic accounting avoids.
    calls = {"n": 0}

    class _Elementwise(torch.nn.Module):
        def forward(self, x):
            calls["n"] += 1
            return x * 2

    model = _Elementwise()
    for _ in range(5):
        flops.calibrate("elementwise", model, torch.randn(4, 100))

    assert calls["n"] == 1


def test_should_probe_a_broken_model_only_once():
    calls = {"n": 0}

    def _broken(_x):
        calls["n"] += 1
        raise RuntimeError("no forward here")

    for _ in range(5):
        flops.calibrate("broken", _broken, torch.randn(2, 100))

    assert calls["n"] == 1


def test_should_not_disturb_batchnorm_statistics_during_calibration():
    # Regression: the probe forward must not update running stats, or
    # calibrating mid-training would silently perturb the run it measures.
    model = torch.nn.Sequential(torch.nn.Linear(100, 200, bias=False), torch.nn.BatchNorm1d(200))
    model.train()
    before = model[1].running_mean.clone()

    flops.calibrate("with-bn", model, torch.randn(8, 100))

    assert torch.equal(model[1].running_mean, before)


def test_should_restore_training_mode_after_calibration():
    model = _Linear()
    model.train()

    flops.calibrate("restore-mode", model, torch.randn(4, 100))

    assert model.training is True


def test_should_preserve_mixed_train_eval_modes_across_calibration():
    # Fine-tuning runs the backbone in train mode while frozen BatchNorm
    # layers stay in eval. A blanket model.train() on restore would
    # silently unfreeze them, so restoration must be per submodule.
    model = torch.nn.Sequential(torch.nn.Linear(100, 200, bias=False), torch.nn.BatchNorm1d(200))
    model.train()
    frozen_bn = model[1]
    frozen_bn.eval()

    flops.calibrate("mixed-mode", model, torch.randn(8, 100))

    assert model[0].training is True
    assert frozen_bn.training is False
