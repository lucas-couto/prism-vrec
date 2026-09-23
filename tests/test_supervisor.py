"""A CUDA context fault resumes the SAME run in a fresh child process.

The supervisor is exercised with real child processes (``python -c``)
that play the pipeline's side of the protocol: record the run directory,
report progress, exit with the fault code.  No GPU is involved.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

from src import supervisor

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Child that records one attempt per line in ``attempts.txt`` and then
#: runs the body given to :func:`_child`.
_PRELUDE = """
import os, signal, sys
from pathlib import Path
from src import supervisor
n = supervisor.attempt()
with open("attempts.txt", "a") as fh:
    fh.write(f"{n} {os.environ['PRISM_RUN_ID']} {supervisor.restart_reason()}\\n")
"""


def _child(body: str) -> list[str]:
    return [sys.executable, "-c", _PRELUDE + body]


def _supervise(tmp_path: Path, monkeypatch, body: str) -> int:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(PROJECT_ROOT))
    monkeypatch.delenv(supervisor.ENV_SUPERVISED, raising=False)
    monkeypatch.delenv(supervisor.ENV_RUN_ID, raising=False)
    return supervisor.supervise(_child(body), delay_s=0, sleep=lambda _s: None)


def _attempts(tmp_path: Path) -> list[list[str]]:
    return [line.split(" ", 2) for line in (tmp_path / "attempts.txt").read_text().splitlines()]


class TestRetryableExits:
    def test_should_retry_the_cuda_fault_exit_code(self) -> None:
        assert supervisor.is_retryable(supervisor.EXIT_CUDA_FAULT)

    def test_should_retry_a_kill_by_sigkill(self) -> None:
        assert supervisor.is_retryable(-signal.SIGKILL)

    @pytest.mark.parametrize("code", [0, 1, 2, 130, -signal.SIGTERM, -signal.SIGINT])
    def test_should_not_retry_success_errors_or_stop_signals(self, code: int) -> None:
        assert not supervisor.is_retryable(code)


class TestSupervise:
    def test_should_resume_the_same_run_after_a_cuda_fault(self, tmp_path, monkeypatch) -> None:
        body = """
if n == 1:
    supervisor.remember_run_dir("single", Path("results/runs/r1"))
    supervisor.note_progress()
    supervisor.note_fault("CUDA error: the launch timed out and was terminated")
    sys.exit(supervisor.EXIT_CUDA_FAULT)
assert supervisor.recorded_run_dir("single") == Path("results/runs/r1").resolve()
"""

        code = _supervise(tmp_path, monkeypatch, body)

        assert code == 0
        attempts = _attempts(tmp_path)
        assert [a[0] for a in attempts] == ["1", "2"]
        assert attempts[0][1] == attempts[1][1]
        assert "launch timed out" in attempts[1][2]

    def test_should_append_the_restart_to_the_session_log(self, tmp_path, monkeypatch) -> None:
        body = "sys.exit(supervisor.EXIT_CUDA_FAULT if n == 1 else 0)"

        _supervise(tmp_path, monkeypatch, body)

        run_id = _attempts(tmp_path)[0][1]
        log = (tmp_path / "logs" / f"run_{run_id}.log").read_text()
        assert "resuming the same run as attempt 2" in log

    def test_should_not_retry_an_ordinary_failure(self, tmp_path, monkeypatch) -> None:
        code = _supervise(tmp_path, monkeypatch, "sys.exit(1)")

        assert code == 1
        assert len(_attempts(tmp_path)) == 1

    def test_should_retry_a_child_killed_by_sigkill(self, tmp_path, monkeypatch) -> None:
        body = "os.kill(os.getpid(), signal.SIGKILL) if n == 1 else sys.exit(0)"

        code = _supervise(tmp_path, monkeypatch, body)

        assert code == 0
        assert len(_attempts(tmp_path)) == 2

    def test_should_give_up_when_restarts_complete_no_new_work(self, tmp_path, monkeypatch) -> None:
        code = _supervise(tmp_path, monkeypatch, "sys.exit(supervisor.EXIT_CUDA_FAULT)")

        assert code == supervisor.EXIT_CUDA_FAULT
        assert len(_attempts(tmp_path)) == supervisor.MAX_RESTARTS_WITHOUT_PROGRESS + 1

    def test_should_keep_retrying_while_each_attempt_makes_progress(
        self, tmp_path, monkeypatch
    ) -> None:
        total = supervisor.MAX_RESTARTS_WITHOUT_PROGRESS + 3
        body = f"""
supervisor.note_progress()
sys.exit(supervisor.EXIT_CUDA_FAULT if n < {total} else 0)
"""

        code = _supervise(tmp_path, monkeypatch, body)

        assert code == 0
        assert len(_attempts(tmp_path)) == total

    def test_should_leave_no_state_directory_behind(self, tmp_path, monkeypatch) -> None:
        body = "Path('state.txt').write_text(os.environ['PRISM_SUPERVISOR_STATE'])"

        _supervise(tmp_path, monkeypatch, body)

        assert not Path((tmp_path / "state.txt").read_text()).exists()


class TestInterrupt:
    """Ctrl+C in ``docker attach`` must not end a run; SIGTERM still does."""

    _BODY = """
import time
Path("started").touch()
time.sleep(20)
"""

    def _supervise_with_signals(self, tmp_path, monkeypatch, signals, *, ignore: bool) -> int:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("PYTHONPATH", str(PROJECT_ROOT))
        monkeypatch.delenv(supervisor.ENV_SUPERVISED, raising=False)

        def send_once_started() -> None:
            while not (tmp_path / "started").exists():
                time.sleep(0.05)
            for signum in signals:
                os.kill(os.getpid(), signum)
                time.sleep(0.5)

        threading.Thread(target=send_once_started, daemon=True).start()
        return supervisor.supervise(
            _child(self._BODY), delay_s=0, sleep=lambda _s: None, ignore_interrupt=ignore
        )

    def test_should_ignore_ctrl_c_inside_the_container(self, tmp_path, monkeypatch, capfd) -> None:
        signals = [signal.SIGINT, signal.SIGTERM]

        code = self._supervise_with_signals(tmp_path, monkeypatch, signals, ignore=True)

        # Only the SIGTERM sent half a second later ended the child.
        assert code == 128 + signal.SIGTERM
        assert "Ctrl+C ignored" in capfd.readouterr().err
        assert len(_attempts(tmp_path)) == 1

    def test_should_stop_on_ctrl_c_outside_the_container(self, tmp_path, monkeypatch) -> None:
        code = self._supervise_with_signals(tmp_path, monkeypatch, [signal.SIGINT], ignore=False)

        assert code != 0
        assert len(_attempts(tmp_path)) == 1


class TestChildHelpersUnsupervised:
    def test_should_be_inert_without_a_supervisor(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv(supervisor.ENV_SUPERVISED, raising=False)
        monkeypatch.setenv(supervisor.ENV_STATE_DIR, str(tmp_path))

        supervisor.remember_run_dir("single", tmp_path / "run")
        supervisor.note_progress()

        assert supervisor.recorded_run_dir("single") is None
        assert supervisor.attempt() == 1
        assert os.listdir(tmp_path) == []
