"""Tests for the "step did no new work" verdict in ``main._run_step``.

Re-running a finished pipeline used to append a full telemetry window
per step, attributing seconds of existence checks — and zero energy — to
an extraction that actually ran for an hour on a previous run.  A step
whose cells were all skipped is now recorded as ``skipped`` with no
telemetry, while steps that emit no cells at all (``preprocess``,
``report``) keep being timed as usual.
"""

from __future__ import annotations

import main


class TestDidNoNewWork:
    def test_all_cells_skipped_is_no_new_work(self):
        # 3 cells entered, all of them skipped.
        assert main._did_no_new_work((0, 0), (0, 3)) is True

    def test_one_recorded_cell_means_work_happened(self):
        assert main._did_no_new_work((0, 0), (1, 5)) is False

    def test_step_without_cells_is_never_marked_skipped(self):
        """``preprocess`` and ``report`` emit no cells; they must be timed."""
        assert main._did_no_new_work((7, 2), (7, 2)) is False

    def test_counts_are_relative_to_the_step_window(self):
        """Cells from earlier steps must not leak into this step's verdict."""
        # Earlier steps left 4 recorded / 6 skipped; this step added
        # only skipped ones.
        assert main._did_no_new_work((4, 6), (4, 9)) is True
        # ...and here it added a real one on top of the skipped ones.
        assert main._did_no_new_work((4, 6), (5, 9)) is False
