"""One sampling cap for every PCA fit in the framework.

Fusion strategies, the per-region component pass, the fusion alignment
and the extractors' fixed projection all fit through
``fit_pca_on_rows``.  Capping there rather than at each call site is
what keeps the estimators comparable: a PCA that saw four million rows
and one that saw three hundred thousand are not the same estimator, and
a battery that mixes them is not a fair comparison.

The cap exists because the per-region pass multiplies the fit set by the
region count: on amazon_women it is 291,812 train items x 4 regions =
1,167,248 rows over 2,816 dims, a 13.1 GiB float32 matrix that sklearn
fits in place -- measured against a 16 GiB container on 2026-09-10.
"""

from __future__ import annotations

import numpy as np

from src.fusions.strategies import PCA_FIT_MAX_ROWS, cap_fit_indices, fit_pca_on_rows


class TestFitCap:
    """The cap is on the INDICES, before the fit matrix is assembled.

    Capping the assembled rows is worse than useless: the full matrix is
    already allocated, and the sample adds a copy on top.  Measured on
    amazon_women -- a 21 GiB peak where the uncapped path peaked at 13.1,
    OOM-killed either way (2026-09-10).
    """

    def test_indices_above_the_cap_are_sampled_down(self, monkeypatch):
        monkeypatch.setattr("src.fusions.strategies.PCA_FIT_MAX_ROWS", 200)

        capped = cap_fit_indices(np.arange(1000), 42, "test")

        assert capped.shape[0] == 200

    def test_indices_below_the_cap_are_returned_unchanged(self, monkeypatch):
        monkeypatch.setattr("src.fusions.strategies.PCA_FIT_MAX_ROWS", 2000)
        idx = np.arange(1000)

        assert cap_fit_indices(idx, 42, "test") is idx

    def test_the_sample_is_a_subset_of_the_fit_set(self, monkeypatch):
        """Never an index the caller did not offer -- a leak guard.

        The fit set is the train-only rows; sampling must not invent an
        index outside it.
        """
        monkeypatch.setattr("src.fusions.strategies.PCA_FIT_MAX_ROWS", 200)
        idx = np.arange(500, 1500)

        capped = cap_fit_indices(idx, 42, "test")

        assert np.isin(capped, idx).all()

    def test_the_sample_is_sorted_and_deterministic(self, monkeypatch):
        monkeypatch.setattr("src.fusions.strategies.PCA_FIT_MAX_ROWS", 200)

        first = cap_fit_indices(np.arange(1000), 42, "test")
        second = cap_fit_indices(np.arange(1000), 42, "test")

        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, np.sort(first))

    def test_an_uncapped_fit_warns_instead_of_copying(self, monkeypatch):
        """`fit_pca_on_rows` must not sample: by then the cost is paid.

        The project logger does not propagate, so the warning is caught
        on the module's own logger rather than through `caplog`.
        """
        import src.fusions.strategies as strategies

        monkeypatch.setattr(strategies, "PCA_FIT_MAX_ROWS", 200)
        warnings: list[str] = []
        monkeypatch.setattr(
            strategies.logger, "warning", lambda msg, *a: warnings.append(msg % a if a else msg)
        )
        rows = np.random.default_rng(0).standard_normal((1000, 8)).astype(np.float32)

        pca = fit_pca_on_rows(rows, 4, 42, "test")

        assert pca.n_samples_ == 1000, "the matrix is already allocated; cutting it saves nothing"
        assert any("did not cap its indices" in w for w in warnings)

    def test_the_cap_fits_the_widest_fit_matrix_in_the_battery(self):
        """The cap is a byte budget expressed in rows.

        The widest PCA fit is the per-region concat of the two fusion
        extractors: 2048 + 768 = 2816 float32 columns.  The budget is
        8 GB of host RAM -- the fit is scikit-learn on CPU and never
        touches the GPU.
        """
        widest_dim = 2048 + 768
        peak_bytes = PCA_FIT_MAX_ROWS * widest_dim * 4

        assert peak_bytes <= 8 * 1024**3
