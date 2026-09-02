"""Guard: the DVBPR split parser must keep item id 0.

The raw partitions store each interaction as ``{"productid": <int>}``.
A truthiness test on the id (``entry.get(...) or ...``) discarded
product 0 from every split, and a user whose held-out was item 0
silently disappeared from val/test.  Measured on 2026-09-02: 55
amazon_men users lost item 0 and 5 of them vanished from test.csv.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.dvbpr import DVBPRDataLoader, _product_id


def _raw(train: dict, val: dict, test: dict, n_users: int, n_items: int) -> np.ndarray:
    """Shape a partition exactly like the DVBPR ``.npy`` payload."""
    wrap = lambda split: {u: [{"productid": i} for i in items] for u, items in split.items()}  # noqa: E731
    return np.array([wrap(train), wrap(val), wrap(test), None, n_users, n_items], dtype=object)


class TestProductId:
    @pytest.mark.parametrize("entry", [{"productid": 0}, {b"productid": 0}, 0, np.int64(0)])
    def test_should_return_zero_for_item_zero(self, entry) -> None:
        assert _product_id(entry) == 0

    def test_should_return_none_when_key_is_absent(self) -> None:
        assert _product_id({"other": 3}) is None


class TestLoadSplitsKeepsItemZero:
    def test_should_keep_item_zero_in_train_val_and_test(self, monkeypatch, tmp_path) -> None:
        loader = DVBPRDataLoader("amazon_men", raw_dir=str(tmp_path))
        raw = _raw(
            train={0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8]},
            val={0: [9], 1: [0], 2: [10]},
            test={0: [11], 1: [12], 2: [0]},
            n_users=3,
            n_items=13,
        )
        monkeypatch.setattr(loader, "_load_raw", lambda: raw)

        train, val, test, n_users, n_items = loader.load_splits()

        assert 0 in train[0]
        assert val[1] == {0}
        assert test[2] == {0}
        assert set(val) == set(test) == {0, 1, 2}
        assert (n_users, n_items) == (3, 13)
