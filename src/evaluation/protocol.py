"""Evaluation protocol for recommendation models.

Two protocols are supported, selected by the ``protocol`` constructor
argument:

* ``"full_ranking"`` (default and primary protocol — recommended for
  thesis-grade comparisons).  Scores every item in the catalogue for
  every test user, masks items seen in train + val, computes top-K
  metrics on the resulting full ranking.
* ``"sampled"``.  For each test user, draws ``n_negatives`` items the
  user has not seen and ranks the held-out positives against that
  smaller pool.  Much cheaper but **statistically inconsistent** with
  full-ranking (Krichene & Rendle, KDD 2020): the relative ordering of
  models can flip between the two protocols, so sampled metrics
  should only be reported for comparability with prior work that
  used the same protocol — never as the primary benchmark number.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.evaluation.metrics import compute_all_metrics
from src.utils.logging import get_logger

logger = get_logger(__name__)


ProtocolName = Literal["full_ranking", "sampled"]


class Evaluator:
    """Full-ranking evaluator with per-user metric computation.

    Parameters
    ----------
    train_interactions:
        Mapping ``{user_id: set_of_item_ids}`` representing the training
        history.  Used to filter out already-seen items from candidates.
    test_interactions:
        Mapping ``{user_id: set_of_item_ids}`` representing the held-out
        items.  In leave-one-out there is exactly one item per user.
    n_items:
        Total number of items in the catalogue (items are assumed to be
        integer-indexed from ``0`` to ``n_items - 1``).
    k_values:
        List of cut-off positions at which metrics are computed.
    """

    def __init__(
        self,
        train_interactions: dict[int, set[int]],
        test_interactions: dict[int, set[int]],
        n_items: int,
        k_values: list[int] | None = None,
        sample_size: int | None = None,
        sample_seed: int = 42,
        protocol: ProtocolName = "full_ranking",
        n_negatives: int = 100,
        negative_sampling_seed: int = 42,
    ) -> None:
        if protocol not in ("full_ranking", "sampled"):
            raise ValueError(f"protocol must be 'full_ranking' or 'sampled'; got {protocol!r}")
        if protocol == "sampled" and n_negatives < 1:
            raise ValueError("n_negatives must be >= 1 when protocol='sampled'")

        self.train_interactions = train_interactions
        self.test_interactions = test_interactions
        self.n_items = n_items
        self.k_values = k_values if k_values is not None else [5, 10, 20]
        self.max_k = max(self.k_values)
        self.protocol: ProtocolName = protocol
        self.n_negatives = n_negatives
        self.negative_sampling_seed = negative_sampling_seed

        # Lazy GPU cache: per-user training-item indices as a LongTensor on
        # the same device used during evaluation.  Built once on first
        # access and reused across epochs to avoid rebuilding tensors.
        self._train_idx_gpu: dict[int, torch.Tensor | None] = {}
        self._train_idx_device: torch.device | None = None

        all_test_users = sorted(test_interactions.keys())

        # When sample_size is set and smaller than the population, draw a
        # deterministic random subset. Used for fast early-stopping during
        # hyperparameter search; final reported metrics should always be
        # produced with sample_size=None.
        if sample_size is not None and sample_size < len(all_test_users):
            rng = np.random.default_rng(sample_seed)
            idx = rng.choice(len(all_test_users), size=sample_size, replace=False)
            self.test_users = sorted(all_test_users[i] for i in idx)
            self.is_sampled = True
        else:
            self.test_users = all_test_users
            self.is_sampled = False

        logger.info(
            "Evaluator initialised: %d test users%s, %d items, k=%s, protocol=%s%s",
            len(self.test_users),
            f" (sampled from {len(all_test_users)}, seed={sample_seed})" if self.is_sampled else "",
            self.n_items,
            self.k_values,
            self.protocol,
            f", n_negatives={self.n_negatives}" if self.protocol == "sampled" else "",
        )

    def evaluate(
        self,
        model: Any,
        device: str = "cuda",
    ) -> dict[str, float]:
        """Compute averaged metrics across all test users.

        Parameters
        ----------
        model:
            A recommendation model whose ``predict(user_id, item_ids)``
            method returns a 1-D tensor of scores for the given items.
        device:
            Torch device string (``"cuda"`` or ``"cpu"``).

        Returns
        -------
        dict
            Averaged metrics, e.g. ``{'precision@5': 0.12, ...}``.
        """
        per_user_df = self.evaluate_per_user(model, device=device)

        metric_cols = [c for c in per_user_df.columns if c != "user_id"]
        return per_user_df[metric_cols].mean().to_dict()

    def evaluate_per_user(
        self,
        model: Any,
        device: str = "cuda",
        batch_size: int = 512,
    ) -> pd.DataFrame:
        """Compute per-user metrics and return them as a DataFrame.

        The returned DataFrame has one row per test user and columns
        ``user_id``, ``precision@5``, ``ndcg@10``, etc.  This is the
        format needed for statistical significance tests (e.g. Wilcoxon
        signed-rank test operating on paired per-user scores).

        Parameters
        ----------
        model:
            Recommendation model (see :meth:`evaluate` for the expected
            interface).
        device:
            Torch device string.
        batch_size:
            Number of users to score in parallel when the model supports
            ``predict_batch``.  Ignored for single-user fallback.

        Returns
        -------
        pd.DataFrame
            Per-user metric values.
        """
        device_obj = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        all_items = torch.arange(self.n_items, device=device_obj)
        per_user_results: list[dict] = []

        model.eval()

        has_batch = hasattr(model, "predict_batch") and callable(model.predict_batch)

        with torch.no_grad():
            if self.protocol == "sampled":
                per_user_results = self._evaluate_sampled(model, device_obj)
            elif has_batch:
                per_user_results = self._evaluate_batched(model, all_items, batch_size)
            else:
                per_user_results = self._evaluate_single(model, all_items)

        df = pd.DataFrame(per_user_results)
        cols = ["user_id"] + [c for c in df.columns if c != "user_id"]
        return df[cols]

    def _evaluate_single(
        self,
        model: Any,
        all_items: torch.Tensor,
    ) -> list[dict]:
        """Fallback: score one user at a time."""
        results: list[dict] = []
        for user_id in tqdm(self.test_users, desc="Evaluating"):
            scores = model.predict(user_id, all_items)
            if isinstance(scores, torch.Tensor):
                user_scores = scores.cpu().numpy()
            else:
                user_scores = np.asarray(scores)

            results.append(self._rank_and_score(user_id, user_scores))
        return results

    def _evaluate_batched(
        self,
        model: Any,
        all_items: torch.Tensor,
        batch_size: int,
    ) -> list[dict]:
        """Score users in batches using model.predict_batch().

        The hot path keeps everything on GPU as long as possible:
        ``predict_batch`` returns ``(B, N)`` scores, training items are
        masked in place via ``index_fill_``, and ``torch.topk`` selects
        the top-K candidates per user.  Only the resulting ``(B, K)``
        index matrix is transferred to CPU, instead of the full ``(B, N)``
        score matrix.  For amazon_women this shrinks the per-batch GPU→CPU
        transfer from hundreds of MB to ~40 KB.
        """
        device = all_items.device
        self._ensure_train_idx_cache(device)

        results: list[dict] = []
        n_users = len(self.test_users)
        neg_inf = float("-inf")

        for start in tqdm(range(0, n_users, batch_size), desc="Evaluating"):
            batch_user_ids = self.test_users[start : start + batch_size]
            user_ids_tensor = torch.tensor(batch_user_ids, dtype=torch.long, device=device)

            batch_scores = model.predict_batch(user_ids_tensor, all_items)

            for i, user_id in enumerate(batch_user_ids):
                idx = self._train_idx_gpu.get(user_id)
                if idx is not None:
                    batch_scores[i].index_fill_(0, idx, neg_inf)

            # Stable descending sort instead of topk: torch.topk's tie
            # order is backend-dependent (CPU vs GPU can rank tied items
            # differently), which breaks reproducibility across devices.
            # A stable sort guarantees the unified rule used by every
            # evaluation path: ties broken by LOWER item index.
            top_indices = torch.sort(batch_scores, dim=1, descending=True, stable=True).indices[
                :, : self.max_k
            ]  # (B, K)
            top_indices_np = top_indices.cpu().numpy()

            for i, user_id in enumerate(batch_user_ids):
                ranked_list = top_indices_np[i].tolist()
                ground_truth = self.test_interactions[user_id]
                user_metrics = compute_all_metrics(ranked_list, ground_truth, self.k_values)
                user_metrics["user_id"] = user_id
                results.append(user_metrics)
        return results

    def _ensure_train_idx_cache(self, device: torch.device) -> None:
        """Build per-user train-item index tensors on the target device.

        Rebuilds the cache only when the device changes (e.g. first call
        or after switching CPU↔GPU).  Each entry maps ``user_id`` to a
        ``LongTensor`` of training item ids, or ``None`` for users with
        no training history.
        """
        if self._train_idx_device == device and self._train_idx_gpu:
            return
        self._train_idx_gpu = {}
        for user_id in self.test_users:
            items = self.train_interactions.get(user_id)
            if items:
                self._train_idx_gpu[user_id] = torch.tensor(
                    list(items),
                    dtype=torch.long,
                    device=device,
                )
            else:
                self._train_idx_gpu[user_id] = None
        self._train_idx_device = device

    def _evaluate_sampled(
        self,
        model: Any,
        device: torch.device,
    ) -> list[dict]:
        """Score each user against ``n_negatives`` negatives plus its positives.

        Krichene & Rendle (KDD 2020) showed that ranking inside a small
        sampled pool does not preserve model ordering compared to
        full-ranking, so this path is opt-in and warns at call time.
        The implementation scores one user at a time because each user
        has its own pool of candidates; per-user RNG seeds make the
        sampling deterministic and resumable.
        """
        results: list[dict] = []
        for user_id in tqdm(self.test_users, desc="Evaluating (sampled)"):
            positives = self.test_interactions[user_id]
            if not positives:
                continue
            seen = self.train_interactions.get(user_id, set())
            forbidden = seen | positives

            negatives = self._sample_negatives(user_id, forbidden)
            pool: list[int] = list(positives) + negatives
            pool_tensor = torch.tensor(pool, dtype=torch.long, device=device)

            scores = model.predict(user_id, pool_tensor)
            if isinstance(scores, torch.Tensor):
                scores_np = scores.cpu().numpy()
            else:
                scores_np = np.asarray(scores)

            # Unified deterministic tie-break: ties by LOWER catalogue
            # item id.  Breaking ties by pool position would favour the
            # positives (listed first in the pool) and inflate metrics.
            pool_ids = np.asarray(pool)
            order = np.lexsort((pool_ids, -scores_np))
            top_k = order[: self.max_k]
            ranked_list = pool_ids[top_k].tolist()

            user_metrics = compute_all_metrics(ranked_list, positives, self.k_values)
            user_metrics["user_id"] = user_id
            results.append(user_metrics)
        return results

    def _sample_negatives(self, user_id: int, forbidden: set[int]) -> list[int]:
        """Draw ``n_negatives`` items not in ``forbidden`` for ``user_id``.

        Uses a per-user RNG seeded from
        ``(negative_sampling_seed, user_id)`` so the sampled pool is
        identical across runs and across model comparisons — paired
        statistical tests rely on identical candidate pools.
        """
        available = self.n_items - len(forbidden)
        if available <= self.n_negatives:
            return [i for i in range(self.n_items) if i not in forbidden]

        rng = np.random.default_rng((self.negative_sampling_seed, int(user_id)))
        negatives: list[int] = []
        chosen: set[int] = set()
        while len(negatives) < self.n_negatives:
            candidates = rng.integers(0, self.n_items, size=self.n_negatives)
            for cand in candidates:
                cand_int = int(cand)
                if cand_int in forbidden or cand_int in chosen:
                    continue
                chosen.add(cand_int)
                negatives.append(cand_int)
                if len(negatives) == self.n_negatives:
                    break
        return negatives

    def _rank_and_score(self, user_id: int, user_scores: np.ndarray) -> dict:
        """Mask training items, rank, and compute metrics for one user.

        Used by the single-user fallback path; the batched path performs
        the same operations on GPU.
        """
        train_items = self.train_interactions.get(user_id, set())
        if train_items:
            train_idx = np.array(list(train_items), dtype=np.int64)
            user_scores[train_idx] = -np.inf

        # Stable full sort instead of argpartition: partition boundaries
        # split tied scores arbitrarily, so tied items could enter or
        # miss the top-k nondeterministically.  Stable sort implements
        # the unified rule (ties broken by lower item index) shared with
        # the batched and sampled paths.
        ranked_list = np.argsort(-user_scores, kind="stable")[: self.max_k].tolist()

        ground_truth = self.test_interactions[user_id]
        user_metrics = compute_all_metrics(ranked_list, ground_truth, self.k_values)
        user_metrics["user_id"] = user_id
        return user_metrics
