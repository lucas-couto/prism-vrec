"""Evaluation protocol for recommendation models.

Two protocols are supported, selected by the ``protocol`` constructor
argument:

* ``"full_ranking"`` (default and primary protocol — recommended for
  thesis-grade comparisons).  Scores every item in the catalogue for
  every test user, masks the interactions passed as the seen/train set
  (train+val at final evaluation), computes top-K metrics on the
  resulting full ranking.
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
from src.utils import telemetry
from src.utils.logging import get_logger

logger = get_logger(__name__)


ProtocolName = Literal["full_ranking", "sampled"]


class NonFiniteScoresError(RuntimeError):
    """A model emitted NaN/inf scores during evaluation.

    Masking is the Evaluator's job (train items are ``-inf``-filled
    AFTER this check), so any non-finite value in a model's raw output
    is a numerical bug — typically an fp16/AMP overflow.  It must fail
    loud: the ranking paths would otherwise disagree silently (the
    batched sort ranks NaN at the TOP, the numpy lexsort at the bottom)
    and a NaN held-out score turns into a spurious ``_rank`` of 1.
    """


def _assert_finite_scores(finite_mask: np.ndarray | torch.Tensor, where: str) -> None:
    """Raise :class:`NonFiniteScoresError` unless *finite_mask* is all-True."""
    if bool(finite_mask.all()):
        return
    n_bad = int((~finite_mask).sum())
    raise NonFiniteScoresError(
        f"model emitted {n_bad} non-finite score(s) ({where}). "
        "Raw scores must be finite — the Evaluator applies the "
        "train-item mask itself. Check the model for fp16/AMP "
        "overflow or uninitialised parameters."
    )


#: Number of top items persisted per user for downstream inspection
#: (the ``_top_items`` column).  Together with ``max_k`` it bounds how
#: much of each full ranking is ever read back.
TOP_ITEMS_PERSISTED = 20

#: Bytes of GPU memory the batched ranking holds per ``(user, item)``
#: pair at its peak: the score matrix and its tie-break reordering
#: (fp32, 4 B each), the sort permutation (int64, 8 B) and the workspace
#: ``torch.sort`` allocates for a stable sort (~12 B, values + indices).
#: Rounded up — overestimating costs a smaller batch, underestimating
#: costs an OOM mid-grid.
RANKING_BYTES_PER_ELEMENT = 28

#: Share of a process's GPU allowance the ranking buffers may take when
#: the caller does not state a budget.  The remainder covers the model,
#: its embedding tables, the optimiser state and the autograd graph.
_DEFAULT_RANKING_VRAM_FRACTION = 0.35


def plan_ranking_batch(requested: int, n_items: int, budget_bytes: int) -> int:
    """Clamp a user-batch size to what the ranking buffers can afford.

    Scoring ``B`` users against ``N`` items allocates ``B * N`` elements
    several times over (see :data:`RANKING_BYTES_PER_ELEMENT`), so a
    batch size that is comfortable on a 166K-item catalogue overflows on
    a 348K-item one.  This turns the caller's value into an upper bound
    and derives the real one from the catalogue size.

    Batching is a pure execution detail here: every row is sorted,
    masked and scored independently, so the metrics are identical for
    any batch size.  Only throughput and peak memory change.

    :param requested: The caller's batch size, treated as a maximum.
    :param n_items: Catalogue size ``N``.
    :param budget_bytes: GPU bytes the ranking buffers may occupy.
    :returns: Batch size in ``[1, requested]``.
    """
    if n_items <= 0 or budget_bytes <= 0:
        return max(1, requested)
    per_user = n_items * RANKING_BYTES_PER_ELEMENT
    return max(1, min(requested, int(budget_bytes // per_user)))


def default_ranking_budget(device: torch.device) -> int:
    """GPU bytes one process may spend on ranking buffers, by default.

    A fixed fraction of this process's VRAM *allowance* (total memory
    scaled by any ``set_per_process_memory_fraction`` cap), so a capped
    worker never budgets ranking buffers off the whole card while
    sharing it with siblings.  Callers that know a tighter allowance
    still pass it explicitly.  Returns ``0`` on CPU, where the caller's
    batch size stands (host RAM is an order of magnitude larger and not
    the constraint).
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0
    try:
        from src.utils.device import vram_allowance_bytes

        allowance = vram_allowance_bytes(device)
    except Exception:  # noqa: BLE001 — probing must never break evaluation
        return 0
    return int(allowance * _DEFAULT_RANKING_VRAM_FRACTION)


class Evaluator:
    """Full-ranking evaluator with per-user metric computation.

    Parameters
    ----------
    train_interactions:
        Mapping ``{user_id: set_of_item_ids}`` representing the training
        history.  Used to filter out already-seen items from candidates.
    test_interactions:
        Mapping ``{user_id: set_of_item_ids}`` representing the held-out
        items.  In leave-one-out there is exactly one item per user;
        multi-item test sets are supported for plain metric evaluation,
        but the per-user sufficient-statistic records
        (:meth:`per_user_records` / :meth:`evaluate_with_records`)
        require leave-one-out and raise otherwise.
    n_items:
        Total number of items in the catalogue (items are assumed to be
        integer-indexed from ``0`` to ``n_items - 1``).
    k_values:
        List of cut-off positions at which metrics are computed.
    ranking_budget_bytes:
        GPU bytes the batched full-ranking buffers may occupy.  Used to
        derive the real user-batch size from the catalogue size (see
        :func:`plan_ranking_batch`).  ``None`` derives a default from
        the device's total memory; a process capped by
        ``torch.cuda.set_per_process_memory_fraction`` must pass its own
        allowance, because that cap is invisible to the device query.
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
        tiebreak_seed: int = 42,
        ranking_budget_bytes: int | None = None,
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
        #: GPU bytes the batched ranking may hold.  ``None`` means "derive
        #: from the device"; a training worker running under
        #: ``set_per_process_memory_fraction`` passes its real allowance,
        #: which the device cannot report.
        self.ranking_budget_bytes = ranking_budget_bytes

        #: True when every test user holds out exactly ONE item.  The
        #: per-user sufficient-statistic records (``_rank``,
        #: ``_tie_block_size``, ...) pick the held-out via
        #: ``next(iter(...))`` — meaningless for a multi-item test set —
        #: so the records-producing entry points refuse to run unless
        #: this holds.  Plain metric evaluation supports any test-set
        #: cardinality.
        self._is_leave_one_out = all(len(items) == 1 for items in test_interactions.values())

        # Random tie-break key: exact-score ties are broken by ascending
        # ``_tiebreak_key[item]`` instead of ascending item_idx.  item_idx
        # correlates with popularity in the DVBPR splits (Spearman -0.34 to
        # -0.45), so an id tie-break would systematically favour popular
        # items inside a tie block.  A fixed permutation seeded from the
        # run's global seed breaks ties uniformly at random yet
        # reproducibly, and is identical for every model/trial of a
        # (dataset, seed) run — so it never becomes between-model variance.
        # ``_tiebreak_order`` lists item ids in ascending key: a stable
        # descending sort over columns reordered by it breaks ties by the
        # key (used by the batched torch path).
        rng = np.random.default_rng(tiebreak_seed)
        self._tiebreak_key = rng.permutation(n_items).astype(np.int64)
        self._tiebreak_order_np = np.argsort(self._tiebreak_key).astype(np.int64)
        self._tiebreak_order_gpu: torch.Tensor | None = None
        self._tiebreak_order_device: torch.device | None = None
        self._tiebreak_key_gpu: torch.Tensor | None = None
        self._tiebreak_key_device: torch.device | None = None

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
        frame = self._per_user_frame(model, device, batch_size)
        self._log_tie_stats(frame)
        return self._metrics_view(frame)

    def _per_user_frame(self, model: Any, device: str, batch_size: int) -> pd.DataFrame:
        """Run the ranking dispatch ONCE, returning the raw per-user frame.

        Rows carry both the metric columns and the ``_``-prefixed
        diagnostics/records (``_rank``, ``_n_candidates``,
        ``_tie_block_size``, ``_top_items``).  Every public entry point
        derives its result from this single pass — the sufficient
        statistic is never recomputed in a second scoring pass.
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

        return pd.DataFrame(per_user_results)

    @staticmethod
    def _metrics_view(frame: pd.DataFrame) -> pd.DataFrame:
        """Metric matrix: drop the ``_``-prefixed diagnostic/record columns.

        Keeps them out of the per-user matrix the statistical step consumes.
        """
        df = frame.drop(columns=[c for c in frame.columns if c.startswith("_")])
        cols = ["user_id"] + [c for c in df.columns if c != "user_id"]
        return df[cols]

    @staticmethod
    def _records_view(frame: pd.DataFrame) -> pd.DataFrame:
        """Per-user sufficient-statistic records (``_x`` columns → ``x``)."""
        rename = {c: c[1:] for c in frame.columns if c.startswith("_")}
        return frame[["user_id", *rename]].rename(columns=rename)

    def _require_leave_one_out(self, caller: str) -> None:
        """Refuse the records path when the test split is not leave-one-out.

        The ``_rank`` / ``_tie_block_size`` records are sufficient
        statistics ONLY when each user holds out exactly one item; for a
        multi-item test set they would silently describe an arbitrary
        held-out.  Plain metric evaluation (:meth:`evaluate_per_user`)
        keeps working for any test-set cardinality.
        """
        if self._is_leave_one_out:
            return
        n_multi = sum(1 for items in self.test_interactions.values() if len(items) != 1)
        raise ValueError(
            f"{caller} requires a leave-one-out test split (exactly one "
            f"held-out item per user); {n_multi} of "
            f"{len(self.test_interactions)} test users violate this. "
            "Use evaluate_per_user() for plain metrics on multi-item "
            "test sets."
        )

    def evaluate_with_records(
        self,
        model: Any,
        device: str = "cuda",
        batch_size: int = 512,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Single pass → ``(metrics_df, records_df)`` (Task F).

        Used by the final evaluate step so the per-user artifact is written
        WITHOUT a second scoring pass.  Requires a leave-one-out test
        split (see :meth:`_require_leave_one_out`); raises ``ValueError``
        otherwise.
        """
        self._require_leave_one_out("evaluate_with_records")
        frame = self._per_user_frame(model, device, batch_size)
        self._log_tie_stats(frame)
        return self._metrics_view(frame), self._records_view(frame)

    def _log_tie_stats(self, df: pd.DataFrame) -> None:
        """Log how often the held-out item lands in an exact-score tie.

        The audit could not measure real exact-tie frequency (it depends
        on a trained model); this turns that unknown into a number logged
        during the battery — the empirical basis for the tie-break note in
        the dissertation.
        """
        if "_tie_block_size" not in df.columns or df.empty:
            return
        blocks = df["_tie_block_size"].to_numpy()
        tied = blocks > 1
        frac_tied = float(tied.mean())
        mean_block = float(blocks[tied].mean()) if tied.any() else 0.0
        logger.info(
            "Tie-break: %.2f%% of held-outs in an exact-score tie "
            "(mean block %.2f, max block %d, over %d users)",
            100.0 * frac_tied,
            mean_block,
            int(blocks.max()),
            len(blocks),
        )

    def per_user_records(
        self,
        model: Any,
        device: str = "cuda",
        batch_size: int = 512,
    ) -> pd.DataFrame:
        """Per-user sufficient statistics for permanent persistence (F/D3).

        Under leave-one-out the held-out's rank is a sufficient statistic
        for every accuracy metric at any k.  Returns one row per test
        user with: ``user_id``; ``rank`` (1-indexed, post-mask,
        post-tiebreak — the seeded permutation resolves ties); effective
        ``n_candidates`` (post-mask, varies per user); ``tie_block_size``
        (exact-score block of the held-out); ``top_items`` (first 20
        item_idx of the masked ranking).  Full-ranking only, and requires
        a leave-one-out test split (raises ``ValueError`` otherwise).
        """
        if self.protocol != "full_ranking":
            raise ValueError("per_user_records requires protocol='full_ranking'.")
        self._require_leave_one_out("per_user_records")
        return self._records_view(self._per_user_frame(model, device, batch_size))

    def _evaluate_single(
        self,
        model: Any,
        all_items: torch.Tensor,
    ) -> list[dict]:
        """Fallback: score one user at a time."""
        results: list[dict] = []
        for user_id in tqdm(self.test_users, desc="Evaluating"):
            scores = model.predict(user_id, all_items)
            telemetry.add_items(1)
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
        masked in place via ``index_fill_``, and only the head of each
        ranking is transferred to CPU, instead of the full ``(B, N)``
        score matrix.  For amazon_women this shrinks the per-batch GPU→CPU
        transfer from hundreds of MB to ~40 KB.

        *batch_size* is an upper bound, not the value used: ``B * N``
        elements are allocated several times over, so the real batch is
        derived from the catalogue size and the process's GPU allowance
        (see :func:`plan_ranking_batch`).  Every row is ranked and scored
        independently, so this changes throughput and peak memory only —
        never the metrics.
        """
        device = all_items.device
        self._ensure_train_idx_cache(device)

        results: list[dict] = []
        n_users = len(self.test_users)
        neg_inf = float("-inf")

        budget = self.ranking_budget_bytes
        if budget is None:
            budget = default_ranking_budget(device)
        batch_size = plan_ranking_batch(batch_size, self.n_items, budget)

        for start in tqdm(range(0, n_users, batch_size), desc="Evaluating"):
            batch_user_ids = self.test_users[start : start + batch_size]
            user_ids_tensor = torch.tensor(batch_user_ids, dtype=torch.long, device=device)

            batch_scores = model.predict_batch(user_ids_tensor, all_items)
            # R3 guard: raw scores must be finite BEFORE the train-item
            # mask goes in — the stable sort below ranks NaN at the top,
            # so a NaN would silently inflate the metrics.
            _assert_finite_scores(
                torch.isfinite(batch_scores),
                f"batched path, users {batch_user_ids[0]}..{batch_user_ids[-1]}",
            )
            # Inference throughput is measured in users ranked per second;
            # see src.utils.flops on why scoring dispatches no counted FLOPs
            # for factorisation models.
            telemetry.add_items(len(batch_user_ids))

            for i, user_id in enumerate(batch_user_ids):
                idx = self._train_idx_gpu.get(user_id)
                if idx is not None:
                    batch_scores[i].index_fill_(0, idx, neg_inf)

            # Stable descending sort instead of topk: torch.topk's tie
            # order is backend-dependent (CPU vs GPU can rank tied items
            # differently), which breaks reproducibility across devices.
            # Columns are first reordered by ``_tiebreak_order`` (ascending
            # random key), so the stable sort breaks exact-score ties by
            # that key rather than by item index — the unified rule shared
            # with the single and sampled paths.
            order = self._tiebreak_order_on(device)  # (n_items,)
            reordered = batch_scores.index_select(1, order)
            sorted_perm = torch.sort(
                reordered, dim=1, descending=True, stable=True
            ).indices  # (B, N) — full order (torch.sort already sorts the whole row)
            del reordered

            # Only the head of each ranking is ever read back.  Slicing
            # the permutation BEFORE mapping it to item ids keeps the
            # gather at (B, keep) instead of allocating a second (B, N)
            # int64 — 1.4 GB per batch on amazon_women, for columns that
            # are then thrown away.  The values are unchanged: taking the
            # first `keep` columns commutes with the element-wise lookup.
            keep = max(self.max_k, TOP_ITEMS_PERSISTED)
            top_ranked = order[sorted_perm[:, :keep]]  # (B, keep) item ids
            del sorted_perm

            metrics_top_np = top_ranked[:, : self.max_k].cpu().numpy()
            top20_np = top_ranked[:, :TOP_ITEMS_PERSISTED].cpu().numpy()  # persisted (D3)

            # Per-user sufficient statistics + tie instrumentation, computed
            # ONCE here and reused by the metric path (dropped) and the
            # persistence writer (Task F). Single transfer per batch;
            # assumes leave-one-out (one held item per user).
            key = self._tiebreak_key_on(device)
            held_ids = torch.tensor(
                [next(iter(self.test_interactions[u])) for u in batch_user_ids],
                dtype=torch.long,
                device=device,
            )
            held_scores = batch_scores.gather(1, held_ids[:, None])  # (B,1)
            tie_mask = batch_scores == held_scores
            greater = (batch_scores > held_scores).sum(dim=1)
            tied_lower = (tie_mask & (key[None, :] < key[held_ids][:, None])).sum(dim=1)
            rank_np = (1 + greater + tied_lower).cpu().numpy()
            n_cand_np = torch.isfinite(batch_scores).sum(dim=1).cpu().numpy()
            tie_blocks_np = tie_mask.sum(dim=1).cpu().numpy()
            # Everything below is numpy.  Release the (B, N) buffers now,
            # so the next iteration's predict_batch does not allocate its
            # scores while this one's are still resident.
            del batch_scores, tie_mask, top_ranked

            for i, user_id in enumerate(batch_user_ids):
                ground_truth = self.test_interactions[user_id]
                user_metrics = compute_all_metrics(
                    metrics_top_np[i].tolist(), ground_truth, self.k_values
                )
                user_metrics["user_id"] = user_id
                user_metrics["_rank"] = int(rank_np[i])
                user_metrics["_n_candidates"] = int(n_cand_np[i])
                user_metrics["_tie_block_size"] = int(tie_blocks_np[i])
                user_metrics["_top_items"] = top20_np[i].tolist()
                results.append(user_metrics)
        return results

    def _tiebreak_order_on(self, device: torch.device) -> torch.Tensor:
        """Item ids in ascending tie-break key, as a LongTensor on *device*."""
        if self._tiebreak_order_device != device or self._tiebreak_order_gpu is None:
            self._tiebreak_order_gpu = torch.as_tensor(
                self._tiebreak_order_np, dtype=torch.long, device=device
            )
            self._tiebreak_order_device = device
        return self._tiebreak_order_gpu

    def _tiebreak_key_on(self, device: torch.device) -> torch.Tensor:
        """Per-item tie-break key, as a LongTensor on *device*."""
        if self._tiebreak_key_device != device or self._tiebreak_key_gpu is None:
            self._tiebreak_key_gpu = torch.as_tensor(
                self._tiebreak_key, dtype=torch.long, device=device
            )
            self._tiebreak_key_device = device
        return self._tiebreak_key_gpu

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
            telemetry.add_items(1)
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
            # R3 guard: the sampled pool has no mask at all, so every
            # score must be finite.
            _assert_finite_scores(np.isfinite(scores_np), f"sampled path, user {user_id}")

            # Unified tie-break: exact-score ties broken by the random
            # ``_tiebreak_key`` (seeded permutation), NOT by pool position
            # (which would favour the positives, listed first) nor by item
            # id (which correlates with popularity).
            pool_ids = np.asarray(pool)
            order = np.lexsort((self._tiebreak_key[pool_ids], -scores_np))
            ranked_list = pool_ids[order[: self.max_k]].tolist()

            user_metrics = compute_all_metrics(ranked_list, positives, self.k_values)
            user_metrics["user_id"] = user_id
            held = next(iter(positives))
            held_pos = pool.index(held)
            held_score = scores_np[held_pos]
            tie_mask = scores_np == held_score
            tied_lower = int(
                np.sum(tie_mask & (self._tiebreak_key[pool_ids] < self._tiebreak_key[held]))
            )
            user_metrics["_rank"] = 1 + int(np.sum(scores_np > held_score)) + tied_lower
            user_metrics["_n_candidates"] = len(pool)
            user_metrics["_tie_block_size"] = int(tie_mask.sum())
            user_metrics["_top_items"] = pool_ids[order[:20]].tolist()
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
        # R3 guard: check the raw output before masking (numpy's lexsort
        # would push NaN to the BOTTOM — the opposite of the batched
        # path — so the two paths would silently disagree).
        _assert_finite_scores(np.isfinite(user_scores), f"single path, user {user_id}")
        train_items = self.train_interactions.get(user_id, set())
        if train_items:
            train_idx = np.array(list(train_items), dtype=np.int64)
            user_scores[train_idx] = -np.inf

        # lexsort over (tiebreak_key, -score): full sort (not argpartition,
        # whose boundaries split tied scores arbitrarily) with exact-score
        # ties broken by the random ``_tiebreak_key`` — the unified rule
        # shared with the batched and sampled paths.
        ranked = np.lexsort((self._tiebreak_key, -user_scores))

        ground_truth = self.test_interactions[user_id]
        user_metrics = compute_all_metrics(
            ranked[: self.max_k].tolist(), ground_truth, self.k_values
        )
        user_metrics["user_id"] = user_id

        # Per-user sufficient statistics, computed once (Task F): reused by
        # the persistence writer, dropped from the metric matrix.
        held = next(iter(ground_truth))
        held_score = user_scores[held]
        tie_mask = user_scores == held_score
        tied_lower = int(np.sum(tie_mask & (self._tiebreak_key < self._tiebreak_key[held])))
        user_metrics["_rank"] = 1 + int(np.sum(user_scores > held_score)) + tied_lower
        user_metrics["_n_candidates"] = int(np.sum(np.isfinite(user_scores)))
        user_metrics["_tie_block_size"] = int(tie_mask.sum())
        user_metrics["_top_items"] = ranked[:20].tolist()
        return user_metrics
