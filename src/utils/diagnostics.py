"""Opt-in, bounded training diagnostics (SDD S04).

The training loop can record, at initialisation and at a fixed set of
optimizer steps, a detached summary of what the model does on a small
**fixed probe** drawn once from the training interactions:

* source / fused visual-feature norm quantiles;
* per-branch pre-activation (pre-ReLU) min / median / max and active
  fraction, for models that expose :meth:`diagnostic_branches` (VNPR),
  separately for the training branches (positive / negative, dropout
  on) and the inference branches ((u, i, i), dropout off);
* score min / max / std / unique fraction, pairwise score differences
  on fixed item pairs and top-score tie frequency;
* loss components (BPR log-loss, L2 term, total), gradient norms and
  parameter-delta norms on tracked slices (probe rows of the embedding
  tables, whole small parameters);
* optimizer steps actually applied (Adam's step counter, so AMP skips
  show as ``steps_attempted - steps_applied``) and the scaler scale;
* at every validation: the selection metric, a separate zero-metric
  flag, the probe tie frequency and whether a winner checkpoint exists.

Everything is computed under ``torch.no_grad`` or on a throw-away
backward whose gradients are cleared, inside ``torch.random.fork_rng``,
so the training trajectory (RNG stream, parameters, optimizer state) is
unchanged whether diagnostics are on or off.  Nothing is retained
between records except the tracked parameter slices; there is no
whole-catalogue activation dump.  Off by default: enable with
``diagnostics.enabled: true`` in the resolved configuration.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.atomic_io import atomic_write

QUANTILES: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
#: Largest tracked parameter slice kept between records (elements).
MAX_TRACKED_ELEMENTS = 1 << 20
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DiagnosticsConfig:
    """Resolved ``diagnostics:`` block; ``enabled`` is false by default."""

    enabled: bool = False
    probe_users: int = 64
    probe_items: int = 256
    probe_pairs: int = 64
    probe_seed: int = 0
    steps: tuple[int, ...] = (1, 10, 100, 1000)
    output_dir: str = "results/diagnostics"

    @classmethod
    def from_config(cls, config: dict) -> DiagnosticsConfig | None:
        """``None`` unless the block exists and says ``enabled: true``."""
        block = config.get("diagnostics") or {}
        if not bool(block.get("enabled", False)):
            return None
        return cls(
            enabled=True,
            probe_users=int(block.get("probe_users", cls.probe_users)),
            probe_items=int(block.get("probe_items", cls.probe_items)),
            probe_pairs=int(block.get("probe_pairs", cls.probe_pairs)),
            probe_seed=int(block.get("probe_seed", cls.probe_seed)),
            steps=tuple(int(s) for s in block.get("steps", cls.steps)),
            output_dir=str(block.get("output_dir", cls.output_dir)),
        )


@dataclass(frozen=True)
class Probe:
    """Fixed users, items, one (pos, neg) per user and fixed item pairs."""

    users: torch.Tensor
    items: torch.Tensor
    pos: torch.Tensor
    neg: torch.Tensor
    pairs: torch.Tensor  # (P, 2) positions into ``items``

    def to(self, device: str | torch.device) -> Probe:
        return Probe(
            *(t.to(device) for t in (self.users, self.items, self.pos, self.neg, self.pairs))
        )


def build_probe(
    train_interactions: dict, n_users: int, n_items: int, config: DiagnosticsConfig
) -> Probe:
    """Seeded probe over users with at least one training interaction."""
    rng = np.random.default_rng(config.probe_seed)
    eligible = np.array(sorted(u for u, s in train_interactions.items() if s), dtype=np.int64)
    if eligible.size == 0 or n_items <= 0:
        raise ValueError("diagnostics probe needs at least one training interaction.")
    users = np.sort(
        rng.choice(eligible, size=min(config.probe_users, eligible.size), replace=False)
    )
    items = np.sort(rng.choice(n_items, size=min(config.probe_items, n_items), replace=False))
    pos = np.array([rng.choice(sorted(train_interactions[int(u)])) for u in users], dtype=np.int64)
    neg = np.array([_draw_negative(rng, train_interactions[int(u)], n_items) for u in users])
    n_pairs = min(config.probe_pairs, items.size * (items.size - 1) // 2)
    pairs = np.array([_draw_pair(rng, items.size) for _ in range(n_pairs)], dtype=np.int64)
    return Probe(
        users=torch.from_numpy(users),
        items=torch.from_numpy(items),
        pos=torch.from_numpy(pos),
        neg=torch.from_numpy(neg.astype(np.int64)),
        pairs=torch.from_numpy(pairs.reshape(-1, 2)),
    )


def _draw_negative(rng: np.random.Generator, seen: set, n_items: int, attempts: int = 64) -> int:
    for _ in range(attempts):
        candidate = int(rng.integers(n_items))
        if candidate not in seen:
            return candidate
    return int(rng.integers(n_items))


def _draw_pair(rng: np.random.Generator, n: int) -> tuple[int, int]:
    i, j = rng.choice(n, size=2, replace=False)
    return int(i), int(j)


def expected_random_metrics(n_items: int, k: int = 10) -> dict[str, float]:
    """Leave-one-out full-ranking expectation of a uniformly random ranker."""
    k = min(k, n_items)
    return {
        f"hit@{k}": k / n_items,
        f"ndcg@{k}": sum(1.0 / math.log2(r + 1) for r in range(1, k + 1)) / n_items,
    }


def _quantiles(values: torch.Tensor) -> list[float]:
    flat = values.detach().float().flatten().cpu()
    if flat.numel() == 0:
        return [math.nan] * len(QUANTILES)
    return [float(v) for v in torch.quantile(flat, torch.tensor(QUANTILES))]


def _tensor_summary(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().float().flatten().cpu()
    if flat.numel() == 0:
        return {"min": math.nan, "median": math.nan, "max": math.nan, "mean": math.nan}
    return {
        "min": float(flat.min()),
        "median": float(flat.median()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
    }


def _feature_norms(model: Any, items: torch.Tensor) -> dict[str, Any] | None:
    """Per-source and fused L2-norm quantiles of the probe items' features."""
    if not getattr(model, "has_visual_features", False):
        return None
    raw = model._raw_visual_rows(items)
    fusion = getattr(model, "_online_fusion", None)
    if raw.dim() == 3:
        sources = [raw[:, m, :] for m in range(raw.shape[1])]
    elif fusion is not None and getattr(fusion, "source_dims", None):
        sources = list(torch.split(raw, list(fusion.source_dims), dim=-1))
    else:
        sources = [raw.flatten(1)]
    fused = model._resolve_visual(items)
    return {
        "source_norm_quantiles": [_quantiles(s.float().norm(dim=-1)) for s in sources],
        "fused_norm_quantiles": _quantiles(fused.float().norm(dim=-1)),
        "fused_dim": int(fused.shape[-1]),
    }


def _branch_stats(model: Any, probe: Probe, *, train_mode: bool) -> dict[str, Any] | None:
    """Pre-activation summaries for models exposing ``diagnostic_branches``."""
    hook = getattr(model, "diagnostic_branches", None)
    if hook is None:
        return None
    model.train(train_mode)
    items_neg = probe.neg if train_mode else probe.pos  # inference feeds (u, i, i)
    branches = hook(probe.users, probe.pos, items_neg)
    out: dict[str, Any] = {}
    for name, pre in branches.items():
        summary = _tensor_summary(pre)
        summary["active_fraction"] = float((pre.detach() > 0).float().mean())
        out[name] = summary
    return out


def _score_stats(model: Any, probe: Probe) -> dict[str, Any]:
    """Spread, ties and pairwise differences of the probe score matrix."""
    model.eval()
    if hasattr(model, "predict_batch"):
        scores = model.predict_batch(probe.users, probe.items)
    else:
        scores = torch.stack([model.predict(int(u), probe.items) for u in probe.users])
    scores = scores.detach().float()
    top = scores.max(dim=1).values
    tied_at_top = (scores == top[:, None]).float().mean(dim=1)
    diffs = (scores[:, probe.pairs[:, 0]] - scores[:, probe.pairs[:, 1]]).flatten()
    return {
        "min": float(scores.min()),
        "max": float(scores.max()),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "finite": bool(torch.isfinite(scores).all()),
        "unique_fraction": float(torch.unique(scores).numel() / scores.numel()),
        "constant_user_fraction": float((scores.std(dim=1) == 0).float().mean()),
        "top_tie_fraction": float(tied_at_top.mean()),
        "all_tied_user_fraction": float((tied_at_top == 1.0).float().mean()),
        "pairwise_zero_fraction": float((diffs == 0).float().mean()),
        "pairwise_abs_diff_quantiles": _quantiles(diffs.abs()),
    }


def _loss_and_gradients(model: Any, probe: Probe) -> tuple[dict[str, Any], dict[str, float]]:
    """One throw-away backward on the probe triple; gradients are cleared."""
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        score_pos, score_neg = model(probe.users, probe.pos, probe.neg)
        bpr = -F.logsigmoid(score_pos - score_neg).mean()
        total = model.bpr_loss(score_pos, score_neg)
        total.backward()
    grads = {
        name: float(p.grad.detach().norm())
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    loss = {
        "bpr": float(bpr.detach()),
        "l2": float((total - bpr).detach()),
        "total": float(total.detach()),
        "finite": bool(torch.isfinite(total.detach())),
        "score_pos_mean": float(score_pos.detach().float().mean()),
        "score_neg_mean": float(score_neg.detach().float().mean()),
    }
    model.zero_grad(set_to_none=True)
    return loss, grads


def _tracked_slice(param: torch.Tensor, probe: Probe, n_users: int, n_items: int) -> torch.Tensor:
    """Probe rows of a user/item table, or the whole (small) parameter."""
    data = param.detach()
    if data.dim() == 2 and data.shape[0] == n_users:
        data = data[probe.users.to(data.device)]
    elif data.dim() == 2 and data.shape[0] == n_items:
        data = data[probe.items.to(data.device)]
    if data.numel() > MAX_TRACKED_ELEMENTS:
        data = data.flatten()[:MAX_TRACKED_ELEMENTS]
    return data.float().cpu().clone()


def _optimizer_stats(optimizer: Any, scaler: Any, steps_attempted: int) -> dict[str, Any]:
    applied = 0
    if optimizer is not None:
        for state in optimizer.state.values():
            step = state.get("step")
            if step is not None:
                applied = max(applied, int(float(step)))
    out: dict[str, Any] = {
        "steps_attempted": int(steps_attempted),
        "steps_applied": applied,
        "steps_skipped": int(steps_attempted) - applied,
    }
    if scaler is not None:
        out["scaler_enabled"] = bool(scaler.is_enabled())
        out["scaler_scale"] = float(scaler.get_scale()) if scaler.is_enabled() else None
    return out


class TrainingDiagnostics:
    """Bounded, detached probe recorder attached to one training run."""

    def __init__(
        self,
        model: Any,
        *,
        config: DiagnosticsConfig,
        probe: Probe,
        identity: dict[str, Any],
        output_path: Path,
        n_users: int,
        n_items: int,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.config = config
        self.identity = dict(identity)
        self.output_path = Path(output_path)
        self.n_users, self.n_items = int(n_users), int(n_items)
        self._probe = probe.to(device)
        self._steps = frozenset(int(s) for s in config.steps)
        self._records: list[dict[str, Any]] = []
        self._init_params: dict[str, torch.Tensor] | None = None
        self._prev_params: dict[str, torch.Tensor] | None = None
        self._fork_devices = [torch.device(device)] if str(device).startswith("cuda") else []

    @classmethod
    def for_run(
        cls,
        model: Any,
        *,
        config: dict,
        train_interactions: dict,
        n_users: int,
        n_items: int,
        identity: dict[str, Any],
        device: str = "cpu",
    ) -> TrainingDiagnostics | None:
        """Build from the resolved config; ``None`` when diagnostics are off."""
        resolved = DiagnosticsConfig.from_config(config)
        if resolved is None:
            return None
        probe = build_probe(train_interactions, n_users, n_items, resolved)
        run_id = str(identity.get("run_id", "run"))
        return cls(
            model,
            config=resolved,
            probe=probe,
            identity=identity,
            output_path=Path(resolved.output_dir) / f"{run_id}.json",
            n_users=n_users,
            n_items=n_items,
            device=device,
        )

    @property
    def records(self) -> list[dict[str, Any]]:
        return list(self._records)

    def wants_step(self, step: int) -> bool:
        return int(step) in self._steps

    def record_init(self) -> dict[str, Any]:
        return self._record("init", step=0, epoch=None, optimizer=None, scaler=None, attempted=0)

    def record_step(
        self, *, step: int, epoch: int, optimizer: Any, scaler: Any, steps_attempted: int
    ) -> dict[str, Any]:
        return self._record(
            "step",
            step=step,
            epoch=epoch,
            optimizer=optimizer,
            scaler=scaler,
            attempted=steps_attempted,
        )

    def record_validation(
        self,
        *,
        epoch: int,
        step: int,
        metrics: dict[str, Any],
        es_metric: str,
        checkpoint_exists: bool,
        has_valid_observation: bool,
    ) -> dict[str, Any]:
        """Selection outcome with zero-metric, tie and checkpoint as separate fields."""
        value = metrics.get(es_metric)
        numeric = float(value) if isinstance(value, int | float) else math.nan
        record = {
            "phase": "validation",
            "epoch": int(epoch),
            "step": int(step),
            "selection_metric": es_metric,
            "metric_value": numeric,
            "metric_present": es_metric in metrics,
            "metric_finite": bool(math.isfinite(numeric)),
            "zero_metric": bool(numeric == 0.0),
            "metrics": {k: float(v) for k, v in metrics.items() if isinstance(v, int | float)},
            "checkpoint_exists": bool(checkpoint_exists),
            "has_valid_observation": bool(has_valid_observation),
            "scores": self._guarded(lambda: _score_stats(self.model, self._probe)),
        }
        record["tie_frequency"] = record["scores"]["top_tie_fraction"]
        self._records.append(record)
        return record

    def _guarded(self, fn):
        was_training = self.model.training
        with torch.random.fork_rng(devices=self._fork_devices), torch.no_grad():
            result = fn()
        self.model.train(was_training)
        return result

    def _record(self, phase: str, *, step: int, epoch: int | None, optimizer, scaler, attempted):
        was_training = self.model.training
        with torch.random.fork_rng(devices=self._fork_devices):
            with torch.no_grad():
                features = _feature_norms(self.model, self._probe.items)
                train_branches = _branch_stats(self.model, self._probe, train_mode=True)
                eval_branches = _branch_stats(self.model, self._probe, train_mode=False)
                scores = _score_stats(self.model, self._probe)
            loss, grads = _loss_and_gradients(self.model, self._probe)
        self.model.train(was_training)
        record = {
            "phase": phase,
            "step": int(step),
            "epoch": epoch,
            "features": features,
            "train_branches": train_branches,
            "eval_branches": eval_branches,
            "scores": scores,
            "loss": loss,
            "gradient_norms": grads,
            "gradient_norm_total": math.sqrt(sum(g * g for g in grads.values())),
            "parameters": self._parameter_stats(),
            "optimizer": _optimizer_stats(optimizer, scaler, attempted),
        }
        self._records.append(record)
        return record

    def _parameter_stats(self) -> dict[str, dict[str, float]]:
        current = {
            name: _tracked_slice(p, self._probe, self.n_users, self.n_items)
            for name, p in self.model.named_parameters()
        }
        if self._init_params is None:
            self._init_params = current
        prev = self._prev_params or self._init_params
        stats: dict[str, dict[str, float]] = {}
        for name, p in self.model.named_parameters():
            tracked = current[name]
            entry = {
                "norm": float(p.detach().float().norm()),
                "tracked_norm": float(tracked.norm()),
                "delta_from_prev": float((tracked - prev[name]).norm()),
                "delta_from_init": float((tracked - self._init_params[name]).norm()),
            }
            if tracked.dim() == 2:
                entry["zero_row_fraction"] = float((tracked.abs().sum(dim=1) == 0).float().mean())
            stats[name] = entry
        self._prev_params = current
        return stats

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "identity": self.identity,
            "config": asdict(self.config),
            "probe": {
                "n_users": int(self._probe.users.numel()),
                "n_items": int(self._probe.items.numel()),
                "n_pairs": int(self._probe.pairs.shape[0]),
                "seed": self.config.probe_seed,
            },
            "random_baseline": expected_random_metrics(self.n_items),
            "records": self._records,
        }

    def write(self) -> Path:
        """Atomically write the JSON report; returns its path."""
        text = json.dumps(self.payload(), indent=2, allow_nan=True)
        atomic_write(lambda tmp: Path(tmp).write_text(text, encoding="utf-8"), self.output_path)
        return self.output_path


__all__ = [
    "DiagnosticsConfig",
    "Probe",
    "TrainingDiagnostics",
    "build_probe",
    "expected_random_metrics",
]
