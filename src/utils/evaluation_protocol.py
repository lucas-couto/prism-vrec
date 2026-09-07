"""Which final-evaluation protocol a run uses, resolved from the YAML.

``folds.enabled`` (``configs/default.yaml``) decides how the ``evaluate``
step scores the frozen winners (researcher decision, 2026-09-07):

* ``true``  — user-level K-fold cross-validation with fold-in
  (:mod:`src.folds.runner`); the canonical per-user artifacts are keyed
  by the partition seed ``folds.seed``;
* ``false`` — the single leave-one-out split (:mod:`src.steps.evaluate`);
  the artifacts are keyed by the run seed.

The resolved choice is execution metadata: it is recorded in the run
manifest under ``evaluation_protocol`` and printed by ``--show-plan``,
and it never enters the scientific identity digest — a fold training
binds its fold through the identity context instead (E04/E07).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

MODE_KFOLD = "kfold"
MODE_SINGLE_SPLIT = "single_split"

#: Defaults mirror ``FoldsConfig`` (``src/utils/config_schema.py``).
_DEFAULT_K = 5
_DEFAULT_SEED = 42


@dataclass(frozen=True)
class EvaluationProtocol:
    """The protocol the evaluate step will run and the seed its artifacts carry."""

    mode: str
    k: int | None
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """Manifest-friendly plain dictionary."""
        return {"mode": self.mode, "k": self.k, "seed": self.seed}

    def describe(self) -> str:
        """One-line human description naming the YAML key that chose it."""
        if self.mode == MODE_KFOLD:
            return f"kfold (k={self.k}, partition seed={self.seed}; folds.enabled: true)"
        return f"single_split (run seed={self.seed}; folds.enabled: false)"


def resolve_evaluation_protocol(config: Mapping[str, Any]) -> EvaluationProtocol:
    """Resolve the evaluation protocol from ``folds.enabled``.

    :param config: The merged run configuration.
    :returns: :class:`EvaluationProtocol` for the run.
    :raises ValueError: If ``folds.enabled`` is not a boolean.
    """
    folds = config.get("folds") or {}
    enabled = folds.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(f"folds.enabled must be a boolean, got {enabled!r}")
    if enabled:
        return EvaluationProtocol(
            mode=MODE_KFOLD,
            k=int(folds.get("k", _DEFAULT_K)),
            seed=int(folds.get("seed", _DEFAULT_SEED)),
        )
    return EvaluationProtocol(
        mode=MODE_SINGLE_SPLIT, k=None, seed=int(config.get("seed", _DEFAULT_SEED))
    )


def artifact_seed(config: Mapping[str, Any]) -> int:
    """Seed the canonical per-user artifacts of this run are keyed by.

    Consumers that discover ``results/per_user/<dataset>/`` by seed
    (``beyond_accuracy``) must look under the partition seed when the
    K-fold protocol produced the artifacts.

    :param config: The merged run configuration.
    :returns: ``folds.seed`` under K-fold, the run ``seed`` otherwise.
    """
    return resolve_evaluation_protocol(config).seed
