"""Reproducibility helpers.

Provides a single function that seeds every relevant PRNG so that
experiments are deterministic across runs (given the same hardware and
library versions).
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set seeds for all random number generators used in the project.

    Seeds Python's built-in :mod:`random`, NumPy, and PyTorch (CPU and,
    when available, all CUDA devices).  Also configures cuDNN for
    deterministic behaviour.

    Parameters
    ----------
    seed:
        The integer seed value to use everywhere.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
