"""Deterministic seeding across python / numpy / torch.

Reproducibility is a first-class requirement for this project: the whole point
of the ablation tables is that two runs differing only in the variable under
test are otherwise identical. Fixing every RNG source is how we earn that.
"""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed python, numpy and (if available) torch.

    Args:
        seed: the integer seed.
        deterministic_torch: if True, also force cuDNN into deterministic mode.
            This can slow training slightly but removes a source of run-to-run
            variance that would otherwise pollute ablation comparisons.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # torch is optional for the pure-python utilities (metrics, randomization).
        pass
