"""Evaluation statistics.

At 20 trials per condition the sampling error on a success rate is large, so we
never report a bare point estimate — every rate comes with a Wilson score 95%
confidence interval. The Wilson interval (not the textbook normal approximation)
is used deliberately: it stays inside [0, 1] and behaves sensibly at the extreme
rates (0/20, 20/20) that a working policy actually produces, where the normal
approximation gives nonsense like a negative lower bound.

Pure python + scipy for the z-quantile; unit-tested without torch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Wilson:
    point: float
    low: float
    high: float

    def as_pct(self) -> tuple[float, float, float]:
        return 100 * self.point, 100 * self.low, 100 * self.high

    def __str__(self) -> str:
        p, lo, hi = self.as_pct()
        return f"{p:.0f}% [{lo:.0f}, {hi:.0f}]"


def _z(confidence: float) -> float:
    """Two-sided z critical value. Uses scipy if present, else a small table."""
    try:
        from scipy.stats import norm

        return float(norm.ppf(1 - (1 - confidence) / 2))
    except ImportError:
        table = {0.90: 1.6449, 0.95: 1.9600, 0.99: 2.5758}
        return table.get(round(confidence, 2), 1.9600)


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Wilson:
    """Wilson score interval for a binomial proportion.

    Args:
        successes: number of successful trials.
        n: total trials. If 0, returns a degenerate (0, 0, 1) interval.
    """
    if n == 0:
        return Wilson(point=0.0, low=0.0, high=1.0)
    if successes < 0 or successes > n:
        raise ValueError(f"successes={successes} out of range for n={n}")

    z = _z(confidence)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return Wilson(point=p, low=max(0.0, center - half), high=min(1.0, center + half))


@dataclass
class ConditionResult:
    condition_id: str
    n: int
    successes: int
    partials: int
    failures: int

    @property
    def success_rate(self) -> Wilson:
        return wilson_interval(self.successes, self.n)

    @property
    def partial_or_success_rate(self) -> Wilson:
        return wilson_interval(self.successes + self.partials, self.n)


def paraphrase_gap(train_rate: float, heldout_rate: float) -> float:
    """Train-phrasing success minus held-out-phrasing success (in [0, 1]).

    A large positive gap means the policy leaned on the exact training wording
    rather than grounding the language — the single most telling number for the
    'does language generalize?' question.
    """
    return train_rate - heldout_rate
