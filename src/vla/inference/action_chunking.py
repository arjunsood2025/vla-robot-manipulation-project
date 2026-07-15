"""Action-chunk buffering with temporal ensembling.

Chunking policies (ACT, OpenVLA-OFT, SmolVLA) predict a short horizon of future
actions per forward pass instead of a single step. Two ways to consume a chunk:

  * Open-loop replay: execute all H actions, then re-query. Simple, but the
    joins between chunks are visibly jerky and the arm ignores the world during
    a chunk.
  * Temporal ensembling (ACT's trick): re-query every tick, and for each
    timestep average ALL the predictions that different (overlapping) chunks made
    for it, weighting older predictions by exp(-m * age). This smooths motion and
    lets fresh observations influence the command every tick.

This module implements the ensembler. It is policy-agnostic (it just consumes
chunks of shape (H, action_dim)) and pure-numpy, so it is unit-tested with no
model or hardware.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np


class TemporalEnsembler:
    """Buffers overlapping action chunks and returns the ensembled action for the
    current timestep.

    Args:
        action_dim: dimensionality of a single action.
        weight_m: exponential decay rate. exp(-m * age); smaller m = smoother
            (older predictions retain more influence). ACT uses m ~= 0.01.
    """

    def __init__(self, action_dim: int, weight_m: float = 0.01):
        self.action_dim = action_dim
        self.weight_m = weight_m
        self.t = 0
        # timestep -> list of (age_at_prediction, action_vector)
        self._predictions: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)

    def add_chunk(self, chunk: np.ndarray) -> None:
        """Register a chunk predicted for timesteps [t, t+H). ``age`` records how
        many steps into the chunk each action was, so newer chunks (predicted for
        the near future) can be weighted against older ones for the same step."""
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[1] != self.action_dim:
            raise ValueError(f"chunk must be (H, {self.action_dim}), got {chunk.shape}")
        h = chunk.shape[0]
        for age in range(h):
            self._predictions[self.t + age].append((age, chunk[age]))

    def step(self) -> np.ndarray:
        """Return the ensembled action for the current timestep and advance."""
        preds = self._predictions.pop(self.t, None)
        if not preds:
            raise RuntimeError(
                f"No prediction available for t={self.t}. Add a chunk before stepping."
            )
        ages = np.array([age for age, _ in preds], dtype=np.float64)
        actions = np.stack([a for _, a in preds], axis=0)   # (k, action_dim)
        weights = np.exp(-self.weight_m * ages)
        weights /= weights.sum()
        action = (weights[:, None] * actions).sum(axis=0)
        self.t += 1
        return action

    def pending_steps(self) -> int:
        """How many future timesteps still have at least one buffered prediction."""
        return sum(1 for k in self._predictions if k >= self.t)

    def reset(self) -> None:
        self.t = 0
        self._predictions.clear()
