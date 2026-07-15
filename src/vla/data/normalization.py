"""Compute and persist per-dimension normalization statistics.

Joint positions and joint-target actions live on very different numeric scales
(a shoulder sweep vs a gripper percentage). Feeding raw values to an MLP makes
the loss dominated by the widest-range dimension. We standardise every dim to
~zero mean / unit std using statistics computed on the TRAINING split only, then
apply the same transform at inference. Computing stats on train-only (never the
val or eval data) avoids leaking distribution information.

Pure numpy so the stats can be recomputed and inspected without torch.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


@dataclass
class NormStats:
    """Per-dimension mean and std for one vector quantity (state or action)."""
    mean: list[float]
    std: list[float]

    @classmethod
    def from_array(cls, x: np.ndarray, eps: float = 1e-6) -> "NormStats":
        """Fit stats from an (N, D) array. ``eps`` floors std to avoid /0 on
        dimensions that never move in the data (e.g. a locked joint)."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"Expected (N, D) array, got shape {x.shape}")
        mean = x.mean(axis=0)
        std = np.maximum(x.std(axis=0), eps)
        return cls(mean=mean.tolist(), std=std.tolist())

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - np.asarray(self.mean)) / np.asarray(self.std)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * np.asarray(self.std) + np.asarray(self.mean)


@dataclass
class DatasetStats:
    state: NormStats
    action: NormStats

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"state": asdict(self.state), "action": asdict(self.action)}, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "DatasetStats":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return cls(
            state=NormStats(**raw["state"]),
            action=NormStats(**raw["action"]),
        )
