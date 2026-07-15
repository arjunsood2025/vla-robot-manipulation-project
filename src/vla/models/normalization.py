"""Torch normalization module holding stats as buffers.

Storing mean/std as registered buffers (not parameters) means they move with
``.to(device)`` and are saved inside the checkpoint, so a loaded policy carries
its own normalization — there is no way to accidentally run inference with the
wrong stats, a classic and silent BC failure mode.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from vla.data.normalization import NormStats


class Normalizer(nn.Module):
    def __init__(self, stats: NormStats):
        super().__init__()
        self.register_buffer("mean", torch.tensor(stats.mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(stats.std, dtype=torch.float32))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean
