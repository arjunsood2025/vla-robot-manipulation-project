"""Adapter around HuggingFace LeRobotDataset.

LeRobotDataset is the canonical on-disk format for this project (parquet for
low-dim signals + MP4 for video, one row per timestep with an episode index and
a per-episode `task` string). ACT / SmolVLA / OpenVLA all consume it directly;
our from-scratch BC baseline consumes it through the torch Dataset in
``dataset.py``. This module isolates the couple of places where we touch the
LeRobot API so a version bump only breaks in one file.

``lerobot`` is imported lazily so the pure-python utilities and unit tests do
not require it (it is a heavy, CUDA-adjacent dependency).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np


@dataclass
class EpisodeIndex:
    """Maps global frame indices to episodes so we can split by episode, not by
    frame — splitting by frame would leak frames of one demonstration across the
    train/val boundary."""
    episode_of_frame: np.ndarray   # (num_frames,) int episode id per frame
    frames_of_episode: dict[int, np.ndarray]

    @property
    def num_episodes(self) -> int:
        return len(self.frames_of_episode)


def _import_lerobot_dataset_cls():
    """Return the ``LeRobotDataset`` class across LeRobot layouts.

    LeRobot moved the module from ``lerobot.common.datasets`` (<=0.1.x) to
    ``lerobot.datasets`` (0.3+). This project is pinned to 0.4.1, but keeping
    both paths means a version bump breaks in exactly one place — which is the
    whole point of this adapter module.
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:  # pragma: no cover - legacy LeRobot layout
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    return LeRobotDataset


def load_lerobot_dataset(repo_id: str, root: str | None = None):
    """Instantiate a LeRobotDataset (downloading from the Hub unless ``root`` is
    a local dataset directory). Returns the LeRobot object untouched.

    Note the dataset must be in LeRobot codebase format v3.0; older v2.1 Hub
    datasets are converted once with
    ``python -m lerobot.datasets.v30.convert_dataset_v21_to_v30``.
    """
    LeRobotDataset = _import_lerobot_dataset_cls()
    return LeRobotDataset(repo_id, root=root)


def build_episode_index(dataset) -> EpisodeIndex:
    """Extract per-frame episode ids from a LeRobotDataset.

    LeRobot exposes an ``episode_index`` column per frame. We materialise it once
    so the BC dataset can do an episode-level train/val split cheaply.
    """
    epi = np.asarray(dataset.hf_dataset["episode_index"])
    frames_of_episode: dict[int, np.ndarray] = {}
    for ep in np.unique(epi):
        frames_of_episode[int(ep)] = np.where(epi == ep)[0]
    return EpisodeIndex(episode_of_frame=epi, frames_of_episode=frames_of_episode)


def iter_low_dim(dataset, state_key: str, action_key: str) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (state, action) numpy vectors per frame — used to fit NormStats
    without decoding any video (fast)."""
    states = np.asarray(dataset.hf_dataset[state_key], dtype=np.float32)
    actions = np.asarray(dataset.hf_dataset[action_key], dtype=np.float32)
    for s, a in zip(states, actions):
        yield np.asarray(s), np.asarray(a)


def collect_low_dim_arrays(dataset, state_key: str, action_key: str,
                           frame_indices: np.ndarray | None = None
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Return (states, actions) as (N, D) arrays, optionally restricted to a set
    of frame indices (e.g. the training split) for stat computation."""
    states = np.asarray(dataset.hf_dataset[state_key], dtype=np.float32)
    actions = np.asarray(dataset.hf_dataset[action_key], dtype=np.float32)
    if frame_indices is not None:
        states = states[frame_indices]
        actions = actions[frame_indices]
    return states, actions
