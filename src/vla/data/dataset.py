"""Torch Dataset for the from-scratch behavior-cloning baseline.

Wraps a LeRobotDataset and yields, per frame:
    images       - (n_cameras, 3, H, W) float tensor, ImageNet-normalised
    state        - (state_dim,) standardised joint positions
    action       - (action_dim,) standardised joint-target (the label)
    instruction  - the raw task string (tokenised inside the model)

Design choices worth defending in review:
  * We predict a SINGLE next action here (not a chunk). Chunking is what ACT and
    the VLAs add on top; the baseline is deliberately the simplest thing that
    exercises the full data path, so its weaknesses motivate the upgrades.
  * Train/val split is by EPISODE, never by frame, to avoid temporally-adjacent
    frames of one demo leaking across the split and inflating val metrics.
  * Normalization stats are fit on the training frames only and passed in, so
    the same object is reused at inference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from vla.data.lerobot_adapter import build_episode_index, collect_low_dim_arrays
from vla.data.normalization import DatasetStats, NormStats

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


@dataclass
class BCDatasetConfig:
    image_keys: list[str]
    state_key: str
    action_key: str
    task_key: str
    image_size: int = 224
    val_fraction: float = 0.1
    seed: int = 42


def split_episodes(num_episodes: int, val_fraction: float, seed: int
                   ) -> tuple[list[int], list[int]]:
    """Deterministically partition episode ids into (train, val)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(num_episodes)
    n_val = max(1, int(round(num_episodes * val_fraction)))
    val = sorted(order[:n_val].tolist())
    train = sorted(order[n_val:].tolist())
    return train, val


class BCDataset(Dataset):
    """A frame-level view over selected episodes of a LeRobotDataset."""

    def __init__(
        self,
        lerobot_dataset,
        cfg: BCDatasetConfig,
        episode_ids: list[int],
        stats: DatasetStats,
        train: bool,
    ):
        self.ds = lerobot_dataset
        self.cfg = cfg
        self.stats = stats
        self.train = train

        index = build_episode_index(lerobot_dataset)
        frames = [index.frames_of_episode[ep] for ep in episode_ids]
        self.frame_ids = np.concatenate(frames) if frames else np.array([], dtype=int)

        # Augmentation is training-only; eval/inference must be deterministic.
        self._build_transforms()

    def _build_transforms(self):
        import torchvision.transforms as T

        size = self.cfg.image_size
        if self.train:
            self.transform = T.Compose([
                T.RandomResizedCrop(size, scale=(0.9, 1.0), antialias=True),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ])
        else:
            self.transform = T.Compose([
                T.Resize(size, antialias=True),
                T.CenterCrop(size),
            ])

    def __len__(self) -> int:
        return len(self.frame_ids)

    def _load_image(self, frame: dict, key: str) -> torch.Tensor:
        """LeRobot returns images as CHW float tensors in [0, 1]. Apply the
        spatial/appearance transform, then ImageNet-normalise."""
        img = frame[key]
        if not isinstance(img, torch.Tensor):
            img = torch.as_tensor(np.asarray(img))
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        if img.ndim == 3 and img.shape[0] not in (1, 3):  # HWC -> CHW
            img = img.permute(2, 0, 1)
        img = self.transform(img)
        return (img - IMAGENET_MEAN) / IMAGENET_STD

    def __getitem__(self, idx: int) -> dict:
        frame = self.ds[int(self.frame_ids[idx])]

        images = torch.stack([self._load_image(frame, k) for k in self.cfg.image_keys], dim=0)

        state = np.asarray(frame[self.cfg.state_key], dtype=np.float32)
        action = np.asarray(frame[self.cfg.action_key], dtype=np.float32)
        state = self.stats.state.normalize(state).astype(np.float32)
        action = self.stats.action.normalize(action).astype(np.float32)

        instruction = frame[self.cfg.task_key]
        if isinstance(instruction, (list, tuple)):
            instruction = instruction[0]

        return {
            "images": images,                       # (n_cam, 3, H, W)
            "state": torch.from_numpy(state),       # (state_dim,)
            "action": torch.from_numpy(action),     # (action_dim,)
            "instruction": instruction,             # str
        }


def bc_collate(batch: list[dict]) -> dict:
    """Collate that keeps instructions as a python list of strings (tokenised in
    the model) and stacks the tensor fields."""
    return {
        "images": torch.stack([b["images"] for b in batch]),
        "state": torch.stack([b["state"] for b in batch]),
        "action": torch.stack([b["action"] for b in batch]),
        "instruction": [b["instruction"] for b in batch],
    }


def fit_stats(lerobot_dataset, cfg: BCDatasetConfig, train_episode_ids: list[int]) -> DatasetStats:
    """Fit normalization stats on the training frames only."""
    index = build_episode_index(lerobot_dataset)
    train_frames = np.concatenate([index.frames_of_episode[ep] for ep in train_episode_ids])
    states, actions = collect_low_dim_arrays(
        lerobot_dataset, cfg.state_key, cfg.action_key, frame_indices=train_frames
    )
    return DatasetStats(
        state=NormStats.from_array(states),
        action=NormStats.from_array(actions),
    )
