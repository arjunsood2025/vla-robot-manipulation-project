"""Uniform inference interface over the different policy families.

The evaluation harness and the policy server must not care whether they are
serving the from-scratch BC baseline, a LeRobot ACT/SmolVLA checkpoint, or an
OpenVLA-OFT model — they all answer the same question: given the current
observation, return an action chunk of shape (H, action_dim) in PHYSICAL units.
Each family gets a thin wrapper conforming to ``Policy`` below. This is exactly
the boundary a clean "add a new policy" PR would slot into.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Policy(Protocol):
    action_dim: int

    def act(self, observation: dict) -> np.ndarray:
        """observation keys: 'images' {cam_name: HWC uint8 array}, 'state'
        (n_joints float array, physical units), 'instruction' str.
        Returns (H, action_dim) physical-unit action chunk (H may be 1)."""
        ...


class BCPolicyWrapper:
    """Wraps the Phase-1 BC baseline (single-step, so H=1)."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        import torch

        from vla.models.bc_policy import BCPolicy, BCPolicyConfig
        from vla.data.normalization import DatasetStats, NormStats

        ckpt = torch.load(checkpoint_path, map_location=device)
        cfg = BCPolicyConfig(**ckpt["policy_config"])
        stats = DatasetStats(
            state=NormStats(**ckpt["stats"]["state"]),
            action=NormStats(**ckpt["stats"]["action"]),
        )
        self.model = BCPolicy(cfg, stats).to(device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.device = device
        self.action_dim = cfg.action_dim
        self.image_keys = ckpt.get("image_keys", ["observation.images.front",
                                                   "observation.images.wrist"])
        self._image_size = cfg.__dict__.get("image_size", 224)

    def act(self, observation: dict) -> np.ndarray:
        import torch
        from vla.data.dataset import IMAGENET_MEAN, IMAGENET_STD

        imgs = []
        for key in self.image_keys:
            cam = key.split(".")[-1]
            arr = np.asarray(observation["images"][cam])
            t = torch.as_tensor(arr).float()
            if t.ndim == 3 and t.shape[0] not in (1, 3):
                t = t.permute(2, 0, 1)
            if t.max() > 1.5:
                t = t / 255.0
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0), size=224, mode="bilinear", align_corners=False
            ).squeeze(0)
            t = (t - IMAGENET_MEAN) / IMAGENET_STD
            imgs.append(t)
        images = torch.stack(imgs, dim=0).unsqueeze(0).to(self.device)  # (1, n_cam, 3, H, W)

        state = torch.as_tensor(np.asarray(observation["state"]), dtype=torch.float32)
        state = state.unsqueeze(0).to(self.device)
        action = self.model.predict_action(images, state, [observation["instruction"]])
        return action.squeeze(0).cpu().numpy()[None, :]  # (1, action_dim)


class LeRobotPolicyWrapper:
    """Wraps any LeRobot policy (ACT, SmolVLA, pi0). LeRobot policies expose a
    ``select_action`` that already handles normalization and chunk buffering, so
    here we return the single action it hands back (H=1 from the caller's view;
    LeRobot does the internal chunking/ensembling)."""

    def __init__(self, checkpoint_dir: str, device: str = "cuda"):
        import torch
        from lerobot.common.policies.factory import make_policy

        self.device = device
        self.policy = make_policy(pretrained_path=checkpoint_dir).to(device)
        self.policy.eval()
        self.action_dim = self.policy.config.action_feature.shape[0]
        self._torch = torch

    def act(self, observation: dict) -> np.ndarray:
        torch = self._torch
        batch = {"observation.state": torch.as_tensor(
            np.asarray(observation["state"]), dtype=torch.float32).unsqueeze(0).to(self.device)}
        for cam, arr in observation["images"].items():
            t = torch.as_tensor(np.asarray(arr)).float()
            if t.ndim == 3 and t.shape[0] not in (1, 3):
                t = t.permute(2, 0, 1)
            if t.max() > 1.5:
                t = t / 255.0
            batch[f"observation.images.{cam}"] = t.unsqueeze(0).to(self.device)
        batch["task"] = [observation["instruction"]]
        with torch.no_grad():
            action = self.policy.select_action(batch)
        return action.squeeze(0).cpu().numpy()[None, :]


def load_policy(kind: str, path: str, device: str = "cuda") -> Policy:
    """Factory: kind in {'bc', 'act', 'smolvla', 'openvla'}."""
    if kind == "bc":
        return BCPolicyWrapper(path, device)
    if kind in ("act", "smolvla", "pi0"):
        return LeRobotPolicyWrapper(path, device)
    if kind == "openvla":
        from vla.inference.openvla_wrapper import OpenVLAPolicyWrapper

        return OpenVLAPolicyWrapper(path, device)
    raise ValueError(f"Unknown policy kind: {kind}")
