"""Uniform inference interface over the different policy families.

The evaluation harness and the policy server must not care whether they are
serving the from-scratch BC baseline, a LeRobot ACT/SmolVLA checkpoint, or an
OpenVLA-OFT model — they all answer the same question: given the current
observation, return an action chunk of shape (H, action_dim) in PHYSICAL units.
Each family gets a thin wrapper conforming to ``Policy`` below. This is exactly
the boundary a clean "add a new policy" PR would slot into.
"""
from __future__ import annotations

from pathlib import Path
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

    def __init__(self, checkpoint_dir: str, device: str = "cuda",
                 rename_map: dict[str, str] | None = None,
                 ds_repo_id: str | None = None):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import (get_policy_class,
                                              make_pre_post_processors)

        # LeRobot writes checkpoints as <ckpt>/pretrained_model/; accept either
        # that directory or its parent so callers can pass whichever they have.
        path = Path(checkpoint_dir)
        if (path / "pretrained_model").is_dir():
            path = path / "pretrained_model"
        path = str(path)

        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.pretrained_path = path

        self.device = device
        self.rename_map = rename_map or {}

        # LeRobot's make_policy() insists on dataset metadata (or a sim env)
        # purely to derive input/output feature shapes. A *trained* checkpoint
        # already records those in its own config, so requiring the training
        # dataset just to serve a policy would be a needless deployment
        # dependency — the policy server should run on a machine that has the
        # weights and nothing else. Build it straight from the checkpoint when
        # the features are present, and only fall back to the dataset when they
        # are not.
        if cfg.input_features and cfg.output_features:
            self.policy = get_policy_class(cfg.type).from_pretrained(
                pretrained_name_or_path=path, config=cfg
            ).to(device)
        else:
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
            from lerobot.policies.factory import make_policy

            if ds_repo_id is None:
                raise ValueError(
                    f"{path} has no input/output features in its config; pass "
                    "ds_repo_id so the shapes can be read from dataset metadata."
                )
            self.policy = make_policy(
                cfg=cfg,
                ds_meta=LeRobotDatasetMetadata(ds_repo_id),
                rename_map=self.rename_map,
            ).to(device)
        self.policy.eval()

        # Normalization/unnormalization live in these pipelines in LeRobot
        # 0.4.x, not inside the model, so inference must run through them or
        # actions come back in normalised units.
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=path,
            preprocessor_overrides={
                "device_processor": {"device": str(device)},
                "rename_observations_processor": {"rename_map": self.rename_map},
            },
        )
        self.action_dim = self.policy.config.action_feature.shape[0]
        self._torch = torch

    def _build_batch(self, observation: dict):
        torch = self._torch
        batch = {"observation.state": torch.as_tensor(
            np.asarray(observation["state"]), dtype=torch.float32).unsqueeze(0)}
        for cam, arr in observation["images"].items():
            t = torch.as_tensor(np.asarray(arr)).float()
            if t.ndim == 3 and t.shape[0] not in (1, 3):
                t = t.permute(2, 0, 1)
            if t.max() > 1.5:
                t = t / 255.0
            batch[f"observation.images.{cam}"] = t.unsqueeze(0)
        batch["task"] = observation["instruction"]
        return batch

    def act(self, observation: dict) -> np.ndarray:
        torch = self._torch
        batch = self._build_batch(observation)
        with torch.no_grad():
            action = self.policy.select_action(self.preprocessor(batch))
            action = self.postprocessor(action)
        return action.squeeze(0).float().cpu().numpy()[None, :]


def load_policy(kind: str, path: str, device: str = "cuda",
                rename_map: dict[str, str] | None = None,
                ds_repo_id: str | None = None) -> Policy:
    """Factory: kind in {'bc', 'act', 'smolvla', 'openvla'}.

    ``rename_map`` maps this dataset's camera keys onto the ones a pretrained
    checkpoint expects (SmolVLA's base model wants camera1/camera2); it is
    ignored by the families that do not need it. ``ds_repo_id`` is only needed
    for a LeRobot checkpoint whose config lacks feature shapes (see
    LeRobotPolicyWrapper); trained checkpoints carry their own.
    """
    if kind == "bc":
        return BCPolicyWrapper(path, device)
    if kind in ("act", "smolvla", "pi0"):
        return LeRobotPolicyWrapper(path, device, rename_map=rename_map,
                                    ds_repo_id=ds_repo_id)
    if kind == "openvla":
        from vla.inference.openvla_wrapper import OpenVLAPolicyWrapper

        return OpenVLAPolicyWrapper(path, device)
    raise ValueError(f"Unknown policy kind: {kind}")
