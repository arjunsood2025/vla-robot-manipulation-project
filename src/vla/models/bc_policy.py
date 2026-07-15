"""Behavior-cloning baseline policy (Phase 1).

    [image feats ‖ text feats ‖ state] -> MLP -> next action (normalised)

This is intentionally the simplest model that touches every part of the system:
vision, language, proprioception, and a continuous action output. Its job is to
prove the data pipeline works and to be the honest floor that ACT / SmolVLA /
OpenVLA are measured against. It has no action chunking, no temporal model, and
no generative head — those are precisely the axes the later models improve, so
the baseline's failure modes are the story the report tells.

The model owns its action Normalizer, so a checkpoint is self-contained: load it,
call ``predict_action``, and you get a physical joint-target back with no
external stats file to misplace.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from vla.data.normalization import DatasetStats
from vla.models.encoders import CLIPTextEncoder, ResNetImageEncoder
from vla.models.normalization import Normalizer


@dataclass
class BCPolicyConfig:
    clip_model_name: str = "openai/clip-vit-base-patch32"
    n_cameras: int = 2
    state_dim: int = 6
    action_dim: int = 6
    hidden_dims: tuple[int, ...] = (512, 512, 256)
    dropout: float = 0.1
    pretrained_vision: bool = True


class BCPolicy(nn.Module):
    def __init__(self, cfg: BCPolicyConfig, stats: DatasetStats):
        super().__init__()
        self.cfg = cfg

        self.image_encoder = ResNetImageEncoder(pretrained=cfg.pretrained_vision)
        self.text_encoder = CLIPTextEncoder(cfg.clip_model_name)

        img_dim = self.image_encoder.output_dim(cfg.n_cameras)
        txt_dim = self.text_encoder.output_dim()
        in_dim = img_dim + txt_dim + cfg.state_dim

        layers: list[nn.Module] = []
        prev = in_dim
        for h in cfg.hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(cfg.dropout)]
            prev = h
        layers.append(nn.Linear(prev, cfg.action_dim))
        self.head = nn.Sequential(*layers)

        # Normalizers live in the module so they travel with the checkpoint.
        self.state_norm = Normalizer(stats.state)
        self.action_norm = Normalizer(stats.action)

    # ---- training-time forward: everything already normalised by the Dataset ----
    def forward(self, images: torch.Tensor, state: torch.Tensor,
                instruction: list[str]) -> torch.Tensor:
        """Return predicted NORMALISED action. Inputs are already normalised
        (the Dataset does it), matching how the loss is computed."""
        device = state.device
        img_feat = self.image_encoder(images)
        txt_feat = self.text_encoder(instruction, device).to(img_feat.dtype)
        x = torch.cat([img_feat, txt_feat, state], dim=-1)
        return self.head(x)

    # ---- inference: raw physical state in, raw physical action out ----
    @torch.no_grad()
    def predict_action(self, images: torch.Tensor, raw_state: torch.Tensor,
                       instruction: list[str]) -> torch.Tensor:
        """images: (B, n_cam, 3, H, W) ImageNet-normalised.
        raw_state: (B, state_dim) in PHYSICAL units (degrees).
        Returns (B, action_dim) in PHYSICAL units — ready for the safety filter."""
        self.eval()
        state = self.state_norm.normalize(raw_state)
        pred_norm = self.forward(images, state, instruction)
        return self.action_norm.denormalize(pred_norm)

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
