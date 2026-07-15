"""OpenVLA-OFT inference wrapper.

Loads the base OpenVLA-7B model plus the LoRA adapter trained in Phase 4 and the
OFT continuous action head, and exposes the same ``act`` interface as the other
policies. OpenVLA consumes a single third-person RGB image + the instruction and
(under OFT) emits a continuous action chunk via a parallel-decoded L1 head, which
we un-normalise with the dataset action statistics saved at training time.

Kept in its own module because it drags in the heaviest dependencies (the 7B
backbone, PEFT); nothing imports it unless an OpenVLA checkpoint is actually served.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class OpenVLAPolicyWrapper:
    def __init__(self, checkpoint_dir: str, device: str = "cuda"):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForVision2Seq, AutoProcessor

        ckpt = Path(checkpoint_dir)
        self.device = device
        self.processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
        base = AutoModelForVision2Seq.from_pretrained(
            ckpt / "base", torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(base, ckpt / "lora").to(device).eval()

        stats_path = ckpt / "dataset_statistics.json"
        stats = json.loads(stats_path.read_text())
        self.action_mean = np.array(stats["action"]["mean"], dtype=np.float64)
        self.action_std = np.array(stats["action"]["std"], dtype=np.float64)
        self.action_dim = len(self.action_mean)
        self._torch = torch

    def act(self, observation: dict) -> np.ndarray:
        torch = self._torch
        # OpenVLA uses a single third-person view; prefer the front camera.
        image = np.asarray(observation["images"]["front"])
        prompt = f"In: What action should the robot take to {observation['instruction']}?\nOut:"
        inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        with torch.no_grad():
            norm_chunk = self.model.predict_action(**inputs)  # (H, action_dim) normalised
        norm_chunk = np.asarray(norm_chunk, dtype=np.float64).reshape(-1, self.action_dim)
        return norm_chunk * self.action_std + self.action_mean
