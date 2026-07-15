#!/usr/bin/env python
"""OpenVLA-7B LoRA fine-tuning (OFT recipe) driver.

This orchestrates the OpenVLA-OFT fine-tuning run from our YAML config. The
model, LoRA wiring (via PEFT), and the continuous L1 action head come from the
OpenVLA / OpenVLA-OFT codebase, which must be installed alongside this project
(`pip install -e` their repo). We keep the hyperparameters in
configs/train_openvla_lora.yaml so the exact recipe is version-controlled next to
the results, and expand them into the OFT launcher's flags here.

Design choices worth defending:
  * LoRA (r=32) not full fine-tune: 7B params on one 24 GB GPU is only feasible
    with parameter-efficient tuning; LoRA also drastically cuts overfitting risk
    on a few-hundred-demo dataset.
  * OFT continuous head + action chunking + parallel decode instead of the
    original 256-bin token head: lifts control frequency from ~5 Hz to 25+ Hz,
    which is the difference between a jerky and a usable real-arm policy.
  * bf16 (QLoRA/4-bit only if the GPU is <24 GB): bf16 keeps the LoRA math stable
    without the extra quantisation error, when memory allows.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vla.utils.config import load_yaml  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune OpenVLA-7B with LoRA (OFT).")
    parser.add_argument("--config", default="configs/train_openvla_lora.yaml")
    parser.add_argument("--oft-launcher", default="vla-scripts/finetune.py",
                        help="Path to the OpenVLA-OFT finetune script in the OpenVLA repo.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    lora, head, train = cfg["lora"], cfg["action_head"], cfg["train"]

    cmd = [
        sys.executable, args.oft_launcher,
        f"--vla_path={cfg['base_model']}",
        f"--data_root_dir={cfg['dataset']['rlds_root']}",
        f"--run_root_dir={cfg['output_dir']}",
        "--use_lora=True",
        f"--lora_rank={lora['r']}",
        f"--lora_dropout={lora['dropout']}",
        "--use_l1_regression=True" if head["type"] == "l1_regression" else "--use_l1_regression=False",
        f"--num_actions_chunk={head['chunk_size']}",
        "--use_film=False",
        f"--batch_size={train['per_device_batch_size']}",
        f"--grad_accumulation_steps={train['grad_accum_steps']}",
        f"--learning_rate={train['lr']}",
        f"--max_steps={train['max_steps']}",
        f"--save_freq={train['save_every_steps']}",
        f"--use_quantization={'True' if train['use_qlora'] else 'False'}",
    ]
    print("[train_openvla] launching OFT fine-tune:")
    print("  " + " ".join(cmd))
    print("[train_openvla] NOTE: requires the OpenVLA-OFT repo installed and on PATH.")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
