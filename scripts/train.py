#!/usr/bin/env python
"""Unified training entrypoint that dispatches to the right backend per policy.

  bc       -> our from-scratch loop in vla.training.train_bc
  act      -> lerobot-train (ACT)
  smolvla  -> lerobot-train (SmolVLA fine-tune)
  openvla  -> the OpenVLA-OFT LoRA fine-tuning script (scripts/train_openvla.py)

Having one entrypoint means the README shows a single command shape and the
per-policy details live in the YAML configs, which are the artifacts you commit
alongside results.

Examples:
    python scripts/train.py --policy bc      --config configs/train_bc.yaml
    python scripts/train.py --policy act     --config configs/train_act.yaml
    python scripts/train.py --policy smolvla --config configs/train_smolvla.yaml
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Make ``src`` importable when run as a plain script (no install needed).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vla.utils.config import load_yaml  # noqa: E402


def train_bc(config: str, overrides: list[str]) -> None:
    from vla.training.train_bc import train

    cfg = load_yaml(config)
    if overrides:
        from vla.utils.config import deep_update, parse_overrides

        cfg = deep_update(cfg, parse_overrides(overrides))
    train(cfg)


def train_lerobot(policy: str, config: str) -> None:
    """Shell out to the LeRobot trainer, translating our YAML to its CLI flags.

    We keep the canonical hyperparameters in YAML (version-controlled next to
    results) and expand them into the LeRobot CLI here, so there is a single
    source of truth even though the actual training is LeRobot's."""
    cfg = load_yaml(config)
    ds = cfg["dataset"]["repo_id"]
    out = cfg["output_dir"]

    if policy == "act":
        p, t = cfg["policy"], cfg["train"]
        cmd = [
            "lerobot-train", "--policy.type=act", f"--dataset.repo_id={ds}",
            f"--batch_size={t['batch_size']}", f"--steps={t['steps']}",
            f"--policy.chunk_size={p['chunk_size']}",
            f"--policy.n_action_steps={p['n_action_steps']}",
            f"--policy.kl_weight={p['kl_weight']}",
            f"--policy.optimizer_lr={t['optimizer_lr']}",
            f"--output_dir={out}", "--wandb.enable=true",
        ]
    elif policy == "smolvla":
        p, t = cfg["policy"], cfg["train"]
        cmd = [
            "lerobot-train", f"--policy.path={p['path']}", f"--dataset.repo_id={ds}",
            f"--batch_size={t['batch_size']}", f"--steps={t['steps']}",
            f"--output_dir={out}", "--wandb.enable=true",
        ]
    else:
        raise ValueError(policy)

    print("[train] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def train_openvla(config: str) -> None:
    cmd = [sys.executable, str(Path(__file__).with_name("train_openvla.py")), "--config", config]
    print("[train] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a manipulation policy.")
    parser.add_argument("--policy", required=True,
                        choices=["bc", "act", "smolvla", "openvla"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", nargs="*", default=[])
    args = parser.parse_args()

    if args.policy == "bc":
        train_bc(args.config, args.override)
    elif args.policy in ("act", "smolvla"):
        train_lerobot(args.policy, args.config)
    elif args.policy == "openvla":
        train_openvla(args.config)


if __name__ == "__main__":
    main()
