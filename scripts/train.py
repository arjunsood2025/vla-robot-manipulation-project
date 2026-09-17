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
import json
import os
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


def _resolve_pretrained(path: str) -> str:
    """Return a local directory for a pretrained policy checkpoint.

    ``--policy.path`` is parsed by LeRobot as a filesystem path, so on Windows a
    Hub id like ``lerobot/smolvla_base`` gets normalised to
    ``lerobot\\smolvla_base`` and then rejected as an invalid repo id. Resolving
    the id to its local snapshot ourselves sidesteps that, and has the side
    benefit of making the training run offline-reproducible once cached.
    """
    if Path(path).is_dir():
        return str(Path(path).resolve())

    from huggingface_hub import snapshot_download

    return str(Path(snapshot_download(path)).resolve())


def train_lerobot(policy: str, config: str) -> None:
    """Shell out to the LeRobot trainer, translating our YAML to its CLI flags.

    We keep the canonical hyperparameters in YAML (version-controlled next to
    results) and expand them into the LeRobot CLI here, so there is a single
    source of truth even though the actual training is LeRobot's."""
    cfg = load_yaml(config)
    ds = cfg["dataset"]["repo_id"]
    out = cfg["output_dir"]
    t = cfg["train"]
    log = cfg.get("logging", {})

    # Train on exactly the episodes the BC baseline trains on. LeRobot would
    # otherwise use all of them, including the held-out ones, and every
    # "held-out action error" we later report for ACT/SmolVLA would be measured
    # on episodes the model had already fitted. See vla.data.splits.
    from vla.data.splits import train_val_episodes

    train_eps, val_eps = train_val_episodes(
        ds,
        val_fraction=cfg["dataset"].get("val_fraction", 0.1),
        seed=cfg.get("seed", 42),
    )
    print(f"[train] holding out {len(val_eps)} episodes: {val_eps}")

    common = [
        f"--dataset.repo_id={ds}",
        f"--dataset.episodes={json.dumps(train_eps, separators=(',', ':'))}",
        f"--batch_size={t['batch_size']}",
        f"--steps={t['steps']}",
        f"--output_dir={out}",
        f"--num_workers={t.get('num_workers', 8)}",
        f"--save_freq={t.get('save_freq', 5000)}",
        f"--log_freq={t.get('log_freq', 100)}",
        # LeRobot 0.4.x refuses to start unless push-to-hub is either configured
        # with a repo id or explicitly disabled. We keep weights local and
        # publish deliberately, not as a training side effect.
        "--policy.push_to_hub=false",
        f"--wandb.enable={'true' if log.get('backend', 'wandb') == 'wandb' else 'false'}",
    ]
    if log.get("backend") == "wandb":
        common += [f"--wandb.project={log.get('project', 'vla-manipulation')}"]

    if policy == "act":
        p = cfg["policy"]
        args = [
            "--policy.type=act",
            f"--policy.chunk_size={p['chunk_size']}",
            f"--policy.n_action_steps={p['n_action_steps']}",
            f"--policy.kl_weight={p['kl_weight']}",
            f"--policy.optimizer_lr={t['optimizer_lr']}",
            *common,
        ]
    elif policy == "smolvla":
        p = cfg["policy"]
        args = [
            f"--policy.path={_resolve_pretrained(p['path'])}",
            f"--policy.optimizer_lr={t['optimizer_lr']}",
            *common,
        ]
    else:
        raise ValueError(policy)

    # Camera-name remapping (e.g. this dataset's front/wrist -> SmolVLA's
    # camera1/camera2). LeRobot takes it as a JSON object on the CLI.
    if cfg.get("rename_map"):
        args.append("--rename_map=" + json.dumps(cfg["rename_map"]))

    # Go through our launcher rather than the ``lerobot-train`` console script:
    # it installs the Windows symlink fallback (see vla.training.lerobot_launch)
    # without which every checkpoint save crashes the run on Windows.
    cmd = [sys.executable, "-m", "vla.training.lerobot_launch", *args]
    print("[train] running:", " ".join(cmd))
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, env=env)


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
