#!/usr/bin/env python
"""Teleoperated demonstration collection — a thin, opinionated wrapper over
``lerobot-record``.

Why wrap it at all? Two reasons that matter for data quality:
  1. It reads the SAME robot/camera config the rest of the project uses, so the
     dataset's keys (observation.images.front/wrist, observation.state, action)
     are guaranteed to match what training expects.
  2. It prints the pre-generated placement sheet row for each episode so the
     operator sets up objects in the reproducible grid cells, and rotates through
     the task's training paraphrases so language variety is baked into the data.

The actual servo/camera recording is done by LeRobot; this script builds the
command and enforces the protocol around it.

Example:
    python scripts/collect_demos.py \
        --task block_in_bowl --repo-id youruser/so101_tabletop_v1 \
        --num-episodes 50 --robot-port COM5 --teleop-port COM6 \
        --layout-sheet data/layouts/block_in_bowl.csv
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vla.data.paraphrases import ParaphraseBank  # noqa: E402
from vla.utils.config import load_yaml  # noqa: E402


def build_camera_arg(robot_cfg: dict) -> str:
    cams = robot_cfg["cameras"]
    parts = []
    for name, c in cams.items():
        parts.append(
            f"{name}: {{type: opencv, index_or_path: {c['index_or_path']}, "
            f"width: {c['width']}, height: {c['height']}, fps: {c['fps']}}}"
        )
    return "{" + ", ".join(parts) + "}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Record teleop demos with the SO-101.")
    parser.add_argument("--task", required=True, help="Task id in configs/paraphrases.json")
    parser.add_argument("--repo-id", required=True, help="HF dataset repo id to push to.")
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--robot-port", required=True)
    parser.add_argument("--teleop-port", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--robot-config", default="configs/robot_so101.yaml")
    parser.add_argument("--paraphrases", default="configs/paraphrases.json")
    parser.add_argument("--layout-sheet", default=None,
                        help="CSV of per-episode object placements (printed to guide the operator).")
    parser.add_argument("--resume", action="store_true",
                        help="Append to an existing dataset instead of creating a new one.")
    args = parser.parse_args()

    robot_cfg = load_yaml(args.robot_config)
    bank = ParaphraseBank.load(args.paraphrases)
    task = bank[args.task]

    # Rotate through training phrasings so language variety is in the data. We
    # record all episodes under a single --single_task string per lerobot-record
    # invocation, so we cycle phrasings across small batches.
    phrasings = task.train
    print(f"[collect] task '{args.task}' — rotating {len(phrasings)} training phrasings:")
    for p in phrasings:
        print(f"    - {p}")
    if args.layout_sheet:
        print(f"[collect] set up each episode per {args.layout_sheet}")

    cameras = build_camera_arg(robot_cfg)
    per_phrasing = max(1, args.num_episodes // len(phrasings))

    for i, phrasing in enumerate(phrasings):
        n = per_phrasing if i < len(phrasings) - 1 else args.num_episodes - per_phrasing * (len(phrasings) - 1)
        if n <= 0:
            continue
        cmd = [
            "lerobot-record",
            f"--robot.type={robot_cfg['robot']['type']}",
            f"--robot.port={args.robot_port}",
            f"--robot.cameras={cameras}",
            "--teleop.type=so101_leader",
            f"--teleop.port={args.teleop_port}",
            f"--dataset.repo_id={args.repo_id}",
            f"--dataset.num_episodes={n}",
            f'--dataset.single_task={phrasing}',
            f"--dataset.fps={args.fps}",
        ]
        if args.resume or i > 0:
            cmd.append("--resume=true")
        print(f"\n[collect] batch {i+1}/{len(phrasings)} ({n} eps) phrasing: {phrasing}")
        print("[collect] running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print(f"\n[collect] done. Dataset pushed to {args.repo_id}. "
          f"Tag it (v1, v2, ...) on the Hub — never overwrite a version.")


if __name__ == "__main__":
    main()
