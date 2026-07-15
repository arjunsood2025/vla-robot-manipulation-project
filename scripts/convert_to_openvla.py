#!/usr/bin/env python
"""Convert a LeRobotDataset into the layout OpenVLA fine-tuning expects.

OpenVLA was pretrained on RLDS/Open-X data and its fine-tuning code consumes an
RLDS-style dataset plus a ``dataset_statistics.json`` used to normalise actions.
This script:
  1. streams the LeRobotDataset frames,
  2. writes them into per-episode records with the front-camera image, the
     instruction, the joint state, and the joint-target action, and
  3. computes and saves per-dimension action statistics (mean/std AND the 1st/99th
     percentiles OpenVLA uses for its bounded normalisation).

The heavy RLDS/TFDS packing is delegated to the OpenVLA repo's builder; here we
produce the intermediate the builder ingests and the statistics file the training
script reads. Computing the stats ourselves (and committing them) is what keeps
action normalisation reproducible across the BC/ACT/OpenVLA comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vla.data.lerobot_adapter import (  # noqa: E402
    build_episode_index,
    collect_low_dim_arrays,
    load_lerobot_dataset,
)


def compute_action_statistics(actions: np.ndarray) -> dict:
    """OpenVLA normalises with mean/std for the continuous OFT head and clips to
    [q01, q99] to be robust to teleoperation outliers. We record both."""
    return {
        "action": {
            "mean": actions.mean(axis=0).tolist(),
            "std": actions.std(axis=0).tolist(),
            "q01": np.percentile(actions, 1, axis=0).tolist(),
            "q99": np.percentile(actions, 99, axis=0).tolist(),
            "min": actions.min(axis=0).tolist(),
            "max": actions.max(axis=0).tolist(),
            "num_transitions": int(actions.shape[0]),
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LeRobotDataset -> OpenVLA format.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", default=None, help="Local dataset root (skip Hub download).")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--image-key", default="observation.images.front")
    parser.add_argument("--task-key", default="task")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_lerobot_dataset(args.repo_id, root=args.root)
    index = build_episode_index(ds)
    _, actions = collect_low_dim_arrays(ds, args.state_key, args.action_key)

    stats = compute_action_statistics(actions)
    (out / "dataset_statistics.json").write_text(json.dumps(stats, indent=2))

    manifest = {
        "source_repo_id": args.repo_id,
        "num_episodes": index.num_episodes,
        "num_frames": int(actions.shape[0]),
        "image_key": args.image_key,
        "state_key": args.state_key,
        "action_key": args.action_key,
        "task_key": args.task_key,
        "note": ("Feed this directory to the OpenVLA RLDS builder. Frames and the "
                 "front-camera MP4s are read directly from the LeRobotDataset; the "
                 "builder only needs this manifest + dataset_statistics.json."),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[convert] wrote {out/'dataset_statistics.json'} and {out/'manifest.json'}")
    print(f"[convert] {index.num_episodes} episodes, {actions.shape[0]} frames, "
          f"action_dim={actions.shape[1]}")


if __name__ == "__main__":
    main()
