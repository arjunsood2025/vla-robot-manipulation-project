#!/usr/bin/env python
"""Lane A evaluation: held-out action-prediction error.

Without a robot there is no task-success rate to report, so the model ladder is
compared on the metric that *is* honestly measurable offline: how closely each
policy reproduces the demonstrator's action on episodes it never trained on.

What this does NOT claim
------------------------
This is an open-loop, single-step metric. Low action error does not guarantee a
policy would complete the task on hardware — errors compound over a closed-loop
rollout, and a policy can track the average demonstration while never actually
closing the gripper. It is a *relative* comparison between rungs of the ladder
under identical conditions, nothing more. Do not translate these numbers into
success rates.

Fairness rules enforced here
----------------------------
* Every model is scored on the same held-out episodes (``vla.data.splits``),
  which none of them trained on.
* Every model sees the same frames in the same order (fixed stride, fixed seed).
* Errors are reported in **physical units** (the dataset's action units), not in
  each model's own normalised space, so the numbers are comparable across
  models that normalise differently.
* Chunked policies (ACT, SmolVLA) are scored on the first action of the chunk,
  which is the one that would actually be executed at that timestep.

Usage:
    python scripts/eval_action_error.py \
        --model bc:outputs/bc_baseline_v1/bc_best.pt \
        --model act:outputs/act_v1/checkpoints/last \
        --model smolvla:outputs/smolvla_v1/checkpoints/last \
        --stride 10
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vla.data.lerobot_adapter import build_episode_index, load_lerobot_dataset  # noqa: E402
from vla.data.splits import train_val_episodes  # noqa: E402
from vla.utils.config import load_yaml  # noqa: E402
from vla.utils.seed import set_seed  # noqa: E402

CAMERA_KEYS = ["observation.images.front", "observation.images.wrist"]


def evaluate_model(kind: str, checkpoint: str, ds, frame_ids, tasks_by_frame,
                   device: str, rename_map: dict[str, str] | None):
    """Return per-frame absolute error arrays for one policy."""
    from vla.inference.policy_wrapper import load_policy

    policy = load_policy(kind, checkpoint, device=device, rename_map=rename_map)

    # Chunked LeRobot policies keep an internal action queue across calls; it
    # must be cleared between frames or we would score a stale queued action
    # against an unrelated timestep.
    reset = getattr(getattr(policy, "policy", None), "reset", None)

    abs_errors, task_of_row = [], []
    for n, fid in enumerate(frame_ids):
        frame = ds[int(fid)]
        obs = {
            "images": {k.split(".")[-1]: frame[k] for k in CAMERA_KEYS},
            "state": np.asarray(frame["observation.state"], dtype=np.float32),
            "instruction": tasks_by_frame[int(fid)],
        }
        if reset is not None:
            reset()
        pred = np.asarray(policy.act(obs), dtype=np.float64)[0]
        truth = np.asarray(frame["action"], dtype=np.float64)
        abs_errors.append(np.abs(pred - truth))
        task_of_row.append(tasks_by_frame[int(fid)])

        if (n + 1) % 100 == 0:
            print(f"    {n + 1}/{len(frame_ids)} frames", flush=True)

    return np.asarray(abs_errors), task_of_row


def summarize(abs_err: np.ndarray) -> dict:
    """Mean/median L1, RMSE and a bootstrap 95% CI on the mean L1."""
    per_frame = abs_err.mean(axis=1)          # mean over joints
    rng = np.random.default_rng(0)
    boot = [rng.choice(per_frame, per_frame.size, replace=True).mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n_frames": int(per_frame.size),
        "mae": float(per_frame.mean()),
        "mae_ci95": [float(lo), float(hi)],
        "median_ae": float(np.median(per_frame)),
        "rmse": float(np.sqrt((abs_err ** 2).mean())),
        "per_joint_mae": [float(x) for x in abs_err.mean(axis=0)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True,
                    help="kind:path, e.g. bc:outputs/bc_baseline_v1/bc_best.pt")
    ap.add_argument("--config", default="configs/train_bc.yaml",
                    help="Config that defines the dataset and the split.")
    ap.add_argument("--stride", type=int, default=10,
                    help="Score every Nth held-out frame (video decode is slow).")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/eval/action_error")
    args = ap.parse_args()

    set_seed(0)
    cfg = load_yaml(args.config)
    repo_id = cfg["dataset"]["repo_id"]

    train_eps, val_eps = train_val_episodes(
        repo_id,
        val_fraction=cfg["dataset"].get("val_fraction", 0.1),
        seed=cfg.get("seed", 42),
    )
    print(f"[eval] dataset {repo_id}")
    print(f"[eval] held-out episodes ({len(val_eps)}): {val_eps}")

    ds = load_lerobot_dataset(repo_id, root=cfg["dataset"].get("root"))
    index = build_episode_index(ds)
    frame_ids = np.concatenate([index.frames_of_episode[e] for e in val_eps])
    frame_ids = frame_ids[:: args.stride]
    print(f"[eval] scoring {len(frame_ids)} frames (stride {args.stride})")

    # Task strings come from the dataset itself, so the language conditioning is
    # exactly what the demonstrator was given.
    tasks_by_frame = {int(f): ds[int(f)]["task"] for f in frame_ids}

    smolvla_rename = {
        "observation.images.front": "observation.images.camera1",
        "observation.images.wrist": "observation.images.camera2",
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for spec in args.model:
        kind, _, ckpt = spec.partition(":")
        if not Path(ckpt).exists():
            print(f"[eval] SKIP {kind}: no checkpoint at {ckpt}")
            continue
        print(f"[eval] {kind} <- {ckpt}")
        abs_err, task_rows = evaluate_model(
            kind, ckpt, ds, frame_ids, tasks_by_frame, args.device,
            rename_map=smolvla_rename if kind == "smolvla" else None,
        )
        summary = summarize(abs_err)

        by_task = {}
        per_frame = abs_err.mean(axis=1)
        buckets = defaultdict(list)
        for e, t in zip(per_frame, task_rows):
            buckets[t].append(e)
        for t, vals in sorted(buckets.items()):
            by_task[t] = {"n_frames": len(vals), "mae": float(np.mean(vals))}
        summary["by_task"] = by_task
        results[kind] = summary

        np.savetxt(out_dir / f"{kind}_abs_errors.csv", abs_err, delimiter=",",
                   header=",".join(f"joint{i}" for i in range(abs_err.shape[1])),
                   comments="")
        print(f"    MAE {summary['mae']:.4f} "
              f"[{summary['mae_ci95'][0]:.4f}, {summary['mae_ci95'][1]:.4f}]  "
              f"RMSE {summary['rmse']:.4f}")

    meta = {
        "dataset": repo_id,
        "held_out_episodes": val_eps,
        "n_train_episodes": len(train_eps),
        "stride": args.stride,
        "n_frames_scored": int(len(frame_ids)),
        "metric": "mean absolute action error, physical units, open-loop single step",
        "results": results,
    }
    (out_dir / "action_error.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    lines = [
        "# Held-out action-prediction error (Lane A)",
        "",
        f"Dataset: `{repo_id}` — {len(train_eps)} train / {len(val_eps)} held-out episodes.",
        f"Scored {len(frame_ids)} held-out frames (every {args.stride}th), "
        "open-loop, single step, physical units.",
        "",
        "| Model | Frames | MAE | 95% CI | Median AE | RMSE |",
        "|---|---|---|---|---|---|",
    ]
    for kind, s in results.items():
        lines.append(
            f"| {kind} | {s['n_frames']} | {s['mae']:.4f} | "
            f"[{s['mae_ci95'][0]:.4f}, {s['mae_ci95'][1]:.4f}] | "
            f"{s['median_ae']:.4f} | {s['rmse']:.4f} |"
        )
    lines += [
        "",
        "Per-task MAE:",
        "",
        "| Model | " + " | ".join(sorted({t for s in results.values() for t in s["by_task"]})) + " |",
    ]
    task_names = sorted({t for s in results.values() for t in s["by_task"]})
    lines.append("|---" * (len(task_names) + 1) + "|")
    for kind, s in results.items():
        row = [f"{s['by_task'].get(t, {}).get('mae', float('nan')):.4f}" for t in task_names]
        lines.append(f"| {kind} | " + " | ".join(row) + " |")
    lines += [
        "",
        "> Open-loop action error is **not** a task-success rate. It compares how "
        "closely each policy reproduces held-out demonstrations; it does not "
        "establish that any of them would complete the task on hardware.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[eval] wrote {out_dir / 'report.md'} and {out_dir / 'action_error.json'}")


if __name__ == "__main__":
    main()
