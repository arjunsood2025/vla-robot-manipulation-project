#!/usr/bin/env python
"""Generate the README/report figures from measured artifacts.

Every figure here is drawn from a file some earlier command actually produced —
no hand-entered numbers. If an input is missing the figure is skipped with a
message rather than invented.

    python scripts/make_figures.py

Inputs                                        Output
------                                        ------
outputs/eval/action_error/action_error.json   docs/results.png       (IMAGE 3)
outputs/train_logs/bc.log                     docs/training_curves.png
outputs/train_logs/act.log, smolvla.log       docs/training_curves.png
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

# One colour per rung, used consistently across every figure.
COLORS = {"bc": "#4C72B0", "act": "#DD8452", "smolvla": "#55A868"}
LABELS = {"bc": "BC (from scratch)", "act": "ACT", "smolvla": "SmolVLA (fine-tuned)"}


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)


def figure_action_error(out: Path) -> bool:
    src = ROOT / "outputs" / "eval" / "action_error" / "action_error.json"
    if not src.is_file():
        print(f"[figures] SKIP results.png — no {src.relative_to(ROOT)} yet "
              "(run scripts/eval_action_error.py first)")
        return False

    meta = json.loads(src.read_text(encoding="utf-8"))
    results = meta["results"]
    if not results:
        print("[figures] SKIP results.png — no models in action_error.json")
        return False

    kinds = [k for k in ("bc", "act", "smolvla") if k in results]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    # Left: overall MAE with bootstrap 95% CI.
    means = [results[k]["mae"] for k in kinds]
    los = [results[k]["mae"] - results[k]["mae_ci95"][0] for k in kinds]
    his = [results[k]["mae_ci95"][1] - results[k]["mae"] for k in kinds]
    ax1.bar(range(len(kinds)), means, yerr=[los, his], capsize=6,
            color=[COLORS.get(k, "#888") for k in kinds], width=0.6)
    ax1.set_xticks(range(len(kinds)))
    ax1.set_xticklabels([LABELS.get(k, k) for k in kinds], fontsize=9)
    ax1.set_ylabel("Mean absolute action error\n(physical units, lower is better)")
    ax1.set_title(f"Held-out action error\n{meta['n_frames_scored']} frames, "
                  f"{len(meta['held_out_episodes'])} unseen episodes", fontsize=10)
    # Label above the upper CI cap, not at the bar top, or the text sits on the
    # error bar.
    headroom = max(results[k]["mae_ci95"][1] for k in kinds) * 0.04
    for i, k in enumerate(kinds):
        ax1.text(i, results[k]["mae_ci95"][1] + headroom, f"{results[k]['mae']:.3f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_ylim(0, max(results[k]["mae_ci95"][1] for k in kinds) * 1.18)
    _style(ax1)

    # Right: per-task breakdown.
    tasks = sorted({t for k in kinds for t in results[k]["by_task"]})
    if tasks:
        width = 0.8 / max(len(kinds), 1)
        for i, k in enumerate(kinds):
            vals = [results[k]["by_task"].get(t, {}).get("mae", 0.0) for t in tasks]
            xs = [x + i * width for x in range(len(tasks))]
            ax2.bar(xs, vals, width=width, label=LABELS.get(k, k),
                    color=COLORS.get(k, "#888"))
        ax2.set_xticks([x + 0.4 - width / 2 for x in range(len(tasks))])
        # Task strings are full sentences; wrap them so the axis stays readable.
        ax2.set_xticklabels([t.replace(" and place into ", "\n→ ") for t in tasks],
                            fontsize=7)
        ax2.set_ylabel("MAE (physical units)")
        ax2.set_title("Per-task held-out action error", fontsize=10)
        ax2.legend(fontsize=8, frameon=False)
        _style(ax2)

    fig.suptitle("Open-loop action error on held-out demonstrations — NOT a task-success rate",
                 fontsize=9, style="italic", y=0.005, color="#555")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {out.relative_to(ROOT)}")
    return True


def _parse_bc_log(path: Path):
    """(epoch, val_mse) pairs from the BC training log."""
    if not path.is_file():
        return []
    pat = re.compile(r"epoch (\d+): val MSE = ([0-9.]+)")
    return [(int(m.group(1)), float(m.group(2)))
            for m in (pat.search(l) for l in path.read_text(errors="ignore").splitlines()) if m]


_SUFFIX = {"": 1, "K": 1_000, "M": 1_000_000}


def _parse_lerobot_log(path: Path):
    """(step, loss) pairs from a LeRobot training log.

    LeRobot abbreviates large step counts ("250", then "1K" ... "100K"), so the
    suffix must be expanded — reading "10K" as 10 silently interleaves early and
    late steps and draws a zig-zag instead of a loss curve.
    """
    if not path.is_file():
        return []
    pat = re.compile(r"step:([0-9.]+)([KM]?)\s.*?loss:([0-9.]+)")
    out = []
    for line in path.read_text(errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            step = int(float(m.group(1)) * _SUFFIX[m.group(2)])
            out.append((step, float(m.group(3))))
    # Abbreviated steps collide (many lines report "12K"); keep the last loss
    # logged at each distinct step so the curve is monotonic in x.
    dedup: dict[int, float] = {}
    for step, loss in out:
        dedup[step] = loss
    return sorted(dedup.items())


def figure_training_curves(out: Path) -> bool:
    logs = ROOT / "outputs" / "train_logs"
    bc = _parse_bc_log(logs / "bc.log")
    act = _parse_lerobot_log(logs / "act.log")
    smol = _parse_lerobot_log(logs / "smolvla.log")
    if not (bc or act or smol):
        print("[figures] SKIP training_curves.png — no training logs yet")
        return False

    panels = [p for p in (("bc", bc), ("act", act), ("smolvla", smol)) if p[1]]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6), squeeze=False)
    for ax, (kind, data) in zip(axes[0], panels):
        xs, ys = zip(*data)
        ax.plot(xs, ys, color=COLORS.get(kind, "#888"), linewidth=1.8)
        ax.set_title(LABELS.get(kind, kind), fontsize=10)
        ax.set_xlabel("epoch" if kind == "bc" else "step")
        ax.set_ylabel("held-out MSE" if kind == "bc" else "training loss")
        if kind != "bc" and min(ys) > 0:
            ax.set_yscale("log")
        _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {out.relative_to(ROOT)}")
    return True


JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]


def figure_dataset(out: Path, config: str) -> bool:
    """Per-joint action histograms + a sample frame per task.

    This is the pre-training sanity check from the notebook, saved as a figure:
    it shows the action distributions are sane and that each task's language
    string matches what the camera actually sees.
    """
    sys_path_added = str(ROOT / "src")
    import sys

    if sys_path_added not in sys.path:
        sys.path.insert(0, sys_path_added)

    import numpy as np

    from vla.data.lerobot_adapter import (build_episode_index,
                                          collect_low_dim_arrays,
                                          load_lerobot_dataset)
    from vla.utils.config import load_yaml

    cfg = load_yaml(config)
    repo_id = cfg["dataset"]["repo_id"]
    try:
        ds = load_lerobot_dataset(repo_id, root=cfg["dataset"].get("root"))
    except Exception as exc:  # dataset not downloaded on this machine
        print(f"[figures] SKIP dataset.png — could not load {repo_id}: {exc}")
        return False

    index = build_episode_index(ds)
    _, actions = collect_low_dim_arrays(ds, cfg["dataset"]["state_key"],
                                        cfg["dataset"]["action_key"])

    # One representative frame per distinct task string.
    first_frame_of_task: dict[str, int] = {}
    for ep in sorted(index.frames_of_episode):
        fid = int(index.frames_of_episode[ep][0])
        task = ds[fid]["task"]
        first_frame_of_task.setdefault(task, fid)
    tasks = sorted(first_frame_of_task)

    # 6 base columns so the task strip and the 3-wide histogram grid both
    # divide the full width evenly (no dangling empty column).
    n_tasks = max(len(tasks), 1)
    ncol = 6
    fig = plt.figure(figsize=(3.1 * max(n_tasks, 3), 6.6))
    gs = fig.add_gridspec(3, ncol, height_ratios=[0.78, 1, 1], hspace=0.42, wspace=0.35)

    span = ncol // n_tasks if n_tasks <= ncol else 1
    for i, task in enumerate(tasks):
        ax = fig.add_subplot(gs[0, i * span:(i + 1) * span])
        img = np.asarray(ds[first_frame_of_task[task]]["observation.images.front"])
        if img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(task.replace(" and place into ", "\n→ "), fontsize=8)

    for j, name in enumerate(JOINT_NAMES[: actions.shape[1]]):
        ax = fig.add_subplot(gs[1 + j // 3, (j % 3) * 2:(j % 3) * 2 + 2])
        ax.hist(actions[:, j], bins=60, color="#4C72B0")
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
        _style(ax)

    fig.suptitle(
        f"{repo_id} — {index.num_episodes} episodes, {len(actions):,} frames, "
        f"{len(tasks)} language tasks\nfront camera per task (top); "
        "per-joint action distributions (bottom)",
        fontsize=10,
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {out.relative_to(ROOT)}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(DOCS))
    ap.add_argument("--config", default="configs/train_bc.yaml")
    ap.add_argument("--skip-dataset", action="store_true",
                    help="Skip the dataset figure (it decodes video, so it is slow).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_action_error(out_dir / "results.png")
    figure_training_curves(out_dir / "training_curves.png")
    if not args.skip_dataset:
        figure_dataset(out_dir / "dataset.png", args.config)


if __name__ == "__main__":
    main()
