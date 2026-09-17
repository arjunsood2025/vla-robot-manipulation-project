#!/usr/bin/env python
"""Measure per-policy inference latency, and the action-chunking speedup.

Two numbers come out of this, both measurable without a robot:

1. **Inference latency** — wall-clock for one ``policy.act()`` call on the GPU:
   image preprocessing, forward pass, and un-normalisation. This is the
   *policy-server* half of the control loop. It excludes camera capture and the
   network hop to the robot, which need hardware; the end-to-end loop figure
   comes from ``demo_offline.py`` instead. Reported as mean / p50 / p95 over a
   fixed number of timed calls after warm-up.

2. **Chunking headroom** — ACT and SmolVLA predict a *chunk* of `n_action_steps`
   actions per forward pass, so the policy is queried once per chunk instead of
   once per control tick. This does NOT raise the control rate: the arm runs at
   a fixed 30 Hz and cannot move faster. What it buys is time — one query must
   merely finish before its chunk finishes executing. Reported as the chunk's
   wall-clock duration, the duty cycle (compute / chunk duration), and whether
   p95 latency fits inside that budget.

    python scripts/bench_latency.py \
        --model bc:outputs/bc_baseline_v1/bc_best.pt \
        --model act:outputs/act_v1/checkpoints/last
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The SO-101 dataset was recorded at 30 fps, so a deployed loop runs at 30 Hz.
CONTROL_HZ = 30.0

CAMERAS = ("front", "wrist")
INSTRUCTION = "Grab pens and place into pen holder"


def bench(kind: str, checkpoint: str, device: str, n_warmup: int, n_iters: int,
          rename_map: dict[str, str] | None, text_cache: bool = True):
    import torch

    from vla.inference.policy_wrapper import load_policy

    policy = load_policy(kind, checkpoint, device=device, rename_map=rename_map)

    # The BC baseline encodes its instruction with a frozen CLIP text tower.
    # Toggling the cache off measures what that costs per control tick.
    text_encoder = getattr(getattr(policy, "model", None), "text_encoder", None)
    if text_encoder is not None:
        text_encoder._cache_enabled = text_cache
        text_encoder.clear_cache()

    rng = np.random.default_rng(0)
    obs = {
        "images": {c: rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for c in CAMERAS},
        "state": rng.normal(0, 1, 6).astype(np.float32),
        "instruction": INSTRUCTION,
    }

    reset = getattr(getattr(policy, "policy", None), "reset", None)

    for _ in range(n_warmup):
        if reset is not None:
            reset()
        policy.act(obs)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    times = []
    for _ in range(n_iters):
        # Reset the chunk queue so every timed call is a real forward pass, not
        # a cheap pop from a queue LeRobot filled on an earlier call. This
        # measures worst-case (per-chunk) latency, which is the honest number.
        if reset is not None:
            reset()
        t0 = time.perf_counter()
        policy.act(obs)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(times)
    n_action_steps = 1
    cfg = getattr(getattr(policy, "policy", None), "config", None)
    if cfg is not None:
        n_action_steps = int(getattr(cfg, "n_action_steps", 1) or 1)

    mean_ms = float(arr.mean())

    # What chunking actually buys is NOT a higher control rate -- the arm runs at
    # a fixed CONTROL_HZ and cannot execute actions faster than it moves. It buys
    # the right to query the policy less often. So the meaningful quantities are:
    #   chunk_duration_ms : how much wall-clock motion one query supplies
    #   duty_cycle        : fraction of that time spent computing the next chunk
    #   realtime_feasible : whether a query finishes before its chunk runs out
    # Reporting "latency x chunk_size" as an effective Hz would produce absurd
    # figures (ACT: 2207 "Hz") that no robot could ever realise.
    chunk_duration_ms = 1000.0 * n_action_steps / CONTROL_HZ
    return {
        "n_iters": n_iters,
        "mean_ms": mean_ms,
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
        "n_action_steps": n_action_steps,
        "query_hz": 1000.0 / mean_ms,
        "control_hz_assumed": CONTROL_HZ,
        "chunk_duration_ms": chunk_duration_ms,
        "duty_cycle": mean_ms / chunk_duration_ms,
        # p95 is the honest bar: one late chunk stalls the arm.
        "realtime_feasible": bool(float(np.percentile(arr, 95)) < chunk_duration_ms),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="kind:path")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out", default="outputs/eval/action_error/latency.json")
    args = ap.parse_args()

    smolvla_rename = {
        "observation.images.front": "observation.images.camera1",
        "observation.images.wrist": "observation.images.camera2",
    }

    results = {}
    for spec in args.model:
        kind, _, ckpt = spec.partition(":")
        if not Path(ckpt).exists():
            print(f"[latency] SKIP {kind}: no checkpoint at {ckpt}")
            continue
        print(f"[latency] {kind} <- {ckpt}")
        r = bench(kind, ckpt, args.device, args.warmup, args.iters,
                  smolvla_rename if kind == "smolvla" else None)
        results[kind] = r
        print(f"    mean {r['mean_ms']:.1f} ms  p95 {r['p95_ms']:.1f} ms  "
              f"chunk {r['n_action_steps']} ({r['chunk_duration_ms']:.0f} ms of motion "
              f"@ {CONTROL_HZ:.0f} Hz)  duty {r['duty_cycle'] * 100:.1f}%  "
              f"realtime={'yes' if r['realtime_feasible'] else 'NO'}")

        if kind == "bc":
            # Same policy, text cache disabled: isolates the cost of re-encoding
            # a constant instruction on every control tick.
            u = bench(kind, ckpt, args.device, args.warmup, args.iters, None,
                      text_cache=False)
            results["bc_no_text_cache"] = u
            r["text_cache_speedup"] = u["mean_ms"] / r["mean_ms"]
            print(f"    without text cache: mean {u['mean_ms']:.1f} ms "
                  f"-> caching is {r['text_cache_speedup']:.2f}x faster")

    if results and "bc" in results:
        # How much less often each policy must be queried than the single-step
        # baseline -- the real benefit of action chunking. The no-cache row is a
        # BC ablation, not a rung of the ladder, so it is excluded.
        base = results["bc"]["chunk_duration_ms"]
        for kind, r in results.items():
            if kind != "bc_no_text_cache":
                r["query_period_vs_bc"] = r["chunk_duration_ms"] / base

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"device": args.device, "results": results}, indent=2),
                   encoding="utf-8")
    print(f"[latency] wrote {out}")


if __name__ == "__main__":
    main()
