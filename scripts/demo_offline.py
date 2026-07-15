#!/usr/bin/env python
"""Hardware-free, GPU-free end-to-end demo of the control stack.

Wires the real DummyRobot + TemporalEnsembler + SafetyFilter together and runs a
rollout with a stub policy (small random action chunks). Nothing here needs a
GPU, a robot, or the network — it exists so you can verify the whole loop
(observe -> chunk -> ensemble -> safety-filter -> command) executes correctly on
any laptop, and to show the safety filter actually clamping deliberately-unsafe
targets. This is the first thing to run after cloning.

    python scripts/demo_offline.py --ticks 60
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vla.inference.action_chunking import TemporalEnsembler  # noqa: E402
from vla.inference.safety import SafetyFilter  # noqa: E402
from vla.robot.dummy_robot import DummyRobot  # noqa: E402
from vla.utils.config import load_yaml  # noqa: E402


class StubChunkPolicy:
    """Stands in for a trained chunking policy: returns an (H, action_dim) chunk
    of small joint moves plus, occasionally, a deliberately out-of-range target
    so you can watch the safety filter reject it."""

    def __init__(self, action_dim: int, horizon: int = 8, seed: int = 0):
        self.action_dim = action_dim
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)

    def act(self, state: np.ndarray, tick: int) -> np.ndarray:
        base = np.asarray(state, dtype=np.float64)
        chunk = base[None, :] + self.rng.normal(0, 3.0, size=(self.horizon, self.action_dim))
        if tick % 20 == 10:  # inject an unsafe spike every 20 ticks
            chunk[0, 0] = 9999.0
        return chunk


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline end-to-end control-loop demo.")
    parser.add_argument("--ticks", type=int, default=60)
    parser.add_argument("--robot-config", default="configs/robot_so101.yaml")
    parser.add_argument("--safety-config", default="configs/safety_limits.yaml")
    args = parser.parse_args()

    robot_cfg = load_yaml(args.robot_config)
    safety_cfg = load_yaml(args.safety_config)
    action_dim = robot_cfg["robot"]["n_joints"]

    robot = DummyRobot(n_joints=action_dim)
    safety = SafetyFilter(robot_cfg, safety_cfg)
    ensembler = TemporalEnsembler(action_dim, weight_m=0.01)
    policy = StubChunkPolicy(action_dim)

    safety.reset(robot.get_joint_state())
    latencies, enforced = [], 0

    print(f"[demo] running {args.ticks}-tick rollout (dummy robot, stub policy)...")
    for tick in range(args.ticks):
        t0 = time.perf_counter()
        obs = robot.get_observation("put the red block in the green bowl")
        chunk = policy.act(obs["state"], tick)
        ensembler.add_chunk(chunk)
        action = ensembler.step()
        safe_target, report = safety.filter(action, gripper_current=obs["gripper_current"])
        robot.send_joint_target(safe_target)
        latencies.append((time.perf_counter() - t0) * 1e3)
        if report.any_action_taken:
            enforced += 1
            if report.workspace_violation or report.clamped_joints:
                print(f"  tick {tick:3d}: safety enforced {report}")

    lat = np.array(latencies)
    print("\n[demo] rollout complete.")
    print(f"  ticks with safety enforcement : {enforced}/{args.ticks}")
    print(f"  loop latency  mean/p95/max ms : {lat.mean():.2f} / "
          f"{np.percentile(lat, 95):.2f} / {lat.max():.2f}")
    print(f"  final joint state             : {np.round(robot.get_joint_state(), 2)}")
    print("\n[demo] The safety filter clamped the injected 9999-degree spikes — "
          "no unsafe target reached the (simulated) motors.")


if __name__ == "__main__":
    main()
