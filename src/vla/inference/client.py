"""Robot-side inference client: the real-time control loop.

Responsibilities per tick (target 30 Hz):
    1. read the current observation (camera frames + joint state) from the robot,
    2. ask the policy server for an action chunk (async — request the next chunk
       before the current one is exhausted so the GPU latency is hidden),
    3. push the chunk through the temporal ensembler for a smooth per-tick action,
    4. run that action through the SafetyFilter, and
    5. command the (safe) joint target to the robot.

The robot itself is behind a small ``RobotInterface`` so this loop is identical
for a LeRobot SO-101, a MuJoCo sim twin, or a bench test with a dummy arm. That
seam is what lets the same evaluation run in sim and on hardware.
"""
from __future__ import annotations

import time
from typing import Protocol

import numpy as np

from vla.inference.action_chunking import TemporalEnsembler
from vla.inference.safety import SafetyFilter
from vla.inference.serialization import pack, unpack
from vla.utils.config import load_yaml


class RobotInterface(Protocol):
    def get_observation(self, instruction: str) -> dict:
        """Return {'images': {cam: HWC array}, 'state': joint array (deg),
        'gripper_current': float | None, 'instruction': str}."""
        ...

    def send_joint_target(self, joint_deg: np.ndarray) -> None: ...

    def get_joint_state(self) -> np.ndarray: ...

    def torque_off(self) -> None: ...


class RobotClient:
    def __init__(
        self,
        server_uri: str = "ws://localhost:8000",
        robot: RobotInterface | None = None,
        robot_config_path: str = "configs/robot_so101.yaml",
        safety_config_path: str = "configs/safety_limits.yaml",
        control_hz: float = 30.0,
        ensemble_weight_m: float = 0.01,
        max_ticks: int = 600,   # 20 s @ 30 Hz hard episode cap
    ):
        self.server_uri = server_uri
        self.robot = robot
        robot_cfg = load_yaml(robot_config_path)
        safety_cfg = load_yaml(safety_config_path)
        self.safety = SafetyFilter(robot_cfg, safety_cfg)
        self.action_dim = robot_cfg["robot"]["n_joints"]
        self.dt = 1.0 / control_hz
        self.ensemble_weight_m = ensemble_weight_m
        self.max_ticks = max_ticks
        self.max_stale_ticks = safety_cfg.get("max_stale_ticks", 3)
        self.latencies_ms: list[float] = []

    # ---- synchronous convenience wrapper used by the eval harness ----
    def run_episode(self, instruction: str, layout: dict, distractors: list, seed: int) -> None:
        """Drive one rollout to completion (or the tick cap). The eval harness
        calls this; success labelling is done by the operator afterwards."""
        import asyncio

        asyncio.run(self._run_episode_async(instruction))

    async def _run_episode_async(self, instruction: str) -> None:
        import websockets

        if self.robot is None:
            raise RuntimeError("RobotClient needs a RobotInterface to run on hardware.")

        ensembler = TemporalEnsembler(self.action_dim, self.ensemble_weight_m)
        self.safety.reset(self.robot.get_joint_state())
        request_id = 0

        async with websockets.connect(self.server_uri, max_size=None) as ws:
            for tick in range(self.max_ticks):
                loop_start = time.perf_counter()

                # 1. observe + 2. request a fresh chunk every tick (temporal ensembling).
                obs = self.robot.get_observation(instruction)
                obs["request_id"] = request_id
                request_id += 1
                await ws.send(pack(obs))
                response = unpack(await ws.recv())
                chunk = np.asarray(response["action_chunk"], dtype=np.float64)
                ensembler.add_chunk(chunk)

                # 3. ensembled action for this tick.
                action = ensembler.step()

                # 4. safety filter.
                safe_target, report = self.safety.filter(
                    action, gripper_current=obs.get("gripper_current")
                )
                if report.workspace_violation:
                    print(f"[client] tick {tick}: workspace violation — holding pose")

                # 5. command.
                self.robot.send_joint_target(safe_target)

                # latency bookkeeping + fixed-rate pacing.
                self.latencies_ms.append((time.perf_counter() - loop_start) * 1e3)
                sleep = self.dt - (time.perf_counter() - loop_start)
                if sleep > 0:
                    await _async_sleep(sleep)

    def latency_summary(self) -> dict[str, float]:
        if not self.latencies_ms:
            return {}
        arr = np.array(self.latencies_ms)
        return {
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "max_ms": float(arr.max()),
        }


async def _async_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)
