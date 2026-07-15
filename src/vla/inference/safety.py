"""Safety filter sitting between the policy and the motors.

This is non-negotiable for real hardware: NOTHING the policy predicts reaches a
servo without passing every check here. The filter is stateful — it remembers
the last commanded target so that when a check trips it can hold position rather
than lunge. Checks, in order:

  1. Joint-limit clamp    - keep every joint inside a margin of its calibrated range.
  2. Velocity clamp       - cap the per-tick change so the arm can't snap.
  3. Workspace check      - reject targets whose FK end-effector leaves the box.
  4. Gripper current      - force the gripper open if it stalls (crush protection).

Design note defensible in review: clamping (joints, velocity) SATURATES toward a
safe value, while the workspace check REJECTS (holds the previous pose). We
prefer holding over projecting-onto-the-box because a projected target can be a
pose the arm was never trained on, whereas the previous target is known-safe.

Pure numpy so the whole envelope is unit-tested without hardware.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vla.robot.kinematics import SO101Geometry, forward_kinematics_ee


@dataclass
class SafetyReport:
    """What the filter did on one tick — logged so rejections are visible, not
    silently swallowed."""
    clamped_joints: bool = False
    clamped_velocity: bool = False
    workspace_violation: bool = False
    gripper_protected: bool = False

    @property
    def any_action_taken(self) -> bool:
        return (self.clamped_joints or self.clamped_velocity
                or self.workspace_violation or self.gripper_protected)


class SafetyFilter:
    def __init__(self, robot_cfg: dict, safety_cfg: dict):
        rc = robot_cfg["robot"]
        self.joint_names: list[str] = rc["joint_names"]
        self.n_joints = rc["n_joints"]
        self.gripper_idx = self.joint_names.index("gripper")

        # Calibrated limits shrunk by the safety margin.
        margin = safety_cfg["joint_limit_margin"]
        limits = rc["joint_limits_deg"]
        lo, hi = [], []
        for name in self.joint_names:
            a, b = limits[name]
            mid = 0.5 * (a + b)
            half = 0.5 * (b - a) * margin
            lo.append(mid - half)
            hi.append(mid + half)
        self.joint_low = np.array(lo, dtype=np.float64)
        self.joint_high = np.array(hi, dtype=np.float64)

        self.max_joint_delta = float(safety_cfg["max_joint_delta_deg"])
        self.max_gripper_delta = float(safety_cfg["max_gripper_delta"])
        self.gripper_current_limit = float(safety_cfg["gripper_current_limit"])

        box = safety_cfg["workspace_bbox_m"]
        self.box_low = np.array([box["x"][0], box["y"][0], box["z"][0]])
        self.box_high = np.array([box["x"][1], box["y"][1], box["z"][1]])

        self.geom = SO101Geometry.from_config(robot_cfg)
        self._last_target: np.ndarray | None = None

    def reset(self, current_joints: np.ndarray) -> None:
        """Seed the 'last target' with where the arm actually is before a rollout."""
        self._last_target = np.asarray(current_joints, dtype=np.float64).copy()

    def in_workspace(self, joint_target: np.ndarray) -> bool:
        ee = forward_kinematics_ee(joint_target, self.geom)
        return bool(np.all(ee >= self.box_low) and np.all(ee <= self.box_high))

    def filter(self, raw_target: np.ndarray, gripper_current: float | None = None
               ) -> tuple[np.ndarray, SafetyReport]:
        """Return a safe joint target and a report of what was enforced."""
        report = SafetyReport()
        target = np.asarray(raw_target, dtype=np.float64).copy()
        if target.shape[0] != self.n_joints:
            raise ValueError(f"Expected {self.n_joints} joints, got {target.shape[0]}")

        last = self._last_target if self._last_target is not None else target.copy()

        # 1. Joint-limit clamp.
        clamped = np.clip(target, self.joint_low, self.joint_high)
        if not np.allclose(clamped, target):
            report.clamped_joints = True
        target = clamped

        # 2. Per-tick velocity clamp (arm joints vs gripper have different caps).
        delta = target - last
        cap = np.full(self.n_joints, self.max_joint_delta)
        cap[self.gripper_idx] = self.max_gripper_delta
        clipped_delta = np.clip(delta, -cap, cap)
        if not np.allclose(clipped_delta, delta):
            report.clamped_velocity = True
        target = last + clipped_delta

        # 3. Workspace check on the resulting pose — reject to last-safe if outside.
        if not self.in_workspace(target):
            report.workspace_violation = True
            target = last.copy()

        # 4. Gripper stall / crush protection.
        if gripper_current is not None and gripper_current > self.gripper_current_limit:
            report.gripper_protected = True
            target[self.gripper_idx] = self.joint_high[self.gripper_idx]  # force open

        self._last_target = target.copy()
        return target, report
