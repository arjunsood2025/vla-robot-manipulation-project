"""Approximate forward kinematics for the SO-101 arm.

The safety filter needs to answer one question cheaply: "if I send these joint
angles, roughly where does the gripper end up?" — so it can reject targets that
would drive the end-effector outside the workspace box. We do NOT need a precise
kinematic model for that; a planar 3-link approximation (shoulder-lift, elbow,
wrist pitch in the vertical plane, rotated by the shoulder-pan yaw) is accurate
to a couple of centimetres, which is plenty for a coarse safety envelope. Full
IK is unnecessary because the learned policies command joint space directly.

Angles are in DEGREES to match the calibrated dataset convention. Pure numpy so
it is unit-tested without ROS or hardware.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SO101Geometry:
    base_height: float = 0.055
    shoulder: float = 0.116
    elbow: float = 0.135
    wrist: float = 0.065

    @classmethod
    def from_config(cls, robot_cfg: dict) -> "SO101Geometry":
        lengths = robot_cfg["robot"]["link_lengths_m"]
        return cls(
            base_height=lengths["base_height"],
            shoulder=lengths["shoulder"],
            elbow=lengths["elbow"],
            wrist=lengths["wrist"],
        )


def forward_kinematics_ee(joint_deg: np.ndarray, geom: SO101Geometry) -> np.ndarray:
    """Return the end-effector (x, y, z) in metres, base frame.

    joint_deg layout (first 4 arm joints used; wrist_roll and gripper don't move
    the EE position):
        [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]

    Convention: shoulder_pan rotates the whole arm about the vertical z axis;
    the remaining pitch joints articulate the arm in the radial-vertical plane.
    Zero pitch = arm pointing straight out horizontally along +radius.
    """
    j = np.asarray(joint_deg, dtype=np.float64)
    pan = np.radians(j[0])
    lift = np.radians(j[1])
    elbow = np.radians(j[2])
    wrist = np.radians(j[3])

    # Accumulate pitch angles along the chain (each is relative to the previous
    # link). Radius grows outward; height accumulates vertically.
    a1 = lift
    a2 = lift + elbow
    a3 = lift + elbow + wrist

    radius = (
        geom.shoulder * np.cos(a1)
        + geom.elbow * np.cos(a2)
        + geom.wrist * np.cos(a3)
    )
    height = (
        geom.base_height
        + geom.shoulder * np.sin(a1)
        + geom.elbow * np.sin(a2)
        + geom.wrist * np.sin(a3)
    )

    x = radius * np.sin(pan)
    y = radius * np.cos(pan)
    z = height
    return np.array([x, y, z], dtype=np.float64)
