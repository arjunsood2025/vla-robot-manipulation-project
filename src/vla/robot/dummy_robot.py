"""A hardware-free RobotInterface for tests, CI, and laptop demos.

Holds an internal joint state, moves it toward commanded targets, and returns
random-noise 'camera' frames. Lets the entire control loop (client -> server ->
safety filter -> command) be exercised end-to-end with no arm and no GPU, which
is how the integration test and the offline demo run.
"""
from __future__ import annotations

import numpy as np


class DummyRobot:
    def __init__(self, n_joints: int = 6, image_hw: tuple[int, int] = (480, 640), seed: int = 0):
        self.n_joints = n_joints
        self.image_hw = image_hw
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(n_joints, dtype=np.float64)

    def _frame(self) -> np.ndarray:
        h, w = self.image_hw
        return self.rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)

    def get_observation(self, instruction: str) -> dict:
        return {
            "images": {"front": self._frame(), "wrist": self._frame()},
            "state": self.state.copy(),
            "gripper_current": 0.0,
            "instruction": instruction,
        }

    def get_joint_state(self) -> np.ndarray:
        return self.state.copy()

    def send_joint_target(self, joint_deg: np.ndarray) -> None:
        # First-order tracking toward the target (a crude servo model).
        self.state += 0.5 * (np.asarray(joint_deg, dtype=np.float64) - self.state)

    def torque_off(self) -> None:
        pass
