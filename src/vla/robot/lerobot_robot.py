"""RobotInterface backed by a LeRobot SO-101 follower arm.

LeRobot already implements servo comms, camera capture, and calibration for the
SO-101, so on real hardware we delegate to it and only adapt the data shapes to
what the inference client expects. This is the class the client uses in the
'real robot' path; the sim path uses ``MuJoCoRobot`` and tests use ``DummyRobot``.
"""
from __future__ import annotations

import numpy as np

from vla.utils.config import load_yaml


class LeRobotSO101(  # implements vla.inference.client.RobotInterface
):
    def __init__(self, robot_config_path: str = "configs/robot_so101.yaml", port: str | None = None):
        from lerobot.common.robots.so101_follower import SO101Follower, SO101FollowerConfig

        cfg = load_yaml(robot_config_path)["robot"]
        cameras = load_yaml(robot_config_path)["cameras"]
        self.joint_names = cfg["joint_names"]

        robot_cfg = SO101FollowerConfig(
            port=port or cfg.get("port"),
            cameras={
                name: {"index_or_path": c["index_or_path"],
                       "width": c["width"], "height": c["height"], "fps": c["fps"]}
                for name, c in cameras.items()
            },
        )
        self.robot = SO101Follower(robot_cfg)
        self.robot.connect()

    def get_observation(self, instruction: str) -> dict:
        obs = self.robot.get_observation()
        images = {name: np.asarray(obs[f"observation.images.{name}"])
                  for name in ("front", "wrist") if f"observation.images.{name}" in obs}
        state = np.asarray(obs["observation.state"], dtype=np.float64)
        return {
            "images": images,
            "state": state,
            "gripper_current": obs.get("gripper_current"),
            "instruction": instruction,
        }

    def get_joint_state(self) -> np.ndarray:
        obs = self.robot.get_observation()
        return np.asarray(obs["observation.state"], dtype=np.float64)

    def send_joint_target(self, joint_deg: np.ndarray) -> None:
        action = {f"{name}.pos": float(v) for name, v in zip(self.joint_names, joint_deg)}
        self.robot.send_action(action)

    def torque_off(self) -> None:
        self.robot.disconnect()
