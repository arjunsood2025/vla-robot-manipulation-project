"""Optional ROS2 bridge for the SO-101 (extreme-version / lab-integration path).

Not required for the LeRobot-native control loop — the client talks to the arm
directly through ``LeRobotSO101``. This node exists for setups that standardise
on ROS2 (multi-robot labs, existing ROS tooling): it subscribes to joint states,
republishes them as observations, and exposes a service/topic that accepts safe
joint targets from the inference client. ``rclpy`` is imported lazily so the rest
of the project has no ROS dependency.

Topics:
    subscribe  /so101/joint_states   (sensor_msgs/JointState)
    publish    /so101/joint_command  (sensor_msgs/JointState) - position targets

The SafetyFilter still runs on the client side BEFORE anything is published here;
this node is a transport, not a safety boundary.
"""
from __future__ import annotations

import numpy as np

from vla.utils.config import load_yaml


class SO101ROS2Controller:
    def __init__(self, robot_config_path: str = "configs/robot_so101.yaml"):
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState

        cfg = load_yaml(robot_config_path)["robot"]
        self.joint_names = cfg["joint_names"]
        self._JointState = JointState

        if not rclpy.ok():
            rclpy.init()
        self.node = Node("so101_controller")
        self._latest_state = np.zeros(len(self.joint_names), dtype=np.float64)

        self.node.create_subscription(
            JointState, "/so101/joint_states", self._on_joint_state, 10
        )
        self.cmd_pub = self.node.create_publisher(JointState, "/so101/joint_command", 10)
        self._rclpy = rclpy

    def _on_joint_state(self, msg) -> None:
        # Reorder incoming positions to our canonical joint order.
        name_to_pos = dict(zip(msg.name, msg.position))
        self._latest_state = np.array(
            [name_to_pos.get(n, 0.0) for n in self.joint_names], dtype=np.float64
        )

    def get_joint_state(self) -> np.ndarray:
        self._rclpy.spin_once(self.node, timeout_sec=0.0)
        return self._latest_state.copy()

    def send_joint_target(self, joint_deg: np.ndarray) -> None:
        msg = self._JointState()
        msg.name = list(self.joint_names)
        msg.position = [float(v) for v in joint_deg]
        self.cmd_pub.publish(msg)

    def shutdown(self) -> None:
        self.node.destroy_node()
        self._rclpy.shutdown()
