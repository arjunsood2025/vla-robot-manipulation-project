#!/usr/bin/env python
"""Free teleoperation (no recording) for checking calibration, camera framing,
and reachability before a collection session.

Drives the follower arm from the leader arm through LeRobot and prints the live
joint state + the safety filter's verdict each tick, so you can confirm the
workspace box and joint limits are sane on THIS arm before you trust them during
autonomous rollouts. Nothing here commands the arm autonomously — it only mirrors
the leader — but it runs the real SafetyFilter in 'observe' mode so you see what
it would clamp.

Example:
    python scripts/teleop.py --robot-port COM5 --teleop-port COM6
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vla.inference.safety import SafetyFilter  # noqa: E402
from vla.utils.config import load_yaml  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Teleop sanity check (no recording).")
    parser.add_argument("--robot-port", required=True)
    parser.add_argument("--teleop-port", required=True)
    parser.add_argument("--robot-config", default="configs/robot_so101.yaml")
    parser.add_argument("--safety-config", default="configs/safety_limits.yaml")
    parser.add_argument("--seconds", type=float, default=60.0)
    args = parser.parse_args()

    from lerobot.common.robots.so101_follower import SO101Follower, SO101FollowerConfig
    from lerobot.common.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

    robot_cfg = load_yaml(args.robot_config)
    safety = SafetyFilter(robot_cfg, load_yaml(args.safety_config))

    robot = SO101Follower(SO101FollowerConfig(port=args.robot_port))
    leader = SO101Leader(SO101LeaderConfig(port=args.teleop_port))
    robot.connect()
    leader.connect()
    safety.reset(robot.get_observation()["observation.state"])

    print("[teleop] mirroring leader -> follower. Ctrl-C to stop.")
    t_end = time.time() + args.seconds
    try:
        while time.time() < t_end:
            action = leader.get_action()
            target = [action[f"{n}.pos"] for n in robot_cfg["robot"]["joint_names"]]
            safe, report = safety.filter(target)
            if report.any_action_taken:
                print(f"[teleop] safety would enforce: {report}")
            robot.send_action({f"{n}.pos": float(v)
                               for n, v in zip(robot_cfg["robot"]["joint_names"], safe)})
            time.sleep(1 / robot_cfg["robot"]["control_hz"])
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()
        leader.disconnect()
        print("[teleop] disconnected.")


if __name__ == "__main__":
    main()
