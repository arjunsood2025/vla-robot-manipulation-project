import numpy as np
import pytest

from vla.inference.safety import SafetyFilter
from vla.utils.config import load_yaml

ROBOT_CFG = load_yaml("configs/robot_so101.yaml")
SAFETY_CFG = load_yaml("configs/safety_limits.yaml")


def make_filter():
    f = SafetyFilter(ROBOT_CFG, SAFETY_CFG)
    f.reset(np.zeros(6))
    return f


def test_joint_limit_clamp():
    f = make_filter()
    # Command a wildly out-of-range shoulder_pan; expect a clamp report and a
    # target inside the calibrated (margin-shrunk) range.
    target, report = f.filter(np.array([9999, 0, 0, 0, 0, 0], dtype=float))
    assert report.clamped_joints
    assert target[0] <= f.joint_high[0] + 1e-9


def test_velocity_clamp_limits_per_tick_delta():
    f = make_filter()
    # A 50-degree jump on one joint must be capped to max_joint_delta_deg per tick.
    target, report = f.filter(np.array([50, 0, 0, 0, 0, 0], dtype=float))
    assert report.clamped_velocity
    assert abs(target[0]) <= SAFETY_CFG["max_joint_delta_deg"] + 1e-9


def test_gripper_has_larger_delta_budget():
    f = make_filter()
    gi = f.gripper_idx
    cmd = np.zeros(6)
    cmd[gi] = 12.0  # between arm cap (6) and gripper cap (15)
    target, report = f.filter(cmd)
    # Gripper should move the full 12 (within its budget), not be clamped to 6.
    assert target[gi] == pytest.approx(12.0)


def test_in_workspace_accepts_nominal_pose_rejects_extreme():
    f = make_filter()
    # All-zero joints put the EE well inside the box (arm extended forward).
    assert f.in_workspace(np.zeros(6))
    # Folding every pitch joint to 90 deg drives the EE up and back out of the box.
    assert not f.in_workspace(np.array([0, 90, 90, 90, 0, 50], dtype=float))


def test_workspace_violation_holds_previous_pose(monkeypatch):
    # Force the workspace check to fail and confirm the filter holds the last
    # commanded target rather than moving. This isolates the hold logic from the
    # exact FK geometry (which is covered by the test above).
    f = make_filter()
    prev, _ = f.filter(np.array([1.0, 1.0, 1.0, 1.0, 0.0, 2.0]))  # a known-safe step
    monkeypatch.setattr(f, "in_workspace", lambda target: False)
    held, report = f.filter(np.array([3.0, 3.0, 3.0, 3.0, 0.0, 4.0]))
    assert report.workspace_violation
    assert np.allclose(held, prev)


def test_gripper_current_forces_open():
    f = make_filter()
    gi = f.gripper_idx
    target, report = f.filter(np.zeros(6), gripper_current=99999)
    assert report.gripper_protected
    assert target[gi] == pytest.approx(f.joint_high[gi])


def test_wrong_dimension_raises():
    f = make_filter()
    with pytest.raises(ValueError):
        f.filter(np.zeros(3))


def test_safe_command_passes_through_unmodified():
    f = make_filter()
    small = np.array([1.0, 1.0, -1.0, 0.5, 0.0, 2.0])
    target, report = f.filter(small)
    # Small, in-range, in-workspace move: nothing should be enforced.
    assert not report.any_action_taken
    assert np.allclose(target, small)
