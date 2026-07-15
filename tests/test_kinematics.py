import numpy as np

from vla.robot.kinematics import SO101Geometry, forward_kinematics_ee

GEOM = SO101Geometry()


def test_zero_pose_reaches_forward():
    ee = forward_kinematics_ee(np.zeros(6), GEOM)
    # All joints zero: arm points straight out along +y, so x~0, y = sum of links.
    assert ee[0] == 0.0
    assert ee[1] > 0.3
    assert ee[2] > 0.0  # above the table by the base height


def test_shoulder_pan_rotates_into_x():
    ee = forward_kinematics_ee(np.array([90, 0, 0, 0, 0, 0], dtype=float), GEOM)
    # 90-degree pan swings the reach from +y into +x.
    assert ee[0] > 0.3
    assert abs(ee[1]) < 1e-6


def test_lifting_raises_end_effector():
    low = forward_kinematics_ee(np.array([0, 0, 0, 0, 0, 0], dtype=float), GEOM)
    high = forward_kinematics_ee(np.array([0, 45, 0, 0, 0, 0], dtype=float), GEOM)
    assert high[2] > low[2]


def test_gripper_and_wrist_roll_do_not_move_position():
    a = forward_kinematics_ee(np.array([10, 20, 30, 5, 0, 0], dtype=float), GEOM)
    b = forward_kinematics_ee(np.array([10, 20, 30, 5, 160, 100], dtype=float), GEOM)
    assert np.allclose(a, b)
