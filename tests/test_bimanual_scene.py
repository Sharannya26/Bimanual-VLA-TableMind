"""Milestone-one validation tests for the bimanual MuJoCo foundation."""

import mujoco

from tablemind.robot_config.so101 import JOINT_NAMES
from tablemind.simulation import BimanualTableSimulation


def test_model_loads_with_two_independent_so101_arms() -> None:
    simulation = BimanualTableSimulation.create()

    assert simulation.arms == ("left", "right")
    assert simulation.model.nu == 12
    for arm in simulation.arms:
        assert len(simulation.joint_names(arm)) == len(JOINT_NAMES)
        for joint_name in simulation.joint_names(arm):
            assert mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name) >= 0


def test_scene_has_workspace_camera_and_dinner_objects() -> None:
    simulation = BimanualTableSimulation.create()

    for geometry in ("table", "plate_1", "plate_2", "glass_1", "glass_2"):
        assert mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_GEOM, geometry) >= 0
    assert mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_CAMERA, "table_overview") >= 0


def test_simulation_steps_and_arm_targets_are_isolated() -> None:
    simulation = BimanualTableSimulation.create()
    right_before = simulation.joint_positions("right")

    simulation.set_joint_targets("left", {"shoulder_pan": 0.20, "elbow_flex": -0.15})
    left_actuator = mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_shoulder_pan")
    right_actuator = mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_shoulder_pan")
    assert simulation.data.ctrl[left_actuator] == 0.20
    assert simulation.data.ctrl[right_actuator] == 0.0

    simulation.step(20)
    assert simulation.data.time > 0.0
    assert simulation.joint_positions("right").keys() == right_before.keys()
