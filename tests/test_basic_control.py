"""Focused tests for the deterministic Milestone 2 SO-101 control layer."""

import mujoco

from tablemind.control import SO101Controller
from tablemind.simulation import BimanualTableSimulation


def test_gripper_controls_are_independent() -> None:
    simulation = BimanualTableSimulation.create()
    controller = SO101Controller(simulation)
    left_gripper = mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_gripper")
    right_gripper = mujoco.mj_name2id(simulation.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_gripper")

    controller.open_gripper("left", settle_steps=0)
    assert simulation.data.ctrl[left_gripper] == simulation.model.actuator_ctrlrange[left_gripper, 1]
    assert simulation.data.ctrl[right_gripper] == 0.0

    controller.close_gripper("right", settle_steps=0)
    assert simulation.data.ctrl[right_gripper] == simulation.model.actuator_ctrlrange[right_gripper, 0]


def test_cartesian_move_accepts_current_end_effector_position() -> None:
    controller = SO101Controller(BimanualTableSimulation.create())

    result = controller.move_to("left", controller.end_effector_position("left"))

    assert result.reached
    assert result.steps == 0
    assert result.position_error == 0.0


def test_placeholder_pick_is_not_reported_as_a_success() -> None:
    controller = SO101Controller(
        BimanualTableSimulation.create()
    )

    result = controller.execute_pick(
        "left",
        "plate_1",
    )

    assert not result.completed
    assert (
        "grasp attachment is not implemented"
        in result.reason
    )