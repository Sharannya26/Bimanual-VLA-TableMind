"""Diagnose SO-101 actuator and Cartesian control."""

from __future__ import annotations

import mujoco
import numpy as np

from tablemind.control import SO101Controller
from tablemind.simulation import BimanualTableSimulation


def main() -> None:
    simulation = BimanualTableSimulation.create()
    controller = SO101Controller(simulation)

    arm = "left"

    print("=" * 60)
    print("SO-101 CONTROL DIAGNOSTIC")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Model information
    # ---------------------------------------------------------
    print("\n[1] MODEL")
    print(f"Number of joints   : {simulation.model.njnt}")
    print(f"Number of DoFs     : {simulation.model.nv}")
    print(f"Number of actuators: {simulation.model.nu}")

    # ---------------------------------------------------------
    # 2. Joint positions before movement
    # ---------------------------------------------------------
    print("\n[2] INITIAL JOINT POSITIONS")

    initial_joints = simulation.joint_positions(arm)

    for name, value in initial_joints.items():
        print(f"{name:20s}: {value:+.6f}")

    # ---------------------------------------------------------
    # 3. End-effector position
    # ---------------------------------------------------------
    initial_ee = controller.end_effector_position(arm)

    print("\n[3] INITIAL END EFFECTOR")
    print(
        "Position: "
        f"x={initial_ee[0]:+.4f}, "
        f"y={initial_ee[1]:+.4f}, "
        f"z={initial_ee[2]:+.4f}"
    )

    # ---------------------------------------------------------
    # 4. Plate position
    # ---------------------------------------------------------
    plate = controller.object_position("plate_1")

    print("\n[4] PLATE")
    print(
        "Position: "
        f"x={plate[0]:+.4f}, "
        f"y={plate[1]:+.4f}, "
        f"z={plate[2]:+.4f}"
    )

    target = (
        plate[0],
        plate[1],
        plate[2] + 0.16,
    )

    print(
        "Target:   "
        f"x={target[0]:+.4f}, "
        f"y={target[1]:+.4f}, "
        f"z={target[2]:+.4f}"
    )

    # ---------------------------------------------------------
    # 5. Test one actuator directly
    # ---------------------------------------------------------
    print("\n[5] DIRECT ACTUATOR TEST")

    joint_name = "shoulder_pan"

    joint_id = mujoco.mj_name2id(
        simulation.model,
        mujoco.mjtObj.mjOBJ_JOINT,
        f"{arm}_{joint_name}",
    )

    actuator_id = mujoco.mj_name2id(
        simulation.model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        f"{arm}_{joint_name}",
    )

    print(f"Joint ID    : {joint_id}")
    print(f"Actuator ID : {actuator_id}")

    qpos_address = simulation.model.jnt_qposadr[joint_id]

    before = float(simulation.data.qpos[qpos_address])

    ctrl_range = simulation.model.actuator_ctrlrange[actuator_id]

    print(f"Current qpos : {before:+.6f}")
    print(
        "Control range: "
        f"[{ctrl_range[0]:+.6f}, {ctrl_range[1]:+.6f}]"
    )

    # Move slightly toward the middle of the actuator range.
    test_target = float(
        np.clip(
            before + 0.20,
            ctrl_range[0],
            ctrl_range[1],
        )
    )

    print(f"Test target  : {test_target:+.6f}")

    simulation.set_joint_targets(
        arm,
        {joint_name: test_target},
    )

    print("\nRunning 100 simulation steps...")

    simulation.step(100)

    after = float(simulation.data.qpos[qpos_address])

    print(f"qpos before  : {before:+.6f}")
    print(f"qpos after   : {after:+.6f}")
    print(f"qpos change  : {after - before:+.6f}")

    # ---------------------------------------------------------
    # 6. Test Cartesian movement
    # ---------------------------------------------------------
    print("\n[6] CARTESIAN TEST")

    print(
        "Current EE: "
        f"{controller.end_effector_position(arm)}"
    )

    print(f"Moving toward: {target}")

    result = controller.move_to(
        arm,
        target,
        max_steps=100,
    )

    print("\nRESULT")
    print(f"Reached       : {result.reached}")
    print(f"Control steps : {result.steps}")
    print(f"Final position: {result.final_position}")
    print(f"Final error   : {result.position_error:.6f} m")

    # ---------------------------------------------------------
    # 7. Final joint positions
    # ---------------------------------------------------------
    print("\n[7] FINAL JOINT POSITIONS")

    final_joints = simulation.joint_positions(arm)

    for name, value in final_joints.items():
        change = value - initial_joints[name]
        print(
            f"{name:20s}: "
            f"{value:+.6f} "
            f"(change {change:+.6f})"
        )

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()