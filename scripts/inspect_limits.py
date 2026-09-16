"""Inspect SO-101 joint and actuator limits."""

from __future__ import annotations

import mujoco

from tablemind.simulation import BimanualTableSimulation
from tablemind.robot_config.so101 import JOINT_NAMES, namespaced


def main() -> None:
    simulation = BimanualTableSimulation.create()

    arm = "left"

    print("=" * 70)
    print("SO-101 JOINT / ACTUATOR LIMIT INSPECTION")
    print("=" * 70)

    for joint_name in JOINT_NAMES:
        joint_full_name = namespaced(arm, joint_name)

        joint_id = mujoco.mj_name2id(
            simulation.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_full_name,
        )

        actuator_id = mujoco.mj_name2id(
            simulation.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            joint_full_name,
        )

        qpos_address = simulation.model.jnt_qposadr[joint_id]

        current = float(
            simulation.data.qpos[qpos_address]
        )

        joint_range = simulation.model.jnt_range[joint_id]
        actuator_range = simulation.model.actuator_ctrlrange[actuator_id]

        print(f"\n{joint_name}")
        print(f"  Current qpos    : {current:+.6f}")
        print(
            f"  Joint range    : "
            f"[{joint_range[0]:+.6f}, {joint_range[1]:+.6f}]"
        )
        print(
            f"  Actuator range : "
            f"[{actuator_range[0]:+.6f}, {actuator_range[1]:+.6f}]"
        )

    print("\n" + "=" * 70)
    print("END")
    print("=" * 70)


if __name__ == "__main__":
    main()