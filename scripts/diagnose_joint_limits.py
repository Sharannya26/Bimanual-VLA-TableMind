from __future__ import annotations

import mujoco

from tablemind.robot_config.so101 import JOINT_NAMES, namespaced
from tablemind.simulation.runtime import BimanualTableSimulation


simulation = BimanualTableSimulation.create()

model = simulation.model
data = simulation.data

print("TABLEMIND - Milestone 4.3 Joint Diagnostic")
print("=" * 70)

for arm in simulation.arms:
    print()
    print(f"{arm.upper()} ARM")
    print("-" * 70)

    for joint_name in JOINT_NAMES:
        full_name = namespaced(arm, joint_name)

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            full_name,
        )

        if joint_id < 0:
            print(f"{joint_name:12s} NOT FOUND")
            continue

        qpos_address = model.jnt_qposadr[joint_id]

        current = float(data.qpos[qpos_address])

        limited = bool(model.jnt_limited[joint_id])

        if limited:
            lower = float(model.jnt_range[joint_id, 0])
            upper = float(model.jnt_range[joint_id, 1])
        else:
            lower = float("-inf")
            upper = float("inf")

        print(
            f"{joint_name:12s} "
            f"current={current:+.4f} rad "
            f"range=[{lower:+.4f}, {upper:+.4f}] "
            f"limited={limited}"
        )

print()
print("Initial end-effector positions")
print("-" * 70)

for arm in simulation.arms:
    site_name = namespaced(arm, "gripperframe")

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    position = data.site_xpos[site_id]

    print(
        f"{arm:6s} "
        f"EE=({position[0]:+.4f}, "
        f"{position[1]:+.4f}, "
        f"{position[2]:+.4f})"
    )