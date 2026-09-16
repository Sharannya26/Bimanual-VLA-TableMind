from __future__ import annotations

import numpy as np
import mujoco

from tablemind.robot_config.so101 import namespaced
from tablemind.simulation.runtime import BimanualTableSimulation


simulation = BimanualTableSimulation.create()

model = simulation.model
data = simulation.data

print("TABLEMIND - Milestone 4.3 Jacobian Diagnostic")
print("=" * 70)

for arm in simulation.arms:
    print()
    print(f"{arm.upper()} ARM")
    print("-" * 70)

    site_name = namespaced(arm, "gripperframe")

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )

    jac_position = np.zeros((3, model.nv))

    mujoco.mj_jacSite(
        model,
        data,
        jac_position,
        None,
        site_id,
    )

    print("Position Jacobian:")
    print(jac_position)

    # Only show the five arm joints used for Cartesian control.
    joint_names = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    )

    columns = []

    for joint_name in joint_names:
        full_name = namespaced(arm, joint_name)

        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            full_name,
        )

        dof_address = model.jnt_dofadr[joint_id]

        columns.append(dof_address)

    arm_jacobian = jac_position[:, columns]

    print()
    print("5-DOF arm Jacobian:")
    print(arm_jacobian)

    print()
    print("Jacobian singular values:")

    singular_values = np.linalg.svd(
        arm_jacobian,
        compute_uv=False,
    )

    for index, value in enumerate(singular_values, start=1):
        print(f"  sigma_{index} = {value:.6f}")

    print()
    print("Approximate Cartesian direction capability:")

    x_strength = np.linalg.norm(arm_jacobian[0])
    y_strength = np.linalg.norm(arm_jacobian[1])
    z_strength = np.linalg.norm(arm_jacobian[2])

    print(f"  X = {x_strength:.6f}")
    print(f"  Y = {y_strength:.6f}")
    print(f"  Z = {z_strength:.6f}")

print()
print("Jacobian diagnostic complete.")