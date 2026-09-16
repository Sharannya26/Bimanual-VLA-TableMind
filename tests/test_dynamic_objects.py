"""Tests for TABLEMIND dynamic manipulation objects."""

from __future__ import annotations

import numpy as np
import mujoco

from tablemind.manipulation.objects import DynamicObjectManager
from tablemind.simulation.runtime import BimanualTableSimulation


def test_dynamic_object_bodies_exist() -> None:
    simulation = BimanualTableSimulation.create()

    manager = DynamicObjectManager(
        simulation.model,
        simulation.data,
    )

    for object_id in (
        "plate_1",
        "plate_2",
        "glass_1",
        "glass_2",
    ):
        assert manager.exists(object_id)


def test_dynamic_objects_have_free_joints() -> None:
    simulation = BimanualTableSimulation.create()

    for object_id in (
        "plate_1",
        "plate_2",
        "glass_1",
        "glass_2",
    ):
        body_id = mujoco.mj_name2id(
            simulation.model,
            mujoco.mjtObj.mjOBJ_BODY,
            object_id,
        )

        assert body_id >= 0

        joint_id = simulation.model.body_jntadr[body_id]

        assert joint_id >= 0
        assert (
            simulation.model.jnt_type[joint_id]
            == mujoco.mjtJoint.mjJNT_FREE
        )


def test_dynamic_objects_have_mass() -> None:
    simulation = BimanualTableSimulation.create()

    for object_id in (
        "plate_1",
        "plate_2",
        "glass_1",
        "glass_2",
    ):
        body_id = mujoco.mj_name2id(
            simulation.model,
            mujoco.mjtObj.mjOBJ_BODY,
            object_id,
        )

        assert simulation.model.body_mass[body_id] > 0.0


def test_object_manager_reports_positions() -> None:
    simulation = BimanualTableSimulation.create()

    manager = DynamicObjectManager(
        simulation.model,
        simulation.data,
    )

    plate = manager.get("plate_1")

    assert plate.object_id == "plate_1"
    assert plate.object_type == "plate"

    np.testing.assert_allclose(
        plate.position,
        (-0.28, 0.0, 0.772),
        atol=1e-5,
    )


def test_objects_stay_on_table_after_settling() -> None:
    simulation = BimanualTableSimulation.create()

    manager = DynamicObjectManager(
        simulation.model,
        simulation.data,
    )

    simulation.step(500)

    for object_id in (
        "plate_1",
        "plate_2",
        "glass_1",
        "glass_2",
    ):
        obj = manager.get(object_id)

        assert obj.position[2] >= 0.75