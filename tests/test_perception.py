"""Tests for TABLEMIND Milestone 3 perception foundations."""

from __future__ import annotations

import numpy as np

from tablemind.perception.camera import TableCamera
from tablemind.perception.scene import SceneObject, SceneObservation
from tablemind.simulation import BimanualTableSimulation


def test_table_camera_exists() -> None:
    simulation = BimanualTableSimulation.create()

    camera_id = simulation.model.camera("table_overview").id

    assert camera_id >= 0


def test_table_camera_captures_rgb_frame() -> None:
    simulation = BimanualTableSimulation.create()

    camera = TableCamera(
        simulation.model,
        simulation.data,
        width=320,
        height=240,
    )

    try:
        image = camera.capture()

        assert isinstance(image, np.ndarray)
        assert image.shape == (240, 320, 3)
        assert image.dtype == np.uint8

    finally:
        camera.close()


def test_scene_observation_representation() -> None:
    observation = SceneObservation(
        objects=(
            SceneObject(
                object_id="plate_1",
                object_type="plate",
                position=(-0.28, 0.16, 0.77),
            ),
        )
    )

    assert len(observation.objects) == 1
    assert observation.get("plate_1") is not None
    assert observation.by_type("plate")[0].object_id == "plate_1"