"""Tests for TABLEMIND camera-to-world grounding."""

from __future__ import annotations

from tablemind.perception.camera_grounding import (
    CameraGrounder,
)
from tablemind.perception.coordinates import (
    WorldPoint,
)
from tablemind.simulation import (
    BimanualTableSimulation,
)


TABLE_TOP_Z = 0.76


def test_camera_grounder_can_be_created() -> None:
    simulation = BimanualTableSimulation.create()

    grounder = CameraGrounder(
        simulation.model,
        simulation.data,
    )

    assert grounder.camera_id >= 0


def test_camera_center_projects_to_table_plane() -> None:
    simulation = BimanualTableSimulation.create()

    grounder = CameraGrounder(
        simulation.model,
        simulation.data,
    )

    point = grounder.pixel_to_world_on_plane(
        320,
        240,
        plane_z=TABLE_TOP_Z,
    )

    assert isinstance(
        point,
        WorldPoint,
    )

    assert abs(
        point.z - TABLE_TOP_Z
    ) < 1e-6


def test_grounded_point_contains_finite_coordinates() -> None:
    simulation = BimanualTableSimulation.create()

    grounder = CameraGrounder(
        simulation.model,
        simulation.data,
    )

    point = grounder.pixel_to_world_on_plane(
        320,
        240,
        plane_z=TABLE_TOP_Z,
    )

    assert all(
        value == value
        for value in point.xyz
    )


def test_bounding_box_center_can_be_grounded() -> None:
    simulation = BimanualTableSimulation.create()

    grounder = CameraGrounder(
        simulation.model,
        simulation.data,
    )

    point = grounder.bounding_box_center_to_world(
        300,
        220,
        340,
        260,
        plane_z=TABLE_TOP_Z,
    )

    assert isinstance(
        point,
        WorldPoint,
    )

    assert abs(
        point.z - TABLE_TOP_Z
    ) < 1e-6