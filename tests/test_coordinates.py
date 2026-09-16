"""Tests for TABLEMIND world-coordinate utilities."""

from __future__ import annotations

import numpy as np
import pytest

from tablemind.perception.coordinates import (
    WorldCoordinateSystem,
    WorldPoint,
)
from tablemind.simulation.runtime import BimanualTableSimulation


def test_world_point_xyz() -> None:
    point = WorldPoint(
        x=1.0,
        y=2.0,
        z=3.0,
    )

    assert point.xyz == (1.0, 2.0, 3.0)
    assert np.allclose(
        point.as_array(),
        np.array([1.0, 2.0, 3.0]),
    )


def test_geom_position() -> None:
    simulation = BimanualTableSimulation.create()

    coordinates = WorldCoordinateSystem(
        simulation.model,
        simulation.data,
    )

    plate = coordinates.geom_position("plate_1")

    assert isinstance(plate, WorldPoint)

    assert plate.x == pytest.approx(-0.28, abs=1e-3)
    assert plate.y == pytest.approx(0.00, abs=1e-3)
    assert plate.z == pytest.approx(0.772, abs=1e-3)


def test_glass_position() -> None:
    simulation = BimanualTableSimulation.create()

    coordinates = WorldCoordinateSystem(
        simulation.model,
        simulation.data,
    )

    glass = coordinates.geom_position("glass_1")

    assert isinstance(glass, WorldPoint)

    assert glass.x == pytest.approx(-0.40, abs=1e-3)
    assert glass.y == pytest.approx(-0.14, abs=1e-3)
    assert glass.z == pytest.approx(0.835, abs=1e-3)


def test_distance() -> None:
    first = WorldPoint(
        x=0.0,
        y=0.0,
        z=0.0,
    )

    second = WorldPoint(
        x=3.0,
        y=4.0,
        z=0.0,
    )

    simulation = BimanualTableSimulation.create()

    coordinates = WorldCoordinateSystem(
        simulation.model,
        simulation.data,
    )

    assert coordinates.distance(first, second) == pytest.approx(5.0)


def test_missing_geometry_raises_error() -> None:
    simulation = BimanualTableSimulation.create()

    coordinates = WorldCoordinateSystem(
        simulation.model,
        simulation.data,
    )

    with pytest.raises(KeyError):
        coordinates.geom_position("does_not_exist")