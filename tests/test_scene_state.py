"""Tests for TABLEMIND structured scene state."""

from __future__ import annotations

import pytest

from tablemind.perception.state import RobotState, SceneState
from tablemind.perception.state_builder import SceneStateBuilder
from tablemind.simulation.runtime import BimanualTableSimulation


def test_scene_state_contains_objects_and_robots() -> None:
    simulation = BimanualTableSimulation.create()

    builder = SceneStateBuilder(
        simulation.model,
        simulation.data,
    )

    state = builder.build()

    assert isinstance(state, SceneState)

    assert len(state.objects.objects) == 4
    assert len(state.robots) == 2


def test_scene_state_contains_both_arms() -> None:
    simulation = BimanualTableSimulation.create()

    builder = SceneStateBuilder(
        simulation.model,
        simulation.data,
    )

    state = builder.build()

    assert state.robot("left") is not None
    assert state.robot("right") is not None


def test_robot_state_has_end_effector_position() -> None:
    simulation = BimanualTableSimulation.create()

    builder = SceneStateBuilder(
        simulation.model,
        simulation.data,
    )

    state = builder.build()

    left = state.robot("left")

    assert left is not None
    assert isinstance(left, RobotState)

    assert len(left.end_effector_position) == 3


def test_scene_state_time_matches_simulation() -> None:
    simulation = BimanualTableSimulation.create()

    simulation.step(10)

    builder = SceneStateBuilder(
        simulation.model,
        simulation.data,
    )

    state = builder.build()

    assert state.simulation_time == pytest.approx(
        simulation.data.time
    )


def test_scene_state_to_dict() -> None:
    simulation = BimanualTableSimulation.create()

    builder = SceneStateBuilder(
        simulation.model,
        simulation.data,
    )

    state = builder.build()

    result = state.to_dict()

    assert "objects" in result
    assert "robots" in result
    assert "simulation_time" in result

    assert len(result["objects"]) == 4
    assert len(result["robots"]) == 2


def test_scene_state_object_positions() -> None:
    simulation = BimanualTableSimulation.create()

    builder = SceneStateBuilder(
        simulation.model,
        simulation.data,
    )

    state = builder.build()

    plate = state.objects.get("plate_1")

    assert plate is not None

    assert plate.position[0] == pytest.approx(
        -0.28,
        abs=1e-3,
    )

    assert plate.position[1] == pytest.approx(
        0.00,
        abs=1e-3,
    )


def test_unknown_robot_returns_none() -> None:
    simulation = BimanualTableSimulation.create()

    builder = SceneStateBuilder(
        simulation.model,
        simulation.data,
    )

    state = builder.build()

    assert state.robot("middle") is None