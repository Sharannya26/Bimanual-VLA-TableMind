"""Tests for the TABLEMIND closed-loop VLA orchestrator."""

from __future__ import annotations

from pathlib import Path

import pytest

from tablemind.integration.closed_loop import (
    ClosedLoopResult,
    ClosedLoopVLA,
)
from tablemind.simulation.runtime import (
    BimanualTableSimulation,
)


def test_closed_loop_module_is_importable() -> None:
    assert ClosedLoopVLA is not None
    assert ClosedLoopResult is not None


def test_closed_loop_create_constructs_real_components() -> None:
    simulation = BimanualTableSimulation.create()

    pipeline = ClosedLoopVLA.create(
        simulation=simulation,
    )

    assert pipeline.simulation is simulation
    assert pipeline.vision is not None
    assert pipeline.reasoning is not None
    assert pipeline.planning is not None
    assert pipeline.execution is not None
    assert pipeline.executor is not None
    assert pipeline.controller is not None
    assert pipeline.verifier is not None

    pipeline.close()


def test_robot_states_use_actual_so101_positions() -> None:
    simulation = BimanualTableSimulation.create()

    pipeline = ClosedLoopVLA.create(
        simulation=simulation,
    )

    states = pipeline._robot_states()

    assert len(states) == 2

    arm_ids = {
        state.arm_id
        for state in states
    }

    assert arm_ids == {
        "left",
        "right",
    }

    for state in states:
        assert len(
            state.end_effector_position
        ) == 3

    pipeline.close()


def test_empty_instruction_is_rejected() -> None:
    simulation = BimanualTableSimulation.create()

    pipeline = ClosedLoopVLA.create(
        simulation=simulation,
    )

    with pytest.raises(ValueError):
        pipeline.run("")

    pipeline.close()


def test_blank_instruction_is_rejected() -> None:
    simulation = BimanualTableSimulation.create()

    pipeline = ClosedLoopVLA.create(
        simulation=simulation,
    )

    with pytest.raises(ValueError):
        pipeline.run("   ")

    pipeline.close()