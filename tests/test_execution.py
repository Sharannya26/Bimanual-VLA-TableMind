"""Tests for TABLEMIND action execution."""

from __future__ import annotations

from tablemind.perception.state_builder import SceneStateBuilder
from tablemind.planning.executor import (
    ActionExecutor,
    ActionResult,
    ExecutionReport,
)
from tablemind.planning.planner import DeterministicTablePlanner
from tablemind.planning.task import (
    Action,
    ActionType,
)
from tablemind.simulation.runtime import BimanualTableSimulation


def build_plan():
    """Build a deterministic test plan."""

    simulation = BimanualTableSimulation.create()

    builder = SceneStateBuilder(
        simulation.model,
        simulation.data,
    )

    scene = builder.build()

    planner = DeterministicTablePlanner()

    plan = planner.plan_set_table(scene)

    return simulation, plan


def test_action_result_structure() -> None:
    result = ActionResult(
        action_id="test_action",
        success=True,
        message="Action completed",
    )

    assert result.action_id == "test_action"
    assert result.success is True
    assert result.message == "Action completed"


def test_execution_report_success() -> None:
    results = (
        ActionResult(
            action_id="a1",
            success=True,
            message="ok",
        ),
        ActionResult(
            action_id="a2",
            success=True,
            message="ok",
        ),
    )

    report = ExecutionReport(results=results)

    assert report.success is True
    assert report.completed_actions == 2


def test_execution_report_failure() -> None:
    results = (
        ActionResult(
            action_id="a1",
            success=True,
            message="ok",
        ),
        ActionResult(
            action_id="a2",
            success=False,
            message="failed",
        ),
    )

    report = ExecutionReport(results=results)

    assert report.success is False
    assert report.completed_actions == 1


def test_open_gripper_executes() -> None:
    simulation, _ = build_plan()

    executor = ActionExecutor(simulation)

    action = Action(
        action_id="open_left",
        action_type=ActionType.OPEN_GRIPPER,
        arm="left",
    )

    result = executor.execute_action(action)

    assert result.success is True
    assert "opened" in result.message


def test_close_gripper_executes() -> None:
    simulation, _ = build_plan()

    executor = ActionExecutor(simulation)

    action = Action(
        action_id="close_left",
        action_type=ActionType.CLOSE_GRIPPER,
        arm="left",
    )

    result = executor.execute_action(action)

    assert result.success is True
    assert "closed" in result.message


def test_invalid_arm_fails_cleanly() -> None:
    simulation, _ = build_plan()

    executor = ActionExecutor(simulation)

    action = Action(
        action_id="bad_arm",
        action_type=ActionType.OPEN_GRIPPER,
        arm="middle",
    )

    result = executor.execute_action(action)

    assert result.success is False
    assert "Unknown arm" in result.message


def test_move_without_target_fails_cleanly() -> None:
    simulation, _ = build_plan()

    executor = ActionExecutor(simulation)

    action = Action(
        action_id="missing_target",
        action_type=ActionType.MOVE,
        arm="left",
    )

    result = executor.execute_action(action)

    assert result.success is False
    assert "no target position" in result.message


def test_execute_single_move() -> None:
    simulation, _ = build_plan()

    executor = ActionExecutor(simulation)

    action = Action(
        action_id="move_left",
        action_type=ActionType.MOVE,
        arm="left",
        target_position=(
            -0.38,
            -0.0386,
            0.9865,
        ),
    )

    result = executor.execute_action(action)

    assert isinstance(result, ActionResult)


def test_execute_plan() -> None:
    simulation, _ = build_plan()

    executor = ActionExecutor(simulation)

    # Use a known reachable Cartesian target from Milestone 2.
    # This test validates the executor, not the planner's
    # object-reaching accuracy.
    actions = (
        Action(
            action_id="test_move",
            action_type=ActionType.MOVE,
            arm="left",
            target_position=(
                -0.38,
                -0.0386,
                0.9865,
            ),
        ),
        Action(
            action_id="test_open",
            action_type=ActionType.OPEN_GRIPPER,
            arm="left",
        ),
        Action(
            action_id="test_close",
            action_type=ActionType.CLOSE_GRIPPER,
            arm="left",
        ),
    )

    report = executor.execute_plan(
        actions,
        stop_on_failure=True,
    )

    assert isinstance(report, ExecutionReport)
    assert len(report.results) == len(actions)
    assert report.completed_actions == len(actions)
    assert report.success is True