"""Tests for TABLEMIND deterministic planning."""

from __future__ import annotations

from tablemind.perception.state_builder import SceneStateBuilder
from tablemind.planning.planner import DeterministicTablePlanner
from tablemind.planning.task import (
    ActionType,
    TaskPlan,
)
from tablemind.simulation.runtime import BimanualTableSimulation


def build_scene():
    """Create a fresh TABLEMIND scene state."""

    simulation = BimanualTableSimulation.create()

    builder = SceneStateBuilder(
        simulation.model,
        simulation.data,
    )

    return simulation, builder.build()


def test_set_table_returns_task_plan() -> None:
    _, scene = build_scene()

    planner = DeterministicTablePlanner()

    plan = planner.plan_set_table(scene)

    assert isinstance(plan, TaskPlan)


def test_set_table_contains_all_objects() -> None:
    _, scene = build_scene()

    planner = DeterministicTablePlanner()

    plan = planner.plan_set_table(scene)

    object_ids = {
        action.object_id
        for action in plan.actions
        if action.object_id is not None
    }

    assert object_ids == {
        "plate_1",
        "plate_2",
        "glass_1",
        "glass_2",
    }


def test_each_object_has_five_actions() -> None:
    _, scene = build_scene()

    planner = DeterministicTablePlanner()

    plan = planner.plan_set_table(scene)

    assert plan.action_count == 20


def test_action_types_are_valid() -> None:
    _, scene = build_scene()

    planner = DeterministicTablePlanner()

    plan = planner.plan_set_table(scene)

    for action in plan.actions:
        assert isinstance(action.action_type, ActionType)


def test_left_objects_use_left_arm() -> None:
    _, scene = build_scene()

    planner = DeterministicTablePlanner()

    plan = planner.plan_set_table(scene)

    left_object_ids = {
        action.object_id
        for action in plan.actions
        if action.object_id in {"plate_1", "glass_1"}
    }

    left_actions = [
        action
        for action in plan.actions
        if action.object_id in left_object_ids
    ]

    assert all(
        action.arm == "left"
        for action in left_actions
    )


def test_right_objects_use_right_arm() -> None:
    _, scene = build_scene()

    planner = DeterministicTablePlanner()

    plan = planner.plan_set_table(scene)

    right_object_ids = {
        action.object_id
        for action in plan.actions
        if action.object_id in {"plate_2", "glass_2"}
    }

    right_actions = [
        action
        for action in plan.actions
        if action.object_id in right_object_ids
    ]

    assert all(
        action.arm == "right"
        for action in right_actions
    )


def test_actions_have_correct_order() -> None:
    _, scene = build_scene()

    planner = DeterministicTablePlanner()

    plan = planner.plan_set_table(scene)

    first_object_actions = plan.actions[:5]

    assert [
        action.action_type
        for action in first_object_actions
    ] == [
        ActionType.APPROACH,
        ActionType.OPEN_GRIPPER,
        ActionType.MOVE,
        ActionType.CLOSE_GRIPPER,
        ActionType.LIFT,
    ]


def test_action_ids_are_unique() -> None:
    _, scene = build_scene()

    planner = DeterministicTablePlanner()

    plan = planner.plan_set_table(scene)

    action_ids = [
        action.action_id
        for action in plan.actions
    ]

    assert len(action_ids) == len(set(action_ids))