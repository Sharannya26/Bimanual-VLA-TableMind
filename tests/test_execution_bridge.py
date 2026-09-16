from tablemind.planning.task import (
    Action,
    ActionType,
    Task,
    TaskPlan,
)
from tablemind.reasoning.execution_bridge import (
    ReasoningExecutionBridge,
)


def make_plan() -> TaskPlan:
    task = Task(
        task_id="set_table",
        description="Set the table.",
    )

    actions = (
        Action(
            action_id="action_1_approach",
            action_type=ActionType.APPROACH,
            arm="left",
            object_id="plate_1",
            target_position=(-0.28, 0.0, 0.89),
        ),
        Action(
            action_id="action_1_open",
            action_type=ActionType.OPEN_GRIPPER,
            arm="left",
            object_id="plate_1",
        ),
        Action(
            action_id="action_1_move",
            action_type=ActionType.MOVE,
            arm="left",
            object_id="plate_1",
            target_position=(-0.28, 0.0, 0.772),
        ),
        Action(
            action_id="action_1_close",
            action_type=ActionType.CLOSE_GRIPPER,
            arm="left",
            object_id="plate_1",
        ),
        Action(
            action_id="action_1_lift",
            action_type=ActionType.LIFT,
            arm="left",
            object_id="plate_1",
            target_position=(-0.28, 0.0, 0.95),
        ),
        Action(
            action_id="action_1_place",
            action_type=ActionType.PLACE,
            arm="left",
            object_id="plate_1",
            target_position=(-0.22, 0.0, 0.772),
        ),
    )

    return TaskPlan(
        task=task,
        actions=actions,
    )


def test_bridge_creates_executable_task():
    result = ReasoningExecutionBridge().build_task(
        make_plan()
    )

    assert result.task.name == "set_table"
    assert result.assignment_count == 1


def test_bridge_preserves_arm_assignment():
    result = ReasoningExecutionBridge().build_task(
        make_plan()
    )

    assignment = result.task.assignment_for("left")

    assert assignment.arm == "left"
    assert assignment.object_id == "plate_1"


def test_bridge_preserves_planned_positions():
    result = ReasoningExecutionBridge().build_task(
        make_plan()
    )

    assignment = result.task.assignment_for("left")

    assert assignment.approach_position == (
        -0.28,
        0.0,
        0.89,
    )

    assert assignment.grasp_position == (
        -0.28,
        0.0,
        0.772,
    )

    assert assignment.lift_position == (
        -0.28,
        0.0,
        0.95,
    )

    assert assignment.place_position == (
        -0.22,
        0.0,
        0.772,
    )


def test_bridge_deduplicates_actions_for_same_object():
    result = ReasoningExecutionBridge().build_task(
        make_plan()
    )

    assert result.assignment_count == 1
    assert result.task.objects == ("plate_1",)


def test_bridge_supports_two_arm_plan():
    plan = make_plan()

    extra_actions = (
        Action(
            action_id="action_2_approach",
            action_type=ActionType.APPROACH,
            arm="right",
            object_id="plate_2",
            target_position=(0.28, 0.0, 0.89),
        ),
        Action(
            action_id="action_2_move",
            action_type=ActionType.MOVE,
            arm="right",
            object_id="plate_2",
            target_position=(0.28, 0.0, 0.772),
        ),
        Action(
            action_id="action_2_lift",
            action_type=ActionType.LIFT,
            arm="right",
            object_id="plate_2",
            target_position=(0.28, 0.0, 0.95),
        ),
        Action(
            action_id="action_2_place",
            action_type=ActionType.PLACE,
            arm="right",
            object_id="plate_2",
            target_position=(0.22, 0.0, 0.772),
        ),
    )

    two_arm_plan = TaskPlan(
        task=plan.task,
        actions=plan.actions + extra_actions,
    )

    result = ReasoningExecutionBridge().build_task(
        two_arm_plan
    )

    assert result.assignment_count == 2
    assert result.task.arms == ("left", "right")
    assert result.task.objects == (
        "plate_1",
        "plate_2",
    )

    right = result.task.assignment_for("right")

    assert right.approach_position == (
        0.28,
        0.0,
        0.89,
    )

    assert right.grasp_position == (
        0.28,
        0.0,
        0.772,
    )

    assert right.lift_position == (
        0.28,
        0.0,
        0.95,
    )

    assert right.place_position == (
        0.22,
        0.0,
        0.772,
    )


def test_bridge_rejects_plan_without_executable_objects():
    task = Task(
        task_id="empty_task",
        description="No manipulation.",
    )

    plan = TaskPlan(
        task=task,
        actions=(
            Action(
                action_id="open",
                action_type=ActionType.OPEN_GRIPPER,
                arm="left",
                object_id="plate_1",
            ),
        ),
    )

    try:
        ReasoningExecutionBridge().build_task(plan)
    except ValueError as exc:
        assert "no executable object assignments" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for an empty executable plan."
        )