
from tablemind.perception.scene import (
    SceneObject,
    SceneObservation,
)
from tablemind.reasoning.context_planner import (
    ContextAwarePlanner,
)
from tablemind.reasoning.task_request import (
    TaskRequest,
)
from tablemind.planning.task import (
    ActionType,
)


def make_scene() -> SceneObservation:
    return SceneObservation(
        objects=(
            SceneObject(
                object_id="plate_1",
                object_type="plate",
                position=(-0.28, 0.0, 0.772),
            ),
            SceneObject(
                object_id="plate_2",
                object_type="plate",
                position=(0.28, 0.0, 0.90),
            ),
            SceneObject(
                object_id="glass_1",
                object_type="glass",
                position=(-0.40, -0.14, 0.835),
            ),
            SceneObject(
                object_id="glass_2",
                object_type="glass",
                position=(0.40, -0.14, 0.90),
            ),
        )
    )



def make_request() -> TaskRequest:
    return TaskRequest(
        raw_instruction="Set the table for two.",
        intent="set_table",
        object_types=("plate", "glass"),
        quantity=2,
    )


def test_planner_returns_task_plan():
    plan = ContextAwarePlanner().plan(
        make_request(),
        make_scene(),
    )

    assert plan.task.task_id == "set_table"


def test_planner_skips_completed_objects():
    plan = ContextAwarePlanner().plan(
        make_request(),
        make_scene(),
    )

    object_ids = {
        action.object_id
        for action in plan.actions
    }

    assert "plate_1" not in object_ids
    assert "glass_1" not in object_ids


def test_planner_includes_incomplete_objects():
    plan = ContextAwarePlanner().plan(
        make_request(),
        make_scene(),
    )

    object_ids = {
        action.object_id
        for action in plan.actions
    }

    assert "plate_2" in object_ids
    assert "glass_2" in object_ids


def test_planner_generates_manipulation_actions():
    plan = ContextAwarePlanner().plan(
        make_request(),
        make_scene(),
    )

    action_types = {
        action.action_type
        for action in plan.actions
    }

    assert ActionType.APPROACH in action_types
    assert ActionType.OPEN_GRIPPER in action_types
    assert ActionType.CLOSE_GRIPPER in action_types
    assert ActionType.LIFT in action_types
    assert ActionType.PLACE in action_types


def test_planner_assigns_arms_by_object_side():
    plan = ContextAwarePlanner().plan(
        make_request(),
        make_scene(),
    )

    assignments = {
        action.object_id: action.arm
        for action in plan.actions
        if action.action_type == ActionType.APPROACH
    }

    assert assignments["plate_2"] == "right"
    assert assignments["glass_2"] == "right"


def test_planner_does_not_create_actions_when_everything_is_complete():
    scene = SceneObservation(
        objects=(
            SceneObject(
                object_id="plate_1",
                object_type="plate",
                position=(-0.28, 0.0, 0.772),
            ),
            SceneObject(
                object_id="plate_2",
                object_type="plate",
                position=(0.28, 0.0, 0.772),
            ),
            SceneObject(
                object_id="glass_1",
                object_type="glass",
                position=(-0.40, -0.14, 0.835),
            ),
            SceneObject(
                object_id="glass_2",
                object_type="glass",
                position=(0.40, -0.14, 0.835),
            ),
        )
    )

    plan = ContextAwarePlanner().plan(
        make_request(),
        scene,
    )

    assert plan.actions == ()

def test_planner_can_build_from_existing_reasoning_result():
    planner = ContextAwarePlanner()

    request = make_request()
    scene = make_scene()

    reasoning = planner.context_reasoner.reason(
        request,
        scene,
    )

    plan = planner.plan_from_reasoning(
        reasoning,
    )

    assert plan.task.task_id == "set_table"

    object_ids = {
        action.object_id
        for action in plan.actions
    }

    assert "plate_1" not in object_ids
    assert "glass_1" not in object_ids
    assert "plate_2" in object_ids
    assert "glass_2" in object_ids


def test_planner_uses_reasoning_completion_state():
    planner = ContextAwarePlanner()

    request = make_request()

    scene = SceneObservation(
        objects=(
            SceneObject(
                object_id="plate_1",
                object_type="plate",
                position=(-0.28, 0.0, 0.90),
            ),
            SceneObject(
                object_id="glass_1",
                object_type="glass",
                position=(-0.40, -0.14, 0.90),
            ),
        )
    )

    reasoning = planner.context_reasoner.reason(
        request,
        scene,
    )

    plan = planner.plan_from_reasoning(
        reasoning,
    )

    object_ids = {
        action.object_id
        for action in plan.actions
    }

    assert "plate_1" in object_ids
    assert "glass_1" in object_ids