from tablemind.perception.scene import SceneObject, SceneObservation
from tablemind.perception.state import RobotState, SceneState
from tablemind.reasoning.decomposition import (
    DecomposedTask,
    TaskStep,
    TaskStepType,
)
from tablemind.reasoning.sequence import TaskSequencePlanner
from tablemind.reasoning.planning_bridge import ReasoningPlanningBridge


def make_scene() -> SceneState:
    objects = SceneObservation(
        objects=(
            SceneObject(
                object_id="plate_1",
                object_type="plate",
                position=(-0.28, 0.0, 0.772),
            ),
        )
    )

    robots = (
        RobotState(
            arm_id="left",
            end_effector_position=(-0.20, 0.0, 0.90),
            gripper_position=1.0,
        ),
        RobotState(
            arm_id="right",
            end_effector_position=(0.20, 0.0, 0.90),
            gripper_position=1.0,
        ),
    )

    return SceneState(
        objects=objects,
        robots=robots,
        simulation_time=0.0,
    )


def make_sequence():
    task = DecomposedTask(
        name="set_table",
        steps=(
            TaskStep(
                step_type=TaskStepType.PICK,
                object_id="plate_1",
                object_type="plate",
                arm="left",
            ),
            TaskStep(
                step_type=TaskStepType.PLACE,
                object_id="plate_1",
                object_type="plate",
                arm="left",
                target_position=(-0.22, 0.0, 0.772),
            ),
        ),
    )

    return TaskSequencePlanner().create_sequence(task)


def test_bridge_creates_plan_from_reasoning_sequence():
    scene = make_scene()
    sequence = make_sequence()

    bridge = ReasoningPlanningBridge()
    plan = bridge.build_plan(sequence, scene)

    assert plan.task.task_id == "set_table"
    assert len(plan.actions) > 0


def test_bridge_preserves_arm_assignment():
    scene = make_scene()
    sequence = make_sequence()

    bridge = ReasoningPlanningBridge()
    plan = bridge.build_plan(sequence, scene)

    assert all(action.arm == "left" for action in plan.actions)


def test_bridge_creates_pick_actions():
    scene = make_scene()
    sequence = make_sequence()

    bridge = ReasoningPlanningBridge()
    plan = bridge.build_plan(sequence, scene)

    action_types = [action.action_type for action in plan.actions]

    assert action_types[0].value == "approach"
    assert action_types[1].value == "open_gripper"
    assert action_types[2].value == "move"
    assert action_types[3].value == "close_gripper"
    assert action_types[4].value == "lift"


def test_bridge_creates_place_action():
    scene = make_scene()
    sequence = make_sequence()

    bridge = ReasoningPlanningBridge()
    plan = bridge.build_plan(sequence, scene)

    place_actions = [
        action
        for action in plan.actions
        if action.action_type.value == "place"
    ]

    assert len(place_actions) == 1

    assert place_actions[0].target_position == (
        -0.22,
        0.0,
        0.772,
    )


def test_bridge_rejects_missing_object():
    scene = make_scene()

    task = DecomposedTask(
        name="invalid_task",
        steps=(
            TaskStep(
                step_type=TaskStepType.PICK,
                object_id="does_not_exist",
                object_type="plate",
                arm="left",
            ),
        ),
    )

    sequence = TaskSequencePlanner().create_sequence(task)

    bridge = ReasoningPlanningBridge()

    try:
        bridge.build_plan(sequence, scene)
    except ValueError as exc:
        assert "does_not_exist" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for missing object."
        )