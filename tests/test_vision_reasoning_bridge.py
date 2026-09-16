"""Tests for TABLEMIND vision-to-reasoning integration."""

from __future__ import annotations

from tablemind.perception.state import RobotState
from tablemind.perception.vision_scene_state import (
    ScenePosition,
    SceneState,
    VisionSceneObject,
)
from tablemind.reasoning.interpreter import TaskInterpreter
from tablemind.reasoning.vision_reasoning_bridge import (
    ReasoningSceneAdapter,
    VisionReasoningBridge,
)


def _vision_scene() -> SceneState:
    """Create a deterministic four-object visual scene."""

    objects = (
        VisionSceneObject(
            object_id="plate_1",
            object_type="plate",
            confidence=0.95,
            bounding_box=(230.0, 75.0, 290.0, 105.0),
            pixel_center=(260.0, 90.0),
            position=ScenePosition(
                x=-0.22,
                y=0.0,
                z=0.772,
            ),
        ),
        VisionSceneObject(
            object_id="plate_2",
            object_type="plate",
            confidence=0.94,
            bounding_box=(350.0, 75.0, 410.0, 105.0),
            pixel_center=(380.0, 90.0),
            position=ScenePosition(
                x=0.22,
                y=0.0,
                z=0.772,
            ),
        ),
        VisionSceneObject(
            object_id="glass_1",
            object_type="glass",
            confidence=0.91,
            bounding_box=(238.0, 68.0, 262.0, 106.0),
            pixel_center=(250.0, 87.0),
            position=ScenePosition(
                x=-0.25,
                y=-0.14,
                z=0.835,
            ),
        ),
        VisionSceneObject(
            object_id="glass_2",
            object_type="glass",
            confidence=0.90,
            bounding_box=(378.0, 68.0, 402.0, 106.0),
            pixel_center=(390.0, 87.0),
            position=ScenePosition(
                x=0.25,
                y=-0.14,
                z=0.835,
            ),
        ),
    )

    return SceneState(
        objects=objects,
        camera_width=640,
        camera_height=480,
        source="test",
    )


def _robot_states() -> tuple[RobotState, ...]:
    """Create deterministic robot states."""

    return (
        RobotState(
            arm_id="left",
            end_effector_position=(-0.48, -0.43, 0.76),
            gripper_position=0.0,
        ),
        RobotState(
            arm_id="right",
            end_effector_position=(0.48, -0.43, 0.76),
            gripper_position=0.0,
        ),
    )


def test_adapter_converts_vision_objects() -> None:
    """Vision objects should become legacy reasoning scene objects."""

    vision_scene = _vision_scene()

    scene = ReasoningSceneAdapter().convert(
        vision_scene=vision_scene,
        robot_states=_robot_states(),
        simulation_time=2.0,
    )

    assert len(scene.objects.objects) == 4
    assert scene.objects.get("plate_1") is not None
    assert scene.objects.get("glass_2") is not None

    assert scene.simulation_time == 2.0


def test_adapter_preserves_world_coordinates() -> None:
    """Calibrated world coordinates should survive the adapter."""

    scene = ReasoningSceneAdapter().convert(
        vision_scene=_vision_scene(),
        robot_states=_robot_states(),
    )

    plate = scene.objects.get("plate_1")

    assert plate is not None

    assert plate.position == (
        -0.22,
        0.0,
        0.772,
    )


def test_set_table_for_two_grounds_four_objects() -> None:
    """A quantity of two should select two plates and two glasses."""

    bridge = VisionReasoningBridge(
        interpreter=TaskInterpreter(),
    )

    result = bridge.build(
        instruction="Set the table for two.",
        vision_scene=_vision_scene(),
        robot_states=_robot_states(),
    )

    assert result.request.intent == "set_table"
    assert result.request.quantity == 2

    assert len(result.grounded.matches) == 4

    object_types = tuple(
        match.object_type
        for match in result.grounded.matches
    )

    assert object_types.count("plate") == 2
    assert object_types.count("glass") == 2


def test_grounding_preserves_object_ids() -> None:
    """Grounding should preserve detector-generated object IDs."""

    bridge = VisionReasoningBridge(
        interpreter=TaskInterpreter(),
    )

    result = bridge.build(
        instruction="Set the table for two.",
        vision_scene=_vision_scene(),
        robot_states=_robot_states(),
    )

    object_ids = {
        match.object_id
        for match in result.grounded.matches
    }

    assert object_ids == {
        "plate_1",
        "plate_2",
        "glass_1",
        "glass_2",
    }


def test_reasoning_decomposition_creates_pick_and_place_steps() -> None:
    """Each grounded object should receive pick and place reasoning steps."""

    bridge = VisionReasoningBridge(
        interpreter=TaskInterpreter(),
    )

    result = bridge.build(
        instruction="Set the table for two.",
        vision_scene=_vision_scene(),
        robot_states=_robot_states(),
    )

    assert result.decomposed_task.step_count == 8

    pick_count = sum(
        step.step_type.value == "pick"
        for step in result.decomposed_task.steps
    )

    place_count = sum(
        step.step_type.value == "place"
        for step in result.decomposed_task.steps
    )

    assert pick_count == 4
    assert place_count == 4


def test_reasoning_sequence_contains_pick_and_place_stages() -> None:
    """The sequence planner should create the expected execution stages."""

    bridge = VisionReasoningBridge(
        interpreter=TaskInterpreter(),
    )

    result = bridge.build(
        instruction="Set the table for two.",
        vision_scene=_vision_scene(),
        robot_states=_robot_states(),
    )

    assert result.sequence.stage_count == 2
    assert result.sequence.step_count == 8

    assert result.sequence.stages[0].step_count == 4
    assert result.sequence.stages[1].step_count == 4


def test_quantity_is_applied_per_object_type() -> None:
    """Quantity should mean N plates and N glasses, not N total objects."""

    bridge = VisionReasoningBridge(
        interpreter=TaskInterpreter(),
    )

    result = bridge.build(
        instruction="Set the table for one.",
        vision_scene=_vision_scene(),
        robot_states=_robot_states(),
    )

    assert len(result.grounded.matches) == 2

    object_types = [
        match.object_type
        for match in result.grounded.matches
    ]

    assert object_types.count("plate") == 1
    assert object_types.count("glass") == 1


def test_all_visible_objects_are_used_when_quantity_is_unspecified() -> None:
    """Without a quantity, all requested object types should be grounded."""

    bridge = VisionReasoningBridge(
        interpreter=TaskInterpreter(),
    )

    result = bridge.build(
        instruction="Set the table.",
        vision_scene=_vision_scene(),
        robot_states=_robot_states(),
    )

    assert len(result.grounded.matches) == 4