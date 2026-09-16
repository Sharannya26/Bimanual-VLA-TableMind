from tablemind.perception.scene import (
    SceneObject,
    SceneObservation,
)
from tablemind.reasoning.dynamic_replanner import (
    DynamicReplanner,
)
from tablemind.reasoning.task_request import (
    TaskRequest,
)


def make_request() -> TaskRequest:
    return TaskRequest(
        raw_instruction="Set the table for two.",
        intent="set_table",
        object_types=("plate", "glass"),
        quantity=2,
    )


def test_replanner_returns_updated_plan():
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

    result = DynamicReplanner().replan(
        make_request(),
        scene,
    )

    assert result.needs_action is True

    assert "plate_2" in result.remaining_objects
    assert "glass_2" in result.remaining_objects

    assert "plate_1" not in result.remaining_objects
    assert "glass_1" not in result.remaining_objects


def test_replanner_updates_after_scene_changes():
    initial_scene = SceneObservation(
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

    replanner = DynamicReplanner()

    first_result = replanner.replan(
        make_request(),
        initial_scene,
    )

    assert first_result.needs_action is True

    updated_scene = SceneObservation(
        objects=(
            SceneObject(
                object_id="plate_1",
                object_type="plate",
                position=(-0.28, 0.0, 0.772),
            ),
            SceneObject(
                object_id="glass_1",
                object_type="glass",
                position=(-0.40, -0.14, 0.835),
            ),
        )
    )

    second_result = replanner.replan(
        make_request(),
        updated_scene,
    )

    assert second_result.needs_action is False
    assert second_result.plan.actions == ()


def test_replanner_preserves_task_identity():
    scene = SceneObservation(
        objects=(
            SceneObject(
                object_id="plate_1",
                object_type="plate",
                position=(-0.28, 0.0, 0.90),
            ),
        )
    )

    result = DynamicReplanner().replan(
        make_request(),
        scene,
    )

    assert result.plan.task.task_id == "set_table"