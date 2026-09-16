from tablemind.perception.scene import (
    SceneObject,
    SceneObservation,
)
from tablemind.reasoning.context import (
    ContextAnalyzer,
)
from tablemind.reasoning.task_request import (
    TaskRequest,
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


def test_context_finds_relevant_objects():
    request = TaskRequest(
        raw_instruction="Set the table for two.",
        intent="set_table",
        object_types=("plate", "glass"),
        quantity=2,
    )

    context = ContextAnalyzer().analyze(
        request,
        make_scene(),
    )

    assert context.available_objects == (
        "plate_1",
        "plate_2",
        "glass_1",
        "glass_2",
    )


def test_context_calculates_missing_quantity():
    request = TaskRequest(
        raw_instruction="Set the table for six.",
        intent="set_table",
        object_types=("plate", "glass"),
        quantity=6,
    )

    context = ContextAnalyzer().analyze(
        request,
        make_scene(),
    )

    assert context.required_quantity == 6
    assert context.missing_quantity == 2


def test_context_is_satisfied_when_enough_objects_exist():
    request = TaskRequest(
        raw_instruction="Set the table for two.",
        intent="set_table",
        object_types=("plate", "glass"),
        quantity=2,
    )

    context = ContextAnalyzer().analyze(
        request,
        make_scene(),
    )

    assert context.is_satisfied is True


def test_context_is_not_satisfied_when_objects_are_missing():
    request = TaskRequest(
        raw_instruction="Set the table for six.",
        intent="set_table",
        object_types=("plate", "glass"),
        quantity=6,
    )

    context = ContextAnalyzer().analyze(
        request,
        make_scene(),
    )

    assert context.is_satisfied is False