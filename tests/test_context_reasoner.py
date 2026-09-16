from tablemind.perception.scene import (
    SceneObject,
    SceneObservation,
)
from tablemind.reasoning.context_reasoner import (
    ContextReasoner,
)
from tablemind.reasoning.task_request import (
    TaskRequest,
)


def make_request() -> TaskRequest:
    return TaskRequest(
        raw_instruction="Set the table for four.",
        intent="set_table",
        object_types=("plate", "glass"),
        quantity=4,
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


def test_reasoner_builds_complete_context():
    result = ContextReasoner().reason(
        make_request(),
        make_scene(),
    )

    assert result.request.intent == "set_table"
    assert result.requirements.object_types == (
        "plate",
        "glass",
    )


def test_reasoner_identifies_completed_objects():
    result = ContextReasoner().reason(
        make_request(),
        make_scene(),
    )

    assert result.completed_objects == (
        "plate_1",
        "glass_1",
    )


def test_reasoner_identifies_incomplete_objects():
    result = ContextReasoner().reason(
        make_request(),
        make_scene(),
    )

    assert result.incomplete_objects == (
        "plate_2",
        "glass_2",
    )


def test_reasoner_identifies_missing_quantities():
    result = ContextReasoner().reason(
        make_request(),
        make_scene(),
    )

    missing = {
        item.object_type: item.missing_quantity
        for item in result.missing
    }

    assert missing["plate"] == 2
    assert missing["glass"] == 2


def test_reasoner_calculates_total_missing():
    result = ContextReasoner().reason(
        make_request(),
        make_scene(),
    )

    assert result.total_missing == 4