from tablemind.perception.scene import (
    SceneObject,
    SceneObservation,
)
from tablemind.reasoning.object_selector import (
    QuantityAwareObjectSelector,
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


def test_selector_respects_quantity_two():
    request = TaskRequest(
        raw_instruction="Set the table for two.",
        intent="set_table",
        object_types=("plate", "glass"),
        quantity=2,
    )

    selected = QuantityAwareObjectSelector().select(
        request,
        make_scene(),
    )

    assert len(selected) == 2


def test_selector_returns_expected_objects_first():
    request = TaskRequest(
        raw_instruction="Set the table for two.",
        intent="set_table",
        object_types=("plate", "glass"),
        quantity=2,
    )

    selected = QuantityAwareObjectSelector().select(
        request,
        make_scene(),
    )

    assert selected == (
        "plate_1",
        "plate_2",
    )


def test_selector_can_select_four_objects():
    request = TaskRequest(
        raw_instruction="Set the table for four.",
        intent="set_table",
        object_types=("plate", "glass"),
        quantity=4,
    )

    selected = QuantityAwareObjectSelector().select(
        request,
        make_scene(),
    )

    assert selected == (
        "plate_1",
        "plate_2",
        "glass_1",
        "glass_2",
    )


def test_selector_returns_available_objects_when_quantity_exceeds_scene():
    request = TaskRequest(
        raw_instruction="Set the table for six.",
        intent="set_table",
        object_types=("plate", "glass"),
        quantity=6,
    )

    selected = QuantityAwareObjectSelector().select(
        request,
        make_scene(),
    )

    assert len(selected) == 4


def test_selector_uses_all_objects_when_quantity_is_missing():
    request = TaskRequest(
        raw_instruction="Set the table.",
        intent="set_table",
        object_types=("plate", "glass"),
        quantity=None,
    )

    selected = QuantityAwareObjectSelector().select(
        request,
        make_scene(),
    )

    assert len(selected) == 4