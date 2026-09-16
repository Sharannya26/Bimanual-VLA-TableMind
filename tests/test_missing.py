from tablemind.perception.scene import (
    SceneObject,
    SceneObservation,
)
from tablemind.reasoning.missing import (
    MissingObjectAnalyzer,
)
from tablemind.reasoning.requirements import (
    TaskRequirementBuilder,
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
            SceneObject(
                object_id="glass_3",
                object_type="glass",
                position=(0.0, -0.14, 0.835),
            ),
        )
    )


def test_analyzer_detects_missing_plates():
    requirements = TaskRequirementBuilder().build(
        make_request()
    )

    results = MissingObjectAnalyzer().analyze(
        requirements,
        make_scene(),
    )

    plate = next(
        result
        for result in results
        if result.object_type == "plate"
    )

    assert plate.required_quantity == 4
    assert plate.available_quantity == 2
    assert plate.missing_quantity == 2


def test_analyzer_detects_missing_glasses():
    requirements = TaskRequirementBuilder().build(
        make_request()
    )

    results = MissingObjectAnalyzer().analyze(
        requirements,
        make_scene(),
    )

    glass = next(
        result
        for result in results
        if result.object_type == "glass"
    )

    assert glass.required_quantity == 4
    assert glass.available_quantity == 3
    assert glass.missing_quantity == 1


def test_analyzer_reports_zero_when_requirement_is_satisfied():
    requirements = TaskRequirementBuilder().build(
        make_request()
    )

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
                object_id="plate_3",
                object_type="plate",
                position=(-0.10, 0.0, 0.772),
            ),
            SceneObject(
                object_id="plate_4",
                object_type="plate",
                position=(0.10, 0.0, 0.772),
            ),
        )
    )

    results = MissingObjectAnalyzer().analyze(
        requirements,
        scene,
    )

    plate = next(
        result
        for result in results
        if result.object_type == "plate"
    )

    assert plate.missing_quantity == 0


def test_analyzer_handles_empty_scene():
    requirements = TaskRequirementBuilder().build(
        make_request()
    )

    scene = SceneObservation(objects=())

    results = MissingObjectAnalyzer().analyze(
        requirements,
        scene,
    )

    assert all(
        result.missing_quantity == 4
        for result in results
    )