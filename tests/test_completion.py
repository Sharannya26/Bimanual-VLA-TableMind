from tablemind.perception.scene import (
    SceneObject,
    SceneObservation,
)
from tablemind.reasoning.completion import (
    CompletionAnalyzer,
)


def test_plate_at_expected_height_is_complete():
    scene = SceneObservation(
        objects=(
            SceneObject(
                object_id="plate_1",
                object_type="plate",
                position=(-0.28, 0.0, 0.772),
            ),
        )
    )

    results = CompletionAnalyzer().analyze(scene)

    assert len(results) == 1
    assert results[0].object_id == "plate_1"
    assert results[0].is_complete is True
    assert results[0].position_error == 0.0


def test_plate_above_table_is_incomplete():
    scene = SceneObservation(
        objects=(
            SceneObject(
                object_id="plate_1",
                object_type="plate",
                position=(-0.28, 0.0, 0.90),
            ),
        )
    )

    results = CompletionAnalyzer().analyze(scene)

    assert results[0].is_complete is False
    assert results[0].position_error > 0.06


def test_glass_at_expected_height_is_complete():
    scene = SceneObservation(
        objects=(
            SceneObject(
                object_id="glass_1",
                object_type="glass",
                position=(0.40, -0.14, 0.835),
            ),
        )
    )

    results = CompletionAnalyzer().analyze(scene)

    assert results[0].is_complete is True


def test_multiple_objects_are_analyzed():
    scene = SceneObservation(
        objects=(
            SceneObject(
                object_id="plate_1",
                object_type="plate",
                position=(-0.28, 0.0, 0.772),
            ),
            SceneObject(
                object_id="glass_1",
                object_type="glass",
                position=(0.40, -0.14, 0.90),
            ),
        )
    )

    results = CompletionAnalyzer().analyze(scene)

    assert len(results) == 2
    assert results[0].is_complete is True
    assert results[1].is_complete is False