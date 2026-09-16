from tablemind.perception.scene import (
    SceneObject,
    SceneObservation,
)
from tablemind.reasoning.grounding import (
    GroundingMatch,
    WorldGrounder,
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


def test_find_plates() -> None:
    grounder = WorldGrounder()

    matches = grounder.find_by_type(
        make_scene(),
        "plate",
    )

    assert len(matches) == 2
    assert matches[0].object_id == "plate_1"
    assert matches[1].object_id == "plate_2"


def test_find_glasses() -> None:
    grounder = WorldGrounder()

    matches = grounder.find_by_type(
        make_scene(),
        "glass",
    )

    assert len(matches) == 2
    assert matches[0].object_id == "glass_1"
    assert matches[1].object_id == "glass_2"


def test_unknown_type_returns_no_matches() -> None:
    grounder = WorldGrounder()

    matches = grounder.find_by_type(
        make_scene(),
        "fork",
    )

    assert matches == ()


def test_match_contains_world_position() -> None:
    grounder = WorldGrounder()

    matches = grounder.find_by_type(
        make_scene(),
        "plate",
    )

    assert isinstance(
        matches[0],
        GroundingMatch,
    )

    assert matches[0].position == (
        -0.28,
        0.0,
        0.772,
    )