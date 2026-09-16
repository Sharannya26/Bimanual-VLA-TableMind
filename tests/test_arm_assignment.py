from tablemind.perception.scene import (
    SceneObject,
    SceneObservation,
)
from tablemind.perception.state import (
    RobotState,
    SceneState,
)
from tablemind.reasoning.arm_assignment import (
    ContextAwareArmAssigner,
)
from tablemind.reasoning.grounding import (
    GroundingMatch,
)


def make_scene() -> SceneState:
    objects = SceneObservation(
        objects=(
            SceneObject(
                object_id="plate_1",
                object_type="plate",
                position=(-0.30, 0.0, 0.772),
            ),
            SceneObject(
                object_id="plate_2",
                object_type="plate",
                position=(0.30, 0.0, 0.772),
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


def make_grounding_matches():
    return (
        GroundingMatch(
            reference="plate",
            object_id="plate_1",
            object_type="plate",
            position=(-0.30, 0.0, 0.772),
        ),
        GroundingMatch(
            reference="plate",
            object_id="plate_2",
            object_type="plate",
            position=(0.30, 0.0, 0.772),
        ),
    )


def test_assigns_left_object_to_left_arm():
    scene = make_scene()

    assignments = ContextAwareArmAssigner().assign(
        (make_grounding_matches()[0],),
        scene,
    )

    assert len(assignments) == 1
    assert assignments[0].object_id == "plate_1"
    assert assignments[0].arm == "left"


def test_assigns_right_object_to_right_arm():
    scene = make_scene()

    assignments = ContextAwareArmAssigner().assign(
        (make_grounding_matches()[1],),
        scene,
    )

    assert len(assignments) == 1
    assert assignments[0].object_id == "plate_2"
    assert assignments[0].arm == "right"


def test_returns_distance():
    scene = make_scene()

    assignments = ContextAwareArmAssigner().assign(
        (make_grounding_matches()[0],),
        scene,
    )

    expected_distance = (
        (
            (-0.30 - (-0.20)) ** 2
            + (0.0 - 0.0) ** 2
            + (0.772 - 0.90) ** 2
        )
        ** 0.5
    )

    assert abs(
        assignments[0].distance - expected_distance
    ) < 1e-9


def test_assigns_multiple_objects():
    scene = make_scene()

    assignments = ContextAwareArmAssigner().assign(
        make_grounding_matches(),
        scene,
    )

    assert len(assignments) == 2

    assert assignments[0].arm == "left"
    assert assignments[1].arm == "right"


def test_requires_both_arms():
    objects = SceneObservation(objects=())

    scene = SceneState(
        objects=objects,
        robots=(
            RobotState(
                arm_id="left",
                end_effector_position=(-0.20, 0.0, 0.90),
                gripper_position=1.0,
            ),
        ),
        simulation_time=0.0,
    )

    grounded = make_grounding_matches()

    try:
        ContextAwareArmAssigner().assign(
            (grounded[0],),
            scene,
        )
    except RuntimeError as exc:
        assert "Both left and right" in str(exc)
    else:
        raise AssertionError(
            "Expected RuntimeError when one arm is missing."
        )