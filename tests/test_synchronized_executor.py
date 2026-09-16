from tablemind.manipulation.coordination import (
    BimanualAssignment,
)
from tablemind.manipulation.synchronized_executor import (
    SynchronizedBimanualExecutor,
)


def test_planner_targets_are_used_for_approach():
    assignment = BimanualAssignment(
        arm="left",
        object_id="plate_1",
        approach_position=(-0.31, -0.02, 0.91),
    )

    target = SynchronizedBimanualExecutor._approach_target(
        assignment,
        (-0.28, 0.0, 0.772),
    )

    assert target == (-0.31, -0.02, 0.91)


def test_planner_targets_are_used_for_grasp():
    assignment = BimanualAssignment(
        arm="left",
        object_id="plate_1",
        grasp_position=(-0.30, 0.01, 0.78),
    )

    target = SynchronizedBimanualExecutor._grasp_target(
        assignment,
        (-0.28, 0.0, 0.772),
    )

    assert target == (-0.30, 0.01, 0.78)


def test_planner_targets_are_used_for_lift():
    assignment = BimanualAssignment(
        arm="left",
        object_id="plate_1",
        lift_position=(-0.25, 0.02, 0.96),
    )

    target = SynchronizedBimanualExecutor._lift_target(
        assignment,
        (-0.28, 0.0, 0.772),
    )

    assert target == (-0.25, 0.02, 0.96)


def test_planner_place_target_controls_transport():
    assignment = BimanualAssignment(
        arm="left",
        object_id="plate_1",
        place_position=(-0.19, 0.04, 0.78),
    )

    target = SynchronizedBimanualExecutor._transport_target(
        assignment,
        (-0.28, 0.0, 0.772),
    )

    assert target == (-0.19, 0.04, 0.93)


def test_planner_place_target_controls_lowering():
    assignment = BimanualAssignment(
        arm="right",
        object_id="plate_2",
        place_position=(0.21, 0.03, 0.78),
    )

    target = SynchronizedBimanualExecutor._lower_target(
        assignment,
        (0.28, 0.0, 0.772),
    )

    assert target == (0.21, 0.03, 0.78)


def test_legacy_fallbacks_still_work():
    assignment = BimanualAssignment(
        arm="left",
        object_id="plate_1",
    )

    initial = (-0.28, 0.0, 0.772)

    assert (
        SynchronizedBimanualExecutor._approach_target(
            assignment,
            initial,
        )
        == (-0.28, 0.0, 0.852)
    )

    assert (
        SynchronizedBimanualExecutor._grasp_target(
            assignment,
            initial,
        )
        == (-0.28, 0.0, 0.792)
    )

    assert (
        SynchronizedBimanualExecutor._lift_target(
            assignment,
            initial,
        )
        == (-0.28, 0.0, 0.922)
    )

    assert (
        SynchronizedBimanualExecutor._transport_target(
            assignment,
            initial,
        )
        == (-0.22, 0.0, 0.922)
    )

    assert (
        SynchronizedBimanualExecutor._lower_target(
            assignment,
            initial,
        )
        == (-0.22, 0.0, 0.792)
    )