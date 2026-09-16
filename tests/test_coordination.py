from tablemind.manipulation.coordination import (
    BimanualAssignment,
    BimanualCoordinator,
)


def test_create_two_arm_task() -> None:
    coordinator = BimanualCoordinator()

    task = coordinator.create_two_arm_task(
        name="set_table",
        left_object="plate_1",
        right_object="plate_2",
    )

    assert task.name == "set_table"
    assert task.arms == ("left", "right")
    assert task.objects == ("plate_1", "plate_2")


def test_assignment_lookup() -> None:
    coordinator = BimanualCoordinator()

    task = coordinator.create_two_arm_task(
        name="set_table",
        left_object="plate_1",
        right_object="plate_2",
    )

    assert task.assignment_for("left").object_id == "plate_1"
    assert task.assignment_for("right").object_id == "plate_2"


def test_duplicate_arm_rejected() -> None:
    coordinator = BimanualCoordinator()

    try:
        coordinator.create_task(
            name="invalid",
            assignments=(
                BimanualAssignment("left", "plate_1"),
                BimanualAssignment("left", "plate_2"),
            ),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate arm should be rejected")


def test_duplicate_object_rejected() -> None:
    coordinator = BimanualCoordinator()

    try:
        coordinator.create_task(
            name="invalid",
            assignments=(
                BimanualAssignment("left", "plate_1"),
                BimanualAssignment("right", "plate_1"),
            ),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Duplicate object should be rejected")