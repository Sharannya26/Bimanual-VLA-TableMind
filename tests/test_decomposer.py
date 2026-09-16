from tablemind.reasoning.decomposer import TaskDecomposer
from tablemind.reasoning.decomposition import TaskStepType
from tablemind.reasoning.grounding import GroundingMatch


def make_object(
    object_id: str,
    object_type: str,
    position: tuple[float, float, float],
) -> GroundingMatch:
    return GroundingMatch(
        reference=object_type,
        object_id=object_id,
        object_type=object_type,
        position=position,
    )


def test_empty_scene_creates_empty_task():
    decomposer = TaskDecomposer()

    task = decomposer.decompose_set_table(())

    assert task.name == "set_table"
    assert task.steps == ()


def test_plate_gets_pick_and_place_steps():
    decomposer = TaskDecomposer()

    objects = (
        make_object(
            "plate_1",
            "plate",
            (-0.28, 0.0, 0.772),
        ),
    )

    task = decomposer.decompose_set_table(objects)

    assert task.step_count == 2
    assert task.steps[0].step_type == TaskStepType.PICK
    assert task.steps[1].step_type == TaskStepType.PLACE


def test_left_object_is_assigned_to_left_arm():
    decomposer = TaskDecomposer()

    objects = (
        make_object(
            "plate_1",
            "plate",
            (-0.28, 0.0, 0.772),
        ),
    )

    task = decomposer.decompose_set_table(objects)

    assert task.steps[0].arm == "left"
    assert task.steps[1].arm == "left"


def test_right_object_is_assigned_to_right_arm():
    decomposer = TaskDecomposer()

    objects = (
        make_object(
            "plate_2",
            "plate",
            (0.28, 0.0, 0.772),
        ),
    )

    task = decomposer.decompose_set_table(objects)

    assert task.steps[0].arm == "right"
    assert task.steps[1].arm == "right"


def test_multiple_objects_are_decomposed():
    decomposer = TaskDecomposer()

    objects = (
        make_object(
            "plate_1",
            "plate",
            (-0.28, 0.0, 0.772),
        ),
        make_object(
            "plate_2",
            "plate",
            (0.28, 0.0, 0.772),
        ),
    )

    task = decomposer.decompose_set_table(objects)

    assert task.step_count == 4
    assert task.objects == ("plate_1", "plate_2")
    assert task.arms == ("left", "right")


def test_plate_place_target_uses_table_height():
    decomposer = TaskDecomposer()

    objects = (
        make_object(
            "plate_1",
            "plate",
            (-0.28, 0.0, 0.900),
        ),
    )

    task = decomposer.decompose_set_table(objects)

    place_step = task.steps[1]

    assert place_step.target_position == (-0.28, 0.0, 0.772)