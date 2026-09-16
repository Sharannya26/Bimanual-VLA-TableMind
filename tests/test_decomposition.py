from tablemind.reasoning.decomposition import (
    DecomposedTask,
    TaskStep,
    TaskStepType,
)


def test_task_step_represents_pick():
    step = TaskStep(
        step_type=TaskStepType.PICK,
        object_id="plate_1",
        object_type="plate",
        arm="left",
    )

    assert step.step_type == TaskStepType.PICK
    assert step.object_id == "plate_1"
    assert step.object_type == "plate"
    assert step.arm == "left"


def test_task_step_can_store_target_position():
    step = TaskStep(
        step_type=TaskStepType.PLACE,
        object_id="plate_1",
        object_type="plate",
        arm="left",
        target_position=(-0.22, 0.0, 0.772),
    )

    assert step.target_position == (-0.22, 0.0, 0.772)


def test_decomposed_task_counts_steps():
    task = DecomposedTask(
        name="set_table",
        steps=(
            TaskStep(
                TaskStepType.PICK,
                "plate_1",
                "plate",
                "left",
            ),
            TaskStep(
                TaskStepType.PLACE,
                "plate_1",
                "plate",
                "left",
            ),
        ),
    )

    assert task.step_count == 2


def test_decomposed_task_reports_unique_arms():
    task = DecomposedTask(
        name="set_table",
        steps=(
            TaskStep(TaskStepType.PICK, "plate_1", "plate", "left"),
            TaskStep(TaskStepType.PLACE, "plate_1", "plate", "left"),
            TaskStep(TaskStepType.PICK, "plate_2", "plate", "right"),
        ),
    )

    assert task.arms == ("left", "right")


def test_decomposed_task_reports_unique_objects():
    task = DecomposedTask(
        name="set_table",
        steps=(
            TaskStep(TaskStepType.PICK, "plate_1", "plate", "left"),
            TaskStep(TaskStepType.PLACE, "plate_1", "plate", "left"),
            TaskStep(TaskStepType.PICK, "plate_2", "plate", "right"),
        ),
    )

    assert task.objects == ("plate_1", "plate_2")