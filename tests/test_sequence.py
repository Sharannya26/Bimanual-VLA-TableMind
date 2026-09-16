from tablemind.reasoning.decomposition import (
    DecomposedTask,
    TaskStep,
    TaskStepType,
)
from tablemind.reasoning.sequence import (
    ExecutionStage,
    TaskSequence,
    TaskSequencePlanner,
)


def make_step(
    step_type: TaskStepType,
    object_id: str,
    arm: str,
) -> TaskStep:
    return TaskStep(
        step_type=step_type,
        object_id=object_id,
        object_type="plate",
        arm=arm,
    )


def test_empty_task_creates_empty_sequence():
    planner = TaskSequencePlanner()

    task = DecomposedTask(
        name="set_table",
        steps=(),
    )

    sequence = planner.create_sequence(task)

    assert sequence.task_name == "set_table"
    assert sequence.stage_count == 0
    assert sequence.step_count == 0


def test_pick_and_place_are_separate_stages():
    planner = TaskSequencePlanner()

    task = DecomposedTask(
        name="set_table",
        steps=(
            make_step(TaskStepType.PICK, "plate_1", "left"),
            make_step(TaskStepType.PLACE, "plate_1", "left"),
        ),
    )

    sequence = planner.create_sequence(task)

    assert sequence.stage_count == 2
    assert sequence.stages[0].steps[0].step_type == TaskStepType.PICK
    assert sequence.stages[1].steps[0].step_type == TaskStepType.PLACE


def test_bimanual_pick_steps_share_one_stage():
    planner = TaskSequencePlanner()

    task = DecomposedTask(
        name="set_table",
        steps=(
            make_step(TaskStepType.PICK, "plate_1", "left"),
            make_step(TaskStepType.PLACE, "plate_1", "left"),
            make_step(TaskStepType.PICK, "plate_2", "right"),
            make_step(TaskStepType.PLACE, "plate_2", "right"),
        ),
    )

    sequence = planner.create_sequence(task)

    pick_stage = sequence.stages[0]

    assert pick_stage.step_count == 2
    assert pick_stage.arms == ("left", "right")


def test_bimanual_place_steps_share_one_stage():
    planner = TaskSequencePlanner()

    task = DecomposedTask(
        name="set_table",
        steps=(
            make_step(TaskStepType.PICK, "plate_1", "left"),
            make_step(TaskStepType.PLACE, "plate_1", "left"),
            make_step(TaskStepType.PICK, "plate_2", "right"),
            make_step(TaskStepType.PLACE, "plate_2", "right"),
        ),
    )

    sequence = planner.create_sequence(task)

    place_stage = sequence.stages[1]

    assert place_stage.step_count == 2
    assert place_stage.arms == ("left", "right")


def test_stage_indices_are_sequential():
    planner = TaskSequencePlanner()

    task = DecomposedTask(
        name="set_table",
        steps=(
            make_step(TaskStepType.PICK, "plate_1", "left"),
            make_step(TaskStepType.PLACE, "plate_1", "left"),
        ),
    )

    sequence = planner.create_sequence(task)

    assert [stage.index for stage in sequence.stages] == [1, 2]


def test_sequence_counts_all_steps():
    planner = TaskSequencePlanner()

    task = DecomposedTask(
        name="set_table",
        steps=(
            make_step(TaskStepType.PICK, "plate_1", "left"),
            make_step(TaskStepType.PLACE, "plate_1", "left"),
            make_step(TaskStepType.PICK, "plate_2", "right"),
            make_step(TaskStepType.PLACE, "plate_2", "right"),
        ),
    )

    sequence = planner.create_sequence(task)

    assert sequence.step_count == 4