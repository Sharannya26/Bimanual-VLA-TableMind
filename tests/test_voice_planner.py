import pytest

from tablemind.conversation.voice_planner import VoicePlannerPipeline
from tablemind.perception.scene import SceneObservation, SceneObject


def test_voice_planner_converts_command_into_task_plan():
    scene = SceneObservation(
        objects=(
            SceneObject(
                object_id="plate_1",
                object_type="plate",
                position=(-0.28, 0.0, 0.842),
            ),
            SceneObject(
                object_id="glass_1",
                object_type="glass",
                position=(0.40, -0.14, 0.835),
            ),
        )
    )

    pipeline = VoicePlannerPipeline()

    request, plan = pipeline.process(
        "Set the table for one.",
        scene,
    )

    assert request.intent == "set_table"
    assert request.quantity == 1
    assert request.object_types == ("plate", "glass")

    assert plan.task.task_id == "set_table"
    assert plan.task.description == "Set the table for one."

    assert len(plan.actions) > 0


def test_voice_planner_rejects_empty_transcript():
    scene = SceneObservation(objects=())

    pipeline = VoicePlannerPipeline()

    with pytest.raises(ValueError, match="cannot be empty"):
        pipeline.process("", scene)