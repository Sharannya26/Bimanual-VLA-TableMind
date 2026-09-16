from tablemind.conversation.voice_planner import VoicePlannerPipeline
from tablemind.reasoning.execution_bridge import ReasoningExecutionBridge
from tablemind.perception.scene import SceneObservation, SceneObject


def test_voice_plan_can_be_converted_into_bimanual_task():
    scene = SceneObservation(
        objects=(
            SceneObject(
                object_id="plate_1",
                object_type="plate",
                position=(-0.28, 0.0, 0.842),
            ),
            SceneObject(
                object_id="plate_2",
                object_type="plate",
                position=(0.28, 0.0, 0.842),
            ),
        )
    )

    voice_planner = VoicePlannerPipeline()

    request, plan = voice_planner.process(
        "Set the table for two.",
        scene,
    )

    bridge = ReasoningExecutionBridge()

    result = bridge.build_task(plan)
    task = result.task

    assert request.intent == "set_table"
    assert request.quantity == 2

    assert len(task.assignments) == 2

    assert task.assignments[0].object_id == "plate_1"
    assert task.assignments[1].object_id == "plate_2"

    assert task.assignments[0].arm == "left"
    assert task.assignments[1].arm == "right"

    assert task.assignments[0].approach_position is not None
    assert task.assignments[0].grasp_position is not None
    assert task.assignments[0].lift_position is not None
    assert task.assignments[0].place_position is not None