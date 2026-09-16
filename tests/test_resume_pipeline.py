from tablemind.conversation.resume_pipeline import (
    ConversationalResumePipeline,
)
from tablemind.reasoning.interpreter import TaskInterpreter
from tablemind.perception.scene import (
    SceneObservation,
    SceneObject,
)
from tablemind.simulation import BimanualTableSimulation


def test_resume_pipeline_replans_modified_task():
    simulation = BimanualTableSimulation.create()

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

    interpreter = TaskInterpreter()

    current_request = interpreter.interpret(
        "Set the table for two."
    )

    pipeline = ConversationalResumePipeline(
        simulation,
    )

    result = pipeline.resume_with_modification(
        current_request,
        "Actually, make it three.",
        scene,
    )

    assert result.modification.quantity == 3
    assert result.updated_request.quantity == 3

    assert result.replanning is not None
    assert result.replanning.needs_action is True