from tablemind.conversation.modification_pipeline import (
    ModificationPipeline,
)
from tablemind.reasoning.interpreter import TaskInterpreter
from tablemind.perception.scene import (
    SceneObservation,
    SceneObject,
)


def test_modification_pipeline_updates_quantity_and_replans():
    interpreter = TaskInterpreter()
    current_request = interpreter.interpret(
        "Set the table for two."
    )

    scene = SceneObservation(
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

    pipeline = ModificationPipeline()

    modification, updated_request, replanning = (
        pipeline.process(
            current_request,
            "Actually, make it three.",
            scene,
        )
    )

    assert modification.is_modification is True
    assert modification.quantity == 3

    assert updated_request.intent == "set_table"
    assert updated_request.quantity == 3

    assert replanning is not None