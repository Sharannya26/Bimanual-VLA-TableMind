from tablemind.conversation.modification import (
    ModificationRequest,
)
from tablemind.conversation.modification_replanner import (
    ModificationReplanner,
)
from tablemind.reasoning.interpreter import TaskInterpreter
from tablemind.perception.scene import (
    SceneObject,
    SceneObservation,
)


def test_modification_replanner_updates_quantity_and_replans():
    interpreter = TaskInterpreter()

    current_request = interpreter.interpret(
        "Set the table for two."
    )

    modification = ModificationRequest(
        raw_instruction="Actually, make it three.",
        intent="set_table",
        quantity=3,
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
                object_id="plate_3",
                object_type="plate",
                position=(0.0, 0.0, 0.842),
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
            SceneObject(
                object_id="glass_3",
                object_type="glass",
                position=(0.0, -0.14, 0.910),
            ),
        )
    )

    replanner = ModificationReplanner()

    result = replanner.replan(
        current_request,
        modification,
        scene,
    )

    assert result.updated_request.quantity == 3

    assert result.replanning.reasoning.requirements.requirement_for(
        "plate"
    ).required_quantity == 3

    assert result.replanning.reasoning.requirements.requirement_for(
        "glass"
    ).required_quantity == 3

    assert result.needs_action is True


def test_modification_replanner_returns_remaining_objects():
    interpreter = TaskInterpreter()

    current_request = interpreter.interpret(
        "Set the table for two."
    )

    modification = ModificationRequest(
        raw_instruction="Make it three.",
        intent="set_table",
        quantity=3,
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
                object_id="plate_3",
                object_type="plate",
                position=(0.0, 0.0, 0.842),
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
            SceneObject(
                object_id="glass_3",
                object_type="glass",
                position=(0.0, -0.14, 0.910),
            ),
        )
    )

    replanner = ModificationReplanner()

    result = replanner.replan(
        current_request,
        modification,
        scene,
    )

    assert "plate_3" in result.remaining_objects
    assert "glass_3" in result.remaining_objects