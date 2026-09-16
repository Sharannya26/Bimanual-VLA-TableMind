from tablemind.conversation.modification import (
    ModificationRequest,
)
from tablemind.conversation.task_updater import (
    TaskUpdateManager,
)
from tablemind.reasoning.interpreter import TaskInterpreter


def test_task_quantity_can_be_updated():
    interpreter = TaskInterpreter()

    current = interpreter.interpret(
        "Set the table for two."
    )

    modification = ModificationRequest(
        raw_instruction="Actually, make it three.",
        intent="set_table",
        quantity=3,
    )

    updater = TaskUpdateManager()

    updated = updater.update(
        current,
        modification,
    )

    assert updated.intent == "set_table"
    assert updated.quantity == 3
    assert updated.object_types == (
        "plate",
        "glass",
    )


def test_task_update_preserves_constraints():
    interpreter = TaskInterpreter()

    current = interpreter.interpret(
        "Set the table for two."
    )

    modification = ModificationRequest(
        raw_instruction="Change that to four.",
        intent="set_table",
        quantity=4,
    )

    updater = TaskUpdateManager()

    updated = updater.update(
        current,
        modification,
    )

    assert updated.quantity == 4
    assert updated.constraints == current.constraints


def test_task_update_preserves_object_types():
    interpreter = TaskInterpreter()

    current = interpreter.interpret(
        "Set the table for two."
    )

    modification = ModificationRequest(
        raw_instruction="Make it three.",
        intent="set_table",
        quantity=3,
    )

    updater = TaskUpdateManager()

    updated = updater.update(
        current,
        modification,
    )

    assert updated.object_types == current.object_types


def test_modification_without_quantity_keeps_current_quantity():
    interpreter = TaskInterpreter()

    current = interpreter.interpret(
        "Set the table for two."
    )

    modification = ModificationRequest(
        raw_instruction="Actually, change that.",
        intent="set_table",
        quantity=None,
    )

    updater = TaskUpdateManager()

    updated = updater.update(
        current,
        modification,
    )

    assert updated.quantity == 2


def test_non_modification_request_is_rejected():
    interpreter = TaskInterpreter()

    current = interpreter.interpret(
        "Set the table for two."
    )

    modification = ModificationRequest(
        raw_instruction="Set the table for three.",
        intent="set_table",
        quantity=3,
        is_modification=False,
    )

    updater = TaskUpdateManager()

    try:
        updater.update(
            current,
            modification,
        )
    except ValueError as exc:
        assert "modification" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected non-modification request to be rejected."
        )


def test_mismatched_intent_is_rejected():
    interpreter = TaskInterpreter()

    current = interpreter.interpret(
        "Set the table for two."
    )

    modification = ModificationRequest(
        raw_instruction="Change the task.",
        intent="different_task",
        quantity=3,
    )

    updater = TaskUpdateManager()

    try:
        updater.update(
            current,
            modification,
        )
    except ValueError as exc:
        assert "intent" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected mismatched intent to be rejected."
        )