from tablemind.reasoning import (
    TaskInterpreter,
    TaskRequest,
)


def test_set_table_numeric_quantity() -> None:
    interpreter = TaskInterpreter()

    task = interpreter.interpret(
        "Set the table for 2."
    )

    assert isinstance(task, TaskRequest)
    assert task.intent == "set_table"
    assert task.quantity == 2
    assert task.object_types == (
        "plate",
        "glass",
    )


def test_set_table_word_quantity() -> None:
    interpreter = TaskInterpreter()

    task = interpreter.interpret(
        "Set the table for four."
    )

    assert task.intent == "set_table"
    assert task.quantity == 4


def test_raw_instruction_is_preserved() -> None:
    interpreter = TaskInterpreter()

    instruction = "Set the table for two."

    task = interpreter.interpret(
        instruction
    )

    assert task.raw_instruction == instruction


def test_task_request_to_dict() -> None:
    task = TaskRequest(
        raw_instruction="Set the table for two.",
        intent="set_table",
        object_types=(
            "plate",
            "glass",
        ),
        quantity=2,
    )

    result = task.to_dict()

    assert result["intent"] == "set_table"
    assert result["quantity"] == 2
    assert result["object_types"] == [
        "plate",
        "glass",
    ]


def test_unknown_instruction_is_rejected() -> None:
    interpreter = TaskInterpreter()

    try:
        interpreter.interpret(
            "Dance with the robots."
        )
    except ValueError:
        return

    raise AssertionError(
        "Unsupported instruction should raise ValueError."
    )