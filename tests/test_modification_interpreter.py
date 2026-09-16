from tablemind.conversation.modification import (
    ModificationRequest,
)
from tablemind.conversation.modification_interpreter import (
    ModificationInterpreter,
)


def test_actual_make_it_three_is_detected():
    interpreter = ModificationInterpreter()

    result = interpreter.interpret(
        "Actually, make it three."
    )

    assert isinstance(
        result,
        ModificationRequest,
    )

    assert result.intent == "set_table"
    assert result.quantity == 3
    assert result.is_modification is True


def test_change_to_four_is_detected():
    interpreter = ModificationInterpreter()

    result = interpreter.interpret(
        "Change that to four."
    )

    assert result.intent == "set_table"
    assert result.quantity == 4
    assert result.is_modification is True


def test_make_it_two_is_detected():
    interpreter = ModificationInterpreter()

    result = interpreter.interpret(
        "Make it two."
    )

    assert result.intent == "set_table"
    assert result.quantity == 2


def test_numeric_quantity_is_detected():
    interpreter = ModificationInterpreter()

    result = interpreter.interpret(
        "Actually, make it 3."
    )

    assert result.quantity == 3


def test_non_modification_is_rejected():
    interpreter = ModificationInterpreter()

    try:
        interpreter.interpret(
            "Set the table for two."
        )
    except ValueError as exc:
        assert "modification" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected non-modification instruction to be rejected."
        )


def test_empty_instruction_is_rejected():
    interpreter = ModificationInterpreter()

    try:
        interpreter.interpret("   ")
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError(
            "Expected empty instruction to be rejected."
        )