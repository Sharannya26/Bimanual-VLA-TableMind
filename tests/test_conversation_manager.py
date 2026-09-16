from tablemind.conversation.manager import (
    ConversationAction,
    ConversationManager,
    ConversationState,
)


def test_new_task_starts_execution():
    manager = ConversationManager()

    turn = manager.process(
        "Set the table for two."
    )

    assert turn.action == ConversationAction.NEW_TASK
    assert turn.state == ConversationState.EXECUTING
    assert turn.task_request is not None
    assert turn.task_request.quantity == 2


def test_stop_interrupts_current_task():
    manager = ConversationManager()

    manager.process(
        "Set the table for two."
    )

    turn = manager.process("Actually, stop.")

    assert turn.action == ConversationAction.STOP
    assert turn.state == ConversationState.INTERRUPTED


def test_modification_preserves_current_task():
    manager = ConversationManager()

    manager.process(
        "Set the table for two."
    )

    turn = manager.process(
        "Actually, make it three."
    )

    assert turn.action == ConversationAction.MODIFY
    assert turn.task_request is not None
    assert turn.task_request.quantity == 2


def test_resume_returns_to_execution():
    manager = ConversationManager()

    manager.process(
        "Set the table for two."
    )

    manager.process("stop")

    turn = manager.process("continue")

    assert turn.action == ConversationAction.RESUME
    assert turn.state == ConversationState.EXECUTING


def test_completion_state():
    manager = ConversationManager()

    manager.process(
        "Set the table for two."
    )

    manager.mark_completed()

    assert manager.state == ConversationState.COMPLETED


def test_reset_returns_to_idle():
    manager = ConversationManager()

    manager.process(
        "Set the table for two."
    )

    manager.reset()

    assert manager.state == ConversationState.IDLE
    assert manager.current_request is None