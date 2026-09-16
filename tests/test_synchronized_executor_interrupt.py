from tablemind.manipulation.execution_control import ExecutionControl
from tablemind.manipulation.synchronized_executor import (
    SynchronizedExecutionResult,
)


def test_interrupted_result_is_reported_correctly():
    result = SynchronizedExecutionResult(
        success=False,
        completed_stages=3,
        total_stages=7,
        message="Synchronized task 'current task' was interrupted by the user.",
    )

    assert result.success is False
    assert result.completed_stages == 3
    assert result.total_stages == 7
    assert "interrupted" in result.message.lower()


def test_stop_request_can_be_shared_with_executor():
    control = ExecutionControl()

    assert control.stop_requested is False

    control.request_stop()

    assert control.stop_requested is True