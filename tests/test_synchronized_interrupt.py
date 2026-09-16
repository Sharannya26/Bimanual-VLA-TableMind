from tablemind.manipulation.execution_control import ExecutionControl


def test_execution_control_can_request_and_clear_stop():
    control = ExecutionControl()

    assert control.stop_requested is False

    control.request_stop()

    assert control.stop_requested is True

    control.clear()

    assert control.stop_requested is False