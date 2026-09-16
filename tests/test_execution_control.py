from tablemind.manipulation.execution_control import ExecutionControl


def test_execution_control_starts_without_stop_request():
    control = ExecutionControl()

    assert control.stop_requested is False


def test_execution_control_can_request_stop():
    control = ExecutionControl()

    control.request_stop()

    assert control.stop_requested is True


def test_execution_control_can_clear_stop_request():
    control = ExecutionControl()

    control.request_stop()
    control.clear()

    assert control.stop_requested is False