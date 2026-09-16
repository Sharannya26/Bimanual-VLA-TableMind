"""Execution control signals for interruptible TABLEMIND tasks."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event


@dataclass
class ExecutionControl:
    """Thread-safe control state for a running robot task."""

    def __post_init__(self) -> None:
        self._stop_event = Event()

    def request_stop(self) -> None:
        """Request that the current execution stop as soon as safely possible."""
        self._stop_event.set()

    def clear(self) -> None:
        """Clear a previous stop request."""
        self._stop_event.clear()

    @property
    def stop_requested(self) -> bool:
        """Return whether a stop has been requested."""
        return self._stop_event.is_set()