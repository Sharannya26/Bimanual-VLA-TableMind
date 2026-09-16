"""Task decomposition utilities for TABLEMIND."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TaskStepType(str, Enum):
    """Types of high-level manipulation steps."""

    PICK = "pick"
    PLACE = "place"


@dataclass(frozen=True)
class TaskStep:
    """A single high-level manipulation step."""

    step_type: TaskStepType
    object_id: str
    object_type: str
    arm: str
    target_position: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class DecomposedTask:
    """A high-level task broken into executable steps."""

    name: str
    steps: tuple[TaskStep, ...]

    @property
    def step_count(self) -> int:
        """Return the number of steps in the task."""

        return len(self.steps)

    @property
    def arms(self) -> tuple[str, ...]:
        """Return the unique arms involved in the task."""

        return tuple(dict.fromkeys(step.arm for step in self.steps))

    @property
    def objects(self) -> tuple[str, ...]:
        """Return the unique objects involved in the task."""

        return tuple(dict.fromkeys(step.object_id for step in self.steps))