"""Task representations for TABLEMIND planning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ActionType(str, Enum):
    """Supported primitive robot actions."""

    APPROACH = "approach"
    OPEN_GRIPPER = "open_gripper"
    CLOSE_GRIPPER = "close_gripper"
    LIFT = "lift"
    MOVE = "move"
    PLACE = "place"


@dataclass(frozen=True)
class Task:
    """High-level task description."""

    task_id: str
    description: str


@dataclass(frozen=True)
class Action:
    """One primitive action in a robot plan."""

    action_id: str
    action_type: ActionType
    arm: str
    object_id: str | None = None
    target_position: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class TaskPlan:
    """Ordered sequence of actions for completing a task."""

    task: Task
    actions: tuple[Action, ...]

    @property
    def action_count(self) -> int:
        """Return the number of primitive actions."""

        return len(self.actions)