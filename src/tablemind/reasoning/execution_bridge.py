"""Bridge reasoning plans into bimanual manipulation execution."""

from __future__ import annotations

from dataclasses import dataclass

from tablemind.manipulation.coordination import (
    BimanualAssignment,
    BimanualCoordinator,
    CoordinatedTask,
)
from tablemind.planning.task import ActionType, TaskPlan


@dataclass(frozen=True)
class ExecutionBridgeResult:
    """Result of converting a task plan into an executable task."""

    task: CoordinatedTask
    assignment_count: int


class ReasoningExecutionBridge:
    """Convert a reasoning TaskPlan into a target-aware CoordinatedTask."""

    EXECUTABLE_ACTIONS = (
        ActionType.APPROACH,
        ActionType.MOVE,
        ActionType.LIFT,
        ActionType.PLACE,
    )

    def __init__(
        self,
        coordinator: BimanualCoordinator | None = None,
    ) -> None:
        self.coordinator = coordinator or BimanualCoordinator()

    def build_task(
        self,
        plan: TaskPlan,
    ) -> ExecutionBridgeResult:
        """Build an executable coordinated task from a reasoning plan.

        Planner-generated target positions are preserved and passed
        into the manipulation layer.
        """

        assignments: dict[str, dict[str, object]] = {}

        for action in plan.actions:
            if action.action_type not in self.EXECUTABLE_ACTIONS:
                continue

            if action.arm is None:
                raise ValueError(
                    f"Action '{action.action_id}' has no arm assignment."
                )

            if action.object_id is None:
                raise ValueError(
                    f"Action '{action.action_id}' has no object assignment."
                )

            key = action.arm

            if key not in assignments:
                assignments[key] = {
                    "object_id": action.object_id,
                    "approach_position": None,
                    "grasp_position": None,
                    "lift_position": None,
                    "place_position": None,
                }

            current = assignments[key]

            if current["object_id"] != action.object_id:
                raise ValueError(
                    f"Arm '{action.arm}' is assigned to multiple objects."
                )

            if action.action_type == ActionType.APPROACH:
                current["approach_position"] = action.target_position

            elif action.action_type == ActionType.MOVE:
                current["grasp_position"] = action.target_position

            elif action.action_type == ActionType.LIFT:
                current["lift_position"] = action.target_position

            elif action.action_type == ActionType.PLACE:
                current["place_position"] = action.target_position

        if not assignments:
            raise ValueError(
                "Task plan contains no executable object assignments."
            )

        bimanual_assignments = tuple(
            BimanualAssignment(
                arm=arm,
                object_id=data["object_id"],
                approach_position=data["approach_position"],
                grasp_position=data["grasp_position"],
                lift_position=data["lift_position"],
                place_position=data["place_position"],
            )
            for arm, data in assignments.items()
        )

        task = self.coordinator.create_task(
            name=plan.task.task_id,
            assignments=bimanual_assignments,
        )

        return ExecutionBridgeResult(
            task=task,
            assignment_count=len(task.assignments),
        )