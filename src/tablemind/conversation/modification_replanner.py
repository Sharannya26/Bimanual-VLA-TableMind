"""Conversational modification and replanning for TABLEMIND."""

from __future__ import annotations

from dataclasses import dataclass

from tablemind.conversation.modification import ModificationRequest
from tablemind.conversation.task_updater import TaskUpdateManager
from tablemind.perception.scene import SceneObservation
from tablemind.planning.task import TaskPlan
from tablemind.reasoning.dynamic_replanner import (
    DynamicReplanner,
    ReplanningResult,
)
from tablemind.reasoning.task_request import TaskRequest


@dataclass(frozen=True)
class ModificationReplanningResult:
    """Result of applying a modification and replanning."""

    updated_request: TaskRequest
    replanning: ReplanningResult

    @property
    def plan(self) -> TaskPlan:
        """Return the updated task plan."""
        return self.replanning.plan

    @property
    def remaining_objects(self) -> tuple[str, ...]:
        """Return objects that still require manipulation."""
        return self.replanning.remaining_objects

    @property
    def needs_action(self) -> bool:
        """Return whether the updated task still requires action."""
        return self.replanning.needs_action


class ModificationReplanner:
    """Apply a conversational modification and replan from the latest scene."""

    def __init__(
        self,
        task_updater: TaskUpdateManager | None = None,
        replanner: DynamicReplanner | None = None,
    ) -> None:
        self.task_updater = (
            task_updater or TaskUpdateManager()
        )

        self.replanner = (
            replanner or DynamicReplanner()
        )

    def replan(
        self,
        current_request: TaskRequest,
        modification: ModificationRequest,
        scene: SceneObservation,
    ) -> ModificationReplanningResult:
        """Apply a modification and replan from the latest scene."""

        updated_request = self.task_updater.update(
            current_request,
            modification,
        )

        replanning = self.replanner.replan(
            updated_request,
            scene,
        )

        return ModificationReplanningResult(
            updated_request=updated_request,
            replanning=replanning,
        )