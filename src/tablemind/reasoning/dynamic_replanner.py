"""Dynamic replanning for TABLEMIND."""

from __future__ import annotations

from dataclasses import dataclass

from tablemind.perception.scene import SceneObservation
from tablemind.planning.task import TaskPlan
from tablemind.reasoning.context_planner import ContextAwarePlanner
from tablemind.reasoning.context_reasoner import (
    ContextReasoner,
    ContextReasoningResult,
)
from tablemind.reasoning.task_request import TaskRequest


@dataclass(frozen=True)
class ReplanningResult:
    """Result of reevaluating a task after the world changes."""

    reasoning: ContextReasoningResult
    plan: TaskPlan

    @property
    def needs_action(self) -> bool:
        """Return True when the updated scene still needs manipulation."""

        return bool(self.plan.actions)

    @property
    def remaining_objects(self) -> tuple[str, ...]:
        """Return object IDs that still require manipulation."""

        return tuple(
            self.reasoning.incomplete_objects
        )


class DynamicReplanner:
    """Replan a task from the latest observed world state."""

    def __init__(
        self,
        context_reasoner: ContextReasoner | None = None,
        planner: ContextAwarePlanner | None = None,
    ) -> None:
        self.context_reasoner = (
            context_reasoner or ContextReasoner()
        )

        self.planner = (
            planner or ContextAwarePlanner(
                context_reasoner=self.context_reasoner,
            )
        )

    def replan(
        self,
        request: TaskRequest,
        scene: SceneObservation,
    ) -> ReplanningResult:
        """Re-evaluate the scene and generate an updated plan."""

        reasoning = self.context_reasoner.reason(
            request,
            scene,
        )

        plan = self.planner.plan_from_reasoning(
            reasoning,
        )

        return ReplanningResult(
            reasoning=reasoning,
            plan=plan,
        )