"""Integrated context reasoning for TABLEMIND."""

from __future__ import annotations

from dataclasses import dataclass

from tablemind.perception.scene import SceneObservation
from tablemind.reasoning.completion import (
    CompletionStatus,
    CompletionAnalyzer,
)
from tablemind.reasoning.missing import (
    MissingObjectAnalyzer,
    MissingObjects,
)
from tablemind.reasoning.requirements import (
    TaskRequirementBuilder,
    TaskRequirements,
)
from tablemind.reasoning.task_request import (
    TaskRequest,
)


@dataclass(frozen=True)
class ContextReasoningResult:
    """Complete reasoning state for a task and current scene."""

    request: TaskRequest
    requirements: TaskRequirements
    completion: tuple[CompletionStatus, ...]
    missing: tuple[MissingObjects, ...]

    @property
    def completed_objects(self) -> tuple[str, ...]:
        """Return IDs of objects already in their completed state."""

        return tuple(
            status.object_id
            for status in self.completion
            if status.is_complete
        )

    @property
    def incomplete_objects(self) -> tuple[str, ...]:
        """Return IDs of objects that still need manipulation."""

        return tuple(
            status.object_id
            for status in self.completion
            if not status.is_complete
        )

    @property
    def total_missing(self) -> int:
        """Return the total number of missing objects."""

        return sum(
            item.missing_quantity
            for item in self.missing
        )


class ContextReasoner:
    """Combine task requirements and scene analysis."""

    def __init__(
        self,
        requirement_builder: TaskRequirementBuilder | None = None,
        completion_analyzer: CompletionAnalyzer | None = None,
        missing_analyzer: MissingObjectAnalyzer | None = None,
    ) -> None:
        self.requirement_builder = (
            requirement_builder or TaskRequirementBuilder()
        )
        self.completion_analyzer = (
            completion_analyzer or CompletionAnalyzer()
        )
        self.missing_analyzer = (
            missing_analyzer or MissingObjectAnalyzer()
        )

    def reason(
        self,
        request: TaskRequest,
        scene: SceneObservation,
    ) -> ContextReasoningResult:
        """Produce a complete context reasoning result."""

        requirements = self.requirement_builder.build(
            request
        )

        completion = self.completion_analyzer.analyze(
            scene
        )

        missing = self.missing_analyzer.analyze(
            requirements,
            scene,
        )

        return ContextReasoningResult(
            request=request,
            requirements=requirements,
            completion=completion,
            missing=missing,
        )