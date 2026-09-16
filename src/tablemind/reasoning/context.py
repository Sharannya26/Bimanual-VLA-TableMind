"""Context-aware task reasoning for TABLEMIND."""

from __future__ import annotations

from dataclasses import dataclass

from tablemind.perception.scene import SceneObservation
from tablemind.reasoning.task_request import TaskRequest


@dataclass(frozen=True)
class TaskContext:
    """Describe the requested task relative to the current scene."""

    request: TaskRequest
    available_objects: tuple[str, ...]
    required_quantity: int
    missing_quantity: int

    @property
    def is_satisfied(self) -> bool:
        """Return True when the current scene satisfies the request."""
        return self.missing_quantity <= 0


class ContextAnalyzer:
    """Analyze a task request against the current scene."""

    def analyze(
        self,
        request: TaskRequest,
        scene: SceneObservation,
    ) -> TaskContext:
        """Build a context describing what the scene currently provides."""

        relevant_objects: list[str] = []

        for object_type in request.object_types:
            relevant_objects.extend(
                obj.object_id
                for obj in scene.by_type(object_type)
            )

        available_objects = tuple(relevant_objects)

        required_quantity = request.quantity or 0
        missing_quantity = max(
            required_quantity - len(available_objects),
            0,
        )

        return TaskContext(
            request=request,
            available_objects=available_objects,
            required_quantity=required_quantity,
            missing_quantity=missing_quantity,
        )