"""Task decomposition logic for TABLEMIND."""

from __future__ import annotations

from tablemind.reasoning.decomposition import (
    DecomposedTask,
    TaskStep,
    TaskStepType,
)
from tablemind.reasoning.grounding import GroundingMatch


class TaskDecomposer:
    """Convert grounded scene objects into manipulation steps."""

    VALID_ARMS = ("left", "right")

    def decompose_set_table(
        self,
        grounded_objects: tuple[GroundingMatch, ...],
    ) -> DecomposedTask:
        """Create pick/place steps for grounded table objects."""

        if not grounded_objects:
            return DecomposedTask(
                name="set_table",
                steps=(),
            )

        assignments = self._assign_arms(grounded_objects)

        steps: list[TaskStep] = []

        for obj in grounded_objects:
            arm = assignments[obj.object_id]

            steps.append(
                TaskStep(
                    step_type=TaskStepType.PICK,
                    object_id=obj.object_id,
                    object_type=obj.object_type,
                    arm=arm,
                )
            )

            steps.append(
                TaskStep(
                    step_type=TaskStepType.PLACE,
                    object_id=obj.object_id,
                    object_type=obj.object_type,
                    arm=arm,
                    target_position=self._placement_position(obj),
                )
            )

        return DecomposedTask(
            name="set_table",
            steps=tuple(steps),
        )

    def _assign_arms(
        self,
        objects: tuple[GroundingMatch, ...],
    ) -> dict[str, str]:
        """Assign each object to the nearest side of the robot workspace."""

        assignments: dict[str, str] = {}

        for obj in objects:
            x = obj.position[0]

            if x < 0:
                assignments[obj.object_id] = "left"
            else:
                assignments[obj.object_id] = "right"

        return assignments

    @staticmethod
    def _placement_position(
        obj: GroundingMatch,
    ) -> tuple[float, float, float]:
        """Return a safe table-level placement target."""

        x, y, z = obj.position

        if obj.object_type == "plate":
            return (x, y, 0.772)

        if obj.object_type == "glass":
            return (x, y, 0.835)

        return (x, y, z)