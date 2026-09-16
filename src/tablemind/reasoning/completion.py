"""Completion analysis for context-aware TABLEMIND planning."""

from __future__ import annotations

from dataclasses import dataclass

from tablemind.perception.scene import SceneObservation


@dataclass(frozen=True)
class CompletionStatus:
    """Describe which objects are already at their intended positions."""

    object_id: str
    object_type: str
    current_position: tuple[float, float, float]
    target_position: tuple[float, float, float]
    position_error: float

    @property
    def is_complete(self) -> bool:
        """Return True when the object is close enough to its target."""
        return self.position_error <= 0.06


class CompletionAnalyzer:
    """Determine whether scene objects are already correctly placed."""

    TOLERANCE = 0.06

    def analyze(
        self,
        scene: SceneObservation,
    ) -> tuple[CompletionStatus, ...]:
        """Evaluate every relevant object in the current scene."""

        results: list[CompletionStatus] = []

        for obj in scene.objects:
            target = self._target_position(obj.object_type, obj.position)

            error = self._distance(
                obj.position,
                target,
            )

            results.append(
                CompletionStatus(
                    object_id=obj.object_id,
                    object_type=obj.object_type,
                    current_position=obj.position,
                    target_position=target,
                    position_error=error,
                )
            )

        return tuple(results)

    @staticmethod
    def _target_position(
        object_type: str,
        current_position: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Return the expected placement height for an object."""

        x, y, z = current_position

        if object_type == "plate":
            return (x, y, 0.772)

        if object_type == "glass":
            return (x, y, 0.835)

        return current_position

    @staticmethod
    def _distance(
        first: tuple[float, float, float],
        second: tuple[float, float, float],
    ) -> float:
        """Calculate Euclidean distance between two positions."""

        return (
            (first[0] - second[0]) ** 2
            + (first[1] - second[1]) ** 2
            + (first[2] - second[2]) ** 2
        ) ** 0.5