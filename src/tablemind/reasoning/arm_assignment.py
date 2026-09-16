"""Context-aware arm assignment for TABLEMIND."""

from __future__ import annotations

from dataclasses import dataclass

from tablemind.perception.state import SceneState
from tablemind.reasoning.grounding import GroundingMatch


@dataclass(frozen=True)
class ArmAssignment:
    """Assignment of one object to the closest robot arm."""

    object_id: str
    arm: str
    distance: float


class ContextAwareArmAssigner:
    """Assign grounded objects to the closest available arm."""

    VALID_ARMS = ("left", "right")

    def assign(
        self,
        objects: tuple[GroundingMatch, ...],
        scene: SceneState,
    ) -> tuple[ArmAssignment, ...]:
        """Assign every object to the closest robot arm."""

        assignments: list[ArmAssignment] = []

        for obj in objects:
            arm, distance = self._closest_arm(
                object_position=obj.position,
                scene=scene,
            )

            assignments.append(
                ArmAssignment(
                    object_id=obj.object_id,
                    arm=arm,
                    distance=distance,
                )
            )

        return tuple(assignments)

    def _closest_arm(
        self,
        object_position: tuple[float, float, float],
        scene: SceneState,
    ) -> tuple[str, float]:
        """Return the closest arm and its Cartesian distance."""

        left = scene.robot("left")
        right = scene.robot("right")

        if left is None or right is None:
            raise RuntimeError(
                "Both left and right robot states are required."
            )

        left_distance = self._distance(
            object_position,
            left.end_effector_position,
        )

        right_distance = self._distance(
            object_position,
            right.end_effector_position,
        )

        if left_distance <= right_distance:
            return "left", left_distance

        return "right", right_distance

    @staticmethod
    def _distance(
        first: tuple[float, float, float],
        second: tuple[float, float, float],
    ) -> float:
        """Calculate Euclidean distance between two 3D points."""

        return (
            (
                (first[0] - second[0]) ** 2
                + (first[1] - second[1]) ** 2
                + (first[2] - second[2]) ** 2
            )
            ** 0.5
        )