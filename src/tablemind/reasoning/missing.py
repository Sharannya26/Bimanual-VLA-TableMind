"""Missing-object analysis for TABLEMIND."""

from __future__ import annotations

from dataclasses import dataclass

from tablemind.perception.scene import SceneObservation
from tablemind.reasoning.requirements import TaskRequirements


@dataclass(frozen=True)
class MissingObjects:
    """Describe missing objects for one object category."""

    object_type: str
    required_quantity: int
    available_quantity: int
    missing_quantity: int


class MissingObjectAnalyzer:
    """Compare task requirements against the current scene."""

    def analyze(
        self,
        requirements: TaskRequirements,
        scene: SceneObservation,
    ) -> tuple[MissingObjects, ...]:
        """Calculate missing quantities for every required type."""

        results: list[MissingObjects] = []

        for requirement in requirements.requirements:
            available_quantity = len(
                scene.by_type(requirement.object_type)
            )

            missing_quantity = max(
                requirement.required_quantity
                - available_quantity,
                0,
            )

            results.append(
                MissingObjects(
                    object_type=requirement.object_type,
                    required_quantity=requirement.required_quantity,
                    available_quantity=available_quantity,
                    missing_quantity=missing_quantity,
                )
            )

        return tuple(results)