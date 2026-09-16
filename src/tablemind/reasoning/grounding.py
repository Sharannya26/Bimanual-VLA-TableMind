"""World-state grounding utilities for TABLEMIND."""

from __future__ import annotations

from dataclasses import dataclass

from tablemind.perception.scene import (
    SceneObject,
    SceneObservation,
)


@dataclass(frozen=True)
class GroundingMatch:
    """A language reference matched to a scene object."""

    reference: str
    object_id: str
    object_type: str
    position: tuple[float, float, float]


class WorldGrounder:
    """Ground language object references in the current scene."""

    def find_by_type(
        self,
        scene: SceneObservation,
        object_type: str,
    ) -> tuple[GroundingMatch, ...]:
        """Return all scene objects matching an object type."""

        matches: list[GroundingMatch] = []

        for obj in scene.by_type(object_type):
            matches.append(
                GroundingMatch(
                    reference=object_type,
                    object_id=obj.object_id,
                    object_type=obj.object_type,
                    position=obj.position,
                )
            )

        return tuple(matches)