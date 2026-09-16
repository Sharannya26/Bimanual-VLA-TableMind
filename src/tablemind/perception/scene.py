"""Scene representation used by TABLEMIND perception."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SceneObject:
    """A detected or observed object in the simulation."""

    object_id: str
    object_type: str
    position: Tuple[float, float, float]


@dataclass(frozen=True)
class SceneObservation:
    """A structured observation of the current table scene."""

    objects: tuple[SceneObject, ...]

    def by_type(self, object_type: str) -> tuple[SceneObject, ...]:
        """Return all objects matching a given type."""

        return tuple(
            obj
            for obj in self.objects
            if obj.object_type == object_type
        )

    def get(self, object_id: str) -> SceneObject | None:
        """Return an object by ID, or None if it is not present."""

        for obj in self.objects:
            if obj.object_id == object_id:
                return obj

        return None