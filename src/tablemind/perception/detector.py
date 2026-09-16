"""Object perception utilities for TABLEMIND."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco

from tablemind.perception.scene import SceneObject, SceneObservation
from tablemind.scene_config.table import GLASS_POSITIONS, PLATE_POSITIONS


@dataclass(frozen=True)
class DetectedObject:
    """Object detected by the perception system."""

    object_id: str
    object_type: str
    position: tuple[float, float, float]


class GroundTruthDetector:
    """
    Deterministic object detector using MuJoCo scene state.

    The backend is intentionally deterministic for the simulation
    environment. The rest of TABLEMIND interacts with the detector
    through the same scene-observation interface that a future
    RGB-based detector can use.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData | None = None,
    ) -> None:
        self.model = model
        self.data = data

    def detect(self) -> SceneObservation:
        """Detect all known table objects."""

        objects: list[SceneObject] = []

        objects.extend(
            self._detect_objects(
                prefix="plate",
                object_type="plate",
                positions=PLATE_POSITIONS,
            )
        )

        objects.extend(
            self._detect_objects(
                prefix="glass",
                object_type="glass",
                positions=GLASS_POSITIONS,
            )
        )

        return SceneObservation(objects=tuple(objects))

    def _detect_objects(
        self,
        prefix: str,
        object_type: str,
        positions: tuple[tuple[float, float, float], ...],
    ) -> list[SceneObject]:
        """Detect all objects of one type."""

        detected: list[SceneObject] = []

        for index, fallback_position in enumerate(positions, start=1):
            object_id = f"{prefix}_{index}"

            geom_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                object_id,
            )

            if geom_id < 0:
                continue

            position = self._world_geom_position(
                geom_id,
                fallback_position,
            )

            detected.append(
                SceneObject(
                    object_id=object_id,
                    object_type=object_type,
                    position=position,
                )
            )

        return detected

    def _world_geom_position(
        self,
        geom_id: int,
        fallback_position: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Return a geometry's world position."""

        if self.data is None:
            # Backward-compatible fallback for callers that only provide
            # the MuJoCo model.
            return tuple(float(value) for value in fallback_position)

        position = self.data.geom_xpos[geom_id]

        return (
            float(position[0]),
            float(position[1]),
            float(position[2]),
        )