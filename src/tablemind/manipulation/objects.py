"""Dynamic object utilities for TABLEMIND."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco


@dataclass(frozen=True)
class DynamicObject:
    """State of one dynamic manipulation object."""

    object_id: str
    object_type: str
    body_id: int
    position: tuple[float, float, float]
    linear_velocity: tuple[float, float, float]


class DynamicObjectManager:
    """
    Access and inspect dynamic manipulation objects in MuJoCo.

    This class establishes the dynamic-object layer that later
    grasping and pick-and-place logic will use.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        self.model = model
        self.data = data

    def get(self, object_id: str) -> DynamicObject:
        """Return the current state of one dynamic object."""

        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            object_id,
        )

        if body_id < 0:
            raise KeyError(
                f"MuJoCo model does not contain dynamic object "
                f"body {object_id!r}"
            )

        position = self.data.xpos[body_id]

        joint_id = self.model.body_jntadr[body_id]

        if joint_id < 0:
            linear_velocity = (0.0, 0.0, 0.0)
        else:
            qvel_address = self.model.jnt_dofadr[joint_id]

            velocity = self.data.qvel[
                qvel_address:qvel_address + 3
            ]

            linear_velocity = (
                float(velocity[0]),
                float(velocity[1]),
                float(velocity[2]),
            )

        return DynamicObject(
            object_id=object_id,
            object_type=self._infer_object_type(object_id),
            body_id=body_id,
            position=(
                float(position[0]),
                float(position[1]),
                float(position[2]),
            ),
            linear_velocity=linear_velocity,
        )

    def exists(self, object_id: str) -> bool:
        """Return whether a dynamic object exists."""

        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            object_id,
        )

        return body_id >= 0

    def all_objects(self) -> tuple[DynamicObject, ...]:
        """Return all known TABLEMIND manipulation objects."""

        objects: list[DynamicObject] = []

        for object_id in self._known_object_ids():
            if self.exists(object_id):
                objects.append(self.get(object_id))

        return tuple(objects)

    def _known_object_ids(self) -> tuple[str, ...]:
        """Return known manipulation object IDs."""

        return (
            "plate_1",
            "plate_2",
            "glass_1",
            "glass_2",
        )

    def _infer_object_type(self, object_id: str) -> str:
        """Infer object type from its identifier."""

        if object_id.startswith("plate_"):
            return "plate"

        if object_id.startswith("glass_"):
            return "glass"

        return "unknown"