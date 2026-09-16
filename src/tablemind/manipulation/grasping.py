"""Grasping utilities for TABLEMIND."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco

from tablemind.control.so101 import SO101Controller
from tablemind.manipulation.objects import DynamicObjectManager
from tablemind.simulation import BimanualTableSimulation


@dataclass(frozen=True)
class GraspResult:
    """Result of a grasp attempt."""

    success: bool
    arm: str
    object_id: str
    distance: float
    reason: str


class GraspManager:
    """Manage physical object grasping using MuJoCo weld constraints."""

    GRASP_DISTANCE = 0.085

    def __init__(
        self,
        simulation: BimanualTableSimulation,
        controller: SO101Controller | None = None,
    ):
        self.simulation = simulation
        self.controller = controller or SO101Controller(simulation)
        self.objects = DynamicObjectManager(
            simulation.model,
            simulation.data,
        )

    def distance_to_object(
        self,
        arm: str,
        object_id: str,
    ) -> float:
        """Return Euclidean distance from the gripper to an object."""
        gripper = self.controller.end_effector_position(arm)
        obj = self.objects.get(object_id)

        dx = gripper[0] - obj.position[0]
        dy = gripper[1] - obj.position[1]
        dz = gripper[2] - obj.position[2]

        return float((dx * dx + dy * dy + dz * dz) ** 0.5)

    def can_grasp(
        self,
        arm: str,
        object_id: str,
    ) -> bool:
        """Return whether the object is within grasping distance."""
        distance = self.distance_to_object(arm, object_id)
        return distance <= self.GRASP_DISTANCE

    def grasp(
        self,
        arm: str,
        object_id: str,
    ) -> GraspResult:
        """Grasp an object using the existing MuJoCo weld constraint."""
        distance = self.distance_to_object(arm, object_id)

        if distance > self.GRASP_DISTANCE:
            return GraspResult(
                False,
                arm,
                object_id,
                distance,
                (
                    f"Object is too far from gripper "
                    f"({distance:.3f} m > {self.GRASP_DISTANCE:.3f} m)"
                ),
            )

        self._activate_grasp_constraint(arm, object_id)
        self.controller.close_gripper(arm)
        self.simulation.step(5)

        return GraspResult(
            True,
            arm,
            object_id,
            distance,
            "Object successfully grasped.",
        )

    def release(
        self,
        arm: str,
        object_id: str,
    ) -> None:
        """Release an object and deactivate its grasp constraint."""
        self.controller.open_gripper(arm)
        self._deactivate_grasp_constraint(arm, object_id)
        self.simulation.step(5)

    def _constraint_name(
        self,
        arm: str,
        object_id: str,
    ) -> str:
        return f"grasp_{arm}_{object_id}"

    def _constraint_id(
        self,
        arm: str,
        object_id: str,
    ) -> int:
        name = self._constraint_name(arm, object_id)

        constraint_id = mujoco.mj_name2id(
            self.simulation.model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            name,
        )

        if constraint_id < 0:
            raise KeyError(
                f"Grasp constraint not found: {name}"
            )

        return constraint_id

    def _activate_grasp_constraint(
        self,
        arm: str,
        object_id: str,
    ) -> None:
        constraint_id = self._constraint_id(arm, object_id)

        self.simulation.data.eq_active[constraint_id] = 1
        mujoco.mj_forward(
            self.simulation.model,
            self.simulation.data,
        )

    def _deactivate_grasp_constraint(
        self,
        arm: str,
        object_id: str,
    ) -> None:
        constraint_id = self._constraint_id(arm, object_id)

        self.simulation.data.eq_active[constraint_id] = 0
        mujoco.mj_forward(
            self.simulation.model,
            self.simulation.data,
        )