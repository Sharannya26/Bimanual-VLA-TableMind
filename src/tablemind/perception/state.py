"""Structured scene state for TABLEMIND."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from tablemind.perception.scene import SceneObservation


@dataclass(frozen=True)
class RobotState:
    """Current state of one robot arm."""

    arm_id: str
    end_effector_position: tuple[float, float, float]
    gripper_position: float


@dataclass(frozen=True)
class SceneState:
    """
    Complete structured representation of the TABLEMIND scene.

    This is the interface that future reasoning and planning
    modules will consume.
    """

    objects: SceneObservation
    robots: tuple[RobotState, ...]
    simulation_time: float

    def robot(self, arm_id: str) -> RobotState | None:
        """Return a robot state by arm ID."""

        for robot in self.robots:
            if robot.arm_id == arm_id:
                return robot

        return None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the scene state into a JSON-friendly dictionary.
        """

        return {
            "objects": [
                {
                    "id": obj.object_id,
                    "type": obj.object_type,
                    "position": list(obj.position),
                }
                for obj in self.objects.objects
            ],
            "robots": [
                asdict(robot)
                for robot in self.robots
            ],
            "simulation_time": self.simulation_time,
        }