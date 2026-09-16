"""Build structured TABLEMIND scene state from simulation data."""

from __future__ import annotations

import mujoco

from tablemind.perception.coordinates import WorldCoordinateSystem
from tablemind.perception.detector import GroundTruthDetector
from tablemind.perception.state import RobotState, SceneState
from tablemind.robot_config.so101 import ARM_PLACEMENTS


class SceneStateBuilder:
    """Build a complete scene state from the MuJoCo simulation."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> None:
        self.model = model
        self.data = data

        self.detector = GroundTruthDetector(model,data)
        self.coordinates = WorldCoordinateSystem(model, data)

    def build(self) -> SceneState:
        """Build the current structured scene state."""

        observation = self.detector.detect()

        robots = tuple(
            self._build_robot_state(arm)
            for arm in ARM_PLACEMENTS
        )

        return SceneState(
            objects=observation,
            robots=robots,
            simulation_time=float(self.data.time),
        )

    def _build_robot_state(self, arm: str) -> RobotState:
        """Build the state representation for one SO-101 arm."""

        site_name = f"{arm}_gripperframe"

        end_effector = self.coordinates.site_position(site_name)

        gripper_joint_name = f"{arm}_gripper"

        joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            gripper_joint_name,
        )

        if joint_id < 0:
            raise KeyError(
                f"MuJoCo model does not contain joint "
                f"{gripper_joint_name!r}"
            )

        qpos_address = self.model.jnt_qposadr[joint_id]

        gripper_position = float(
            self.data.qpos[qpos_address]
        )

        return RobotState(
            arm_id=arm,
            end_effector_position=end_effector.xyz,
            gripper_position=gripper_position,
        )