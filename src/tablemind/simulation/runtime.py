"""Small, explicit control surface for the milestone-one MuJoCo model."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco

from tablemind.robot_config.so101 import ARM_PLACEMENTS, JOINT_NAMES, namespaced
from tablemind.simulation.composer import build_bimanual_scene


@dataclass
class BimanualTableSimulation:
    """Loaded MuJoCo model with independently addressable SO-101 arms."""

    model: mujoco.MjModel
    data: mujoco.MjData

    @classmethod
    def create(cls) -> "BimanualTableSimulation":
        scene_path = build_bimanual_scene()
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return cls(model=model, data=data)

    @property
    def arms(self) -> tuple[str, ...]:
        return tuple(ARM_PLACEMENTS)

    def joint_names(self, arm: str) -> tuple[str, ...]:
        self._validate_arm(arm)
        return tuple(namespaced(arm, name) for name in JOINT_NAMES)

    def actuator_names(self, arm: str) -> tuple[str, ...]:
        return self.joint_names(arm)

    def set_joint_targets(self, arm: str, targets: dict[str, float]) -> None:
        """Set position-actuator targets by local joint name for exactly one arm."""

        self._validate_arm(arm)
        unknown = set(targets).difference(JOINT_NAMES)
        if unknown:
            raise ValueError(f"Unknown {arm} joint target(s): {sorted(unknown)}")
        for joint_name, target in targets.items():
            actuator_id = self._name_id(mujoco.mjtObj.mjOBJ_ACTUATOR, namespaced(arm, joint_name))
            self.data.ctrl[actuator_id] = target

    def joint_positions(self, arm: str) -> dict[str, float]:
        """Read one arm's six generalized positions keyed by local joint name."""

        self._validate_arm(arm)
        result: dict[str, float] = {}
        for joint_name in JOINT_NAMES:
            joint_id = self._name_id(mujoco.mjtObj.mjOBJ_JOINT, namespaced(arm, joint_name))
            result[joint_name] = float(self.data.qpos[self.model.jnt_qposadr[joint_id]])
        return result

    def step(self, count: int = 1) -> None:
        if count < 1:
            raise ValueError("count must be at least 1")
        for _ in range(count):
            mujoco.mj_step(self.model, self.data)

    def _validate_arm(self, arm: str) -> None:
        if arm not in ARM_PLACEMENTS:
            raise ValueError(f"Unknown arm {arm!r}; expected one of {self.arms}")

    def _name_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise KeyError(f"MuJoCo model does not contain {name!r}")
        return object_id
