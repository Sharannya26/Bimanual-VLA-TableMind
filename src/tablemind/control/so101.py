"""Deterministic Cartesian and gripper control for the simulated SO-101 arms."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from tablemind.robot_config.so101 import ARM_PLACEMENTS, JOINT_NAMES, namespaced
from tablemind.simulation import BimanualTableSimulation


# The SO-101 model has five arm joints followed by one gripper joint.
ARM_JOINT_NAMES = JOINT_NAMES[:-1]


@dataclass(frozen=True)
class CartesianMoveResult:
    """Outcome of a bounded Cartesian movement."""

    reached: bool
    steps: int
    final_position: tuple[float, float, float]
    position_error: float


@dataclass(frozen=True)
class ObjectManipulationResult:
    """Outcome of an attempted pick/place operation."""

    completed: bool
    reason: str


class SO101Controller:
    """Reusable controller for the simulated SO-101 arms.

    Cartesian movement uses damped least-squares position IK.

    Each newly calculated joint target is held for multiple MuJoCo
    simulation steps. This gives the position actuators time to physically
    track the target instead of changing the target after only one step.

    The optional ``should_stop`` callback allows a higher-level system,
    such as the conversational TABLEMIND executor, to interrupt a
    Cartesian movement safely.
    """

    def __init__(
        self,
        simulation: BimanualTableSimulation,
        *,
        control_steps_per_update: int = 20,
    ) -> None:
        if control_steps_per_update < 1:
            raise ValueError("control_steps_per_update must be at least 1")

        self.simulation = simulation
        self.control_steps_per_update = control_steps_per_update

    def end_effector_position(
        self,
        arm: str,
    ) -> tuple[float, float, float]:
        """Return the current world position of the arm gripper frame."""

        self._validate_arm(arm)

        site_id = self._name_id(
            mujoco.mjtObj.mjOBJ_SITE,
            namespaced(arm, "gripperframe"),
        )

        return tuple(
            float(value)
            for value in self.simulation.data.site_xpos[site_id]
        )

    def move_to(
        self,
        arm: str,
        target: tuple[float, float, float],
        *,
        tolerance: float = 0.025,
        should_stop: Callable[[], bool] | None = None,
        max_steps: int = 300,
        max_cartesian_step: float = 0.015,
        damping: float = 0.05,
    ) -> CartesianMoveResult:
        """Move the end effector toward a world-frame Cartesian target.

        Parameters
        ----------
        arm:
            "left" or "right".

        target:
            Desired world-frame XYZ position.

        tolerance:
            Maximum Cartesian error considered successful.

        should_stop:
            Optional callback checked between control updates.
            If it returns True, the movement stops safely at the
            current position.

        max_steps:
            Maximum number of IK/control updates.

        max_cartesian_step:
            Maximum Cartesian movement requested per IK update.

        damping:
            Damping coefficient for the least-squares IK solver.
        """

        self._validate_arm(arm)

        if tolerance <= 0:
            raise ValueError("tolerance must be positive")

        if max_steps < 0:
            raise ValueError("max_steps cannot be negative")

        if max_cartesian_step <= 0:
            raise ValueError("max_cartesian_step must be positive")

        if damping <= 0:
            raise ValueError("damping must be positive")

        target_array = np.asarray(target, dtype=float)

        if target_array.shape != (3,):
            raise ValueError("target must contain exactly three coordinates")

        if not np.isfinite(target_array).all():
            raise ValueError("target must contain finite coordinates")

        site_id = self._name_id(
            mujoco.mjtObj.mjOBJ_SITE,
            namespaced(arm, "gripperframe"),
        )

        # Resolve the five arm joints once.
        joint_ids = [
            self._name_id(
                mujoco.mjtObj.mjOBJ_JOINT,
                namespaced(arm, joint_name),
            )
            for joint_name in ARM_JOINT_NAMES
        ]

        dof_ids = np.asarray(
            [
                self.simulation.model.jnt_dofadr[joint_id]
                for joint_id in joint_ids
            ],
            dtype=int,
        )

        for update in range(max_steps + 1):

            # Check for a conversational or external stop request
            # before calculating the next movement update.
            if should_stop is not None and should_stop():
                mujoco.mj_forward(
                    self.simulation.model,
                    self.simulation.data,
                )

                current = np.array(
                    self.simulation.data.site_xpos[site_id],
                    dtype=float,
                )

                error_norm = float(
                    np.linalg.norm(target_array - current)
                )

                return CartesianMoveResult(
                    reached=False,
                    steps=update,
                    final_position=tuple(current),
                    position_error=error_norm,
                )

            # Make sure all derived quantities are up to date.
            mujoco.mj_forward(
                self.simulation.model,
                self.simulation.data,
            )

            current = np.array(
                self.simulation.data.site_xpos[site_id],
                dtype=float,
            )

            error = target_array - current
            error_norm = float(np.linalg.norm(error))

            # Successful convergence.
            if error_norm <= tolerance:
                return CartesianMoveResult(
                    reached=True,
                    steps=update,
                    final_position=tuple(current),
                    position_error=error_norm,
                )

            # Target was not reached within the requested update budget.
            if update == max_steps:
                return CartesianMoveResult(
                    reached=False,
                    steps=update,
                    final_position=tuple(current),
                    position_error=error_norm,
                )

            # Limit the requested Cartesian movement per update.
            desired_delta = error * min(
                1.0,
                max_cartesian_step / error_norm,
            )

            # Position Jacobian for the gripper frame.
            jacobian_position = np.zeros(
                (3, self.simulation.model.nv),
                dtype=float,
            )

            jacobian_rotation = np.zeros(
                (3, self.simulation.model.nv),
                dtype=float,
            )

            mujoco.mj_jacSite(
                self.simulation.model,
                self.simulation.data,
                jacobian_position,
                jacobian_rotation,
                site_id,
            )

            arm_jacobian = jacobian_position[:, dof_ids]

            # Damped least-squares inverse Jacobian:
            #
            # dq = J^T (J J^T + λ²I)^-1 dx
            #
            # This is more stable near singular configurations than
            # a direct matrix inverse.
            identity = np.eye(3)

            joint_delta = arm_jacobian.T @ np.linalg.solve(
                arm_jacobian @ arm_jacobian.T
                + damping**2 * identity,
                desired_delta,
            )

            # Convert the calculated joint increments into actuator targets.
            targets: dict[str, float] = {}

            for joint_name, joint_id, delta in zip(
                ARM_JOINT_NAMES,
                joint_ids,
                joint_delta,
                strict=True,
            ):
                actuator_id = self._name_id(
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    namespaced(arm, joint_name),
                )

                qpos_address = self.simulation.model.jnt_qposadr[joint_id]

                current_joint_position = float(
                    self.simulation.data.qpos[qpos_address]
                )

                control_range = (
                    self.simulation.model.actuator_ctrlrange[actuator_id]
                )

                requested_target = (
                    current_joint_position + float(delta)
                )

                targets[joint_name] = float(
                    np.clip(
                        requested_target,
                        control_range[0],
                        control_range[1],
                    )
                )

            # Send the new target to the position actuators.
            self.simulation.set_joint_targets(
                arm,
                targets,
            )

            # Allow the position controllers to track the target
            # for several simulation steps before calculating the
            # next IK correction.
            self.simulation.step(
                self.control_steps_per_update
            )

        raise AssertionError(
            "move_to loop must return before reaching this point"
        )

    def open_gripper(
        self,
        arm: str,
        *,
        settle_steps: int = 100,
    ) -> None:
        """Open the selected arm's gripper."""

        self._set_gripper(
            arm,
            limit_index=1,
            settle_steps=settle_steps,
        )

    def close_gripper(
        self,
        arm: str,
        *,
        settle_steps: int = 100,
    ) -> None:
        """Close the selected arm's gripper."""

        self._set_gripper(
            arm,
            limit_index=0,
            settle_steps=settle_steps,
        )

    def execute_pick(
        self,
        arm: str,
        object_name: str,
    ) -> ObjectManipulationResult:
        """Attempt a pick operation.

        Milestone 2 intentionally does not claim physical grasping because
        the current plate and glass objects are fixed placeholder geometry.
        """

        self._validate_arm(arm)

        if not self._is_movable_geom(object_name):
            return ObjectManipulationResult(
                completed=False,
                reason=(
                    f"{object_name!r} is fixed placeholder geometry "
                    "and cannot be physically grasped yet"
                ),
            )

        return ObjectManipulationResult(
            completed=False,
            reason=(
                "Movable-object grasp attachment is not implemented "
                "in Milestone 2"
            ),
        )

    def execute_place(
        self,
        arm: str,
        target: tuple[float, float, float],
    ) -> CartesianMoveResult:
        """Move to a placement position and open the gripper."""

        result = self.move_to(
            arm,
            target,
        )

        if result.reached:
            self.open_gripper(arm)

        return result

    def object_position(
        self,
        object_name: str,
    ) -> tuple[float, float, float]:
        """Return the current world position of a named geometry."""

        geometry_id = self._name_id(
            mujoco.mjtObj.mjOBJ_GEOM,
            object_name,
        )

        return tuple(
            float(value)
            for value in self.simulation.data.geom_xpos[geometry_id]
        )

    def _set_gripper(
        self,
        arm: str,
        *,
        limit_index: int,
        settle_steps: int,
    ) -> None:
        """Command the gripper and let the actuator settle."""

        self._validate_arm(arm)

        if settle_steps < 0:
            raise ValueError(
                "settle_steps cannot be negative"
            )

        actuator_id = self._name_id(
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            namespaced(arm, "gripper"),
        )

        target = float(
            self.simulation.model.actuator_ctrlrange[
                actuator_id
            ][limit_index]
        )

        self.simulation.set_joint_targets(
            arm,
            {"gripper": target},
        )

        if settle_steps:
            self.simulation.step(settle_steps)

    def _is_movable_geom(
        self,
        object_name: str,
    ) -> bool:
        """Check whether a geometry belongs to a free-body object."""

        geometry_id = self._name_id(
            mujoco.mjtObj.mjOBJ_GEOM,
            object_name,
        )

        body_id = int(
            self.simulation.model.geom_bodyid[geometry_id]
        )

        joint_start = int(
            self.simulation.model.body_jntadr[body_id]
        )

        joint_count = int(
            self.simulation.model.body_jntnum[body_id]
        )

        return any(
            self.simulation.model.jnt_type[
                joint_start + offset
            ]
            == mujoco.mjtJoint.mjJNT_FREE
            for offset in range(joint_count)
        )

    def _validate_arm(
        self,
        arm: str,
    ) -> None:
        """Validate an arm name."""

        if arm not in ARM_PLACEMENTS:
            raise ValueError(
                f"Unknown arm {arm!r}; "
                f"expected one of {self.simulation.arms}"
            )

    def _name_id(
        self,
        object_type: mujoco.mjtObj,
        name: str,
    ) -> int:
        """Resolve a MuJoCo object name to its integer ID."""

        object_id = mujoco.mj_name2id(
            self.simulation.model,
            object_type,
            name,
        )

        if object_id < 0:
            raise KeyError(
                f"MuJoCo model does not contain {name!r}"
            )

        return object_id