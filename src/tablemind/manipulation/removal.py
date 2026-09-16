"""Physical removal primitives for TABLEMIND Milestone 9.5."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco

from tablemind.control.so101 import SO101Controller
from tablemind.manipulation.grasping import GraspManager
from tablemind.perception.scene import SceneObject


Position = tuple[float, float, float]


@dataclass(frozen=True)
class RemovalResult:
    """Result of physically removing one object."""

    object_id: str
    arm: str
    success: bool
    reason: str = ""


class PhysicalRemovalExecutor:
    """
    Physically remove objects belonging to a conversationally
    deactivated place-setting slot.

    A complete place setting is removed in this order:

        glass
          ↓
        plate

    Removing the glass first prevents the plate transport path
    from disturbing or launching the glass in MuJoCo.
    """

    APPROACH_HEIGHT = 0.12
    LIFT_HEIGHT = 0.12

    # Robot-facing side of the table.
    STAGING_Y = -0.30

    GRASP_OFFSET = 0.02

    MOVE_TOLERANCE = 0.04
    MAX_STEPS = 700

    GRASP_DISTANCE = 0.085

    def __init__(
        self,
        simulation,
        controller=None,
        grasp_manager=None,
    ):
        self.simulation = simulation

        self.controller = controller or SO101Controller(
            simulation
        )

        self.grasp_manager = grasp_manager or GraspManager(
            simulation,
            self.controller,
        )

    # ================================================================
    # PUBLIC API
    # ================================================================

    def remove_object(
        self,
        obj: SceneObject,
    ) -> RemovalResult:
        """Remove one object from the active place-setting."""

        # ------------------------------------------------------------
        # Read the live MuJoCo position.
        # ------------------------------------------------------------

        source_position = self._current_object_position(
            obj.object_id,
            obj.position,
        )

        arm = self._assign_arm(
            source_position[0]
        )

        target_z = source_position[2]

        # ------------------------------------------------------------
        # APPROACH
        # ------------------------------------------------------------

        approach_position = (
            source_position[0],
            source_position[1],
            target_z + self.APPROACH_HEIGHT,
        )

        # ------------------------------------------------------------
        # GRASP
        # ------------------------------------------------------------

        grasp_position = (
            source_position[0],
            source_position[1],
            target_z + self.GRASP_OFFSET,
        )

        # ------------------------------------------------------------
        # LIFT
        # ------------------------------------------------------------

        lift_position = (
            source_position[0],
            source_position[1],
            target_z + self.LIFT_HEIGHT,
        )

        # ------------------------------------------------------------
        # STAGING
        # ------------------------------------------------------------

        staging_x = self._staging_x(
            source_position[0]
        )

        staging_position = (
            staging_x,
            self.STAGING_Y,
            target_z,
        )

        staging_approach = (
            staging_x,
            self.STAGING_Y,
            target_z + self.APPROACH_HEIGHT,
        )

        try:

            # ========================================================
            # RESET / SETTLE
            # ========================================================

            try:
                self.grasp_manager.release(
                    arm,
                    obj.object_id,
                )
            except Exception:
                pass

            self.simulation.step(30)

            # --------------------------------------------------------
            # Refresh after settling.
            # --------------------------------------------------------

            source_position = self._current_object_position(
                obj.object_id,
                source_position,
            )

            target_z = source_position[2]

            approach_position = (
                source_position[0],
                source_position[1],
                target_z + self.APPROACH_HEIGHT,
            )

            grasp_position = (
                source_position[0],
                source_position[1],
                target_z + self.GRASP_OFFSET,
            )

            lift_position = (
                source_position[0],
                source_position[1],
                target_z + self.LIFT_HEIGHT,
            )

            staging_x = self._staging_x(
                source_position[0]
            )

            staging_position = (
                staging_x,
                self.STAGING_Y,
                target_z,
            )

            staging_approach = (
                staging_x,
                self.STAGING_Y,
                target_z + self.APPROACH_HEIGHT,
            )

            # ========================================================
            # 1. APPROACH
            # ========================================================

            approach_result = self.controller.move_to(
                arm,
                approach_position,
                tolerance=self.MOVE_TOLERANCE,
                max_steps=self.MAX_STEPS,
            )

            if not approach_result.reached:
                return RemovalResult(
                    obj.object_id,
                    arm,
                    False,
                    (
                        "Failed to reach removal approach position: "
                        f"error={approach_result.position_error:.3f} m"
                    ),
                )

            # ========================================================
            # 2. GRASP POSITION
            # ========================================================

            grasp_position_result = self.controller.move_to(
                arm,
                grasp_position,
                tolerance=self.MOVE_TOLERANCE,
                max_steps=self.MAX_STEPS,
            )

            if not grasp_position_result.reached:
                return RemovalResult(
                    obj.object_id,
                    arm,
                    False,
                    (
                        "Failed to reach removal grasp position: "
                        f"error={grasp_position_result.position_error:.3f} m"
                    ),
                )

            # ========================================================
            # 3. VERIFY PHYSICAL PROXIMITY
            # ========================================================

            actual_ee = self._safe_end_effector_position(
                arm
            )

            actual_object = self._current_object_position(
                obj.object_id,
                source_position,
            )

            if actual_ee is None:
                return RemovalResult(
                    obj.object_id,
                    arm,
                    False,
                    "Could not read end-effector position.",
                )

            physical_distance = self._distance(
                actual_ee,
                actual_object,
            )

            print(
                f"[REMOVAL] {obj.object_id} "
                f"arm={arm}"
            )

            print(
                f"  actual EE       = "
                f"({actual_ee[0]:.3f}, "
                f"{actual_ee[1]:.3f}, "
                f"{actual_ee[2]:.3f})"
            )

            print(
                f"  actual object   = "
                f"({actual_object[0]:.3f}, "
                f"{actual_object[1]:.3f}, "
                f"{actual_object[2]:.3f})"
            )

            print(
                f"  physical dist   = "
                f"{physical_distance:.3f} m"
            )

            if physical_distance > self.GRASP_DISTANCE:
                return RemovalResult(
                    obj.object_id,
                    arm,
                    False,
                    (
                        "Physical gripper/object distance is too large: "
                        f"{physical_distance:.3f} m"
                    ),
                )

            # ========================================================
            # 4. GRASP
            # ========================================================

            self.simulation.step(5)

            grasp_result = self.grasp_manager.grasp(
                arm,
                obj.object_id,
            )

            if not grasp_result.success:
                return RemovalResult(
                    obj.object_id,
                    arm,
                    False,
                    (
                        "GraspManager rejected the grasp: "
                        f"{grasp_result.reason}. "
                        f"Measured EE-object distance="
                        f"{physical_distance:.3f} m."
                    ),
                )

            # ========================================================
            # 5. LIFT
            # ========================================================

            lift_result = self.controller.move_to(
                arm,
                lift_position,
                tolerance=self.MOVE_TOLERANCE,
                max_steps=self.MAX_STEPS,
            )

            if not lift_result.reached:

                self.grasp_manager.release(
                    arm,
                    obj.object_id,
                )

                return RemovalResult(
                    obj.object_id,
                    arm,
                    False,
                    (
                        "Failed to lift removed object: "
                        f"error={lift_result.position_error:.3f} m"
                    ),
                )

            # ========================================================
            # 6. STAGING APPROACH
            # ========================================================

            staging_result = self.controller.move_to(
                arm,
                staging_approach,
                tolerance=self.MOVE_TOLERANCE,
                max_steps=self.MAX_STEPS,
            )

            if not staging_result.reached:

                self.grasp_manager.release(
                    arm,
                    obj.object_id,
                )

                return RemovalResult(
                    obj.object_id,
                    arm,
                    False,
                    (
                        "Failed to reach staging approach position: "
                        f"error={staging_result.position_error:.3f} m"
                    ),
                )

            # ========================================================
            # 7. LOWER
            # ========================================================

            lower_result = self.controller.move_to(
                arm,
                staging_position,
                tolerance=self.MOVE_TOLERANCE,
                max_steps=self.MAX_STEPS,
            )

            if not lower_result.reached:

                self.grasp_manager.release(
                    arm,
                    obj.object_id,
                )

                return RemovalResult(
                    obj.object_id,
                    arm,
                    False,
                    (
                        "Failed to reach staging position: "
                        f"error={lower_result.position_error:.3f} m"
                    ),
                )

            # ========================================================
            # 8. RELEASE
            # ========================================================

            self.grasp_manager.release(
                arm,
                obj.object_id,
            )

            self.simulation.step(30)

            return RemovalResult(
                obj.object_id,
                arm,
                True,
                "Object moved to staging area.",
            )

        except Exception as exc:

            try:
                self.grasp_manager.release(
                    arm,
                    obj.object_id,
                )
            except Exception:
                pass

            return RemovalResult(
                obj.object_id,
                arm,
                False,
                f"Removal failed: {exc}",
            )

    # ================================================================
    # PLACE-SETTING REMOVAL
    # ================================================================

    def remove_setting(
        self,
        plate: SceneObject,
        glass: SceneObject,
    ) -> tuple[RemovalResult, RemovalResult]:
        """
        Remove a complete place setting.

        IMPORTANT:
        Glass is removed BEFORE plate.

        This prevents the plate's transport path from physically
        disturbing the glass before the glass can be grasped.
        """

        # ------------------------------------------------------------
        # 1. REMOVE GLASS FIRST
        # ------------------------------------------------------------

        glass_result = self.remove_object(
            glass
        )

        if not glass_result.success:

            return (
                RemovalResult(
                    plate.object_id,
                    self._assign_arm(
                        plate.position[0]
                    ),
                    False,
                    "Skipped because glass removal failed.",
                ),
                glass_result,
            )

        # ------------------------------------------------------------
        # Let the glass settle completely.
        # ------------------------------------------------------------

        self.simulation.step(50)

        # ------------------------------------------------------------
        # 2. REFRESH PLATE POSITION
        # ------------------------------------------------------------

        plate_position = self._current_object_position(
            plate.object_id,
            plate.position,
        )

        refreshed_plate = SceneObject(
            object_id=plate.object_id,
            object_type=plate.object_type,
            position=plate_position,
        )

        # ------------------------------------------------------------
        # 3. REMOVE PLATE
        # ------------------------------------------------------------

        plate_result = self.remove_object(
            refreshed_plate
        )

        # ------------------------------------------------------------
        # Return results in logical order:
        #
        #     plate, glass
        #
        # even though physical execution was:
        #
        #     glass, plate
        # ------------------------------------------------------------

        return (
            plate_result,
            glass_result,
        )

    # ================================================================
    # PHYSICAL STATE
    # ================================================================

    def _current_object_position(
        self,
        object_id: str,
        fallback: Position,
    ) -> Position:
        """
        Read the current physical MuJoCo body position.

        This intentionally uses the same data.xpos source used by
        DynamicObjectManager.
        """

        try:

            body_id = mujoco.mj_name2id(
                self.simulation.model,
                mujoco.mjtObj.mjOBJ_BODY,
                object_id,
            )

            if body_id < 0:
                raise KeyError(
                    f"MuJoCo model does not contain body "
                    f"{object_id!r}"
                )

            position = self.simulation.data.xpos[
                body_id
            ]

            return (
                float(position[0]),
                float(position[1]),
                float(position[2]),
            )

        except Exception:

            return (
                float(fallback[0]),
                float(fallback[1]),
                float(fallback[2]),
            )

    def _safe_end_effector_position(
        self,
        arm: str,
    ) -> Position | None:
        """Safely read the current end-effector position."""

        try:

            position = (
                self.controller.end_effector_position(
                    arm
                )
            )

            return (
                float(position[0]),
                float(position[1]),
                float(position[2]),
            )

        except Exception:

            return None

    # ================================================================
    # GEOMETRY
    # ================================================================

    @staticmethod
    def _assign_arm(
        x_position: float,
    ) -> str:
        return (
            "left"
            if x_position < 0
            else "right"
        )

    @staticmethod
    def _staging_x(
        x_position: float,
    ) -> float:

        if x_position < 0:
            return max(
                -0.48,
                x_position,
            )

        return min(
            0.48,
            x_position,
        )

    @staticmethod
    def _distance(
        first: Position,
        second: Position,
    ) -> float:

        return (
            (first[0] - second[0]) ** 2
            + (first[1] - second[1]) ** 2
            + (first[2] - second[2]) ** 2
        ) ** 0.5