"""Physical execution of conversational goal additions for TABLEMIND."""

from __future__ import annotations

from dataclasses import dataclass

from tablemind.control.so101 import SO101Controller
from tablemind.manipulation.grasping import GraspManager
from tablemind.perception.scene import SceneObject
from tablemind.reasoning.goal_reconciliation import PlaceSettingSlot
from tablemind.simulation import BimanualTableSimulation


Position = tuple[float, float, float]


@dataclass(frozen=True)
class AdditionResult:
    """Result of restoring one object to an active place-setting slot."""

    success: bool
    arm: str
    object_id: str
    message: str


class GoalAdditionExecutor:
    """
    Physically restore objects belonging to a conversationally
    reactivated place-setting slot.

    This executor is intentionally sequential because both objects
    belonging to one place setting may require the same physical arm.
    """

    APPROACH_HEIGHT = 0.12
    GLASS_APPROACH_HEIGHT = 0.20
    GRASP_OFFSET = 0.02
    LIFT_HEIGHT = 0.12

    PLATE_Z = 0.772
    GLASS_Z = 0.835

    def __init__(
        self,
        simulation: BimanualTableSimulation,
        controller: SO101Controller | None = None,
        grasping: GraspManager | None = None,
    ) -> None:
        self.simulation = simulation

        self.controller = (
            controller
            or SO101Controller(simulation)
        )

        self.grasping = (
            grasping
            or GraspManager(
                simulation,
                self.controller,
            )
        )

    def add_setting(
        self,
        slot: PlaceSettingSlot,
        plate: SceneObject,
        glass: SceneObject,
    ) -> tuple[AdditionResult, ...]:
        """
        Restore a complete place setting.

        Plate and glass are intentionally executed sequentially.
        """

        results: list[AdditionResult] = []

        plate_result = self._move_object(
            object_id=plate.object_id,
            object_type="plate",
            source_position=plate.position,
            target_position=slot.plate_position,
        )

        results.append(plate_result)

        if not plate_result.success:
            return tuple(results)

        glass_result = self._move_object(
            object_id=glass.object_id,
            object_type="glass",
            source_position=glass.position,
            target_position=slot.glass_position,
        )

        results.append(glass_result)

        return tuple(results)

    def _move_object(
        self,
        object_id: str,
        object_type: str,
        source_position: Position,
        target_position: Position,
    ) -> AdditionResult:
        """Move one object from its current/staged position to its slot."""

        arm = (
            "left"
            if source_position[0] < 0
            else "right"
        )

        if object_type == "glass":
            approach_height = self.GLASS_APPROACH_HEIGHT
        else:
            approach_height = self.APPROACH_HEIGHT

        approach_position = (
            source_position[0],
            source_position[1],
            source_position[2] + approach_height,
        )

        grasp_position = (
            source_position[0],
            source_position[1],
            source_position[2] + self.GRASP_OFFSET,
        )

        lift_position = (
            source_position[0],
            source_position[1],
            source_position[2] + self.LIFT_HEIGHT,
        )

        transport_position = (
            target_position[0],
            target_position[1],
            target_position[2] + self.LIFT_HEIGHT,
        )

        # --------------------------------------------------
        # APPROACH
        # --------------------------------------------------

        approach_result = self.controller.move_to(
            arm,
            approach_position,
            tolerance=0.04,
            max_steps=700,
        )

        if not approach_result.reached:
            return AdditionResult(
                success=False,
                arm=arm,
                object_id=object_id,
                message=(
                    "Approach failed: "
                    f"error={approach_result.position_error:.3f} m"
                ),
            )

        # --------------------------------------------------
        # GRASP POSITION
        # --------------------------------------------------

        grasp_position_result = self.controller.move_to(
            arm,
            grasp_position,
            tolerance=0.04,
            max_steps=700,
        )

        if not grasp_position_result.reached:
            return AdditionResult(
                success=False,
                arm=arm,
                object_id=object_id,
                message=(
                    "Grasp positioning failed: "
                    f"error={grasp_position_result.position_error:.3f} m"
                ),
            )

        # --------------------------------------------------
        # GRASP
        # --------------------------------------------------

        grasp_result = self.grasping.grasp(
            arm,
            object_id,
        )

        if not grasp_result.success:
            return AdditionResult(
                success=False,
                arm=arm,
                object_id=object_id,
                message=(
                    "Grasp failed: "
                    f"{grasp_result.reason}"
                ),
            )

        # --------------------------------------------------
        # LIFT
        # --------------------------------------------------

        lift_result = self.controller.move_to(
            arm,
            lift_position,
            tolerance=0.04,
            max_steps=700,
        )

        if not lift_result.reached:
            self.grasping.release(
                arm,
                object_id,
            )

            return AdditionResult(
                success=False,
                arm=arm,
                object_id=object_id,
                message=(
                    "Lift failed: "
                    f"error={lift_result.position_error:.3f} m"
                ),
            )

        # --------------------------------------------------
        # TRANSPORT
        # --------------------------------------------------

        transport_result = self.controller.move_to(
            arm,
            transport_position,
            tolerance=0.04,
            max_steps=700,
        )

        if not transport_result.reached:
            self.grasping.release(
                arm,
                object_id,
            )

            return AdditionResult(
                success=False,
                arm=arm,
                object_id=object_id,
                message=(
                    "Transport failed: "
                    f"error={transport_result.position_error:.3f} m"
                ),
            )

        # --------------------------------------------------
        # LOWER
        # --------------------------------------------------

        lower_result = self.controller.move_to(
            arm,
            target_position,
            tolerance=0.04,
            max_steps=700,
        )

        if not lower_result.reached:
            self.grasping.release(
                arm,
                object_id,
            )

            return AdditionResult(
                success=False,
                arm=arm,
                object_id=object_id,
                message=(
                    "Lowering failed: "
                    f"error={lower_result.position_error:.3f} m"
                ),
            )

        # --------------------------------------------------
        # RELEASE
        # --------------------------------------------------

        self.grasping.release(
            arm,
            object_id,
        )

        self.simulation.step(20)

        return AdditionResult(
            success=True,
            arm=arm,
            object_id=object_id,
            message=(
                f"{object_id} successfully restored "
                f"to place-setting slot."
            ),
        )