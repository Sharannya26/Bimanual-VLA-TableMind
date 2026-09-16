"""Recovery utilities for TABLEMIND manipulation."""

from __future__ import annotations

from dataclasses import dataclass

from tablemind.control.so101 import SO101Controller
from tablemind.manipulation.grasping import GraspManager
from tablemind.manipulation.verification import (
    ManipulationVerifier,
    VerificationResult,
)


@dataclass(frozen=True)
class RecoveryResult:
    """Result of one manipulation recovery attempt."""

    success: bool
    attempts: int
    verification: VerificationResult
    reason: str


class ManipulationRecovery:
    """Recover from failed object placement."""

    def __init__(
        self,
        controller: SO101Controller,
        grasping: GraspManager,
        verifier: ManipulationVerifier,
    ) -> None:
        self.controller = controller
        self.grasping = grasping
        self.verifier = verifier

    def recover_placement(
        self,
        arm: str,
        object_id: str,
        corrected_position: tuple[float, float, float],
        *,
        tolerance: float = 0.06,
    ) -> RecoveryResult:
        """Attempt to correct a failed placement."""

        print(
            f"Recovery: attempting corrected placement "
            f"for {object_id}..."
        )

        object_state = self.grasping.objects.get(object_id)

        # Re-approach the object.
        approach_target = (
            object_state.position[0],
            object_state.position[1],
            object_state.position[2] + 0.08,
        )

        result = self.controller.move_to(
            arm,
            approach_target,
            tolerance=0.04,
            max_steps=700,
        )

        if not result.reached:
            verification = self.verifier.verify_position(
                object_id,
                corrected_position,
                tolerance=tolerance,
            )

            return RecoveryResult(
                success=False,
                attempts=1,
                verification=verification,
                reason="Recovery approach failed.",
            )

        # Move into grasp position.
        grasp_target = (
            object_state.position[0],
            object_state.position[1],
            object_state.position[2] + 0.02,
        )

        result = self.controller.move_to(
            arm,
            grasp_target,
            tolerance=0.04,
            max_steps=700,
        )

        if not result.reached:
            verification = self.verifier.verify_position(
                object_id,
                corrected_position,
                tolerance=tolerance,
            )

            return RecoveryResult(
                success=False,
                attempts=1,
                verification=verification,
                reason="Recovery grasp positioning failed.",
            )

        # Re-grasp.
        grasp_result = self.grasping.grasp(
            arm,
            object_id,
        )

        if not grasp_result.success:
            verification = self.verifier.verify_position(
                object_id,
                corrected_position,
                tolerance=tolerance,
            )

            return RecoveryResult(
                success=False,
                attempts=1,
                verification=verification,
                reason="Recovery grasp failed.",
            )

        # Lift before transporting again.
        lift_target = (
            object_state.position[0],
            object_state.position[1],
            object_state.position[2] + 0.15,
        )

        result = self.controller.move_to(
            arm,
            lift_target,
            tolerance=0.04,
            max_steps=700,
        )

        if not result.reached:
            self.grasping.release(
                arm,
                object_id,
            )

            verification = self.verifier.verify_position(
                object_id,
                corrected_position,
                tolerance=tolerance,
            )

            return RecoveryResult(
                success=False,
                attempts=1,
                verification=verification,
                reason="Recovery lift failed.",
            )

        # Transport to corrected placement.
        transport_target = (
            corrected_position[0],
            corrected_position[1],
            object_state.position[2] + 0.15,
        )

        result = self.controller.move_to(
            arm,
            transport_target,
            tolerance=0.04,
            max_steps=700,
        )

        if not result.reached:
            self.grasping.release(
                arm,
                object_id,
            )

            verification = self.verifier.verify_position(
                object_id,
                corrected_position,
                tolerance=tolerance,
            )

            return RecoveryResult(
                success=False,
                attempts=1,
                verification=verification,
                reason="Recovery transport failed.",
            )

        # Lower.
        release_target = (
            corrected_position[0],
            corrected_position[1],
            object_state.position[2] + 0.02,
        )

        result = self.controller.move_to(
            arm,
            release_target,
            tolerance=0.04,
            max_steps=700,
        )

        if not result.reached:
            self.grasping.release(
                arm,
                object_id,
            )

            verification = self.verifier.verify_position(
                object_id,
                corrected_position,
                tolerance=tolerance,
            )

            return RecoveryResult(
                success=False,
                attempts=1,
                verification=verification,
                reason="Recovery lowering failed.",
            )

        # Release and let physics settle.
        self.grasping.release(
            arm,
            object_id,
        )

        self.grasping.simulation.step(40)

        # Verify the corrected placement.
        verification = self.verifier.verify_position(
            object_id,
            corrected_position,
            tolerance=tolerance,
        )

        if verification.success:
            return RecoveryResult(
                success=True,
                attempts=1,
                verification=verification,
                reason="Placement successfully recovered.",
            )

        return RecoveryResult(
            success=False,
            attempts=1,
            verification=verification,
            reason="Corrected placement still failed verification.",
        )