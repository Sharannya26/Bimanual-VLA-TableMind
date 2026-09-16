"""Demonstrate closed-loop placement verification and recovery."""

from __future__ import annotations

from tablemind.control.so101 import SO101Controller
from tablemind.manipulation.grasping import GraspManager
from tablemind.manipulation.recovery import ManipulationRecovery
from tablemind.manipulation.verification import ManipulationVerifier
from tablemind.simulation import BimanualTableSimulation


def main() -> None:
    simulation = BimanualTableSimulation.create()

    controller = SO101Controller(simulation)
    grasping = GraspManager(simulation, controller)

    verifier = ManipulationVerifier(
        grasping.objects
    )

    recovery = ManipulationRecovery(
        controller,
        grasping,
        verifier,
    )

    arm = "left"
    object_id = "plate_1"

    print("=" * 64)
    print("TABLEMIND - Milestone 5.5.3")
    print("Closed-Loop Verification + Recovery")
    print("=" * 64)

    initial = grasping.objects.get(object_id)

    print(
        f"\nInitial {object_id} position:"
        f"\n{initial.position}"
    )

    # ------------------------------------------------------------
    # 1. NORMAL PICK
    # ------------------------------------------------------------

    approach_target = (
        initial.position[0],
        initial.position[1],
        initial.position[2] + 0.08,
    )

    print("\n[1] Approaching object...")

    result = controller.move_to(
        arm,
        approach_target,
        tolerance=0.04,
        max_steps=700,
    )

    print(
        f"Reached: {result.reached} | "
        f"error={result.position_error:.3f} m"
    )

    if not result.reached:
        print("❌ Initial approach failed.")
        return

    grasp_target = (
        initial.position[0],
        initial.position[1],
        initial.position[2] + 0.02,
    )

    print("\n[2] Moving into grasp position...")

    result = controller.move_to(
        arm,
        grasp_target,
        tolerance=0.04,
        max_steps=700,
    )

    print(
        f"Reached: {result.reached} | "
        f"error={result.position_error:.3f} m"
    )

    if not result.reached:
        print("❌ Initial grasp positioning failed.")
        return

    print("\n[3] Grasping...")

    grasp_result = grasping.grasp(
        arm,
        object_id,
    )

    print(
        f"Success: {grasp_result.success} | "
        f"distance={grasp_result.distance:.3f} m"
    )

    if not grasp_result.success:
        print("❌ Initial grasp failed.")
        return

    # ------------------------------------------------------------
    # 2. DELIBERATELY BAD PLACEMENT
    # ------------------------------------------------------------

    bad_target = (
        -0.10,
        0.00,
        initial.position[2] + 0.15,
    )

    print("\n[4] Executing intentionally imperfect placement...")
    print(f"Requested target: {bad_target}")

    lift_target = (
        initial.position[0],
        initial.position[1],
        initial.position[2] + 0.15,
    )

    result = controller.move_to(
        arm,
        lift_target,
        tolerance=0.04,
        max_steps=700,
    )

    if not result.reached:
        print("❌ Lift failed.")
        grasping.release(
            arm,
            object_id,
        )
        return

    result = controller.move_to(
        arm,
        bad_target,
        tolerance=0.04,
        max_steps=700,
    )

    print(
        f"Transport reached: {result.reached} | "
        f"error={result.position_error:.3f} m"
    )

    if not result.reached:
        print(
            "⚠️ Transport target was not fully reached."
        )

    release_target = (
        bad_target[0],
        bad_target[1],
        initial.position[2] + 0.02,
    )

    controller.move_to(
        arm,
        release_target,
        tolerance=0.04,
        max_steps=700,
    )

    grasping.release(
        arm,
        object_id,
    )

    simulation.step(40)

    # ------------------------------------------------------------
    # 3. VERIFY
    # ------------------------------------------------------------

    print("\n[5] Verifying placement...")

    verification = verifier.verify_position(
        object_id,
        bad_target,
        tolerance=0.04,
    )

    print(
        f"Verification success: "
        f"{verification.success}"
    )

    print(
        f"Actual position: "
        f"{verification.actual_position}"
    )

    print(
        f"Position error: "
        f"{verification.position_error:.3f} m"
    )

    print(
        f"Reason: {verification.reason}"
    )

    # ------------------------------------------------------------
    # 4. RECOVERY
    # ------------------------------------------------------------

    if verification.success:
        print(
            "\nUnexpectedly successful placement; "
            "no recovery required."
        )
        return

    corrected_target = (
        -0.22,
        0.00,
        initial.position[2],
    )

    print("\n[6] VERIFICATION FAILED")
    print("Initiating recovery...")
    print(
        f"Corrected target: "
        f"{corrected_target}"
    )

    recovery_result = recovery.recover_placement(
        arm,
        object_id,
        corrected_target,
        tolerance=0.06,
    )

    print("\n[7] Recovery result")

    print(
        f"Success: {recovery_result.success}"
    )

    print(
        f"Attempts: {recovery_result.attempts}"
    )

    print(
        f"Final position error: "
        f"{recovery_result.verification.position_error:.3f} m"
    )

    print(
        f"Reason: {recovery_result.reason}"
    )

    print("\n" + "=" * 64)

    if recovery_result.success:
        print("✅ CLOSED-LOOP RECOVERY SUCCESSFUL")
        print(
            "TABLEMIND detected the failed placement "
            "and recovered automatically."
        )
    else:
        print("❌ RECOVERY FAILED")

    print("=" * 64)


if __name__ == "__main__":
    main()