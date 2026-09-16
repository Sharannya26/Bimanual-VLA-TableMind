"""Demonstrate a complete pick-and-place operation."""

from __future__ import annotations

from tablemind.control.so101 import SO101Controller
from tablemind.manipulation.grasping import GraspManager
from tablemind.simulation import BimanualTableSimulation


def main() -> None:
    simulation = BimanualTableSimulation.create()

    controller = SO101Controller(simulation)
    grasping = GraspManager(simulation, controller)

    arm = "left"
    object_id = "plate_1"

    print("=" * 60)
    print("TABLEMIND - Milestone 5.3 Pick + Place")
    print("=" * 60)

    # ------------------------------------------------------------
    # 1. Read the initial object position
    # ------------------------------------------------------------

    initial_object = grasping.objects.get(object_id)

    print(f"\nInitial {object_id} position:")
    print(initial_object.position)

    # ------------------------------------------------------------
    # 2. Move above the object
    # ------------------------------------------------------------

    approach_target = (
        initial_object.position[0],
        initial_object.position[1],
        initial_object.position[2] + 0.08,
    )

    print("\n[1/6] Moving above object...")

    result = controller.move_to(
        arm,
        approach_target,
        tolerance=0.03,
        max_steps=500,
    )

    print(
        f"Reached: {result.reached} | "
        f"error={result.position_error:.3f} m"
    )

    if not result.reached:
        print("❌ Could not reach approach position.")
        return

    # ------------------------------------------------------------
    # 3. Move into grasp position
    # ------------------------------------------------------------

    grasp_target = (
        initial_object.position[0],
        initial_object.position[1],
        initial_object.position[2] + 0.02,
    )

    print("\n[2/6] Moving into grasp position...")

    result = controller.move_to(
        arm,
        grasp_target,
        tolerance=0.03,
        max_steps=500,
    )

    print(
        f"Reached: {result.reached} | "
        f"error={result.position_error:.3f} m"
    )

    if not result.reached:
        print("❌ Could not reach grasp position.")
        return

    # ------------------------------------------------------------
    # 4. Grasp
    # ------------------------------------------------------------

    print("\n[3/6] Grasping object...")

    grasp_result = grasping.grasp(
        arm,
        object_id,
    )

    print(f"Grasp success: {grasp_result.success}")
    print(f"Distance: {grasp_result.distance:.3f} m")

    if not grasp_result.success:
        print("❌ Grasp failed.")
        return

    # ------------------------------------------------------------
    # 5. Lift and move to new location
    # ------------------------------------------------------------

    lifted_target = (
        initial_object.position[0],
        initial_object.position[1],
        initial_object.position[2] + 0.15,
    )

    print("\n[4/6] Lifting object...")

    result = controller.move_to(
        arm,
        lifted_target,
        tolerance=0.04,
        max_steps=600,
    )

    print(
        f"Lift reached: {result.reached} | "
        f"error={result.position_error:.3f} m"
    )

    if not result.reached:
        print("❌ Lift failed.")
        grasping.release(arm, object_id)
        return

    # New placement location.
    # We move the plate toward the center-right side of the table.
    place_target = (
        -0.22,
        0.00,
        initial_object.position[2] + 0.15,
    )

    print("\n[5/6] Moving to placement location...")

    result = controller.move_to(
        arm,
        place_target,
        tolerance=0.04,
        max_steps=700,
    )

    print(
        f"Move reached: {result.reached} | "
        f"error={result.position_error:.3f} m"
    )

    if not result.reached:
        print("❌ Could not reach placement location.")
        grasping.release(arm, object_id)
        return

    # ------------------------------------------------------------
    # 6. Lower and release
    # ------------------------------------------------------------

    release_target = (
        -0.22,
        0.00,
        initial_object.position[2] + 0.02,
    )

    print("\nLowering object...")

    result = controller.move_to(
        arm,
        release_target,
        tolerance=0.04,
        max_steps=600,
    )

    print(
        f"Lower reached: {result.reached} | "
        f"error={result.position_error:.3f} m"
    )

    print("\n[6/6] Releasing object...")

    grasping.release(
        arm,
        object_id,
    )

    simulation.step(40)

    final_object = grasping.objects.get(object_id)

    print(f"\nFinal {object_id} position:")
    print(final_object.position)

    # ------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------

    displacement = (
        (
            (final_object.position[0] - initial_object.position[0]) ** 2
            + (final_object.position[1] - initial_object.position[1]) ** 2
            + (final_object.position[2] - initial_object.position[2]) ** 2
        )
        ** 0.5
    )

    print(f"\nObject displacement: {displacement:.3f} m")

    print("\n" + "=" * 60)

    if displacement > 0.05:
        print("✅ PICK + PLACE SUCCESSFUL")
        print("The plate was grasped, transported, and released.")
    else:
        print("❌ PICK + PLACE FAILED")
        print("The plate did not move sufficiently.")

    print("=" * 60)


if __name__ == "__main__":
    main()