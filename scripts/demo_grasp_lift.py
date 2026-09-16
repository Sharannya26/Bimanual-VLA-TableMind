"""Demonstrate grasping and lifting a dynamic dinnerware object."""

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
    print("TABLEMIND - Milestone 5.2.2 Grasp + Lift")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Read initial object position
    # ---------------------------------------------------------

    initial_object = grasping.objects.get(object_id)

    print(
        f"\nInitial {object_id} position: "
        f"{initial_object.position}"
    )

    # ---------------------------------------------------------
    # 2. Move gripper above the object
    # ---------------------------------------------------------

    target = (
        initial_object.position[0],
        initial_object.position[1],
        initial_object.position[2] + 0.08,
    )

    print("\nMoving gripper above object...")

    result = controller.move_to(
        arm,
        target,
        tolerance=0.03,
        max_steps=500,
    )

    print(
        f"Reached: {result.reached} | "
        f"error={result.position_error:.3f} m"
    )

    # ---------------------------------------------------------
    # 3. Move closer to the object
    # ---------------------------------------------------------

    grasp_target = (
        initial_object.position[0],
        initial_object.position[1],
        initial_object.position[2] + 0.02,
    )

    print("\nMoving into grasp position...")

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

    # ---------------------------------------------------------
    # 4. Attempt grasp
    # ---------------------------------------------------------

    print("\nAttempting grasp...")

    grasp_result = grasping.grasp(
        arm,
        object_id,
    )

    print(
        f"Grasp success: {grasp_result.success}"
    )

    print(
        f"Grasp distance: "
        f"{grasp_result.distance:.3f} m"
    )

    print(
        f"Reason: {grasp_result.reason}"
    )

    if not grasp_result.success:
        print("\n❌ Grasp failed.")
        return

    # ---------------------------------------------------------
    # 5. Lift the object
    # ---------------------------------------------------------

    lifted_target = (
        initial_object.position[0],
        initial_object.position[1],
        initial_object.position[2] + 0.15,
    )

    print("\nLifting object...")

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

    # Allow MuJoCo to settle.
    simulation.step(20)

    # ---------------------------------------------------------
    # 6. Check object position
    # ---------------------------------------------------------

    lifted_object = grasping.objects.get(object_id)

    print(
        f"\nAfter lift {object_id} position: "
        f"{lifted_object.position}"
    )

    vertical_change = (
        lifted_object.position[2]
        - initial_object.position[2]
    )

    print(
        f"Object vertical movement: "
        f"{vertical_change:.3f} m"
    )

    # ---------------------------------------------------------
    # 7. Release
    # ---------------------------------------------------------

    print("\nReleasing object...")

    grasping.release(
        arm,
        object_id,
    )

    simulation.step(30)

    released_object = grasping.objects.get(object_id)

    print(
        f"After release position: "
        f"{released_object.position}"
    )

    print("\n" + "=" * 60)

    if vertical_change > 0.05:
        print("✅ GRASP + LIFT SUCCESSFUL")
        print("The object followed the robot during the lift.")
    else:
        print("❌ GRASP + LIFT FAILED")
        print("The object did not move sufficiently with the gripper.")

    print("=" * 60)


if __name__ == "__main__":
    main()