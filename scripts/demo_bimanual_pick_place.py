"""Demonstrate independent bimanual pick-and-place."""

from __future__ import annotations

from tablemind.control.so101 import SO101Controller
from tablemind.manipulation.grasping import GraspManager
from tablemind.simulation import BimanualTableSimulation


def move_checked(
    controller: SO101Controller,
    arm: str,
    target: tuple[float, float, float],
    label: str,
    *,
    tolerance: float = 0.04,
    max_steps: int = 700,
) -> bool:
    """Move one arm and report the result."""

    result = controller.move_to(
        arm,
        target,
        tolerance=tolerance,
        max_steps=max_steps,
    )

    print(
        f"{label} [{arm}] | "
        f"reached={result.reached} | "
        f"error={result.position_error:.3f} m"
    )

    return result.reached


def main() -> None:
    simulation = BimanualTableSimulation.create()

    controller = SO101Controller(simulation)
    grasping = GraspManager(simulation, controller)

    assignments = (
        ("left", "plate_1"),
        ("right", "plate_2"),
    )

    print("=" * 64)
    print("TABLEMIND - Milestone 5.4 Bimanual Pick + Place")
    print("=" * 64)

    initial_positions = {}

    for arm, object_id in assignments:
        obj = grasping.objects.get(object_id)
        initial_positions[object_id] = obj.position

        print(
            f"\n{arm.upper()} ARM -> {object_id} "
            f"initial position: {obj.position}"
        )

    # ------------------------------------------------------------
    # 1. APPROACH
    # ------------------------------------------------------------

    print("\n[1/7] Approaching both objects...")

    for arm, object_id in assignments:
        obj = grasping.objects.get(object_id)

        target = (
            obj.position[0],
            obj.position[1],
            obj.position[2] + 0.08,
        )

        if not move_checked(
            controller,
            arm,
            target,
            "Approach",
        ):
            print(f"❌ {arm} arm could not reach {object_id}.")
            return

    # ------------------------------------------------------------
    # 2. MOVE INTO GRASP POSITION
    # ------------------------------------------------------------

    print("\n[2/7] Moving both arms into grasp position...")

    for arm, object_id in assignments:
        obj = grasping.objects.get(object_id)

        target = (
            obj.position[0],
            obj.position[1],
            obj.position[2] + 0.02,
        )

        if not move_checked(
            controller,
            arm,
            target,
            "Grasp position",
        ):
            print(f"❌ {arm} arm could not reach {object_id}.")
            return

    # ------------------------------------------------------------
    # 3. GRASP
    # ------------------------------------------------------------

    print("\n[3/7] Grasping both objects...")

    for arm, object_id in assignments:
        result = grasping.grasp(
            arm,
            object_id,
        )

        print(
            f"Grasp [{arm}] -> {object_id} | "
            f"success={result.success} | "
            f"distance={result.distance:.3f} m"
        )

        if not result.success:
            print(f"❌ {arm} arm failed to grasp {object_id}.")

            # Release anything already grasped.
            for previous_arm, previous_object in assignments:
                if previous_arm == arm:
                    break

                grasping.release(
                    previous_arm,
                    previous_object,
                )

            return

    # ------------------------------------------------------------
    # 4. LIFT
    # ------------------------------------------------------------

    print("\n[4/7] Lifting both objects...")

    for arm, object_id in assignments:
        initial = initial_positions[object_id]

        target = (
            initial[0],
            initial[1],
            initial[2] + 0.15,
        )

        if not move_checked(
            controller,
            arm,
            target,
            "Lift",
            tolerance=0.04,
            max_steps=700,
        ):
            print(f"❌ {arm} arm failed to lift {object_id}.")

            grasping.release(
                arm,
                object_id,
            )

            return

    # ------------------------------------------------------------
    # 5. TRANSPORT
    # ------------------------------------------------------------

    print("\n[5/7] Transporting both objects...")

    # Move each plate inward by a modest amount.
    # These targets are deliberately conservative because
    # the SO-101 workspace is limited.

    placement_targets = {
        "plate_1": (
            -0.22,
            0.00,
            initial_positions["plate_1"][2] + 0.15,
        ),
        "plate_2": (
            0.22,
            0.00,
            initial_positions["plate_2"][2] + 0.15,
        ),
    }

    for arm, object_id in assignments:
        target = placement_targets[object_id]

        if not move_checked(
            controller,
            arm,
            target,
            "Transport",
            tolerance=0.04,
            max_steps=700,
        ):
            print(
                f"❌ {arm} arm could not transport "
                f"{object_id}."
            )

            grasping.release(
                arm,
                object_id,
            )

            return

    # ------------------------------------------------------------
    # 6. LOWER + RELEASE
    # ------------------------------------------------------------

    print("\n[6/7] Lowering both objects...")

    for arm, object_id in assignments:
        initial = initial_positions[object_id]
        placement = placement_targets[object_id]

        target = (
            placement[0],
            placement[1],
            initial[2] + 0.02,
        )

        if not move_checked(
            controller,
            arm,
            target,
            "Lower",
            tolerance=0.04,
            max_steps=700,
        ):
            print(
                f"⚠️ {arm} arm could not perfectly "
                f"reach release height."
            )

    print("\nReleasing both objects...")

    for arm, object_id in assignments:
        grasping.release(
            arm,
            object_id,
        )

    simulation.step(40)

    # ------------------------------------------------------------
    # 7. VERIFY
    # ------------------------------------------------------------

    print("\n[7/7] Verifying both placements...")

    successful_objects = 0

    for arm, object_id in assignments:
        initial = initial_positions[object_id]
        final = grasping.objects.get(object_id).position

        displacement = (
            (
                (final[0] - initial[0]) ** 2
                + (final[1] - initial[1]) ** 2
                + (final[2] - initial[2]) ** 2
            )
            ** 0.5
        )

        print(f"\n{object_id}:")
        print(f"  Initial: {initial}")
        print(f"  Final:   {final}")
        print(f"  Displacement: {displacement:.3f} m")

        if displacement > 0.04:
            successful_objects += 1
            print("  ✅ Object successfully moved.")
        else:
            print("  ❌ Object did not move sufficiently.")

    print("\n" + "=" * 64)

    if successful_objects == 2:
        print("✅ BILATERAL PICK + PLACE SUCCESSFUL")
        print("Both SO-101 arms manipulated objects successfully.")
    elif successful_objects == 1:
        print("⚠️ PARTIAL BILATERAL SUCCESS")
        print(f"{successful_objects}/2 objects were moved successfully.")
    else:
        print("❌ BILATERAL PICK + PLACE FAILED")
        print("Neither object moved sufficiently.")

    print("=" * 64)


if __name__ == "__main__":
    main()