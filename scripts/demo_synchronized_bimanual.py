"""Demonstrate synchronized bimanual manipulation."""

from __future__ import annotations

from tablemind.manipulation.coordination import (
    BimanualCoordinator,
)
from tablemind.manipulation.synchronized_executor import (
    SynchronizedBimanualExecutor,
)
from tablemind.simulation import BimanualTableSimulation


def main() -> None:
    simulation = BimanualTableSimulation.create()

    coordinator = BimanualCoordinator()

    task = coordinator.create_two_arm_task(
        name="synchronized_plate_placement",
        left_object="plate_1",
        right_object="plate_2",
    )

    executor = SynchronizedBimanualExecutor(
        simulation
    )

    print("=" * 64)
    print("TABLEMIND - Milestone 5.4.4")
    print("Synchronized Bimanual Execution")
    print("=" * 64)

    print("\nCoordinated task:")
    print(f"  {task.name}")

    print("\nAssignments:")

    for assignment in task.assignments:
        print(
            f"  {assignment.arm.upper()} ARM "
            f"-> {assignment.object_id}"
        )

    print("\nExecuting synchronized manipulation...")

    result = executor.execute(task)

    print("\n" + "=" * 64)
    print("EXECUTION RESULT")
    print("=" * 64)

    print(
        f"Success: {result.success}"
    )

    print(
        f"Stages completed: "
        f"{result.completed_stages}/"
        f"{result.total_stages}"
    )

    print(
        f"Message: {result.message}"
    )

    print("\n" + "=" * 64)

    if result.success:
        print("✅ SYNCHRONIZED BIMANUAL EXECUTION SUCCESSFUL")
        print(
            "Both SO-101 arms completed the coordinated "
            "manipulation task."
        )
    else:
        print("❌ SYNCHRONIZED EXECUTION FAILED")

    print("=" * 64)


if __name__ == "__main__":
    main()