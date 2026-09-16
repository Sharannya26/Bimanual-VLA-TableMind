"""Run a TABLEMIND coordinated bimanual task."""

from __future__ import annotations

from tablemind.manipulation.coordination import (
    BimanualCoordinator,
)
from tablemind.manipulation.coordinator_executor import (
    CoordinatedTaskExecutor,
)
from tablemind.simulation import BimanualTableSimulation


def main() -> None:
    simulation = BimanualTableSimulation.create()

    coordinator = BimanualCoordinator()

    task = coordinator.create_two_arm_task(
        name="coordinated_plate_placement",
        left_object="plate_1",
        right_object="plate_2",
    )

    executor = CoordinatedTaskExecutor(
        simulation
    )

    print("=" * 64)
    print("TABLEMIND - Milestone 5.4.3")
    print("Coordinated Bimanual Task")
    print("=" * 64)

    print(f"\nTask: {task.name}")

    for assignment in task.assignments:
        print(
            f"  {assignment.arm.upper()} ARM "
            f"-> {assignment.object_id}"
        )

    print("\nExecuting coordinated task...")

    result = executor.execute(task)

    print("\n" + "=" * 64)
    print("EXECUTION RESULT")
    print("=" * 64)

    print(
        f"Success: {result.success}"
    )

    print(
        f"Assignments completed: "
        f"{result.completed_assignments}/"
        f"{result.total_assignments}"
    )

    print(
        f"Message: {result.message}"
    )

    print("=" * 64)


if __name__ == "__main__":
    main()