"""Demonstrate dynamic TABLEMIND manipulation objects."""

from __future__ import annotations

import mujoco
import mujoco.viewer
from tablemind.manipulation.objects import DynamicObjectManager
from tablemind.simulation.runtime import BimanualTableSimulation


def main() -> None:
    print()
    print("TABLEMIND - Milestone 5.1 Dynamic Objects")
    print("=" * 48)
    print()

    simulation = BimanualTableSimulation.create()

    manager = DynamicObjectManager(
        simulation.model,
        simulation.data,
    )

    print("Initial object state:")
    print("-" * 70)

    for obj in manager.all_objects():
        print(
            f"{obj.object_id:10s} | "
            f"type={obj.object_type:6s} | "
            f"position=("
            f"{obj.position[0]:+.3f}, "
            f"{obj.position[1]:+.3f}, "
            f"{obj.position[2]:+.3f})"
        )

    print()
    print("Settling simulation...")
    print()

    simulation.step(500)

    print("Object state after physics settling:")
    print("-" * 70)

    for obj in manager.all_objects():
        print(
            f"{obj.object_id:10s} | "
            f"position=("
            f"{obj.position[0]:+.3f}, "
            f"{obj.position[1]:+.3f}, "
            f"{obj.position[2]:+.3f}) | "
            f"velocity=("
            f"{obj.linear_velocity[0]:+.4f}, "
            f"{obj.linear_velocity[1]:+.4f}, "
            f"{obj.linear_velocity[2]:+.4f})"
        )

    print()
    print("Launching visual MuJoCo simulation...")
    print("Close the window to finish.")
    print()

    viewer = mujoco.viewer.launch_passive(
        simulation.model,
        simulation.data,
    )

    try:
        while viewer.is_running():
            simulation.step(5)
            viewer.sync()
    finally:
        viewer.close()


if __name__ == "__main__":
    main()