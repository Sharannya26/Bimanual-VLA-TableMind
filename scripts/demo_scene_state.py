"""Demonstrate TABLEMIND structured scene state."""

from __future__ import annotations

import json

from tablemind.perception.state_builder import SceneStateBuilder
from tablemind.simulation.runtime import BimanualTableSimulation


def main() -> None:
    print("TABLEMIND - Milestone 3.4 Structured Scene State")
    print("=" * 60)

    simulation = BimanualTableSimulation.create()

    builder = SceneStateBuilder(
        simulation.model,
        simulation.data,
    )

    state = builder.build()

    print("\nObjects:")
    print("-" * 60)

    for obj in state.objects.objects:
        print(
            f"{obj.object_id:10s} | "
            f"type={obj.object_type:6s} | "
            f"position={obj.position}"
        )

    print("\nRobots:")
    print("-" * 60)

    for robot in state.robots:
        print(
            f"{robot.arm_id:5s} | "
            f"EE={robot.end_effector_position} | "
            f"gripper={robot.gripper_position:.4f}"
        )

    print("\nSimulation time:")
    print("-" * 60)

    print(
        f"{state.simulation_time:.3f} seconds"
    )

    print("\nJSON representation:")
    print("-" * 60)

    print(
        json.dumps(
            state.to_dict(),
            indent=2,
        )
    )

    print("\nMilestone 3.4 scene-state validation complete.")


if __name__ == "__main__":
    main()