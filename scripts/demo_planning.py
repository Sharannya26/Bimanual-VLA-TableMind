"""Demonstrate TABLEMIND deterministic task planning."""

from __future__ import annotations

from tablemind.perception.state_builder import SceneStateBuilder
from tablemind.planning.planner import DeterministicTablePlanner
from tablemind.simulation.runtime import BimanualTableSimulation


def main() -> None:
    print("TABLEMIND - Milestone 4.1 Task Planning")
    print("=" * 60)

    simulation = BimanualTableSimulation.create()

    builder = SceneStateBuilder(
        simulation.model,
        simulation.data,
    )

    scene = builder.build()

    planner = DeterministicTablePlanner()

    plan = planner.plan_set_table(scene)

    print("\nTask:")
    print("-" * 60)

    print(plan.task.description)

    print("\nGenerated action plan:")
    print("-" * 60)

    for number, action in enumerate(plan.actions, start=1):
        print(
            f"{number:02d}. "
            f"{action.action_type.value:15s} | "
            f"arm={action.arm:5s} | "
            f"object={action.object_id!s:10s} | "
            f"target={action.target_position}"
        )

    print("\nPlan summary:")
    print("-" * 60)

    print(f"Total actions: {plan.action_count}")

    arms = sorted(
        {
            action.arm
            for action in plan.actions
        }
    )

    print(f"Arms involved: {', '.join(arms)}")

    print("\nMilestone 4.1 planning validation complete.")


if __name__ == "__main__":
    main()