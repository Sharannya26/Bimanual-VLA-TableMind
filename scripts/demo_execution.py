"""Demonstrate TABLEMIND task-plan execution."""

from __future__ import annotations

from tablemind.perception.state_builder import SceneStateBuilder
from tablemind.planning.executor import ActionExecutor
from tablemind.planning.planner import DeterministicTablePlanner
from tablemind.simulation.runtime import BimanualTableSimulation


def main() -> None:
    print("TABLEMIND - Milestone 4.2 Action Execution")
    print("=" * 60)

    simulation = BimanualTableSimulation.create()

    builder = SceneStateBuilder(
        simulation.model,
        simulation.data,
    )

    scene = builder.build()

    planner = DeterministicTablePlanner()

    plan = planner.plan_set_table(scene)

    executor = ActionExecutor(simulation)

    # Execute only the first three actions.
    #
    # The complete plan contains grasping actions, but the current
    # table objects are fixed placeholder geometries. We therefore
    # demonstrate controller execution without claiming object
    # transfer.
    actions_to_execute = plan.actions[:3]

    print("\nExecuting first three planned actions:")
    print("-" * 60)

    for action in actions_to_execute:
        print(
            f"{action.action_id:25s} | "
            f"{action.action_type.value:15s} | "
            f"arm={action.arm}"
        )

    report = executor.execute_plan(
        actions_to_execute,
        stop_on_failure=True,
    )

    print("\nExecution results:")
    print("-" * 60)

    for result in report.results:
        status = "SUCCESS" if result.success else "FAILED"

        print(
            f"{status:7s} | "
            f"{result.action_id:25s} | "
            f"{result.message}"
        )

    print("\nExecution summary:")
    print("-" * 60)

    print(
        f"Actions attempted: {len(report.results)}"
    )

    print(
        f"Actions completed: {report.completed_actions}"
    )

    print(
        f"Overall execution success: {report.success}"
    )

    print(
        "\nNote: physical object transfer is not claimed because "
        "the current plate/glass geometries are fixed placeholders."
    )

    print(
        "\nMilestone 4.2 execution validation complete."
    )


if __name__ == "__main__":
    main()