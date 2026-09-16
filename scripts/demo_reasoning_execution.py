"""End-to-end TABLEMIND reasoning -> bimanual execution demo."""

from __future__ import annotations

import mujoco

from tablemind.manipulation.grasping import GraspManager
from tablemind.manipulation.synchronized_executor import (
    SynchronizedBimanualExecutor,
)
from tablemind.reasoning.context_planner import ContextAwarePlanner
from tablemind.reasoning.execution_bridge import ReasoningExecutionBridge
from tablemind.reasoning.interpreter import TaskInterpreter
from tablemind.simulation import BimanualTableSimulation
from tablemind.perception.state_builder import SceneStateBuilder


def perturb_plate(
    simulation: BimanualTableSimulation,
    object_id: str,
    z_offset: float,
) -> None:
    """Move one freejoint plate vertically to create an incomplete task state."""

    body_id = mujoco.mj_name2id(
        simulation.model,
        mujoco.mjtObj.mjOBJ_BODY,
        object_id,
    )

    if body_id < 0:
        raise KeyError(
            f"MuJoCo model does not contain body {object_id!r}"
        )

    joint_id = simulation.model.body_jntadr[body_id]

    if joint_id < 0:
        raise ValueError(
            f"Object {object_id!r} does not have a freejoint."
        )

    qpos_address = simulation.model.jnt_qposadr[joint_id]

    simulation.data.qpos[qpos_address + 2] += z_offset

    mujoco.mj_forward(
        simulation.model,
        simulation.data,
    )


def main() -> None:
    print("=" * 72)
    print("TABLEMIND - MILESTONE 6.6.10-C")
    print("End-to-End Reasoning -> Bimanual Execution")
    print("=" * 72)

    # ------------------------------------------------------------
    # 1. Create MuJoCo simulation
    # ------------------------------------------------------------

    simulation = BimanualTableSimulation.create()

    print("\n[1/9] SIMULATION")
    print("Dual SO-101 MuJoCo environment created.")
    print(f"Arms: {simulation.arms}")

    # ------------------------------------------------------------
    # 2. Create a deliberately incomplete task state
    #
    # The two plates are moved slightly upward.
    # The glasses remain correctly positioned.
    #
    # This gives the reasoning system:
    #
    #   plates  -> incomplete
    #   glasses -> already complete
    #
    # Therefore the planner should generate exactly two
    # manipulation assignments: plate_1 and plate_2.
    # ------------------------------------------------------------

    perturb_plate(
        simulation,
        "plate_1",
        0.07,
    )

    perturb_plate(
        simulation,
        "plate_2",
        0.07,
    )

    print("\n[2/9] SCENE SETUP")
    print("Created an incomplete table-setting state.")
    print("plate_1 and plate_2 require placement.")
    print("glasses are already correctly positioned.")

    # ------------------------------------------------------------
    # 3. Natural-language instruction
    # ------------------------------------------------------------

    instruction = "Set the table for two."

    interpreter = TaskInterpreter()

    request = interpreter.interpret(
        instruction,
    )

    print("\n[3/9] NATURAL-LANGUAGE UNDERSTANDING")
    print(f"Human: {instruction}")
    print(f"Intent: {request.intent}")
    print(f"Objects: {', '.join(request.object_types)}")
    print(f"Quantity: {request.quantity}")

    # ------------------------------------------------------------
    # 4. Build structured scene state
    # ------------------------------------------------------------

    state_builder = SceneStateBuilder(
        simulation.model,
        simulation.data,
    )

    scene_state = state_builder.build()

    print("\n[4/9] SCENE UNDERSTANDING")
    print(
        f"Detected objects: "
        f"{len(scene_state.objects.objects)}"
    )

    for obj in scene_state.objects.objects:
        position = tuple(
            round(value, 3)
            for value in obj.position
        )

        print(
            f"  {obj.object_id:<10}"
            f"{obj.object_type:<8}"
            f"position={position}"
        )

    # ------------------------------------------------------------
    # 5. Context-aware reasoning
    # ------------------------------------------------------------

    planner = ContextAwarePlanner()

    reasoning = planner.context_reasoner.reason(
        request,
        scene_state.objects,
    )

    print("\n[5/9] CONTEXT-AWARE REASONING")

    plate_requirement = (
        reasoning.requirements.requirement_for("plate")
    )

    glass_requirement = (
        reasoning.requirements.requirement_for("glass")
    )

    print(
        f"Required quantities: "
        f"plate={plate_requirement.required_quantity}, "
        f"glass={glass_requirement.required_quantity}"
    )

    print(
        f"Completed objects: "
        f"{len(reasoning.completed_objects)}"
    )

    print(
        f"Incomplete objects: "
        f"{len(reasoning.incomplete_objects)}"
    )

    print(
        f"Missing objects: "
        f"{reasoning.total_missing}"
    )

    if reasoning.completed_objects:
        print("\nAlready complete:")

        for object_id in reasoning.completed_objects:
            print(f"  ✓ {object_id}")

    if reasoning.incomplete_objects:
        print("\nObjects requiring action:")

        for object_id in reasoning.incomplete_objects:
            print(f"  → {object_id}")

    # ------------------------------------------------------------
    # 6. Convert reasoning into a robot task plan
    # ------------------------------------------------------------

    plan = planner.plan(
        request,
        scene_state.objects,
    )

    print("\n[6/9] TASK PLANNING")
    print(f"Task ID: {plan.task.task_id}")
    print(f"Description: {plan.task.description}")
    print(f"Actions generated: {plan.action_count}")

    for action in plan.actions:
        target = None

        if action.target_position is not None:
            target = tuple(
                round(value, 3)
                for value in action.target_position
            )

        print(
            f"  {action.action_type.value:<14}"
            f"arm={action.arm:<6}"
            f"object={action.object_id:<10}"
            f"target={target}"
        )

    # ------------------------------------------------------------
    # 7. Bridge reasoning plan -> bimanual manipulation task
    # ------------------------------------------------------------

    bridge = ReasoningExecutionBridge()

    bridge_result = bridge.build_task(
        plan,
    )

    coordinated_task = bridge_result.task

    print("\n[7/9] REASONING -> EXECUTION BRIDGE")

    print(
        f"Bimanual assignments: "
        f"{bridge_result.assignment_count}"
    )

    for assignment in coordinated_task.assignments:
        print(
            f"\n  {assignment.arm.upper()} ARM"
            f" -> {assignment.object_id}"
        )

        print(
            f"    approach = "
            f"{assignment.approach_position}"
        )

        print(
            f"    grasp    = "
            f"{assignment.grasp_position}"
        )

        print(
            f"    lift     = "
            f"{assignment.lift_position}"
        )

        print(
            f"    place    = "
            f"{assignment.place_position}"
        )

    # ------------------------------------------------------------
    # 8. Execute synchronized bimanual manipulation
    # ------------------------------------------------------------

    grasping = GraspManager(
        simulation,
    )

    executor = SynchronizedBimanualExecutor(
        simulation=simulation,
        grasping=grasping,
    )

    print("\n[8/9] SYNCHRONIZED BIMANUAL EXECUTION")

    result = executor.execute(
        coordinated_task,
    )

    print("\nExecution result:")
    print(f"Success: {result.success}")

    print(
        f"Stages completed: "
        f"{result.completed_stages}/"
        f"{result.total_stages}"
    )

    print(f"Message: {result.message}")

    # ------------------------------------------------------------
    # 9. Final result
    # ------------------------------------------------------------

    print("\n[9/9] TABLEMIND RESULT")
    print("=" * 72)

    if result.success:
        print("TABLEMIND END-TO-END PIPELINE SUCCESSFUL")
        print()
        print(
            "Natural language"
            " -> Scene understanding"
            " -> Context reasoning"
            " -> Task planning"
            " -> Execution bridge"
            " -> Bimanual manipulation"
        )
    else:
        print("TABLEMIND END-TO-END PIPELINE FAILED")

    print("=" * 72)


if __name__ == "__main__":
    main()