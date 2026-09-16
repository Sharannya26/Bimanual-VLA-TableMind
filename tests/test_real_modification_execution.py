import mujoco

from tablemind.conversation.modification import (
    ModificationRequest,
)
from tablemind.conversation.modification_replanner import (
    ModificationReplanner,
)
from tablemind.control.so101 import SO101Controller
from tablemind.manipulation.execution_control import ExecutionControl
from tablemind.manipulation.grasping import GraspManager
from tablemind.manipulation.synchronized_executor import (
    SynchronizedBimanualExecutor,
)
from tablemind.reasoning.interpreter import TaskInterpreter
from tablemind.reasoning.execution_bridge import (
    ReasoningExecutionBridge,
)
from tablemind.perception.state_builder import SceneStateBuilder
from tablemind.simulation import BimanualTableSimulation


def move_body_up(simulation, body_name, distance):
    """Move a free-jointed MuJoCo body upward."""

    body_id = simulation.model.body(body_name).id

    joint_id = simulation.model.body_jntadr[body_id]

    if joint_id < 0:
        raise ValueError(
            f"Body '{body_name}' does not have a joint."
        )

    qpos_address = simulation.model.jnt_qposadr[joint_id]

    # MuJoCo free joint:
    #
    # x, y, z, qw, qx, qy, qz
    #
    # Therefore +2 is the Z position.
    simulation.data.qpos[qpos_address + 2] += distance


def test_real_mujoco_modification_replans_current_scene():
    # ---------------------------------------------------------
    # 1. CREATE REAL MUJOCO SIMULATION
    # ---------------------------------------------------------

    simulation = BimanualTableSimulation.create()

    # ---------------------------------------------------------
    # 2. CREATE AN INCOMPLETE INITIAL SCENE
    # ---------------------------------------------------------
    #
    # The default scene is already correctly set.
    #
    # Move both plates upward so the original task
    # "Set the table for two" requires action.

    move_body_up(
        simulation,
        "plate_1",
        0.07,
    )

    move_body_up(
        simulation,
        "plate_2",
        0.07,
    )

    mujoco.mj_forward(
        simulation.model,
        simulation.data,
    )

    # ---------------------------------------------------------
    # 3. BUILD CURRENT SCENE STATE
    # ---------------------------------------------------------

    scene_state = SceneStateBuilder(
        simulation.model,
        simulation.data,
    ).build()

    # ---------------------------------------------------------
    # 4. ORIGINAL TASK
    # ---------------------------------------------------------

    interpreter = TaskInterpreter()

    current_request = interpreter.interpret(
        "Set the table for two."
    )

    assert current_request.intent == "set_table"
    assert current_request.quantity == 2

    # ---------------------------------------------------------
    # 5. CONVERSATIONAL MODIFICATION
    # ---------------------------------------------------------

    modification = ModificationRequest(
        raw_instruction="Actually, make it three.",
        intent="set_table",
        quantity=3,
    )

    # ---------------------------------------------------------
    # 6. MODIFICATION-AWARE REPLANNING
    # ---------------------------------------------------------

    modification_replanner = ModificationReplanner()

    result = modification_replanner.replan(
        current_request,
        modification,
        scene_state.objects,
    )

    # ---------------------------------------------------------
    # 7. VERIFY TASK WAS UPDATED
    # ---------------------------------------------------------

    assert result.updated_request.intent == "set_table"
    assert result.updated_request.quantity == 3

    # ---------------------------------------------------------
    # 8. VERIFY SCENE-AWARE REASONING
    # ---------------------------------------------------------

    plate_requirement = (
        result.replanning.reasoning.requirements
        .requirement_for("plate")
    )

    glass_requirement = (
        result.replanning.reasoning.requirements
        .requirement_for("glass")
    )

    assert plate_requirement.required_quantity == 3
    assert glass_requirement.required_quantity == 3

    # ---------------------------------------------------------
    # 9. VERIFY REPLANNING PRODUCED WORK
    # ---------------------------------------------------------

    assert result.needs_action is True
    assert len(result.plan.actions) > 0

    # ---------------------------------------------------------
    # 10. CONVERT PLAN TO EXECUTABLE TASK
    # ---------------------------------------------------------

    bridge = ReasoningExecutionBridge()

    execution_task = bridge.build_task(
        result.plan,
    ).task

    # ---------------------------------------------------------
    # 11. VERIFY ASSIGNMENTS
    # ---------------------------------------------------------
    #
    # At this point the scene only contains the original
    # two plates and two glasses.
    #
    # Therefore the new requirement of three cannot yet
    # be physically satisfied because plate_3/glass_3
    # do not exist.
    #
    # The important proof here is that reasoning detects
    # the changed requirement and produces an executable
    # plan from the latest scene.

    assert len(execution_task.assignments) > 0

    # Every assignment must contain an object ID.
    for assignment in execution_task.assignments:
        assert assignment.object_id

    # ---------------------------------------------------------
    # 12. VERIFY CURRENT SCENE IS STILL THE SOURCE OF TRUTH
    # ---------------------------------------------------------

    remaining_objects = result.remaining_objects

    assert len(remaining_objects) > 0