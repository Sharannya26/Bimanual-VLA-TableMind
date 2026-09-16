import mujoco

from tablemind.control.so101 import SO101Controller
from tablemind.manipulation.execution_control import ExecutionControl
from tablemind.manipulation.grasping import GraspManager
from tablemind.manipulation.synchronized_executor import (
    SynchronizedBimanualExecutor,
)
from tablemind.reasoning.context_planner import ContextAwarePlanner
from tablemind.reasoning.execution_bridge import ReasoningExecutionBridge
from tablemind.reasoning.interpreter import TaskInterpreter
from tablemind.perception.state_builder import SceneStateBuilder
from tablemind.simulation import BimanualTableSimulation


class StopDuringExecutionController(SO101Controller):
    """Request a stop once the executor reaches the lift stage."""

    def __init__(self, simulation, control):
        super().__init__(simulation)
        self.control = control
        self.move_calls = 0

    def move_to(self, *args, **kwargs):
        self.move_calls += 1

        # Two approach moves + two grasp-position moves
        # happen before the lift stage.
        #
        # Requesting stop on the 5th movement means:
        # - both objects have been grasped
        # - the first lift movement is interrupted
        if self.move_calls == 5:
            self.control.request_stop()

        return super().move_to(*args, **kwargs)


def move_body_up(simulation, body_name, distance):
    """Move a free-jointed MuJoCo body upward."""

    body_id = simulation.model.body(body_name).id

    # Get the first joint belonging to this body.
    joint_id = simulation.model.body_jntadr[body_id]

    if joint_id < 0:
        raise ValueError(
            f"Body '{body_name}' does not have a joint."
        )

    # Get the position address of this joint in qpos.
    qpos_address = simulation.model.jnt_qposadr[joint_id]

    # A MuJoCo free joint stores:
    #
    # x, y, z, qw, qx, qy, qz
    #
    # Therefore index +2 is the Z position.
    simulation.data.qpos[qpos_address + 2] += distance


def test_real_mujoco_bimanual_execution_can_be_interrupted_safely():
    # ---------------------------------------------------------
    # 1. CREATE REAL MUJOCO SIMULATION
    # ---------------------------------------------------------
    simulation = BimanualTableSimulation.create()

    # ---------------------------------------------------------
    # 2. MAKE BOTH PLATES INCOMPLETE
    # ---------------------------------------------------------
    #
    # The default scene is already correctly set.
    # Move both plates upward by 7 cm so that the planner
    # has real work to perform.
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

    # Recompute MuJoCo state after changing qpos.
    mujoco.mj_forward(
        simulation.model,
        simulation.data,
    )

    # ---------------------------------------------------------
    # 3. EXECUTION CONTROL
    # ---------------------------------------------------------
    control = ExecutionControl()

    # ---------------------------------------------------------
    # 4. CONTROLLER
    # ---------------------------------------------------------
    controller = StopDuringExecutionController(
        simulation,
        control,
    )

    # ---------------------------------------------------------
    # 5. GRASPING
    # ---------------------------------------------------------
    grasping = GraspManager(
        simulation,
        controller,
    )

    # ---------------------------------------------------------
    # 6. SYNCHRONIZED EXECUTOR
    # ---------------------------------------------------------
    executor = SynchronizedBimanualExecutor(
        simulation,
        controller=controller,
        grasping=grasping,
        control=control,
    )

    # ---------------------------------------------------------
    # 7. BUILD CURRENT SCENE STATE
    # ---------------------------------------------------------
    scene_state = SceneStateBuilder(
        simulation.model,
        simulation.data,
    ).build()

    # ---------------------------------------------------------
    # 8. NATURAL-LANGUAGE REQUEST
    # ---------------------------------------------------------
    interpreter = TaskInterpreter()

    request = interpreter.interpret(
        "Set the table for two."
    )

    # ---------------------------------------------------------
    # 9. CONTEXT-AWARE PLANNING
    # ---------------------------------------------------------
    planner = ContextAwarePlanner()

    plan = planner.plan(
        request,
        scene_state.objects,
    )

    # The plates were deliberately moved out of their
    # expected positions, so the planner should generate
    # executable actions.
    assert len(plan.actions) > 0

    # ---------------------------------------------------------
    # 10. REASONING -> EXECUTION BRIDGE
    # ---------------------------------------------------------
    bridge = ReasoningExecutionBridge()

    result = bridge.build_task(plan)

    task = result.task

    # We expect one assignment for each plate.
    assert len(task.assignments) == 2

    # ---------------------------------------------------------
    # 11. EXECUTE WITH FORCED INTERRUPTION
    # ---------------------------------------------------------
    execution_result = executor.execute(task)

    # ---------------------------------------------------------
    # 12. VERIFY SAFE INTERRUPTION
    # ---------------------------------------------------------

    # Execution should NOT report success because we
    # deliberately requested a stop during execution.
    assert execution_result.success is False

    # A full synchronized execution contains 7 stages.
    # Since we interrupted it, fewer than 7 stages
    # should have completed.
    assert execution_result.completed_stages < 7

    # The executor should explicitly report interruption.
    assert "interrupted" in (
        execution_result.message.lower()
    )

    # The stop signal should remain visible after execution.
    assert control.stop_requested is True