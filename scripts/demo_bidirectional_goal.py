"""
TABLEMIND — Milestone 9.5 Bidirectional Goal Modification Demo

Demonstrates:

    1 setting
        ↓
    "Actually, make it 2."
        ↓
    physically restore setting #2
        ↓
    "Actually, one guest isn't coming."
        ↓
    physically remove setting #2
        ↓
    back to 1 setting

This demo intentionally does NOT modify the existing 9.4 closed-loop
pipeline or demo.

It uses the real MuJoCo simulation, SO-101 controllers, GraspManager,
GoalAdditionExecutor, PhysicalRemovalExecutor, and GoalReconciler.
"""

from __future__ import annotations

import time

import mujoco

from tablemind.control.so101 import SO101Controller
from tablemind.conversation.goal_modification_pipeline import (
    BidirectionalGoalModificationPipeline,
)
from tablemind.manipulation.goal_addition import GoalAdditionExecutor
from tablemind.manipulation.grasping import GraspManager
from tablemind.manipulation.removal import PhysicalRemovalExecutor
from tablemind.perception.scene import SceneObject
from tablemind.reasoning.goal_reconciliation import GoalReconciler
from tablemind.reasoning.task_request import TaskRequest
from tablemind.simulation.runtime import BimanualTableSimulation


# ============================================================================
# Canonical TABLEMIND 9.5 place-setting geometry
# ============================================================================

PLATE_SLOT_1 = (-0.18, -0.04, 0.772)
PLATE_SLOT_2 = (0.18, -0.04, 0.772)

GLASS_SLOT_1 = (-0.40, -0.14, 0.835)
GLASS_SLOT_2 = (0.40, -0.14, 0.835)


# ============================================================================
# MuJoCo helpers
# ============================================================================


def get_free_body_position(
    simulation: BimanualTableSimulation,
    body_name: str,
) -> tuple[float, float, float]:
    """Return the XYZ position of a freejoint body."""

    body_id = mujoco.mj_name2id(
        simulation.model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )

    if body_id < 0:
        raise RuntimeError(
            f"Could not find body '{body_name}'."
        )

    joint_id = simulation.model.body_jntadr[body_id]

    if joint_id < 0:
        raise RuntimeError(
            f"Body '{body_name}' does not have a joint."
        )

    qpos_address = simulation.model.jnt_qposadr[joint_id]

    return tuple(
        float(value)
        for value in simulation.data.qpos[
            qpos_address : qpos_address + 3
        ]
    )


def set_free_body_position(
    simulation: BimanualTableSimulation,
    body_name: str,
    position: tuple[float, float, float],
) -> None:
    """Set the XYZ position of a freejoint body."""

    body_id = mujoco.mj_name2id(
        simulation.model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )

    if body_id < 0:
        raise RuntimeError(
            f"Could not find body '{body_name}'."
        )

    joint_id = simulation.model.body_jntadr[body_id]

    if joint_id < 0:
        raise RuntimeError(
            f"Body '{body_name}' does not have a joint."
        )

    qpos_address = simulation.model.jnt_qposadr[joint_id]

    simulation.data.qpos[
        qpos_address : qpos_address + 3
    ] = position


# ============================================================================
# Scene adapter
# ============================================================================


class MuJoCoSceneAdapter:
    """
    Small scene adapter used only by the 9.5 demo.

    It converts the actual MuJoCo object positions into the minimal scene
    interface required by GoalReconciler and the conversational pipeline.

    This avoids modifying the established perception/vision infrastructure.
    """

    OBJECTS = (
        ("plate_1", "plate"),
        ("plate_2", "plate"),
        ("glass_1", "glass"),
        ("glass_2", "glass"),
    )

    def __init__(
        self,
        simulation: BimanualTableSimulation,
    ) -> None:
        self.simulation = simulation

    def snapshot(self):
        objects = []

        for object_id, object_type in self.OBJECTS:
            position = get_free_body_position(
                self.simulation,
                object_id,
            )

            objects.append(
                SceneObject(
                    object_id=object_id,
                    object_type=object_type,
                    position=position,
                )
            )

        return _SceneSnapshot(tuple(objects))


class _SceneSnapshot:
    """
    Minimal scene interface required by GoalReconciler and the 9.5 pipeline.
    """

    def __init__(
        self,
        objects: tuple[SceneObject, ...],
    ) -> None:
        self.objects = objects

    def by_type(
        self,
        object_type: str,
    ) -> tuple[SceneObject, ...]:
        return tuple(
            obj
            for obj in self.objects
            if obj.object_type == object_type
        )

    def get(
        self,
        object_id: str,
    ) -> SceneObject | None:
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj

        return None


# ============================================================================
# Verification
# ============================================================================


def distance(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    """Euclidean distance between two 3D positions."""

    return (
        (first[0] - second[0]) ** 2
        + (first[1] - second[1]) ** 2
        + (first[2] - second[2]) ** 2
    ) ** 0.5


def verify_object(
    simulation: BimanualTableSimulation,
    object_id: str,
    expected_position: tuple[float, float, float],
    tolerance: float = 0.065,
) -> bool:
    """Verify one object's MuJoCo position."""

    actual = get_free_body_position(
        simulation,
        object_id,
    )

    error = distance(
        actual,
        expected_position,
    )

    passed = error <= tolerance

    print(
        f"  {object_id:<8} "
        f"{'PASS' if passed else 'FAIL'} "
        f"error={error:.4f} m"
    )

    print(
        f"      expected = "
        f"({expected_position[0]:.3f}, "
        f"{expected_position[1]:.3f}, "
        f"{expected_position[2]:.3f})"
    )

    print(
        f"      actual   = "
        f"({actual[0]:.3f}, "
        f"{actual[1]:.3f}, "
        f"{actual[2]:.3f})"
    )

    return passed


def verify_one_setting(
    simulation: BimanualTableSimulation,
) -> bool:
    """Verify place setting #1."""

    print()
    print("VERIFYING ONE PLACE SETTING")
    print("-" * 60)

    plate_ok = verify_object(
        simulation,
        "plate_1",
        PLATE_SLOT_1,
    )

    glass_ok = verify_object(
        simulation,
        "glass_1",
        GLASS_SLOT_1,
    )

    return plate_ok and glass_ok


def verify_two_settings(
    simulation: BimanualTableSimulation,
) -> bool:
    """Verify both active place settings."""

    print()
    print("VERIFYING TWO PLACE SETTINGS")
    print("-" * 60)

    setting_1_plate = verify_object(
        simulation,
        "plate_1",
        PLATE_SLOT_1,
    )

    setting_1_glass = verify_object(
        simulation,
        "glass_1",
        GLASS_SLOT_1,
    )

    setting_2_plate = verify_object(
        simulation,
        "plate_2",
        PLATE_SLOT_2,
    )

    setting_2_glass = verify_object(
        simulation,
        "glass_2",
        GLASS_SLOT_2,
    )

    return (
        setting_1_plate
        and setting_1_glass
        and setting_2_plate
        and setting_2_glass
    )


def verify_removed_setting(
    simulation: BimanualTableSimulation,
) -> bool:
    """
    Verify setting #2 has left its active slot.

    The removal executor stages objects at y=0.32.
    """

    print()
    print("VERIFYING SETTING #2 REMOVAL")
    print("-" * 60)

    plate_position = get_free_body_position(
        simulation,
        "plate_2",
    )

    glass_position = get_free_body_position(
        simulation,
        "glass_2",
    )

    plate_removed = (
        distance(
            plate_position,
            PLATE_SLOT_2,
        )
        > 0.10
    )

    glass_removed = (
        distance(
            glass_position,
            GLASS_SLOT_2,
        )
        > 0.10
    )

    print(
        f"  plate_2  "
        f"{'PASS' if plate_removed else 'FAIL'} "
        f"no longer in active slot"
    )

    print(
        f"  glass_2  "
        f"{'PASS' if glass_removed else 'FAIL'} "
        f"no longer in active slot"
    )

    return plate_removed and glass_removed


# ============================================================================
# Task requests
# ============================================================================


def make_request(
    quantity: int,
) -> TaskRequest:
    """Create the conversational task state."""

    return TaskRequest(
        raw_instruction=f"Set the table for {quantity}.",
        intent="set_table",
        object_types=("plate", "glass"),
        quantity=quantity,
        constraints=(),
    )


# ============================================================================
# Scene preparation
# ============================================================================


def prepare_demo_scene(
    simulation: BimanualTableSimulation,
) -> None:
    """
    Prepare the proven 9.4 workspace.

    Both plate objects are placed at the canonical active slots.

    The first phase intentionally moves plate_2 away from its active
    slot so that the initial physical goal contains exactly one complete
    place setting.
    """

    # Keep plate_1 active.
    set_free_body_position(
        simulation,
        "plate_1",
        PLATE_SLOT_1,
    )

    # Move plate_2 back to its original source position.
    #
    # This is the object that the robot will later pick when the user
    # increases the goal from 1 → 2.
    set_free_body_position(
        simulation,
        "plate_2",
        (0.28, 0.00, 0.772),
    )

    # Glass_1 remains active.
    set_free_body_position(
        simulation,
        "glass_1",
        GLASS_SLOT_1,
    )

    # Glass_2 remains available at its source position.
    set_free_body_position(
        simulation,
        "glass_2",
        (0.40, -0.14, 0.835),
    )

    mujoco.mj_forward(
        simulation.model,
        simulation.data,
    )

    simulation.step(20)


# ============================================================================
# Main demo
# ============================================================================


def main() -> None:
    print()
    print("=" * 72)
    print("TABLEMIND — MILESTONE 9.5")
    print("BIDIRECTIONAL CONVERSATIONAL GOAL MODIFICATION")
    print("=" * 72)
    print()

    print("Creating real MuJoCo simulation...")
    simulation = BimanualTableSimulation.create()

    controller = SO101Controller(
        simulation,
    )

    grasp_manager = GraspManager(
        simulation,
        controller,
    )

    addition_executor = GoalAdditionExecutor(
        simulation=simulation,
        controller=controller,
        grasping=grasp_manager,
    )

    removal_executor = PhysicalRemovalExecutor(
        simulation=simulation,
        controller=controller,
        grasp_manager=grasp_manager,
    )

    pipeline = BidirectionalGoalModificationPipeline(
        goal_reconciler=GoalReconciler(),
        addition_executor=addition_executor,
        removal_executor=removal_executor,
    )

    scene_adapter = MuJoCoSceneAdapter(
        simulation,
    )

    # ------------------------------------------------------------------
    # PREPARE INITIAL STATE
    # ------------------------------------------------------------------

    prepare_demo_scene(
        simulation,
    )

    scene = scene_adapter.snapshot()

    print()
    print("INITIAL PHYSICAL STATE")
    print("-" * 60)

    print(
        f"plate_1 = "
        f"{get_free_body_position(simulation, 'plate_1')}"
    )

    print(
        f"plate_2 = "
        f"{get_free_body_position(simulation, 'plate_2')}"
    )

    print(
        f"glass_1 = "
        f"{get_free_body_position(simulation, 'glass_1')}"
    )

    print(
        f"glass_2 = "
        f"{get_free_body_position(simulation, 'glass_2')}"
    )

    # ------------------------------------------------------------------
    # INITIAL VERIFICATION
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print("PHASE 1 — INITIAL GOAL: ONE GUEST")
    print("=" * 72)

    initial_ok = verify_one_setting(
        simulation,
    )

    if not initial_ok:
        raise RuntimeError(
            "Initial one-setting state was not prepared correctly."
        )

    print()
    print("Initial state verified: 1 complete place setting.")

    # ------------------------------------------------------------------
    # CONVERSATIONAL INCREASE
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print('PHASE 2 — USER: "Actually, make it 2."')
    print("=" * 72)
    print()

    current_request = make_request(
        1,
    )

    increase_result = pipeline.reconcile(
        current_request=current_request,
        instruction="Actually, make it 2.",
        scene=scene,
    )

    print(
        f"Current quantity : "
        f"{increase_result.reconciliation.current_quantity}"
    )

    print(
        f"Desired quantity : "
        f"{increase_result.reconciliation.desired_quantity}"
    )

    print(
        f"Delta            : "
        f"{increase_result.reconciliation.delta}"
    )

    print(
        "Additions        : "
        f"{[slot.index for slot in increase_result.reconciliation.additions]}"
    )

    if not increase_result.increased:
        raise RuntimeError(
            "Goal reconciler did not detect the 1 → 2 increase."
        )

    print()
    print("Executing REAL MuJoCo addition...")

    increase_executed = pipeline.execute(
        result=increase_result,
        scene=scene,
    )

    for result in increase_executed.addition_results:
        print(
            f"  {'PASS' if result.success else 'FAIL'} "
            f"{result.object_id}: "
            f"{result.message}"
        )

    if not increase_executed.execution_succeeded:
        raise RuntimeError(
            "Physical 1 → 2 goal modification failed."
        )

    # Let the simulation settle.
    simulation.step(40)
    mujoco.mj_forward(
        simulation.model,
        simulation.data,
    )

    # ------------------------------------------------------------------
    # VERIFY INCREASE
    # ------------------------------------------------------------------

    increase_ok = verify_two_settings(
        simulation,
    )

    if not increase_ok:
        raise RuntimeError(
            "Vision-equivalent physical verification failed after 1 → 2."
        )

    print()
    print("🔥 BIDIRECTIONAL INCREASE VERIFIED")
    print("   Physical goal changed from 1 → 2.")

    # ------------------------------------------------------------------
    # CONVERSATIONAL DECREASE
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print('PHASE 3 — USER: "Actually, one guest isn\'t coming."')
    print("=" * 72)
    print()

    scene = scene_adapter.snapshot()

    current_request = make_request(
        2,
    )

    decrease_result = pipeline.reconcile(
        current_request=current_request,
        instruction="Actually, one guest isn't coming.",
        scene=scene,
    )

    print(
        f"Current quantity : "
        f"{decrease_result.reconciliation.current_quantity}"
    )

    print(
        f"Desired quantity : "
        f"{decrease_result.reconciliation.desired_quantity}"
    )

    print(
        f"Delta            : "
        f"{decrease_result.reconciliation.delta}"
    )

    print(
        "Removals         : "
        f"{[state.slot.index for state in decrease_result.reconciliation.removals]}"
    )

    if not decrease_result.decreased:
        raise RuntimeError(
            "Goal reconciler did not detect the 2 → 1 decrease."
        )

    print()
    print("Executing REAL MuJoCo removal...")

    decrease_executed = pipeline.execute(
        result=decrease_result,
        scene=scene,
    )

    for result in decrease_executed.removal_results:
        print(
            f"  {'PASS' if result.success else 'FAIL'} "
            f"{result.object_id}: "
            f"{result.reason}"
        )

    if not decrease_executed.execution_succeeded:
        raise RuntimeError(
            "Physical 2 → 1 goal modification failed."
        )

    simulation.step(40)
    mujoco.mj_forward(
        simulation.model,
        simulation.data,
    )

    # ------------------------------------------------------------------
    # VERIFY DECREASE
    # ------------------------------------------------------------------

    remaining_ok = verify_one_setting(
        simulation,
    )

    removed_ok = verify_removed_setting(
        simulation,
    )

    if not remaining_ok or not removed_ok:
        raise RuntimeError(
            "Physical verification failed after 2 → 1."
        )

    print()
    print("🔥 BIDIRECTIONAL DECREASE VERIFIED")
    print("   Physical goal changed from 2 → 1.")

    # ------------------------------------------------------------------
    # FINAL RESULT
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print("FINAL RESULT")
    print("=" * 72)
    print()

    print("✓ Initial 1-person table verified")
    print("✓ Conversational increase detected")
    print("✓ Setting #2 physically restored")
    print("✓ Two-person table physically verified")
    print("✓ Conversational decrease detected")
    print("✓ Setting #2 physically removed")
    print("✓ One-person table physically verified")
    print()

    print("🔥 TABLEMIND 9.5 BIDIRECTIONAL GOAL DEMO SUCCESSFUL")
    print()
    print("    1  →  2  →  1")
    print()
    print("The robot did not merely replan.")
    print("It reconciled the user's new goal with the physical world.")
    print()

    # Give a short window for visual inspection if running a viewer.
    time.sleep(2)


if __name__ == "__main__":
    main()