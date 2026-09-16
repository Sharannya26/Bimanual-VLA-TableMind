"""Closed-loop VLA integration for TABLEMIND."""

from __future__ import annotations

from dataclasses import dataclass

from tablemind.control.so101 import SO101Controller

from tablemind.manipulation.coordination import CoordinatedTask

from tablemind.manipulation.synchronized_executor import (
    SynchronizedBimanualExecutor,
    SynchronizedExecutionResult,
)

from tablemind.manipulation.verification import (
    ManipulationVerifier,
    VerificationResult,
)

from tablemind.perception.state import RobotState

from tablemind.perception.vision_pipeline import (
    DEFAULT_CALIBRATION_PATH,
    DEFAULT_MODEL_PATH,
    VisionObservation,
    VisionPipeline,
)

from tablemind.planning.task import Action, TaskPlan

from tablemind.reasoning.execution_bridge import (
    ExecutionBridgeResult,
    ReasoningExecutionBridge,
)

from tablemind.reasoning.interpreter import TaskInterpreter

from tablemind.reasoning.planning_bridge import (
    ReasoningPlanningBridge,
)

from tablemind.reasoning.sequence import TaskSequencePlanner

from tablemind.reasoning.vision_reasoning_bridge import (
    VisionReasoningBridge,
)

from tablemind.simulation.runtime import BimanualTableSimulation


@dataclass(frozen=True)
class ClosedLoopBatchResult:
    """Result of one synchronized bimanual batch."""

    batch_index: int
    object_ids: tuple[str, ...]
    execution_result: SynchronizedExecutionResult
    verification_results: tuple[VerificationResult, ...]

    @property
    def success(self) -> bool:
        """Return True when execution and verification succeed."""

        return (
            self.execution_result.success
            and bool(self.verification_results)
            and all(
                result.success
                for result in self.verification_results
            )
        )


@dataclass(frozen=True)
class ClosedLoopResult:
    """Complete result of a TABLEMIND closed-loop run."""

    success: bool
    instruction: str
    vision_observation: VisionObservation
    reasoning_result: object
    task_plan: TaskPlan
    execution_task: CoordinatedTask | None
    execution_result: ExecutionBridgeResult | None
    synchronized_result: SynchronizedExecutionResult | None
    verification_results: tuple[VerificationResult, ...]
    batches: tuple[ClosedLoopBatchResult, ...]


class ClosedLoopVLA:
    """End-to-end closed-loop TABLEMIND controller."""

    def __init__(
        self,
        simulation: BimanualTableSimulation,
        vision_pipeline: VisionPipeline,
        reasoning_bridge: VisionReasoningBridge | None = None,
        planning_bridge: ReasoningPlanningBridge | None = None,
        execution_bridge: ReasoningExecutionBridge | None = None,
        synchronized_executor: SynchronizedBimanualExecutor | None = None,
    ) -> None:
        self.simulation = simulation

        # ==============================================================
        # VISION
        # ==============================================================

        self.vision_pipeline = vision_pipeline

        # Compatibility name expected by existing tests.
        self.vision = self.vision_pipeline

        # ==============================================================
        # REASONING
        # ==============================================================

        self.reasoning_bridge = (
            reasoning_bridge
            or VisionReasoningBridge(
                interpreter=TaskInterpreter(),
                sequence_planner=TaskSequencePlanner(),
            )
        )

        # Compatibility name expected by existing tests.
        self.reasoning = self.reasoning_bridge

        # ==============================================================
        # PLANNING
        # ==============================================================

        self.planning_bridge = (
            planning_bridge
            or ReasoningPlanningBridge()
        )

        # Compatibility name expected by existing tests.
        self.planning = self.planning_bridge

        # ==============================================================
        # EXECUTION BRIDGE
        # ==============================================================

        self.execution_bridge = (
            execution_bridge
            or ReasoningExecutionBridge()
        )

        # Compatibility name expected by existing tests.
        self.execution = self.execution_bridge

        # ==============================================================
        # SYNCHRONIZED EXECUTOR
        # ==============================================================

        self.synchronized_executor = (
            synchronized_executor
            or SynchronizedBimanualExecutor(
                simulation
            )
        )

        # Compatibility name expected by existing tests.
        self.executor = self.synchronized_executor

        # ==============================================================
        # LOW-LEVEL ROBOT CONTROL
        # ==============================================================

        self.controller = SO101Controller(
            simulation
        )

        # ==============================================================
        # VERIFICATION
        # ==============================================================

        self.verifier = ManipulationVerifier(
            self.synchronized_executor.grasping.objects
        )

    @classmethod
    def create(
        cls,
        simulation: BimanualTableSimulation,
        model_path=DEFAULT_MODEL_PATH,
        calibration_path=DEFAULT_CALIBRATION_PATH,
        camera_name: str = "table_overview",
        image_width: int = 640,
        image_height: int = 480,
        confidence_threshold: float = 0.25,
    ) -> "ClosedLoopVLA":
        """Create the complete TABLEMIND closed-loop pipeline."""

        vision_pipeline = VisionPipeline.create(
            simulation=simulation,
            model_path=model_path,
            calibration_path=calibration_path,
            camera_name=camera_name,
            image_width=image_width,
            image_height=image_height,
            confidence_threshold=confidence_threshold,
        )

        return cls(
            simulation=simulation,
            vision_pipeline=vision_pipeline,
        )

    # ==================================================================
    # ROBOT STATE
    # ==================================================================

    def _robot_states(self) -> tuple[RobotState, ...]:
        """Read the current state of both robot arms."""

        states: list[RobotState] = []

        for arm in self.simulation.arms:
            end_effector = (
                self.controller.end_effector_position(
                    arm
                )
            )

            states.append(
                RobotState(
                    arm_id=arm,
                    end_effector_position=end_effector,
                    gripper_position=0.0,
                )
            )

        return tuple(states)

    # ==================================================================
    # OBJECT HELPERS
    # ==================================================================

    @staticmethod
    def _object_order(
        task_plan: TaskPlan,
    ) -> tuple[str, ...]:
        """Return object IDs in first-appearance order."""

        ordered: list[str] = []

        for action in task_plan.actions:
            if action.object_id is None:
                continue

            if action.object_id not in ordered:
                ordered.append(
                    action.object_id
                )

        return tuple(ordered)

    @staticmethod
    def _actions_for_objects(
        task_plan: TaskPlan,
        object_ids: tuple[str, ...],
    ) -> tuple[Action, ...]:
        """Extract actions belonging to specific objects."""

        object_id_set = set(object_ids)

        return tuple(
            action
            for action in task_plan.actions
            if action.object_id in object_id_set
        )

    @staticmethod
    def _build_bimanual_batches(
        task_plan: TaskPlan,
    ) -> tuple[tuple[str, ...], ...]:
        """Split a multi-object task into synchronized batches.

        Example:

            left:
                plate_1
                glass_1

            right:
                plate_2
                glass_2

        becomes:

            batch 1:
                left  -> plate_1
                right -> plate_2

            batch 2:
                left  -> glass_1
                right -> glass_2
        """

        left_objects: list[str] = []
        right_objects: list[str] = []

        seen_left: set[str] = set()
        seen_right: set[str] = set()

        for action in task_plan.actions:
            if action.object_id is None:
                continue

            if action.arm == "left":
                if action.object_id not in seen_left:
                    left_objects.append(
                        action.object_id
                    )
                    seen_left.add(
                        action.object_id
                    )

            elif action.arm == "right":
                if action.object_id not in seen_right:
                    right_objects.append(
                        action.object_id
                    )
                    seen_right.add(
                        action.object_id
                    )

        if not left_objects and not right_objects:
            raise ValueError(
                "Task plan contains no arm-assigned objects."
            )

        if len(left_objects) != len(right_objects):
            raise ValueError(
                "Cannot create synchronized bimanual batches: "
                f"left has {len(left_objects)} objects while "
                f"right has {len(right_objects)} objects."
            )

        batches: list[tuple[str, ...]] = []

        for left_object, right_object in zip(
            left_objects,
            right_objects,
        ):
            batches.append(
                (
                    left_object,
                    right_object,
                )
            )

        return tuple(batches)

    # ==================================================================
    # VERIFICATION
    # ==================================================================

    def _verify_batch(
        self,
        object_ids: tuple[str, ...],
        task_plan: TaskPlan,
    ) -> tuple[VerificationResult, ...]:
        """Verify the final physical position of each object."""

        results: list[VerificationResult] = []

        for object_id in object_ids:
            expected_position = None

            for action in task_plan.actions:
                if action.object_id != object_id:
                    continue

                if action.target_position is not None:
                    expected_position = (
                        action.target_position
                    )

            if expected_position is None:
                raise ValueError(
                    "No expected placement position found for "
                    f"object '{object_id}'."
                )

            verification = (
                self.verifier.verify_position(
                    object_id=object_id,
                    expected_position=expected_position,
                    tolerance=0.06,
                )
            )

            results.append(
                verification
            )

        return tuple(results)

    # ==================================================================
    # MAIN CLOSED LOOP
    # ==================================================================

    def run(
        self,
        instruction: str,
    ) -> ClosedLoopResult:
        """Run the complete TABLEMIND closed-loop pipeline."""

        print()
        print("=" * 72)
        print("TABLEMIND CLOSED-LOOP VLA")
        print("=" * 72)

        # ==============================================================
        # 1. USER INSTRUCTION
        # ==============================================================

        print()
        print("[1/6] USER INSTRUCTION")
        print(f"  {instruction}")

        # ==============================================================
        # 2. VISION
        # ==============================================================

        print()
        print("[2/6] VISION")

        vision_observation = (
            self.vision_pipeline.observe()
        )

        print(
            f"  detected="
            f"{len(vision_observation.detections)} objects"
        )

        for detection in vision_observation.detections:
            bbox = tuple(
                round(value, 2)
                for value in detection.bounding_box
            )

            print(
                "  "
                f"{detection.object_type} "
                f"conf={detection.confidence:.3f} "
                f"bbox={bbox}"
            )

        print(
            "  "
            f"scene_objects="
            f"{vision_observation.scene_state.object_count}"
        )

        for obj in vision_observation.scene_state.objects:
            print(
                "  "
                f"{obj.object_id} "
                f"xyz=("
                f"{obj.position.x:.5f}, "
                f"{obj.position.y:.5f}, "
                f"{obj.position.z:.3f}"
                f")"
            )

        # ==============================================================
        # 3. REASONING
        # ==============================================================

        print()
        print("[3/6] REASONING")

        robot_states = self._robot_states()

        reasoning_result = (
            self.reasoning_bridge.build(
                instruction=instruction,
                vision_scene=(
                    vision_observation.scene_state
                ),
                robot_states=robot_states,
                simulation_time=(
                    self.simulation.data.time
                ),
            )
        )

        print(
            f"  intent="
            f"{reasoning_result.request.intent}"
        )

        print(
            f"  objects="
            f"{reasoning_result.request.object_types}"
        )

        print(
            f"  quantity="
            f"{reasoning_result.request.quantity}"
        )

        print(
            f"  grounded="
            f"{len(reasoning_result.grounded.matches)}"
        )

        print(
            f"  sequence_stages="
            f"{len(reasoning_result.sequence.stages)}"
        )

        # ==============================================================
        # 4. PLANNING
        # ==============================================================

        print()
        print("[4/6] PLANNING")

        task_plan = (
            self.planning_bridge.build_plan(
                sequence=reasoning_result.sequence,
                scene=reasoning_result.scene,
            )
        )

        print(
            f"  task="
            f"{task_plan.task.task_id}"
        )

        print(
            f"  actions="
            f"{len(task_plan.actions)}"
        )

        planned_object_ids = (
            self._object_order(
                task_plan
            )
        )

        print(
            f"  objects="
            f"{planned_object_ids}"
        )

        batches = (
            self._build_bimanual_batches(
                task_plan
            )
        )

        print(
            f"  synchronized_batches="
            f"{len(batches)}"
        )

        # ==============================================================
        # 5. EXECUTION + VERIFICATION
        # ==============================================================

        print()
        print("[5/6] EXECUTION + VERIFICATION")

        batch_results: list[
            ClosedLoopBatchResult
        ] = []

        all_verifications: list[
            VerificationResult
        ] = []

        first_execution_task: (
            CoordinatedTask | None
        ) = None

        first_execution_result: (
            ExecutionBridgeResult | None
        ) = None

        first_synchronized_result: (
            SynchronizedExecutionResult | None
        ) = None

        for batch_index, object_ids in enumerate(
            batches,
            start=1,
        ):
            print()
            print(
                f"  BATCH "
                f"{batch_index}/{len(batches)}"
            )

            print(
                f"    objects="
                f"{object_ids}"
            )

            # ----------------------------------------------------------
            # Build a plan containing only this batch.
            # ----------------------------------------------------------

            batch_actions = (
                self._actions_for_objects(
                    task_plan=task_plan,
                    object_ids=object_ids,
                )
            )

            batch_task_plan = TaskPlan(
                task=task_plan.task,
                actions=batch_actions,
            )

            # ----------------------------------------------------------
            # Build coordinated bimanual task.
            # ----------------------------------------------------------

            execution_result = (
                self.execution_bridge.build_task(
                    batch_task_plan
                )
            )

            execution_task = (
                execution_result.task
            )

            if first_execution_task is None:
                first_execution_task = (
                    execution_task
                )

            if first_execution_result is None:
                first_execution_result = (
                    execution_result
                )

            print(
                f"    assignments="
                f"{execution_result.assignment_count}"
            )

            for assignment in execution_task.assignments:
                print(
                    "    "
                    f"{assignment.arm}: "
                    f"{assignment.object_id}"
                )

            # ----------------------------------------------------------
            # Execute synchronized manipulation.
            # ----------------------------------------------------------

            synchronized_result = (
                self.synchronized_executor.execute(
                    execution_task
                )
            )

            if first_synchronized_result is None:
                first_synchronized_result = (
                    synchronized_result
                )

            print(
                f"    stages="
                f"{synchronized_result.completed_stages}/"
                f"{synchronized_result.total_stages}"
            )

            print(
                "    execution="
                f"{'PASS' if synchronized_result.success else 'FAIL'}"
            )

            # ----------------------------------------------------------
            # Verify physical placement.
            # ----------------------------------------------------------

            verification_results = (
                self._verify_batch(
                    object_ids=object_ids,
                    task_plan=task_plan,
                )
            )

            for verification in verification_results:
                print(
                    "    "
                    f"{verification.object_id}: "
                    f"{'PASS' if verification.success else 'FAIL'} "
                    f"error="
                    f"{verification.position_error:.4f}m"
                )

            all_verifications.extend(
                verification_results
            )

            batch_result = (
                ClosedLoopBatchResult(
                    batch_index=batch_index,
                    object_ids=object_ids,
                    execution_result=(
                        synchronized_result
                    ),
                    verification_results=(
                        verification_results
                    ),
                )
            )

            batch_results.append(
                batch_result
            )

            if not batch_result.success:
                print()
                print(
                    "    ⚠️ Batch failed."
                )

                print(
                    "    Closed-loop execution stopped."
                )

                break

        # ==============================================================
        # 6. FINAL RESULT
        # ==============================================================

        print()
        print("[6/6] FINAL RESULT")

        planned_object_ids = (
            self._object_order(
                task_plan
            )
        )

        final_success = (
            len(all_verifications)
            == len(planned_object_ids)
            and all(
                result.success
                for result in all_verifications
            )
            and len(batch_results)
            == len(batches)
            and all(
                batch.success
                for batch in batch_results
            )
        )

        if final_success:
            print()
            print(
                "🎉 TABLEMIND CLOSED-LOOP SUCCESS"
            )

            print(
                "  Vision → reasoning → planning → "
                "bimanual execution → verification"
            )

            print(
                "  All objects successfully verified."
            )

        else:
            print()
            print(
                "❌ TABLEMIND CLOSED-LOOP FAILED"
            )

            print(
                f"  planned_objects="
                f"{len(planned_object_ids)}"
            )

            print(
                f"  verified_objects="
                f"{len(all_verifications)}"
            )

        print()
        print("=" * 72)

        return ClosedLoopResult(
            success=final_success,
            instruction=instruction,
            vision_observation=(
                vision_observation
            ),
            reasoning_result=(
                reasoning_result
            ),
            task_plan=task_plan,
            execution_task=(
                first_execution_task
            ),
            execution_result=(
                first_execution_result
            ),
            synchronized_result=(
                first_synchronized_result
            ),
            verification_results=tuple(
                all_verifications
            ),
            batches=tuple(
                batch_results
            ),
        )

    # ==================================================================
    # RESOURCE MANAGEMENT
    # ==================================================================

    def close(self) -> None:
        """Close resources owned by the vision pipeline."""

        self.vision_pipeline.close()

    def __enter__(self) -> "ClosedLoopVLA":
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()