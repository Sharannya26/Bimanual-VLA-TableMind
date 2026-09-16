"""Synchronized bimanual manipulation execution for TABLEMIND."""

from __future__ import annotations

from dataclasses import dataclass

from tablemind.control.so101 import SO101Controller
from tablemind.manipulation.coordination import (
    BimanualAssignment,
    CoordinatedTask,
)
from tablemind.manipulation.execution_control import ExecutionControl
from tablemind.manipulation.grasping import GraspManager
from tablemind.simulation import BimanualTableSimulation


Position = tuple[float, float, float]


@dataclass(frozen=True)
class SynchronizedExecutionResult:
    """Result of one synchronized bimanual execution."""

    success: bool
    completed_stages: int
    total_stages: int
    message: str


class SynchronizedBimanualExecutor:
    """Execute both arms through shared manipulation stages."""

    STAGES = (
        "approach",
        "grasp_position",
        "grasp",
        "lift",
        "transport",
        "lower",
        "release",
    )

    def __init__(
        self,
        simulation: BimanualTableSimulation,
        controller: SO101Controller | None = None,
        grasping: GraspManager | None = None,
        control: ExecutionControl | None = None,
    ) -> None:
        self.simulation = simulation

        self.controller = controller or SO101Controller(
            simulation
        )

        self.grasping = grasping or GraspManager(
            simulation,
            self.controller,
        )

        self.control = control or ExecutionControl()

    def execute(
        self,
        task: CoordinatedTask,
    ) -> SynchronizedExecutionResult:
        """Execute both assignments stage-by-stage."""

        if len(task.assignments) != 2:
            raise ValueError(
                "Synchronized bimanual execution requires "
                "exactly two assignments."
            )

        assignments = task.assignments

        initial_states = {
            assignment.object_id: self.grasping.objects.get(
                assignment.object_id
            )
            for assignment in assignments
        }

        completed_stages = 0

        # ----------------------------------------------------
        # 1. APPROACH
        # ----------------------------------------------------

        print("\n[1/7] Both arms approaching...")

        for assignment in assignments:
            if self.control.stop_requested:
                return self._interrupted(
                    assignments,
                    completed_stages,
                )

            initial = initial_states[
                assignment.object_id
            ]

            target = self._approach_target(
                assignment,
                initial.position,
            )

            print(
                "\n[TABLEMIND DIAGNOSTIC] "
                f"{assignment.object_id} BEFORE APPROACH"
            )
            self._print_object_position(
                assignment.object_id
            )

            result = self.controller.move_to(
                assignment.arm,
                target,
                tolerance=0.04,
                max_steps=700,
                should_stop=lambda: self.control.stop_requested,
            )

            print(
                f"{assignment.arm} -> "
                f"{assignment.object_id} | "
                f"reached={result.reached} | "
                f"error={result.position_error:.3f} m"
            )

            print(
                "\n[TABLEMIND DIAGNOSTIC] "
                f"{assignment.object_id} AFTER APPROACH"
            )
            self._print_object_position(
                assignment.object_id
            )

            if self.control.stop_requested:
                return self._interrupted(
                    assignments,
                    completed_stages,
                )

            if not result.reached:
                return self._failure(
                    completed_stages,
                    "Approach stage failed.",
                )

        completed_stages += 1

        # ----------------------------------------------------
        # 2. GRASP POSITION
        # ----------------------------------------------------

        print("\n[2/7] Both arms moving into grasp position...")

        for assignment in assignments:
            if self.control.stop_requested:
                return self._interrupted(
                    assignments,
                    completed_stages,
                )

            initial = initial_states[
                assignment.object_id
            ]

            target = self._grasp_target(
                assignment,
                initial.position,
            )

            print(
                "\n[TABLEMIND DIAGNOSTIC] "
                f"{assignment.object_id} BEFORE GRASP POSITION"
            )
            self._print_object_position(
                assignment.object_id
            )

            print(
                "[TABLEMIND DIAGNOSTIC] "
                f"{assignment.object_id} GRASP TARGET = "
                f"({target[0]:+.4f}, "
                f"{target[1]:+.4f}, "
                f"{target[2]:+.4f})"
            )

            result = self.controller.move_to(
                assignment.arm,
                target,
                tolerance=0.04,
                max_steps=700,
                should_stop=lambda: self.control.stop_requested,
            )

            print(
                f"{assignment.arm} | "
                f"reached={result.reached} | "
                f"error={result.position_error:.3f} m"
            )

            print(
                "\n[TABLEMIND DIAGNOSTIC] "
                f"{assignment.object_id} AFTER GRASP POSITION"
            )

            self._print_object_position(
                assignment.object_id
            )

            gripper_position = (
                self.controller.end_effector_position(
                    assignment.arm
                )
            )

            object_position = (
                self.grasping.objects
                .get(assignment.object_id)
                .position
            )

            print(
                "[TABLEMIND DIAGNOSTIC] "
                f"{assignment.arm} GRIPPER = "
                f"({gripper_position[0]:+.4f}, "
                f"{gripper_position[1]:+.4f}, "
                f"{gripper_position[2]:+.4f})"
            )

            print(
                "[TABLEMIND DIAGNOSTIC] "
                f"{assignment.object_id} OBJECT = "
                f"({object_position[0]:+.4f}, "
                f"{object_position[1]:+.4f}, "
                f"{object_position[2]:+.4f})"
            )

            dx = (
                gripper_position[0]
                - object_position[0]
            )
            dy = (
                gripper_position[1]
                - object_position[1]
            )
            dz = (
                gripper_position[2]
                - object_position[2]
            )

            distance = (
                dx * dx
                + dy * dy
                + dz * dz
            ) ** 0.5

            print(
                "[TABLEMIND DIAGNOSTIC] "
                f"{assignment.object_id} "
                f"GRIPPER-OBJECT DISTANCE = "
                f"{distance:.4f} m"
            )

            if self.control.stop_requested:
                return self._interrupted(
                    assignments,
                    completed_stages,
                )

            if not result.reached:
                return self._failure(
                    completed_stages,
                    "Grasp positioning stage failed.",
                )

        completed_stages += 1

        # ----------------------------------------------------
        # 3. GRASP
        # ----------------------------------------------------

        print("\n[3/7] Both arms grasping...")

        for assignment in assignments:
            if self.control.stop_requested:
                return self._interrupted(
                    assignments,
                    completed_stages,
                )

            print(
                "\n[TABLEMIND DIAGNOSTIC] "
                f"{assignment.object_id} IMMEDIATELY BEFORE GRASP"
            )

            self._print_object_position(
                assignment.object_id
            )

            gripper_position = (
                self.controller.end_effector_position(
                    assignment.arm
                )
            )

            object_position = (
                self.grasping.objects
                .get(assignment.object_id)
                .position
            )

            print(
                "[TABLEMIND DIAGNOSTIC] "
                f"{assignment.arm} GRIPPER = "
                f"({gripper_position[0]:+.4f}, "
                f"{gripper_position[1]:+.4f}, "
                f"{gripper_position[2]:+.4f})"
            )

            print(
                "[TABLEMIND DIAGNOSTIC] "
                f"{assignment.object_id} OBJECT = "
                f"({object_position[0]:+.4f}, "
                f"{object_position[1]:+.4f}, "
                f"{object_position[2]:+.4f})"
            )

            result = self.grasping.grasp(
                assignment.arm,
                assignment.object_id,
            )

            print(
                f"{assignment.arm} -> "
                f"{assignment.object_id} | "
                f"success={result.success} | "
                f"distance={result.distance:.3f} m | "
                f"reason={result.reason}"
            )

            print(
                "\n[TABLEMIND DIAGNOSTIC] "
                f"{assignment.object_id} AFTER GRASP"
            )

            self._print_object_position(
                assignment.object_id
            )

            if self.control.stop_requested:
                return self._interrupted(
                    assignments,
                    completed_stages,
                )

            if not result.success:
                self._release_all(assignments)

                return self._failure(
                    completed_stages,
                    "Grasp stage failed.",
                )

        completed_stages += 1

        # ----------------------------------------------------
        # 4. LIFT
        # ----------------------------------------------------

        print("\n[4/7] Both arms lifting...")

        for assignment in assignments:
            if self.control.stop_requested:
                return self._interrupted(
                    assignments,
                    completed_stages,
                )

            initial = initial_states[
                assignment.object_id
            ]

            target = self._lift_target(
                assignment,
                initial.position,
            )

            result = self.controller.move_to(
                assignment.arm,
                target,
                tolerance=0.04,
                max_steps=700,
                should_stop=lambda: self.control.stop_requested,
            )

            print(
                f"{assignment.arm} | "
                f"reached={result.reached} | "
                f"error={result.position_error:.3f} m"
            )

            if self.control.stop_requested:
                return self._interrupted(
                    assignments,
                    completed_stages,
                )

            if not result.reached:
                self._release_all(assignments)

                return self._failure(
                    completed_stages,
                    "Lift stage failed.",
                )

        completed_stages += 1

        # ----------------------------------------------------
        # 5. TRANSPORT
        # ----------------------------------------------------

        print("\n[5/7] Both arms transporting...")

        for assignment in assignments:
            if self.control.stop_requested:
                return self._interrupted(
                    assignments,
                    completed_stages,
                )

            initial = initial_states[
                assignment.object_id
            ]

            target = self._transport_target(
                assignment,
                initial.position,
            )

            result = self.controller.move_to(
                assignment.arm,
                target,
                tolerance=0.04,
                max_steps=700,
                should_stop=lambda: self.control.stop_requested,
            )

            print(
                f"{assignment.arm} | "
                f"reached={result.reached} | "
                f"error={result.position_error:.3f} m"
            )

            if self.control.stop_requested:
                return self._interrupted(
                    assignments,
                    completed_stages,
                )

            if not result.reached:
                self._release_all(assignments)

                return self._failure(
                    completed_stages,
                    "Transport stage failed.",
                )

        completed_stages += 1

        # ----------------------------------------------------
        # 6. LOWER
        # ----------------------------------------------------

        print("\n[6/7] Both arms lowering...")

        for assignment in assignments:
            if self.control.stop_requested:
                return self._interrupted(
                    assignments,
                    completed_stages,
                )

            initial = initial_states[
                assignment.object_id
            ]

            target = self._lower_target(
                assignment,
                initial.position,
            )

            result = self.controller.move_to(
                assignment.arm,
                target,
                tolerance=0.04,
                max_steps=700,
                should_stop=lambda: self.control.stop_requested,
            )

            print(
                f"{assignment.arm} | "
                f"reached={result.reached} | "
                f"error={result.position_error:.3f} m"
            )

            if self.control.stop_requested:
                return self._interrupted(
                    assignments,
                    completed_stages,
                )

            if not result.reached:
                self._release_all(assignments)

                return self._failure(
                    completed_stages,
                    "Lowering stage failed.",
                )

        completed_stages += 1

        # ----------------------------------------------------
        # 7. RELEASE
        # ----------------------------------------------------

        print("\n[7/7] Both arms releasing...")

        if self.control.stop_requested:
            return self._interrupted(
                assignments,
                completed_stages,
            )

        self._release_all(assignments)

        self.simulation.step(40)

        completed_stages += 1

        return SynchronizedExecutionResult(
            success=True,
            completed_stages=completed_stages,
            total_stages=len(self.STAGES),
            message=(
                f"Synchronized task '{task.name}' "
                "completed successfully."
            ),
        )

    # ========================================================
    # Target selection
    # ========================================================

    @staticmethod
    def _approach_target(
        assignment: BimanualAssignment,
        initial_position: Position,
    ) -> Position:
        """Return planner target or the legacy approach fallback."""

        if assignment.approach_position is not None:
            return assignment.approach_position

        return (
            initial_position[0],
            initial_position[1],
            initial_position[2] + 0.08,
        )

    @staticmethod
    def _grasp_target(
        assignment: BimanualAssignment,
        initial_position: Position,
    ) -> Position:
        """Return planner target or the legacy grasp fallback."""

        if assignment.grasp_position is not None:
            return assignment.grasp_position

        return (
            initial_position[0],
            initial_position[1],
            initial_position[2] + 0.02,
        )

    @staticmethod
    def _lift_target(
        assignment: BimanualAssignment,
        initial_position: Position,
    ) -> Position:
        """Return planner target or the legacy lift fallback."""

        if assignment.lift_position is not None:
            return assignment.lift_position

        return (
            initial_position[0],
            initial_position[1],
            initial_position[2] + 0.15,
        )

    @staticmethod
    def _transport_target(
        assignment: BimanualAssignment,
        initial_position: Position,
    ) -> Position:
        """Return planner placement target at transport height."""

        if assignment.place_position is not None:
            place = assignment.place_position

            return (
                place[0],
                place[1],
                place[2] + 0.15,
            )

        if assignment.arm == "left":
            placement_x = -0.22
        else:
            placement_x = 0.22

        return (
            placement_x,
            0.00,
            initial_position[2] + 0.15,
        )

    @staticmethod
    def _lower_target(
        assignment: BimanualAssignment,
        initial_position: Position,
    ) -> Position:
        """Return planner placement target or legacy lower target."""

        if assignment.place_position is not None:
            return assignment.place_position

        if assignment.arm == "left":
            placement_x = -0.22
        else:
            placement_x = 0.22

        return (
            placement_x,
            0.00,
            initial_position[2] + 0.02,
        )

    # ========================================================
    # Diagnostic helpers
    # ========================================================

    def _print_object_position(
        self,
        object_id: str,
    ) -> None:
        """Print the current world position of one object."""

        position = self.grasping.objects.get(
            object_id
        ).position

        print(
            f"    object = "
            f"({position[0]:+.4f}, "
            f"{position[1]:+.4f}, "
            f"{position[2]:+.4f})"
        )

    # ========================================================
    # Helpers
    # ========================================================

    def _release_all(
        self,
        assignments: tuple[BimanualAssignment, ...],
    ) -> None:
        """Release all active objects."""

        for assignment in assignments:
            self.grasping.release(
                assignment.arm,
                assignment.object_id,
            )

    def _interrupted(
        self,
        assignments: tuple[BimanualAssignment, ...],
        completed_stages: int,
    ) -> SynchronizedExecutionResult:
        """Safely stop an interrupted execution."""

        print("\n[TABLEMIND] STOP REQUEST RECEIVED")

        self._release_all(assignments)

        self.simulation.step(20)

        print("[TABLEMIND] Active grasps released.")
        print("[TABLEMIND] Execution interrupted safely.")

        return SynchronizedExecutionResult(
            success=False,
            completed_stages=completed_stages,
            total_stages=len(self.STAGES),
            message=(
                f"Synchronized task '{self._task_name_placeholder()}' "
                "was interrupted by the user."
            ),
        )

    @staticmethod
    def _task_name_placeholder() -> str:
        """Return a stable label for interrupted execution."""

        return "current task"

    def _failure(
        self,
        completed_stages: int,
        message: str,
    ) -> SynchronizedExecutionResult:
        """Create a failed execution result."""

        return SynchronizedExecutionResult(
            success=False,
            completed_stages=completed_stages,
            total_stages=len(self.STAGES),
            message=message,
        )