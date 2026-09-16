"""Execution of coordinated bimanual manipulation tasks."""

from __future__ import annotations

from dataclasses import dataclass

from tablemind.control.so101 import SO101Controller
from tablemind.manipulation.coordination import CoordinatedTask
from tablemind.manipulation.grasping import GraspManager
from tablemind.simulation import BimanualTableSimulation


@dataclass(frozen=True)
class CoordinatedExecutionResult:
    """Result of executing one coordinated task."""

    success: bool
    completed_assignments: int
    total_assignments: int
    message: str


class CoordinatedTaskExecutor:
    """Execute coordinated manipulation assignments."""

    def __init__(
        self,
        simulation: BimanualTableSimulation,
        controller: SO101Controller | None = None,
        grasping: GraspManager | None = None,
    ) -> None:
        self.simulation = simulation

        self.controller = controller or SO101Controller(
            simulation
        )

        self.grasping = grasping or GraspManager(
            simulation,
            self.controller,
        )

    def execute(
        self,
        task: CoordinatedTask,
    ) -> CoordinatedExecutionResult:
        """Execute every assignment in a coordinated task."""

        completed = 0

        for assignment in task.assignments:
            success = self._execute_assignment(
                assignment.arm,
                assignment.object_id,
            )

            if success:
                completed += 1
            else:
                return CoordinatedExecutionResult(
                    success=False,
                    completed_assignments=completed,
                    total_assignments=len(task.assignments),
                    message=(
                        f"Assignment failed: "
                        f"{assignment.arm} -> "
                        f"{assignment.object_id}"
                    ),
                )

        return CoordinatedExecutionResult(
            success=True,
            completed_assignments=completed,
            total_assignments=len(task.assignments),
            message=(
                f"Coordinated task '{task.name}' "
                f"completed successfully."
            ),
        )

    def _execute_assignment(
        self,
        arm: str,
        object_id: str,
    ) -> bool:
        """Execute one complete pick-and-place assignment."""

        initial = self.grasping.objects.get(object_id)

        approach_target = (
            initial.position[0],
            initial.position[1],
            initial.position[2] + 0.08,
        )

        result = self.controller.move_to(
            arm,
            approach_target,
            tolerance=0.04,
            max_steps=700,
        )

        if not result.reached:
            return False

        grasp_target = (
            initial.position[0],
            initial.position[1],
            initial.position[2] + 0.02,
        )

        result = self.controller.move_to(
            arm,
            grasp_target,
            tolerance=0.04,
            max_steps=700,
        )

        if not result.reached:
            return False

        grasp_result = self.grasping.grasp(
            arm,
            object_id,
        )

        if not grasp_result.success:
            return False

        lift_target = (
            initial.position[0],
            initial.position[1],
            initial.position[2] + 0.15,
        )

        result = self.controller.move_to(
            arm,
            lift_target,
            tolerance=0.04,
            max_steps=700,
        )

        if not result.reached:
            self.grasping.release(
                arm,
                object_id,
            )
            return False

        # Conservative inward placement target.
        if arm == "left":
            placement_x = -0.22
        else:
            placement_x = 0.22

        transport_target = (
            placement_x,
            0.00,
            initial.position[2] + 0.15,
        )

        result = self.controller.move_to(
            arm,
            transport_target,
            tolerance=0.04,
            max_steps=700,
        )

        if not result.reached:
            self.grasping.release(
                arm,
                object_id,
            )
            return False

        release_target = (
            placement_x,
            0.00,
            initial.position[2] + 0.02,
        )

        result = self.controller.move_to(
            arm,
            release_target,
            tolerance=0.04,
            max_steps=700,
        )

        if not result.reached:
            self.grasping.release(
                arm,
                object_id,
            )
            return False

        self.grasping.release(
            arm,
            object_id,
        )

        self.simulation.step(40)

        final = self.grasping.objects.get(object_id)

        displacement = (
            (
                (final.position[0] - initial.position[0]) ** 2
                + (final.position[1] - initial.position[1]) ** 2
                + (final.position[2] - initial.position[2]) ** 2
            )
            ** 0.5
        )

        return displacement > 0.04