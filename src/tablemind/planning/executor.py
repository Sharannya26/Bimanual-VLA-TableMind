"""Action execution for TABLEMIND."""

from __future__ import annotations

from dataclasses import dataclass

from tablemind.control.so101 import SO101Controller
from tablemind.planning.task import Action, ActionType
from tablemind.simulation.runtime import BimanualTableSimulation


@dataclass(frozen=True)
class ActionResult:
    """Result of executing one planned action."""

    action_id: str
    success: bool
    message: str


@dataclass(frozen=True)
class ExecutionReport:
    """Results from executing a complete task plan."""

    results: tuple[ActionResult, ...]

    @property
    def success(self) -> bool:
        """Return True when every executed action succeeded."""

        return bool(self.results) and all(
            result.success
            for result in self.results
        )

    @property
    def completed_actions(self) -> int:
        """Return the number of successful actions."""

        return sum(
            result.success
            for result in self.results
        )


class ActionExecutor:
    """
    Execute TABLEMIND primitive actions.

    This executor connects the high-level task planner
    to the existing SO-101 controller.

    Physical object transfer is not claimed yet because
    the current table objects are fixed placeholder
    geometries.
    """

    def __init__(
        self,
        simulation: BimanualTableSimulation,
    ) -> None:
        self.simulation = simulation

        # One controller manages both arms.
        # The arm is specified when executing each action.
        self.controller = SO101Controller(
            simulation,
        )

    def execute_action(
        self,
        action: Action,
    ) -> ActionResult:
        """Execute one primitive action."""

        if action.arm not in self.simulation.arms:
            return ActionResult(
                action_id=action.action_id,
                success=False,
                message=f"Unknown arm: {action.arm}",
            )

        try:
            if action.action_type == ActionType.APPROACH:
                return self._execute_move(
                    action=action,
                    label="Approach",
                )

            if action.action_type == ActionType.MOVE:
                return self._execute_move(
                    action=action,
                    label="Move",
                )

            if action.action_type == ActionType.OPEN_GRIPPER:
                self.controller.open_gripper(
                    action.arm,
                )

                return ActionResult(
                    action_id=action.action_id,
                    success=True,
                    message=(
                        f"{action.arm} gripper opened"
                    ),
                )

            if action.action_type == ActionType.CLOSE_GRIPPER:
                self.controller.close_gripper(
                    action.arm,
                )

                return ActionResult(
                    action_id=action.action_id,
                    success=True,
                    message=(
                        f"{action.arm} gripper closed; "
                        "object attachment is not claimed"
                    ),
                )

            if action.action_type == ActionType.LIFT:
                return self._execute_move(
                    action=action,
                    label="Lift",
                )

            if action.action_type == ActionType.PLACE:
                return self._execute_move(
                    action=action,
                    label="Place",
                )

            return ActionResult(
                action_id=action.action_id,
                success=False,
                message=(
                    f"Unsupported action type: "
                    f"{action.action_type}"
                ),
            )

        except Exception as exc:
            return ActionResult(
                action_id=action.action_id,
                success=False,
                message=f"Execution error: {exc}",
            )

    def execute_plan(
        self,
        actions: tuple[Action, ...],
        stop_on_failure: bool = True,
    ) -> ExecutionReport:
        """
        Execute an ordered sequence of actions.

        Parameters
        ----------
        actions:
            Planned primitive actions.

        stop_on_failure:
            Stop execution when an action fails.
        """

        results: list[ActionResult] = []

        for action in actions:
            result = self.execute_action(action)

            results.append(result)

            if not result.success and stop_on_failure:
                break

        return ExecutionReport(
            results=tuple(results),
        )

    def _execute_move(
        self,
        action: Action,
        label: str,
    ) -> ActionResult:
        """Execute a Cartesian movement."""

        if action.target_position is None:
            return ActionResult(
                action_id=action.action_id,
                success=False,
                message=(
                    f"{label} action has no target position"
                ),
            )

        result = self.controller.move_to(
            action.arm,
            action.target_position,
        )

        if result.reached:
            return ActionResult(
                action_id=action.action_id,
                success=True,
                message=(
                    f"{label} target reached at "
                    f"{action.target_position}; "
                ),
            )

        return ActionResult(
            action_id=action.action_id,
            success=False,
            message=(
                f"{label} target not reached; "
            ),
        )