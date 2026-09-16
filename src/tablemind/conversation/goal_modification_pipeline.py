"""Bidirectional conversational goal modification for TABLEMIND."""

from __future__ import annotations

from dataclasses import dataclass

from tablemind.conversation.modification import ModificationRequest
from tablemind.conversation.modification_pipeline import ModificationPipeline
from tablemind.manipulation.goal_addition import (
    AdditionResult,
    GoalAdditionExecutor,
)
from tablemind.manipulation.removal import (
    PhysicalRemovalExecutor,
    RemovalResult,
)
from tablemind.perception.scene import SceneObservation
from tablemind.reasoning.goal_reconciliation import (
    GoalReconciliation,
    GoalReconciler,
)
from tablemind.reasoning.task_request import TaskRequest


@dataclass(frozen=True)
class GoalModificationResult:
    """Result of reconciling a conversational goal."""

    modification: ModificationRequest
    updated_request: TaskRequest
    reconciliation: GoalReconciliation
    addition_results: tuple[AdditionResult, ...] = ()
    removal_results: tuple[RemovalResult, ...] = ()

    @property
    def changed(self) -> bool:
        return self.reconciliation.needs_change

    @property
    def increased(self) -> bool:
        return self.reconciliation.is_increase

    @property
    def decreased(self) -> bool:
        return self.reconciliation.is_decrease

    @property
    def already_satisfied(self) -> bool:
        return self.reconciliation.already_satisfied

    @property
    def addition_succeeded(self) -> bool:
        if not self.addition_results:
            return True

        return all(
            result.success
            for result in self.addition_results
        )

    @property
    def removal_succeeded(self) -> bool:
        if not self.removal_results:
            return True

        return all(
            result.success
            for result in self.removal_results
        )

    @property
    def execution_succeeded(self) -> bool:
        return (
            self.addition_succeeded
            and self.removal_succeeded
        )


class BidirectionalGoalModificationPipeline:
    """
    Handle both increases and decreases in the conversational goal.

    The pipeline deliberately separates:

    1. conversational interpretation,
    2. physical-state reconciliation,
    3. physical addition,
    4. physical removal.

    Existing 9.4 execution infrastructure remains untouched.
    """

    def __init__(
        self,
        modification_pipeline: ModificationPipeline | None = None,
        goal_reconciler: GoalReconciler | None = None,
        addition_executor: GoalAdditionExecutor | None = None,
        removal_executor: PhysicalRemovalExecutor | None = None,
    ) -> None:
        self.modification_pipeline = (
            modification_pipeline
            or ModificationPipeline()
        )

        self.goal_reconciler = (
            goal_reconciler
            or GoalReconciler()
        )

        self.addition_executor = addition_executor
        self.removal_executor = removal_executor

    # ======================================================
    # Conversation
    # ======================================================

    def interpret(
        self,
        instruction: str,
    ) -> ModificationRequest:
        """Interpret a conversational modification."""

        return self.modification_pipeline.interpret(
            instruction
        )

    def update_request(
        self,
        current_request: TaskRequest,
        modification: ModificationRequest,
    ) -> TaskRequest:
        """Create the updated task request."""

        return self.modification_pipeline.update_request(
            current_request,
            modification,
        )

    # ======================================================
    # Goal reconciliation
    # ======================================================

    def reconcile(
        self,
        current_request: TaskRequest,
        instruction: str,
        scene: SceneObservation,
    ) -> GoalModificationResult:
        """
        Interpret a modification and compare the requested
        goal against the actual physical scene.
        """

        modification = self.interpret(
            instruction
        )

        updated_request = self.update_request(
            current_request,
            modification,
        )

        reconciliation = self.goal_reconciler.reconcile(
            desired_quantity=updated_request.quantity or 0,
            scene=scene,
        )

        return GoalModificationResult(
            modification=modification,
            updated_request=updated_request,
            reconciliation=reconciliation,
        )

    # ======================================================
    # Increase
    # ======================================================

    def execute_increase(
        self,
        reconciliation: GoalReconciliation,
        scene: SceneObservation,
    ) -> tuple[AdditionResult, ...]:
        """
        Physically restore missing place settings.

        Example:

            current = 1
            desired = 2

        The reconciler selects setting #2 and this method
        restores plate_2 and glass_2.
        """

        if not reconciliation.is_increase:
            return ()

        if self.addition_executor is None:
            raise RuntimeError(
                "A GoalAdditionExecutor is required to execute "
                "an increase in the physical goal."
            )

        results: list[AdditionResult] = []

        for slot in reconciliation.additions:
            plate = scene.get(slot.plate_id)
            glass = scene.get(slot.glass_id)

            if plate is None:
                raise RuntimeError(
                    f"Required object {slot.plate_id!r} "
                    "was not found in the scene."
                )

            if glass is None:
                raise RuntimeError(
                    f"Required object {slot.glass_id!r} "
                    "was not found in the scene."
                )

            setting_results = (
                self.addition_executor.add_setting(
                    slot=slot,
                    plate=plate,
                    glass=glass,
                )
            )

            results.extend(setting_results)

            if not all(
                result.success
                for result in setting_results
            ):
                break

        return tuple(results)

    # ======================================================
    # Decrease
    # ======================================================

    def execute_decrease(
        self,
        reconciliation: GoalReconciliation,
    ) -> tuple[RemovalResult, ...]:
        """Physically remove excess place settings."""

        if not reconciliation.is_decrease:
            return ()

        if self.removal_executor is None:
            raise RuntimeError(
                "A PhysicalRemovalExecutor is required to execute "
                "a decrease in the physical goal."
            )

        results: list[RemovalResult] = []

        for state in reconciliation.removals:
            if state.plate is None:
                raise RuntimeError(
                    f"Place setting {state.slot.index} "
                    "is missing its plate."
                )

            if state.glass is None:
                raise RuntimeError(
                    f"Place setting {state.slot.index} "
                    "is missing its glass."
                )

            setting_results = (
                self.removal_executor.remove_setting(
                    state.plate,
                    state.glass,
                )
            )

            results.extend(setting_results)

            if not all(
                result.success
                for result in setting_results
            ):
                break

        return tuple(results)

    # ======================================================
    # Unified execution
    # ======================================================

    def execute(
        self,
        result: GoalModificationResult,
        scene: SceneObservation,
    ) -> GoalModificationResult:
        """
        Execute whichever physical change is required.

        This is the main 9.5 entry point after reconciliation.
        """

        reconciliation = result.reconciliation

        if reconciliation.is_already_satisfied:
            return result

        if reconciliation.is_increase:
            addition_results = self.execute_increase(
                reconciliation=reconciliation,
                scene=scene,
            )

            return GoalModificationResult(
                modification=result.modification,
                updated_request=result.updated_request,
                reconciliation=reconciliation,
                addition_results=addition_results,
                removal_results=(),
            )

        if reconciliation.is_decrease:
            removal_results = self.execute_decrease(
                reconciliation=reconciliation,
            )

            return GoalModificationResult(
                modification=result.modification,
                updated_request=result.updated_request,
                reconciliation=reconciliation,
                addition_results=(),
                removal_results=removal_results,
            )

        return result

    def process(
        self,
        current_request: TaskRequest,
        instruction: str,
        scene: SceneObservation,
    ) -> GoalModificationResult:
        """
        Convenience method for the complete conversational
        modification → reconciliation → execution flow.
        """

        result = self.reconcile(
            current_request=current_request,
            instruction=instruction,
            scene=scene,
        )

        return self.execute(
            result=result,
            scene=scene,
        )