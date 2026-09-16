"""Conversational stop-and-resume orchestration for TABLEMIND."""

from __future__ import annotations

from dataclasses import dataclass

from tablemind.conversation.modification_pipeline import (
    ModificationPipeline,
)
from tablemind.control.so101 import SO101Controller
from tablemind.manipulation.execution_control import ExecutionControl
from tablemind.manipulation.grasping import GraspManager
from tablemind.manipulation.synchronized_executor import (
    SynchronizedBimanualExecutor,
)
from tablemind.reasoning.execution_bridge import (
    ReasoningExecutionBridge,
)
from tablemind.reasoning.task_request import TaskRequest
from tablemind.perception.scene import SceneObservation
from tablemind.simulation import BimanualTableSimulation


@dataclass(frozen=True)
class ResumeResult:
    """Result of replanning and resuming a modified task."""

    modification: object
    updated_request: TaskRequest
    replanning: object
    execution: object


class ConversationalResumePipeline:
    """Stop, update, replan, and resume TABLEMIND execution."""

    def __init__(
        self,
        simulation: BimanualTableSimulation,
        control: ExecutionControl | None = None,
        modification_pipeline: ModificationPipeline | None = None,
    ) -> None:
        self.simulation = simulation
        self.control = control or ExecutionControl()

        self.modification_pipeline = (
            modification_pipeline
            or ModificationPipeline()
        )

        self.controller = SO101Controller(
            simulation
        )

        self.grasping = GraspManager(
            simulation,
            self.controller,
        )

        self.executor = SynchronizedBimanualExecutor(
            simulation,
            controller=self.controller,
            grasping=self.grasping,
            control=self.control,
        )

        self.bridge = ReasoningExecutionBridge()

    def request_stop(self) -> None:
        """Request a safe stop of the current execution."""

        self.control.request_stop()

    def resume_with_modification(
        self,
        current_request: TaskRequest,
        modification_text: str,
        scene: SceneObservation,
    ) -> ResumeResult:
        """Apply a modification, replan, and resume execution."""

        # The previous execution must be stopped before
        # starting a new plan.
        self.control.clear()

        (
            modification,
            updated_request,
            replanning,
        ) = self.modification_pipeline.process(
            current_request,
            modification_text,
            scene,
        )

        if not replanning.needs_action:
            return ResumeResult(
                modification=modification,
                updated_request=updated_request,
                replanning=replanning,
                execution=None,
            )

        bridge_result = self.bridge.build_task(
            replanning.plan
        )

        execution = self.executor.execute(
            bridge_result.task
        )

        return ResumeResult(
            modification=modification,
            updated_request=updated_request,
            replanning=replanning,
            execution=execution,
        )