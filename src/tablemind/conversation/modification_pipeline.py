"""Conversational modification pipeline for TABLEMIND."""

from __future__ import annotations

from tablemind.conversation.modification import ModificationRequest
from tablemind.conversation.modification_interpreter import (
    ModificationInterpreter,
)
from tablemind.reasoning.dynamic_replanner import (
    DynamicReplanner,
    ReplanningResult,
)
from tablemind.reasoning.task_request import TaskRequest


class ModificationPipeline:
    """Convert conversational modifications into updated task plans."""

    def __init__(
        self,
        modification_interpreter: ModificationInterpreter | None = None,
        replanner: DynamicReplanner | None = None,
    ) -> None:
        self.modification_interpreter = (
            modification_interpreter
            or ModificationInterpreter()
        )

        self.replanner = (
            replanner
            or DynamicReplanner()
        )

    def interpret(
        self,
        instruction: str,
    ) -> ModificationRequest:
        """Interpret a spoken conversational modification."""

        return self.modification_interpreter.interpret(
            instruction,
        )

    def update_request(
        self,
        current_request: TaskRequest,
        modification: ModificationRequest,
    ) -> TaskRequest:
        """Apply a modification to the current task request."""

        if modification.intent != current_request.intent:
            raise ValueError(
                "Modification intent does not match the current task."
            )

        quantity = (
            modification.quantity
            if modification.quantity is not None
            else current_request.quantity
        )

        return TaskRequest(
            raw_instruction=modification.raw_instruction,
            intent=current_request.intent,
            object_types=current_request.object_types,
            quantity=quantity,
            constraints=current_request.constraints,
        )

    def replan(
        self,
        request: TaskRequest,
        scene,
    ) -> ReplanningResult:
        """Replan the updated task using the latest scene."""

        return self.replanner.replan(
            request,
            scene,
        )

    def process(
        self,
        current_request: TaskRequest,
        instruction: str,
        scene,
    ) -> tuple[
        ModificationRequest,
        TaskRequest,
        ReplanningResult,
    ]:
        """Interpret, apply, and replan a conversational modification."""

        modification = self.interpret(
            instruction,
        )

        updated_request = self.update_request(
            current_request,
            modification,
        )

        replanning = self.replan(
            updated_request,
            scene,
        )

        return (
            modification,
            updated_request,
            replanning,
        )