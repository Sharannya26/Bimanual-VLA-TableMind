"""Update TABLEMIND task requests from conversational modifications."""

from __future__ import annotations

from tablemind.conversation.modification import ModificationRequest
from tablemind.reasoning.task_request import TaskRequest


class TaskUpdateManager:
    """Apply conversational modifications to an existing task request."""

    def update(
        self,
        current: TaskRequest,
        modification: ModificationRequest,
    ) -> TaskRequest:
        """Return an updated task request."""

        if not modification.is_modification:
            raise ValueError(
                "Request is not marked as a modification."
            )

        if current.intent != modification.intent:
            raise ValueError(
                "Modification intent does not match the current task."
            )

        quantity = (
            modification.quantity
            if modification.quantity is not None
            else current.quantity
        )

        return TaskRequest(
            raw_instruction=modification.raw_instruction,
            intent=current.intent,
            object_types=current.object_types,
            quantity=quantity,
            constraints=current.constraints,
        )