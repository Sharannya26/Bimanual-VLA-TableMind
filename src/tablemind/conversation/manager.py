"""Multi-turn conversation management for TABLEMIND."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tablemind.conversation.modification_interpreter import (
    ModificationInterpreter,
)
from tablemind.reasoning.interpreter import TaskInterpreter
from tablemind.reasoning.task_request import TaskRequest


class ConversationState(str, Enum):
    IDLE = "idle"
    EXECUTING = "executing"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"


class ConversationAction(str, Enum):
    NEW_TASK = "new_task"
    MODIFY = "modify"
    STOP = "stop"
    RESUME = "resume"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ConversationTurn:
    """Interpretation of one conversational turn."""

    raw_text: str
    action: ConversationAction
    state: ConversationState
    task_request: TaskRequest | None = None


class ConversationManager:
    """Maintain TABLEMIND's conversational task state."""

    _STOP_PHRASES = (
        "stop",
        "pause",
        "halt",
        "cancel",
    )

    _RESUME_PHRASES = (
        "resume",
        "continue",
        "go ahead",
        "carry on",
    )

    def __init__(
        self,
        task_interpreter: TaskInterpreter | None = None,
        modification_interpreter: ModificationInterpreter | None = None,
    ) -> None:
        self.task_interpreter = (
            task_interpreter or TaskInterpreter()
        )
        self.modification_interpreter = (
            modification_interpreter
            or ModificationInterpreter()
        )

        self.state = ConversationState.IDLE
        self.current_request: TaskRequest | None = None

    def process(self, text: str) -> ConversationTurn:
        """Process one conversational turn."""

        text = text.strip()

        if not text:
            raise ValueError(
                "Conversation input cannot be empty."
            )

        lowered = text.lower()

        # --------------------------------------------------
        # STOP
        # --------------------------------------------------

        if any(
            phrase in lowered
            for phrase in self._STOP_PHRASES
        ):
            self.state = ConversationState.INTERRUPTED

            return ConversationTurn(
                raw_text=text,
                action=ConversationAction.STOP,
                state=self.state,
                task_request=self.current_request,
            )

        # --------------------------------------------------
        # RESUME
        # --------------------------------------------------

        if any(
            phrase in lowered
            for phrase in self._RESUME_PHRASES
        ):
            if self.current_request is None:
                return ConversationTurn(
                    raw_text=text,
                    action=ConversationAction.UNKNOWN,
                    state=self.state,
                )

            self.state = ConversationState.EXECUTING

            return ConversationTurn(
                raw_text=text,
                action=ConversationAction.RESUME,
                state=self.state,
                task_request=self.current_request,
            )

        # --------------------------------------------------
        # MODIFICATION
        # --------------------------------------------------

        try:
            modification = (
                self.modification_interpreter.interpret(
                    text
                )
            )

            if self.current_request is not None:
                self.state = ConversationState.EXECUTING

                return ConversationTurn(
                    raw_text=text,
                    action=ConversationAction.MODIFY,
                    state=self.state,
                    task_request=self.current_request,
                )

            return ConversationTurn(
                raw_text=text,
                action=ConversationAction.UNKNOWN,
                state=self.state,
            )

        except ValueError:
            pass

        # --------------------------------------------------
        # NEW TASK
        # --------------------------------------------------

        request = self.task_interpreter.interpret(text)

        self.current_request = request
        self.state = ConversationState.EXECUTING

        return ConversationTurn(
            raw_text=text,
            action=ConversationAction.NEW_TASK,
            state=self.state,
            task_request=request,
        )

    def mark_completed(self) -> None:
        """Mark the current task as completed."""

        self.state = ConversationState.COMPLETED

    def reset(self) -> None:
        """Reset the conversation to an idle state."""

        self.state = ConversationState.IDLE
        self.current_request = None