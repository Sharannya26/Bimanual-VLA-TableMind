"""Voice-to-task pipeline for TABLEMIND."""

from __future__ import annotations

from tablemind.reasoning.interpreter import TaskInterpreter, TaskRequest


class VoiceTaskPipeline:
    """Convert a completed spoken utterance into a TABLEMIND TaskRequest."""

    def __init__(
        self,
        interpreter: TaskInterpreter | None = None,
    ) -> None:
        self.interpreter = interpreter or TaskInterpreter()

    def process(self, transcript: str) -> TaskRequest:
        """Convert a completed voice transcript into a structured task request."""
        transcript = transcript.strip()

        if not transcript:
            raise ValueError("Voice transcript cannot be empty.")

        return self.interpreter.interpret(transcript)