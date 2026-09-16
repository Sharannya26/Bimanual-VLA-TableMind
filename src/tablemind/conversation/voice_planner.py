"""Voice-to-planning pipeline for TABLEMIND."""

from __future__ import annotations

from tablemind.conversation.voice_pipeline import VoiceTaskPipeline
from tablemind.reasoning.context_planner import ContextAwarePlanner
from tablemind.reasoning.task_request import TaskRequest


class VoicePlannerPipeline:
    """Convert a spoken command into a TABLEMIND task plan."""

    def __init__(
        self,
        voice_pipeline: VoiceTaskPipeline | None = None,
        planner: ContextAwarePlanner | None = None,
    ) -> None:
        self.voice_pipeline = (
            voice_pipeline or VoiceTaskPipeline()
        )
        self.planner = (
            planner or ContextAwarePlanner()
        )

    def process(
        self,
        transcript: str,
        scene,
    ):
        """Convert a voice transcript into a planned TABLEMIND task."""

        request: TaskRequest = (
            self.voice_pipeline.process(transcript)
        )

        plan = self.planner.plan(
            request,
            scene,
        )

        return request, plan