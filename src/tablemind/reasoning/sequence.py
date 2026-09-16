"""Execution sequencing utilities for TABLEMIND."""

from __future__ import annotations

from dataclasses import dataclass

from tablemind.reasoning.decomposition import (
    DecomposedTask,
    TaskStep,
    TaskStepType,
)


@dataclass(frozen=True)
class ExecutionStage:
    """A group of manipulation steps that can execute together."""

    index: int
    steps: tuple[TaskStep, ...]

    @property
    def step_count(self) -> int:
        """Return the number of steps in this stage."""

        return len(self.steps)

    @property
    def arms(self) -> tuple[str, ...]:
        """Return the unique robot arms involved."""

        return tuple(dict.fromkeys(step.arm for step in self.steps))


@dataclass(frozen=True)
class TaskSequence:
    """An ordered sequence of execution stages."""

    task_name: str
    stages: tuple[ExecutionStage, ...]

    @property
    def stage_count(self) -> int:
        """Return the number of execution stages."""

        return len(self.stages)

    @property
    def step_count(self) -> int:
        """Return the total number of manipulation steps."""

        return sum(stage.step_count for stage in self.stages)


class TaskSequencePlanner:
    """Convert a decomposed task into an ordered execution sequence."""

    def create_sequence(
        self,
        task: DecomposedTask,
    ) -> TaskSequence:
        """Create a bimanual-friendly execution sequence."""

        if not task.steps:
            return TaskSequence(
                task_name=task.name,
                stages=(),
            )

        pick_steps = tuple(
            step
            for step in task.steps
            if step.step_type == TaskStepType.PICK
        )

        place_steps = tuple(
            step
            for step in task.steps
            if step.step_type == TaskStepType.PLACE
        )

        stages: list[ExecutionStage] = []

        if pick_steps:
            stages.append(
                ExecutionStage(
                    index=len(stages) + 1,
                    steps=pick_steps,
                )
            )

        if place_steps:
            stages.append(
                ExecutionStage(
                    index=len(stages) + 1,
                    steps=place_steps,
                )
            )

        return TaskSequence(
            task_name=task.name,
            stages=tuple(stages),
        )