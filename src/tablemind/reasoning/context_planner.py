"""Context-aware task planning for TABLEMIND."""

from __future__ import annotations

from tablemind.perception.scene import SceneObservation
from tablemind.planning.task import (
    Action,
    ActionType,
    Task,
    TaskPlan,
)
from tablemind.reasoning.context_reasoner import (
    ContextReasoner,
    ContextReasoningResult,
)
from tablemind.reasoning.task_request import TaskRequest


class ContextAwarePlanner:
    """Generate manipulation actions from unified context reasoning."""

    def __init__(
        self,
        context_reasoner: ContextReasoner | None = None,
    ) -> None:
        self.context_reasoner = (
            context_reasoner or ContextReasoner()
        )

    def plan(
        self,
        request: TaskRequest,
        scene: SceneObservation,
    ) -> TaskPlan:
        """Reason about the current scene and build a task plan."""

        reasoning = self.context_reasoner.reason(
            request,
            scene,
        )

        return self.plan_from_reasoning(
            reasoning,
        )

    def plan_from_reasoning(
        self,
        reasoning: ContextReasoningResult,
    ) -> TaskPlan:
        """Build a task plan from an existing reasoning result."""

        request = reasoning.request

        completed_ids = set(
            reasoning.completed_objects
        )

        relevant_objects = [
            obj
            for obj in reasoning_scene_objects(reasoning)
            if (
                obj.object_type in request.object_types
                and obj.object_id not in completed_ids
            )
        ]

        actions: list[Action] = []

        for index, obj in enumerate(
            relevant_objects,
            start=1,
        ):
            arm = self._assign_arm(
                obj.position[0]
            )

            actions.extend(
                self._create_object_actions(
                    index=index,
                    arm=arm,
                    object_id=obj.object_id,
                    position=obj.position,
                    object_type=obj.object_type,
                )
            )

        task = Task(
            task_id=request.intent,
            description=request.raw_instruction,
        )

        return TaskPlan(
            task=task,
            actions=tuple(actions),
        )

    @staticmethod
    def _assign_arm(
        x_position: float,
    ) -> str:
        """Assign an object to the arm nearest its table side."""

        return (
            "left"
            if x_position < 0
            else "right"
        )

    @staticmethod
    def _create_object_actions(
        index: int,
        arm: str,
        object_id: str,
        position: tuple[float, float, float],
        object_type: str,
    ) -> tuple[Action, ...]:
        """Create a pick-and-place sequence for one object."""

        x, y, _ = position

        if object_type == "plate":
            target_z = 0.772
        elif object_type == "glass":
            target_z = 0.835
        else:
            target_z = position[2]

        approach_position = (
            x,
            y,
            target_z + 0.12,
        )

        target_position = (
            x,
            y,
            target_z,
        )

        prefix = (
            f"action_{index}_{object_id}"
        )

        return (
            Action(
                action_id=f"{prefix}_approach",
                action_type=ActionType.APPROACH,
                arm=arm,
                object_id=object_id,
                target_position=approach_position,
            ),
            Action(
                action_id=f"{prefix}_open",
                action_type=ActionType.OPEN_GRIPPER,
                arm=arm,
                object_id=object_id,
            ),
            Action(
                action_id=f"{prefix}_move",
                action_type=ActionType.MOVE,
                arm=arm,
                object_id=object_id,
                target_position=target_position,
            ),
            Action(
                action_id=f"{prefix}_close",
                action_type=ActionType.CLOSE_GRIPPER,
                arm=arm,
                object_id=object_id,
            ),
            Action(
                action_id=f"{prefix}_lift",
                action_type=ActionType.LIFT,
                arm=arm,
                object_id=object_id,
                target_position=(
                    x,
                    y,
                    target_z + 0.12,
                ),
            ),
            Action(
                action_id=f"{prefix}_place",
                action_type=ActionType.PLACE,
                arm=arm,
                object_id=object_id,
                target_position=target_position,
            ),
        )


def reasoning_scene_objects(
    reasoning: ContextReasoningResult,
):
    """Recover scene objects represented by the reasoning result."""

    object_ids = {
        status.object_id
        for status in reasoning.completion
    }

    objects = []

    for status in reasoning.completion:
        objects.append(
            _CompletionObject(
                object_id=status.object_id,
                object_type=status.object_type,
                position=status.current_position,
            )
        )

    return tuple(
        obj
        for obj in objects
        if obj.object_id in object_ids
    )


class _CompletionObject:
    """Small internal representation of a scene object."""

    def __init__(
        self,
        object_id: str,
        object_type: str,
        position: tuple[float, float, float],
    ) -> None:
        self.object_id = object_id
        self.object_type = object_type
        self.position = position