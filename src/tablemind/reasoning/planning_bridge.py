"""Bridge between TABLEMIND reasoning and manipulation planning."""

from __future__ import annotations

from tablemind.perception.state import SceneState
from tablemind.planning.task import (
    Action,
    ActionType,
    Task,
    TaskPlan,
)
from tablemind.reasoning.sequence import TaskSequence


class ReasoningPlanningBridge:
    """Convert a reasoning sequence into the existing planning format."""

    def build_plan(
        self,
        sequence: TaskSequence,
        scene: SceneState,
    ) -> TaskPlan:
        """Create a TaskPlan from a reasoning-generated sequence."""

        actions: list[Action] = []

        action_index = 1

        for stage in sequence.stages:
            for step in stage.steps:
                scene_object = scene.objects.get(step.object_id)

                if scene_object is None:
                    raise ValueError(
                        f"Object '{step.object_id}' "
                        "is not present in the current scene."
                    )

                position = scene_object.position

                if step.step_type.value == "pick":
                    actions.extend(
                        self._create_pick_actions(
                            action_index=action_index,
                            object_id=step.object_id,
                            object_type=scene_object.object_type,
                            arm=step.arm,
                            position=position,
                        )
                    )

                elif step.step_type.value == "place":
                    target = step.target_position or position

                    actions.append(
                        Action(
                            action_id=f"action_{action_index}_place",
                            action_type=ActionType.PLACE,
                            arm=step.arm,
                            object_id=step.object_id,
                            target_position=target,
                        )
                    )

                action_index += 1

        task = Task(
            task_id=sequence.task_name,
            description=(
                f"Reasoning-generated task '{sequence.task_name}'."
            ),
        )

        return TaskPlan(
            task=task,
            actions=tuple(actions),
        )

    @staticmethod
    def _create_pick_actions(
        action_index: int,
        object_id: str,
        object_type: str,
        arm: str,
        position: tuple[float, float, float],
    ) -> list[Action]:
        """Create primitive actions needed to pick one object."""

        if object_type == "glass":
            # Approach from well above the glass so the gripper
            # does not collide with or push the glass sideways.
            approach_position = (
                position[0],
                position[1],
                position[2] + 0.20,
            )

            # The SO-101 controller can stop within roughly
            # 4 cm of the requested Cartesian target. Therefore
            # place the nominal grasp point slightly below the
            # glass center so the actual gripper still enters
            # the grasp-distance envelope.
            grasp_position = (
                position[0],
                position[1],
                position[2] + 0.02,
            )
        else:
            approach_position = (
                position[0],
                position[1],
                position[2] + 0.12,
            )

            grasp_position = position

        return [
            Action(
                action_id=f"action_{action_index}_approach",
                action_type=ActionType.APPROACH,
                arm=arm,
                object_id=object_id,
                target_position=approach_position,
            ),
            Action(
                action_id=f"action_{action_index}_open",
                action_type=ActionType.OPEN_GRIPPER,
                arm=arm,
                object_id=object_id,
            ),
            Action(
                action_id=f"action_{action_index}_move",
                action_type=ActionType.MOVE,
                arm=arm,
                object_id=object_id,
                target_position=grasp_position,
            ),
            Action(
                action_id=f"action_{action_index}_close",
                action_type=ActionType.CLOSE_GRIPPER,
                arm=arm,
                object_id=object_id,
            ),
            Action(
                action_id=f"action_{action_index}_lift",
                action_type=ActionType.LIFT,
                arm=arm,
                object_id=object_id,
                target_position=(
                    position[0],
                    position[1],
                    position[2] + 0.15,
                ),
            ),
        ]