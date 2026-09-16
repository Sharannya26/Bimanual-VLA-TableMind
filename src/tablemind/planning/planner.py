"""Deterministic task planner for TABLEMIND."""

from __future__ import annotations

from tablemind.perception.state import SceneState
from tablemind.planning.task import (
    Action,
    ActionType,
    Task,
    TaskPlan,
)


class DeterministicTablePlanner:
    """
    Deterministic planner for simple table-setting tasks.

    This planner deliberately avoids language models at this stage.
    Its purpose is to establish a reliable planning interface before
    adding language understanding.
    """

    def plan_set_table(
        self,
        scene: SceneState,
    ) -> TaskPlan:
        """
        Create a deterministic plan for setting the table.

        The planner uses the objects currently visible in the scene.
        """

        task = Task(
            task_id="set_table",
            description="Set the table using the available plates and glasses.",
        )

        actions: list[Action] = []

        plates = scene.objects.by_type("plate")
        glasses = scene.objects.by_type("glass")

        ordered_objects = (
            list(sorted(plates, key=lambda obj: obj.position[0]))
            + list(sorted(glasses, key=lambda obj: obj.position[0]))
        )

        for index, obj in enumerate(ordered_objects, start=1):
            arm = self._select_arm(
                object_x=obj.position[0],
                scene=scene,
            )

            actions.extend(
                self._create_pick_and_place_sequence(
                    index=index,
                    object_id=obj.object_id,
                    arm=arm,
                    position=obj.position,
                )
            )

        return TaskPlan(
            task=task,
            actions=tuple(actions),
        )

    def _select_arm(
        self,
        object_x: float,
        scene: SceneState,
    ) -> str:
        """
        Select the closest arm based on current end-effector X position.
        """

        left = scene.robot("left")
        right = scene.robot("right")

        if left is None or right is None:
            raise RuntimeError(
                "Both left and right robot states are required."
            )

        left_distance = abs(
            object_x - left.end_effector_position[0]
        )

        right_distance = abs(
            object_x - right.end_effector_position[0]
        )

        return "left" if left_distance <= right_distance else "right"

    def _create_pick_and_place_sequence(
        self,
        index: int,
        object_id: str,
        arm: str,
        position: tuple[float, float, float],
    ) -> list[Action]:
        """Create primitive actions for one object."""

        approach_position = (
            position[0],
            position[1],
            position[2] + 0.12,
        )

        lift_position = (
            position[0],
            position[1],
            position[2] + 0.18,
        )

        return [
            Action(
                action_id=f"action_{index}_approach",
                action_type=ActionType.APPROACH,
                arm=arm,
                object_id=object_id,
                target_position=approach_position,
            ),
            Action(
                action_id=f"action_{index}_open",
                action_type=ActionType.OPEN_GRIPPER,
                arm=arm,
                object_id=object_id,
            ),
            Action(
                action_id=f"action_{index}_move",
                action_type=ActionType.MOVE,
                arm=arm,
                object_id=object_id,
                target_position=position,
            ),
            Action(
                action_id=f"action_{index}_close",
                action_type=ActionType.CLOSE_GRIPPER,
                arm=arm,
                object_id=object_id,
            ),
            Action(
                action_id=f"action_{index}_lift",
                action_type=ActionType.LIFT,
                arm=arm,
                object_id=object_id,
                target_position=lift_position,
            ),
        ]