"""Bridge calibrated vision state into TABLEMIND reasoning."""

from __future__ import annotations

from dataclasses import dataclass

from tablemind.perception.scene import SceneObject, SceneObservation
from tablemind.perception.state import RobotState
from tablemind.perception.state import SceneState as LegacySceneState
from tablemind.perception.vision_scene_state import SceneState as VisionSceneState
from tablemind.reasoning.decomposition import DecomposedTask
from tablemind.reasoning.decomposer import TaskDecomposer
from tablemind.reasoning.grounding import GroundingMatch, WorldGrounder
from tablemind.reasoning.sequence import TaskSequence, TaskSequencePlanner
from tablemind.reasoning.task_request import TaskRequest


@dataclass(frozen=True)
class ReasoningSceneAdapter:
    """Convert the new vision SceneState into the existing reasoning format."""

    def convert(
        self,
        vision_scene: VisionSceneState,
        robot_states: tuple[RobotState, ...],
        simulation_time: float = 0.0,
    ) -> LegacySceneState:
        """Convert calibrated vision objects into the legacy scene format."""

        scene_objects = tuple(
            SceneObject(
                object_id=obj.object_id,
                object_type=obj.object_type,
                position=obj.position.xyz,
            )
            for obj in vision_scene.objects
        )

        observation = SceneObservation(
            objects=scene_objects,
        )

        return LegacySceneState(
            objects=observation,
            robots=robot_states,
            simulation_time=simulation_time,
        )


@dataclass(frozen=True)
class GroundedTask:
    """A language request grounded against the current visual scene."""

    request: TaskRequest
    matches: tuple[GroundingMatch, ...]


@dataclass(frozen=True)
class VisionReasoningResult:
    """Result of vision-to-reasoning integration."""

    request: TaskRequest
    scene: LegacySceneState
    grounded: GroundedTask
    decomposed_task: DecomposedTask
    sequence: TaskSequence


class VisionReasoningBridge:
    """
    Connect calibrated vision with TABLEMIND reasoning.

    Pipeline:

        VisionSceneState
            ↓
        Legacy SceneState
            ↓
        WorldGrounder
            ↓
        TaskDecomposer
            ↓
        TaskSequencePlanner
    """

    def __init__(
        self,
        interpreter,
        grounder: WorldGrounder | None = None,
        decomposer: TaskDecomposer | None = None,
        sequence_planner: TaskSequencePlanner | None = None,
    ) -> None:
        self.interpreter = interpreter
        self.grounder = grounder or WorldGrounder()
        self.decomposer = decomposer or TaskDecomposer()
        self.sequence_planner = sequence_planner or TaskSequencePlanner()

    def build(
        self,
        instruction: str,
        vision_scene: VisionSceneState,
        robot_states: tuple[RobotState, ...],
        simulation_time: float = 0.0,
    ) -> VisionReasoningResult:
        """
        Convert a natural-language instruction and visual scene into
        a grounded, decomposed, sequenced task.
        """

        request = self.interpreter.interpret(instruction)

        scene = ReasoningSceneAdapter().convert(
            vision_scene=vision_scene,
            robot_states=robot_states,
            simulation_time=simulation_time,
        )

        grounded_matches = self._ground_request(
            request=request,
            scene=scene,
        )

        grounded_task = GroundedTask(
            request=request,
            matches=grounded_matches,
        )

        decomposed_task = self.decomposer.decompose_set_table(
            grounded_objects=grounded_matches,
        )

        sequence = self.sequence_planner.create_sequence(
            decomposed_task,
        )

        return VisionReasoningResult(
            request=request,
            scene=scene,
            grounded=grounded_task,
            decomposed_task=decomposed_task,
            sequence=sequence,
        )

    def _ground_request(
        self,
        request: TaskRequest,
        scene: LegacySceneState,
    ) -> tuple[GroundingMatch, ...]:
        """Ground every requested object type against the current scene."""

        matches: list[GroundingMatch] = []

        for object_type in request.object_types:
            type_matches = self.grounder.find_by_type(
                scene.objects,
                object_type,
            )

            matches.extend(type_matches)

        return self._apply_quantity(
            request=request,
            matches=tuple(matches),
        )

    @staticmethod
    def _apply_quantity(
        request: TaskRequest,
        matches: tuple[GroundingMatch, ...],
    ) -> tuple[GroundingMatch, ...]:
        """
        Apply the requested quantity independently to each object type.

        For example:

            "Set the table for two."

        means:

            2 plates
            2 glasses
        """

        if request.quantity is None:
            return matches

        selected: list[GroundingMatch] = []

        for object_type in request.object_types:
            object_matches = tuple(
                match
                for match in matches
                if match.object_type == object_type
            )

            selected.extend(
                object_matches[: request.quantity]
            )

        return tuple(selected)