"""Integrated vision pipeline for TABLEMIND Milestone 9.2.

Pipeline:

    MuJoCo camera
        -> RGB image
        -> OpenVINO/YOLO detector
        -> VisionDetection objects
        -> calibrated world coordinates
        -> Vision SceneState
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tablemind.perception.camera import TableCamera
from tablemind.perception.camera_grounding import CameraGrounder
from tablemind.perception.vision_detector import (
    OpenVINOTableDetector,
    VisionDetection,
)
from tablemind.perception.vision_scene_state import (
    CalibrationSet,
    SceneState,
    VisionSceneStateBuilder,
)
from tablemind.simulation.runtime import BimanualTableSimulation


DEFAULT_MODEL_PATH = Path(
    "runs/detect/artifacts/tablemind_mujoco_training/"
    "plate_glass_mujoco_v1/weights/best_openvino_model"
)

DEFAULT_CALIBRATION_PATH = Path(
    "artifacts/camera_calibration_8_9_7.json"
)


@dataclass(frozen=True)
class VisionObservation:
    """One complete observation from the TABLEMIND vision pipeline."""

    image: np.ndarray
    detections: tuple[VisionDetection, ...]
    scene_state: SceneState


class VisionPipeline:
    """Capture, detect, calibrate, and build a vision-grounded SceneState."""

    def __init__(
        self,
        simulation: BimanualTableSimulation,
        detector: OpenVINOTableDetector,
        scene_state_builder: VisionSceneStateBuilder,
        camera: TableCamera,
    ) -> None:
        self.simulation = simulation
        self.detector = detector
        self.scene_state_builder = scene_state_builder
        self.camera = camera

    @classmethod
    def create(
        cls,
        simulation: BimanualTableSimulation,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        calibration_path: str | Path = DEFAULT_CALIBRATION_PATH,
        camera_name: str = "table_overview",
        image_width: int = 640,
        image_height: int = 480,
        confidence_threshold: float = 0.25,
    ) -> "VisionPipeline":
        """Create a fully configured vision pipeline."""

        model_path = Path(model_path)
        calibration_path = Path(calibration_path)

        camera = TableCamera(
            model=simulation.model,
            data=simulation.data,
            camera_name=camera_name,
            width=image_width,
            height=image_height,
        )

        # Constructing CameraGrounder here validates that the MuJoCo camera
        # geometry and intrinsics are available and internally consistent.
        CameraGrounder(
            model=simulation.model,
            data=simulation.data,
            camera_name=camera_name,
            image_width=image_width,
            image_height=image_height,
        )

        detector = OpenVINOTableDetector(
            model_path=model_path,
            image_size=image_width,
            confidence_threshold=confidence_threshold,
        )

        calibration = CalibrationSet.from_json(calibration_path)

        scene_state_builder = VisionSceneStateBuilder(
            calibration=calibration,
            camera_width=image_width,
            camera_height=image_height,
            confidence_threshold=confidence_threshold,
        )

        return cls(
            simulation=simulation,
            detector=detector,
            scene_state_builder=scene_state_builder,
            camera=camera,
        )

    def capture(self) -> np.ndarray:
        """Capture the current table camera frame."""

        return self.camera.capture()

    def detect(
        self,
        image: np.ndarray,
    ) -> tuple[VisionDetection, ...]:
        """Run the custom OpenVINO detector on an RGB image."""

        detections = self.detector.detect(image)
        return tuple(detections)

    def build_scene_state(
        self,
        detections: tuple[VisionDetection, ...] | list[VisionDetection],
    ) -> SceneState:
        """Convert detections into calibrated world-space SceneState."""

        return self.scene_state_builder.build(detections)

    def observe(self) -> VisionObservation:
        """Capture, detect, and build a complete vision SceneState."""

        image = self.capture()

        detections = self.detect(image)

        scene_state = self.build_scene_state(detections)

        return VisionObservation(
            image=image,
            detections=detections,
            scene_state=scene_state,
        )

    def close(self) -> None:
        """Release the MuJoCo renderer."""

        self.camera.close()

    def __enter__(self) -> "VisionPipeline":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()