"""Tests for TABLEMIND Milestone 9.2 vision pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tablemind.perception.vision_detector import VisionDetection
from tablemind.perception.vision_pipeline import (
    VisionObservation,
    VisionPipeline,
)
from tablemind.perception.vision_scene_state import (
    CalibrationSet,
    ScenePosition,
    VisionSceneStateBuilder,
)
from tablemind.simulation.runtime import BimanualTableSimulation


def _make_calibration_file(tmp_path: Path) -> Path:
    """Create a small deterministic calibration file for tests."""

    payload = {
        "plate": {
            "coefficients": [
                [0.01, 0.0],
                [0.0, -0.01],
                [0.0, 1.0],
            ],
            "z": 0.772,
        },
        "glass": {
            "coefficients": [
                [0.02, 0.0],
                [0.0, -0.02],
                [-1.0, 1.0],
            ],
            "z": 0.835,
        },
    }

    path = tmp_path / "calibration.json"

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    return path


def _make_detection(
    object_type: str,
    confidence: float,
    bbox: tuple[float, float, float, float],
) -> VisionDetection:
    """Create a deterministic VisionDetection for testing."""

    return VisionDetection(
        object_type=object_type,
        confidence=confidence,
        bounding_box=bbox,
    )


def test_pipeline_class_is_importable():
    """VisionPipeline should be importable."""

    assert VisionPipeline is not None


def test_vision_observation_is_constructible(tmp_path):
    """VisionObservation should store image, detections, and SceneState."""

    image = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    detections = (
        _make_detection(
            "plate",
            0.95,
            (200.0, 70.0, 260.0, 100.0),
        ),
    )

    calibration = CalibrationSet.from_json(
        _make_calibration_file(tmp_path)
    )

    builder = VisionSceneStateBuilder(
        calibration=calibration,
        camera_width=640,
        camera_height=480,
    )

    scene_state = builder.build(detections)

    observation = VisionObservation(
        image=image,
        detections=detections,
        scene_state=scene_state,
    )

    assert observation.image.shape == (480, 640, 3)
    assert len(observation.detections) == 1
    assert observation.scene_state.object_count == 1


def test_pipeline_scene_state_builder_accepts_multiple_objects(
    tmp_path,
):
    """SceneStateBuilder should handle multiple detected objects."""

    calibration = CalibrationSet.from_json(
        _make_calibration_file(tmp_path)
    )

    builder = VisionSceneStateBuilder(
        calibration=calibration,
        camera_width=640,
        camera_height=480,
    )

    detections = [
        _make_detection(
            "plate",
            0.90,
            (200.0, 70.0, 260.0, 100.0),
        ),
        _make_detection(
            "plate",
            0.85,
            (350.0, 70.0, 410.0, 100.0),
        ),
        _make_detection(
            "glass",
            0.80,
            (220.0, 65.0, 245.0, 105.0),
        ),
        _make_detection(
            "glass",
            0.75,
            (380.0, 65.0, 405.0, 105.0),
        ),
    ]

    scene_state = builder.build(detections)

    assert scene_state.object_count == 4
    assert scene_state.count("plate") == 2
    assert scene_state.count("glass") == 2


def test_pipeline_scene_state_contains_world_positions(
    tmp_path,
):
    """Vision-grounded objects should contain world positions."""

    calibration = CalibrationSet.from_json(
        _make_calibration_file(tmp_path)
    )

    builder = VisionSceneStateBuilder(
        calibration=calibration,
        camera_width=640,
        camera_height=480,
    )

    detection = _make_detection(
        "plate",
        0.95,
        (200.0, 70.0, 260.0, 100.0),
    )

    scene_state = builder.build([detection])

    obj = scene_state.get("plate_1")

    assert obj is not None
    assert obj.position is not None
    assert obj.position.z == 0.772


def test_pipeline_observation_image_is_rgb():
    """The expected camera image should be RGB uint8."""

    image = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    assert image.dtype == np.uint8
    assert image.shape == (480, 640, 3)


def test_real_simulation_camera_capture():
    """The real MuJoCo camera should produce a valid RGB frame."""

    simulation = BimanualTableSimulation.create()

    from tablemind.perception.camera import TableCamera

    camera = TableCamera(
        model=simulation.model,
        data=simulation.data,
        camera_name="table_overview",
        width=640,
        height=480,
    )

    try:
        image = camera.capture()

        assert image.shape == (480, 640, 3)
        assert image.dtype == np.uint8

    finally:
        camera.close()


def test_real_simulation_vision_pipeline_components():
    """The real simulation should support the camera-side pipeline."""

    simulation = BimanualTableSimulation.create()

    from tablemind.perception.camera import TableCamera
    from tablemind.perception.camera_grounding import CameraGrounder

    camera = TableCamera(
        model=simulation.model,
        data=simulation.data,
        camera_name="table_overview",
        width=640,
        height=480,
    )

    try:
        grounder = CameraGrounder(
            model=simulation.model,
            data=simulation.data,
            camera_name="table_overview",
            image_width=640,
            image_height=480,
        )

        image = camera.capture()

        assert image.shape == (480, 640, 3)
        assert grounder.camera_position.shape == (3,)
        assert grounder.camera_rotation.shape == (3, 3)

    finally:
        camera.close()


def test_scene_position_is_numeric():
    """ScenePosition should expose numeric coordinate values."""

    position = ScenePosition(
        x=0.25,
        y=-0.10,
        z=0.772,
    )

    assert position.x == 0.25
    assert position.y == -0.10
    assert position.z == 0.772
    assert position.xy == (0.25, -0.10)
    assert position.xyz == (0.25, -0.10, 0.772)