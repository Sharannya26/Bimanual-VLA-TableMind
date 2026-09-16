from pathlib import Path

import numpy as np

from tablemind.perception.vision_detector import (
    OpenVINOTableDetector,
    VisionDetection,
)


MODEL_PATH = Path(
    "runs/detect/artifacts/tablemind_training/"
    "plate_glass_v1-2/weights/best_openvino_model"
)


def test_vision_detection_center():
    detection = VisionDetection(
        object_type="plate",
        confidence=0.99,
        bounding_box=(100.0, 200.0, 300.0, 400.0),
    )

    assert detection.center == (200.0, 300.0)


def test_vision_detection_bottom_center():
    detection = VisionDetection(
        object_type="glass",
        confidence=0.99,
        bounding_box=(100.0, 200.0, 300.0, 400.0),
    )

    assert detection.bottom_center == (200.0, 400.0)


def test_detector_can_load_custom_openvino_model():
    detector = OpenVINOTableDetector(MODEL_PATH)

    assert detector.model is not None


def test_detector_returns_table_objects():
    detector = OpenVINOTableDetector(MODEL_PATH)

    image = np.zeros(
        (480, 640, 3),
        dtype=np.uint8,
    )

    detections = detector.detect(image)

    assert isinstance(detections, tuple)

    for detection in detections:
        assert detection.object_type in {"plate", "glass"}
        assert 0.0 <= detection.confidence <= 1.0