"""Tests for TABLEMIND Milestone 9.1 SceneState."""

from __future__ import annotations

import json

import numpy as np

from tablemind.perception.vision_scene_state import (
    AffineCalibration,
    CalibrationSet,
    ScenePosition,
    VisionSceneObject,
    VisionSceneStateBuilder,
)


class FakeDetection:
    """Minimal VisionDetection-compatible test object."""

    def __init__(
        self,
        object_type: str,
        confidence: float,
        bounding_box: tuple[
            float,
            float,
            float,
            float,
        ],
    ) -> None:

        self.object_type = object_type
        self.confidence = confidence
        self.bounding_box = bounding_box


def make_calibration() -> CalibrationSet:
    """Create deterministic test calibrations."""

    plate = AffineCalibration(
        np.asarray(
            [
                [0.001, 0.000],
                [0.000, 0.001],
                [0.000, 0.000],
            ]
        )
    )

    glass = AffineCalibration(
        np.asarray(
            [
                [0.002, 0.000],
                [0.000, 0.002],
                [0.000, 0.000],
            ]
        )
    )

    return CalibrationSet(
        calibrations={
            "plate": plate,
            "glass": glass,
        },
        z_values={
            "plate": 0.772,
            "glass": 0.835,
        },
    )


def test_scene_position_properties() -> None:

    position = ScenePosition(
        x=0.1,
        y=-0.2,
        z=0.772,
    )

    assert position.xy == (
        0.1,
        -0.2,
    )

    assert position.xyz == (
        0.1,
        -0.2,
        0.772,
    )


def test_affine_calibration() -> None:

    calibration = AffineCalibration(
        np.asarray(
            [
                [0.001, 0.000],
                [0.000, 0.002],
                [0.100, -0.100],
            ]
        )
    )

    x, y = calibration.pixel_to_world(
        200.0,
        50.0,
    )

    assert np.isclose(
        x,
        0.300,
    )

    assert np.isclose(
        y,
        0.000,
    )


def test_calibration_set_ground() -> None:

    calibration = make_calibration()

    position = calibration.ground(
        object_type="plate",
        pixel_x=200.0,
        pixel_y=100.0,
    )

    assert np.isclose(
        position.x,
        0.2,
    )

    assert np.isclose(
        position.y,
        0.1,
    )

    assert np.isclose(
        position.z,
        0.772,
    )


def test_builder_creates_scene_state() -> None:

    builder = VisionSceneStateBuilder(
        calibration=make_calibration()
    )

    detections = [
        FakeDetection(
            object_type="plate",
            confidence=0.95,
            bounding_box=(
                180.0,
                90.0,
                220.0,
                110.0,
            ),
        ),
        FakeDetection(
            object_type="glass",
            confidence=0.88,
            bounding_box=(
                300.0,
                60.0,
                320.0,
                100.0,
            ),
        ),
    ]

    state = builder.build(
        detections
    )

    assert state.object_count == 2

    assert state.count(
        "plate"
    ) == 1

    assert state.count(
        "glass"
    ) == 1

    assert state.contains(
        "plate"
    )

    assert state.contains(
        "glass"
    )


def test_builder_uses_bbox_center() -> None:

    builder = VisionSceneStateBuilder(
        calibration=make_calibration()
    )

    detection = FakeDetection(
        object_type="plate",
        confidence=0.95,
        bounding_box=(
            100.0,
            50.0,
            200.0,
            150.0,
        ),
    )

    state = builder.build(
        [detection]
    )

    obj = state.get(
        "plate_1"
    )

    assert obj.pixel_center == (
        150.0,
        100.0,
    )

    assert np.isclose(
        obj.x,
        0.150,
    )

    assert np.isclose(
        obj.y,
        0.100,
    )

    assert np.isclose(
        obj.z,
        0.772,
    )


def test_builder_filters_low_confidence() -> None:

    builder = VisionSceneStateBuilder(
        calibration=make_calibration(),
        confidence_threshold=0.5,
    )

    detections = [
        FakeDetection(
            object_type="plate",
            confidence=0.9,
            bounding_box=(
                100.0,
                100.0,
                150.0,
                150.0,
            ),
        ),
        FakeDetection(
            object_type="glass",
            confidence=0.2,
            bounding_box=(
                200.0,
                200.0,
                250.0,
                250.0,
            ),
        ),
    ]

    state = builder.build(
        detections
    )

    assert state.object_count == 1

    assert state.contains(
        "plate"
    )

    assert not state.contains(
        "glass"
    )


def test_scene_state_closest_to() -> None:

    objects = [
        VisionSceneObject(
            object_id="plate_1",
            object_type="plate",
            confidence=0.9,
            bounding_box=(
                0.0,
                0.0,
                10.0,
                10.0,
            ),
            pixel_center=(
                5.0,
                5.0,
            ),
            position=ScenePosition(
                x=-0.2,
                y=0.0,
                z=0.772,
            ),
        ),
        VisionSceneObject(
            object_id="plate_2",
            object_type="plate",
            confidence=0.9,
            bounding_box=(
                20.0,
                20.0,
                30.0,
                30.0,
            ),
            pixel_center=(
                25.0,
                25.0,
            ),
            position=ScenePosition(
                x=0.2,
                y=0.0,
                z=0.772,
            ),
        ),
    ]

    from tablemind.perception.vision_scene_state import (
        SceneState,
    )

    state = SceneState(
        objects=objects
    )

    closest = state.closest_to(
        object_type="plate",
        x=0.15,
        y=0.0,
    )

    assert closest.object_id == (
        "plate_2"
    )


def test_scene_state_serialization() -> None:

    builder = VisionSceneStateBuilder(
        calibration=make_calibration()
    )

    state = builder.build(
        [
            FakeDetection(
                object_type="plate",
                confidence=0.95,
                bounding_box=(
                    100.0,
                    100.0,
                    200.0,
                    200.0,
                ),
            )
        ]
    )

    payload = state.to_dict()

    assert payload[
        "source"
    ] == "openvino_calibrated_vision"

    assert len(
        payload["objects"]
    ) == 1

    assert (
        payload["objects"][0][
            "object_type"
        ]
        == "plate"
    )

    # Make sure it is actually JSON serializable.
    json.dumps(
        payload
    )