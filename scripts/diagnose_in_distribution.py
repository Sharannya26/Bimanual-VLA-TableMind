"""Test detector localization on an in-distribution MuJoCo scene."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco

from tablemind.perception.camera import TableCamera
from tablemind.perception.camera_grounding import CameraGrounder
from tablemind.perception.vision_detector import OpenVINOTableDetector
from tablemind.simulation import BimanualTableSimulation


IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

MODEL_PATH = Path(
    "runs/detect/artifacts/tablemind_mujoco_training/"
    "plate_glass_mujoco_v1/weights/best_openvino_model"
)

# These positions are deliberately closer to the image center
# than the normal runtime configuration.
TEST_POSITIONS = {
    "plate_1": ("plate", (-0.22, 0.00, 0.772)),
    "plate_2": ("plate", (0.22, 0.00, 0.772)),
    "glass_1": ("glass", (-0.25, -0.14, 0.835)),
    "glass_2": ("glass", (0.25, -0.14, 0.835)),
}


def project_world_point(
    grounder: CameraGrounder,
    world_point: tuple[float, float, float],
) -> tuple[float, float]:

    camera_position = grounder.camera_position
    camera_rotation = grounder.camera_rotation
    fx, fy = grounder.focal_lengths
    cx, cy = grounder.principal_point

    point_world = np.asarray(
        world_point,
        dtype=float,
    )

    point_camera = camera_rotation.T @ (
        point_world - camera_position
    )

    x_camera = point_camera[0]
    y_camera = point_camera[1]
    z_camera = point_camera[2]

    if z_camera >= 0:
        raise ValueError(
            "World point is behind the camera."
        )

    pixel_x = (
        cx
        + fx * (x_camera / -z_camera)
    )

    pixel_y = (
        cy
        - fy * (y_camera / -z_camera)
    )

    return (
        float(pixel_x),
        float(pixel_y),
    )


def pixel_distance(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:

    dx = first[0] - second[0]
    dy = first[1] - second[1]

    return float(
        np.sqrt(dx * dx + dy * dy)
    )


def set_object_position(
    simulation: BimanualTableSimulation,
    object_id: str,
    position: tuple[float, float, float],
) -> None:
    """Move a free-joint object to a new position."""

    body_id = mujoco.mj_name2id(
        simulation.model,
        mujoco.mjtObj.mjOBJ_BODY,
        object_id,
    )

    if body_id < 0:
        raise KeyError(
            f"Body '{object_id}' not found."
        )

    joint_id = simulation.model.body_jntadr[body_id]

    if joint_id < 0:
        raise ValueError(
            f"Body '{object_id}' has no joint."
        )

    qpos_address = simulation.model.jnt_qposadr[joint_id]

    simulation.data.qpos[qpos_address:qpos_address + 3] = np.asarray(
        position,
        dtype=float,
    )


def main() -> None:

    print()
    print("=" * 64)
    print("TABLEMIND — IN-DISTRIBUTION LOCALIZATION DIAGNOSTIC")
    print("=" * 64)
    print()

    # ------------------------------------------------------------------
    # 1. Create simulation
    # ------------------------------------------------------------------

    simulation = BimanualTableSimulation.create()

    print("[1/5] MuJoCo simulation created.")

    # ------------------------------------------------------------------
    # 2. Move objects to controlled in-distribution positions
    # ------------------------------------------------------------------

    for object_id, (_, position) in TEST_POSITIONS.items():
        set_object_position(
            simulation,
            object_id,
            position,
        )

    mujoco.mj_forward(
        simulation.model,
        simulation.data,
    )

    print(
        "[2/5] Objects moved to "
        "in-distribution test positions."
    )

    print()
    print("## TEST WORLD POSITIONS")
    print()

    for object_id, (
        object_type,
        position,
    ) in TEST_POSITIONS.items():

        print(
            f"{object_id:<10} "
            f"{object_type:<6} "
            f"({position[0]:+.3f}, "
            f"{position[1]:+.3f}, "
            f"{position[2]:.3f})"
        )

    # ------------------------------------------------------------------
    # 3. Capture image
    # ------------------------------------------------------------------

    camera = TableCamera(
        simulation.model,
        simulation.data,
        camera_name="table_overview",
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
    )

    image = camera.capture()

    print()
    print(
        f"[3/5] Camera captured "
        f"{image.shape[1]}x{image.shape[0]}"
    )

    # ------------------------------------------------------------------
    # 4. Ground-truth projection
    # ------------------------------------------------------------------

    grounder = CameraGrounder(
        simulation.model,
        simulation.data,
        camera_name="table_overview",
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
    )

    projected_truth = {}

    print()
    print("## GROUND-TRUTH PROJECTED PIXELS")
    print()

    for object_id, (
        object_type,
        position,
    ) in TEST_POSITIONS.items():

        pixel = project_world_point(
            grounder,
            position,
        )

        projected_truth[object_id] = pixel

        print(
            f"{object_id:<10} "
            f"{object_type:<6} "
            f"pixel=({pixel[0]:.2f}, "
            f"{pixel[1]:.2f})"
        )

    # ------------------------------------------------------------------
    # 5. Run detector
    # ------------------------------------------------------------------

    detector = OpenVINOTableDetector(
        MODEL_PATH,
        image_size=640,
        confidence_threshold=0.25,
    )

    detections = detector.detect(image)

    print()
    print(
        f"[4/5] OpenVINO detected "
        f"{len(detections)} objects."
    )

    print()
    print("## DETECTOR RESULTS")
    print()

    for index, detection in enumerate(
        detections,
        start=1,
    ):

        print(
            f"{index}. "
            f"{detection.object_type:<6} "
            f"confidence="
            f"{detection.confidence:.3f} "
            f"center="
            f"({detection.center[0]:.2f}, "
            f"{detection.center[1]:.2f})"
        )

    # ------------------------------------------------------------------
    # Match detections to nearest same-class truth
    # ------------------------------------------------------------------

    remaining = dict(TEST_POSITIONS)

    pixel_errors = []

    print()
    print("## LOCALIZATION ERROR")
    print()

    for detection in detections:

        candidates = [
            object_id
            for object_id, (
                object_type,
                _,
            ) in remaining.items()
            if object_type == detection.object_type
        ]

        if not candidates:
            continue

        best_object = min(
            candidates,
            key=lambda object_id:
            pixel_distance(
                detection.center,
                projected_truth[object_id],
            ),
        )

        truth_pixel = projected_truth[
            best_object
        ]

        detected_pixel = detection.center

        error = pixel_distance(
            detected_pixel,
            truth_pixel,
        )

        pixel_errors.append(error)

        dx = (
            detected_pixel[0]
            - truth_pixel[0]
        )

        dy = (
            detected_pixel[1]
            - truth_pixel[1]
        )

        print(
            f"{best_object:<10} "
            f"{detection.object_type:<6} "
            f"error={error:.2f} px"
        )

        print(
            f"   truth:    "
            f"({truth_pixel[0]:.2f}, "
            f"{truth_pixel[1]:.2f})"
        )

        print(
            f"   detected: "
            f"({detected_pixel[0]:.2f}, "
            f"{detected_pixel[1]:.2f})"
        )

        print(
            f"   delta:    "
            f"({dx:+.2f}, {dy:+.2f})"
        )

        print()

        del remaining[best_object]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print("=" * 64)
    print("IN-DISTRIBUTION DIAGNOSTIC SUMMARY")
    print("=" * 64)
    print()

    print(
        f"Detections:       {len(detections)}"
    )

    print(
        f"Matched objects:  {len(pixel_errors)}"
    )

    if pixel_errors:

        mean_error = float(
            np.mean(pixel_errors)
        )

        print(
            f"Mean pixel error: "
            f"{mean_error:.2f} px"
        )

        print(
            f"Minimum error:    "
            f"{np.min(pixel_errors):.2f} px"
        )

        print(
            f"Maximum error:    "
            f"{np.max(pixel_errors):.2f} px"
        )

    print()
    print(
        "[5/5] In-distribution diagnostic complete."
    )
    print()


if __name__ == "__main__":
    main()