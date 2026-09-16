"""Diagnose TABLEMIND camera projection vs YOLO detections."""

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

GROUND_TRUTH = {
    "plate_1": ("plate", (-0.28, 0.00, 0.772)),
    "plate_2": ("plate", (0.28, 0.00, 0.772)),
    "glass_1": ("glass", (-0.40, -0.14, 0.835)),
    "glass_2": ("glass", (0.40, -0.14, 0.835)),
}


def project_world_point(
    grounder: CameraGrounder,
    world_point: tuple[float, float, float],
) -> tuple[float, float]:
    """Project a world-space point into image pixels."""

    camera_position = grounder.camera_position
    camera_rotation = grounder.camera_rotation
    fx, fy = grounder.focal_lengths
    cx, cy = grounder.principal_point

    point_world = np.asarray(world_point, dtype=float)

    # MuJoCo camera rotation matrix maps camera coordinates to world.
    # Therefore world -> camera uses its transpose.
    point_camera = camera_rotation.T @ (
        point_world - camera_position
    )

    x_camera = point_camera[0]
    y_camera = point_camera[1]
    z_camera = point_camera[2]

    if z_camera >= 0:
        raise ValueError(
            "Point is not in front of the camera."
        )

    pixel_x = cx + fx * (x_camera / -z_camera)
    pixel_y = cy - fy * (y_camera / -z_camera)

    return float(pixel_x), float(pixel_y)


def pixel_distance(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    """Return Euclidean pixel distance."""

    dx = first[0] - second[0]
    dy = first[1] - second[1]

    return float(np.sqrt(dx * dx + dy * dy))


def main() -> None:
    print()
    print("=" * 64)
    print("TABLEMIND — GROUNDING CALIBRATION DIAGNOSTIC")
    print("=" * 64)
    print()

    # ------------------------------------------------------------------
    # 1. Create simulation
    # ------------------------------------------------------------------

    simulation = BimanualTableSimulation.create()

    print("[1/4] MuJoCo simulation created.")

    # ------------------------------------------------------------------
    # 2. Create camera + grounding system
    # ------------------------------------------------------------------

    camera = TableCamera(
        simulation.model,
        simulation.data,
        camera_name="table_overview",
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
    )

    grounder = CameraGrounder(
        simulation.model,
        simulation.data,
        camera_name="table_overview",
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
    )

    image = camera.capture()

    print(
        f"[2/4] Camera captured {image.shape[1]}x{image.shape[0]}"
    )

    # ------------------------------------------------------------------
    # 3. Project ground-truth 3D positions into image
    # ------------------------------------------------------------------

    print()
    print("## GROUND-TRUTH → IMAGE PROJECTION")
    print()

    projected_truth = {}

    for object_id, (object_type, world_position) in GROUND_TRUTH.items():

        pixel = project_world_point(
            grounder,
            world_position,
        )

        projected_truth[object_id] = pixel

        print(
            f"{object_id:<10} "
            f"{object_type:<6} "
            f"world=({world_position[0]:.3f}, "
            f"{world_position[1]:.3f}, "
            f"{world_position[2]:.3f})"
        )

        print(
            f"           projected pixel="
            f"({pixel[0]:.2f}, {pixel[1]:.2f})"
        )

    # ------------------------------------------------------------------
    # 4. Run YOLO/OpenVINO detector
    # ------------------------------------------------------------------

    print()

    detector = OpenVINOTableDetector(
        MODEL_PATH,
        image_size=640,
        confidence_threshold=0.25,
    )

    detections = detector.detect(image)

    print(
        f"[3/4] OpenVINO detected {len(detections)} objects."
    )

    print()
    print("## YOLO DETECTIONS")
    print()

    for index, detection in enumerate(detections, start=1):

        print(
            f"{index}. "
            f"{detection.object_type:<6} "
            f"confidence={detection.confidence:.3f} "
            f"center=({detection.center[0]:.2f}, "
            f"{detection.center[1]:.2f})"
        )

    # ------------------------------------------------------------------
    # Match each detection to nearest same-type ground truth
    # ------------------------------------------------------------------

    remaining = dict(GROUND_TRUTH)

    print()
    print("## PIXEL PROJECTION ERROR")
    print()

    pixel_errors = []

    for detection in detections:

        candidates = [
            (
                object_id,
                object_type,
                world_position,
            )
            for object_id, (
                object_type,
                world_position,
            ) in remaining.items()
            if object_type == detection.object_type
        ]

        if not candidates:
            continue

        best = min(
            candidates,
            key=lambda candidate: pixel_distance(
                detection.center,
                projected_truth[candidate[0]],
            ),
        )

        object_id, object_type, world_position = best

        truth_pixel = projected_truth[object_id]
        detected_pixel = detection.center

        error = pixel_distance(
            detected_pixel,
            truth_pixel,
        )

        pixel_errors.append(error)

        print(
            f"{object_id:<10} "
            f"{object_type:<6} "
            f"pixel error={error:.2f} px"
        )

        print(
            f"   truth pixel:    "
            f"({truth_pixel[0]:.2f}, "
            f"{truth_pixel[1]:.2f})"
        )

        print(
            f"   detected pixel: "
            f"({detected_pixel[0]:.2f}, "
            f"{detected_pixel[1]:.2f})"
        )

        print(
            f"   pixel delta:    "
            f"({detected_pixel[0] - truth_pixel[0]:+.2f}, "
            f"{detected_pixel[1] - truth_pixel[1]:+.2f})"
        )

        print()

        del remaining[object_id]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print("=" * 64)
    print("CALIBRATION DIAGNOSTIC SUMMARY")
    print("=" * 64)
    print()

    if pixel_errors:
        print(
            f"Matched detections: {len(pixel_errors)}"
        )

        print(
            f"Mean pixel error:   "
            f"{np.mean(pixel_errors):.2f} px"
        )

        print(
            f"Maximum pixel error:"
            f" {np.max(pixel_errors):.2f} px"
        )

        print(
            f"Minimum pixel error:"
            f" {np.min(pixel_errors):.2f} px"
        )

    else:
        print("No detections could be matched.")

    print()
    print("[4/4] Diagnostic complete.")
    print()


if __name__ == "__main__":
    main()