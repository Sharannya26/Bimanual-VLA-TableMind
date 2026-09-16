"""TABLEMIND — Vision → World grounding demo.

Milestone 8.8:
Compare the MuJoCo-native OpenVINO detector against the
known ground-truth MuJoCo object positions using the same
camera-to-world projection pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tablemind.perception.camera import TableCamera
from tablemind.perception.camera_grounding import CameraGrounder
from tablemind.perception.vision_detector import OpenVINOTableDetector
from tablemind.simulation import BimanualTableSimulation


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

TABLE_TOP_Z = 0.76

# Object-center heights used by the camera-to-world grounding model.
OBJECT_GROUNDING_HEIGHTS = {
    "plate": 0.772,
    "glass": 0.835,
}

# Ground-truth MuJoCo object positions.
GROUND_TRUTH = {
    "plate_1": ("plate", (-0.28, 0.00, 0.772)),
    "plate_2": ("plate", (0.28, 0.00, 0.772)),
    "glass_1": ("glass", (-0.40, -0.14, 0.835)),
    "glass_2": ("glass", (0.40, -0.14, 0.835)),
}

# IMPORTANT:
# This is the NEW MuJoCo-native model from Milestone 8.6.
MODEL_PATH = Path(
    "runs/detect/artifacts/tablemind_mujoco_training/"
    "plate_glass_mujoco_v1/weights/best_openvino_model"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def xy_distance(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    """Return XY-plane Euclidean distance in metres."""

    dx = first[0] - second[0]
    dy = first[1] - second[1]

    return float(np.sqrt(dx * dx + dy * dy))


def match_detections_to_ground_truth(
    detections,
    grounded_points,
):
    """Match each detection to the nearest unused ground-truth object.

    Matching is restricted by object type.
    """

    remaining = dict(GROUND_TRUTH)
    matches = []

    for detection, world_point in grounded_points:
        candidates = [
            (object_id, object_type, position)
            for object_id, (object_type, position) in remaining.items()
            if object_type == detection.object_type
        ]

        if not candidates:
            continue

        best_candidate = min(
            candidates,
            key=lambda candidate: xy_distance(
                world_point.xyz,
                candidate[2],
            ),
        )

        object_id, object_type, truth_position = best_candidate

        matches.append(
            (
                detection,
                world_point,
                object_id,
                object_type,
                truth_position,
            )
        )

        del remaining[object_id]

    return matches


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------


def main() -> None:
    print()
    print("=" * 64)
    print("TABLEMIND — VISION GROUNDED PERCEPTION")
    print("=" * 64)
    print()

    # -----------------------------------------------------------------------
    # 1. Create MuJoCo simulation
    # -----------------------------------------------------------------------

    simulation = BimanualTableSimulation.create()

    print("[1/5] MuJoCo simulation created.")

    # -----------------------------------------------------------------------
    # 2. Capture camera image
    # -----------------------------------------------------------------------

    camera = TableCamera(
        simulation.model,
        simulation.data,
        camera_name="table_overview",
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
    )

    image = camera.capture()

    print(
        f"[2/5] Camera captured image: "
        f"{image.shape[1]}x{image.shape[0]}"
    )

    # Save the exact image used by the detector.
    image_path = Path("artifacts/table_camera.png")
    image_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import cv2

        cv2.imwrite(
            str(image_path),
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        )
    except ImportError:
        pass

    # -----------------------------------------------------------------------
    # 3. Run NEW MuJoCo-native OpenVINO detector
    # -----------------------------------------------------------------------

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "MuJoCo-native OpenVINO model not found:\n"
            f"{MODEL_PATH.resolve()}"
        )

    detector = OpenVINOTableDetector(
        MODEL_PATH,
        image_size=640,
        confidence_threshold=0.25,
    )

    detections = detector.detect(image)

    print(
        f"[3/5] OpenVINO detected {len(detections)} objects."
    )

    print()
    print("## DETECTIONS")
    print()

    # -----------------------------------------------------------------------
    # 4. Camera → world grounding
    # -----------------------------------------------------------------------

    grounder = CameraGrounder(
        simulation.model,
        simulation.data,
        camera_name="table_overview",
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
    )

    grounded_points = []

    for index, detection in enumerate(detections, start=1):

        grounding_z = OBJECT_GROUNDING_HEIGHTS.get(
            detection.object_type
        )

        if grounding_z is None:
            print(
                f"Skipping unsupported object type: "
                f"{detection.object_type}"
            )
            continue

        pixel_x, pixel_y = detection.center

        world_point = grounder.pixel_to_world_on_plane(
            pixel_x,
            pixel_y,
            plane_z=grounding_z,
        )

        grounded_points.append(
            (
                detection,
                world_point,
            )
        )

        print(
            f"{index}. {detection.object_type} "
            f"confidence={detection.confidence:.3f}"
        )

        print(
            "   bbox: "
            f"({detection.bounding_box[0]:.1f}, "
            f"{detection.bounding_box[1]:.1f}, "
            f"{detection.bounding_box[2]:.1f}, "
            f"{detection.bounding_box[3]:.1f})"
        )

        print(
            "   center pixel: "
            f"({pixel_x:.1f}, {pixel_y:.1f})"
        )

        print(
            "   grounding height: "
            f"{grounding_z:.3f} m"
        )

        print(
            "   grounded world: "
            f"({world_point.x:.4f}, "
            f"{world_point.y:.4f}, "
            f"{world_point.z:.4f})"
        )

    # -----------------------------------------------------------------------
    # 5. Compare against MuJoCo ground truth
    # -----------------------------------------------------------------------

    matches = match_detections_to_ground_truth(
        detections,
        grounded_points,
    )

    print()
    print("## GROUNDING ERROR")
    print()

    errors = []

    for (
        detection,
        world_point,
        object_id,
        object_type,
        truth_position,
    ) in matches:

        error_m = xy_distance(
            world_point.xyz,
            truth_position,
        )

        error_cm = error_m * 100.0

        errors.append(error_cm)

        print(
            f"{object_id:<10} "
            f"{object_type:<6} "
            f"error={error_cm:.2f} cm"
        )

        print(
            "vision: "
            f"({world_point.x:.4f}, "
            f"{world_point.y:.4f}, "
            f"{world_point.z:.4f})"
        )

        print(
            "truth:  "
            f"({truth_position[0]:.4f}, "
            f"{truth_position[1]:.4f}, "
            f"{truth_position[2]:.4f})"
        )

        print()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------

    print("=" * 64)
    print("VISION GROUNDING SUMMARY")
    print("=" * 64)
    print()

    print(
        f"Detections:       {len(detections)}"
    )

    print(
        f"Matched objects:  {len(matches)}"
    )

    if errors:
        mean_error = float(np.mean(errors))
        max_error = float(np.max(errors))
        min_error = float(np.min(errors))

        print(
            f"Mean XY error:    {mean_error:.2f} cm"
        )

        print(
            f"Minimum XY error: {min_error:.2f} cm"
        )

        print(
            f"Maximum XY error: {max_error:.2f} cm"
        )

        print()
        print(
            "BASELINE (old detector): 8.59 cm"
        )

        improvement = 8.59 - mean_error
        improvement_percent = (
            improvement / 8.59 * 100.0
        )

        print(
            f"Change vs baseline: {improvement:+.2f} cm"
        )

        print(
            f"Relative improvement: "
            f"{improvement_percent:+.1f}%"
        )

    else:
        print("Mean XY error:    unavailable")

    print()
    print("VISION → WORLD GROUNDING COMPLETE")
    print()


if __name__ == "__main__":
    main()