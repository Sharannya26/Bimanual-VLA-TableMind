"""Validate 2D bounding-box-center to 3D grounding."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

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

TEST_POSITIONS = {
    "plate_1": ("plate", (-0.22, 0.00, 0.772)),
    "plate_2": ("plate", (+0.22, 0.00, 0.772)),
    "glass_1": ("glass", (-0.25, -0.14, 0.835)),
    "glass_2": ("glass", (+0.25, -0.14, 0.835)),
}

OBJECT_HEIGHTS = {
    "plate": 0.772,
    "glass": 0.835,
}


def set_object_position(
    simulation: BimanualTableSimulation,
    object_id: str,
    position: tuple[float, float, float],
) -> None:

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

    simulation.data.qpos[
        qpos_address:qpos_address + 3
    ] = np.asarray(
        position,
        dtype=float,
    )


def capture_segmentation(
    simulation: BimanualTableSimulation,
) -> np.ndarray:

    renderer = mujoco.Renderer(
        simulation.model,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
    )

    try:

        renderer.enable_segmentation_rendering()

        camera_id = mujoco.mj_name2id(
            simulation.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            "table_overview",
        )

        renderer.update_scene(
            simulation.data,
            camera=camera_id,
        )

        return np.asarray(
            renderer.render()
        ).copy()

    finally:
        renderer.close()


def segmentation_boxes(
    simulation: BimanualTableSimulation,
    segmentation: np.ndarray,
) -> dict[str, tuple[float, float, float, float]]:

    boxes = {}

    for object_id in TEST_POSITIONS:

        geom_id = mujoco.mj_name2id(
            simulation.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            object_id,
        )

        if geom_id < 0:
            continue

        mask = (
            segmentation[:, :, 0]
            == geom_id
        )

        ys, xs = np.where(mask)

        if len(xs) == 0:
            continue

        boxes[object_id] = (
            float(xs.min()),
            float(ys.min()),
            float(xs.max()),
            float(ys.max()),
        )

    return boxes


def box_center(
    box: tuple[float, float, float, float],
) -> tuple[float, float]:

    left, top, right, bottom = box

    return (
        (left + right) / 2.0,
        (top + bottom) / 2.0,
    )


def xy_error(
    estimated: tuple[float, float, float],
    truth: tuple[float, float, float],
) -> float:

    return float(
        np.linalg.norm(
            np.asarray(estimated[:2])
            - np.asarray(truth[:2])
        )
    )


def main() -> None:

    print()
    print("=" * 68)
    print("TABLEMIND — BBOX CENTER → WORLD GROUNDING DIAGNOSTIC")
    print("=" * 68)
    print()

    # ---------------------------------------------------------------
    # 1. Create controlled scene
    # ---------------------------------------------------------------

    simulation = BimanualTableSimulation.create()

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
        "[1/5] Controlled MuJoCo scene created."
    )

    # ---------------------------------------------------------------
    # 2. Capture camera image
    # ---------------------------------------------------------------

    camera = TableCamera(
        simulation.model,
        simulation.data,
        camera_name="table_overview",
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
    )

    image = camera.capture()

    print(
        "[2/5] Camera image captured."
    )

    # ---------------------------------------------------------------
    # 3. Ground-truth segmentation bbox centers
    # ---------------------------------------------------------------

    segmentation = capture_segmentation(
        simulation
    )

    truth_boxes = segmentation_boxes(
        simulation,
        segmentation,
    )

    truth_bbox_centers = {
        object_id: box_center(box)
        for object_id, box in truth_boxes.items()
    }

    print()
    print("## SEGMENTATION BBOX CENTERS")
    print()

    for object_id, center in truth_bbox_centers.items():

        print(
            f"{object_id:<10} "
            f"center=({center[0]:.2f}, "
            f"{center[1]:.2f})"
        )

    # ---------------------------------------------------------------
    # 4. Run YOLO
    # ---------------------------------------------------------------

    detector = OpenVINOTableDetector(
        MODEL_PATH,
        image_size=640,
        confidence_threshold=0.25,
    )

    detections = detector.detect(image)

    print()
    print(
        f"[3/5] OpenVINO detected "
        f"{len(detections)} objects."
    )

    # Match each detection to nearest segmentation center
    remaining = dict(truth_bbox_centers)

    matched_detections = []

    for detection in detections:

        candidates = [
            object_id
            for object_id in remaining
            if TEST_POSITIONS[object_id][0]
            == detection.object_type
        ]

        if not candidates:
            continue

        best_object = min(
            candidates,
            key=lambda object_id:
            np.linalg.norm(
                np.asarray(detection.center)
                - np.asarray(
                    remaining[object_id]
                )
            ),
        )

        matched_detections.append(
            (
                best_object,
                detection,
            )
        )

        del remaining[best_object]

    # ---------------------------------------------------------------
    # 5. Ground bbox centers into world coordinates
    # ---------------------------------------------------------------

    grounder = CameraGrounder(
        simulation.model,
        simulation.data,
        camera_name="table_overview",
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
    )

    print()
    print("## WORLD GROUNDING")
    print()

    errors = []

    for object_id, detection in matched_detections:

        object_type = TEST_POSITIONS[
            object_id
        ][0]

        truth_position = TEST_POSITIONS[
            object_id
        ][1]

        bbox_center = detection.center

        grounding_z = OBJECT_HEIGHTS[
            object_type
        ]

        grounded = grounder.pixel_to_world_on_plane(
            bbox_center[0],
            bbox_center[1],
            plane_z=grounding_z,
        )

        estimated_position = (
            grounded.x,
            grounded.y,
            grounded.z,
        )

        error = xy_error(
            estimated_position,
            truth_position,
        )

        errors.append(error)

        print(
            f"{object_id:<10} "
            f"{object_type:<6}"
        )

        print(
            f"   bbox center: "
            f"({bbox_center[0]:.2f}, "
            f"{bbox_center[1]:.2f})"
        )

        print(
            f"   grounded:    "
            f"({estimated_position[0]:+.4f}, "
            f"{estimated_position[1]:+.4f}, "
            f"{estimated_position[2]:.4f})"
        )

        print(
            f"   truth:       "
            f"({truth_position[0]:+.4f}, "
            f"{truth_position[1]:+.4f}, "
            f"{truth_position[2]:.4f})"
        )

        print(
            f"   XY error:    "
            f"{error * 100:.2f} cm"
        )

        print()

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print("=" * 68)
    print("GROUNDING SUMMARY")
    print("=" * 68)
    print()

    if errors:

        print(
            f"Objects evaluated: "
            f"{len(errors)}"
        )

        print(
            f"Mean XY error:     "
            f"{np.mean(errors) * 100:.2f} cm"
        )

        print(
            f"Minimum XY error:  "
            f"{np.min(errors) * 100:.2f} cm"
        )

        print(
            f"Maximum XY error:  "
            f"{np.max(errors) * 100:.2f} cm"
        )

    print()
    print(
        "[5/5] Bbox-center grounding diagnostic complete."
    )
    print()


if __name__ == "__main__":
    main()