"""Compare MuJoCo segmentation boxes against YOLO prediction boxes."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from tablemind.perception.camera import TableCamera
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
    "plate_2": ("plate", (0.22, 0.00, 0.772)),
    "glass_1": ("glass", (-0.25, -0.14, 0.835)),
    "glass_2": ("glass", (0.25, -0.14, 0.835)),
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

        segmentation = renderer.render()

        return np.asarray(
            segmentation
        ).copy()

    finally:
        renderer.close()


def segmentation_boxes(
    simulation: BimanualTableSimulation,
    segmentation: np.ndarray,
) -> dict[str, tuple[float, float, float, float]]:

    boxes = {}

    object_ids = {
        "plate_1": "plate_1",
        "plate_2": "plate_2",
        "glass_1": "glass_1",
        "glass_2": "glass_2",
    }

    for logical_id, geom_name in object_ids.items():

        geom_id = mujoco.mj_name2id(
            simulation.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_name,
        )

        if geom_id < 0:
            continue

        mask = (
            segmentation[:, :, 0]
            == geom_id
        )

        ys, xs = np.where(mask)

        if len(xs) == 0:
            print(
                f"WARNING: no segmentation pixels "
                f"for {logical_id}"
            )
            continue

        left = float(xs.min())
        right = float(xs.max())
        top = float(ys.min())
        bottom = float(ys.max())

        boxes[logical_id] = (
            left,
            top,
            right,
            bottom,
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


def box_width(
    box: tuple[float, float, float, float],
) -> float:

    return box[2] - box[0]


def box_height(
    box: tuple[float, float, float, float],
) -> float:

    return box[3] - box[1]


def box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:

    first_left, first_top, first_right, first_bottom = first
    second_left, second_top, second_right, second_bottom = second

    intersection_left = max(
        first_left,
        second_left,
    )

    intersection_top = max(
        first_top,
        second_top,
    )

    intersection_right = min(
        first_right,
        second_right,
    )

    intersection_bottom = min(
        first_bottom,
        second_bottom,
    )

    intersection_width = max(
        0.0,
        intersection_right - intersection_left,
    )

    intersection_height = max(
        0.0,
        intersection_bottom - intersection_top,
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    first_area = (
        max(0.0, first_right - first_left)
        * max(0.0, first_bottom - first_top)
    )

    second_area = (
        max(0.0, second_right - second_left)
        * max(0.0, second_bottom - second_top)
    )

    union_area = (
        first_area
        + second_area
        - intersection_area
    )

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def main() -> None:

    print()
    print("=" * 64)
    print("TABLEMIND — BOUNDING BOX GEOMETRY DIAGNOSTIC")
    print("=" * 64)
    print()

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
        "[1/4] Controlled MuJoCo scene created."
    )

    segmentation = capture_segmentation(
        simulation
    )

    truth_boxes = segmentation_boxes(
        simulation,
        segmentation,
    )

    print(
        "[2/4] Ground-truth segmentation boxes extracted."
    )

    camera = TableCamera(
        simulation.model,
        simulation.data,
        camera_name="table_overview",
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
    )

    image = camera.capture()

    detector = OpenVINOTableDetector(
        MODEL_PATH,
        image_size=640,
        confidence_threshold=0.25,
    )

    detections = detector.detect(image)

    print(
        f"[3/4] YOLO detected "
        f"{len(detections)} objects."
    )

    print()
    print("## GROUND-TRUTH BOXES")
    print()

    for object_id, box in truth_boxes.items():

        center = box_center(box)

        print(
            f"{object_id:<10} "
            f"box=({box[0]:.1f}, "
            f"{box[1]:.1f}, "
            f"{box[2]:.1f}, "
            f"{box[3]:.1f}) "
            f"center=({center[0]:.1f}, "
            f"{center[1]:.1f}) "
            f"size=({box_width(box):.1f} x "
            f"{box_height(box):.1f})"
        )

    print()
    print("## YOLO BOXES")
    print()

    for index, detection in enumerate(
        detections,
        start=1,
    ):

        box = detection.bounding_box
        center = detection.center

        print(
            f"{index}. "
            f"{detection.object_type:<6} "
            f"confidence={detection.confidence:.3f} "
            f"box=({box[0]:.1f}, "
            f"{box[1]:.1f}, "
            f"{box[2]:.1f}, "
            f"{box[3]:.1f}) "
            f"center=({center[0]:.1f}, "
            f"{center[1]:.1f}) "
            f"size=({box_width(box):.1f} x "
            f"{box_height(box):.1f})"
        )

    print()
    print("## MATCHED BOX COMPARISON")
    print()

    remaining = dict(truth_boxes)

    ious = []
    center_errors = []

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
            key=lambda object_id: (
                (
                    box_center(
                        remaining[object_id]
                    )[0]
                    - detection.center[0]
                ) ** 2
                +
                (
                    box_center(
                        remaining[object_id]
                    )[1]
                    - detection.center[1]
                ) ** 2
            ),
        )

        truth_box = remaining[best_object]
        predicted_box = detection.bounding_box

        truth_center = box_center(
            truth_box
        )

        predicted_center = detection.center

        center_error = float(
            np.linalg.norm(
                np.asarray(predicted_center)
                - np.asarray(truth_center)
            )
        )

        iou = box_iou(
            truth_box,
            predicted_box,
        )

        ious.append(iou)
        center_errors.append(center_error)

        print(
            f"{best_object:<10} "
            f"IoU={iou:.3f} "
            f"center_error={center_error:.2f}px"
        )

        print(
            f"   truth box: "
            f"({truth_box[0]:.1f}, "
            f"{truth_box[1]:.1f}, "
            f"{truth_box[2]:.1f}, "
            f"{truth_box[3]:.1f})"
        )

        print(
            f"   YOLO box:  "
            f"({predicted_box[0]:.1f}, "
            f"{predicted_box[1]:.1f}, "
            f"{predicted_box[2]:.1f}, "
            f"{predicted_box[3]:.1f})"
        )

        print()

        del remaining[best_object]

    print("=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print()

    if ious:

        print(
            f"Mean IoU:          "
            f"{np.mean(ious):.3f}"
        )

        print(
            f"Mean center error: "
            f"{np.mean(center_errors):.2f} px"
        )

    print()
    print(
        "[4/4] Bounding-box geometry diagnostic complete."
    )
    print()


if __name__ == "__main__":
    main()