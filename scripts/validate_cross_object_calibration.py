"""TABLEMIND 8.9.7 — Cross-object camera calibration validation.

Tests whether one plate-derived affine pixel->world mapping
also works for glasses.

If cross-object error is poor, a separate glass calibration
is fitted and compared.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from tablemind.simulation import BimanualTableSimulation


IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

TABLE_Z = {
    "plate": 0.772,
    "glass": 0.835,
}

OBJECTS = {
    "plate": {
        "body": "plate_1",
        "geom": "plate_1",
        "z": 0.772,
    },
    "glass": {
        "body": "glass_1",
        "geom": "glass_1",
        "z": 0.835,
    },
}

# Calibration grid.
CALIBRATION_POINTS = tuple(
    (x, y)
    for x in (-0.30, -0.20, -0.10, 0.00, 0.10, 0.20, 0.30)
    for y in (-0.10, -0.05, 0.00, 0.05)
)

# Completely held-out validation positions.
VALIDATION_POINTS = (
    (-0.27, -0.075),
    (-0.17, -0.025),
    (-0.07, 0.025),
    (0.07, -0.075),
    (0.17, -0.025),
    (0.27, 0.025),
)


OUTPUT_PATH = Path(
    "artifacts"
) / "camera_calibration_8_9_7.json"


def set_object_position(
    simulation: BimanualTableSimulation,
    body_name: str,
    position: tuple[float, float, float],
) -> None:

    body_id = mujoco.mj_name2id(
        simulation.model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )

    if body_id < 0:
        raise KeyError(
            f"Body '{body_name}' not found."
        )

    joint_id = simulation.model.body_jntadr[body_id]

    if joint_id < 0:
        raise ValueError(
            f"Body '{body_name}' has no joint."
        )

    qpos_address = simulation.model.jnt_qposadr[joint_id]

    simulation.data.qpos[
        qpos_address:qpos_address + 3
    ] = np.asarray(
        position,
        dtype=float,
    )

    mujoco.mj_forward(
        simulation.model,
        simulation.data,
    )


def bbox_center_from_segmentation(
    simulation: BimanualTableSimulation,
    geom_name: str,
) -> tuple[float, float]:

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

        segmentation = np.asarray(
            renderer.render()
        ).copy()

    finally:
        renderer.close()

    geom_id = mujoco.mj_name2id(
        simulation.model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_name,
    )

    if geom_id < 0:
        raise KeyError(
            f"Geom '{geom_name}' not found."
        )

    mask = (
        segmentation[:, :, 0]
        == geom_id
    )

    ys, xs = np.where(mask)

    if len(xs) == 0:
        raise RuntimeError(
            f"No segmentation pixels found for "
            f"geom '{geom_name}'."
        )

    return (
        float(xs.min() + xs.max()) / 2.0,
        float(ys.min() + ys.max()) / 2.0,
    )


def collect_samples(
    simulation: BimanualTableSimulation,
    object_type: str,
    points: tuple[tuple[float, float], ...],
) -> tuple[np.ndarray, np.ndarray]:

    object_info = OBJECTS[object_type]

    pixels: list[tuple[float, float]] = []
    world: list[tuple[float, float]] = []

    print()
    print(
        f"Collecting {object_type} samples..."
    )

    for x, y in points:

        set_object_position(
            simulation,
            object_info["body"],
            (
                x,
                y,
                object_info["z"],
            ),
        )

        pixel = bbox_center_from_segmentation(
            simulation,
            object_info["geom"],
        )

        pixels.append(pixel)
        world.append((x, y))

    return (
        np.asarray(pixels, dtype=float),
        np.asarray(world, dtype=float),
    )


def fit_affine(
    pixels: np.ndarray,
    world: np.ndarray,
) -> np.ndarray:

    design = np.column_stack(
        [
            pixels[:, 0],
            pixels[:, 1],
            np.ones(len(pixels)),
        ]
    )

    coefficients, *_ = np.linalg.lstsq(
        design,
        world,
        rcond=None,
    )

    return coefficients


def predict(
    pixels: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:

    design = np.column_stack(
        [
            pixels[:, 0],
            pixels[:, 1],
            np.ones(len(pixels)),
        ]
    )

    return design @ coefficients


def calculate_errors(
    predicted: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray:

    return np.linalg.norm(
        predicted - truth,
        axis=1,
    )


def print_coefficients(
    name: str,
    coefficients: np.ndarray,
) -> None:

    print()
    print(
        f"### {name.upper()} CALIBRATION"
    )
    print()

    print(
        "World X = "
        f"{coefficients[0, 0]:+.8f} * pixel_x "
        f"{coefficients[1, 0]:+.8f} * pixel_y "
        f"{coefficients[2, 0]:+.8f}"
    )

    print(
        "World Y = "
        f"{coefficients[0, 1]:+.8f} * pixel_x "
        f"{coefficients[1, 1]:+.8f} * pixel_y "
        f"{coefficients[2, 1]:+.8f}"
    )


def print_validation(
    object_type: str,
    pixels: np.ndarray,
    truth: np.ndarray,
    predicted: np.ndarray,
    errors: np.ndarray,
) -> None:

    print()
    print(
        f"## {object_type.upper()} VALIDATION"
    )
    print()

    for index in range(len(truth)):

        print(
            f"{index + 1}. "
            f"pixel=("
            f"{pixels[index, 0]:.2f}, "
            f"{pixels[index, 1]:.2f}) "
            f"truth=("
            f"{truth[index, 0]:+.3f}, "
            f"{truth[index, 1]:+.3f}) "
            f"pred=("
            f"{predicted[index, 0]:+.3f}, "
            f"{predicted[index, 1]:+.3f}) "
            f"error="
            f"{errors[index] * 100:.2f} cm"
        )

    print()

    print(
        f"{object_type} mean error: "
        f"{np.mean(errors) * 100:.3f} cm"
    )

    print(
        f"{object_type} max error: "
        f"{np.max(errors) * 100:.3f} cm"
    )


def save_calibration(
    plate_coefficients: np.ndarray,
    glass_coefficients: np.ndarray,
) -> None:

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "version": "8.9.7",
        "mapping": "affine_pixel_to_world",
        "image_width": IMAGE_WIDTH,
        "image_height": IMAGE_HEIGHT,
        "plate": {
            "z": OBJECTS["plate"]["z"],
            "coefficients": plate_coefficients.tolist(),
        },
        "glass": {
            "z": OBJECTS["glass"]["z"],
            "coefficients": glass_coefficients.tolist(),
        },
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Calibration saved to: "
        f"{OUTPUT_PATH}"
    )


def main() -> None:

    print()
    print("=" * 72)
    print(
        "TABLEMIND — MILESTONE 8.9.7"
    )
    print(
        "CROSS-OBJECT CALIBRATION VALIDATION"
    )
    print("=" * 72)

    simulation = (
        BimanualTableSimulation.create()
    )

    # ==============================================================
    # 1. COLLECT PLATE CALIBRATION DATA
    # ==============================================================

    plate_calibration_pixels, plate_calibration_world = (
        collect_samples(
            simulation,
            "plate",
            CALIBRATION_POINTS,
        )
    )

    plate_coefficients = fit_affine(
        plate_calibration_pixels,
        plate_calibration_world,
    )

    print_coefficients(
        "plate",
        plate_coefficients,
    )

    # ==============================================================
    # 2. COLLECT GLASS CALIBRATION DATA
    # ==============================================================

    glass_calibration_pixels, glass_calibration_world = (
        collect_samples(
            simulation,
            "glass",
            CALIBRATION_POINTS,
        )
    )

    glass_coefficients = fit_affine(
        glass_calibration_pixels,
        glass_calibration_world,
    )

    print_coefficients(
        "glass",
        glass_coefficients,
    )

    # ==============================================================
    # 3. HELD-OUT PLATE VALIDATION
    # ==============================================================

    plate_validation_pixels, plate_validation_world = (
        collect_samples(
            simulation,
            "plate",
            VALIDATION_POINTS,
        )
    )

    plate_predictions = predict(
        plate_validation_pixels,
        plate_coefficients,
    )

    plate_errors = calculate_errors(
        plate_predictions,
        plate_validation_world,
    )

    print_validation(
        "plate",
        plate_validation_pixels,
        plate_validation_world,
        plate_predictions,
        plate_errors,
    )

    # ==============================================================
    # 4. HELD-OUT GLASS VALIDATION
    # ==============================================================

    glass_validation_pixels, glass_validation_world = (
        collect_samples(
            simulation,
            "glass",
            VALIDATION_POINTS,
        )
    )

    glass_predictions = predict(
        glass_validation_pixels,
        glass_coefficients,
    )

    glass_errors = calculate_errors(
        glass_predictions,
        glass_validation_world,
    )

    print_validation(
        "glass",
        glass_validation_pixels,
        glass_validation_world,
        glass_predictions,
        glass_errors,
    )

    # ==============================================================
    # 5. CRITICAL CROSS-OBJECT TEST
    #
    # Apply PLATE calibration to GLASS observations.
    # ==============================================================

    glass_with_plate_calibration = predict(
        glass_validation_pixels,
        plate_coefficients,
    )

    cross_object_errors = calculate_errors(
        glass_with_plate_calibration,
        glass_validation_world,
    )

    print()
    print(
        "## CRITICAL CROSS-OBJECT TEST"
    )
    print()

    for index in range(
        len(glass_validation_world)
    ):

        print(
            f"{index + 1}. "
            f"glass pixel=("
            f"{glass_validation_pixels[index, 0]:.2f}, "
            f"{glass_validation_pixels[index, 1]:.2f}) "
            f"truth=("
            f"{glass_validation_world[index, 0]:+.3f}, "
            f"{glass_validation_world[index, 1]:+.3f}) "
            f"plate-calibration=("
            f"{glass_with_plate_calibration[index, 0]:+.3f}, "
            f"{glass_with_plate_calibration[index, 1]:+.3f}) "
            f"error="
            f"{cross_object_errors[index] * 100:.2f} cm"
        )

    cross_object_mean = np.mean(
        cross_object_errors
    )

    cross_object_max = np.max(
        cross_object_errors
    )

    glass_specific_mean = np.mean(
        glass_errors
    )

    plate_mean = np.mean(
        plate_errors
    )

    # ==============================================================
    # 6. SUMMARY
    # ==============================================================

    print()
    print("=" * 72)
    print("8.9.7 SUMMARY")
    print("=" * 72)
    print()

    print(
        f"Plate-specific mean error: "
        f"{plate_mean * 100:.3f} cm"
    )

    print(
        f"Glass-specific mean error: "
        f"{glass_specific_mean * 100:.3f} cm"
    )

    print(
        f"Plate calibration applied to glasses: "
        f"{cross_object_mean * 100:.3f} cm"
    )

    print(
        f"Cross-object maximum error: "
        f"{cross_object_max * 100:.3f} cm"
    )

    # ==============================================================
    # DECISION
    # ==============================================================

    print()

    if cross_object_mean <= 0.02:

        print(
            "✅ ONE SHARED CALIBRATION IS SUFFICIENT."
        )

        print(
            "The plate-derived mapping generalizes "
            "well to glasses."
        )

    else:

        print(
            "⚠️ OBJECT-SPECIFIC CALIBRATION IS RECOMMENDED."
        )

        print(
            "The glass-specific mapping should be used "
            "for glass detections."
        )

    # Save both mappings regardless.
    save_calibration(
        plate_coefficients,
        glass_coefficients,
    )

    print()
    print(
        "Milestone 8.9.7 validation complete."
    )
    print()


if __name__ == "__main__":
    main()