"""Calibrate camera bbox pixels to MuJoCo world coordinates."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from tablemind.simulation import BimanualTableSimulation


IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Calibration grid.
CALIBRATION_X = (-0.30, -0.20, -0.10, 0.00, 0.10, 0.20, 0.30)
CALIBRATION_Y = (-0.10, -0.05, 0.00, 0.05)

# Held-out points. These are deliberately different from
# the calibration grid.
VALIDATION_POINTS = (
    (-0.27, -0.075),
    (-0.17, -0.025),
    (-0.07, 0.025),
    (0.07, -0.075),
    (0.17, -0.025),
    (0.27, 0.025),
)

OBJECTS = {
    "plate_1": ("plate", 0.772),
    "plate_2": ("plate", 0.772),
}

PLATE_Y_OFFSET = 0.0


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
            f"No segmentation pixels found for {geom_name}."
        )

    return (
        float(xs.min() + xs.max()) / 2.0,
        float(ys.min() + ys.max()) / 2.0,
    )


def fit_affine(
    pixels: np.ndarray,
    world: np.ndarray,
) -> np.ndarray:

    # [u, v, 1]
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


def main() -> None:

    print()
    print("=" * 68)
    print("TABLEMIND — BBOX → WORLD AFFINE CALIBRATION")
    print("=" * 68)
    print()

    simulation = BimanualTableSimulation.create()

    calibration_pixels = []
    calibration_world = []

    # ---------------------------------------------------------------
    # Generate calibration samples
    # ---------------------------------------------------------------

    print("## GENERATING CALIBRATION DATA")
    print()

    sample_number = 0

    for x in CALIBRATION_X:
        for y in CALIBRATION_Y:

            # Keep the object safely above the table.
            z = 0.772

            position = (
                x,
                y,
                z,
            )

            set_object_position(
                simulation,
                "plate_1",
                position,
            )

            mujoco.mj_forward(
                simulation.model,
                simulation.data,
            )

            pixel = bbox_center_from_segmentation(
                simulation,
                "plate_1",
            )

            calibration_pixels.append(pixel)
            calibration_world.append(
                (x, y)
            )

            sample_number += 1

    calibration_pixels = np.asarray(
        calibration_pixels,
        dtype=float,
    )

    calibration_world = np.asarray(
        calibration_world,
        dtype=float,
    )

    print(
        f"Calibration samples: {sample_number}"
    )

    # ---------------------------------------------------------------
    # Fit affine mapping
    # ---------------------------------------------------------------

    coefficients = fit_affine(
        calibration_pixels,
        calibration_world,
    )

    print()
    print("## CALIBRATION COEFFICIENTS")
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

    # ---------------------------------------------------------------
    # Calibration-set residual
    # ---------------------------------------------------------------

    calibration_predictions = predict(
        calibration_pixels,
        coefficients,
    )

    calibration_errors = np.linalg.norm(
        calibration_predictions
        - calibration_world,
        axis=1,
    )

    print()
    print("## CALIBRATION RESIDUAL")
    print()

    print(
        f"Mean calibration error: "
        f"{np.mean(calibration_errors) * 100:.3f} cm"
    )

    print(
        f"Maximum calibration error: "
        f"{np.max(calibration_errors) * 100:.3f} cm"
    )

    # ---------------------------------------------------------------
    # Held-out validation
    # ---------------------------------------------------------------

    print()
    print("## HELD-OUT VALIDATION")
    print()

    validation_pixels = []
    validation_world = []

    for x, y in VALIDATION_POINTS:

        position = (
            x,
            y,
            0.772,
        )

        set_object_position(
            simulation,
            "plate_1",
            position,
        )

        mujoco.mj_forward(
            simulation.model,
            simulation.data,
        )

        pixel = bbox_center_from_segmentation(
            simulation,
            "plate_1",
        )

        validation_pixels.append(pixel)
        validation_world.append(
            (x, y)
        )

    validation_pixels = np.asarray(
        validation_pixels,
        dtype=float,
    )

    validation_world = np.asarray(
        validation_world,
        dtype=float,
    )

    validation_predictions = predict(
        validation_pixels,
        coefficients,
    )

    validation_errors = np.linalg.norm(
        validation_predictions
        - validation_world,
        axis=1,
    )

    for index in range(
        len(validation_world)
    ):

        truth = validation_world[index]
        predicted = validation_predictions[index]
        pixel = validation_pixels[index]

        error = validation_errors[index]

        print(
            f"{index + 1}. "
            f"pixel=({pixel[0]:.2f}, "
            f"{pixel[1]:.2f}) "
            f"truth=("
            f"{truth[0]:+.3f}, "
            f"{truth[1]:+.3f}) "
            f"pred=("
            f"{predicted[0]:+.3f}, "
            f"{predicted[1]:+.3f}) "
            f"error={error * 100:.2f} cm"
        )

    print()
    print("=" * 68)
    print("CALIBRATION SUMMARY")
    print("=" * 68)
    print()

    print(
        f"Validation samples: "
        f"{len(validation_errors)}"
    )

    print(
        f"Mean validation error: "
        f"{np.mean(validation_errors) * 100:.3f} cm"
    )

    print(
        f"Minimum validation error: "
        f"{np.min(validation_errors) * 100:.3f} cm"
    )

    print(
        f"Maximum validation error: "
        f"{np.max(validation_errors) * 100:.3f} cm"
    )

    print()
    print(
        "Calibration experiment complete."
    )
    print()


if __name__ == "__main__":
    main()