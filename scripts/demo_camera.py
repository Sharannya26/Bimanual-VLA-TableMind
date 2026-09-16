"""Milestone 3 camera demonstration."""

from __future__ import annotations

import time

import mujoco.viewer

from tablemind.perception.camera import TableCamera
from tablemind.simulation import BimanualTableSimulation


def main() -> None:
    print("TABLEMIND - Milestone 3 Camera Demo")
    print("=" * 45)

    simulation = BimanualTableSimulation.create()

    camera = TableCamera(
        simulation.model,
        simulation.data,
        camera_name="table_overview",
        width=960,
        height=720,
    )

    image = camera.capture()

    print(f"Camera frame shape: {image.shape}")
    print(f"Camera frame dtype: {image.dtype}")

    try:
        with mujoco.viewer.launch_passive(
            simulation.model,
            simulation.data,
        ) as viewer:

            print()
            print("MuJoCo simulation is running.")
            print("Camera frame captured successfully.")
            print("Close the MuJoCo window to finish the demo.")

            while viewer.is_running():
                simulation.step(1)
                viewer.sync()
                time.sleep(0.01)

    finally:
        camera.close()

    print("Camera demo complete.")


if __name__ == "__main__":
    main()