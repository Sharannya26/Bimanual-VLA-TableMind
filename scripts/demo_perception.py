"""Milestone 3.2 object perception demonstration."""

from __future__ import annotations

import time

import mujoco.viewer

from tablemind.perception import GroundTruthDetector, TableCamera
from tablemind.simulation import BimanualTableSimulation


def main() -> None:
    print("TABLEMIND - Milestone 3.2 Object Perception")
    print("=" * 50)

    simulation = BimanualTableSimulation.create()

    camera = TableCamera(
        simulation.model,
        simulation.data,
        camera_name="table_overview",
        width=640,
        height=480,
    )

    detector = GroundTruthDetector(simulation.model)

    try:
        image = camera.capture()
        observation = detector.detect()

        print()
        print(f"Camera frame: {image.shape}")
        print(f"Objects detected: {len(observation.objects)}")
        print()

        for obj in observation.objects:
            print(
                f"{obj.object_id:10s} | "
                f"type={obj.object_type:6s} | "
                f"position={obj.position}"
            )

        print()
        print("MuJoCo simulation is running.")
        print("Close the window to finish.")

        with mujoco.viewer.launch_passive(
            simulation.model,
            simulation.data,
        ) as viewer:

            while viewer.is_running():
                simulation.step(1)
                viewer.sync()
                time.sleep(0.01)

    finally:
        camera.close()

    print()
    print("Perception demo complete.")


if __name__ == "__main__":
    main()