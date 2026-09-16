"""Simple visual SO-101 movement test."""

from __future__ import annotations

import time

import mujoco
import mujoco.viewer

from tablemind.control import SO101Controller
from tablemind.simulation import BimanualTableSimulation


def main() -> None:
    simulation = BimanualTableSimulation.create()
    controller = SO101Controller(simulation)

    arm = "left"

    start = controller.end_effector_position(arm)

    # Simple target: move mostly upward from the starting position.
    target = (
        start[0],
        start[1],
        start[2],
    )

    print("SO-101 SIMPLE MOVEMENT TEST")
    print(f"Starting position: {start}")
    print(f"Target position  : {target}")

    with mujoco.viewer.launch_passive(
        simulation.model,
        simulation.data,
    ) as viewer:

        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.distance = 1.8
        viewer.cam.lookat[:] = [-0.1, 0.0, 0.8]

        for _ in range(50):
            mujoco.mj_step(
                simulation.model,
                simulation.data,
            )
            viewer.sync()
            time.sleep(0.01)

        print("\nMoving 10 cm upward...")

        result = controller.move_to(
            arm,
            target,
            tolerance=0.02,
            max_steps=100,
        )

        print(f"Reached       : {result.reached}")
        print(f"Final position: {result.final_position}")
        print(f"Final error   : {result.position_error:.4f} m")

        print("\nClose the MuJoCo window to exit.")

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()