"""Launch the TableMind milestone-one MuJoCo scene."""

from __future__ import annotations

import argparse
import time

import mujoco

from tablemind.simulation import BimanualTableSimulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the TableMind bimanual SO-101 MuJoCo scene.")
    parser.add_argument("--headless", action="store_true", help="Step without opening the MuJoCo viewer.")
    parser.add_argument("--steps", type=int, default=1_000, help="Headless step count (default: 1000).")
    args = parser.parse_args()

    simulation = BimanualTableSimulation.create()
    print(f"Loaded {simulation.model.nbody} bodies, {simulation.model.nu} actuators, arms={simulation.arms}")

    if args.headless:
        simulation.step(args.steps)
        print(f"Headless validation complete after {args.steps} steps (t={simulation.data.time:.3f}s).")
        return

    import mujoco.viewer

    with mujoco.viewer.launch_passive(simulation.model, simulation.data) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()
            simulation.step()
            viewer.sync()
            remaining = simulation.model.opt.timestep - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
