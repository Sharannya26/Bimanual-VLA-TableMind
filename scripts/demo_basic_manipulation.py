"""Run the deterministic Milestone 2 SO-101 reach and gripper demonstration."""

from __future__ import annotations

from tablemind.control import SO101Controller
from tablemind.simulation import BimanualTableSimulation


def _report(label: str, reached: bool, error: float) -> None:
    status = "reached" if reached else "not reached"
    print(f"{label}: {status}; final Cartesian error={error:.3f} m")


def main() -> None:
    simulation = BimanualTableSimulation.create()
    controller = SO101Controller(simulation)
    arm = "left"
    plate = controller.object_position("plate_1")
    approach = (plate[0], plate[1], plate[2] + 0.16)
    contact = (plate[0], plate[1], plate[2] + 0.05)
    lift = (plate[0], plate[1], plate[2] + 0.20)
    placement = (-0.10, 0.02, plate[2] + 0.16)

    print("Milestone 2 deterministic reach/gripper demo (left SO-101)")
    controller.open_gripper(arm)
    result = controller.move_to(arm, approach)
    _report("Approach plate_1", result.reached, result.position_error)
    result = controller.move_to(arm, contact)
    _report("Lower toward plate_1", result.reached, result.position_error)
    controller.close_gripper(arm)
    result = controller.move_to(arm, lift)
    _report("Lift", result.reached, result.position_error)
    result = controller.move_to(arm, placement)
    _report("Move to placement location", result.reached, result.position_error)
    controller.open_gripper(arm)

    pick = controller.execute_pick(arm, "plate_1")
    print(f"Physical pick completed: {pick.completed}. {pick.reason}")
    print("The plate is fixed placeholder geometry in Milestone 2; no object transfer is claimed.")


if __name__ == "__main__":
    main()
