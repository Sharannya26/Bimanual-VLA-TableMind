from __future__ import annotations

import mujoco

from tablemind.integration.closed_loop import ClosedLoopVLA
from tablemind.simulation.runtime import BimanualTableSimulation


def get_free_body_position(
    simulation: BimanualTableSimulation,
    body_name: str,
) -> tuple[float, float, float]:
    """Return the current XYZ position of a freejoint body."""

    body_id = mujoco.mj_name2id(
        simulation.model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )

    if body_id < 0:
        raise RuntimeError(
            f"Could not find body '{body_name}'."
        )

    joint_id = simulation.model.body_jntadr[body_id]

    if joint_id < 0:
        raise RuntimeError(
            f"Body '{body_name}' does not have a joint."
        )

    qpos_address = simulation.model.jnt_qposadr[joint_id]

    return tuple(
        float(value)
        for value in simulation.data.qpos[
            qpos_address : qpos_address + 3
        ]
    )


def set_free_body_position(
    simulation: BimanualTableSimulation,
    body_name: str,
    position: tuple[float, float, float],
) -> None:
    """Set the XYZ position of a freejoint body."""

    body_id = mujoco.mj_name2id(
        simulation.model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )

    if body_id < 0:
        raise RuntimeError(
            f"Could not find body '{body_name}'."
        )

    joint_id = simulation.model.body_jntadr[body_id]

    if joint_id < 0:
        raise RuntimeError(
            f"Body '{body_name}' does not have a joint."
        )

    qpos_address = simulation.model.jnt_qposadr[joint_id]

    simulation.data.qpos[
        qpos_address : qpos_address + 3
    ] = position


def print_verification(
    verification,
    prefix: str = "  ",
) -> None:
    """Print detailed verification coordinates."""

    if verification is None:
        return

    object_id = getattr(
        verification,
        "object_id",
        "unknown",
    )

    expected = getattr(
        verification,
        "expected_position",
        None,
    )

    actual = getattr(
        verification,
        "actual_position",
        None,
    )

    error = getattr(
        verification,
        "position_error",
        None,
    )

    success = getattr(
        verification,
        "success",
        None,
    )

    print(
        f"{prefix}{object_id}: "
        f"{'PASS' if success else 'FAIL'}"
    )

    if expected is not None:
        print(
            f"{prefix}  expected = "
            f"({expected[0]:.5f}, "
            f"{expected[1]:.5f}, "
            f"{expected[2]:.5f})"
        )

    if actual is not None:
        print(
            f"{prefix}  actual   = "
            f"({actual[0]:.5f}, "
            f"{actual[1]:.5f}, "
            f"{actual[2]:.5f})"
        )

    if error is not None:
        print(
            f"{prefix}  error    = "
            f"{error:.5f} m"
        )


def print_batch_verifications(result) -> None:
    """Print detailed verification information from all batches."""

    batch_results = getattr(
        result,
        "batch_results",
        None,
    )

    if not batch_results:
        print("No batch verification details exposed.")
        return

    print()
    print("=" * 72)
    print("DETAILED VERIFICATION")
    print("=" * 72)

    for batch_index, batch in enumerate(
        batch_results,
        start=1,
    ):
        print()
        print(f"BATCH {batch_index}")

        verification = getattr(
            batch,
            "verification",
            None,
        )

        if verification is None:
            print("  No verification object exposed.")
            continue

        if isinstance(
            verification,
            (tuple, list),
        ):
            for item in verification:
                print_verification(item)
        else:
            print_verification(verification)


def main() -> None:
    print()
    print("=" * 72)
    print("TABLEMIND — 9.4.3 REAL CLOSED-LOOP VLA DEMO")
    print("=" * 72)
    print()

    simulation = BimanualTableSimulation.create()

    system = ClosedLoopVLA.create(
        simulation=simulation,
    )

    # ------------------------------------------------------------------
    # DEMO SCENARIO
    # ------------------------------------------------------------------
    #
    # Keep the canonical scene unchanged for the test suite.
    #
    # The real SO-101 reachability diagnostic established that both
    # arms can reliably reach the following symmetric plate positions:
    #
    #   left  = (-0.18, -0.04)
    #   right = (+0.18, -0.04)
    #
    # We therefore prepare ONLY this demo instance inside that proven
    # reachable workspace.
    # ------------------------------------------------------------------

    plate_1 = get_free_body_position(
        simulation,
        "plate_1",
    )

    plate_2 = get_free_body_position(
        simulation,
        "plate_2",
    )

    set_free_body_position(
        simulation,
        "plate_1",
        (
            -0.18,
            -0.04,
            plate_1[2],
        ),
    )

    set_free_body_position(
        simulation,
        "plate_2",
        (
            0.18,
            -0.04,
            plate_2[2],
        ),
    )

    mujoco.mj_forward(
        simulation.model,
        simulation.data,
    )

    print("Starting scene prepared.")
    print("plates: plate_1, plate_2")
    print("glasses: glass_1, glass_2")
    print()
    print("Demo plate positions:")
    print(
        f"  plate_1 = "
        f"({get_free_body_position(simulation, 'plate_1')[0]:.3f}, "
        f"{get_free_body_position(simulation, 'plate_1')[1]:.3f}, "
        f"{get_free_body_position(simulation, 'plate_1')[2]:.3f})"
    )
    print(
        f"  plate_2 = "
        f"({get_free_body_position(simulation, 'plate_2')[0]:.3f}, "
        f"{get_free_body_position(simulation, 'plate_2')[1]:.3f}, "
        f"{get_free_body_position(simulation, 'plate_2')[2]:.3f})"
    )
    print()
    print(
        "Plates prepared inside the proven bimanual workspace."
    )
    print()

    # ------------------------------------------------------------------
    # FULL CLOSED-LOOP PIPELINE
    # ------------------------------------------------------------------

    result = system.run(
        "Set the table for two.",
    )

    print_batch_verifications(result)

    print()
    print("=" * 72)
    print("FINAL RESULT")
    print("=" * 72)
    print()

    print(
        f"success: "
        f"{getattr(result, 'success', 'unknown')}"
    )

    if result.success:
        print()
        print(
            "🔥 TABLEMIND CLOSED-LOOP VLA DEMO SUCCESSFUL"
        )
        return

    print()
    print(
        "⚠️ TABLEMIND CLOSED-LOOP VLA DEMO FAILED"
    )


if __name__ == "__main__":
    main()