from __future__ import annotations

from tablemind.control.so101 import SO101Controller
from tablemind.simulation.runtime import BimanualTableSimulation


simulation = BimanualTableSimulation.create()
controller = SO101Controller(simulation)

print("TABLEMIND - Milestone 4.3 Staged Reach Test")
print("=" * 70)

target = (-0.28, 0.16, 0.892)

print()
print(f"Final target: {target}")

# ---------------------------------------------------------
# Stage 1: move primarily in X/Z while keeping Y near the
# initial workspace region.
# ---------------------------------------------------------

stage_1 = (-0.30, -0.02, 0.90)

print()
print(f"Stage 1 target: {stage_1}")

result_1 = controller.move_to(
    "left",
    stage_1,
    tolerance=0.025,
    max_steps=300,
)

print(
    f"Stage 1 result: "
    f"reached={result_1.reached}, "
    f"error={result_1.position_error:.4f} m, "
    f"steps={result_1.steps}"
)

print(
    f"Stage 1 final position: "
    f"{result_1.final_position}"
)

# ---------------------------------------------------------
# Stage 2: move from the intermediate position to the
# actual target.
# ---------------------------------------------------------

print()
print(f"Stage 2 target: {target}")

result_2 = controller.move_to(
    "left",
    target,
    tolerance=0.025,
    max_steps=300,
)

print(
    f"Stage 2 result: "
    f"reached={result_2.reached}, "
    f"error={result_2.position_error:.4f} m, "
    f"steps={result_2.steps}"
)

print(
    f"Stage 2 final position: "
    f"{result_2.final_position}"
)

print()
print("=" * 70)

if result_1.reached and result_2.reached:
    print("SUCCESS: staged Cartesian movement reached the target.")
else:
    print("Target still not reached.")