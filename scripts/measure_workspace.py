from __future__ import annotations

import numpy as np

from tablemind.control.so101 import SO101Controller
from tablemind.simulation.runtime import BimanualTableSimulation


simulation = BimanualTableSimulation.create()
controller = SO101Controller(simulation)

print("TABLEMIND - SO-101 Workspace Measurement")
print("=" * 70)

# Test progressively farther positive-Y positions
# while keeping X approximately aligned with the arm.
targets = [
    (-0.48, 0.00, 0.90),
    (-0.48, 0.05, 0.90),
    (-0.48, 0.10, 0.90),
    (-0.48, 0.15, 0.90),
    (-0.48, 0.20, 0.90),
    (-0.48, 0.25, 0.90),
    (-0.48, 0.30, 0.90),
    (-0.48, 0.35, 0.90),
]

for target in targets:
    simulation = BimanualTableSimulation.create()
    controller = SO101Controller(simulation)

    result = controller.move_to(
        "left",
        target,
        tolerance=0.025,
        max_steps=300,
    )

    print(
        f"target={target} | "
        f"reached={result.reached} | "
        f"error={result.position_error:.4f} m | "
        f"final={tuple(round(float(v), 4) for v in result.final_position)}"
    )

print()
print("Workspace measurement complete.")