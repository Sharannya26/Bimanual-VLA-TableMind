from __future__ import annotations

from tablemind.control.so101 import SO101Controller
from tablemind.simulation.runtime import BimanualTableSimulation


TARGETS = [
    (-0.24, -0.02, 0.835),
    (-0.24, 0.00, 0.835),
    (-0.20, -0.02, 0.835),
    (-0.20, 0.00, 0.835),
    (-0.16, -0.02, 0.835),
    (-0.16, 0.00, 0.835),
]


print("TABLEMIND - Left Glass Height Test")
print("=" * 70)

for target in TARGETS:
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
        f"error={result.position_error:.4f} m"
    )

print()
print("Height test complete.")