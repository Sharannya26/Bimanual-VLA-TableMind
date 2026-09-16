from __future__ import annotations

from tablemind.control.so101 import SO101Controller
from tablemind.simulation.runtime import BimanualTableSimulation


X_VALUES = (-0.34, -0.32, -0.30, -0.28, -0.26, -0.24)
Y_VALUES = (-0.06, -0.04, -0.02, 0.00, 0.02)
Z = 0.835


print("TABLEMIND - Left Arm Glass Workspace")
print("=" * 70)
print(f"Z = {Z:.3f} m")
print()

print("       " + " ".join(f"{y:+.2f}" for y in Y_VALUES))
print("-" * 70)

for x in X_VALUES:
    row = []

    for y in Y_VALUES:
        simulation = BimanualTableSimulation.create()
        controller = SO101Controller(simulation)

        result = controller.move_to(
            "left",
            (x, y, Z),
            tolerance=0.025,
            max_steps=300,
        )

        if result.reached:
            row.append(" OK ")
        else:
            row.append(f"{result.position_error:.2f}")

    print(f"x={x:+.2f} | " + " ".join(row))

print()
print("Workspace scan complete.")