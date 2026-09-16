from __future__ import annotations

import numpy as np

from tablemind.control.so101 import SO101Controller
from tablemind.simulation.runtime import BimanualTableSimulation


Z = 0.892

X_VALUES = np.arange(-0.42, -0.17, 0.05)
Y_VALUES = np.arange(-0.02, 0.31, 0.05)


print("TABLEMIND - Milestone 4.3 Reachability Scan")
print("=" * 70)
print(f"Target height: z={Z:.3f} m")
print()

print("Each cell shows final Cartesian error in meters.")
print("OK = target reached within controller tolerance.")
print()

print("        " + " ".join(f"{y:+.2f}" for y in Y_VALUES))
print("-" * 70)

for x in X_VALUES:
    row = []

    for y in Y_VALUES:
        simulation = BimanualTableSimulation.create()
        controller = SO101Controller(simulation)

        result = controller.move_to(
            "left",
            (float(x), float(y), Z),
        )

        if result.reached:
            cell = " OK "
        else:
            cell = f"{result.position_error:.2f}"

        row.append(cell)

    print(f"x={x:+.2f} | " + " ".join(row))

print()
print("Reachability scan complete.")