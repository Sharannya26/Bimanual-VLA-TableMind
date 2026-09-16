from __future__ import annotations

from tablemind.control.so101 import SO101Controller
from tablemind.simulation.runtime import BimanualTableSimulation


X_VALUES = (-0.24, -0.20, -0.16, -0.12, -0.08)
Y_VALUES = (-0.02, 0.00, 0.02, 0.04, 0.06)
Z = 0.955


print("TABLEMIND - Left Glass Workspace Search")
print("=" * 70)
print(f"Target Z = {Z:.3f} m")
print()

best = None

for x in X_VALUES:
    for y in Y_VALUES:
        target = (x, y, Z)

        simulation = BimanualTableSimulation.create()
        controller = SO101Controller(simulation)

        result = controller.move_to(
            "left",
            target,
            tolerance=0.025,
            max_steps=300,
        )

        print(
            f"target=({x:+.2f}, {y:+.2f}, {Z:.3f}) | "
            f"reached={result.reached} | "
            f"error={result.position_error:.4f} m"
        )

        if best is None or result.position_error < best[0]:
            best = (result.position_error, target, result)

print()
print("=" * 70)
print("BEST POSITION")
print("=" * 70)

error, target, result = best

print(f"target={target}")
print(f"reached={result.reached}")
print(f"error={error:.4f} m")
print(
    "final="
    f"{tuple(round(float(v), 4) for v in result.final_position)}"
)

print()
print("Search complete.")