from __future__ import annotations

from tablemind.control.so101 import SO101Controller
from tablemind.simulation.runtime import BimanualTableSimulation


TARGETS = {
    "left_glass": ("left", (-0.30, -0.02, 0.835)),
    "right_glass": ("right", (0.30, -0.02, 0.835)),
}


print("TABLEMIND - Final Glass Position Test")
print("=" * 70)

for name, (arm, target) in TARGETS.items():
    simulation = BimanualTableSimulation.create()
    controller = SO101Controller(simulation)

    result = controller.move_to(
        arm,
        target,
        tolerance=0.025,
        max_steps=300,
    )

    print(
        f"{name:12s} | "
        f"arm={arm:5s} | "
        f"target={target} | "
        f"reached={result.reached} | "
        f"error={result.position_error:.4f} m"
    )

print()
print("Glass position test complete.")