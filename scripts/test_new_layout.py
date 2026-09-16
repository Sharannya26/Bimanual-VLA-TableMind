from __future__ import annotations

from tablemind.control.so101 import SO101Controller
from tablemind.simulation.runtime import BimanualTableSimulation


TARGETS = {
    "left_plate": ("left", (-0.28, 0.00, 0.892)),
    "right_plate": ("right", (0.28, 0.00, 0.892)),
    "left_glass": ("left", (-0.16, 0.04, 0.955)),
    "right_glass": ("right", (0.40, 0.04, 0.955)),
}


print("TABLEMIND - Proposed Layout Reachability Test")
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
print("Proposed layout reachability test complete.")