"""Demonstrate TABLEMIND world-coordinate perception."""

from __future__ import annotations

from tablemind.perception.coordinates import WorldCoordinateSystem
from tablemind.simulation.runtime import BimanualTableSimulation


def main() -> None:
    print("TABLEMIND - Milestone 3.3 World Coordinates")
    print("=" * 50)

    simulation = BimanualTableSimulation.create()

    coordinates = WorldCoordinateSystem(
        simulation.model,
        simulation.data,
    )

    objects = [
        "plate_1",
        "plate_2",
        "glass_1",
        "glass_2",
    ]

    print("\nWorld coordinates:")
    print("-" * 50)

    for object_name in objects:
        point = coordinates.geom_position(object_name)

        print(
            f"{object_name:10s} | "
            f"x={point.x:+.3f} m | "
            f"y={point.y:+.3f} m | "
            f"z={point.z:+.3f} m"
        )

    print("\nDistance checks:")
    print("-" * 50)

    plate_1 = coordinates.geom_position("plate_1")
    plate_2 = coordinates.geom_position("plate_2")

    distance = coordinates.distance(
        plate_1,
        plate_2,
    )

    print(
        f"Distance between plate_1 and plate_2: "
        f"{distance:.3f} m"
    )

    print("\nMilestone 3.3 coordinate validation complete.")


if __name__ == "__main__":
    main()