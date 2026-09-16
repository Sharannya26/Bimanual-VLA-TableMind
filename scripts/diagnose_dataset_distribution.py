"""Diagnose horizontal object-center distribution in the MuJoCo dataset."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np


DATASET_ROOT = Path("artifacts/tablemind_mujoco_dataset")
LABEL_DIR = DATASET_ROOT / "labels"

IMAGE_WIDTH = 640

CLASS_NAMES = {
    0: "plate",
    1: "glass",
}


def load_centers() -> dict[str, list[float]]:
    """Read normalized YOLO box centers from every label file."""

    centers: dict[str, list[float]] = defaultdict(list)

    label_files = sorted(LABEL_DIR.glob("*.txt"))

    if not label_files:
        raise FileNotFoundError(
            f"No label files found in {LABEL_DIR.resolve()}"
        )

    for label_file in label_files:
        lines = label_file.read_text(encoding="utf-8").splitlines()

        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue

            parts = line.split()

            if len(parts) != 5:
                raise ValueError(
                    f"Invalid YOLO label in {label_file} "
                    f"line {line_number}: {line}"
                )

            class_id = int(parts[0])
            center_x = float(parts[1])

            if class_id not in CLASS_NAMES:
                raise ValueError(
                    f"Unknown class {class_id} in {label_file}"
                )

            if not 0.0 <= center_x <= 1.0:
                raise ValueError(
                    f"Invalid center_x={center_x} "
                    f"in {label_file}"
                )

            centers[CLASS_NAMES[class_id]].append(
                center_x
            )

    return centers


def summarize(
    name: str,
    values: list[float],
) -> None:
    """Print statistical summary for one object class."""

    array = np.asarray(values, dtype=float)

    pixels = array * IMAGE_WIDTH

    print()
    print(f"## {name.upper()}")
    print()

    print(f"Instances:       {len(array)}")
    print(f"Mean center:     {pixels.mean():.2f} px")
    print(f"Median center:   {np.median(pixels):.2f} px")
    print(f"Minimum center:  {pixels.min():.2f} px")
    print(f"Maximum center:  {pixels.max():.2f} px")
    print(f"Std deviation:   {pixels.std():.2f} px")

    print()
    print("Percentiles:")

    for percentile in (1, 5, 10, 25, 50, 75, 90, 95, 99):
        value = np.percentile(pixels, percentile)

        print(
            f"  P{percentile:02d}: "
            f"{value:7.2f} px"
        )

    # Image-region coverage.
    bins = {
        "0–80 px": (0, 80),
        "80–160 px": (80, 160),
        "160–240 px": (160, 240),
        "240–320 px": (240, 320),
        "320–400 px": (320, 400),
        "400–480 px": (400, 480),
        "480–560 px": (480, 560),
        "560–640 px": (560, 640),
    }

    print()
    print("Horizontal image coverage:")

    for label, (lower, upper) in bins.items():

        count = int(
            np.sum(
                (pixels >= lower)
                & (pixels < upper)
            )
        )

        percentage = (
            count / len(pixels) * 100.0
        )

        print(
            f"  {label:<12} "
            f"{count:4d} "
            f"({percentage:5.1f}%)"
        )


def main() -> None:
    print()
    print("=" * 64)
    print("TABLEMIND — DATASET DISTRIBUTION DIAGNOSTIC")
    print("=" * 64)
    print()

    print(
        f"Dataset: {DATASET_ROOT.resolve()}"
    )

    print(
        f"Labels:  {LABEL_DIR.resolve()}"
    )

    centers = load_centers()

    total_instances = sum(
        len(values)
        for values in centers.values()
    )

    print()
    print(
        f"Total labeled instances: {total_instances}"
    )

    for class_name in ("plate", "glass"):

        values = centers.get(class_name, [])

        if not values:
            print()
            print(
                f"WARNING: no {class_name} instances found."
            )
            continue

        summarize(
            class_name,
            values,
        )

    # ---------------------------------------------------------------
    # Runtime reference positions
    # ---------------------------------------------------------------

    print()
    print("=" * 64)
    print("RUNTIME REFERENCE")
    print("=" * 64)
    print()

    print(
        "The current runtime camera projection gives:"
    )

    print()
    print(
        "plate_1 → 223.82 px"
    )
    print(
        "plate_2 → 415.18 px"
    )
    print(
        "glass_1 → 171.40 px"
    )
    print(
        "glass_2 → 467.60 px"
    )

    print()
    print(
        "These positions should fall well inside "
        "the training distribution."
    )

    print()
    print("=" * 64)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 64)
    print()


if __name__ == "__main__":
    main()