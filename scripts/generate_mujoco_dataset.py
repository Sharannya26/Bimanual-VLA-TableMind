"""Generate a MuJoCo-native TABLEMIND vision dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from tablemind.perception.dataset_generator import (
    generate_mujoco_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate MuJoCo-native TABLEMIND "
            "vision training data."
        )
    )

    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of images to generate.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/tablemind_mujoco_dataset"
        ),
        help="Dataset output directory.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    print("=" * 60)
    print(
        "TABLEMIND — MUJOCO-NATIVE DATASET GENERATION"
    )
    print("=" * 60)

    print()
    print(f"Images: {args.count}")
    print(f"Output: {args.output}")
    print(f"Seed:   {args.seed}")

    samples = generate_mujoco_dataset(
        output_dir=args.output,
        count=args.count,
        seed=args.seed,
    )

    total_boxes = sum(
        len(sample.boxes)
        for sample in samples
    )

    print()
    print("DATASET SUMMARY")
    print("-" * 60)
    print(
        f"Images generated: {len(samples)}"
    )
    print(
        f"Bounding boxes:    {total_boxes}"
    )

    if samples:
        first_sample = samples[0]

        print()
        print("First sample:")
        print(
            f"Image: {first_sample.image_path}"
        )
        print(
            f"Label: {first_sample.label_path}"
        )

        for box in first_sample.boxes:
            print(
                "  "
                f"class={box.class_id} "
                f"cx={box.center_x:.4f} "
                f"cy={box.center_y:.4f} "
                f"w={box.width:.4f} "
                f"h={box.height:.4f}"
            )

    print()
    print(
        "MUJOCO-NATIVE DATASET GENERATION COMPLETE"
    )


if __name__ == "__main__":
    main()