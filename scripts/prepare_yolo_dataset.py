"""Prepare TABLEMIND YOLO dataset splits."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


CLASS_NAMES = {
    0: "plate",
    1: "glass",
}

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Prepare TABLEMIND YOLO dataset splits."
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=Path("artifacts/tablemind_dataset_random"),
        help="Source dataset containing images/ and labels/.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/tablemind_yolo"),
        help="Output YOLO dataset directory.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic splitting.",
    )

    return parser.parse_args()


def find_source_files(
    source_dir: Path,
) -> list[tuple[Path, Path]]:
    """Find image/label pairs in the source dataset."""

    image_dir = source_dir / "images"
    label_dir = source_dir / "labels"

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory does not exist: {image_dir}"
        )

    if not label_dir.exists():
        raise FileNotFoundError(
            f"Label directory does not exist: {label_dir}"
        )

    image_files = sorted(
        image_dir.glob("*.png")
    )

    if not image_files:
        raise ValueError(
            f"No PNG images found in {image_dir}."
        )

    pairs: list[tuple[Path, Path]] = []

    for image_path in image_files:
        label_path = (
            label_dir
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():
            raise ValueError(
                "Missing label for image: "
                f"{image_path}"
            )

        pairs.append(
            (image_path, label_path)
        )

    return pairs


def validate_label(
    label_path: Path,
) -> None:
    """Validate one YOLO label file."""

    text = label_path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        raise ValueError(
            f"Label file is empty: {label_path}"
        )

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"Invalid YOLO label in "
                f"{label_path} at line {line_number}: "
                f"expected 5 values, got {len(parts)}."
            )

        try:
            class_id = int(parts[0])

            center_x = float(parts[1])
            center_y = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

        except ValueError as exc:
            raise ValueError(
                f"Non-numeric YOLO label in "
                f"{label_path} at line {line_number}."
            ) from exc

        if class_id not in CLASS_NAMES:
            raise ValueError(
                f"Unknown class ID {class_id} in "
                f"{label_path} at line {line_number}."
            )

        values = (
            center_x,
            center_y,
            width,
            height,
        )

        if not all(
            0.0 <= value <= 1.0
            for value in values
        ):
            raise ValueError(
                f"YOLO coordinates must be in "
                f"[0, 1] in {label_path} "
                f"at line {line_number}."
            )

        if width <= 0.0 or height <= 0.0:
            raise ValueError(
                f"Bounding-box width and height "
                f"must be positive in {label_path} "
                f"at line {line_number}."
            )


def validate_dataset(
    pairs: list[tuple[Path, Path]],
) -> None:
    """Validate all image/label pairs."""

    for image_path, label_path in pairs:
        if image_path.stat().st_size == 0:
            raise ValueError(
                f"Image file is empty: {image_path}"
            )

        validate_label(label_path)


def split_dataset(
    pairs: list[tuple[Path, Path]],
    seed: int,
) -> dict[str, list[tuple[Path, Path]]]:
    """Split pairs deterministically into train/val/test."""

    shuffled = list(pairs)

    rng = random.Random(seed)
    rng.shuffle(shuffled)

    total = len(shuffled)

    train_count = int(
        total * TRAIN_RATIO
    )

    val_count = int(
        total * VAL_RATIO
    )

    test_count = (
        total
        - train_count
        - val_count
    )

    if train_count == 0:
        raise ValueError(
            "Dataset is too small to create "
            "a training split."
        )

    if val_count == 0 or test_count == 0:
        raise ValueError(
            "Dataset is too small to create "
            "validation and test splits."
        )

    return {
        "train": shuffled[
            :train_count
        ],
        "val": shuffled[
            train_count:train_count + val_count
        ],
        "test": shuffled[
            train_count + val_count:
        ],
    }


def prepare_output_directory(
    output_dir: Path,
) -> None:
    """Create a clean YOLO dataset directory."""

    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in (
        "train",
        "val",
        "test",
    ):
        (
            output_dir
            / "images"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            output_dir
            / "labels"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )


def copy_split(
    pairs: list[tuple[Path, Path]],
    output_dir: Path,
    split: str,
) -> None:
    """Copy one dataset split."""

    image_output = (
        output_dir
        / "images"
        / split
    )

    label_output = (
        output_dir
        / "labels"
        / split
    )

    for image_path, label_path in pairs:
        shutil.copy2(
            image_path,
            image_output / image_path.name,
        )

        shutil.copy2(
            label_path,
            label_output / label_path.name,
        )


def write_data_yaml(
    output_dir: Path,
) -> None:
    """Write the YOLO dataset configuration."""

    yaml_path = output_dir / "data.yaml"

    yaml_text = (
        f"path: {output_dir.resolve().as_posix()}\n"
        "\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "names:\n"
        "  0: plate\n"
        "  1: glass\n"
    )

    yaml_path.write_text(
        yaml_text,
        encoding="utf-8",
    )


def count_files(
    output_dir: Path,
    split: str,
) -> tuple[int, int]:
    """Count images and labels in one split."""

    image_count = len(
        list(
            (
                output_dir
                / "images"
                / split
            ).glob("*.png")
        )
    )

    label_count = len(
        list(
            (
                output_dir
                / "labels"
                / split
            ).glob("*.txt")
        )
    )

    return image_count, label_count


def verify_output(
    output_dir: Path,
    expected_total: int,
) -> None:
    """Verify the prepared dataset."""

    total_images = 0
    total_labels = 0

    expected_counts = {
        "train": int(
            expected_total * TRAIN_RATIO
        ),
        "val": int(
            expected_total * VAL_RATIO
        ),
        "test": expected_total
        - int(expected_total * TRAIN_RATIO)
        - int(expected_total * VAL_RATIO),
    }

    for split, expected_count in (
        expected_counts.items()
    ):
        image_count, label_count = count_files(
            output_dir,
            split,
        )

        if image_count != expected_count:
            raise RuntimeError(
                f"{split}: expected "
                f"{expected_count} images, "
                f"found {image_count}."
            )

        if label_count != expected_count:
            raise RuntimeError(
                f"{split}: expected "
                f"{expected_count} labels, "
                f"found {label_count}."
            )

        total_images += image_count
        total_labels += label_count

    if total_images != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} total images, "
            f"found {total_images}."
        )

    if total_labels != expected_total:
        raise RuntimeError(
            f"Expected {expected_total} total labels, "
            f"found {total_labels}."
        )


def main() -> None:
    """Prepare the dataset."""

    args = parse_args()

    source_dir = args.source
    output_dir = args.output

    print("=" * 60)
    print("TABLEMIND — YOLO DATASET PREPARATION")
    print("=" * 60)

    print()
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Seed:   {args.seed}")

    pairs = find_source_files(
        source_dir
    )

    print()
    print("SOURCE DATASET")
    print("-" * 60)
    print(
        f"Images found: {len(pairs)}"
    )
    print(
        f"Expected labels: {len(pairs)}"
    )

    validate_dataset(pairs)

    print(
        "Label validation: PASSED"
    )

    splits = split_dataset(
        pairs,
        args.seed,
    )

    print()
    print("DATASET SPLIT")
    print("-" * 60)
    print(
        f"Train: {len(splits['train'])}"
    )
    print(
        f"Val:   {len(splits['val'])}"
    )
    print(
        f"Test:  {len(splits['test'])}"
    )

    prepare_output_directory(
        output_dir
    )

    for split, split_pairs in splits.items():
        copy_split(
            split_pairs,
            output_dir,
            split,
        )

    write_data_yaml(
        output_dir
    )

    verify_output(
        output_dir,
        len(pairs),
    )

    print()
    print("OUTPUT VERIFICATION")
    print("-" * 60)

    for split in (
        "train",
        "val",
        "test",
    ):
        image_count, label_count = count_files(
            output_dir,
            split,
        )

        print(
            f"{split:5s}: "
            f"{image_count} images, "
            f"{label_count} labels"
        )

    print()
    print(
        f"Total: {len(pairs)} images, "
        f"{len(pairs)} labels"
    )

    print()
    print(
        f"data.yaml: "
        f"{output_dir / 'data.yaml'}"
    )

    print()
    print(
        "YOLO DATASET PREPARATION COMPLETE"
    )


if __name__ == "__main__":
    main()