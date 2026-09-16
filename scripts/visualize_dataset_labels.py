"""Visualize YOLO labels for TABLEMIND synthetic datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CLASS_NAMES = {
    0: "plate",
    1: "glass",
}


def load_yolo_labels(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> list[tuple[int, float, float, float, float]]:
    """Load YOLO labels and convert them to pixel coordinates."""

    labels = []

    for line in label_path.read_text(
        encoding="utf-8"
    ).splitlines():

        line = line.strip()

        if not line:
            continue

        values = line.split()

        if len(values) != 5:
            raise ValueError(
                f"Invalid YOLO label line: {line!r}"
            )

        class_id = int(values[0])
        center_x = float(values[1]) * image_width
        center_y = float(values[2]) * image_height
        width = float(values[3]) * image_width
        height = float(values[4]) * image_height

        left = center_x - width / 2
        top = center_y - height / 2
        right = center_x + width / 2
        bottom = center_y + height / 2

        labels.append(
            (
                class_id,
                left,
                top,
                right,
                bottom,
            )
        )

    return labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize TABLEMIND YOLO labels."
    )

    parser.add_argument(
        "--dataset",
        default="artifacts/tablemind_dataset",
        help="Dataset directory.",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Frame index to visualize.",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    dataset_dir = project_root / args.dataset

    image_path = (
        dataset_dir
        / "images"
        / f"frame_{args.index:06d}.png"
    )

    label_path = (
        dataset_dir
        / "labels"
        / f"frame_{args.index:06d}.txt"
    )

    output_path = (
        dataset_dir
        / f"frame_{args.index:06d}_labeled.png"
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not label_path.exists():
        raise FileNotFoundError(
            f"Label file not found: {label_path}"
        )

    image = Image.open(image_path).convert("RGB")

    width, height = image.size

    labels = load_yolo_labels(
        label_path,
        width,
        height,
    )

    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            18,
        )
    except OSError:
        font = ImageFont.load_default()

    for class_id, left, top, right, bottom in labels:

        class_name = CLASS_NAMES.get(
            class_id,
            f"class_{class_id}",
        )

        draw.rectangle(
            (
                left,
                top,
                right,
                bottom,
            ),
            outline="red",
            width=3,
        )

        label = f"{class_name} ({class_id})"

        text_bbox = draw.textbbox(
            (left, top),
            label,
            font=font,
        )

        text_width = (
            text_bbox[2] - text_bbox[0]
        )

        text_height = (
            text_bbox[3] - text_bbox[1]
        )

        text_top = max(
            0,
            top - text_height - 6,
        )

        draw.rectangle(
            (
                left,
                text_top,
                left + text_width + 6,
                text_top + text_height + 4,
            ),
            fill="red",
        )

        draw.text(
            (
                left + 3,
                text_top + 2,
            ),
            label,
            fill="white",
            font=font,
        )

    image.save(output_path)

    print("Dataset label visualization created.")
    print(f"Dataset: {dataset_dir}")
    print(f"Image:   {image_path}")
    print(f"Labels:  {label_path}")
    print(f"Output:  {output_path}")
    print(f"Boxes:   {len(labels)}")


if __name__ == "__main__":
    main()