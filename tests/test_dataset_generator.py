from pathlib import Path

from tablemind.perception.dataset_generator import (
    BoundingBox,
    SyntheticDatasetGenerator,
)
from tablemind.simulation import BimanualTableSimulation


def test_bounding_box_serializes_to_yolo():
    box = BoundingBox(
        class_id=0,
        center_x=0.5,
        center_y=0.4,
        width=0.2,
        height=0.3,
    )

    assert box.to_yolo() == (
        "0 0.500000 0.400000 0.200000 0.300000"
    )


def test_synthetic_dataset_generator_creates_sample(tmp_path: Path):
    simulation = BimanualTableSimulation.create()

    generator = SyntheticDatasetGenerator(
        simulation,
        output_dir=tmp_path,
    )

    try:
        samples = generator.generate(1)

        assert len(samples) == 1

        sample = samples[0]

        assert sample.image_path.exists()
        assert sample.label_path.exists()

        assert sample.image_path.suffix == ".png"
        assert sample.label_path.suffix == ".txt"

        assert len(sample.boxes) == 4

        labels = sample.label_path.read_text(
            encoding="utf-8"
        ).strip().splitlines()

        assert len(labels) == 4

        for box in sample.boxes:
            assert 0.0 <= box.center_x <= 1.0
            assert 0.0 <= box.center_y <= 1.0
            assert 0.0 < box.width <= 1.0
            assert 0.0 < box.height <= 1.0

    finally:
        generator.close()