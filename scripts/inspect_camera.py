"""Inspect the TABLEMIND MuJoCo camera output."""

from pathlib import Path

from tablemind.perception.camera import TableCamera
from tablemind.simulation import BimanualTableSimulation


def main() -> None:
    simulation = BimanualTableSimulation.create()

    camera = TableCamera(
        simulation.model,
        simulation.data,
        width=640,
        height=480,
    )

    try:
        image = camera.capture()

        output_path = Path("artifacts") / "table_camera.png"
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # MuJoCo returns RGB; PIL expects RGB as well.
        from PIL import Image

        Image.fromarray(image).save(output_path)

        print("Camera capture successful.")
        print(f"Shape: {image.shape}")
        print(f"Dtype: {image.dtype}")
        print(f"Saved: {output_path}")

    finally:
        camera.close()


if __name__ == "__main__":
    main()