"""Inspect MuJoCo segmentation-render output for TABLEMIND."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from tablemind.simulation import BimanualTableSimulation


def main() -> None:
    print("=" * 60)
    print("TABLEMIND — MUJOCO SEGMENTATION DIAGNOSTIC")
    print("=" * 60)

    simulation = BimanualTableSimulation.create()

    renderer = mujoco.Renderer(
        simulation.model,
        height=480,
        width=640,
    )

    renderer.enable_segmentation_rendering()

    renderer.update_scene(
        simulation.data,
        camera="table_overview",
    )

    segmentation = np.asarray(
        renderer.render()
    ).copy()

    print("\nSEGMENTATION OUTPUT")
    print("-" * 60)

    print(f"Shape: {segmentation.shape}")
    print(f"Dtype: {segmentation.dtype}")
    print(f"Min:   {segmentation.min()}")
    print(f"Max:   {segmentation.max()}")

    if segmentation.ndim == 3:
        for channel in range(segmentation.shape[2]):
            values = np.unique(
                segmentation[:, :, channel]
            )

            print(
                f"\nChannel {channel}:"
            )
            print(
                f"  Unique values: {len(values)}"
            )
            print(
                f"  First values: {values[:30]}"
            )

    print("\nEXPECTED OBJECT GEOM IDS")
    print("-" * 60)

    for object_name in (
        "plate_1",
        "plate_2",
        "glass_1",
        "glass_2",
    ):
        geom_id = mujoco.mj_name2id(
            simulation.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            object_name,
        )

        print(
            f"{object_name:<10} geom_id={geom_id}"
        )

    # Save raw segmentation channels for inspection.
    output_dir = Path(
        "artifacts/segmentation_debug"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_dir / "segmentation.npy",
        segmentation,
    )

    if segmentation.ndim == 3:
        for channel in range(
            segmentation.shape[2]
        ):
            channel_image = segmentation[
                :, :, channel
            ]

            np.save(
                output_dir
                / f"channel_{channel}.npy",
                channel_image,
            )

    print("\nSaved diagnostic arrays to:")
    print(output_dir)

    renderer.close()

    print("\nSEGMENTATION DIAGNOSTIC COMPLETE")


if __name__ == "__main__":
    main()