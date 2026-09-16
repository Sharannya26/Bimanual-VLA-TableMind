"""MuJoCo-native perception dataset generation for TABLEMIND."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from tablemind.perception.camera import TableCamera
from tablemind.simulation import BimanualTableSimulation


@dataclass(frozen=True)
class BoundingBox:
    """A normalized YOLO-style bounding box."""

    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float

    def to_yolo(self) -> str:
        """Return the bounding box in YOLO label format."""

        return (
            f"{self.class_id} "
            f"{self.center_x:.6f} "
            f"{self.center_y:.6f} "
            f"{self.width:.6f} "
            f"{self.height:.6f}"
        )


@dataclass(frozen=True)
class DatasetSample:
    """One generated RGB image and its labels."""

    image_path: Path
    label_path: Path
    boxes: tuple[BoundingBox, ...]


class SyntheticDatasetGenerator:
    """
    Generate automatically labelled TABLEMIND perception samples.

    RGB images are rendered from the actual MuJoCo camera.

    Bounding boxes are generated from MuJoCo's segmentation renderer,
    so the labels correspond to the actual visible object pixels rather
    than an approximate geometric projection.
    """

    PLATE_CLASS = 0
    GLASS_CLASS = 1

    OBJECTS = (
        ("plate_1", PLATE_CLASS),
        ("plate_2", PLATE_CLASS),
        ("glass_1", GLASS_CLASS),
        ("glass_2", GLASS_CLASS),
    )

    # Safe visual region for the current TABLEMIND camera.
    X_RANGE = (-0.43, 0.43)
    Y_RANGE = (-0.20, 0.08)

    # Minimum XY distance between object centers.
    MIN_OBJECT_DISTANCE = 0.18

    def __init__(
        self,
        simulation: BimanualTableSimulation,
        *,
        output_dir: str | Path,
        width: int = 640,
        height: int = 480,
        seed: int = 42,
    ) -> None:
        self.simulation = simulation
        self.output_dir = Path(output_dir)
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)

        self.image_dir = self.output_dir / "images"
        self.label_dir = self.output_dir / "labels"

        self.image_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.label_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.camera = TableCamera(
            simulation.model,
            simulation.data,
            camera_name="table_overview",
            width=width,
            height=height,
        )

        self.segmentation_renderer = mujoco.Renderer(
            simulation.model,
            height=height,
            width=width,
        )

        self.segmentation_renderer.enable_segmentation_rendering()

        self._geom_to_object = self._build_geom_mapping()

    def close(self) -> None:
        """Release renderer resources."""

        self.camera.close()
        self.segmentation_renderer.close()

    def generate_sample(
        self,
        sample_index: int,
        *,
        randomize: bool = True,
    ) -> DatasetSample:
        """Generate one RGB image and its MuJoCo-derived labels."""

        if randomize:
            self.randomize_scene()

        mujoco.mj_forward(
            self.simulation.model,
            self.simulation.data,
        )

        image = self.camera.capture()

        segmentation = self._capture_segmentation()

        boxes = self._boxes_from_segmentation(
            segmentation
        )

        image_path = (
            self.image_dir
            / f"frame_{sample_index:06d}.png"
        )

        label_path = (
            self.label_dir
            / f"frame_{sample_index:06d}.txt"
        )

        Image.fromarray(image).save(image_path)

        label_text = "\n".join(
            box.to_yolo()
            for box in boxes
        )

        if label_text:
            label_text += "\n"

        label_path.write_text(
            label_text,
            encoding="utf-8",
        )

        return DatasetSample(
            image_path=image_path,
            label_path=label_path,
            boxes=boxes,
        )

    def generate(
        self,
        count: int,
        *,
        randomize: bool = True,
    ) -> tuple[DatasetSample, ...]:
        """Generate multiple MuJoCo-native samples."""

        if count <= 0:
            raise ValueError(
                "count must be greater than zero."
            )

        return tuple(
            self.generate_sample(
                index,
                randomize=randomize,
            )
            for index in range(count)
        )

    def randomize_scene(self) -> None:
        """Randomize active object positions and orientations."""

        positions: list[np.ndarray] = []

        for object_name, _ in self.OBJECTS:
            position = self._sample_position(
                positions
            )

            positions.append(position)

            self._set_free_body_pose(
                object_name,
                position,
            )

    def _sample_position(
        self,
        existing_positions: list[np.ndarray],
    ) -> np.ndarray:
        """Sample a collision-safe table position."""

        for _ in range(1000):
            x = self.rng.uniform(
                self.X_RANGE[0],
                self.X_RANGE[1],
            )

            y = self.rng.uniform(
                self.Y_RANGE[0],
                self.Y_RANGE[1],
            )

            candidate = np.array(
                [x, y],
                dtype=float,
            )

            if all(
                np.linalg.norm(
                    candidate - existing
                )
                >= self.MIN_OBJECT_DISTANCE
                for existing in existing_positions
            ):
                return candidate

        raise RuntimeError(
            "Unable to sample a non-overlapping object layout."
        )

    def _set_free_body_pose(
        self,
        body_name: str,
        position: np.ndarray,
    ) -> None:
        """Set the position and yaw of a free-jointed body."""

        body_id = mujoco.mj_name2id(
            self.simulation.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )

        if body_id < 0:
            raise KeyError(
                f"MuJoCo model does not contain body {body_name!r}"
            )

        joint_id = (
            self.simulation.model.body_jntadr[body_id]
        )

        if joint_id < 0:
            raise ValueError(
                f"Body {body_name!r} does not have a joint."
            )

        joint_type = (
            self.simulation.model.jnt_type[joint_id]
        )

        if joint_type != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(
                f"Body {body_name!r} is not free-jointed."
            )

        qpos_address = (
            self.simulation.model.jnt_qposadr[joint_id]
        )

        qpos = self.simulation.data.qpos

        qpos[qpos_address] = position[0]
        qpos[qpos_address + 1] = position[1]

        if body_name.startswith("plate"):
            z = 0.772
        else:
            z = 0.835

        qpos[qpos_address + 2] = z

        yaw = self.rng.uniform(
            -np.pi,
            np.pi,
        )

        qpos[qpos_address + 3] = np.cos(
            yaw / 2.0
        )

        qpos[qpos_address + 4] = 0.0
        qpos[qpos_address + 5] = 0.0

        qpos[qpos_address + 6] = np.sin(
            yaw / 2.0
        )

    def _build_geom_mapping(self) -> dict[int, tuple[str, int]]:
        """
        Map MuJoCo geom IDs to TABLEMIND object names/classes.

        The object geom names in the MuJoCo scene are:

            plate_1
            plate_2
            glass_1
            glass_2
        """

        mapping: dict[int, tuple[str, int]] = {}

        for object_name, class_id in self.OBJECTS:
            geom_id = mujoco.mj_name2id(
                self.simulation.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                object_name,
            )

            if geom_id < 0:
                raise KeyError(
                    f"MuJoCo model does not contain geom "
                    f"{object_name!r}"
                )

            mapping[geom_id] = (
                object_name,
                class_id,
            )

        return mapping

    def _capture_segmentation(self) -> np.ndarray:
        """Render the current scene using MuJoCo segmentation."""

        self.segmentation_renderer.update_scene(
            self.simulation.data,
            camera="table_overview",
        )

        segmentation = (
            self.segmentation_renderer.render()
        )

        return np.asarray(
            segmentation
        ).copy()

    def _boxes_from_segmentation(
        self,
        segmentation: np.ndarray,
    ) -> tuple[BoundingBox, ...]:
        """
        Extract exact visible-object bounding boxes.

        IMPORTANT:

        For the installed MuJoCo version, segmentation output is:

            channel 0 = object ID
            channel 1 = object type

        Therefore channel 0 must be compared with the MuJoCo geom ID.
        """

        if segmentation.ndim != 3:
            raise ValueError(
                "Unexpected MuJoCo segmentation shape: "
                f"{segmentation.shape}"
            )

        if segmentation.shape[2] < 2:
            raise ValueError(
                "MuJoCo segmentation output must contain "
                "at least two channels."
            )

        boxes: list[BoundingBox] = []

        for geom_id, (
            object_name,
            class_id,
        ) in self._geom_to_object.items():

            # MuJoCo segmentation channel 0 contains
            # the object/geom ID.
            mask = (
                segmentation[:, :, 0]
                == geom_id
            )

            ys, xs = np.where(mask)

            if len(xs) == 0:
                # Object is either outside the camera view
                # or completely occluded.
                continue

            min_x = max(
                0.0,
                float(np.min(xs)),
            )

            max_x = min(
                float(self.width - 1),
                float(np.max(xs)),
            )

            min_y = max(
                0.0,
                float(np.min(ys)),
            )

            max_y = min(
                float(self.height - 1),
                float(np.max(ys)),
            )

            if max_x <= min_x or max_y <= min_y:
                continue

            center_x = (
                (min_x + max_x) / 2.0
            ) / self.width

            center_y = (
                (min_y + max_y) / 2.0
            ) / self.height

            box_width = (
                max_x - min_x
            ) / self.width

            box_height = (
                max_y - min_y
            ) / self.height

            boxes.append(
                BoundingBox(
                    class_id=class_id,
                    center_x=center_x,
                    center_y=center_y,
                    width=box_width,
                    height=box_height,
                )
            )

        boxes.sort(
            key=lambda box: (
                box.class_id,
                box.center_x,
            )
        )

        return tuple(boxes)


def generate_mujoco_dataset(
    output_dir: str | Path,
    count: int = 200,
    *,
    seed: int = 42,
) -> tuple[DatasetSample, ...]:
    """Convenience function for generating a MuJoCo-native dataset."""

    simulation = BimanualTableSimulation.create()

    generator = SyntheticDatasetGenerator(
        simulation,
        output_dir=output_dir,
        seed=seed,
    )

    try:
        return generator.generate(
            count
        )
    finally:
        generator.close()