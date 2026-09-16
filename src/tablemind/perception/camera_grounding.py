"""Camera-to-world grounding utilities for TABLEMIND."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from tablemind.perception.coordinates import WorldPoint


@dataclass(frozen=True)
class CameraGrounder:
    """
    Project camera pixels onto a horizontal world plane.

    TABLEMIND currently uses a known table plane because the
    environment is simulation-first and the table height is known.
    """

    model: mujoco.MjModel
    data: mujoco.MjData
    camera_name: str = "table_overview"
    image_width: int = 640
    image_height: int = 480

    def __post_init__(self) -> None:
        camera_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            self.camera_name,
        )

        if camera_id < 0:
            raise KeyError(
                f"MuJoCo model does not contain camera "
                f"{self.camera_name!r}"
            )

        object.__setattr__(
            self,
            "camera_id",
            camera_id,
        )

    @property
    def camera_position(self) -> np.ndarray:
        """Return the camera position in world coordinates."""

        return np.asarray(
            self.data.cam_xpos[self.camera_id],
            dtype=float,
        ).copy()

    @property
    def camera_rotation(self) -> np.ndarray:
        """Return the camera rotation matrix."""

        return np.asarray(
            self.data.cam_xmat[self.camera_id],
            dtype=float,
        ).reshape(3, 3).copy()

    @property
    def focal_lengths(self) -> tuple[float, float]:
        """
        Estimate focal lengths in pixels from the MuJoCo camera FOV.
        """

        fovy_degrees = float(
            self.model.cam_fovy[self.camera_id]
        )

        fovy = np.deg2rad(fovy_degrees)

        fy = (
            self.image_height / 2.0
        ) / np.tan(fovy / 2.0)

        aspect_ratio = (
            self.image_width
            / self.image_height
        )

        fx = fy * aspect_ratio

        return fx, fy

    @property
    def principal_point(self) -> tuple[float, float]:
        """Return the image principal point."""

        return (
            (self.image_width - 1) / 2.0,
            (self.image_height - 1) / 2.0,
        )

    def pixel_to_world_on_plane(
        self,
        pixel_x: float,
        pixel_y: float,
        *,
        plane_z: float,
    ) -> WorldPoint:
        """
        Project one image pixel onto a horizontal world plane.

        Parameters
        ----------
        pixel_x:
            Horizontal image coordinate in pixels.

        pixel_y:
            Vertical image coordinate in pixels.

        plane_z:
            Z coordinate of the world plane in meters.
        """

        fx, fy = self.focal_lengths
        cx, cy = self.principal_point

        # Image coordinates:
        #   +X -> right
        #   +Y -> downward
        #
        # Convert the pixel into a camera-space ray.
        ray_camera = np.array(
            [
                (pixel_x - cx) / fx,
                -(pixel_y - cy) / fy,
                -1.0,
            ],
            dtype=float,
        )

        ray_camera /= np.linalg.norm(
            ray_camera
        )

        # Convert camera-space ray into world-space.
        ray_world = (
            self.camera_rotation
            @ ray_camera
        )

        ray_world /= np.linalg.norm(
            ray_world
        )

        camera_position = (
            self.camera_position
        )

        if abs(ray_world[2]) < 1e-9:
            raise ValueError(
                "Camera ray is parallel to the grounding plane."
            )

        distance = (
            plane_z - camera_position[2]
        ) / ray_world[2]

        if distance <= 0:
            raise ValueError(
                "Grounding plane lies behind the camera ray."
            )

        world_position = (
            camera_position
            + distance * ray_world
        )

        return WorldPoint(
            x=float(world_position[0]),
            y=float(world_position[1]),
            z=float(world_position[2]),
        )

    def bounding_box_center_to_world(
        self,
        left: float,
        top: float,
        right: float,
        bottom: float,
        *,
        plane_z: float,
    ) -> WorldPoint:
        """Ground the center of a 2D bounding box."""

        center_x = (
            left + right
        ) / 2.0

        center_y = (
            top + bottom
        ) / 2.0

        return self.pixel_to_world_on_plane(
            center_x,
            center_y,
            plane_z=plane_z,
        )