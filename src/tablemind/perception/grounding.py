"""Camera-to-world grounding utilities for TABLEMIND."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class GroundedPoint:
    """A point grounded into the MuJoCo world coordinate frame."""

    x: float
    y: float
    z: float

    @property
    def xyz(self) -> tuple[float, float, float]:
        """Return the point as an XYZ tuple."""
        return (self.x, self.y, self.z)


class CameraGrounder:
    """Project camera pixels onto a horizontal world plane."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        camera_name: str = "table_overview",
        image_width: int = 640,
        image_height: int = 480,
    ) -> None:
        self.model = model
        self.data = data
        self.camera_name = camera_name
        self.image_width = image_width
        self.image_height = image_height

        self.camera_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            self.camera_name,
        )

        if self.camera_id < 0:
            raise KeyError(
                f"MuJoCo model does not contain camera "
                f"{self.camera_name!r}."
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
        """Return camera-local-to-world rotation."""

        return np.asarray(
            self.data.cam_xmat[self.camera_id],
            dtype=float,
        ).reshape(3, 3).copy()

    @property
    def focal_lengths(self) -> tuple[float, float]:
        """Return approximate pixel focal lengths."""

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
    ) -> GroundedPoint:
        """
        Project an image pixel onto a horizontal world plane.

        Parameters
        ----------
        pixel_x:
            Horizontal image coordinate in pixels.

        pixel_y:
            Vertical image coordinate in pixels.

        plane_z:
            World-space Z coordinate of the plane.
        """

        fx, fy = self.focal_lengths
        cx, cy = self.principal_point

        # Camera coordinates:
        #
        # x -> right
        # y -> up
        # z -> forward is -Z
        #
        # Image Y grows downward, so invert it.
        ray_camera = np.array(
            [
                (pixel_x - cx) / fx,
                -(pixel_y - cy) / fy,
                -1.0,
            ],
            dtype=float,
        )

        ray_camera /= np.linalg.norm(ray_camera)

        rotation = self.camera_rotation

        ray_world = rotation @ ray_camera
        ray_world /= np.linalg.norm(ray_world)

        camera_position = self.camera_position

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

        world_point = (
            camera_position
            + distance * ray_world
        )

        return GroundedPoint(
            x=float(world_point[0]),
            y=float(world_point[1]),
            z=float(world_point[2]),
        )

    def bounding_box_center_to_world(
        self,
        left: float,
        top: float,
        right: float,
        bottom: float,
        *,
        plane_z: float,
    ) -> GroundedPoint:
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