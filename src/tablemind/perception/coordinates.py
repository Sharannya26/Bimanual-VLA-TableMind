"""World-coordinate utilities for TABLEMIND perception."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class WorldPoint:
    """A point expressed in the MuJoCo world coordinate frame."""

    x: float
    y: float
    z: float

    @property
    def xyz(self) -> tuple[float, float, float]:
        """Return the point as an XYZ tuple."""

        return (self.x, self.y, self.z)

    def as_array(self) -> np.ndarray:
        """Return the point as a NumPy array."""

        return np.array(self.xyz, dtype=float)


class WorldCoordinateSystem:
    """
    Coordinate helper for the TABLEMIND MuJoCo world frame.

    MuJoCo uses a right-handed Cartesian coordinate system.
    Positions are expressed in meters.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data

    def geom_position(self, geom_name: str) -> WorldPoint:
        """
        Return the current world position of a MuJoCo geometry.

        Parameters
        ----------
        geom_name:
            Name of the MuJoCo geometry.

        Returns
        -------
        WorldPoint
            Geometry center expressed in world coordinates.
        """

        geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_name,
        )

        if geom_id < 0:
            raise KeyError(
                f"MuJoCo model does not contain geometry {geom_name!r}"
            )

        position = self.data.geom_xpos[geom_id]

        return WorldPoint(
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
        )

    def site_position(self, site_name: str) -> WorldPoint:
        """
        Return the current world position of a MuJoCo site.
        """

        site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )

        if site_id < 0:
            raise KeyError(
                f"MuJoCo model does not contain site {site_name!r}"
            )

        position = self.data.site_xpos[site_id]

        return WorldPoint(
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
        )

    def body_position(self, body_name: str) -> WorldPoint:
        """
        Return the current world position of a MuJoCo body.
        """

        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )

        if body_id < 0:
            raise KeyError(
                f"MuJoCo model does not contain body {body_name!r}"
            )

        position = self.data.xpos[body_id]

        return WorldPoint(
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
        )

    def distance(
        self,
        first: WorldPoint,
        second: WorldPoint,
    ) -> float:
        """
        Calculate Euclidean distance between two world points.

        Returns
        -------
        float
            Distance in meters.
        """

        first_array = first.as_array()
        second_array = second.as_array()

        return float(np.linalg.norm(first_array - second_array))


def make_world_coordinate_system(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> WorldCoordinateSystem:
    """Create a TABLEMIND world-coordinate helper."""

    return WorldCoordinateSystem(model=model, data=data)