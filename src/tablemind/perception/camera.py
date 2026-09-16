from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class TableCamera:
    model: mujoco.MjModel
    data: mujoco.MjData
    camera_name: str = "table_overview"
    width: int = 640
    height: int = 480

    def __post_init__(self):
        self.camera_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            self.camera_name,
        )

        if self.camera_id < 0:
            raise ValueError(
                f"Camera '{self.camera_name}' was not found."
            )

        self.renderer = mujoco.Renderer(
            self.model,
            height=self.height,
            width=self.width,
        )

    def capture(self) -> np.ndarray:
        self.renderer.update_scene(
            self.data,
            camera=self.camera_id,
        )

        image = self.renderer.render()

        return np.asarray(image).copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()