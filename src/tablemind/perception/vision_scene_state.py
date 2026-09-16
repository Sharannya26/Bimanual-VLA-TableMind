"""Vision-grounded SceneState for TABLEMIND.

Milestone 9.1
-------------
Converts calibrated vision detections into a structured world state
that can be consumed by TABLEMIND reasoning and planning.

Pipeline:

    OpenVINO YOLO
          |
          v
    VisionDetection
          |
          v
    Object-specific calibration
          |
          v
    VisionSceneObject
          |
          v
    SceneState
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import json

import numpy as np


# ---------------------------------------------------------------------------
# Basic world position
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenePosition:
    """3D world position of a detected object."""

    x: float
    y: float
    z: float

    @property
    def xy(self) -> tuple[float, float]:
        """Return the planar world position."""
        return (self.x, self.y)

    @property
    def xyz(self) -> tuple[float, float, float]:
        """Return the full 3D world position."""
        return (self.x, self.y, self.z)


# ---------------------------------------------------------------------------
# Vision-grounded object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VisionSceneObject:
    """One object produced by the vision-grounding pipeline."""

    object_id: str
    object_type: str
    confidence: float

    bounding_box: tuple[
        float,
        float,
        float,
        float,
    ]

    pixel_center: tuple[
        float,
        float,
    ]

    position: ScenePosition

    @property
    def x(self) -> float:
        return self.position.x

    @property
    def y(self) -> float:
        return self.position.y

    @property
    def z(self) -> float:
        return self.position.z

    @property
    def xy(self) -> tuple[float, float]:
        return self.position.xy

    @property
    def xyz(self) -> tuple[float, float, float]:
        return self.position.xyz


# ---------------------------------------------------------------------------
# SceneState
# ---------------------------------------------------------------------------


@dataclass
class SceneState:
    """Structured world state produced from vision.

    This is the representation that later reasoning/planning stages
    should consume instead of directly depending on raw YOLO detections.
    """

    objects: list[VisionSceneObject] = field(
        default_factory=list
    )

    camera_width: int = 640
    camera_height: int = 480

    source: str = "vision"

    @property
    def object_count(self) -> int:
        """Number of detected objects."""
        return len(self.objects)

    @property
    def object_types(self) -> tuple[str, ...]:
        """Unique object types currently visible."""
        return tuple(
            sorted(
                {
                    obj.object_type
                    for obj in self.objects
                }
            )
        )

    def by_type(
        self,
        object_type: str,
    ) -> list[VisionSceneObject]:
        """Return all visible objects of a given type."""

        return [
            obj
            for obj in self.objects
            if obj.object_type == object_type
        ]

    def get(
        self,
        object_id: str,
    ) -> VisionSceneObject:
        """Return an object by ID."""

        for obj in self.objects:
            if obj.object_id == object_id:
                return obj

        raise KeyError(
            f"Object '{object_id}' not found in SceneState."
        )

    def count(
        self,
        object_type: str,
    ) -> int:
        """Return the number of objects of a given type."""

        return len(
            self.by_type(object_type)
        )

    def contains(
        self,
        object_type: str,
    ) -> bool:
        """Return whether at least one object of a type exists."""

        return self.count(object_type) > 0

    def closest_to(
        self,
        object_type: str,
        x: float,
        y: float,
    ) -> VisionSceneObject:
        """Return the closest visible object of a given type."""

        candidates = self.by_type(
            object_type
        )

        if not candidates:
            raise ValueError(
                f"No '{object_type}' objects "
                "exist in SceneState."
            )

        return min(
            candidates,
            key=lambda obj: (
                (obj.x - x) ** 2
                + (obj.y - y) ** 2
            ),
        )

    def summary(self) -> dict[str, int]:
        """Return object counts grouped by type."""

        result: dict[str, int] = {}

        for obj in self.objects:
            result[obj.object_type] = (
                result.get(
                    obj.object_type,
                    0,
                )
                + 1
            )

        return result

    def to_dict(self) -> dict:
        """Serialize SceneState to a JSON-compatible dictionary."""

        return {
            "source": self.source,
            "camera": {
                "width": self.camera_width,
                "height": self.camera_height,
            },
            "objects": [
                {
                    "object_id": obj.object_id,
                    "object_type": obj.object_type,
                    "confidence": obj.confidence,
                    "bounding_box": list(
                        obj.bounding_box
                    ),
                    "pixel_center": list(
                        obj.pixel_center
                    ),
                    "position": {
                        "x": obj.x,
                        "y": obj.y,
                        "z": obj.z,
                    },
                }
                for obj in self.objects
            ],
        }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AffineCalibration:
    """2D affine pixel -> world mapping.

    The mapping is:

        world_x = a0*u + a1*v + a2
        world_y = b0*u + b1*v + b2
    """

    coefficients: np.ndarray

    def __post_init__(self) -> None:
        coefficients = np.asarray(
            self.coefficients,
            dtype=float,
        )

        if coefficients.shape != (3, 2):
            raise ValueError(
                "Affine calibration coefficients "
                "must have shape (3, 2)."
            )

        object.__setattr__(
            self,
            "coefficients",
            coefficients,
        )

    def pixel_to_world(
        self,
        pixel_x: float,
        pixel_y: float,
    ) -> tuple[float, float]:

        vector = np.asarray(
            [
                pixel_x,
                pixel_y,
                1.0,
            ],
            dtype=float,
        )

        world = vector @ self.coefficients

        return (
            float(world[0]),
            float(world[1]),
        )


@dataclass(frozen=True)
class CalibrationSet:
    """Object-specific calibration mappings."""

    calibrations: dict[
        str,
        AffineCalibration,
    ]

    z_values: dict[
        str,
        float,
    ]

    @classmethod
    def from_json(
        cls,
        path: str | Path,
    ) -> "CalibrationSet":
        """Load Milestone 8.9.7 calibration data."""

        calibration_path = Path(path)

        if not calibration_path.exists():
            raise FileNotFoundError(
                f"Calibration file not found: "
                f"{calibration_path}"
            )

        payload = json.loads(
            calibration_path.read_text(
                encoding="utf-8"
            )
        )

        calibrations: dict[
            str,
            AffineCalibration,
        ] = {}

        z_values: dict[
            str,
            float,
        ] = {}

        for object_type in (
            "plate",
            "glass",
        ):

            if object_type not in payload:
                raise KeyError(
                    f"Calibration file does not "
                    f"contain '{object_type}'."
                )

            entry = payload[object_type]

            calibrations[
                object_type
            ] = AffineCalibration(
                np.asarray(
                    entry["coefficients"],
                    dtype=float,
                )
            )

            z_values[
                object_type
            ] = float(
                entry["z"]
            )

        return cls(
            calibrations=calibrations,
            z_values=z_values,
        )

    def ground(
        self,
        object_type: str,
        pixel_x: float,
        pixel_y: float,
    ) -> ScenePosition:
        """Convert a pixel coordinate into a world position."""

        if object_type not in self.calibrations:
            raise KeyError(
                f"No calibration available for "
                f"object type '{object_type}'."
            )

        if object_type not in self.z_values:
            raise KeyError(
                f"No z value available for "
                f"object type '{object_type}'."
            )

        world_x, world_y = (
            self.calibrations[
                object_type
            ].pixel_to_world(
                pixel_x,
                pixel_y,
            )
        )

        return ScenePosition(
            x=world_x,
            y=world_y,
            z=self.z_values[
                object_type
            ],
        )


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class VisionSceneStateBuilder:
    """Build SceneState from VisionDetection objects."""

    def __init__(
        self,
        calibration: CalibrationSet,
        camera_width: int = 640,
        camera_height: int = 480,
        confidence_threshold: float = 0.25,
    ) -> None:

        self.calibration = calibration

        self.camera_width = camera_width
        self.camera_height = camera_height

        self.confidence_threshold = (
            confidence_threshold
        )

    def build(
        self,
        detections: Iterable,
    ) -> SceneState:
        """Build a SceneState from vision detections.

        The input detections are expected to provide:

            object_type
            confidence
            bounding_box

        This matches TABLEMIND's VisionDetection interface.

        Object IDs are assigned deterministically from calibrated
        world X position rather than raw detector order.

        Therefore, for each object type:

            most negative world X -> <type>_1
            next object           -> <type>_2
            ...

        This keeps vision object identities aligned with the
        left-to-right physical layout of the MuJoCo scene.
        """

        grounded_objects: list[
            tuple[
                str,
                float,
                tuple[float, float, float, float],
                tuple[float, float],
                ScenePosition,
            ]
        ] = []

        for detection in detections:

            object_type = str(
                detection.object_type
            )

            confidence = float(
                detection.confidence
            )

            if (
                confidence
                < self.confidence_threshold
            ):
                continue

            x1, y1, x2, y2 = (
                detection.bounding_box
            )

            pixel_x = (
                float(x1) + float(x2)
            ) / 2.0

            pixel_y = (
                float(y1) + float(y2)
            ) / 2.0

            position = (
                self.calibration.ground(
                    object_type=object_type,
                    pixel_x=pixel_x,
                    pixel_y=pixel_y,
                )
            )

            grounded_objects.append(
                (
                    object_type,
                    confidence,
                    (
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                    ),
                    (
                        pixel_x,
                        pixel_y,
                    ),
                    position,
                )
            )

        # ------------------------------------------------------------------
        # Deterministic identity assignment
        # ------------------------------------------------------------------
        #
        # OpenVINO/YOLO detection order is not a physical identity.
        #
        # The previous implementation assigned plate_1, plate_2, etc.
        # according to whatever order the detector happened to return
        # its detections. That caused the physical +X plate to sometimes
        # become plate_1 and the physical -X plate to become plate_2.
        #
        # We instead sort each object type by calibrated world X.
        # This makes IDs stable:
        #
        #   left / negative X  -> object_1
        #   right / positive X -> object_2
        #
        # This matches the existing TABLEMIND arm assignment convention.
        # ------------------------------------------------------------------

        grouped_objects: dict[
            str,
            list[
                tuple[
                    str,
                    float,
                    tuple[float, float, float, float],
                    tuple[float, float],
                    ScenePosition,
                ]
            ],
        ] = {}

        for grounded_object in grounded_objects:
            object_type = grounded_object[0]

            grouped_objects.setdefault(
                object_type,
                [],
            ).append(
                grounded_object
            )

        objects: list[
            VisionSceneObject
        ] = []

        for object_type in sorted(
            grouped_objects
        ):

            typed_objects = grouped_objects[
                object_type
            ]

            typed_objects.sort(
                key=lambda item: (
                    item[4].x,
                    item[4].y,
                    item[3][0],
                )
            )

            for index, (
                _object_type,
                confidence,
                bounding_box,
                pixel_center,
                position,
            ) in enumerate(
                typed_objects,
                start=1,
            ):

                object_id = (
                    f"{object_type}_{index}"
                )

                objects.append(
                    VisionSceneObject(
                        object_id=object_id,
                        object_type=object_type,
                        confidence=confidence,
                        bounding_box=bounding_box,
                        pixel_center=pixel_center,
                        position=position,
                    )
                )

        # Keep the existing public SceneState ordering behavior:
        # object type first, then highest-confidence objects first.
        objects.sort(
            key=lambda obj: (
                obj.object_type,
                -obj.confidence,
            )
        )

        return SceneState(
            objects=objects,
            camera_width=self.camera_width,
            camera_height=self.camera_height,
            source="openvino_calibrated_vision",
        )