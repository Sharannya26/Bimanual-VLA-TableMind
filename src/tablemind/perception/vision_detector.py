"""Vision-based object detection for TABLEMIND."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class VisionDetection:
    """One object detected by the vision model."""

    object_type: str
    confidence: float
    bounding_box: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        """Return the center pixel of the bounding box."""

        left, top, right, bottom = self.bounding_box

        return (
            (left + right) / 2.0,
            (top + bottom) / 2.0,
        )

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Return the bottom-center pixel of the bounding box."""

        left, _, right, bottom = self.bounding_box

        return (
            (left + right) / 2.0,
            bottom,
        )


class OpenVINOTableDetector:
    """
    TABLEMIND table-object detector using the custom YOLO OpenVINO model.

    The exported Ultralytics OpenVINO model is loaded through the
    Ultralytics API so that preprocessing and YOLO postprocessing
    remain consistent with the model used during validation.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        image_size: int = 640,
        confidence_threshold: float = 0.25,
    ) -> None:
        self.model_path = Path(model_path)
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"TABLEMIND vision model not found: {self.model_path}"
            )

        self.model = YOLO(str(self.model_path))

    def detect(
        self,
        image: np.ndarray,
    ) -> tuple[VisionDetection, ...]:
        """Run OpenVINO YOLO inference on an RGB image."""

        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a NumPy array.")

        if image.ndim != 3:
            raise ValueError(
                "image must have shape (height, width, channels)."
            )

        if image.shape[2] != 3:
            raise ValueError(
                "image must contain exactly 3 color channels."
            )

        results = self.model.predict(
            source=image,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            verbose=False,
        )

        detections: list[VisionDetection] = []

        for result in results:
            if result.boxes is None:
                continue

            names = result.names

            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())

                coordinates = box.xyxy[0].cpu().numpy()

                left, top, right, bottom = (
                    float(value)
                    for value in coordinates
                )

                object_type = str(names[class_id])

                detections.append(
                    VisionDetection(
                        object_type=object_type,
                        confidence=confidence,
                        bounding_box=(
                            left,
                            top,
                            right,
                            bottom,
                        ),
                    )
                )

        return tuple(detections)