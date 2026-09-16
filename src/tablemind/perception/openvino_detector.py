"""OpenVINO inference utilities for TABLEMIND."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openvino as ov


@dataclass(frozen=True)
class OpenVINOInferenceResult:
    """Raw inference outputs produced by an OpenVINO model."""

    outputs: tuple[np.ndarray, ...]


class OpenVINOInferenceEngine:
    """Small OpenVINO runtime wrapper used by TABLEMIND perception."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "CPU",
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"OpenVINO model not found: {self.model_path}"
            )

        self.core = ov.Core()
        self.model = self.core.read_model(self.model_path)

        self.compiled_model = self.core.compile_model(
            self.model,
            self.device,
        )

        self.infer_request = (
            self.compiled_model.create_infer_request()
        )

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Return the model's static input tensor shape."""

        shape = self.model.input(0).partial_shape

        if not shape.is_static:
            raise ValueError(
                "TABLEMIND currently requires a static model input shape."
            )

        return tuple(
            dimension.get_length()
            for dimension in shape
        )

    @property
    def input_dtype(self) -> np.dtype:
        """Return the NumPy dtype expected by the model."""

        return np.dtype(
            self.model.input(0).element_type.to_dtype()
        )

    def infer(
        self,
        input_tensor: np.ndarray,
    ) -> OpenVINOInferenceResult:
        """Run inference on a prepared input tensor."""

        if not isinstance(input_tensor, np.ndarray):
            raise TypeError(
                "input_tensor must be a NumPy array."
            )

        result = self.infer_request.infer(
            {
                self.compiled_model.input(0): input_tensor
            }
        )

        outputs = tuple(
            np.asarray(result[output])
            for output in self.compiled_model.outputs
        )

        return OpenVINOInferenceResult(
            outputs=outputs
        )