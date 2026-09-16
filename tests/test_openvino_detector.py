import numpy as np
import openvino as ov

from tablemind.perception.openvino_detector import (
    OpenVINOInferenceEngine,
)


def create_identity_model(path):
    """Create a tiny OpenVINO model for testing."""
    parameter = ov.opset13.parameter(
        [1, 3],
        ov.Type.f32,
    )

    result = ov.opset13.result(parameter)

    model = ov.Model(
        [result],
        [parameter],
        "tablemind_identity_test",
    )

    ov.serialize(model, str(path))


def test_openvino_engine_loads_and_runs(tmp_path):
    model_path = tmp_path / "identity.xml"

    create_identity_model(model_path)

    engine = OpenVINOInferenceEngine(model_path)

    assert engine.input_shape == (1, 3)

    input_tensor = np.array(
        [[1.0, 2.0, 3.0]],
        dtype=np.float32,
    )

    result = engine.infer(input_tensor)

    assert len(result.outputs) == 1
    np.testing.assert_allclose(
        result.outputs[0],
        input_tensor,
    )