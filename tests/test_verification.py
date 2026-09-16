from tablemind.manipulation.verification import (
    ManipulationVerifier,
)


class FakeObject:
    def __init__(self, position: tuple[float, float, float]) -> None:
        self.position = position


class FakeObjects:
    def __init__(self) -> None:
        self.objects = {
            "plate_1": FakeObject(
                (0.10, 0.20, 0.30)
            )
        }

    def get(self, object_id: str) -> FakeObject:
        return self.objects[object_id]


def test_verification_succeeds_inside_tolerance() -> None:
    verifier = ManipulationVerifier(
        FakeObjects()
    )

    result = verifier.verify_position(
        "plate_1",
        (0.10, 0.20, 0.30),
    )

    assert result.success is True
    assert result.position_error == 0.0


def test_verification_fails_outside_tolerance() -> None:
    verifier = ManipulationVerifier(
        FakeObjects()
    )

    result = verifier.verify_position(
        "plate_1",
        (0.30, 0.20, 0.30),
        tolerance=0.05,
    )

    assert result.success is False
    assert result.position_error > 0.05