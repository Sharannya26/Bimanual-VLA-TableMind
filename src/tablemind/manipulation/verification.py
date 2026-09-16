"""Manipulation verification utilities for TABLEMIND."""

from __future__ import annotations

from dataclasses import dataclass

from tablemind.manipulation.objects import DynamicObjectManager


@dataclass(frozen=True)
class VerificationResult:
    """Result of verifying an object's final position."""

    success: bool
    object_id: str
    expected_position: tuple[float, float, float]
    actual_position: tuple[float, float, float]
    position_error: float
    tolerance: float
    reason: str


class ManipulationVerifier:
    """Verify physical manipulation outcomes."""

    def __init__(
        self,
        objects: DynamicObjectManager,
    ) -> None:
        self.objects = objects

    def verify_position(
        self,
        object_id: str,
        expected_position: tuple[float, float, float],
        *,
        tolerance: float = 0.06,
    ) -> VerificationResult:
        """Verify that an object reached an expected position."""

        actual = self.objects.get(object_id).position

        dx = actual[0] - expected_position[0]
        dy = actual[1] - expected_position[1]
        dz = actual[2] - expected_position[2]

        error = (
            dx * dx
            + dy * dy
            + dz * dz
        ) ** 0.5

        success = error <= tolerance

        if success:
            reason = (
                f"{object_id} reached the expected region "
                f"(error={error:.3f} m)."
            )
        else:
            reason = (
                f"{object_id} missed the expected region "
                f"(error={error:.3f} m > "
                f"{tolerance:.3f} m)."
            )

        return VerificationResult(
            success=success,
            object_id=object_id,
            expected_position=expected_position,
            actual_position=actual,
            position_error=error,
            tolerance=tolerance,
            reason=reason,
        )