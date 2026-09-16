"""SO-101 naming and placement used by the bimanual scene."""

from dataclasses import dataclass


JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@dataclass(frozen=True)
class ArmPlacement:
    """World placement for one fixed-base SO-101 instance."""

    name: str
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]


ARM_PLACEMENTS: dict[str, ArmPlacement] = {
    # Both bases sit on the rear edge of the work surface and face the table centre.
    "left": ArmPlacement("left", (-0.48, -0.43, 0.76), (0.7071068, 0.0, 0.0, 0.7071068)),
    "right": ArmPlacement("right", (0.48, -0.43, 0.76), (0.7071068, 0.0, 0.0, 0.7071068)),
}


def namespaced(arm: str, name: str) -> str:
    """Return the model-wide MuJoCo name for an arm-local element."""

    if arm not in ARM_PLACEMENTS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {tuple(ARM_PLACEMENTS)}")
    return f"{arm}_{name}"
