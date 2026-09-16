"""Goal reconciliation for bidirectional TABLEMIND task modification."""

from __future__ import annotations

from dataclasses import dataclass

from tablemind.perception.scene import SceneObject, SceneObservation


Position = tuple[float, float, float]


@dataclass(frozen=True)
class PlaceSettingSlot:
    """Logical location of one complete place setting."""

    index: int
    plate_position: Position
    glass_position: Position

    @property
    def plate_id(self) -> str:
        return f"plate_{self.index}"

    @property
    def glass_id(self) -> str:
        return f"glass_{self.index}"


@dataclass(frozen=True)
class PlaceSettingState:
    """Observed state of one logical place-setting slot."""

    slot: PlaceSettingSlot
    plate: SceneObject | None
    glass: SceneObject | None

    @property
    def is_complete(self) -> bool:
        return self.plate is not None and self.glass is not None


@dataclass(frozen=True)
class GoalReconciliation:
    """Difference between the desired and currently active table goal."""

    desired_quantity: int
    current_quantity: int
    delta: int
    additions: tuple[PlaceSettingSlot, ...]
    removals: tuple[PlaceSettingState, ...]

    @property
    def needs_change(self) -> bool:
        return self.delta != 0

    @property
    def is_increase(self) -> bool:
        return self.delta > 0

    @property
    def is_decrease(self) -> bool:
        return self.delta < 0

    @property
    def is_already_satisfied(self) -> bool:
        return self.delta == 0

    # Compatibility alias used by the 9.5 tests and future callers.
    @property
    def already_satisfied(self) -> bool:
        return self.is_already_satisfied


class GoalReconciler:
    """Reconcile a conversational goal against the physical table state."""

    DEFAULT_SLOTS = (
        PlaceSettingSlot(
            index=1,
            plate_position=(-0.18, -0.04, 0.772),
            glass_position=(-0.40, -0.14, 0.835),
        ),
        PlaceSettingSlot(
            index=2,
            plate_position=(0.18, -0.04, 0.772),
            glass_position=(0.40, -0.14, 0.835),
        ),
    )

    XY_TOLERANCE = 0.10
    Z_TOLERANCE = 0.065

    def __init__(
        self,
        slots: tuple[PlaceSettingSlot, ...] | None = None,
    ) -> None:
        self.slots = slots or self.DEFAULT_SLOTS

    def inspect(
        self,
        scene: SceneObservation,
    ) -> tuple[PlaceSettingState, ...]:
        """Map observed objects onto the logical place-setting slots."""

        states: list[PlaceSettingState] = []

        for slot in self.slots:
            plate = self._find_matching_object(
                objects=scene.by_type("plate"),
                target=slot.plate_position,
            )

            glass = self._find_matching_object(
                objects=scene.by_type("glass"),
                target=slot.glass_position,
            )

            states.append(
                PlaceSettingState(
                    slot=slot,
                    plate=plate,
                    glass=glass,
                )
            )

        return tuple(states)

    def reconcile(
        self,
        desired_quantity: int,
        scene: SceneObservation,
    ) -> GoalReconciliation:
        """Calculate what must change to reach the desired quantity."""

        if desired_quantity < 0:
            raise ValueError("Desired quantity cannot be negative.")

        if desired_quantity > len(self.slots):
            raise ValueError(
                f"Desired quantity {desired_quantity} exceeds the "
                f"{len(self.slots)} supported place-setting slots."
            )

        states = self.inspect(scene)

        active_states = tuple(
            state for state in states if state.is_complete
        )

        current_quantity = len(active_states)
        delta = desired_quantity - current_quantity

        # Increase:
        # select inactive slots in ascending logical order.
        additions: tuple[PlaceSettingSlot, ...]

        if delta > 0:
            additions = tuple(
                state.slot
                for state in states
                if not state.is_complete
            )[:delta]
        else:
            additions = ()

        # Decrease:
        # remove the highest-numbered active settings first.
        removals: tuple[PlaceSettingState, ...]

        if delta < 0:
            removals = tuple(
                sorted(
                    active_states,
                    key=lambda state: state.slot.index,
                    reverse=True,
                )
            )[: abs(delta)]
        else:
            removals = ()

        return GoalReconciliation(
            desired_quantity=desired_quantity,
            current_quantity=current_quantity,
            delta=delta,
            additions=additions,
            removals=removals,
        )

    def verify(
        self,
        desired_quantity: int,
        scene: SceneObservation,
    ) -> bool:
        """Return True when the requested number of complete settings exists."""

        states = self.inspect(scene)

        active_quantity = sum(
            state.is_complete
            for state in states
        )

        return active_quantity == desired_quantity

    def _find_matching_object(
        self,
        objects: tuple[SceneObject, ...],
        target: Position,
    ) -> SceneObject | None:
        """Find the closest object within the slot tolerance."""

        best_object: SceneObject | None = None
        best_distance: float | None = None

        for obj in objects:
            distance = self._distance(obj.position, target)

            if not self._within_tolerance(obj.position, target):
                continue

            if best_distance is None or distance < best_distance:
                best_object = obj
                best_distance = distance

        return best_object

    def _within_tolerance(
        self,
        position: Position,
        target: Position,
    ) -> bool:
        xy_distance = (
            (position[0] - target[0]) ** 2
            + (position[1] - target[1]) ** 2
        ) ** 0.5

        z_distance = abs(position[2] - target[2])

        return (
            xy_distance <= self.XY_TOLERANCE
            and z_distance <= self.Z_TOLERANCE
        )

    @staticmethod
    def _distance(
        first: Position,
        second: Position,
    ) -> float:
        return (
            (first[0] - second[0]) ** 2
            + (first[1] - second[1]) ** 2
            + (first[2] - second[2]) ** 2
        ) ** 0.5