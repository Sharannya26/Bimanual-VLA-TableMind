"""Tests for TABLEMIND Milestone 9.5 goal reconciliation."""

from __future__ import annotations

from tablemind.perception.scene import SceneObject, SceneObservation
from tablemind.reasoning.goal_reconciliation import GoalReconciler


def make_scene(
    *,
    active_settings: int,
) -> SceneObservation:
    objects: list[SceneObject] = []

    if active_settings >= 1:
        objects.extend(
            [
                SceneObject(
                    "plate_1",
                    "plate",
                    (-0.18, -0.04, 0.772),
                ),
                SceneObject(
                    "glass_1",
                    "glass",
                    (-0.40, -0.14, 0.835),
                ),
            ]
        )

    if active_settings >= 2:
        objects.extend(
            [
                SceneObject(
                    "plate_2",
                    "plate",
                    (0.18, -0.04, 0.772),
                ),
                SceneObject(
                    "glass_2",
                    "glass",
                    (0.40, -0.14, 0.835),
                ),
            ]
        )

    return SceneObservation(objects=tuple(objects))


def test_one_to_two_is_positive_delta() -> None:
    reconciler = GoalReconciler()

    scene = make_scene(active_settings=1)

    result = reconciler.reconcile(
        desired_quantity=2,
        scene=scene,
    )

    assert result.current_quantity == 1
    assert result.desired_quantity == 2
    assert result.delta == 1
    assert result.is_increase
    assert result.additions[0].index == 2
    assert result.removals == ()


def test_two_to_one_is_negative_delta() -> None:
    reconciler = GoalReconciler()

    scene = make_scene(active_settings=2)

    result = reconciler.reconcile(
        desired_quantity=1,
        scene=scene,
    )

    assert result.current_quantity == 2
    assert result.desired_quantity == 1
    assert result.delta == -1
    assert result.is_decrease
    assert len(result.removals) == 1
    assert result.removals[0].slot.index == 2


def test_two_to_two_requires_no_change() -> None:
    reconciler = GoalReconciler()

    scene = make_scene(active_settings=2)

    result = reconciler.reconcile(
        desired_quantity=2,
        scene=scene,
    )

    assert result.current_quantity == 2
    assert result.delta == 0
    assert result.already_satisfied
    assert result.additions == ()
    assert result.removals == ()


def test_incomplete_setting_is_not_counted() -> None:
    reconciler = GoalReconciler()

    scene = SceneObservation(
        objects=(
            SceneObject(
                "plate_1",
                "plate",
                (-0.18, -0.04, 0.772),
            ),
            SceneObject(
                "glass_1",
                "glass",
                (-0.40, 0.10, 0.835),
            ),
        )
    )

    result = reconciler.reconcile(
        desired_quantity=1,
        scene=scene,
    )

    assert result.current_quantity == 0
    assert result.delta == 1
    assert result.additions[0].index == 1


def test_staged_object_is_not_counted_as_active_setting() -> None:
    reconciler = GoalReconciler()

    scene = SceneObservation(
        objects=(
            SceneObject(
                "plate_1",
                "plate",
                (-0.18, -0.04, 0.772),
            ),
            SceneObject(
                "glass_1",
                "glass",
                (-0.40, -0.14, 0.835),
            ),
            # Staged second setting.
            SceneObject(
                "plate_2",
                "plate",
                (0.18, 0.32, 0.772),
            ),
            SceneObject(
                "glass_2",
                "glass",
                (0.40, 0.32, 0.835),
            ),
        )
    )

    result = reconciler.reconcile(
        desired_quantity=1,
        scene=scene,
    )

    assert result.current_quantity == 1
    assert result.delta == 0
    assert result.already_satisfied


def test_decrease_removes_highest_numbered_setting_first() -> None:
    reconciler = GoalReconciler()

    scene = make_scene(active_settings=2)

    result = reconciler.reconcile(
        desired_quantity=0,
        scene=scene,
    )

    assert result.current_quantity == 2
    assert result.desired_quantity == 0
    assert result.delta == -2

    removed_indexes = [
        state.slot.index
        for state in result.removals
    ]

    assert removed_indexes == [2, 1]


def test_verification_uses_active_slots_not_object_count() -> None:
    reconciler = GoalReconciler()

    scene = SceneObservation(
        objects=(
            SceneObject(
                "plate_1",
                "plate",
                (-0.18, -0.04, 0.772),
            ),
            SceneObject(
                "glass_1",
                "glass",
                (-0.40, -0.14, 0.835),
            ),
            SceneObject(
                "plate_2",
                "plate",
                (0.18, 0.32, 0.772),
            ),
            SceneObject(
                "glass_2",
                "glass",
                (0.40, 0.32, 0.835),
            ),
        )
    )

    assert reconciler.verify(
        desired_quantity=1,
        scene=scene,
    )

    assert not reconciler.verify(
        desired_quantity=2,
        scene=scene,
    )