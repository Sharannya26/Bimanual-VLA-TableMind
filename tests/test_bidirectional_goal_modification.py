"""Tests for TABLEMIND Milestone 9.5 bidirectional goal modification."""

from __future__ import annotations

from types import SimpleNamespace

from tablemind.conversation.goal_modification_pipeline import (
    BidirectionalGoalModificationPipeline,
)
from tablemind.manipulation.goal_addition import AdditionResult
from tablemind.manipulation.removal import RemovalResult
from tablemind.reasoning.goal_reconciliation import GoalReconciler
from tablemind.reasoning.task_request import TaskRequest


# ============================================================
# Fake scene
# ============================================================


class FakeScene:
    """Minimal scene implementation required by the 9.5 layer."""

    def __init__(self, objects):
        self.objects = tuple(objects)

    def by_type(self, object_type):
        return tuple(
            obj
            for obj in self.objects
            if obj.object_type == object_type
        )

    def get(self, object_id):
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj

        return None


def make_object(object_id, object_type, position):
    """Create a lightweight scene object for testing."""

    return SimpleNamespace(
        object_id=object_id,
        object_type=object_type,
        position=position,
    )


def make_scene(quantity):
    """
    Create a logical physical scene.

    Setting 1:
        plate_1 + glass_1

    Setting 2:
        plate_2 + glass_2

    Important:
    Even when the current goal is only one setting, the physical
    objects for setting 2 still exist somewhere in the scene.
    They represent available objects that can be picked and moved
    into the second place-setting slot when the user increases
    the goal from 1 to 2.
    """

    objects = []

    # --------------------------------------------------------
    # Setting 1
    # --------------------------------------------------------

    objects.extend(
        [
            make_object(
                "plate_1",
                "plate",
                (-0.18, -0.04, 0.772),
            ),
            make_object(
                "glass_1",
                "glass",
                (-0.40, -0.14, 0.835),
            ),
        ]
    )

    # --------------------------------------------------------
    # Setting 2
    # --------------------------------------------------------
    #
    # When quantity == 2, the objects are already in their
    # active logical slots.
    #
    # When quantity == 1, the objects remain available at
    # source positions. This models the real physical world:
    # increasing the conversational goal does not create new
    # objects; the robot must find and move existing objects.

    if quantity >= 2:
        plate_2_position = (
            0.18,
            -0.04,
            0.772,
        )

        glass_2_position = (
            0.40,
            -0.14,
            0.835,
        )

    else:
        plate_2_position = (
            0.28,
            0.00,
            0.772,
        )

        glass_2_position = (
            0.40,
            -0.14,
            0.835,
        )

    objects.extend(
        [
            make_object(
                "plate_2",
                "plate",
                plate_2_position,
            ),
            make_object(
                "glass_2",
                "glass",
                glass_2_position,
            ),
        ]
    )

    return FakeScene(objects)


# ============================================================
# Fake conversational layer
# ============================================================


class FakeModification:
    """Minimal modification request for pipeline tests."""

    def __init__(self, quantity):
        self.raw_instruction = (
            f"change quantity to {quantity}"
        )
        self.intent = "set_table"
        self.quantity = quantity
        self.is_modification = True


class FakeModificationPipeline:
    """
    Deterministic conversational layer.

    These tests intentionally isolate the 9.5 goal
    reconciliation layer from natural-language parsing.
    """

    def __init__(self, quantity):
        self.quantity = quantity

    def interpret(self, instruction):
        return FakeModification(self.quantity)

    def update_request(
        self,
        current_request,
        modification,
    ):
        return TaskRequest(
            raw_instruction=modification.raw_instruction,
            intent=current_request.intent,
            object_types=current_request.object_types,
            quantity=modification.quantity,
            constraints=current_request.constraints,
        )


def make_request(quantity):
    """Create a basic TABLEMIND task request."""

    return TaskRequest(
        raw_instruction=f"Set table for {quantity}",
        intent="set_table",
        object_types=("plate", "glass"),
        quantity=quantity,
        constraints=(),
    )


# ============================================================
# Fake physical executors
# ============================================================


class FakeAdditionExecutor:
    """Records conversationally requested additions."""

    def __init__(self):
        self.calls = []

    def add_setting(
        self,
        slot,
        plate,
        glass,
    ):
        self.calls.append(
            (
                slot.index,
                plate.object_id,
                glass.object_id,
            )
        )

        return (
            AdditionResult(
                success=True,
                arm="right",
                object_id=plate.object_id,
                message="Plate restored.",
            ),
            AdditionResult(
                success=True,
                arm="right",
                object_id=glass.object_id,
                message="Glass restored.",
            ),
        )


class FakeRemovalExecutor:
    """Records conversationally requested removals."""

    def __init__(self):
        self.calls = []

    def remove_setting(
        self,
        plate,
        glass,
    ):
        self.calls.append(
            (
                plate.object_id,
                glass.object_id,
            )
        )

        return (
            RemovalResult(
                object_id=plate.object_id,
                arm="right",
                success=True,
                reason="Plate staged.",
            ),
            RemovalResult(
                object_id=glass.object_id,
                arm="right",
                success=True,
                reason="Glass staged.",
            ),
        )


# ============================================================
# Increase tests
# ============================================================


def test_increase_from_one_to_two():
    """1 → 2 should request exactly one additional setting."""

    addition_executor = FakeAdditionExecutor()

    pipeline = BidirectionalGoalModificationPipeline(
        modification_pipeline=FakeModificationPipeline(2),
        goal_reconciler=GoalReconciler(),
        addition_executor=addition_executor,
    )

    scene = make_scene(1)

    result = pipeline.reconcile(
        current_request=make_request(1),
        instruction="Actually, make it 2.",
        scene=scene,
    )

    assert result.increased
    assert result.reconciliation.current_quantity == 1
    assert result.reconciliation.desired_quantity == 2
    assert result.reconciliation.delta == 1

    assert tuple(
        slot.index
        for slot in result.reconciliation.additions
    ) == (2,)


def test_execute_increase_from_one_to_two():
    """1 → 2 should physically restore setting 2."""

    addition_executor = FakeAdditionExecutor()

    pipeline = BidirectionalGoalModificationPipeline(
        modification_pipeline=FakeModificationPipeline(2),
        goal_reconciler=GoalReconciler(),
        addition_executor=addition_executor,
    )

    scene = make_scene(1)

    result = pipeline.reconcile(
        current_request=make_request(1),
        instruction="Actually, make it 2.",
        scene=scene,
    )

    executed = pipeline.execute(
        result=result,
        scene=scene,
    )

    assert executed.addition_succeeded
    assert executed.execution_succeeded

    assert addition_executor.calls == [
        (
            2,
            "plate_2",
            "glass_2",
        )
    ]


# ============================================================
# Decrease tests
# ============================================================


def test_decrease_from_two_to_one():
    """2 → 1 should select the second complete place setting."""

    removal_executor = FakeRemovalExecutor()

    pipeline = BidirectionalGoalModificationPipeline(
        modification_pipeline=FakeModificationPipeline(1),
        goal_reconciler=GoalReconciler(),
        removal_executor=removal_executor,
    )

    scene = make_scene(2)

    result = pipeline.reconcile(
        current_request=make_request(2),
        instruction="Actually, one guest isn't coming.",
        scene=scene,
    )

    assert result.decreased
    assert result.reconciliation.current_quantity == 2
    assert result.reconciliation.desired_quantity == 1
    assert result.reconciliation.delta == -1

    assert tuple(
        state.slot.index
        for state in result.reconciliation.removals
    ) == (2,)


def test_execute_decrease_from_two_to_one():
    """2 → 1 should physically remove setting 2."""

    removal_executor = FakeRemovalExecutor()

    pipeline = BidirectionalGoalModificationPipeline(
        modification_pipeline=FakeModificationPipeline(1),
        goal_reconciler=GoalReconciler(),
        removal_executor=removal_executor,
    )

    scene = make_scene(2)

    result = pipeline.reconcile(
        current_request=make_request(2),
        instruction="Actually, one guest isn't coming.",
        scene=scene,
    )

    executed = pipeline.execute(
        result=result,
        scene=scene,
    )

    assert executed.removal_succeeded
    assert executed.execution_succeeded

    assert removal_executor.calls == [
        (
            "plate_2",
            "glass_2",
        )
    ]


# ============================================================
# No-op test
# ============================================================


def test_already_satisfied_goal_is_noop():
    """2 → 2 should require no physical action."""

    addition_executor = FakeAdditionExecutor()
    removal_executor = FakeRemovalExecutor()

    pipeline = BidirectionalGoalModificationPipeline(
        modification_pipeline=FakeModificationPipeline(2),
        goal_reconciler=GoalReconciler(),
        addition_executor=addition_executor,
        removal_executor=removal_executor,
    )

    scene = make_scene(2)

    result = pipeline.reconcile(
        current_request=make_request(2),
        instruction="Actually, keep it at 2.",
        scene=scene,
    )

    assert result.already_satisfied
    assert result.reconciliation.delta == 0

    executed = pipeline.execute(
        result=result,
        scene=scene,
    )

    assert executed.execution_succeeded
    assert addition_executor.calls == []
    assert removal_executor.calls == []


# ============================================================
# Round-trip test
# ============================================================


def test_bidirectional_round_trip():
    """
    The same architecture must support:

        1 → 2
        2 → 1

    without changing the underlying 9.4 execution model.
    """

    addition_executor = FakeAdditionExecutor()
    removal_executor = FakeRemovalExecutor()

    # --------------------------------------------------------
    # Increase: 1 → 2
    # --------------------------------------------------------

    pipeline_up = BidirectionalGoalModificationPipeline(
        modification_pipeline=FakeModificationPipeline(2),
        goal_reconciler=GoalReconciler(),
        addition_executor=addition_executor,
    )

    scene_one = make_scene(1)

    increase = pipeline_up.reconcile(
        current_request=make_request(1),
        instruction="Actually, make it 2.",
        scene=scene_one,
    )

    increase_executed = pipeline_up.execute(
        result=increase,
        scene=scene_one,
    )

    assert increase_executed.execution_succeeded

    assert addition_executor.calls == [
        (
            2,
            "plate_2",
            "glass_2",
        )
    ]

    # --------------------------------------------------------
    # Decrease: 2 → 1
    # --------------------------------------------------------

    pipeline_down = BidirectionalGoalModificationPipeline(
        modification_pipeline=FakeModificationPipeline(1),
        goal_reconciler=GoalReconciler(),
        removal_executor=removal_executor,
    )

    scene_two = make_scene(2)

    decrease = pipeline_down.reconcile(
        current_request=make_request(2),
        instruction="Actually, one guest isn't coming.",
        scene=scene_two,
    )

    decrease_executed = pipeline_down.execute(
        result=decrease,
        scene=scene_two,
    )

    assert decrease_executed.execution_succeeded

    assert removal_executor.calls == [
        (
            "plate_2",
            "glass_2",
        )
    ]