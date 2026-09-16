"""Quantity-aware object selection for TABLEMIND."""

from __future__ import annotations

from tablemind.perception.scene import SceneObservation
from tablemind.reasoning.task_request import TaskRequest


class QuantityAwareObjectSelector:
    """Select scene objects according to the requested quantity."""

    def select(
        self,
        request: TaskRequest,
        scene: SceneObservation,
    ) -> tuple[str, ...]:
        """Return object IDs selected for the requested task."""

        quantity = request.quantity

        if quantity is None:
            quantity = len(scene.objects)

        selected: list[str] = []

        for object_type in request.object_types:
            candidates = [
                obj
                for obj in scene.by_type(object_type)
            ]

            for obj in candidates:
                if obj.object_id not in selected:
                    selected.append(obj.object_id)

                if len(selected) >= quantity:
                    return tuple(selected)

        return tuple(selected)