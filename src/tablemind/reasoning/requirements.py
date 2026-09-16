"""Semantic quantity requirements for TABLEMIND."""

from __future__ import annotations

from dataclasses import dataclass

from tablemind.reasoning.task_request import TaskRequest


@dataclass(frozen=True)
class QuantityRequirement:
    """Required quantity for one object category."""

    object_type: str
    required_quantity: int


@dataclass(frozen=True)
class TaskRequirements:
    """Semantic requirements derived from a task request."""

    requirements: tuple[QuantityRequirement, ...]

    def requirement_for(
        self,
        object_type: str,
    ) -> QuantityRequirement:
        """Return the requirement for an object type."""

        for requirement in self.requirements:
            if requirement.object_type == object_type:
                return requirement

        raise KeyError(
            f"No requirement exists for {object_type!r}"
        )

    @property
    def object_types(self) -> tuple[str, ...]:
        """Return all required object types."""

        return tuple(
            requirement.object_type
            for requirement in self.requirements
        )


class TaskRequirementBuilder:
    """Convert task language into semantic object requirements."""

    def build(
        self,
        request: TaskRequest,
    ) -> TaskRequirements:
        """Build per-object-type requirements."""

        quantity = request.quantity or 0

        requirements = tuple(
            QuantityRequirement(
                object_type=object_type,
                required_quantity=quantity,
            )
            for object_type in request.object_types
        )

        return TaskRequirements(
            requirements=requirements,
        )