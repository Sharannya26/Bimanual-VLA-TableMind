"""Natural-language task representations for TABLEMIND."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskRequest:
    """Structured representation of a user's manipulation request."""

    raw_instruction: str
    intent: str
    object_types: tuple[str, ...] = field(default_factory=tuple)
    quantity: int | None = None
    constraints: tuple[str, ...] = field(default_factory=tuple)

    def has_object_type(self, object_type: str) -> bool:
        """Return whether the task mentions a given object type."""

        return object_type in self.object_types

    def to_dict(self) -> dict[str, object]:
        """Convert the task request into a serializable dictionary."""

        return {
            "raw_instruction": self.raw_instruction,
            "intent": self.intent,
            "object_types": list(self.object_types),
            "quantity": self.quantity,
            "constraints": list(self.constraints),
        }