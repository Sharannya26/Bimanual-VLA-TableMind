"""Conversational task modification support for TABLEMIND."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModificationRequest:
    """Structured representation of a conversational task modification."""

    raw_instruction: str
    intent: str
    quantity: int | None = None
    is_modification: bool = True

    def to_dict(self) -> dict[str, object]:
        """Return the modification request as a serializable dictionary."""

        return {
            "raw_instruction": self.raw_instruction,
            "intent": self.intent,
            "quantity": self.quantity,
            "is_modification": self.is_modification,
        }