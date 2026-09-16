"""Interpret conversational modifications for TABLEMIND."""

from __future__ import annotations

import re

from tablemind.conversation.modification import ModificationRequest


class ModificationInterpreter:
    """Convert conversational corrections into structured requests."""

    _QUANTITY_WORDS = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
    }

    _MODIFICATION_PATTERNS = (
        r"\bactually\b",
        r"\bchange\b",
        r"\bmake it\b",
        r"\bmake that\b",
        r"\binstead\b",
        r"\bmodify\b",
        r"\bupdate\b",
    )

    def interpret(
        self,
        instruction: str,
    ) -> ModificationRequest:
        """Interpret a conversational modification."""

        instruction = instruction.strip()

        if not instruction:
            raise ValueError(
                "Modification instruction cannot be empty."
            )

        lowered = instruction.lower()

        is_modification = any(
            re.search(pattern, lowered)
            for pattern in self._MODIFICATION_PATTERNS
        )

        if not is_modification:
            raise ValueError(
                "Instruction does not appear to be a modification."
            )

        quantity = self._extract_quantity(lowered)

        return ModificationRequest(
            raw_instruction=instruction,
            intent="set_table",
            quantity=quantity,
        )

    def _extract_quantity(
        self,
        instruction: str,
    ) -> int | None:
        """Extract a numeric or word-based quantity."""

        numeric_match = re.search(
            r"\b([1-5])\b",
            instruction,
        )

        if numeric_match:
            return int(numeric_match.group(1))

        for word, value in self._QUANTITY_WORDS.items():
            if re.search(
                rf"\b{word}\b",
                instruction,
            ):
                return value

        return None