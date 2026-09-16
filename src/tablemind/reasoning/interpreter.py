"""Simple natural-language task interpreter for TABLEMIND."""

from __future__ import annotations

import re

from tablemind.reasoning.task_request import TaskRequest


class TaskInterpreter:
    """Convert simple natural-language instructions into TaskRequest objects."""

    def interpret(self, instruction: str) -> TaskRequest:
        """Interpret one natural-language manipulation instruction."""

        normalized = instruction.strip().lower()

        if not normalized:
            raise ValueError(
                "Instruction cannot be empty."
            )

        if "set the table" in normalized:
            return self._interpret_set_table(
                instruction,
                normalized,
            )

        raise ValueError(
            f"Unsupported task instruction: {instruction!r}"
        )

    def _interpret_set_table(
        self,
        instruction: str,
        normalized: str,
    ) -> TaskRequest:
        """Interpret a table-setting instruction."""

        quantity = self._extract_quantity(
            normalized
        )

        return TaskRequest(
            raw_instruction=instruction,
            intent="set_table",
            object_types=(
                "plate",
                "glass",
            ),
            quantity=quantity,
        )

    def _extract_quantity(
        self,
        instruction: str,
    ) -> int | None:
        """Extract a numeric quantity from an instruction."""

        match = re.search(
            r"\b(\d+)\b",
            instruction,
        )

        if match:
            return int(match.group(1))

        word_numbers = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
        }

        for word, value in word_numbers.items():
            if re.search(
                rf"\b{word}\b",
                instruction,
            ):
                return value

        return None