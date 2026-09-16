"""Transcript accumulation for TABLEMIND conversational input."""

from __future__ import annotations


class TranscriptBuffer:
    """Accumulates Speechmatics final transcript segments."""

    def __init__(self) -> None:
        self._parts: list[str] = []

    def add_final(self, text: str) -> None:
        """Add one finalized transcript segment."""
        text = text.strip()
        if text:
            self._parts.append(text)

    def get_text(self) -> str:
        """Return the accumulated transcript as one utterance."""
        return " ".join(self._parts).strip()

    def clear(self) -> None:
        """Clear the current utterance."""
        self._parts.clear()

    def has_text(self) -> bool:
        """Return whether any transcript has been accumulated."""
        return bool(self.get_text())