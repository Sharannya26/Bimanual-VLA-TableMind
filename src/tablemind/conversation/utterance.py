"""Utterance finalization for TABLEMIND conversational input."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable


_SENTENCE_END_RE = re.compile(r"""[.!?]["')\]]*$""")


class UtteranceCoordinator:
    """Build complete user utterances from Speechmatics partial hypotheses."""

    def __init__(
        self,
        *,
        silence_timeout: float = 1.0,
        on_utterance: Callable[[str], Awaitable[None] | None],
    ) -> None:
        self.silence_timeout = silence_timeout
        self.on_utterance = on_utterance

        self._best_partial = ""
        self._timer_task: asyncio.Task[None] | None = None
        self._finalizing = False

    def add_partial(self, text: str) -> None:
        """Store the best complete partial hypothesis seen so far."""
        text = text.strip()

        if not text or self._finalizing:
            return

        # Speechmatics partials can shrink as the utterance progresses.
        # Keep the longest hypothesis because it is usually the most
        # complete representation of the spoken command.
        if len(text) > len(self._best_partial):
            self._best_partial = text

        # Any new partial means the recognizer is still active.
        # Restart the debounce timer.
        self._reset_timer()

    def add_final(self, text: str) -> None:
        """
        Ignore individual final segments.

        Speechmatics can emit several final segments for one spoken
        command, so they must not define the utterance boundary.
        """
        return

    def _reset_timer(self) -> None:
        """Restart the silence debounce timer."""
        if self._timer_task is not None:
            self._timer_task.cancel()

        self._timer_task = asyncio.create_task(
            self._finalize_after_silence()
        )

    async def _finalize_after_silence(self) -> None:
        """Finalize the best complete hypothesis after silence."""
        try:
            await asyncio.sleep(self.silence_timeout)
        except asyncio.CancelledError:
            return

        candidate = self._best_partial.strip()

        if not candidate:
            return

        if not _SENTENCE_END_RE.search(candidate):
            return

        await self.finalize()

    async def finalize(self) -> None:
        """Send the best complete utterance to the callback."""
        if self._finalizing:
            return

        candidate = self._best_partial.strip()

        if not candidate:
            return

        if not _SENTENCE_END_RE.search(candidate):
            return

        self._finalizing = True

        if self._timer_task is not None:
            self._timer_task.cancel()
            self._timer_task = None

        self._best_partial = ""

        result = self.on_utterance(candidate)

        if asyncio.iscoroutine(result):
            await result

        self._finalizing = False

    async def close(self) -> None:
        """Finalize any remaining complete hypothesis."""
        if self._timer_task is not None:
            self._timer_task.cancel()
            self._timer_task = None

        candidate = self._best_partial.strip()

        if candidate and _SENTENCE_END_RE.search(candidate):
            await self.finalize()

    def has_text(self) -> bool:
        """Return whether a partial hypothesis is available."""
        return bool(self._best_partial)