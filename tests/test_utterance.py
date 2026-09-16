import asyncio

from tablemind.conversation.utterance import UtteranceCoordinator


def test_complete_partial_finalizes_after_debounce() -> None:
    results: list[str] = []

    async def on_utterance(text: str) -> None:
        results.append(text)

    async def run() -> None:
        coordinator = UtteranceCoordinator(
            silence_timeout=0.05,
            on_utterance=on_utterance,
        )

        coordinator.add_partial("Set the table for two.")

        await asyncio.sleep(0.1)

        assert results == ["Set the table for two."]

    asyncio.run(run())


def test_final_segments_do_not_change_partial_hypothesis() -> None:
    results: list[str] = []

    async def on_utterance(text: str) -> None:
        results.append(text)

    async def run() -> None:
        coordinator = UtteranceCoordinator(
            silence_timeout=0.05,
            on_utterance=on_utterance,
        )

        coordinator.add_partial("Set the table for two.")
        coordinator.add_final("Set")
        coordinator.add_final("the")
        coordinator.add_final("table")
        coordinator.add_final("for")
        coordinator.add_final("two.")

        await asyncio.sleep(0.1)

        assert results == ["Set the table for two."]

    asyncio.run(run())


def test_latest_partial_replaces_previous_hypothesis() -> None:
    results: list[str] = []

    async def on_utterance(text: str) -> None:
        results.append(text)

    async def run() -> None:
        coordinator = UtteranceCoordinator(
            silence_timeout=0.05,
            on_utterance=on_utterance,
        )

        coordinator.add_partial("Set")
        coordinator.add_partial("Set the")
        coordinator.add_partial("Set the table")
        coordinator.add_partial("Set the table for two.")

        await asyncio.sleep(0.1)

        assert results == ["Set the table for two."]

    asyncio.run(run())


def test_incomplete_partial_does_not_finalize() -> None:
    results: list[str] = []

    async def on_utterance(text: str) -> None:
        results.append(text)

    async def run() -> None:
        coordinator = UtteranceCoordinator(
            silence_timeout=0.05,
            on_utterance=on_utterance,
        )

        coordinator.add_partial("Set the table for two")

        await asyncio.sleep(0.1)

        assert results == []

    asyncio.run(run())


def test_close_finalizes_complete_partial() -> None:
    results: list[str] = []

    async def on_utterance(text: str) -> None:
        results.append(text)

    async def run() -> None:
        coordinator = UtteranceCoordinator(
            silence_timeout=10.0,
            on_utterance=on_utterance,
        )

        coordinator.add_partial("Set the table for two.")

        await coordinator.close()

        assert results == ["Set the table for two."]

    asyncio.run(run())


def test_empty_transcripts_are_ignored() -> None:
    results: list[str] = []

    async def on_utterance(text: str) -> None:
        results.append(text)

    async def run() -> None:
        coordinator = UtteranceCoordinator(
            silence_timeout=0.05,
            on_utterance=on_utterance,
        )

        coordinator.add_partial("")
        coordinator.add_partial("   ")
        coordinator.add_final("")
        coordinator.add_final("   ")

        await asyncio.sleep(0.1)

        assert results == []

    asyncio.run(run())