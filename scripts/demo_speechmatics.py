"""Live Speechmatics -> TABLEMIND NLU demo."""

from __future__ import annotations

import asyncio

import pyaudio

from tablemind.conversation.speechmatics import SpeechmaticsTranscriber
from tablemind.conversation.utterance import UtteranceCoordinator
from tablemind.reasoning.interpreter import TaskInterpreter


DEVICE_INDEX = 0
SAMPLE_RATE = 16_000
CHANNELS = 1
CHUNK_SIZE = 4096
SILENCE_TIMEOUT = 1.0


async def main() -> None:
    transcriber = SpeechmaticsTranscriber()
    interpreter = TaskInterpreter()

    print("=" * 60)
    print("TABLEMIND — Speechmatics → NLU")
    print("=" * 60)
    print()
    print('Say something like: "Set the table for two."')
    print("Pause briefly after speaking.")
    print("Press Ctrl+C to stop.")
    print()

    async def handle_utterance(transcript: str) -> None:
        print()
        print(f"[utterance] {transcript}")
        print()

        try:
            request = interpreter.interpret(transcript)

            print("TABLEMIND NLU")
            print("-" * 40)
            print(f"Intent:    {request.intent}")
            print(f"Objects:   {request.object_types}")
            print(f"Quantity:  {request.quantity}")
            print(f"Command:   {request.raw_instruction}")
            print("-" * 40)
            print()

        except Exception as exc:
            print(f"[NLU error] {exc}")
            print()

    coordinator = UtteranceCoordinator(
        silence_timeout=SILENCE_TIMEOUT,
        on_utterance=handle_utterance,
    )

    audio = pyaudio.PyAudio()

    print(f"Opening microphone device {DEVICE_INDEX}...")

    stream = audio.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=DEVICE_INDEX,
        frames_per_buffer=CHUNK_SIZE,
    )

    print("Microphone opened successfully.")
    print()

    async def audio_source(chunk_size: int) -> bytes:
        return await asyncio.to_thread(
            stream.read,
            chunk_size,
            exception_on_overflow=False,
        )

    def handle_partial(text: str) -> None:
        coordinator.add_partial(text)
        print(f"\r[partial] {text}", end="", flush=True)

    def handle_final(text: str) -> None:
        print(f"\n[final]   {text}")
        coordinator.add_final(text)

    try:
        await transcriber.transcribe(
            audio_source,
            on_partial=handle_partial,
            on_final=handle_final,
        )

    finally:
        await coordinator.close()

        stream.stop_stream()
        stream.close()
        audio.terminate()

        print("\nMicrophone closed.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nSpeechmatics demo stopped.")