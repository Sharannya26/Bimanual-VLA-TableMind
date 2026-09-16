"""
Streamlit voice bridge for TABLEMIND.

Speechmatics realtime microphone
        ↓
UtteranceCoordinator
        ↓
complete natural-language command
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pyaudio

from tablemind.conversation.speechmatics import SpeechmaticsTranscriber
from tablemind.conversation.utterance import UtteranceCoordinator


SAMPLE_RATE = 16_000
CHANNELS = 1
CHUNK_SIZE = 4096
DEVICE_INDEX = 0
SILENCE_TIMEOUT = 1.0


@dataclass(frozen=True)
class VoiceCaptureResult:
    transcript: str
    success: bool
    error: str = ""


def capture_voice_command() -> VoiceCaptureResult:
    """Capture one complete spoken TABLEMIND command."""

    try:
        return asyncio.run(_capture_voice_command())

    except KeyboardInterrupt:
        return VoiceCaptureResult(
            transcript="",
            success=False,
            error="Voice capture interrupted.",
        )

    except Exception as exc:
        return VoiceCaptureResult(
            transcript="",
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )


async def _capture_voice_command() -> VoiceCaptureResult:
    audio = pyaudio.PyAudio()
    stream = None

    transcript_result: dict[str, str] = {}
    utterance_received = asyncio.Event()

    try:
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=DEVICE_INDEX,
            frames_per_buffer=CHUNK_SIZE,
        )

        async def handle_utterance(text: str):
            transcript_result["text"] = text.strip()
            utterance_received.set()

        coordinator = UtteranceCoordinator(
            silence_timeout=SILENCE_TIMEOUT,
            on_utterance=handle_utterance,
        )

        async def audio_source(chunk_size: int):
            if utterance_received.is_set():
                return None

            return await asyncio.to_thread(
                stream.read,
                chunk_size,
                exception_on_overflow=False,
            )

        def handle_partial(text: str):
            coordinator.add_partial(text)

        def handle_final(text: str):
            coordinator.add_final(text)

        transcriber = SpeechmaticsTranscriber()

        await transcriber.transcribe(
            audio_source,
            on_partial=handle_partial,
            on_final=handle_final,
        )

        await coordinator.close()

        transcript = transcript_result.get("text", "").strip()

        if not transcript:
            return VoiceCaptureResult(
                transcript="",
                success=False,
                error="Speechmatics did not produce a complete command.",
            )

        return VoiceCaptureResult(
            transcript=transcript,
            success=True,
        )

    except Exception as exc:
        return VoiceCaptureResult(
            transcript=transcript_result.get("text", "").strip(),
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    finally:
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass

            try:
                stream.close()
            except Exception:
                pass

        audio.terminate()