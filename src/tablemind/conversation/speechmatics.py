"""Speechmatics real-time transcription adapter for TABLEMIND."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from dotenv import load_dotenv
from speechmatics.rt import (
    AsyncClient,
    AudioEncoding,
    AudioFormat,
    OperatingPoint,
    ServerMessageType,
    TranscriptResult,
    TranscriptionConfig,
)


@dataclass(frozen=True)
class SpeechmaticsConfig:
    """Configuration for real-time Speechmatics transcription."""

    language: str = "en"
    sample_rate: int = 16_000
    chunk_size: int = 4096
    enable_partials: bool = True
    operating_point: OperatingPoint = OperatingPoint.ENHANCED


class SpeechmaticsCredentialError(RuntimeError):
    """Raised when the Speechmatics API key is unavailable."""


class SpeechmaticsTranscriber:
    """Convert live audio into final and partial transcripts."""

    def __init__(
        self,
        api_key: str | None = None,
        config: SpeechmaticsConfig | None = None,
    ) -> None:
        load_dotenv()

        self.api_key = api_key or os.getenv("SPEECHMATICS_API_KEY")
        self.config = config or SpeechmaticsConfig()

        if not self.api_key:
            raise SpeechmaticsCredentialError(
                "SPEECHMATICS_API_KEY is not configured."
            )

    def _audio_format(self) -> AudioFormat:
        """Build the audio format expected by Speechmatics."""
        return AudioFormat(
            encoding=AudioEncoding.PCM_S16LE,
            sample_rate=self.config.sample_rate,
            chunk_size=self.config.chunk_size,
        )

    def _transcription_config(self) -> TranscriptionConfig:
        """Build the Speechmatics transcription configuration."""
        return TranscriptionConfig(
            language=self.config.language,
            enable_partials=self.config.enable_partials,
            operating_point=self.config.operating_point,
        )

    async def transcribe(
        self,
        audio_source: Callable[[int], object],
        *,
        on_partial: Callable[[str], None] | None = None,
        on_final: Callable[[str], None] | None = None,
    ) -> None:
        """Stream audio chunks to Speechmatics until the source stops.

        Args:
            audio_source:
                Callable receiving the desired chunk size and returning
                audio bytes. Returning ``None`` ends the stream.
            on_partial:
                Optional callback for partial transcripts.
            on_final:
                Optional callback for final transcripts.
        """
        audio_format = self._audio_format()
        transcription_config = self._transcription_config()

        async with AsyncClient(api_key=self.api_key) as client:

            @client.on(ServerMessageType.ADD_TRANSCRIPT)
            def handle_final_transcript(message: object) -> None:
                result = TranscriptResult.from_message(message)
                transcript = result.metadata.transcript

                if transcript and on_final is not None:
                    on_final(transcript)

            @client.on(ServerMessageType.ADD_PARTIAL_TRANSCRIPT)
            def handle_partial_transcript(message: object) -> None:
                result = TranscriptResult.from_message(message)
                transcript = result.metadata.transcript

                if transcript and on_partial is not None:
                    on_partial(transcript)

            await client.start_session(
                transcription_config=transcription_config,
                audio_format=audio_format,
            )

            while True:
                frame = await audio_source(audio_format.chunk_size)

                if frame is None:
                    break

                await client.send_audio(frame)