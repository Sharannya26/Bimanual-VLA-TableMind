"""Tests for the TABLEMIND Speechmatics adapter."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tablemind.conversation.speechmatics import (
    SpeechmaticsConfig,
    SpeechmaticsCredentialError,
    SpeechmaticsTranscriber,
)


def test_speechmatics_config_defaults() -> None:
    """Verify the default real-time transcription configuration."""
    config = SpeechmaticsConfig()

    assert config.language == "en"
    assert config.sample_rate == 16_000
    assert config.chunk_size == 4096
    assert config.enable_partials is True


def test_speechmatics_uses_explicit_api_key() -> None:
    """An explicitly supplied key should be accepted."""
    transcriber = SpeechmaticsTranscriber(api_key="test-key")

    assert transcriber.api_key == "test-key"


def test_speechmatics_loads_api_key_from_environment() -> None:
    """The API key should be loaded from the environment."""
    with patch.dict(
        "os.environ",
        {"SPEECHMATICS_API_KEY": "environment-key"},
        clear=True,
    ):
        transcriber = SpeechmaticsTranscriber()

    assert transcriber.api_key == "environment-key"


def test_speechmatics_requires_api_key() -> None:
    """Missing credentials should produce a clear error."""
    with patch(
        "tablemind.conversation.speechmatics.load_dotenv"
    ):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(SpeechmaticsCredentialError):
                SpeechmaticsTranscriber()