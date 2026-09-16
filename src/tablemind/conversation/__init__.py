"""Conversation and speech interfaces for TABLEMIND."""

from tablemind.conversation.speechmatics import (
    SpeechmaticsConfig,
    SpeechmaticsCredentialError,
    SpeechmaticsTranscriber,
)

__all__ = [
    "SpeechmaticsConfig",
    "SpeechmaticsCredentialError",
    "SpeechmaticsTranscriber",
]