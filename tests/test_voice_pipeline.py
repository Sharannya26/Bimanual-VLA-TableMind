import pytest

from tablemind.conversation.voice_pipeline import VoiceTaskPipeline


def test_voice_pipeline_converts_transcript_to_task_request() -> None:
    pipeline = VoiceTaskPipeline()

    request = pipeline.process("Set the table for two.")

    assert request.raw_instruction == "Set the table for two."
    assert request.intent == "set_table"
    assert request.quantity == 2
    assert request.object_types == ("plate", "glass")


def test_voice_pipeline_strips_transcript_whitespace() -> None:
    pipeline = VoiceTaskPipeline()

    request = pipeline.process("   Set the table for two.   ")

    assert request.raw_instruction == "Set the table for two."


def test_voice_pipeline_rejects_empty_transcript() -> None:
    pipeline = VoiceTaskPipeline()

    with pytest.raises(ValueError, match="cannot be empty"):
        pipeline.process("")


def test_voice_pipeline_rejects_whitespace_only_transcript() -> None:
    pipeline = VoiceTaskPipeline()

    with pytest.raises(ValueError, match="cannot be empty"):
        pipeline.process("   ")