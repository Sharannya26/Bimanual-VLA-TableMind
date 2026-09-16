from tablemind.conversation.transcript_buffer import TranscriptBuffer


def test_accumulates_final_transcript_segments() -> None:
    buffer = TranscriptBuffer()

    buffer.add_final("Set the")
    buffer.add_final("table")
    buffer.add_final("for")
    buffer.add_final("two.")

    assert buffer.get_text() == "Set the table for two."


def test_ignores_empty_segments() -> None:
    buffer = TranscriptBuffer()

    buffer.add_final("Set the")
    buffer.add_final("")
    buffer.add_final("  ")
    buffer.add_final("table")

    assert buffer.get_text() == "Set the table"


def test_clear_resets_buffer() -> None:
    buffer = TranscriptBuffer()

    buffer.add_final("Set the table")
    buffer.clear()

    assert buffer.get_text() == ""
    assert buffer.has_text() is False


def test_has_text() -> None:
    buffer = TranscriptBuffer()

    assert buffer.has_text() is False

    buffer.add_final("Hello")

    assert buffer.has_text() is True