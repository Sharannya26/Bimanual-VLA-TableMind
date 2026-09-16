from tablemind.conversation.transcript_buffer import TranscriptBuffer
from tablemind.reasoning.interpreter import TaskInterpreter


def test_speech_transcript_reaches_task_interpreter() -> None:
    buffer = TranscriptBuffer()

    # Simulate the final transcript segments produced by Speechmatics.
    buffer.add_final("Set the")
    buffer.add_final("table")
    buffer.add_final("for")
    buffer.add_final("two.")

    transcript = buffer.get_text()

    interpreter = TaskInterpreter()
    request = interpreter.interpret(transcript)

    assert request.raw_instruction == "Set the table for two."
    assert request.intent == "set_table"
    assert request.quantity == 2
    assert request.object_types == ("plate", "glass")