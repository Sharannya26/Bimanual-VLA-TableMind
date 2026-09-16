"""Demonstrate natural-language task interpretation."""

from __future__ import annotations

from tablemind.reasoning import TaskInterpreter


def main() -> None:
    interpreter = TaskInterpreter()

    instructions = (
        "Set the table for two.",
        "Set the table for four.",
        "Set the table for 3.",
    )

    print("=" * 64)
    print("TABLEMIND - Milestone 6.1")
    print("Natural-Language Task Representation")
    print("=" * 64)

    for instruction in instructions:
        print(f"\nHuman: {instruction}")

        task = interpreter.interpret(
            instruction
        )

        print("TABLEMIND:")
        print(f"  Intent: {task.intent}")
        print(
            f"  Objects: "
            f"{', '.join(task.object_types)}"
        )
        print(f"  Quantity: {task.quantity}")
        print(
            f"  Constraints: "
            f"{task.constraints}"
        )

    print("\n" + "=" * 64)
    print("✅ NATURAL-LANGUAGE TASK REPRESENTATION WORKING")
    print(
        "Human instructions can now be converted "
        "into structured TaskRequest objects."
    )
    print("=" * 64)


if __name__ == "__main__":
    main()