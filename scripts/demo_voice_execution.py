"""Live Speechmatics voice command -> TABLEMIND bimanual execution demo."""

from __future__ import annotations

import asyncio

import mujoco
import pyaudio

from tablemind.conversation.speechmatics import SpeechmaticsTranscriber
from tablemind.conversation.utterance import UtteranceCoordinator
from tablemind.conversation.voice_planner import VoicePlannerPipeline
from tablemind.manipulation.grasping import GraspManager
from tablemind.manipulation.synchronized_executor import (
    SynchronizedBimanualExecutor,
)
from tablemind.perception.state_builder import SceneStateBuilder
from tablemind.reasoning.execution_bridge import ReasoningExecutionBridge
from tablemind.simulation import BimanualTableSimulation


DEVICE_INDEX = 0
SAMPLE_RATE = 16_000
CHANNELS = 1
CHUNK_SIZE = 4096
SILENCE_TIMEOUT = 1.0


def perturb_plate(
    simulation: BimanualTableSimulation,
    object_id: str,
    z_offset: float,
) -> None:
    """Move one freejoint plate vertically to create an incomplete task state."""

    body_id = mujoco.mj_name2id(
        simulation.model,
        mujoco.mjtObj.mjOBJ_BODY,
        object_id,
    )

    if body_id < 0:
        raise KeyError(
            f"MuJoCo model does not contain body {object_id!r}"
        )

    joint_id = simulation.model.body_jntadr[body_id]

    if joint_id < 0:
        raise ValueError(
            f"Object {object_id!r} does not have a freejoint."
        )

    qpos_address = simulation.model.jnt_qposadr[joint_id]

    simulation.data.qpos[qpos_address + 2] += z_offset

    mujoco.mj_forward(
        simulation.model,
        simulation.data,
    )


async def main() -> None:
    print("=" * 72)
    print("TABLEMIND - MILESTONE 7.2.4")
    print("Voice -> Reasoning -> Bimanual Execution")
    print("=" * 72)

    # ------------------------------------------------------------
    # 1. Create MuJoCo simulation
    # ------------------------------------------------------------

    simulation = BimanualTableSimulation.create()

    print("\n[1/10] SIMULATION")
    print("Dual SO-101 MuJoCo environment created.")
    print(f"Arms: {simulation.arms}")

    # ------------------------------------------------------------
    # 2. Create an intentionally incomplete scene
    # ------------------------------------------------------------

    perturb_plate(
        simulation,
        "plate_1",
        0.07,
    )

    perturb_plate(
        simulation,
        "plate_2",
        0.07,
    )

    print("\n[2/10] SCENE SETUP")
    print("Created an incomplete table-setting state.")
    print("plate_1 and plate_2 require placement.")
    print("glasses remain correctly positioned.")

    # ------------------------------------------------------------
    # 3. Build structured scene state
    # ------------------------------------------------------------

    state_builder = SceneStateBuilder(
        simulation.model,
        simulation.data,
    )

    scene_state = state_builder.build()

    print("\n[3/10] SCENE UNDERSTANDING")
    print(
        f"Detected objects: "
        f"{len(scene_state.objects.objects)}"
    )

    for obj in scene_state.objects.objects:
        position = tuple(
            round(value, 3)
            for value in obj.position
        )

        print(
            f"  {obj.object_id:<10}"
            f"{obj.object_type:<8}"
            f"position={position}"
        )

    # ------------------------------------------------------------
    # 4. Create TABLEMIND voice/planning pipeline
    # ------------------------------------------------------------

    voice_planner = VoicePlannerPipeline()
    bridge = ReasoningExecutionBridge()

    grasping = GraspManager(
        simulation,
    )

    executor = SynchronizedBimanualExecutor(
        simulation=simulation,
        grasping=grasping,
    )

    # ------------------------------------------------------------
    # 5. Create Speechmatics transcriber
    # ------------------------------------------------------------

    transcriber = SpeechmaticsTranscriber()

    print("\n[4/10] SPEECHMATICS")
    print("Speechmatics voice interface initialized.")

    # ------------------------------------------------------------
    # 6. Handle completed voice commands
    # ------------------------------------------------------------

    async def handle_utterance(transcript: str) -> None:
        print()
        print("=" * 72)
        print("[5/10] VOICE COMMAND RECEIVED")
        print("=" * 72)
        print(f'Human: "{transcript}"')

        try:
            # ----------------------------------------------------
            # Voice -> NLU -> Planner
            # ----------------------------------------------------

            request, plan = voice_planner.process(
                transcript,
                scene_state.objects,
            )

            print("\n[6/10] VOICE -> REASONING")
            print(f"Intent:    {request.intent}")
            print(f"Objects:   {request.object_types}")
            print(f"Quantity:  {request.quantity}")
            print(f"Command:   {request.raw_instruction}")

            print("\n[7/10] TASK PLANNING")
            print(f"Task ID:   {plan.task.task_id}")
            print(f"Actions:   {plan.action_count}")

            for action in plan.actions:
                target = None

                if action.target_position is not None:
                    target = tuple(
                        round(value, 3)
                        for value in action.target_position
                    )

                print(
                    f"  {action.action_type.value:<14}"
                    f"arm={action.arm:<6}"
                    f"object={action.object_id:<10}"
                    f"target={target}"
                )

            # ----------------------------------------------------
            # Planner -> Bimanual execution task
            # ----------------------------------------------------

            bridge_result = bridge.build_task(
                plan,
            )

            coordinated_task = bridge_result.task

            print("\n[8/10] REASONING -> EXECUTION")
            print(
                f"Bimanual assignments: "
                f"{bridge_result.assignment_count}"
            )

            for assignment in coordinated_task.assignments:
                print(
                    f"\n  {assignment.arm.upper()} ARM"
                    f" -> {assignment.object_id}"
                )

                print(
                    f"    approach = "
                    f"{assignment.approach_position}"
                )

                print(
                    f"    grasp    = "
                    f"{assignment.grasp_position}"
                )

                print(
                    f"    lift     = "
                    f"{assignment.lift_position}"
                )

                print(
                    f"    place    = "
                    f"{assignment.place_position}"
                )

            # ----------------------------------------------------
            # Actual synchronized robot execution
            # ----------------------------------------------------

            print("\n[9/10] SYNCHRONIZED BIMANUAL EXECUTION")
            print("Both SO-101 arms executing the voice command...")

            result = executor.execute(
                coordinated_task,
            )

            print("\nExecution result:")
            print(f"Success: {result.success}")

            print(
                f"Stages completed: "
                f"{result.completed_stages}/"
                f"{result.total_stages}"
            )

            print(f"Message: {result.message}")

            # ----------------------------------------------------
            # Final result
            # ----------------------------------------------------

            print("\n[10/10] TABLEMIND RESULT")
            print("=" * 72)

            if result.success:
                print("VOICE-CONTROLLED TABLEMIND EXECUTION SUCCESSFUL")
                print()
                print(
                    "Speechmatics"
                    " -> NLU"
                    " -> Context reasoning"
                    " -> Task planning"
                    " -> Execution bridge"
                    " -> Bimanual manipulation"
                )
            else:
                print("VOICE-CONTROLLED EXECUTION FAILED")

            print("=" * 72)

        except Exception as exc:
            print()
            print("[EXECUTION ERROR]")
            print(f"{type(exc).__name__}: {exc}")
            print()

    # ------------------------------------------------------------
    # 7. Create utterance coordinator
    # ------------------------------------------------------------

    coordinator = UtteranceCoordinator(
        silence_timeout=SILENCE_TIMEOUT,
        on_utterance=handle_utterance,
    )

    # ------------------------------------------------------------
    # 8. Open microphone
    # ------------------------------------------------------------

    audio = pyaudio.PyAudio()

    print("\nOpening microphone...")
    print(f"Device index: {DEVICE_INDEX}")

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
    print("=" * 72)
    print("READY FOR VOICE COMMAND")
    print("=" * 72)
    print()
    print('Say: "Set the table for two."')
    print()
    print("Pause briefly after speaking.")
    print("Press Ctrl+C after the execution completes to stop.")
    print()

    # ------------------------------------------------------------
    # 9. Audio source
    # ------------------------------------------------------------

    async def audio_source(chunk_size: int) -> bytes:
        return await asyncio.to_thread(
            stream.read,
            chunk_size,
            exception_on_overflow=False,
        )

    def handle_partial(text: str) -> None:
        coordinator.add_partial(text)

        print(
            f"\r[partial] {text}",
            end="",
            flush=True,
        )

    def handle_final(text: str) -> None:
        print(
            f"\n[final]   {text}"
        )

        coordinator.add_final(text)

    # ------------------------------------------------------------
    # 10. Start live Speechmatics transcription
    # ------------------------------------------------------------

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
        print("\n\nTABLEMIND voice execution stopped.")