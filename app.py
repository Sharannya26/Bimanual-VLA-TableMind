from __future__ import annotations

import contextlib
import io
import runpy
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import streamlit as st

from tablemind.conversation.streamlit_voice import capture_voice_command
from tablemind.integration.closed_loop import ClosedLoopVLA
from tablemind.perception.camera import TableCamera
from tablemind.simulation.runtime import BimanualTableSimulation


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TABLEMIND | Physical AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parent


# ============================================================
# DEMO SCRIPTS
# ============================================================

BIDIRECTIONAL_SCRIPT = (
    ROOT / "scripts" / "demo_bidirectional_goal.py"
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL APP
       ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 15% 0%,
                rgba(120, 70, 255, 0.13),
                transparent 32%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(0, 220, 255, 0.07),
                transparent 30%
            ),
            #05070b;
        color: #e8ecf5;
    }

    [data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0);
    }

    [data-testid="stToolbar"] {
        visibility: hidden;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }


    /* ========================================================
       TYPOGRAPHY
       ======================================================== */

    h1 {
        font-family:
            Inter,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif !important;

        font-size: 4.6rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.065em !important;
        line-height: 0.95 !important;

        background:
            linear-gradient(
                100deg,
                #f5f7fb 0%,
                #f5f7fb 42%,
                #a970ff 65%,
                #d5a9ff 100%
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    h2 {
        color: #f2f4fa !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
    }

    h3 {
        color: #eef1f7 !important;
        font-weight: 650 !important;
    }

    p {
        color: #8d98aa;
    }


    /* ========================================================
       SECTION LABELS
       ======================================================== */

    .section-label {
        color: #727e94;

        font-family:
            "JetBrains Mono",
            monospace;

        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;

        margin-top: 2.3rem;
        margin-bottom: 0.85rem;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero-kicker {
        color: #b990ff;

        font-family:
            "JetBrains Mono",
            monospace;

        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.2em;
        text-transform: uppercase;

        margin-bottom: 0.8rem;
    }

    .hero-description {
        max-width: 850px;
        color: #8995a9;
        font-size: 1.08rem;
        line-height: 1.8;
        margin-top: 1.1rem;
    }

    .hero-pill {
        display: inline-block;

        margin-top: 1.4rem;
        padding: 0.45rem 0.8rem;

        border:
            1px solid
            rgba(169, 112, 255, 0.28);

        border-radius: 999px;

        color: #b9a1dc;

        background:
            rgba(169, 112, 255, 0.06);

        font-family:
            "JetBrains Mono",
            monospace;

        font-size: 0.65rem;
        letter-spacing: 0.12em;
    }


    /* ========================================================
       ONLINE STATUS
       ======================================================== */

    .online-status {
        text-align: right;

        color: #48ef9b;

        font-family:
            "JetBrains Mono",
            monospace;

        font-size: 0.7rem;
        letter-spacing: 0.12em;

        padding-top: 0.7rem;
    }


    /* ========================================================
       CARDS
       ======================================================== */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background:
            linear-gradient(
                145deg,
                rgba(16, 20, 29, 0.96),
                rgba(7, 10, 16, 0.96)
            );

        border:
            1px solid
            rgba(112, 125, 151, 0.16) !important;

        border-radius: 16px !important;

        box-shadow:
            0 15px 45px rgba(0, 0, 0, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.015);
    }


    /* ========================================================
       TEXT INPUT
       ======================================================== */

    [data-testid="stTextInput"] label {
        color: #cbd2df !important;
        font-weight: 650 !important;
    }

    [data-testid="stTextInput"] input {
        background: #0b0f17 !important;
        color: #edf0f7 !important;

        border:
            1px solid
            rgba(169, 112, 255, 0.22) !important;

        border-radius: 10px !important;
        padding: 0.85rem 1rem !important;
    }

    [data-testid="stTextInput"] input:focus {
        border-color: #a970ff !important;

        box-shadow:
            0 0 0 1px rgba(169, 112, 255, 0.35),
            0 0 22px rgba(169, 112, 255, 0.08) !important;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        width: 100%;
        min-height: 46px;

        background:
            linear-gradient(
                135deg,
                rgba(169, 112, 255, 0.12),
                rgba(90, 55, 150, 0.08)
            ) !important;

        color: #e9e3f5 !important;

        border:
            1px solid
            rgba(169, 112, 255, 0.32) !important;

        border-radius: 10px !important;

        font-family:
            "JetBrains Mono",
            monospace !important;

        font-size: 0.72rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.07em !important;

        transition:
            border-color 0.18s ease,
            background 0.18s ease,
            transform 0.18s ease;
    }

    .stButton > button:hover {
        background:
            linear-gradient(
                135deg,
                rgba(169, 112, 255, 0.23),
                rgba(90, 55, 150, 0.14)
            ) !important;

        border-color: #a970ff !important;

        transform: translateY(-1px);

        box-shadow:
            0 8px 28px rgba(169, 112, 255, 0.10);
    }


    /* ========================================================
       PIPELINE
       ======================================================== */

    .pipeline-number {
        color: #a970ff;

        font-family:
            "JetBrains Mono",
            monospace;

        font-size: 0.62rem;
        letter-spacing: 0.12em;
    }

    .pipeline-icon {
        color: #d3b9ff;
        font-size: 1.5rem;
        margin-top: 0.35rem;
    }

    .pipeline-name {
        color: #edf0f6;
        font-weight: 700;
        font-size: 0.88rem;
        margin-top: 0.2rem;
    }

    .pipeline-description {
        color: #697487;

        font-family:
            "JetBrains Mono",
            monospace;

        font-size: 0.61rem;
        margin-top: 0.35rem;
    }


    /* ========================================================
       METRICS
       ======================================================== */

    [data-testid="stMetric"] {
        background: rgba(8, 11, 17, 0.72);

        border:
            1px solid
            rgba(110, 124, 151, 0.14);

        border-radius: 12px;
        padding: 1rem;
    }

    [data-testid="stMetricLabel"] {
        color: #69758a !important;

        font-family:
            "JetBrains Mono",
            monospace;

        font-size: 0.62rem !important;
        letter-spacing: 0.08em;
    }

    [data-testid="stMetricValue"] {
        color: #d9dff0 !important;
    }


    /* ========================================================
       STATUS
       ======================================================== */

    .status-ready {
        color: #48ef9b;

        font-family:
            "JetBrains Mono",
            monospace;

        font-size: 0.68rem;
        letter-spacing: 0.08em;

        padding: 0.7rem 0.85rem;

        background:
            rgba(56, 242, 154, 0.045);

        border:
            1px solid
            rgba(56, 242, 154, 0.18);

        border-radius: 9px;
    }


    /* ========================================================
       COMMAND DISPLAY
       ======================================================== */

    .command-display {
        color: #ddd7e8;

        font-family:
            "JetBrains Mono",
            monospace;

        font-size: 0.84rem;
        line-height: 1.7;

        padding: 1rem;

        background: #080b11;

        border:
            1px solid
            rgba(169, 112, 255, 0.16);

        border-radius: 10px;
    }


    /* ========================================================
       VOICE DISPLAY
       ======================================================== */

    .voice-display {
        color: #d9d1ea;

        font-family:
            "JetBrains Mono",
            monospace;

        font-size: 0.78rem;
        line-height: 1.7;

        padding: 0.9rem 1rem;

        background:
            linear-gradient(
                135deg,
                rgba(169, 112, 255, 0.08),
                rgba(0, 229, 255, 0.025)
            );

        border:
            1px solid
            rgba(169, 112, 255, 0.20);

        border-radius: 10px;

        margin-top: 0.8rem;
    }


    /* ========================================================
       EXECUTION MODE
       ======================================================== */

    .execution-banner {
        padding: 1.4rem 1.5rem;

        background:
            linear-gradient(
                135deg,
                rgba(169, 112, 255, 0.12),
                rgba(0, 229, 255, 0.035)
            );

        border:
            1px solid
            rgba(169, 112, 255, 0.28);

        border-radius: 14px;

        margin-bottom: 1rem;
    }

    .execution-label {
        color: #a970ff;

        font-family:
            "JetBrains Mono",
            monospace;

        font-size: 0.68rem;
        letter-spacing: 0.14em;
    }

    .execution-title {
        color: #f2f3f8;
        font-size: 1.55rem;
        font-weight: 750;
        margin-top: 0.3rem;
    }


    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {
        border-color:
            rgba(112, 125, 151, 0.12) !important;

        margin-top: 2rem !important;
        margin-bottom: 2rem !important;
    }


    /* ========================================================
       CODE OUTPUT
       ======================================================== */

    .stCode {
        border-radius: 12px !important;
    }

    pre {
        background: #05070b !important;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;

        color: #4f5b6f;

        font-family:
            "JetBrains Mono",
            monospace;

        font-size: 0.62rem;
        letter-spacing: 0.08em;

        margin-top: 3rem;
        padding-top: 1.5rem;

        border-top:
            1px solid
            rgba(112, 125, 151, 0.10);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DEMO EXECUTION HELPER
# ============================================================

def run_script(script_name: str) -> tuple[int, str, str]:
    """
    Run one TABLEMIND demo script.

    Returns:
        return_code
        stdout
        stderr
    """

    script_path = ROOT / "scripts" / script_name

    if not script_path.exists():
        return (
            -1,
            "",
            f"Script not found:\n{script_path}",
        )

    try:
        result = subprocess.run(
            [
                sys.executable,
                str(script_path),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        return (
            result.returncode,
            result.stdout,
            result.stderr,
        )

    except Exception as exc:
        return (
            -1,
            "",
            f"{type(exc).__name__}: {exc}",
        )


# ============================================================
# DEMO RESULT DISPLAY
# ============================================================

def show_demo_result(
    return_code: int,
    stdout: str,
    stderr: str,
) -> None:

    if stdout.strip():
        st.code(
            stdout,
            language="text",
        )

    if stderr.strip():
        st.warning(
            "Diagnostic output was produced:"
        )

        st.code(
            stderr,
            language="text",
        )

    if return_code == 0:
        st.success(
            "TABLEMIND verified the physical outcome successfully."
        )
    else:
        st.error(
            f"Demo exited with return code {return_code}."
        )


# ============================================================
# UI-ONLY PRESENTATION CAMERA
# ============================================================

class PresentationCamera:
    """
    UI-only MuJoCo presentation camera.

    This camera is deliberately separate from TABLEMIND's
    calibrated table_overview perception camera.

    Vision / YOLO / OpenVINO continues using the original
    calibrated perception camera.
    """

    def __init__(
        self,
        model,
        data,
        width=640,
        height=480,
    ):
        self.model = model
        self.data = data

        self.renderer = mujoco.Renderer(
            model,
            height=height,
            width=width,
        )

        self.camera = mujoco.MjvCamera()

        mujoco.mjv_defaultCamera(
            self.camera
        )

        self.camera.type = (
            mujoco.mjtCamera.mjCAMERA_FREE
        )

        self.camera.lookat[:] = np.array(
            [0.0, -0.05, 0.65],
            dtype=np.float64,
        )

        self.camera.distance = 2.45
        self.camera.azimuth = 180.0
        self.camera.elevation = -22.0

    def capture(self):
        self.renderer.update_scene(
            self.data,
            camera=self.camera,
        )

        image = self.renderer.render()

        return np.asarray(image).copy()

    def close(self):
        self.renderer.close()


# ============================================================
# LIVE 9.5 BIDIRECTIONAL GOAL DEMO
# ============================================================

def run_live_bidirectional_goal_demo() -> tuple[
    bool,
    object | None,
    object | None,
    str,
    str,
    str,
]:
    """
    Run the proven Milestone 9.5 demo inside the Streamlit
    process while capturing the exact MuJoCo simulation.

    The 9.5 robotics implementation itself is unchanged.
    """

    simulation_holder = {
        "simulation": None,
        "before_frame": None,
    }

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    original_create_descriptor = (
        BimanualTableSimulation.__dict__["create"]
    )

    original_create = (
        BimanualTableSimulation.create
    )

    @classmethod
    def captured_create(cls):
        simulation = original_create()

        simulation_holder["simulation"] = simulation

        camera = None

        try:
            camera = PresentationCamera(
                simulation.model,
                simulation.data,
                width=640,
                height=480,
            )

            simulation_holder["before_frame"] = (
                camera.capture()
            )

        finally:
            if camera is not None:
                camera.close()

        return simulation

    BimanualTableSimulation.create = captured_create

    try:
        with contextlib.redirect_stdout(
            stdout_buffer
        ):
            with contextlib.redirect_stderr(
                stderr_buffer
            ):
                runpy.run_path(
                    str(BIDIRECTIONAL_SCRIPT),
                    run_name="__main__",
                )

        stdout = stdout_buffer.getvalue()
        stderr = stderr_buffer.getvalue()

        simulation = simulation_holder["simulation"]

        success = (
            "TABLEMIND 9.5 BIDIRECTIONAL GOAL DEMO SUCCESSFUL"
            in stdout
        )

        before_frame = (
            simulation_holder["before_frame"]
        )

        after_frame = None

        if simulation is not None:

            camera = None

            try:
                camera = PresentationCamera(
                    simulation.model,
                    simulation.data,
                    width=640,
                    height=480,
                )

                after_frame = camera.capture()

            finally:
                if camera is not None:
                    camera.close()

        if success:

            message = (
                "1 → 2 → 1 VERIFIED — the proven "
                "Milestone 9.5 goal-reconciliation demo "
                "ran inside the same MuJoCo process and "
                "the final physical state was rendered in "
                "the UI."
            )

        else:

            message = (
                "The Milestone 9.5 demo completed without "
                "its success marker. Open the diagnostic "
                "trace below."
            )

        return (
            success,
            before_frame,
            after_frame,
            message,
            stdout,
            stderr,
        )

    except Exception as exc:

        return (
            False,
            simulation_holder["before_frame"],
            None,
            f"{type(exc).__name__}: {exc}",
            stdout_buffer.getvalue(),
            stderr_buffer.getvalue(),
        )

    finally:
        BimanualTableSimulation.create = (
            original_create_descriptor
        )


# ============================================================
# LIVE CLOSED-LOOP MUJOCO DEMO
# ============================================================

def run_live_mujoco_demo(
    instruction: str,
) -> tuple[
    bool,
    object | None,
    object | None,
    str,
]:
    """
    Execute TABLEMIND directly through the real MuJoCo
    simulation instead of launching a demo subprocess.

    Returns:
        success
        before_frame
        after_frame
        message
    """

    simulation = None
    camera = None
    vision = None

    try:

        simulation = (
            BimanualTableSimulation.create()
        )

        camera = PresentationCamera(
            simulation.model,
            simulation.data,
            width=640,
            height=480,
        )

        before_frame = camera.capture()

        vision = ClosedLoopVLA.create(
            simulation
        )

        result = vision.run(
            instruction
        )

        simulation.step(1)

        after_frame = camera.capture()

        success = bool(
            getattr(
                result,
                "success",
                False,
            )
        )

        if success:

            message = (
                "TABLEMIND successfully completed "
                "the physical task and verified the "
                "MuJoCo world."
            )

        else:

            message = (
                "TABLEMIND executed the task, but "
                "final verification reported failure."
            )

        return (
            success,
            before_frame,
            after_frame,
            message,
        )

    except Exception as exc:

        return (
            False,
            None,
            None,
            f"{type(exc).__name__}: {exc}",
        )

    finally:

        if vision is not None:
            try:
                vision.close()
            except Exception:
                pass

        if camera is not None:
            try:
                camera.close()
            except Exception:
                pass


# ============================================================
# SESSION STATE
# ============================================================

if "mode" not in st.session_state:
    st.session_state.mode = "command"

if "instruction" not in st.session_state:
    st.session_state.instruction = (
        "Set the table for two."
    )

if "voice_transcript" not in st.session_state:
    st.session_state.voice_transcript = ""


# ============================================================
# EXECUTION MODE
# ============================================================

if st.session_state.mode == "execution":

    st.markdown(
        "### TABLEMIND / LIVE MUJOCO EXECUTION"
    )

    st.caption(
        "Physical AI pipeline active"
    )

    instruction = (
        st.session_state.instruction
    )

    left_column, right_column = st.columns(
        [3, 1]
    )

    with left_column:

        st.subheader(
            "ACTIVE INSTRUCTION"
        )

        st.code(
            instruction,
            language="text",
        )

    with right_column:

        if st.button(
            "← RETURN TO COMMAND CENTER",
            key="execution_return",
        ):
            st.session_state.mode = "command"
            st.rerun()

    st.divider()

    st.subheader(
        "MUJOCO / BIMANUAL SO-101"
    )

    st.caption(
        "The following execution uses the actual "
        "MuJoCo simulation, not a mock visualization."
    )

    status_placeholder = st.empty()

    status_placeholder.info(
        "Initializing dual SO-101 simulation..."
    )

    (
        success,
        before_frame,
        after_frame,
        message,
    ) = run_live_mujoco_demo(
        instruction
    )

    status_placeholder.empty()

    if before_frame is not None:

        st.markdown(
            "### INITIAL PHYSICAL STATE"
        )

        st.image(
            before_frame,
            use_container_width=True,
        )

    if after_frame is not None:

        st.markdown(
            "### FINAL PHYSICAL STATE"
        )

        st.image(
            after_frame,
            use_container_width=True,
        )

    st.divider()

    if success:

        st.success(
            f"✓ {message}"
        )

    else:

        st.error(
            f"✗ {message}"
        )

    st.divider()

    st.subheader(
        "COGNITIVE PIPELINE"
    )

    execution_columns = st.columns(
        6,
        gap="small",
    )

    execution_steps = [
        (
            "01",
            "LANGUAGE",
            "Natural instruction",
        ),
        (
            "02",
            "VISION",
            "Scene grounding",
        ),
        (
            "03",
            "REASONING",
            "Goal understanding",
        ),
        (
            "04",
            "PLANNING",
            "Task decomposition",
        ),
        (
            "05",
            "BIMANUAL",
            "Dual-arm coordination",
        ),
        (
            "06",
            "VERIFY",
            "Physical validation",
        ),
    ]

    for column, step in zip(
        execution_columns,
        execution_steps,
    ):

        number, name, description = step

        with column:

            with st.container(
                border=True
            ):

                st.caption(number)

                st.markdown(
                    f"**{name}**"
                )

                st.caption(
                    description
                )

    if st.button(
        "← BACK TO TABLEMIND",
        key="execution_back",
    ):

        st.session_state.mode = "command"

        st.rerun()

    st.markdown(
        """
        <div class="footer">
            TABLEMIND / PHYSICAL AI ·
            INTEL AI INFRA SUMMIT 2026 ·
            LIVE MUJOCO BIMANUAL VLA
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# COMMAND CENTER
# ============================================================

header_left, header_right = st.columns(
    [5, 1]
)


# ============================================================
# HERO LEFT
# ============================================================

with header_left:

    st.markdown(
        """
        <div class="hero-kicker">
            PHYSICAL AI / BIMANUAL VLA
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.title(
        "TABLEMIND"
    )

    st.markdown(
        """
        <div class="hero-description">
            A self-correcting conversational Physical AI
            agent that grounds language in the physical world,
            coordinates two robotic manipulators,
            and verifies the outcome.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero-pill">
            MUJOCO / SO-101 / YOLO26n / OPENVINO
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# ONLINE STATUS
# ============================================================

with header_right:

    st.markdown(
        """
        <div class="online-status">
            ● SYSTEM ONLINE
        </div>
        """,
        unsafe_allow_html=True,
    )


st.divider()


# ============================================================
# 01 COMMAND CENTER
# ============================================================

st.markdown(
    """
    <div class="section-label">
        01 / COMMAND CENTER
    </div>
    """,
    unsafe_allow_html=True,
)

command_left, command_right = st.columns(
    [1.45, 1],
    gap="large",
)


# ============================================================
# COMMAND INPUT
# ============================================================

with command_left:

    with st.container(
        border=True
    ):

        st.subheader(
            "Natural-language command"
        )

        st.caption(
            "Tell TABLEMIND what should happen "
            "in the physical world."
        )

        instruction = st.text_input(
            "Instruction",
            value=(
                st.session_state.instruction
            ),
            label_visibility="collapsed",
            placeholder="Set the table for two.",
            key="instruction_input",
        )

        st.session_state.instruction = (
            instruction
        )

        # ----------------------------------------------------
        # COMMAND BUTTONS
        # ----------------------------------------------------

        execute_column, voice_column, goal_column = (
            st.columns(
                3,
                gap="small",
            )
        )

        # ----------------------------------------------------
        # CLOSED-LOOP BUTTON
        # ----------------------------------------------------

        with execute_column:

            if st.button(
                "▶  EXECUTE CLOSED-LOOP VLA",
                type="primary",
                key="execute_closed_loop",
            ):

                st.session_state.instruction = (
                    instruction
                )

                st.session_state.mode = (
                    "execution"
                )

                st.rerun()

        # ----------------------------------------------------
        # SPEECHMATICS VOICE BUTTON
        # ----------------------------------------------------

        with voice_column:

            if st.button(
                "🎤  SPEAK COMMAND",
                key="speak_command",
            ):

                with st.status(
                    "Listening with Speechmatics...",
                    expanded=True,
                ) as status:

                    st.write(
                        "◉ Microphone active"
                    )

                    st.write(
                        "◉ Waiting for a complete command..."
                    )

                    voice_result = (
                        capture_voice_command()
                    )

                    if voice_result.success:

                        spoken_instruction = (
                            voice_result.transcript.strip()
                        )

                        st.session_state.voice_transcript = (
                            spoken_instruction
                        )

                        st.session_state.instruction = (
                            spoken_instruction
                        )

                        status.update(
                            label="Voice command captured",
                            state="complete",
                        )

                        st.success(
                            f'Heard: "{spoken_instruction}"'
                        )

                        st.session_state.mode = (
                            "execution"
                        )

                        st.rerun()

                    else:

                        status.update(
                            label="Voice capture failed",
                            state="error",
                        )

                        st.error(
                            voice_result.error
                            or (
                                "Speechmatics did not "
                                "return a command."
                            )
                        )

        # ----------------------------------------------------
        # GOAL DEMO BUTTON
        # ----------------------------------------------------

        with goal_column:

            if st.button(
                "◆  RUN 1 → 2 → 1",
                key="command_center_goal_demo",
            ):

                with st.status(
                    "Running bidirectional goal reconciliation in MuJoCo...",
                    expanded=True,
                ) as status:

                    st.write(
                        "◉ Initial physical goal → 1 guest"
                    )

                    st.write(
                        "◉ User modification → 2 guests"
                    )

                    st.write(
                        "◉ Physical addition → plate + glass"
                    )

                    st.write(
                        "◉ Verification → 2 complete place settings"
                    )

                    st.write(
                        "◉ User modification → 1 guest"
                    )

                    st.write(
                        "◉ Physical removal → setting #2"
                    )

                    st.write(
                        "◉ Final verification → 1 complete place setting"
                    )

                    (
                        success,
                        before_frame,
                        after_frame,
                        message,
                        stdout,
                        stderr,
                    ) = (
                        run_live_bidirectional_goal_demo()
                    )

                    status.update(
                        label=(
                            "Goal reconciliation verified"
                            if success
                            else "Goal reconciliation failed"
                        ),
                        state=(
                            "complete"
                            if success
                            else "error"
                        ),
                    )

                st.divider()

                if before_frame is not None:

                    st.markdown(
                        "### INITIAL PHYSICAL STATE"
                    )

                    st.image(
                        before_frame,
                        use_container_width=True,
                    )

                if after_frame is not None:

                    st.markdown(
                        "### FINAL PHYSICAL STATE"
                    )

                    st.image(
                        after_frame,
                        use_container_width=True,
                    )

                if success:

                    st.success(
                        message
                    )

                else:

                    st.error(
                        message
                    )

                if stderr.strip():

                    with st.expander(
                        "Diagnostics"
                    ):

                        st.code(
                            stderr,
                            language="text",
                        )

                with st.expander(
                    "9.5 goal-reconciliation trace"
                ):

                    st.code(
                        stdout,
                        language="text",
                    )

        # ----------------------------------------------------
        # LAST VOICE TRANSCRIPT
        # ----------------------------------------------------

        if st.session_state.voice_transcript:

            st.markdown(
                """
                <div class="voice-display">
                    🎤 LAST VOICE COMMAND
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                st.session_state.voice_transcript
            )


# ============================================================
# CURRENT COMMAND
# ============================================================

with command_right:

    with st.container(
        border=True
    ):

        st.caption(
            "CURRENT COMMAND"
        )

        st.markdown(
            f"""
            <div class="command-display">
                "{instruction}"
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        st.markdown(
            """
            <div class="status-ready">
                ● READY FOR EXECUTION
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.voice_transcript:

            st.write("")

            st.caption(
                "SOURCE: SPEECHMATICS"
            )


# ============================================================
# 02 COGNITIVE PIPELINE
# ============================================================

st.markdown(
    """
    <div class="section-label">
        02 / COGNITIVE PIPELINE
    </div>
    """,
    unsafe_allow_html=True,
)

pipeline_columns = st.columns(
    6,
    gap="small",
)

pipeline_steps = [
    (
        "01",
        "◌",
        "LANGUAGE",
        "Natural instruction",
    ),
    (
        "02",
        "◉",
        "VISION",
        "Scene grounding",
    ),
    (
        "03",
        "✦",
        "REASONING",
        "Goal understanding",
    ),
    (
        "04",
        "☷",
        "PLANNING",
        "Task decomposition",
    ),
    (
        "05",
        "♧",
        "BIMANUAL",
        "Dual-arm coordination",
    ),
    (
        "06",
        "✓",
        "VERIFY",
        "Physical validation",
    ),
]

for column, step in zip(
    pipeline_columns,
    pipeline_steps,
):

    number, icon, name, description = step

    with column:

        with st.container(
            border=True
        ):

            st.markdown(
                f"""
                <div class="pipeline-number">
                    {number}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="pipeline-icon">
                    {icon}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="pipeline-name">
                    {name}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="pipeline-description">
                    {description}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# 03 SYSTEM CAPABILITIES
# ============================================================

st.markdown(
    """
    <div class="section-label">
        03 / SYSTEM CAPABILITIES
    </div>
    """,
    unsafe_allow_html=True,
)

metric_columns = st.columns(
    4,
    gap="medium",
)

with metric_columns[0]:

    st.metric(
        label="VISION MODEL",
        value="YOLO26n",
    )

with metric_columns[1]:

    st.metric(
        label="INTEL INFERENCE",
        value="OpenVINO",
    )

with metric_columns[2]:

    st.metric(
        label="MANIPULATORS",
        value="2 × SO-101",
    )

with metric_columns[3]:

    st.metric(
        label="GOAL RECONCILIATION",
        value="1 → 2 → 1",
    )


# ============================================================
# 04 CONVERSATIONAL GOAL RECONCILIATION
# ============================================================

st.markdown(
    """
    <div class="section-label">
        04 / CONVERSATIONAL GOAL RECONCILIATION
    </div>
    """,
    unsafe_allow_html=True,
)

goal_left, goal_right = st.columns(
    [1.15, 0.85],
    gap="large",
)


# ============================================================
# GOAL STATE
# ============================================================

with goal_left:

    with st.container(
        border=True
    ):

        st.subheader(
            "The robot maintains a physical goal state"
        )

        st.caption(
            "The user can change their mind "
            "without restarting the physical world."
        )

        goal_columns = st.columns(
            5
        )

        with goal_columns[0]:

            st.markdown(
                "### 1"
            )

            st.caption(
                "GUEST"
            )

        with goal_columns[1]:

            st.markdown(
                "### →"
            )

        with goal_columns[2]:

            st.markdown(
                "### 2"
            )

            st.caption(
                "GUESTS"
            )

        with goal_columns[3]:

            st.markdown(
                "### →"
            )

        with goal_columns[4]:

            st.markdown(
                "### 1"
            )

            st.caption(
                "GUEST"
            )

        st.caption(
            "Goal reconciliation = additions + "
            "physical removals + verification."
        )


# ============================================================
# GOAL DEMO CARD
# ============================================================

with goal_right:

    with st.container(
        border=True
    ):

        st.subheader(
            "Run the full conversational demo"
        )

        st.caption(
            "Proven TABLEMIND 9.5 demonstration."
        )

        st.markdown(
            """
            **1 guest**

            `↓` Actually, make it 2.

            **2 guests**

            `↓` Actually, one guest isn't coming.

            **1 guest**
            """
        )

        if st.button(
            "▶  RUN 1 → 2 → 1 DEMO",
            key="goal_reconciliation_demo",
        ):

            with st.status(
                "Running conversational goal reconciliation in MuJoCo...",
                expanded=True,
            ) as status:

                st.write(
                    "◉ Initial physical goal → 1 guest"
                )

                st.write(
                    "◉ User modification → 2 guests"
                )

                st.write(
                    "◉ Physical addition → plate + glass"
                )

                st.write(
                    "◉ Verification → 2 complete place settings"
                )

                st.write(
                    "◉ User modification → 1 guest"
                )

                st.write(
                    "◉ Physical removal → setting #2"
                )

                st.write(
                    "◉ Final verification → 1 complete place setting"
                )

                (
                    success,
                    before_frame,
                    after_frame,
                    message,
                    stdout,
                    stderr,
                ) = (
                    run_live_bidirectional_goal_demo()
                )

                status.update(
                    label=(
                        "Goal reconciliation verified"
                        if success
                        else "Goal reconciliation failed"
                    ),
                    state=(
                        "complete"
                        if success
                        else "error"
                    ),
                )

            st.divider()

            if after_frame is not None:

                st.markdown(
                    "### FINAL PHYSICAL STATE"
                )

                st.image(
                    after_frame,
                    use_container_width=True,
                )

            if success:

                st.success(
                    message
                )

            else:

                st.error(
                    message
                )

            if stderr.strip():

                with st.expander(
                    "Diagnostics"
                ):

                    st.code(
                        stderr,
                        language="text",
                    )

            with st.expander(
                "9.5 goal-reconciliation trace"
            ):

                st.code(
                    stdout,
                    language="text",
                )


# ============================================================
# 05 SYSTEM ARCHITECTURE
# ============================================================

st.markdown(
    """
    <div class="section-label">
        05 / SYSTEM ARCHITECTURE
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(
    border=True
):

    architecture_columns = st.columns(
        7
    )

    architecture = [
        "HUMAN",
        "LANGUAGE",
        "VISION",
        "REASONING",
        "PLANNING",
        "BIMANUAL",
        "VERIFICATION",
    ]

    for index, name in enumerate(
        architecture
    ):

        with architecture_columns[index]:

            st.markdown(
                f"**{name}**"
            )

            if index < len(architecture) - 1:

                st.caption(
                    "→"
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        TABLEMIND / PHYSICAL AI ·
        INTEL AI INFRA SUMMIT 2026 ·
        CLOSED-LOOP BIMANUAL VLA
    </div>
    """,
    unsafe_allow_html=True,
)