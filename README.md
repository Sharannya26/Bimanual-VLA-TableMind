# TABLEMIND

## Self-Correcting Conversational Bimanual VLA for Physical AI

**TABLEMIND** is an end-to-end Physical AI system that combines natural-language interaction, visual scene understanding, multi-modal reasoning, bimanual robotic manipulation, physical verification, and conversational goal modification.

It uses **two simulated SO-101 robotic arms in MuJoCo** to understand table-setting instructions, reason over the physical scene, coordinate both manipulators, execute multi-step actions, verify the result, and adapt when the user's goal changes.

> **Understand the human. See the world. Act with both hands. Verify the result. Adapt when the goal changes.**

---

## 🚀 Core Idea

Traditional robotic demonstrations often follow a fixed sequence:

```text
Instruction
    ↓
Plan
    ↓
Execute
    ↓
Stop
````

TABLEMIND instead operates as a **closed-loop conversational Physical AI agent**:

```text
Human Speech / Text
        ↓
Natural-Language Understanding
        ↓
Goal & World-State Grounding
        ↓
Visual Scene Understanding
        ↓
Task Reasoning
        ↓
Bimanual Planning
        ↓
Dual SO-101 Execution
        ↓
MuJoCo Physical Simulation
        ↓
Visual Verification
        ↓
Success / Recovery / Replanning
        ↓
Updated Conversational Goal
```

The system does not simply execute a predefined animation.

It continuously connects:

**language → perception → reasoning → planning → action → verification → adaptation**

---

# 🤖 Bimanual Robotic Manipulation

TABLEMIND uses **two simulated SO-101 robotic arms** inside MuJoCo.

The two manipulators can coordinate their actions to perform multi-object table-setting tasks.

### Example

```text
"Set the table for two."
```

The system interprets the request as:

```text
Intent:      set_table
Objects:     plate + glass
Quantity:    2
```

The planner then decomposes the task into coordinated manipulation actions.

```text
              TABLEMIND
                  │
        ┌─────────┴─────────┐
        ↓                   ↓
   LEFT SO-101         RIGHT SO-101
        │                   │
      Plate                Plate
        │                   │
      Glass                Glass
        └─────────┬─────────┘
                  ↓
          Verified Table State
```

The two arms are coordinated through synchronized execution batches rather than being treated as independent robots.

---

# 👁️ Vision-Grounded Physical AI

TABLEMIND does not rely only on predefined object positions.

The perception pipeline observes the simulated table and constructs a structured scene representation.

The system uses a **custom YOLO26n object detector** trained using synthetic MuJoCo-native data.

### Detected objects

* Plates
* Glasses
* Other table-setting objects supported by the scene

### MuJoCo-native validation results

| Metric    | Validation |  Test |
| --------- | ---------: | ----: |
| Precision |      0.998 | 0.999 |
| Recall    |      0.999 | 0.999 |
| mAP@50    |      0.995 | 0.995 |
| mAP@50-95 |      0.955 | 0.964 |

The perception system converts camera observations into a structured world state containing object identity, position, and task-relevant information.

---

# ⚡ OpenVINO Optimization

TABLEMIND uses **OpenVINO** to optimize neural-network inference for Intel hardware.

The YOLO26n model is converted and executed through the OpenVINO runtime.

The system was validated locally on an:

**Intel Core i5-1135G7**

with OpenVINO runtime.

### Local inference result

```text
2 plates + 2 glasses
        ↓
OpenVINO CPU inference
        ↓
~36.4 ms
```

This demonstrates how Intel-optimized inference can be used even when a newer Intel Core Ultra Series 2/3 system is not available locally.

The architecture is designed to benefit further from newer Intel Core Ultra hardware.

---

# 🔄 Closed-Loop VLA

The core TABLEMIND pipeline is a **closed-loop Vision-Language-Action system**.

Instead of assuming that the robot successfully completed an action, TABLEMIND verifies the resulting physical state.

```text
Language
   ↓
Reasoning
   ↓
Planning
   ↓
Action
   ↓
Physical Simulation
   ↓
Camera Observation
   ↓
Scene Verification
   ↓
Goal State?
  / \
YES  NO
 |    |
 ↓    ↓
Done  Replan
```

This enables the system to reason about the difference between:

```text
Expected World State
        vs.
Actual World State
```

and respond accordingly.

---

# 🧠 Conversational Goal Reconciliation

One of TABLEMIND's key capabilities is that the user can **change the goal after execution has already begun**.

The system reconciles the new goal with the current physical state instead of simply restarting the entire task.

For example:

```text
User:
"Set the table for one."
```

The system creates and verifies one complete place setting.

Then the user says:

```text
"Actually, make it 2."
```

TABLEMIND determines:

```text
Current quantity : 1
Desired quantity : 2
Delta            : +1
```

It physically restores the missing place setting.

Then the user changes the goal again:

```text
"Actually, one guest isn't coming."
```

The system determines:

```text
Current quantity : 2
Desired quantity : 1
Delta            : -1
```

It physically removes the second place setting and verifies the resulting world state.

### Key Demonstration

```text
1 GUEST
   ↓
"Actually, make it 2."
   ↓
2 GUESTS
   ↓
"Actually, one guest isn't coming."
   ↓
1 GUEST
```

The complete **1 → 2 → 1** sequence was executed and verified in MuJoCo.

The important distinction is that TABLEMIND **reconciles changes in the user's goal with the current physical state**.

---

# 🎙️ Voice Interface

TABLEMIND supports conversational interaction through **Speechmatics**.

The voice pipeline follows:

```text
Microphone
    ↓
Speechmatics
    ↓
Speech Transcription
    ↓
TABLEMIND NLU
    ↓
Task Request
    ↓
Planning
    ↓
Robot Execution
```

Example:

```text
User speaks:

"Set the table for two."

        ↓

Speechmatics transcript:

"Set the table for two."

        ↓

TABLEMIND NLU:

Intent: set_table
Objects: plate, glass
Quantity: 2

        ↓

Bimanual execution
```

Speechmatics was also tested through the standalone microphone pipeline with successful transcription and NLU grounding.

---

# 🖥️ Interactive Demonstration

TABLEMIND includes a **Streamlit command-center interface**.

The interface provides:

* Natural-language task input
* Voice command capture
* Closed-loop VLA execution
* Live MuJoCo visualization
* Before/after scene observations
* Goal modification demonstration
* Execution status
* Verification results
* Cognitive pipeline visualization
* System architecture information

The interface is designed to make the Physical AI pipeline understandable during a live hackathon demonstration.

---

# 🏗️ System Architecture

```text
                         HUMAN
                           │
                    Speech / Text
                           │
                           ▼
                ┌─────────────────────┐
                │ Speechmatics / NLU  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Goal Understanding  │
                │ & World Grounding   │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Vision Pipeline     │
                │ YOLO26n + OpenVINO  │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Scene State         │
                │ Representation      │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Task Reasoning      │
                │ & Decomposition     │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Bimanual Planner    │
                └──────────┬──────────┘
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
          ┌─────────────┐     ┌─────────────┐
          │ LEFT SO-101 │     │ RIGHT SO-101│
          └──────┬──────┘     └──────┬──────┘
                 │                   │
                 └─────────┬─────────┘
                           ▼
                    ┌────────────┐
                    │  MuJoCo    │
                    │ Simulation │
                    └─────┬──────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Camera Feedback │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Verification    │
                 └────────┬────────┘
                          │
                   ┌──────┴──────┐
                   ▼             ▼
                 SUCCESS       MISMATCH
                   │             │
                   ▼             ▼
                  DONE       RECOVERY /
                             REPLANNING
```

---

# 📁 Project Structure

```text
Bimanual-VLA-TableMind/
│
├── app.py
├── README.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
│
├── assets/
│   └── mujoco/
│       └── so101/
│           ├── assets/
│           ├── so101_new_calib.xml
│           └── ...
│
├── artifacts/
│   └── camera_calibration_8_9_7.json
│
├── scripts/
│   ├── launch_sim.py
│   ├── demo_basic_manipulation.py
│   ├── demo_closed_loop.py
│   ├── demo_bidirectional_goal.py
│   ├── demo_voice_execution.py
│   ├── demo_speechmatics.py
│   └── ...
│
├── src/
│   └── tablemind/
│       │
│       ├── conversation/
│       │   ├── manager.py
│       │   ├── speechmatics.py
│       │   ├── streamlit_voice.py
│       │   ├── transcript_buffer.py
│       │   ├── utterance.py
│       │   ├── voice_pipeline.py
│       │   ├── voice_planner.py
│       │   └── ...
│       │
│       ├── control/
│       │   └── so101.py
│       │
│       ├── integration/
│       │   └── closed_loop.py
│       │
│       ├── perception/
│       │   ├── camera.py
│       │   ├── vision_scene_state.py
│       │   └── ...
│       │
│       ├── reasoning/
│       │   ├── interpreter.py
│       │   ├── context_planner.py
│       │   ├── goal_reconciler.py
│       │   └── ...
│       │
│       ├── simulation/
│       │   ├── runtime.py
│       │   └── ...
│       │
│       └── ...
│
└── tests/
    ├── test_bimanual_scene.py
    ├── test_bidirectional_goal_modification.py
    └── ...
```

---

# ⚙️ Installation

## Requirements

* Python 3.12–3.13
* MuJoCo
* Streamlit
* OpenVINO
* Speechmatics API access for voice interaction
* Intel CPU recommended for OpenVINO optimization

Python 3.13.14 was used during development and validation.

---

## Clone the Repository

```bash
git clone https://github.com/Sharannya26/Bimanual-VLA-TableMind.git
cd Bimanual-VLA-TableMind
```

---

## Create Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

---

## Install Dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

---

# 🔐 Environment Variables

Create a `.env` file in the project root.

```env
SPEECHMATICS_API_KEY=your_speechmatics_api_key
```

Do **not** commit `.env` or API keys to GitHub.

---

# ▶️ Run TABLEMIND

Launch the Streamlit interface:

```powershell
python -m streamlit run app.py
```

The application will open in the browser.

---

# 🤖 Robotics Demos

## Launch MuJoCo Simulation

```powershell
python scripts/launch_sim.py
```

Basic headless validation:

```text
Loaded 15 bodies, 12 actuators, arms=('left', 'right')
Headless validation complete after 1000 steps
```

---

## Closed-Loop VLA Demo

```powershell
python scripts/demo_closed_loop.py
```

The demonstration performs:

```text
Vision
  ↓
Scene Understanding
  ↓
Task Reasoning
  ↓
Planning
  ↓
Bimanual Execution
  ↓
Physical Verification
```

Expected final result:

```text
🔥 TABLEMIND CLOSED-LOOP VLA DEMO SUCCESSFUL
```

---

# 🔄 Bidirectional Goal Modification Demo

Run:

```powershell
python scripts/demo_bidirectional_goal.py
```

The demonstration performs:

```text
1 Guest
   ↓
Increase Goal
   ↓
2 Guests
   ↓
Decrease Goal
   ↓
1 Guest
```

The physical state is verified after every change.

Expected final result:

```text
🔥 TABLEMIND 9.5 BIDIRECTIONAL GOAL DEMO SUCCESSFUL

    1  →  2  →  1
```

---

# 🎙️ Speechmatics Demo

Run:

```powershell
python scripts/demo_speechmatics.py
```

Example command:

```text
"Set the table for two."
```

Expected NLU interpretation:

```text
Intent:    set_table
Objects:   ('plate', 'glass')
Quantity:  2
```

---

# 🧪 Testing

TABLEMIND currently contains a comprehensive automated test suite covering:

* MuJoCo scene construction
* Dual SO-101 configuration
* Robot control
* Cartesian IK
* Gripper control
* Manipulation
* Scene perception
* Vision grounding
* Planning
* Bimanual execution
* Closed-loop verification
* Goal reconciliation
* Conversational modifications
* Voice/NLU components
* Integration behavior

Current validation:

```text
244 passed
```

The full suite has been executed successfully.

---

# 🧠 Milestone Progress

```text
MILESTONE 1
Dual SO-101 MuJoCo Scene
        ✓

MILESTONE 2
SO-101 Cartesian IK + Manipulation
        ✓

MILESTONE 3
Camera + Scene Perception
        ✓

MILESTONE 4
Deterministic Task Planning
        ✓

MILESTONE 5
Dynamic Objects + Grasping + Bimanual Execution
        ✓

MILESTONE 6
Natural Language Understanding + World Grounding
        ✓

MILESTONE 7
Speechmatics Voice Interaction + Conversation Pipeline
        ✓

MILESTONE 8
Custom YOLO26n + OpenVINO Optimization
        ✓

MILESTONE 9.1–9.3
Vision-Grounded VLA
        ✓

MILESTONE 9.4
Real Closed-Loop VLA
        ✓

MILESTONE 9.5
Bidirectional Conversational Goal Reconciliation
        ✓

MILESTONE 9.6
Final Integration + UI + Optimization + Polish
        ✓
```

---

# 🏆 Hackathon Track Alignment

TABLEMIND directly addresses the requirements of the **Bimanual VLA Manipulation with Multi-Modal Reasoning** challenge.

| Challenge Requirement            | TABLEMIND |
| -------------------------------- | --------- |
| Natural-language instructions    | ✅         |
| Visual scene understanding       | ✅         |
| Multi-modal reasoning            | ✅         |
| Two SO-101 arms                  | ✅         |
| MuJoCo simulation                | ✅         |
| Multi-step manipulation          | ✅         |
| Bimanual coordination            | ✅         |
| Physical verification            | ✅         |
| Closed-loop execution            | ✅         |
| Conversational goal modification | ✅         |
| Replanning                       | ✅         |
| Voice interaction                | ✅         |
| OpenVINO optimization            | ✅         |
| Intel CPU deployment             | ✅         |

---

# 🔥 Key Demonstration

TABLEMIND can reconcile a changing user goal without restarting the physical simulation.

```text
1 GUEST
   ↓
"Actually, make it 2."
   ↓
2 GUESTS
   ↓
"Actually, one guest isn't coming."
   ↓
1 GUEST
```

The complete sequence was executed in the real MuJoCo simulation.

The second place setting was:

1. Physically restored
2. Visually verified
3. Physically removed
4. Verified as no longer occupying the active goal slot

This demonstrates that TABLEMIND does not merely generate a new plan.

It **reconciles the user's new goal with the current physical world state**.

---

# 💡 What Makes TABLEMIND Different?

Most robotic task demonstrations can be summarized as:

```text
Command → Plan → Execute
```

TABLEMIND extends this into:

```text
Command
   ↓
Understand
   ↓
Observe
   ↓
Reason
   ↓
Plan
   ↓
Coordinate
   ↓
Act
   ↓
Verify
   ↓
Listen Again
   ↓
Reconcile
   ↓
Adapt
```

The system therefore combines several capabilities into one end-to-end Physical AI workflow:

### 1. Conversational interaction

The robot understands natural-language instructions rather than requiring hard-coded task parameters.

### 2. Visual grounding

The system reasons over objects observed in the simulated physical environment.

### 3. Bimanual coordination

Two SO-101 manipulators coordinate their actions to perform the task.

### 4. Closed-loop verification

The robot checks whether the physical world actually matches the expected state.

### 5. Goal reconciliation

The user can modify the task after execution has already occurred.

### 6. Hardware-aware inference

OpenVINO is used to optimize the vision pipeline for Intel hardware.

### 7. Human-facing interface

The complete pipeline is exposed through an interactive Streamlit command center.

---

# 🎬 Demonstration

The primary demonstration shows:

```text
Natural Language
      ↓
Vision
      ↓
Reasoning
      ↓
Bimanual Planning
      ↓
SO-101 Manipulation
      ↓
MuJoCo
      ↓
Verification
      ↓
Conversational Modification
      ↓
Physical Reconciliation
```

### Example interaction

```text
USER:
"Set the table for one."

TABLEMIND:
→ Understands the request
→ Detects plate + glass
→ Plans manipulation
→ Executes with SO-101 arms
→ Verifies the result


USER:
"Actually, make it 2."

TABLEMIND:
→ Observes current state
→ Computes missing place setting
→ Plans only the required addition
→ Executes plate + glass placement
→ Verifies two place settings


USER:
"Actually, one guest isn't coming."

TABLEMIND:
→ Observes current state
→ Computes required reduction
→ Identifies setting #2
→ Physically removes its plate + glass
→ Verifies the final one-person table
```

---

# 📊 Final System Capabilities

| Capability                       | Status |
| -------------------------------- | ------ |
| Dual SO-101 simulation           | ✅      |
| MuJoCo physics                   | ✅      |
| Cartesian IK                     | ✅      |
| Gripper control                  | ✅      |
| Bimanual manipulation            | ✅      |
| Object perception                | ✅      |
| Custom YOLO26n                   | ✅      |
| OpenVINO inference               | ✅      |
| Intel CPU validation             | ✅      |
| Natural-language understanding   | ✅      |
| Speechmatics voice interaction   | ✅      |
| Task decomposition               | ✅      |
| Context-aware planning           | ✅      |
| Closed-loop execution            | ✅      |
| Physical verification            | ✅      |
| Conversational goal modification | ✅      |
| Goal increase                    | ✅      |
| Goal decrease                    | ✅      |
| Streamlit interface              | ✅      |
| Automated test suite             | ✅      |
| 1 → 2 → 1 physical demonstration | ✅      |

---

# 🔮 Future Extensions

Potential future extensions include:

* Real SO-101 hardware deployment
* Intel Core Ultra Series 2/3 acceleration
* NPU-aware inference
* More complex household manipulation tasks
* Longer conversational task sequences
* Richer object categories
* More advanced recovery behaviors
* Multi-object rearrangement
* Additional multimodal inputs
* Learning-based manipulation policies

---

# 👤 Author

**Sharannya**

Computer Engineering Student
Physical AI / Robotics / AI Systems

---

# 📜 License

This project was created as a hackathon project for educational and research purposes.

---

# 🔗 Repository

**GitHub:**

[https://github.com/Sharannya26/Bimanual-VLA-TableMind](https://github.com/Sharannya26/Bimanual-VLA-TableMind)

---

## TABLEMIND

> **Understand the human. See the world. Act with both hands. Verify the result. Adapt when the goal changes.**

```


