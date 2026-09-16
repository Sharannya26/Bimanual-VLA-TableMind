# TableMind

Milestone 1 provides a MuJoCo foundation with two separately addressable SO-101 arms, a table workspace, placeholder dinnerware, lighting, and an overview camera. Milestone 2 adds deterministic Cartesian reach control and gripper control for either arm. It deliberately contains no perception, planning, conversational, VLA, or recovery functionality.

## Launch the simulation

From the active project virtual environment, install TableMind in editable mode once. This makes the `src/tablemind` package available to the launch script without setting `PYTHONPATH`:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Then launch from the repository root:

```powershell
python scripts/launch_sim.py
```

The first launch composes `assets/mujoco/bimanual_table.xml` from the vendored upstream SO-101 MJCF and opens the MuJoCo viewer. Close the viewer window to exit.

For a non-visual validation run:

```powershell
python scripts/launch_sim.py --headless --steps 1000
python -m pytest
```

## Milestone 2: basic manipulation control

`tablemind.control.SO101Controller` is the reusable low-level interface. `move_to()` uses damped least-squares Cartesian position IK against each namespaced arm's `gripperframe` site; `open_gripper()` and `close_gripper()` command the separate gripper actuator. `execute_pick()` and `execute_place()` provide an explicit future-facing manipulation API.

Run the deterministic left-arm reach/gripper demonstration from the repository root:

```powershell
python scripts/demo_basic_manipulation.py
```

### Current limitation

The plates and glasses in the Milestone 1 scene are fixed world geoms, not free bodies, so they cannot be grasped or moved by physics. The demo deliberately reports the pick as incomplete and does **not** claim object transfer. It validates deterministic Cartesian reach, safe approach targets, and gripper commands only. Movable tableware and grasp/contact validation remain a later Milestone 2 extension.

## SO-101 asset provenance

`assets/mujoco/so101/` contains the `so101_new_calib.xml` model and meshes from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100), commit `eecbe3e`, licensed under Apache-2.0 (included in that directory). The scene composer keeps the model geometry, joints, limits, and actuator setup intact; it creates two instances by prefixing model-wide names with `left_` and `right_`.

This is compatible with LeRobot's current SO-101 convention: the upstream model uses the new calibration with virtual zero at mid-range. The gripper's native MuJoCo hinge remains in radians in this milestone; LeRobot's hardware-level `0..100` gripper mapping is intentionally not introduced yet.
