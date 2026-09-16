"""Filesystem paths rooted at the repository, independent of the launch directory."""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MUJOCO_ASSETS = REPOSITORY_ROOT / "assets" / "mujoco"
SO101_MODEL = MUJOCO_ASSETS / "so101" / "so101_new_calib.xml"
COMPOSED_SCENE = MUJOCO_ASSETS / "bimanual_table.xml"
