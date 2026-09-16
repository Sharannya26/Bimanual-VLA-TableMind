"""Compose two namespaced instances of the upstream SO-101 MJCF model."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

from tablemind.robot_config.so101 import ARM_PLACEMENTS
from tablemind.scene_config.table import (
    GLASS_POSITIONS,
    PLATE_POSITIONS,
    TABLE_SIZE,
    TABLE_TOP_Z,
)
from tablemind.utils.paths import COMPOSED_SCENE, SO101_MODEL


def _prefix_robot_names(
    element: ET.Element,
    prefix: str,
) -> None:
    """Namespace model-wide body, joint, site, and actuator names."""

    reference_attributes = {
        "joint",
        "body",
        "site",
        "tendon",
        "actuator",
        "sensor",
    }

    for node in element.iter():
        if "name" in node.attrib:
            node.set(
                "name",
                f"{prefix}_{node.get('name')}",
            )

        for attribute in reference_attributes:
            if attribute in node.attrib:
                node.set(
                    attribute,
                    f"{prefix}_{node.get(attribute)}",
                )


def _add_plate(
    worldbody: ET.Element,
    plate_name: str,
    position: tuple[float, float, float],
) -> None:
    """Add one dynamic plate to the MuJoCo world."""

    x, y, z = position

    body = ET.SubElement(
        worldbody,
        "body",
        name=plate_name,
        pos=f"{x} {y} {z}",
    )

    ET.SubElement(
        body,
        "freejoint",
        name=f"{plate_name}_freejoint",
    )

    ET.SubElement(
        body,
        "geom",
        name=plate_name,
        type="cylinder",
        size="0.115 0.012",
        density="300",
        friction="1.2 0.5 0.1",
        rgba="0.95 0.95 0.95 1",
        contype="1",
        conaffinity="1",
    )

    # Invisible point used as the grasp reference.
    ET.SubElement(
        body,
        "site",
        name=f"{plate_name}_grasp",
        type="sphere",
        pos="0 0 0",
        size="0.008",
        rgba="0 0 0 0",
    )


def _add_glass(
    worldbody: ET.Element,
    glass_name: str,
    position: tuple[float, float, float],
) -> None:
    """Add one dynamic glass to the MuJoCo world."""

    x, y, z = position

    body = ET.SubElement(
        worldbody,
        "body",
        name=glass_name,
        pos=f"{x} {y} {z}",
    )

    ET.SubElement(
        body,
        "freejoint",
        name=f"{glass_name}_freejoint",
    )

    # Main glass body.
    ET.SubElement(
        body,
        "geom",
        name=glass_name,
        type="cylinder",
        size="0.038 0.060",
        density="100",
        friction="1.2 0.5 0.1",
        rgba="0.25 0.55 0.70 0.65",
        contype="1",
        conaffinity="1",
    )

    # Wider bottom/base for stability.
    ET.SubElement(
        body,
        "geom",
        name=f"{glass_name}_base",
        type="cylinder",
        pos="0 0 -0.052",
        size="0.052 0.008",
        density="250",
        friction="1.5 0.5 0.1",
        rgba="0.20 0.45 0.60 0.85",
        contype="1",
        conaffinity="1",
    )

    # Invisible point used as the grasp reference.
    ET.SubElement(
        body,
        "site",
        name=f"{glass_name}_grasp",
        type="sphere",
        pos="0 0 0",
        size="0.008",
        rgba="0 0 0 0",
    )


def _add_table_objects(
    worldbody: ET.Element,
) -> None:
    """Add the table and active dinnerware."""

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------

    ET.SubElement(
        worldbody,
        "geom",
        name="table",
        type="box",
        pos=f"0 0 {TABLE_TOP_Z - TABLE_SIZE[2]}",
        size=" ".join(map(str, TABLE_SIZE)),
        rgba="0.32 0.12 0.05 1",
        friction="1.0 0.5 0.2",
    )

    # ------------------------------------------------------------------
    # Active plates
    # ------------------------------------------------------------------

    for index, position in enumerate(
        PLATE_POSITIONS,
        start=1,
    ):
        _add_plate(
            worldbody,
            f"plate_{index}",
            position,
        )

    # ------------------------------------------------------------------
    # Active glasses
    # ------------------------------------------------------------------

    for index, position in enumerate(
        GLASS_POSITIONS,
        start=1,
    ):
        _add_glass(
            worldbody,
            f"glass_{index}",
            position,
        )


def _add_grasp_constraints(
    root: ET.Element,
) -> None:
    """Add inactive weld constraints for all active arm/object pairs."""

    equality = ET.SubElement(
        root,
        "equality",
    )

    arms = (
        "left",
        "right",
    )

    objects = (
        "plate_1",
        "plate_2",
        "glass_1",
        "glass_2",
    )

    for arm in arms:
        for object_id in objects:
            ET.SubElement(
                equality,
                "weld",
                name=f"grasp_{arm}_{object_id}",
                site1=f"{arm}_gripperframe",
                site2=f"{object_id}_grasp",
                active="false",
            )


def build_bimanual_scene(
    output_path: Path = COMPOSED_SCENE,
) -> Path:
    """Build an MJCF scene using two uniquely namespaced SO-101 instances."""

    if not SO101_MODEL.is_file():
        raise FileNotFoundError(
            f"Upstream SO-101 MJCF asset is missing: {SO101_MODEL}"
        )

    upstream = ET.parse(
        SO101_MODEL,
    ).getroot()

    robot_body = upstream.find(
        "worldbody/body",
    )

    robot_actuators = upstream.find(
        "actuator",
    )

    robot_assets = upstream.find(
        "asset",
    )

    if (
        robot_body is None
        or robot_actuators is None
        or robot_assets is None
    ):
        raise ValueError(
            "The upstream SO-101 MJCF does not have "
            "the expected body/assets/actuator layout"
        )

    root = ET.Element(
        "mujoco",
        model="tablemind_bimanual_so101",
    )

    ET.SubElement(
        root,
        "compiler",
        angle="radian",
        meshdir="so101/assets",
        autolimits="true",
    )

    ET.SubElement(
        root,
        "option",
        timestep="0.002",
        gravity="0 0 -9.81",
        integrator="implicitfast",
    )

    for default in upstream.findall("default"):
        root.append(
            deepcopy(default),
        )

    # ------------------------------------------------------------------
    # Assets
    # ------------------------------------------------------------------

    asset = ET.SubElement(
        root,
        "asset",
    )

    for item in robot_assets:
        asset.append(
            deepcopy(item),
        )

    ET.SubElement(
        asset,
        "material",
        name="table_wood",
        rgba="0.34 0.16 0.07 1",
    )

    ET.SubElement(
        asset,
        "material",
        name="ceramic",
        rgba="0.94 0.94 0.90 1",
    )

    ET.SubElement(
        asset,
        "material",
        name="glass",
        rgba="0.38 0.72 0.86 0.45",
    )

    # ------------------------------------------------------------------
    # World
    # ------------------------------------------------------------------

    worldbody = ET.SubElement(
        root,
        "worldbody",
    )

    ET.SubElement(
        worldbody,
        "light",
        name="key_light",
        pos="0 -0.4 2.5",
        dir="0 0 -1",
        directional="true",
    )

    ET.SubElement(
        worldbody,
        "light",
        name="fill_light",
        pos="-1 1 1.6",
        directional="false",
        diffuse="0.45 0.45 0.45",
    )

    ET.SubElement(
        worldbody,
        "geom",
        name="floor",
        type="plane",
        size="0 0 0.1",
        rgba="0.16 0.18 0.20 1",
    )

    ET.SubElement(
        worldbody,
        "camera",
        name="table_overview",
        pos="0 -1.85 1.55",
        xyaxes="1 0 0 0 0.64 0.77",
        fovy="52",
    )

    _add_table_objects(
        worldbody,
    )

    # ------------------------------------------------------------------
    # Robot actuators and bodies
    # ------------------------------------------------------------------

    actuator = ET.SubElement(
        root,
        "actuator",
    )

    for arm, placement in ARM_PLACEMENTS.items():
        body = deepcopy(
            robot_body,
        )

        _prefix_robot_names(
            body,
            arm,
        )

        body.set(
            "pos",
            " ".join(
                map(
                    str,
                    placement.position,
                )
            ),
        )

        body.set(
            "quat",
            " ".join(
                map(
                    str,
                    placement.quaternion,
                )
            ),
        )

        worldbody.append(
            body,
        )

        for source_actuator in robot_actuators:
            arm_actuator = deepcopy(
                source_actuator,
            )

            _prefix_robot_names(
                arm_actuator,
                arm,
            )

            actuator.append(
                arm_actuator,
            )

    # ------------------------------------------------------------------
    # Grasp constraints
    # ------------------------------------------------------------------

    _add_grasp_constraints(
        root,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ET.indent(
        root,
        space="  ",
    )

    ET.ElementTree(
        root,
    ).write(
        output_path,
        encoding="utf-8",
        xml_declaration=True,
    )

    return output_path