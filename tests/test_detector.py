"""Tests for TABLEMIND object perception."""

from __future__ import annotations

from tablemind.perception import GroundTruthDetector
from tablemind.simulation import BimanualTableSimulation


def test_detector_finds_table_objects() -> None:
    simulation = BimanualTableSimulation.create()

    detector = GroundTruthDetector(simulation.model)

    observation = detector.detect()

    assert len(observation.objects) > 0


def test_detector_finds_plates() -> None:
    simulation = BimanualTableSimulation.create()

    detector = GroundTruthDetector(simulation.model)

    observation = detector.detect()

    plates = observation.by_type("plate")

    assert len(plates) == 2


def test_detector_finds_glasses() -> None:
    simulation = BimanualTableSimulation.create()

    detector = GroundTruthDetector(simulation.model)

    observation = detector.detect()

    glasses = observation.by_type("glass")

    assert len(glasses) == 2


def test_detected_objects_have_positions() -> None:
    simulation = BimanualTableSimulation.create()

    detector = GroundTruthDetector(simulation.model)

    observation = detector.detect()

    for obj in observation.objects:
        assert len(obj.position) == 3