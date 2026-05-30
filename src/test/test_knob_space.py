#!/usr/bin/env python3
"""Unit tests for KnobSpace conversions and validation."""

# ruff: noqa: E402

import sys
import unittest
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from util.knob_space import KnobSpace


class TestKnobSpace(unittest.TestCase):
    def setUp(self) -> None:
        self.knob_names = ["k1", "k2", "k3"]
        self.physical_ranges = {
            "k1": (0.1, 10.0),
            "k2": (1.0, 100.0),
            "k3": (0.5, 50.0),
        }

    def test_default_vector(self) -> None:
        knob_space = KnobSpace(self.knob_names)
        self.assertEqual(knob_space.get_default_vector(), [0.5, 0.5, 0.5])

    def test_default_vector_uses_physical_default_from_search_space(self) -> None:
        knob_space = KnobSpace.from_search_space([
            {"var": "k1", "min": 0.1, "max": 100.0, "default": 1.0},
        ])

        default_vector = knob_space.get_default_vector()
        default_config = knob_space.vector_to_config(default_vector)
        normalized_space = knob_space.get_normalized_search_space()

        self.assertAlmostEqual(default_vector[0], 1.0 / 3.0)
        self.assertAlmostEqual(default_config["k1"], 1.0)
        self.assertAlmostEqual(normalized_space[0]["default"], 1.0 / 3.0)

    def test_normalized_search_space(self) -> None:
        knob_space = KnobSpace(self.knob_names)
        normalized = knob_space.get_normalized_search_space()
        self.assertEqual(len(normalized), 3)
        for item in normalized:
            self.assertEqual(item["min"], 0.0)
            self.assertEqual(item["max"], 1.0)
            self.assertEqual(item["default"], 0.5)

    def test_vector_to_config_and_back_with_physical_ranges(self) -> None:
        knob_space = KnobSpace(self.knob_names, self.physical_ranges)
        vector = [0.2, 0.5, 0.8]
        config = knob_space.vector_to_config(vector)
        back_vector = knob_space.config_to_vector(config)

        self.assertEqual(set(config.keys()), set(self.knob_names))
        for original, restored in zip(vector, back_vector):
            self.assertAlmostEqual(original, restored, places=7)

    def test_validate_and_clamp_vector(self) -> None:
        knob_space = KnobSpace(self.knob_names)
        self.assertTrue(knob_space.validate_vector([0.0, 0.5, 1.0]))
        self.assertFalse(knob_space.validate_vector([0.2, 0.5]))
        self.assertFalse(knob_space.validate_vector([0.2, -0.1, 0.9]))
        self.assertFalse(knob_space.validate_vector([0.2, float("nan"), 0.9]))
        self.assertFalse(knob_space.validate_vector([0.2, float("inf"), 0.9]))

        clamped = knob_space.clamp_vector([-0.4, 0.3, 3.0, float("nan"), float("inf"), "bad"])
        self.assertEqual(clamped, [0.0, 0.3, 1.0, 0.5, 0.5, 0.5])

    def test_from_search_space(self) -> None:
        search_space = [
            {"var": "k1", "min": 0.1, "max": 10.0},
            {"var": "k2", "min": 1.0, "max": 100.0},
        ]
        knob_space = KnobSpace.from_search_space(search_space)
        self.assertEqual(knob_space.dimension, 2)
        config = knob_space.vector_to_config([0.5, 0.5])
        self.assertIn("k1", config)
        self.assertIn("k2", config)

    def test_physical_ranges_must_be_positive_for_log_scale(self) -> None:
        with self.assertRaises(ValueError):
            KnobSpace(["k1"], {"k1": (0.0, 1.0)})

        with self.assertRaises(ValueError):
            KnobSpace(["k1"], {"k1": (10.0, 1.0)})

    def test_config_to_vector_rejects_non_positive_physical_value(self) -> None:
        knob_space = KnobSpace(["k1"], {"k1": (0.1, 10.0)})

        with self.assertRaises(ValueError):
            knob_space.config_to_vector({"k1": 0.0})

    def test_vector_to_config_rejects_invalid_normalized_coordinate(self) -> None:
        knob_space = KnobSpace(["k1"], {"k1": (0.1, 10.0)})

        for value in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    knob_space.vector_to_config([value])

    def test_config_to_vector_rejects_non_finite_normalized_coordinate(self) -> None:
        knob_space = KnobSpace(["k1"])

        with self.assertRaises(ValueError):
            knob_space.config_to_vector({"k1": float("nan")})

    def test_rejects_physical_default_outside_search_space(self) -> None:
        with self.assertRaises(ValueError):
            KnobSpace.from_search_space([
                {"var": "k1", "min": 0.1, "max": 10.0, "default": 100.0},
            ])


if __name__ == "__main__":
    unittest.main(verbosity=2)
