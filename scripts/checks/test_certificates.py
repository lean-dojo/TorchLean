"""Numerical-region and input-validation checks for Python certificate producers."""

import contextlib
import copy
from fractions import Fraction
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import MappingProxyType
import unittest
from unittest import mock
from unittest.mock import patch


def load_producer(name, relative):
    source = Path(__file__).resolve().parents[1] / "verification" / relative
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


regions = load_producer("lirpa_export", "lirpa/export_cert.py")
exporter = load_producer("margin_export", "robustness/export_margin_cert.py")
leaf = load_producer("leaf_export", "abcrown/export_leaf_artifact.py")
centered_box = regions.centered_box
image_box = regions.image_box
layernorm_range_interval = regions.layernorm_range_interval
ArtifactExportError = leaf.ArtifactExportError
build_abcrown_leaf_artifact = leaf.build_abcrown_leaf_artifact
write_abcrown_leaf_artifact = leaf.write_abcrown_leaf_artifact


class CenteredBoxTests(unittest.TestCase):
    def assert_encloses(self, center, eps, lo, hi):
        radius = Fraction.from_float(eps)
        self.assertEqual(len(center), len(lo))
        self.assertEqual(len(center), len(hi))
        for x, lower, upper in zip(center, lo, hi):
            exact = Fraction.from_float(x)
            self.assertLessEqual(Fraction.from_float(lower), exact - radius)
            self.assertLessEqual(exact + radius, Fraction.from_float(upper))

    def test_unrepresentable_endpoints(self):
        # Ordinary 1 ± 2^-55 both round to 1, losing both exact endpoints.
        center, eps = [1.0, -1.0], 2.0**-55
        self.assert_encloses(center, eps, *centered_box(center, eps))

    def test_fixture_and_zero_radius_regions(self):
        for center, eps in (
            ([1.0, 2.0, 3.0], 1.0),
            ([1.0, 2.0, 3.0, 4.0], 0.5),
            ([1.0, -1.0, 0.0], 0.1),
            ([1.0, -1.0, 0.0], 0.0),
            ([], 0.5),
        ):
            with self.subTest(center=center, eps=eps):
                self.assert_encloses(center, eps, *centered_box(center, eps))

    def test_cnn_region(self):
        lo, hi = image_box(0.1)
        flat_lo = [value for channel in lo for row in channel for value in row]
        flat_hi = [value for channel in hi for row in channel for value in row]
        self.assert_encloses([1.0] * 16, 0.1, flat_lo, flat_hi)


class LayerNormRangeTests(unittest.TestCase):
    def test_four_element_host_endpoints(self):
        # Lean widens sqrt(4) upward, then widens the subtraction 0 - radius downward.
        lo, hi = layernorm_range_interval(4)
        self.assertEqual(lo, [float.fromhex("-0x1.0000000000002p+1")] * 4)
        self.assertEqual(hi, [float.fromhex("0x1.0000000000001p+1")] * 4)

    def test_empty_and_singleton_rows(self):
        self.assertEqual(layernorm_range_interval(0), ([], []))
        self.assertEqual(layernorm_range_interval(1), ([0.0], [0.0]))

    def test_exact_sqrt_enclosure(self):
        for length in (2, 3, 4, 7, 16):
            with self.subTest(length=length):
                lo, hi = layernorm_range_interval(length)
                self.assertEqual(len(lo), length)
                self.assertEqual(len(hi), length)
                for lower, upper in zip(lo, hi):
                    lower = Fraction.from_float(lower)
                    upper = Fraction.from_float(upper)
                    self.assertLess(lower, 0)
                    self.assertGreater(upper, 0)
                    self.assertGreaterEqual(lower * lower, length)
                    self.assertGreaterEqual(upper * upper, length)


class RootOverrideTests(unittest.TestCase):
    def setUp(self):
        self.leaf = {
            "x_L": [-0.5, -0.5], "x_U": [0.5, 0.5],
            "lower_bounds": [1.0], "thresholds": [0.0],
        }
        self.raw = {"domains": [self.leaf]}
        self.artifact = build_abcrown_leaf_artifact(self.raw)
        self.inputs = [self.leaf, [self.leaf], self.raw, self.artifact]

    def test_override_on_every_input_shape(self):
        for raw in self.inputs:
            with self.subTest(raw=raw):
                original = copy.deepcopy(raw)
                result = build_abcrown_leaf_artifact(raw, root_lo=[-1, -1], root_hi=[1, 1])
                self.assertEqual(result["root"], {"lo": [-1, -1], "hi": [1, 1]})
                self.assertEqual(result["leaves"], self.artifact["leaves"])
                self.assertEqual(raw, original)

    def test_partial_override_is_rejected(self):
        for raw in self.inputs:
            for bounds in [{"root_lo": [-1, -1]}, {"root_hi": [1, 1]}]:
                with self.subTest(raw=raw, bounds=bounds):
                    with self.assertRaisesRegex(ArtifactExportError, "requires both bounds"):
                        build_abcrown_leaf_artifact(raw, **bounds)

    def test_partial_embedded_root_is_rejected(self):
        for bounds in [{"root_lo": [-1, -1]}, {"root_hi": [1, 1]},
                       {"input_lo": [-1, -1]}, {"x_U": [1, 1]}]:
            with self.subTest(bounds=bounds):
                with self.assertRaisesRegex(ArtifactExportError, "requires both bounds"):
                    build_abcrown_leaf_artifact({**self.raw, **bounds})

    @mock.patch.dict("os.environ", {}, clear=True)
    def test_missing_output_path_is_rejected(self):
        with self.assertRaisesRegex(ArtifactExportError, "no output path supplied"):
            write_abcrown_leaf_artifact(root_lo=[-1, -1], root_hi=[1, 1],
                                       leaves=[self.leaf])

    def test_no_override_preserves_artifact(self):
        self.assertEqual(build_abcrown_leaf_artifact(self.artifact), self.artifact)
        self.assertEqual(build_abcrown_leaf_artifact(self.raw), self.artifact)

    def test_invalid_override_is_rejected(self):
        for lo, hi in [([0], [1]), ([2, 2], [1, 1]),
                       ([float("nan"), 0], [1, 1]), ([-1, -1], [1])]:
            with self.subTest(lo=lo, hi=hi):
                with self.assertRaises(ArtifactExportError):
                    build_abcrown_leaf_artifact(self.artifact, root_lo=lo, root_hi=hi)

    def test_overflowed_witness_margin_is_rejected(self):
        leaf = {**self.leaf, "lower_bounds": [sys.float_info.max],
                "thresholds": [-sys.float_info.max]}
        with self.assertRaisesRegex(ArtifactExportError, "witness margin.*finite"):
            build_abcrown_leaf_artifact([leaf])

    def test_unrepresentable_input_number_is_exporter_error(self):
        leaf = {**self.leaf, "lower_bounds": [10**400]}
        with self.assertRaises(ArtifactExportError):
            build_abcrown_leaf_artifact([leaf])

    def test_sequence_and_mapping_inputs_follow_public_signature(self):
        leaf = MappingProxyType(self.leaf)
        result = build_abcrown_leaf_artifact((leaf,))
        self.assertEqual(result, self.artifact)


class MarginCertificateTests(unittest.TestCase):
    def run_export(self, weights, example, eps="0.125"):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "weights.json").write_text(json.dumps(weights))
            (root / "data.json").write_text(json.dumps({"examples": [example]}))
            argv = ["export", "--weights", str(root / "weights.json"),
                    "--dataset", str(root / "data.json"), "--out", str(root / "out.json"),
                    f"--eps={eps}"]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()), \
                 contextlib.redirect_stderr(io.StringIO()):
                exporter.main()
            return json.loads((root / "out.json").read_text())

    def test_exact_bounds_and_first_tie(self):
        weights = {"layers.0.weight": [[2.0, -1.0], [-2.0, 1.0]], "layers.0.bias": [1.0, 0.0]}
        result = self.run_export(weights, {"x": [0.5, 0.25], "y": 0})
        self.assertEqual(result["examples"][0]["logits_lo"], [1.375, -1.125])
        self.assertEqual(result["examples"][0]["logits_hi"], [2.125, -0.375])
        self.assertEqual(exporter.argmax([1.0, 1.0]), 0)

    def test_reject_malformed_numeric_inputs(self):
        weights = {"layers.0.weight": [[1.0], [0.0]], "layers.0.bias": [0.0, 0.0]}
        valid = {"x": [0.5], "y": 0}
        for eps in ("nan", "inf", "-0.1"):
            with self.assertRaises(SystemExit):
                self.run_export(weights, valid, eps)
        for example in ({"x": [float("nan")], "y": 0}, {"x": [1.1], "y": 0},
                        {"x": [0.5], "y": 0.5}, {"x": [0.5], "y": 0, "id": -1}):
            with self.assertRaises(SystemExit):
                self.run_export(weights, example)
        for bad in ({"layers.0.weight": [[1.0]], "layers.0.bias": [0.0]},
                    {"layers.0.weight": [[1.0], []], "layers.0.bias": [0.0, 0.0]},
                    {"layers.0.weight": [[float("inf")], [0.0]], "layers.0.bias": [0.0, 0.0]}):
            with self.assertRaises(SystemExit):
                self.run_export(bad, valid)


if __name__ == "__main__":
    unittest.main()
