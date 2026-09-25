"""Exact-rational regressions for the regions used by LiRPA certificate producers."""

from fractions import Fraction
import unittest

from common import centered_box, layernorm_range_interval
from export_cnn_cert import seed_input_box


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
        lo, hi = seed_input_box(0.1)
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


if __name__ == "__main__":
    unittest.main()
