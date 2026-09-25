"""Exact-rational regressions for the regions used by LiRPA certificate producers."""

from fractions import Fraction
import unittest

from common import centered_box
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


if __name__ == "__main__":
    unittest.main()
