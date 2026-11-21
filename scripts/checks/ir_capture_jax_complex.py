"""JAX's complex arithmetic captured and checked with ``verify ir`` (``JaxComplexTests``): the
products, quotients, phases and powers that the Jaxpr importer computes on the parts of complex
operands (``python/torchlean/_jax_complex.py``), the complex functions, ``Op.complex``, and the
complex power, ``Op.complexPow``, against their definitions rounded once, and the complex orders,
folds, window reductions, contractions, convolutions and scatters. Run them through
``scripts/checks/test_ir_capture.py``, which describes the checks.
"""

import unittest

from ir_capture_support import (CHECKER, COMPARE, EXECUTE, FORMATS, HAS_JAX, CheckerCase,
                                _midpoints, _names, _nearest, _outputs, _records, _same, _statuses, _tags)
from ir_capture_jax import JaxCaptureTests


# The formats in which the import computes a complex product or quotient with binary32 and
# binary64 parts: precision 2 (emax - emin) + 3 p + 3 and an exponent bias above
# 2 (emax - emin + p) + 1 for parts of precision p and exponents emin to emax
# (python/torchlean/_jax_complex.py, _wide).
_WIDE_JSON = {"float32": {"kind": "binary", "exponent_bits": 11, "fraction_bits": 580,
                          "bias": 1023, "encoding": "ieee"},
              "float64": {"kind": "binary", "exponent_bits": 14, "fraction_bits": 4251,
                          "bias": 8191, "encoding": "ieee"}}
# The formats in which it computes a complex phase (lax.sign): precision
# 2 emax - 3 emin + 4 p + 6 and the same exponents (_jax_complex.py, phase).
_PHASE_JSON = {dtype: {**wide, "fraction_bits": bits}
               for (dtype, wide), bits in zip(_WIDE_JSON.items(), (733, 5329))}


def _complex_reference(kind, a, b, c, d, dtype):
    # The parts of (a + ib)(c + id) ("mul") or (a + ib) / (c + id) ("div") of the floats a, b,
    # c and d as the import defines them, ac - bd and ad + bc, or (ac + bd) / (c^2 + d^2) and
    # (bc - ad) / (c^2 + d^2), each computed exactly and rounded once to dtype. The operations
    # are IEEE 754-2019's without rounding, on exact rationals, signed zeros, infinities and
    # NaN: x - y is x + (-y), a zero has the sign that 6.3 gives it, 0 × inf, inf - inf, 0 / 0
    # and inf / inf are NaN (7.2), a nonzero finite value over a zero is an infinity (7.3), as is
    # an infinity over a finite value or a zero, and a finite value over an infinity is a zero.
    import math
    from fractions import Fraction

    def special(x, test):
        return isinstance(x, float) and test(x)

    def nan(x):
        return special(x, math.isnan)

    def infinite(x):
        return special(x, math.isinf)

    def zero(x):
        return special(x, lambda v: v == 0)

    def negative(x):
        return x < 0 if isinstance(x, Fraction) else math.copysign(1.0, x) < 0

    def signed(value, x, y):
        return -value if negative(x) != negative(y) else value

    def mul(x, y):
        if nan(x) or nan(y) or (infinite(x) and zero(y)) or (zero(x) and infinite(y)):
            return math.nan
        if infinite(x) or infinite(y) or zero(x) or zero(y):
            return signed(math.inf if infinite(x) or infinite(y) else 0.0, x, y)
        return x * y

    def add(x, y):
        if nan(x) or nan(y) or (infinite(x) and infinite(y) and x != y):
            return math.nan
        if zero(x) and zero(y):
            return -0.0 if negative(x) and negative(y) else 0.0
        if infinite(x) or zero(y):
            return x
        if infinite(y) or zero(x):
            return y
        return x + y or 0.0

    def div(x, y):
        if nan(x) or nan(y) or (infinite(x) and infinite(y)) or (zero(x) and zero(y)):
            return math.nan
        if infinite(x) or zero(y) or zero(x) or infinite(y):
            return signed(math.inf if infinite(x) or zero(y) else 0.0, x, y)
        return x / y

    a, b, c, d = (Fraction(v) if math.isfinite(v) and v else float(v) for v in (a, b, c, d))
    if kind == "mul":
        parts = (add(mul(a, c), -mul(b, d)), add(mul(a, d), mul(b, c)))
    else:
        norm = add(mul(c, c), mul(d, d))
        parts = (div(add(mul(a, c), mul(b, d)), norm), div(add(mul(b, c), -mul(a, d)), norm))
    return [_nearest(v, dtype) for v in parts]


def _phase_reference(a, b, dtype):
    # The parts of the phase (a + ib) / |a + ib| of the floats a and b as the import defines it,
    # a / sqrt(a^2 + b^2) and b / sqrt(a^2 + b^2), each the exact part rounded once to dtype,
    # and a + ib itself where both are zero. An exact part u is replaced by
    # floor(|u| 2^K) / 2^K, an integer square root, which rounds as u does: for parts of
    # precision p and exponents emin to emax, u is more than 2^(4 emin - 4 p - 2 emax - 2) from
    # every midpoint of the format (python/torchlean/_jax_complex.py, phase), and 2^-K is less.
    # At an infinite or NaN part the operations are IEEE 754-2019's without rounding: a NaN
    # makes the sum a^2 + b^2 and both parts NaN; otherwise |a + ib| is inf, a finite part over
    # it a zero of that part's sign and an infinite one NaN (7.2).
    import math
    from fractions import Fraction

    if a == 0 and b == 0:
        return [a, b]
    if math.isnan(a) or math.isnan(b):
        return [math.nan, math.nan]
    if math.isinf(a) or math.isinf(b):
        return [math.nan if math.isinf(v) else math.copysign(0.0, v) for v in (a, b)]
    precision, emin, emax = FORMATS[dtype]
    shift = 2 * emax + 4 * precision - 4 * emin + 4
    norm = Fraction(a) ** 2 + Fraction(b) ** 2
    parts = []
    for v in (a, b):
        square = Fraction(v) ** 2 * 4 ** shift / norm
        root = Fraction(math.isqrt(square.numerator // square.denominator), 2 ** shift)
        parts.append(_nearest(root if v > 0 else -root, dtype) if v else v)
    return parts


def _root_reference(a, b, dtype):
    # The parts of the principal square root of a + ib of the floats a and b: T + i copysign(U, b)
    # where a is not negative and U + i copysign(T, b) where it is, for T = sqrt((|z| + |a|) / 2)
    # and U = sqrt((|z| - |a|) / 2), each the exact part rounded once to dtype, and +0 + ib at
    # zero. An exact part W is replaced by floor(W 2^K) / 2^K, from integer square roots of
    # N 16^K and of (floor(|z| 4^K) ± |a| 4^K) / 2, which rounds as W does: for parts of precision
    # p and exponents emin to emax, W is more than 2^(4 (emin - p)) / (48 W |z|) >
    # 2^(4 emin - 4 p - 2 emax - 9) from every midpoint m of the format, since, for W' the other
    # part and A = |a|, 4 m^4 ∓ 4 A m^2 - b^2 = 4 (m - W)(m + W)(m^2 + W'^2) is a nonzero multiple
    # of 2^(4 (emin - p)), and 2^-K is less. Infinities and NaN give C11's values (G.6.4.2),
    # with the sign of a NaN b where C leaves it unspecified.
    import math
    from fractions import Fraction

    if math.isinf(b):
        return [math.inf, b]
    if math.isnan(a) or math.isnan(b):
        return [math.inf if a == math.inf else math.nan,
                math.copysign(math.inf, b) if a == -math.inf else math.nan]
    if a == 0 and b == 0:
        return [0.0, b]
    if math.isinf(a):
        return [math.inf, math.copysign(0.0, b)] if a > 0 else [0.0, math.copysign(math.inf, b)]
    precision, emin, emax = FORMATS[dtype]
    shift = 4 * precision - 4 * emin + 2 * emax + 10
    magnitude = abs(Fraction(a))
    norm = math.isqrt(int((magnitude ** 2 + Fraction(b) ** 2) * 16 ** shift))
    scaled = int(magnitude * 4 ** shift)
    large, small = [Fraction(math.isqrt((norm + s) // 2), 2 ** shift) for s in (scaled, -scaled)]
    real, imag = (small, large) if a < 0 else (large, small)
    return [_nearest(real, dtype), math.copysign(_nearest(imag, dtype), b)]


def _reciprocal_root_reference(a, b, dtype):
    # The parts of 1 / √(a + ib) of the floats a and b as the import defines it:
    # X - i copysign(Y, b) where a is not negative and Y - i copysign(X, b) where it is, for
    # X = T / |z| and Y = U / |z| with T and U the parts of the root (_root_reference), each the
    # exact part rounded once to dtype; +0 - i copysign(0, b) where a part is infinite;
    # +inf + NaN i at zero; and NaN + NaN i at a NaN elsewhere. An exact part W is replaced by
    # integer square roots of X^2 = (|z| + |a|) / (2 N) and Y^2 = b^2 / (2 N (|z| + |a|)) for
    # N = a^2 + b^2, within 2^-P W of W, which rounds as W does: for parts of precision p and
    # exponents emin to emax, W is more than 2^(3 - P) W from every midpoint of the format, for
    # P = 2 emax - 8 emin + 8 p + 5, since at a midpoint m, for W' the other part and A = |a|,
    # 4 N^2 m^4 ∓ 4 A N m^2 - b^2 = 4 N^2 (m^2 - W^2)(m^2 + W'^2) is a nonzero multiple of
    # 2^(8 (emin - p) + 6).
    import math
    from fractions import Fraction

    if math.isinf(a) or math.isinf(b):
        return [0.0, math.copysign(0.0, -b)]
    if math.isnan(a) or math.isnan(b):
        return [math.nan, math.nan]
    if a == 0 and b == 0:
        return [math.inf, math.nan]
    precision, emin, emax = FORMATS[dtype]
    bits = 2 * emax - 8 * emin + 8 * precision + 5

    def root(q):
        # sqrt(q), less than 2^(-1 - bits) sqrt(q) below it.
        size = q.numerator.bit_length() - q.denominator.bit_length()
        shift = max(0, bits + 2 - size // 2)
        return Fraction(math.isqrt(q.numerator * 4 ** shift // q.denominator), 2 ** shift)

    magnitude, imag = abs(Fraction(a)), Fraction(b)
    norm = magnitude ** 2 + imag ** 2
    total = root(norm) + magnitude
    large, small = root(total / (2 * norm)), root(imag ** 2 / (2 * norm * total))
    real, imag = (small, large) if a < 0 else (large, small)
    return [_nearest(real, dtype), math.copysign(_nearest(imag, dtype), -b)]


def _exact_rounded(value, dtype):
    # The real mpmath number `value`, nonzero and finite, rounded once to dtype, where it lies more
    # than 2^-250 of its size from the midpoints around that rounding; mpmath's values at 400 bits
    # are far closer to the exact ones, which then round alike.
    from fractions import Fraction

    sign, man, exp, _ = value._mpf_
    exact = (-1) ** sign * Fraction(int(man)) * Fraction(2) ** int(exp)
    candidate = _nearest(exact, dtype)
    low, high = _midpoints(abs(candidate), dtype)
    for m in (low, high):
        if m is not None and m and abs(abs(exact) - m) <= abs(exact) / Fraction(2) ** 250:
            raise AssertionError(f"uncertified rounding of {value}")
    return candidate


def _function_reference(name, a, b, dtype):
    # The parts of the complex function `name` of jax.lax at a + ib for floats a and b off the
    # axes, each mpmath's part of the principal value rounded once to dtype (_exact_rounded).
    import mpmath

    with mpmath.workprec(400):
        value = getattr(mpmath, name)(mpmath.mpc(a, b))
        return [_exact_rounded(v, dtype) for v in (value.real, value.imag)]


@unittest.skipUnless(CHECKER and HAS_JAX, "needs the built verify executable and JAX")
class JaxComplexTests(CheckerCase):
    setUp, verify = JaxCaptureTests.setUp, JaxCaptureTests.verify
    artifact = JaxCaptureTests.artifact

    def test_complex_product_rounds_its_definition_once(self):
        # lax.mul of a + ib and c + id is the definition, re = ac - bd and im = ad + bc, each
        # the exact part rounded once (_complex_reference), as the import computes it in a format
        # where the products and sums are exact (_WIDE_JSON). JAX documents x × y and no rounding
        # of the parts, and XLA's CPU code fuses products, so XLA's words are not the reference.
        # Random factors, eager and under jax.jit, with a real factor that jnp
        # converts (imaginary part +0); (2^100 + 2^100 i)^2, whose real part stays 0 where the
        # other overflows, and (2^600 + 2^600 i)^2 likewise; parts just below the overflow threshold
        # T, a subnormal midpoint and a midpoint between normal neighbours,
        # where rounding to binary64 or binary128 first gave inf, 0 and the even neighbour:
        # (31 2^51)(1082401 2^52) - 1 = T - 1 in binary32 is the greatest finite value, and
        # ((2^27 - 1) 2^485)((2^27 + 1) 2^485) - 1 in binary64, 2^-150 + 2^-298 is 2^-149 and
        # 2^-1075 + 2^-2148 is 2^-1074, and 1 + 2^-24 + 2^-120 is 1 + 2^-23 and
        # 1 + 2^-53 + 2^-200 is 1 + 2^-52; lax.mul's out_dtype,
        # which converts each operand first, a real one to the real part (x = [2, 3] times
        # y = [1, 1] is [2, 3]); and at infinities, NaNs and signed zeros the formula's
        # IEEE values, without C's Annex G recovery: (inf + inf i)(inf + 0i) is NaN + NaN i, and
        # (inf + 2^600 i)(1 + 2^600 i) is inf + inf i, with the exact 2^1200.
        import math

        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(17)

        def products(a, b, c, d):
            z, w = lax.complex(a, b), lax.complex(c, d)
            return [v for p in (z * w, z * c) for v in (p.real, p.imag)]

        def expected(a, b, c, d, dtype):
            columns = [np.asarray(v, np.float64).ravel() for v in (a, b, c, d)]
            cases = [list(zip(*columns)), [(p, q, r, 0.0) for p, q, r, _ in zip(*columns)]]
            values = [[_complex_reference("mul", *entry, dtype) for entry in case]
                      for case in cases]
            return [[part[k] for part in case] for case in values for k in (0, 1)]

        def check(function, operands, dtype, wanted=None):
            report = self.verify(function, *operands, checks=EXECUTE)
            self.assert_passes(report)
            got = [[float(v) for v in actual.double().ravel()] for actual in _outputs(report)]
            for actual, parts in zip(got, wanted or expected(*operands, dtype), strict=True):
                _same(self, actual, parts)
            return got

        rows = {"float32": [(31 * 2.0 ** 51, 1.0, 1082401 * 2.0 ** 52, 1.0),
                            (2.0 ** -75, 2.0 ** -149, 2.0 ** -75, -(2.0 ** -149)),
                            (24929 * 2.0 ** -15, 2.0 ** -60, 673 * 2.0 ** -9, -(2.0 ** -60))],
                "float64": [((2 ** 27 - 1) * 2.0 ** 485, 1.0, (2 ** 27 + 1) * 2.0 ** 485, 1.0),
                            (2.0 ** -537, 2.0 ** -1074, 2.0 ** -538, -(2.0 ** -1074)),
                            (3.0, 2.0 ** -100, 3002399751580331 * 2.0 ** -53, -(2.0 ** -100))]}
        reals = {"float32": [float(np.finfo(np.float32).max), 2.0 ** -149, 1 + 2.0 ** -23],
                 "float64": [float(np.finfo(np.float64).max), 2.0 ** -1074, 1 + 2.0 ** -52]}
        for dtype, edge in (("float32", 2.0 ** 100), ("float64", 2.0 ** 600)):
            with jax.enable_x64(dtype == "float64"):
                factors = [jnp.asarray(rng.standard_normal((4, 64)), dtype) for _ in range(4)]
                for mode, function in (("eager", products), ("jit", jax.jit(products))):
                    with self.subTest(dtype=dtype, mode=mode):
                        check(function, factors, dtype)
                with self.subTest(dtype=dtype, edge=True):
                    got = check(products, [jnp.full((1,), edge, dtype)] * 4, dtype)
                    _same(self, [got[0][0], got[1][0]], [0.0, math.inf])
                with self.subTest(dtype=dtype, rounding=True):
                    got = check(products, [jnp.asarray(column, dtype)
                                           for column in zip(*rows[dtype])], dtype)
                    _same(self, got[0], reals[dtype])
                inf, nan = math.inf, math.nan
                special = [complex(inf, 0), 0j, complex(-0.0, 0), complex(nan, 1),
                           complex(1, inf), complex(-0.0, -0.0), complex(inf, inf),
                           complex(1, -1), complex(-inf, 2), complex(inf, edge),
                           complex(1, edge)]
                pairs = [(z, w) for z in special for w in special]
                parts = [jnp.asarray([getattr(pair[k], part) for pair in pairs], dtype)
                         for k in (0, 1) for part in ("real", "imag")]
                with self.subTest(dtype=dtype, special=True):
                    got = check(products, parts, dtype)
                    index = pairs.index((complex(inf, inf), complex(inf, 0)))
                    self.assertTrue(math.isnan(got[0][index]) and math.isnan(got[1][index]))
                    index = pairs.index((complex(inf, edge), complex(1, edge)))
                    _same(self, [got[0][index], got[1][index]], [math.inf, math.inf])
                artifact = self.artifact(products, *factors)
                self.assertNotIn("fma", _tags(artifact))
                self.assertIn(_WIDE_JSON[dtype], artifact["artifact"]["scalars"])
        x, y = jnp.asarray([2.0, 3.0]), jnp.asarray([1.0, 1.0])
        widened = self.verify(lambda p, q: lax.real(lax.mul(p, q, out_dtype=jnp.complex64)), x, y,
                              checks=COMPARE)
        self.assert_passes(widened)
        self.assertEqual(_outputs(widened)[0].tolist(), [2.0, 3.0])
        with jax.enable_x64(True):
            def converted(a, b, c, d):
                z, w = lax.complex(a, b), lax.complex(c, d)
                return [f(lax.mul(p, q, out_dtype=jnp.complex64)) for p, q in ((z, w), (z, c))
                        for f in (lax.real, lax.imag)]

            operands = [jnp.asarray(rng.standard_normal(32)) for _ in range(4)]
            rounded = [np.asarray(v, np.float32) for v in operands]
            check(converted, operands, "float32", expected(*rounded, "float32"))

    def test_complex_quotient_rounds_its_definition_once(self):
        # lax.div of a + ib by c + id is the definition, ((ac + bd) + (bc - ad) i) / (c^2 + d^2),
        # each part the exact part rounded once (_complex_reference), as the import computes it
        # in a format where the numerators and the denominator are exact and rounding the
        # quotient there first does not change its rounding to the parts' format (_WIDE_JSON).
        # XLA's CPU code divides by Smith's algorithm, whose words differ. Random pairs, eager and
        # under jax.jit, with a real dividend and a complex scalar divisor; quotients of operands
        # at the ends of the range, among them (2^127 + 2^127 i) / (2^127 + 2^127 i) = 1 + 0i,
        # where XLA gives NaN + 0i; the real part T / (1 + 2^-54) just below the overflow
        # threshold T of binary32, of (2^123 - 2^125 i) / (2^-5 + 2^-32 i), and T / (1 + 2^-114)
        # of binary64,
        # of (2^1019 - 2^1022 i) / (2^-5 + 2^-62 i), which are the greatest finite values where
        # rounding to binary64 or binary128 first gave inf, and real parts just above the
        # midpoint 2^-150, (2^50 + 2^-149) / (2^200 + 1), and 2^-1075, which are the least
        # subnormals where it gave 0; and at infinities, NaNs and signed zeros the formula's IEEE
        # values, without C's Annex G recovery: (1 - i) / 0 and (1 - i) / (inf + inf i) are
        # NaN + NaN i, and (inf + 2^600 i) / (1 + 2^600 i) is inf - inf i, with the exact
        # denominator 1 + 2^1200.
        import itertools
        import math

        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(29)

        def quotients(a, b, c, d):
            z, w = lax.complex(a, b), lax.complex(c, d)
            return [v for q in (z / w, a / w, z / (2.5 - 1.25j)) for v in (q.real, q.imag)]

        def expected(a, b, c, d, dtype):
            columns = [np.asarray(v, np.float64).ravel() for v in (a, b, c, d)]
            cases = [list(zip(*columns)), [(p, 0.0, r, s) for p, _, r, s in zip(*columns)],
                     [(p, q, 2.5, -1.25) for p, q, _, _ in zip(*columns)]]
            values = [[_complex_reference("div", *entry, dtype) for entry in case]
                      for case in cases]
            return [[part[k] for part in case] for case in values for k in (0, 1)]

        def check(function, operands, dtype):
            report = self.verify(function, *operands, checks=EXECUTE)
            self.assert_passes(report)
            got = [[float(v) for v in actual.double().ravel()] for actual in _outputs(report)]
            for actual, wanted in zip(got, expected(*operands, dtype), strict=True):
                _same(self, actual, wanted)
            return got

        rows = {"float32": [(2.0 ** 123, -(2.0 ** 125), 2.0 ** -5, 2.0 ** -32),
                            (2.0 ** -50, 2.0 ** -149, 2.0 ** 100, 1.0)],
                "float64": [(2.0 ** 1019, -(2.0 ** 1022), 2.0 ** -5, 2.0 ** -62),
                            (2.0 ** -575, 2.0 ** -1074, 2.0 ** 500, 1.0)]}
        reals = {"float32": [float(np.finfo(np.float32).max), 2.0 ** -149],
                 "float64": [float(np.finfo(np.float64).max), 2.0 ** -1074]}
        for dtype, top, bottom, edge in (("float32", 127, -149, 2.0 ** 100),
                                         ("float64", 1023, -1074, 2.0 ** 600)):
            with jax.enable_x64(dtype == "float64"):
                factors = [jnp.asarray(rng.standard_normal((4, 64)), dtype) for _ in range(4)]
                for mode, function in (("eager", quotients), ("jit", jax.jit(quotients))):
                    with self.subTest(dtype=dtype, mode=mode):
                        check(function, factors, dtype)
                with self.subTest(dtype=dtype, rounding=True):
                    got = check(quotients, [jnp.asarray(column, dtype)
                                            for column in zip(*rows[dtype])], dtype)
                    _same(self, got[0], reals[dtype])
                with self.subTest(dtype=dtype, edge=True):
                    ends = [2.0 ** top, 2.0 ** bottom, -(2.0 ** top), 1.5 * 2.0 ** (top - 1)]
                    combos = list(itertools.product(ends, repeat=4))
                    got = check(quotients, [jnp.asarray([combo[k] for combo in combos], dtype)
                                            for k in range(4)], dtype)
                    _same(self, [got[0][0], got[1][0]], [1.0, 0.0])
                inf, nan = math.inf, math.nan
                special = [complex(inf, 0), 0j, complex(-0.0, 0), complex(nan, 1),
                           complex(1, inf), complex(-0.0, -0.0), complex(inf, inf),
                           complex(1, -1), complex(-inf, 2), complex(0, -0.0),
                           complex(nan, nan), complex(2, 0), complex(2, 1), complex(inf, edge),
                           complex(1, edge)]
                pairs = [(z, w) for z in special for w in special]
                parts = [jnp.asarray([getattr(pair[k], part) for pair in pairs], dtype)
                         for k in (0, 1) for part in ("real", "imag")]
                with self.subTest(dtype=dtype, special=True):
                    got = check(quotients, parts, dtype)
                    for w in (0j, complex(inf, inf)):
                        index = pairs.index((complex(1, -1), w))
                        self.assertTrue(math.isnan(got[0][index]) and math.isnan(got[1][index]))
                    index = pairs.index((complex(inf, edge), complex(1, edge)))
                    _same(self, [got[0][index], got[1][index]], [math.inf, -math.inf])
                artifact = self.artifact(quotients, *factors)
                self.assertNotIn("fma", _tags(artifact))
                self.assertIn(_WIDE_JSON[dtype], artifact["artifact"]["scalars"])

    def test_complex_sign_rounds_its_definition_once(self):
        # lax.sign of a complex z = a + ib is the phase z / |z| that JAX documents,
        # a / sqrt(a^2 + b^2) and b / sqrt(a^2 + b^2), each part the exact part rounded once
        # (_phase_reference), and z itself at zero, as the import computes it in a format of 734
        # or 5330 significand bits (_PHASE_JSON). XLA's CPU code is a unit in the last place away
        # at most random values and returns z at the least subnormal pair and 0 at the greatest
        # finite one, so its words are not the reference. Random values, eager and under jax.jit,
        # and across the exponent range; signed zeros, infinities and NaN, where
        # inf + 1i gives NaN + 0i; parts where a^2 + b^2 underflows or overflows the parts'
        # format; and phases so near a midpoint of the format that computing them in the next
        # IEEE format, binary64 for binary32 parts and binary128 for binary64 parts, and rounding
        # again rounds the other way: the real parts of 14212668 + 10345879i in binary32 and
        # 4307224548994623 + 4322286462541681i in binary64, convergents of the slope b / a at
        # which the phase is a midpoint.
        import math

        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(43)
        inf, nan = math.inf, math.nan

        def phases(a, b):
            z = lax.sign(lax.complex(a, b))
            return [z.real, z.imag]

        def check(function, a, b, dtype):
            report = self.verify(function, a, b, checks=EXECUTE)
            self.assert_passes(report)
            got = [[float(v) for v in actual.double().ravel()] for actual in _outputs(report)]
            parts = [np.asarray(v, np.float64).ravel().tolist() for v in (a, b)]
            wanted = [_phase_reference(p, q, dtype) for p, q in zip(*parts)]
            for k in (0, 1):
                _same(self, got[k], [w[k] for w in wanted])
            return got

        witnesses = {"float32": ((14212668.0, 10345879.0), 0.8084818720817566),
                     "float64": ((4307224548994623.0, 4322286462541681.0), 0.7058715246911654)}
        for dtype in ("float32", "float64"):
            with jax.enable_x64(dtype == "float64"):
                precision, emin, emax = FORMATS[dtype]
                tiny, top = 2.0 ** (emin - precision + 1), float(np.finfo(dtype).max)
                a, b = [jnp.asarray(v, dtype) for v in rng.standard_normal((2, 4, 16))]
                for mode, function in (("eager", phases), ("jit", jax.jit(phases))):
                    with self.subTest(dtype=dtype, mode=mode):
                        check(function, a, b, dtype)
                scales = rng.integers(emin - 30, emax - 2, (2, 64))
                with self.subTest(dtype=dtype, range=True):
                    check(phases, *[jnp.asarray(v, dtype) for v in
                                    np.ldexp(rng.standard_normal((2, 64)), scales)], dtype)
                rows = [(0.0, 0.0), (-0.0, 0.0), (0.0, -0.0), (-0.0, -0.0), (inf, 1.0),
                        (1.0, -inf), (inf, inf), (nan, 1.0), (nan, inf), (-0.0, 5.0),
                        (3.0, 4.0), (-tiny, 0.0), (tiny, tiny), (top, top), (top, tiny),
                        witnesses[dtype][0]]
                with self.subTest(dtype=dtype, special=True):
                    got = check(phases, *[jnp.asarray(v, dtype) for v in zip(*rows)], dtype)
                    _same(self, [got[0][1], got[1][1], got[0][4], got[1][4], got[0][11]],
                          [-0.0, 0.0, nan, 0.0, -1.0])
                    self.assertEqual(got[0][-1], witnesses[dtype][1])
                artifact = self.artifact(phases, a, b)
                self.assertIn(_PHASE_JSON[dtype], artifact["artifact"]["scalars"])

    def test_complex_sqrt_rounds_its_definition_once(self):
        # lax.sqrt of a complex z = a + ib is the principal square root, T + i copysign(U, b)
        # where a is not negative and U + i copysign(T, b) where it is, for
        # T = sqrt((|z| + |a|) / 2) and U = |b| / (2 T), each part the exact part rounded once
        # (_root_reference), with C11's special values (G.6.4.2), as FloatLib's ExecComplex.sqrt
        # computes it (Op.complex). XLA's CPU code gave the root
        # of +0 for a -0 or subnormal part, 0 for a subnormal part of the root, and a unit or two
        # in the last place away at about a quarter of the parts of 500 random pairs, so its words
        # are not the reference. Random values, eager and under jax.jit, and across the exponent
        # range; signed zeros,
        # infinities and NaN, where -inf + NaN i gives NaN + inf i and NaN - inf i gives
        # inf - inf i; exact roots, 3 + 4i and -3 - 4i; the negative axis, where -4 - 0i gives
        # 0 - 2i; and parts where |z| overflows or underflows the parts' format or where the
        # root's smaller part is subnormal.
        import math

        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(59)
        inf, nan = math.inf, math.nan

        def roots(a, b):
            z = lax.sqrt(lax.complex(a, b))
            return [z.real, z.imag]

        def check(function, a, b, dtype):
            report = self.verify(function, a, b, checks=EXECUTE)
            self.assert_passes(report)
            got = [[float(v) for v in actual.double().ravel()] for actual in _outputs(report)]
            parts = [np.asarray(v, np.float64).ravel().tolist() for v in (a, b)]
            wanted = [_root_reference(p, q, dtype) for p, q in zip(*parts)]
            for k in (0, 1):
                _same(self, got[k], [w[k] for w in wanted])
            return got

        for dtype in ("float32", "float64"):
            with jax.enable_x64(dtype == "float64"):
                precision, emin, emax = FORMATS[dtype]
                tiny, top = 2.0 ** (emin - precision + 1), float(np.finfo(dtype).max)
                a, b = [jnp.asarray(v, dtype) for v in rng.standard_normal((2, 4, 16))]
                for mode, function in (("eager", roots), ("jit", jax.jit(roots))):
                    with self.subTest(dtype=dtype, mode=mode):
                        check(function, a, b, dtype)
                scales = rng.integers(emin - 30, emax - 2, (2, 64))
                with self.subTest(dtype=dtype, range=True):
                    check(roots, *[jnp.asarray(v, dtype) for v in
                                   np.ldexp(rng.standard_normal((2, 64)), scales)], dtype)
                rows = [(0.0, 0.0), (-0.0, 0.0), (0.0, -0.0), (-0.0, -0.0), (-4.0, -0.0),
                        (4.0, -0.0), (3.0, 4.0), (-3.0, -4.0), (inf, 1.0), (inf, -0.0),
                        (-inf, 1.0), (-inf, -1.0), (1.0, inf), (nan, -inf), (inf, nan),
                        (-inf, nan), (nan, 1.0), (1.0, nan), (nan, 0.0), (top, top),
                        (-top, -top), (-top, tiny), (top, -tiny), (tiny, tiny), (-tiny, 0.0),
                        (-0.0, tiny), (2.0, 0.0)]
                large, small = 1.5 * 4.0 ** (emax // 4), 2.0 ** (emin + emax // 4)
                rows += [(large, small), (-large, -small)]
                with self.subTest(dtype=dtype, special=True):
                    got = check(roots, *[jnp.asarray(v, dtype) for v in zip(*rows)], dtype)
                    self.assertTrue(0 < got[1][27] < 2.0 ** emin and 0 < got[0][28] < 2.0 ** emin)
                    _same(self, [got[0][3], got[1][3], got[0][4], got[1][4], got[0][6], got[1][6],
                                 got[0][7], got[1][7], got[0][13], got[1][13], got[0][15],
                                 got[1][15]],
                          [0.0, -0.0, 0.0, -2.0, 2.0, 1.0, 1.0, -2.0, inf, -inf, nan, inf])
                names = {e.primitive.name for e in jax.make_jaxpr(roots)(a, b).jaxpr.eqns}
                self.assertIn("sqrt", names)
                artifact = self.artifact(roots, a, b)
                self.assertIn("complex", _tags(artifact))

    def test_complex_rsqrt_rounds_each_part_once(self):
        # lax.rsqrt of a complex z is 1 / √z, as JAX documents, each part the exact part rounded
        # once (_reciprocal_root_reference), FloatLib's ExecComplex.rsqrt (Op.complex), not
        # 1 over the rounded root. Random values, eager and under jax.jit, and across
        # the exponent range; 4 + 0i, 3 + 4i and -4 -+ 0i, where rsqrt(-4 - 0i) is 0 + 0.5i;
        # infinities, where 1 / √z tends to 0; zeros, where it is C's (1 + 0i) / (0 + 0i),
        # inf + NaN i, as FloatLib's ExecComplex.rsqrt is; and a NaN.
        import math

        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(61)
        inf, nan = math.inf, math.nan

        def reciprocals(a, b):
            z = lax.rsqrt(lax.complex(a, b))
            return [z.real, z.imag]

        def check(function, a, b, dtype):
            report = self.verify(function, a, b, checks=EXECUTE)
            self.assert_passes(report)
            got = [[float(v) for v in actual.double().ravel()] for actual in _outputs(report)]
            parts = [np.asarray(v, np.float64).ravel().tolist() for v in (a, b)]
            wanted = [_reciprocal_root_reference(p, q, dtype) for p, q in zip(*parts)]
            for k in (0, 1):
                _same(self, got[k], [w[k] for w in wanted])
            return got

        for dtype in ("float32", "float64"):
            with jax.enable_x64(dtype == "float64"):
                _, emin, emax = FORMATS[dtype]
                a, b = [jnp.asarray(v, dtype) for v in rng.standard_normal((2, 4, 16))]
                for mode, function in (("eager", reciprocals), ("jit", jax.jit(reciprocals))):
                    with self.subTest(dtype=dtype, mode=mode):
                        check(function, a, b, dtype)
                scales = rng.integers(emin - 30, emax - 2, (2, 64))
                with self.subTest(dtype=dtype, range=True):
                    check(reciprocals, *[jnp.asarray(v, dtype) for v in
                                         np.ldexp(rng.standard_normal((2, 64)), scales)], dtype)
                rows = [(4.0, 0.0), (3.0, 4.0), (-4.0, -0.0), (-4.0, 0.0), (0.0, 0.0),
                        (-0.0, -0.0), (inf, 0.0), (-inf, 1.0), (1.0, inf), (nan, 1.0)]
                with self.subTest(dtype=dtype, special=True):
                    got = check(reciprocals, *[jnp.asarray(v, dtype) for v in zip(*rows)], dtype)
                    _same(self, [got[k][i] for i in range(10) if i != 1 for k in (0, 1)],
                          [0.5, -0.0, 0.0, 0.5, 0.0, -0.5, inf, nan, inf, nan, 0.0, -0.0, 0.0,
                           -0.0, 0.0, -0.0, nan, nan])
                names = {e.primitive.name for e in jax.make_jaxpr(reciprocals)(a, b).jaxpr.eqns}
                self.assertIn("rsqrt", names)
                artifact = self.artifact(reciprocals, a, b)
                self.assertIn("complex", _tags(artifact))

    def test_complex_functions_round_each_part_once(self):
        # lax.exp, expm1, log, log1p, sin, cos, tan, sinh, cosh, tanh, asin, acos, atan, asinh,
        # acosh and atanh of a complex z are Op.complex, FloatLib's ExecComplex function of the
        # parts, each part at a finite z the exact part rounded once (sqrt and rsqrt above). Random
        # values off the axes, eager and under jax.jit, against mpmath's principal values
        # (_function_reference); and the branch cuts, where the sign of a zero part selects the
        # side (C23 7.3.3): log(-1 ± 0i) and log1p(-2 ± 0i) are 0 ± πi, asin(2 ± 0i) is
        # π/2 ± i acosh(2), acos(2 ± 0i) is 0 ∓ i acosh(2), acosh(-2 ± 0i) is acosh(2) ± πi,
        # asinh(±0 + 2i) is ±acosh(2) + iπ/2, atan(±0 + 2i) is ±π/2 + i log(3)/2 and
        # atanh(2 ± 0i) is log(3)/2 ± iπ/2, with π, π/2, acosh(2) and log(3)/2 rounded once.
        # XLA's CPU words are not the reference. At 500 random pairs like these its parts
        # differed from the exact parts rounded once at 2 to 449 of them for each function and
        # part, by at most 7 units in the last place except in expm1 (37 and 159 in binary32's
        # real and imaginary parts, 58 and 26236 in binary64's), binary32's tan (17 and 15) and
        # the real part of sinh and imaginary part of cosh (484276 and 605736 in binary32, 62174
        # and 69129 in binary64). On the cuts its asin, acos, acosh and asinh gave the value of
        # the +0 side at -0 too.
        import mpmath
        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(67)
        names = ("exp", "expm1", "log", "log1p", "sin", "cos", "tan", "sinh", "cosh", "tanh",
                 "asin", "acos", "atan", "asinh", "acosh", "atanh")

        def functions(a, b):
            z = lax.complex(a, b)
            return [part for name in names for w in [getattr(lax, name)(z)]
                    for part in (w.real, w.imag)]

        def check(function, a, b):
            report = self.verify(function, a, b, checks=EXECUTE)
            self.assert_passes(report)
            return [[float(v) for v in actual.double().ravel()] for actual in _outputs(report)]

        for dtype in ("float32", "float64"):
            with jax.enable_x64(dtype == "float64"):
                a, b = [jnp.asarray(v, dtype) for v in
                        np.ldexp(rng.standard_normal((2, 4, 6)), rng.integers(-12, 3, (2, 4, 6)))]
                parts = [np.asarray(v, np.float64).ravel().tolist() for v in (a, b)]
                wanted = [[_function_reference(name, p, q, dtype) for p, q in zip(*parts)]
                          for name in names]
                for mode, function in (("eager", functions), ("jit", jax.jit(functions))):
                    with self.subTest(dtype=dtype, mode=mode):
                        got = check(function, a, b)
                        for k, name in enumerate(names):
                            for part in (0, 1):
                                _same(self, got[2 * k + part], [w[part] for w in wanted[k]],
                                      name)
                with mpmath.workprec(400):
                    pi, half_pi, acosh2, half_log3 = [
                        _exact_rounded(v, dtype) for v in
                        (mpmath.pi, mpmath.pi / 2, mpmath.acosh(2), mpmath.log(3) / 2)]
                # Each function's points on its cut and the parts there.
                cuts = {"log": ([(-1.0, 0.0), (-1.0, -0.0)], [(0.0, pi), (0.0, -pi)]),
                        "log1p": ([(-2.0, 0.0), (-2.0, -0.0)], [(0.0, pi), (0.0, -pi)]),
                        "asin": ([(2.0, 0.0), (2.0, -0.0)],
                                 [(half_pi, acosh2), (half_pi, -acosh2)]),
                        "acos": ([(2.0, 0.0), (2.0, -0.0)], [(0.0, -acosh2), (0.0, acosh2)]),
                        "acosh": ([(-2.0, 0.0), (-2.0, -0.0)], [(acosh2, pi), (acosh2, -pi)]),
                        "asinh": ([(0.0, 2.0), (-0.0, 2.0)],
                                  [(acosh2, half_pi), (-acosh2, half_pi)]),
                        "atan": ([(0.0, 2.0), (-0.0, 2.0)],
                                 [(half_pi, half_log3), (-half_pi, half_log3)]),
                        "atanh": ([(2.0, 0.0), (2.0, -0.0)],
                                  [(half_log3, half_pi), (half_log3, -half_pi)])}
                for name, (points, values) in cuts.items():
                    with self.subTest(dtype=dtype, cut=name):

                        def on_cut(a, b, name=name):
                            w = getattr(lax, name)(lax.complex(a, b))
                            return [w.real, w.imag]

                        got = check(on_cut, *[jnp.asarray(v, dtype) for v in zip(*points)])
                        for part in (0, 1):
                            _same(self, got[part], [v[part] for v in values], name)
                artifact = self.artifact(functions, a, b)
                self.assertEqual(
                    [record["fields"][0] for record in _records(artifact)
                     if record["tag"] == "complex"],
                    [{"name": name} for name in names])

    def test_complex_powers_multiply_in_jax_order(self):
        # lax.integer_pow of a complex z multiplies in the order of JAX's lax._integer_pow,
        # square-and-multiply and then 1 / product for a negative exponent (lax.integer_pow
        # documents O(log2 y) multiplications), each product the complex product rounded once
        # (_complex_reference) and the reciprocal that quotient; z^0 is 1 + 0i, at a NaN too, and
        # lax.square is z^2 rounded once. Random values, eager and under jax.jit, for exponents
        # -6 to 13; signed zeros, infinities and NaN, where 0^-1 is NaN + NaN i; and powers that
        # overflow or underflow the parts' format: (2^64 + 2^64 i)^2 is 0 + inf i in binary32, as
        # is (2^512 + 2^512 i)^2 in binary64.
        import math

        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(47)
        inf, nan = math.inf, math.nan
        exponents = [2, 0, 1, 3, 5, 6, 7, 13, -1, -2, -3, -6]

        def powers(a, b):
            z = lax.complex(a, b)
            return [v for p in [lax.square(z), *[lax.integer_pow(z, n) for n in exponents[1:]]]
                    for v in (p.real, p.imag)]

        def power(a, b, n, dtype):
            if n == 0:
                return [1.0, 0.0]
            base, result, m = [a, b], None, abs(n)
            while m:
                if m & 1:
                    result = base if result is None else \
                        _complex_reference("mul", *result, *base, dtype)
                m >>= 1
                if m:
                    base = _complex_reference("mul", *base, *base, dtype)
            return _complex_reference("div", 1.0, 0.0, *result, dtype) if n < 0 else result

        def check(function, a, b, dtype):
            report = self.verify(function, a, b, checks=EXECUTE)
            self.assert_passes(report)
            got = [[float(v) for v in actual.double().ravel()] for actual in _outputs(report)]
            parts = [np.asarray(v, np.float64).ravel().tolist() for v in (a, b)]
            for k, n in enumerate(exponents):
                wanted = [power(p, q, n, dtype) for p, q in zip(*parts)]
                for j in (0, 1):
                    _same(self, got[2 * k + j], [w[j] for w in wanted], (n, j))
            return got

        for dtype, edge in (("float32", 2.0 ** 64), ("float64", 2.0 ** 512)):
            with jax.enable_x64(dtype == "float64"):
                precision, emin, _ = FORMATS[dtype]
                a, b = [jnp.asarray(v, dtype) for v in rng.standard_normal((2, 4, 16))]
                for mode, function in (("eager", powers), ("jit", jax.jit(powers))):
                    with self.subTest(dtype=dtype, mode=mode):
                        check(function, a, b, dtype)
                rows = [(0.0, 0.0), (-0.0, 0.0), (0.0, -0.0), (-0.0, -0.0), (nan, 0.0),
                        (inf, 0.0), (inf, inf), (1.0, -inf), (-inf, 1.0), (1.0, 1.0), (0.0, 1.0),
                        (edge, edge), (2.0 ** (emin // 2), -(2.0 ** (emin // 2))),
                        (2.0 ** (emin - precision + 1), 1.0)]
                with self.subTest(dtype=dtype, special=True):
                    got = check(powers, *[jnp.asarray(v, dtype) for v in zip(*rows)], dtype)
                    _same(self, [got[0][11], got[1][11], got[2][4], got[3][4], got[16][0]],
                          [0.0, inf, 1.0, 0.0, nan])
                names = {e.primitive.name for e in jax.make_jaxpr(powers)(a, b).jaxpr.eqns}
                self.assertLessEqual({"square", "integer_pow"}, names)
                artifact = self.artifact(powers, a, b)
                self.assertIn(_WIDE_JSON[dtype], artifact["artifact"]["scalars"])

    def test_complex_pow_rounds_each_part_once(self):
        # lax.pow of a complex base is Op.complexPow, FloatLib's ExecComplex.pow, C23's cpow
        # exp(w log z), and lax.exp2 of a complex z is the power of the base 2 + 0i, 2^z as JAX
        # documents it. At random bases and exponents, eager and under jax.jit, at an integer
        # exponent array, which JAX converts to the complex type, at a scalar literal exponent
        # and at a real exponent array, each part is mpmath's part of the principal value rounded
        # once (_exact_rounded). On the cut of log the sign of the base's zero imaginary part
        # selects the side: (-4 ± 0i)^0.5 is 0 ± 2i. (1 + i)^2 is 2i; z^0 is 1 ± 0i, at a NaN
        # too, with the sign of the base's imaginary part; 0^w is +0 + 0i, inf + NaN i or
        # NaN + NaN i as the real part of w is positive, negative or zero; and an infinite part
        # gives NaN + NaN i. XLA's CPU words are not the reference: in binary32 it gave
        # NaN + NaN i at a zero base, inf + 0i for inf^1, -4.37e-8 + i for (-1 + 0i)^0.5,
        # -8.74e-8 + 1.9999999i for (1 + i)^2 and parts of (1 + 2i)^(3 - i) 4 and 2 units in
        # the last place from the exact parts rounded once.
        import math

        import mpmath
        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(71)
        inf, nan = math.inf, math.nan

        def powers(a, b, c, d, n, e):
            z, w = lax.complex(a, b), lax.complex(c, d)
            return [part for p in (lax.pow(z, w), lax.pow(z, n), z ** 2.5, z ** e, lax.exp2(w))
                    for part in (p.real, p.imag)]

        def power(a, b, c, d):
            p = lax.pow(lax.complex(a, b), lax.complex(c, d))
            return [p.real, p.imag]

        def reference(z, w, dtype):
            # mpmath's parts of the principal value of z^w rounded once, and 1 ± 0i at w = 0.
            if w == 0:
                return [1.0, math.copysign(0.0, z.imag)]
            with mpmath.workprec(400):
                value = mpmath.power(mpmath.mpc(z), mpmath.mpc(w))
                return [_exact_rounded(v, dtype) for v in (value.real, value.imag)]

        def check(function, args, wanted):
            report = self.verify(function, *args, checks=EXECUTE)
            self.assert_passes(report)
            got = [[float(v) for v in actual.double().ravel()] for actual in _outputs(report)]
            self.assertEqual(len(got), 2 * len(wanted))
            for k, values in enumerate(wanted):
                for part in (0, 1):
                    _same(self, got[2 * k + part], [v[part] for v in values], (k, part))

        for dtype in ("float32", "float64"):
            with jax.enable_x64(dtype == "float64"):
                a, b = np.ldexp(rng.standard_normal((2, 4, 6)), rng.integers(-2, 3, (2, 4, 6)))
                c, d, e = rng.standard_normal((3, 4, 6))
                args = [*[jnp.asarray(v, dtype) for v in (a, b, c, d)],
                        jnp.asarray(rng.integers(-3, 5, (4, 6)), jnp.int32), jnp.asarray(e, dtype)]
                a, b, c, d, n, e = [np.asarray(v, np.float64).ravel().tolist() for v in args]
                z, w = [complex(p, q) for p, q in zip(a, b)], [complex(p, q) for p, q in zip(c, d)]
                wanted = [[reference(x, y, dtype) for x, y in zip(z, w)],
                          [reference(x, complex(m), dtype) for x, m in zip(z, n)],
                          [reference(x, 2.5, dtype) for x in z],
                          [reference(x, complex(m), dtype) for x, m in zip(z, e)],
                          [reference(2, y, dtype) for y in w]]
                for mode, function in (("eager", powers), ("jit", jax.jit(powers))):
                    with self.subTest(dtype=dtype, mode=mode):
                        check(function, args, wanted)
                # Each base, exponent and power.
                rows = [((-4.0, 0.0), (0.5, 0.0), (0.0, 2.0)),
                        ((-4.0, -0.0), (0.5, 0.0), (0.0, -2.0)),
                        ((-1.0, 0.0), (0.5, 0.0), (0.0, 1.0)), ((1.0, 1.0), (2.0, 0.0), (0.0, 2.0)),
                        ((nan, 0.0), (0.0, 0.0), (1.0, 0.0)),
                        ((2.0, -0.0), (0.0, 0.0), (1.0, -0.0)),
                        ((0.0, 0.0), (2.0, 1.0), (0.0, 0.0)), ((0.0, 0.0), (-1.0, 0.0), (inf, nan)),
                        ((0.0, 0.0), (0.0, 1.0), (nan, nan)), ((inf, 0.0), (1.0, 0.0), (nan, nan)),
                        ((1.0, 2.0), (3.0, -1.0), reference(1 + 2j, 3 - 1j, dtype))]
                with self.subTest(dtype=dtype, special=True):
                    check(power, [jnp.asarray(v, dtype) for v in
                                  zip(*[(*z, *w) for z, w, _ in rows])], [[r[2] for r in rows]])
                pows = [e for e in jax.make_jaxpr(powers)(*args).jaxpr.eqns
                        if e.primitive.name == "pow"]
                pair = {"float32": "complex64", "float64": "complex128"}[dtype]
                self.assertEqual([str(e.invars[1].aval.dtype) for e in pows],
                                 [pair, "int32", pair, pair])
                artifact = self.artifact(powers, *args)
                self.assertEqual([record["fields"] for record in _records(artifact)
                                  if record["tag"] == "complex_pow"], [[{"shape": [4, 6]}]] * 5)

    def test_complex_max_and_min_compare_lexicographically(self):
        # lax.max and lax.min of complex operands compare the (real, imaginary) pairs
        # lexicographically, as JAX documents them, and where a NaN part decides the comparison,
        # which JAX does not document, as its lowering does (jax._src.interpreters.mlir.
        # _minmax_hlo): x where the real parts compare so, or are equal and the imaginary parts
        # do, by IEEE comparisons, and y elsewhere. So y where the real parts are unordered or
        # equal beside unordered imaginary parts, x for max(2 + NaN i, 1 + 0i), and 0 + 0i for
        # max(-0 + 0i, 0 + 0i). Parts drawn from a few values, so that ties are common, with
        # signed zeros, infinities and NaN, eager and under jax.jit, against that reference and
        # against XLA's words, which are the same selection.
        import math

        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(53)
        inf, nan = math.inf, math.nan

        def extremes(a, b, c, d):
            z, w = lax.complex(a, b), lax.complex(c, d)
            return [f(p) for p in (lax.max(z, w), lax.min(z, w)) for f in (lax.real, lax.imag)]

        def reference(a, b, c, d, order):
            return [a, b] if (order(b, d) if a == c else order(a, c)) else [c, d]

        orders = (lambda x, y: x > y, lambda x, y: x < y)
        rows = [(-0.0, 0.0, 0.0, 0.0), (0.0, 0.0, -0.0, 0.0), (nan, 0.0, 1.0, 0.0),
                (1.0, 0.0, nan, 0.0), (1.0, nan, 1.0, 2.0), (inf, -1.0, inf, 1.0)]
        values = [-inf, -1.5, -0.0, 0.0, 1.5, 2.0, inf, nan]
        drawn = list(zip(*rng.choice(values, (4, 250)).tolist()))
        for dtype in ("float32", "float64"):
            with jax.enable_x64(dtype == "float64"):
                parts = [jnp.asarray(column, dtype) for column in zip(*rows, *drawn)]
                wanted = [[reference(*entry, order)[k] for entry in [*rows, *drawn]]
                          for order in orders for k in (0, 1)]
                for mode, function in (("eager", extremes), ("jit", jax.jit(extremes))):
                    with self.subTest(dtype=dtype, mode=mode):
                        report = self.verify(function, *parts, checks=EXECUTE)
                        self.assert_passes(report)
                        got = [[float(v) for v in actual.double().ravel()]
                               for actual in _outputs(report)]
                        xla = [np.asarray(v, np.float64).tolist() for v in function(*parts)]
                        for actual, expected, framework in zip(got, wanted, xla, strict=True):
                            _same(self, actual, expected)
                            _same(self, framework, expected)
                        _same(self, [got[0][0], got[0][1], got[0][2], got[0][3], got[2][2]],
                              [0.0, -0.0, 1.0, nan, 1.0])

    def test_complex_sort_orders_pairs_as_jax_does(self):
        # lax.sort, stable, with complex keys: the pairs in the lexicographic order of JAX's
        # comparator (jax._src.lax.lax._sort_lt_comparator), of the parts with -0 taken as +0 and
        # every NaN as one that follows +inf, equal entries in their order and every word kept.
        # Parts drawn from a few values, so that ties are common, with signed zeros, infinities
        # and NaN. One operand, as jnp.sort calls it, along each axis of a 3-d operand, eager and
        # under jax.jit; and several, eager: jnp.argsort, lax.sort_key_val with a complex payload
        # by a complex and by a real key, lax.sort by two complex keys and the JVP of jnp.sort.
        # Each against that reference and against XLA's words, which are the same sort. An
        # unstable sort is refused.
        import math

        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(59)
        inf, nan = math.inf, math.nan

        def key(parts):
            return tuple(v for p in parts for v in ((1, 0.0) if math.isnan(p) else (0, p)))

        def order(keys, axis):
            # The stable order of each fiber along axis by the parts of the keys, each the tuple
            # of a complex key's real and imaginary parts or of a real key alone.
            parts = [np.moveaxis(part, axis, -1) for k in keys for part in k]
            result = np.empty(parts[0].shape, np.int64)
            for c in np.ndindex(parts[0].shape[:-1]):
                result[c] = sorted(range(parts[0].shape[-1]),
                                   key=lambda k: key([float(part[c][k]) for part in parts]))
            return np.moveaxis(result, -1, axis)

        def taken(at, axis, *parts):
            return [np.take_along_axis(part, at, axis).ravel().tolist() for part in parts]

        def check(function, inputs, wanted, **subtest):
            with self.subTest(**subtest):
                report = self.verify(function, *map(jnp.asarray, inputs), checks=EXECUTE)
                self.assert_passes(report)
                got = [[float(v) for v in actual.double().ravel()] for actual in _outputs(report)]
                xla = [np.asarray(v, np.float64).ravel().tolist() for v in function(*inputs)]
                for actual, expected, framework in zip(got, wanted, xla, strict=True):
                    _same(self, actual, expected)
                    _same(self, framework, expected)

        def several(a, b, c, d):
            z, w = lax.complex(a, b), lax.complex(c, d)
            s, t = lax.sort_key_val(z, w, dimension=2)
            u, v = lax.sort_key_val(c, z, dimension=0)
            p, q, r = lax.sort((z, w, a), dimension=1, num_keys=2)
            tangent = jax.jvp(lambda z: jnp.sort(z, axis=1), (z,), (w,))[1]
            return [jnp.argsort(z, axis=2), s.real, s.imag, t.real, t.imag, u, v.real, v.imag,
                    p.real, p.imag, q.real, q.imag, r, tangent.real, tangent.imag]

        values = [-inf, -1.5, -0.0, 0.0, 1.5, inf, nan]
        for dtype in ("float32", "float64"):
            with jax.enable_x64(dtype == "float64"):
                a, b, c, d = [np.asarray(rng.choice(values, (3, 5, 8)), dtype) for _ in range(4)]
                for axis in (0, 1, 2):
                    def sort(a, b, axis=axis):
                        z = jnp.sort(lax.complex(a, b), axis=axis)
                        return [z.real, z.imag]

                    wanted = taken(order([(a, b)], axis), axis, a, b)
                    for mode, function in (("eager", sort), ("jit", jax.jit(sort))):
                        check(function, (a, b), wanted, dtype=dtype, axis=axis, mode=mode)
                last, first, both, jvp = (order([(a, b)], 2), order([(c,)], 0),
                                          order([(a, b), (c, d)], 1), order([(a, b)], 1))
                check(several, (a, b, c, d),
                      [last.ravel().tolist(), *taken(last, 2, a, b, c, d),
                       *taken(first, 0, c, a, b), *taken(both, 1, a, b, c, d, a),
                       *taken(jvp, 1, c, d)], dtype=dtype, case="several")
        x = jnp.asarray(rng.choice(values, (2, 4)), jnp.float32)
        self.assert_refused(self.verify(lambda x: lax.sort(lax.complex(x, x), is_stable=False).real,
                                        x, checks=("structure",)),
                            "sort: only a stable sort")

    def test_complex_folds_start_from_the_first_entry(self):
        # lax.reduce_prod, reduce_max and reduce_min, as jnp.prod, jnp.max and jnp.min call them,
        # and lax.cumprod, cummax and cummin of complex values: the left fold from the first entry,
        # the accumulator first, of the complex product rounded once (_complex_reference) or of
        # lax.max or lax.min (x where the real parts compare so, or are equal and the imaginary
        # parts do, by IEEE comparisons, y elsewhere), over the reduced coordinates in row-major
        # order, last to first for a reverse scan; over no entry 1 + 0i, -inf + 0i and inf + 0i.
        # Random values over each axis of a 3-d operand, two and all three, eager and under jax.jit;
        # and parts drawn from small integers, signed zeros, infinities and NaN, where the start and
        # the order decide. XLA's words there are those of the left fold from JAX's initial value
        # instead, a reverse scan folding each suffix first to last, along 2 to 32 entries, here 3
        # to 24; it returns a lone entry, and past 32 its extrema follow
        # no fold. So (inf + 0i)(1 + 0i) is inf + NaN i, which XLA gives as NaN + NaN i, the maximum
        # of -inf - i and -inf - i is -inf - i, which XLA gives as -inf + 0i, and the minimum of inf
        # + i and inf + i is inf + i, which XLA gives as inf + 0i.
        import math

        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(71)
        inf, nan = math.inf, math.nan
        reductions = [(0,), (1,), (2,), (0, 2), (0, 1, 2)]
        scans = [(0, False), (1, True), (2, False), (2, True)]
        kinds = {"prod": (jnp.prod, lax.cumprod), "max": (jnp.max, lax.cummax),
                 "min": (jnp.min, lax.cummin)}
        seeds = {"prod": [1.0, 0.0], "max": [-inf, 0.0], "min": [inf, 0.0]}

        def folds(a, b):
            z = lax.complex(a, b)
            results = [reduce(z, axis=axes) for reduce, _ in kinds.values() for axes in reductions]
            results += [scan(z, axis=axis, reverse=reverse) for reduce, scan in kinds.values()
                        for axis, reverse in scans]
            return [v for p in results for v in (p.real, p.imag)]

        def step(z, w, kind, dtype):
            if kind == "prod":
                return _complex_reference("mul", *z, *w, dtype)
            (a, b), (c, d) = z, w
            order = (lambda p, q: p > q) if kind == "max" else (lambda p, q: p < q)
            return list(z) if (order(b, d) if a == c else order(a, c)) else list(w)

        def fold(entries, kind, dtype, xla=False):
            # From the first entry, or as XLA's CPU code folds: from the seed but for one entry.
            entries = [list(w) for w in entries]
            if not entries:
                return seeds[kind]
            z, rest = (seeds[kind], entries) if xla else (entries[0], entries[1:])
            for w in rest:
                z = step(z, w, kind, dtype)
            return z

        def reference(a, b, dtype, xla=False):
            reduced, scanned = [], []
            for kind in kinds:
                for axes in reductions:
                    kept = [d for d in range(a.ndim) if d not in axes]
                    lead = [a.shape[d] for d in kept]
                    p, q = [np.transpose(x, kept + list(axes)).reshape(lead + [-1]) for x in (a, b)]
                    seeded = xla and p.shape[-1] > 1
                    zs = [fold(zip(p[c].tolist(), q[c].tolist()), kind, dtype, seeded)
                          for c in np.ndindex(*lead)]
                    reduced += [[z[0] for z in zs], [z[1] for z in zs]]
                for axis, reverse in scans:
                    p, q = np.moveaxis(a, axis, -1), np.moveaxis(b, axis, -1)
                    parts = [np.empty(p.shape), np.empty(p.shape)]
                    for c in np.ndindex(*p.shape[:-1]):
                        entries = list(zip(p[c].tolist(), q[c].tolist()))
                        for k in range(len(entries)):
                            run = (entries[:k + 1] if not reverse else entries[k:] if xla
                                   else entries[k:][::-1])
                            parts[0][c + (k,)], parts[1][c + (k,)] = fold(run, kind, dtype, xla)
                    scanned += [np.moveaxis(x, -1, axis).ravel().tolist() for x in parts]
            return reduced + scanned

        drawn = [-inf, -2.0, -1.0, -0.0, 0.0, 1.0, 2.0, inf, nan]
        # Every output but the reductions over all 144 entries.
        compared = [k for k in range(54) if k not in (8, 9, 18, 19, 28, 29)]
        for dtype in ("float32", "float64"):
            with jax.enable_x64(dtype == "float64"):
                normal = [np.asarray(v, dtype) for v in rng.standard_normal((2, 3, 4, 5))]
                special = [np.asarray(rng.choice(drawn, (3, 6, 8)), dtype) for _ in range(2)]
                for source, (a, b) in (("normal", normal), ("special", special)):
                    wanted = reference(a, b, dtype)
                    for mode, function in (("eager", folds), ("jit", jax.jit(folds))):
                        with self.subTest(dtype=dtype, source=source, mode=mode):
                            report = self.verify(function, jnp.asarray(a), jnp.asarray(b),
                                                 checks=EXECUTE)
                            self.assert_passes(report)
                            got = [[float(v) for v in actual.double().ravel()]
                                   for actual in _outputs(report)]
                            for k, (actual, expected) in enumerate(zip(got, wanted, strict=True)):
                                _same(self, actual, expected, k)
                            if source == "special":
                                xla = [np.asarray(v, np.float64).ravel().tolist()
                                       for v in function(a, b)]
                                model = reference(a, b, dtype, xla=True)
                                for k in compared:
                                    _same(self, xla[k], model[k], k)

                def edges(a, b):
                    z = lax.complex(a, b)
                    results = [f(z[:, :n], axis=1) for n in (1, 2)
                               for f in (jnp.prod, jnp.max, jnp.min)]
                    results += [f(z[:, :0], axes=(1,))
                                for f in (lax.reduce_prod, lax.reduce_max, lax.reduce_min)]
                    return [v for p in results for v in (p.real, p.imag)]

                with self.subTest(dtype=dtype, edges=True):
                    rows = [[(inf, 0.0), (1.0, 0.0)], [(-inf, -1.0), (-inf, -1.0)],
                            [(inf, 1.0), (inf, 1.0)], [(-0.0, -0.0), (2.0, 0.0)]]
                    a, b = [jnp.asarray([[w[k] for w in row] for row in rows], dtype)
                            for k in (0, 1)]
                    report = self.verify(edges, a, b, checks=EXECUTE)
                    self.assert_passes(report)
                    got = [[float(v) for v in actual.double().ravel()]
                           for actual in _outputs(report)]
                    xla = [np.asarray(v, np.float64).ravel().tolist() for v in edges(a, b)]
                    for k, (n, kind) in enumerate([(n, kind) for n in (1, 2, 0) for kind in kinds]):
                        wanted = [fold(row[:n], kind, dtype) for row in rows]
                        model = [fold(row[:n], kind, dtype, n > 1) for row in rows]
                        for j in (0, 1):
                            _same(self, got[2 * k + j], [z[j] for z in wanted], (n, kind))
                            _same(self, xla[2 * k + j], [z[j] for z in model], (n, kind))
                    # The lone entries, as they are; then (inf + 0i)(1 + 0i), the maximum of
                    # -inf - i and -inf - i and the minimum of inf + i and inf + i, and XLA's.
                    _same(self, [got[0][0], got[0][3], got[1][0], got[1][3]],
                          [inf, -0.0, 0.0, -0.0])
                    _same(self, [got[6][0], got[7][0], got[8][1], got[9][1], got[10][2],
                                 got[11][2]], [inf, nan, -inf, -1.0, inf, 1.0])
                    _same(self, [xla[6][0], xla[7][0], xla[8][1], xla[9][1], xla[10][2],
                                 xla[11][2]], [nan, nan, -inf, 0.0, inf, 0.0])
                    _same(self, [v for g in got[12:] for v in g[:1]],
                          [1.0, 0.0, -inf, 0.0, inf, 0.0])
        names = {e.primitive.name for e in jax.make_jaxpr(folds)(*normal).jaxpr.eqns}
        self.assertLessEqual({"reduce_prod", "cumprod", "reduce_max", "reduce_min", "cummax",
                              "cummin"}, names)

    def test_complex_window_reductions_fold_from_the_init_value(self):
        # reduce_window_sum, reduce_window_max and reduce_window_min of complex values, and
        # reduce_window with an add, a mul or a max as its body, or a max or an add of its arguments
        # swapped: each window folded from the init value over its cells in row-major order, by each
        # part's sum, the complex product rounded once (_complex_reference) or lax.max or lax.min (x
        # where the real parts compare so, or are equal and the imaginary parts do, by IEEE
        # comparisons, y elsewhere), on the operand dilated and padded with the init value. The
        # named reductions' init values are 0, -inf + 0i and inf + 0i; the others' are not
        # identities. Random parts and parts drawn from small integers, signed zeros, infinities and
        # NaN, eager and under jax.jit, at both widths. XLA's words are the same for the unpadded
        # sum, for the minimum with holes and the maximum padded with an init value that is not an
        # identity, where an init cell changes nothing, for the swapped maximum, and for the padded
        # maximum of random parts. Elsewhere its loop skips the padding and the holes, reorders
        # padded windows, and rounds the product otherwise.
        import math

        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(73)
        inf, nan = math.inf, math.nan
        bodies = {"add": lax.add, "swapped": lambda p, q: q + p, "mul": lax.mul, "max": lax.max,
                  "min": lax.min, "flipped": lambda p, q: lax.max(q, p)}
        # (body, init value or None for the named reduction's, window, strides, padding, base
        # dilation, window dilation)
        specs = [("add", None, (2, 3), (1, 2), ((0, 0), (0, 0)), (1, 1), (1, 1)),
                 ("add", None, (2, 2), (2, 1), ((1, 0), (0, 2)), (1, 1), (1, 1)),
                 ("max", None, (2, 2), (1, 1), ((1, 1), (0, 1)), (1, 1), (1, 1)),
                 ("min", None, (2, 1), (1, 1), ((0, 0), (0, 0)), (1, 2), (1, 1)),
                 ("mul", 1 + 0j, (1, 2), (1, 1), ((0, 0), (0, 0)), (1, 1), (1, 1)),
                 ("swapped", 1 + 1j, (2, 2), (1, 1), ((1, 1), (0, 0)), (2, 1), (1, 2)),
                 ("max", 0.5 - 1j, (1, 3), (1, 1), ((0, 0), (1, 0)), (1, 1), (1, 1)),
                 ("flipped", 0.5 - 1j, (2, 3), (1, 2), ((0, 0), (0, 0)), (1, 1), (1, 1))]
        named = {"add": 0j, "max": complex(-inf, 0.0), "min": complex(inf, 0.0)}

        def windows(a, b):
            z = lax.complex(a, b)
            results = [lax.reduce_window(z, named[body] if init is None else
                                         jnp.asarray(init, z.dtype), bodies[body], *rest)
                       for body, init, *rest in specs]
            return [v for r in results for v in (r.real, r.imag)]

        def step(body, z, w, dtype):
            if body in ("add", "swapped"):
                number = np.dtype(dtype).type
                return [float(number(p) + number(q)) for p, q in zip(z, w)]
            if body == "mul":
                return _complex_reference("mul", *z, *w, dtype)
            if body == "flipped":
                return step("max", w, z, dtype)
            (a, b), (c, d) = z, w
            order = (lambda p, q: p > q) if body == "max" else (lambda p, q: p < q)
            return list(z) if (order(b, d) if a == c else order(a, c)) else list(w)

        def reference(a, b, dtype):
            wanted = []
            for body, init, dims, strides, padding, base, dilation in specs:
                init = named[body] if init is None else init
                size = [(n - 1) * s + 1 + lo + hi for n, s, (lo, hi) in zip(a.shape, base, padding)]
                parts = [np.full(size, v, np.float64) for v in (init.real, init.imag)]
                at = tuple(slice(lo, lo + (n - 1) * s + 1, s)
                           for n, s, (lo, _) in zip(a.shape, base, padding))
                parts[0][at], parts[1][at] = a, b
                out = [(m - (w - 1) * d - 1) // s + 1
                       for m, w, d, s in zip(size, dims, dilation, strides)]
                folded = [np.empty(out), np.empty(out)]
                for o in np.ndindex(*out):
                    z = [init.real, init.imag]
                    for c in np.ndindex(*dims):
                        cell = tuple(i * s + j * d for i, s, j, d in zip(o, strides, c, dilation))
                        z = step(body, z, [float(p[cell]) for p in parts], dtype)
                    folded[0][o], folded[1][o] = z
                wanted += [f.ravel().tolist() for f in folded]
            return wanted

        drawn = [-inf, -2.0, -1.0, -0.0, 0.0, 1.0, 2.0, inf, nan]
        for dtype in ("float32", "float64"):
            with jax.enable_x64(dtype == "float64"):
                normal = [np.asarray(v, dtype) for v in rng.standard_normal((2, 4, 6))]
                special = [np.asarray(rng.choice(drawn, (4, 6)), dtype) for _ in range(2)]
                for source, (a, b) in (("normal", normal), ("special", special)):
                    wanted = reference(a, b, dtype)
                    for mode, function in (("eager", windows), ("jit", jax.jit(windows))):
                        with self.subTest(dtype=dtype, source=source, mode=mode):
                            report = self.verify(function, jnp.asarray(a), jnp.asarray(b),
                                                 checks=EXECUTE)
                            self.assert_passes(report)
                            got = [[float(v) for v in actual.double().ravel()]
                                   for actual in _outputs(report)]
                            for k, (actual, expected) in enumerate(zip(got, wanted, strict=True)):
                                _same(self, actual, expected, k)
                            xla = [np.asarray(v, np.float64).ravel().tolist()
                                   for v in function(a, b)]
                            for k in [0, 1, 6, 7, 12, 13, 14, 15] + [4, 5] * (source == "normal"):
                                _same(self, xla[k], wanted[k], k)
        names = {e.primitive.name for e in jax.make_jaxpr(windows)(*normal).jaxpr.eqns}
        self.assertLessEqual({"reduce_window_sum", "reduce_window_max", "reduce_window_min",
                              "reduce_window"}, names)

    def test_complex_contraction_is_four_real_contractions(self):
        # dot_general of two complex arrays is four real contractions of their parts a + ib and
        # c + id, re = a c - b d and im = b c + a d, each summed in the IR's order. Independent
        # NumPy products and left-fold additions check that schedule at every size. XLA can
        # fuse and regroup the arithmetic even for small matrices; its exact-word comparison
        # must report the words it actually returns, rather than assume a kernel's schedule.
        import numpy as np

        from torchlean.jaxpr import encode

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(41)

        def flat(values):
            return [float(v) for v in np.asarray(values, np.float64).ravel()]

        def product(a, b, c, d):
            p = lax.complex(a, b) @ lax.complex(c, d)
            return [p.real, p.imag]

        def batched(a, b, c, d):
            p = jnp.einsum("bijk,bjkl->bil", lax.complex(a, b), lax.complex(c, d))
            return [p.real, p.imag]

        def real_operand(a, c, d):
            p = a @ lax.complex(c, d)
            return [p.real, p.imag]

        def sums(u, v):
            total = np.zeros(u.shape[:-1] + (v.shape[-1],), u.dtype)
            for k in range(u.shape[-1]):
                total = total + u[..., k, None] * v[..., None, k, :]
            return total

        def four_sums(a, b, c, d):
            a, b, c, d = map(np.asarray, (a, b, c, d))
            with np.errstate(invalid="ignore"):
                return [sums(a, c) - sums(b, d), sums(b, c) + sums(a, d)]

        def checked(function, operands, expected):
            report = self.verify(function, *operands,
                                 checks=("structure", "types", "execute", "compare_bitwise"))
            statuses = _statuses(report)
            for name, status in (("structure", "checked"), ("types", "checked"),
                                 ("execute", "executed")):
                self.assertEqual(statuses[name][0], status, statuses)
            for output, actual, wanted in zip(report["outputs"], _outputs(report), expected,
                                               strict=True):
                self.assertEqual((output["element_type"], output["shape"]),
                                 (wanted.dtype.name, list(wanted.shape)))
                _same(self, flat(actual.double()), flat(wanted))
            matches = all(output["bits"] == encode(value)["bits"]
                          for output, value in zip(report["outputs"], function(*operands),
                                                   strict=True))
            self.assertEqual(statuses["compare_bitwise"][0], "compared" if matches else "failed",
                             statuses)
            self.assertEqual(report["ok"], matches)
            return report

        for dtype in ("float32", "float64"):
            with jax.enable_x64(dtype == "float64"):
                def draw(*shapes):
                    return [jnp.asarray(rng.standard_normal(shape), dtype) for shape in shapes]

                for terms in (1, 2, 3, 5, 7, 64):
                    parts = draw(*[(3, terms)] * 2, *[(terms, 4)] * 2)
                    expected = four_sums(*parts)
                    for mode, function in (("eager", product), ("jit", jax.jit(product))):
                        with self.subTest(dtype=dtype, terms=terms, mode=mode):
                            checked(function, parts, expected)
                real = draw((3, 5), (5, 4), (5, 4))
                batch = draw(*[(2, 3, 2, 3)] * 2, *[(2, 2, 3, 4)] * 2)
                flattened = [np.asarray(value).reshape(shape) for value, shape in zip(
                    batch, [(2, 3, 6)] * 2 + [(2, 6, 4)] * 2, strict=True)]
                cases = {"batch": (batched, batch, four_sums(*flattened)),
                         "real operand": (real_operand, real, four_sums(
                             real[0], np.zeros_like(real[0]), *real[1:]))}
                for case, (function, operands, expected) in cases.items():
                    for mode, run in (("eager", function), ("jit", jax.jit(function))):
                        with self.subTest(dtype=dtype, case=case, mode=mode):
                            checked(run, operands, expected)
                real[2] = real[2].at[2, 1].set(np.inf)
                expected = four_sums(real[0], np.zeros_like(real[0]), *real[1:])
                for mode, run in (("eager", real_operand), ("jit", jax.jit(real_operand))):
                    with self.subTest(dtype=dtype, case="infinite d", mode=mode):
                        report = checked(run, real, expected)
                        actual = flat(_outputs(report)[0].double())
                        self.assertTrue(all(v != v for v in actual[1::4]))
        self.assertIn("contract", _tags(self.artifact(product, *[np.eye(3, dtype="float32")] * 4)))

    def test_complex_convolution_is_four_real_convolutions(self):
        # conv_general_dilated of complex values and a constant kernel is four real convolutions
        # of the parts a + ib and c + id, re = a c - b d and im = b c + a d, as dot_general
        # contracts: through jnp.convolve, which flips its kernel, jnp.correlate, which
        # conjugates it, and lax.conv_general_dilated with padding, strides, dilation and groups,
        # eager and under jax.jit, at both widths. The words are those of the four real
        # convolutions of the parts, imported from a real function; at parts drawn from small
        # integers, where every sum is exact, they are XLA's too. A tiled complex value is its
        # parts tiled. A kernel that is not known at capture, an input or a slice of one, gives
        # its parts to the four convolutions as their kernel operands.
        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(83)

        def conv(u, w):
            return lax.conv_general_dilated(u, w, (1, 2), ((1, 0), (2, 1)), rhs_dilation=(2, 1),
                                            feature_group_count=2)

        for dtype, real in (("complex64", "float32"), ("complex128", "float64")):
            with jax.enable_x64(dtype == "complex128"):
                for source in ("integers", "normal"):
                    def draw(*shape):
                        parts = [rng.integers(-3, 4, shape) if source == "integers" else
                                 rng.standard_normal(shape) for _ in range(2)]
                        return parts[0].astype(real), parts[1].astype(real)

                    c, d = draw(4)
                    k, m = draw(4, 2, 2, 3)
                    taps, stencil = (c + 1j * d).astype(dtype), (k + 1j * m).astype(dtype)

                    def complexes(s, t, x, y):
                        v, z = lax.complex(s, t), lax.complex(x, y)
                        out = [jnp.convolve(v, taps), jnp.correlate(v, taps, "same"),
                               conv(z, stencil), jnp.tile(v, 2)]
                        return [w for o in out for w in (o.real, o.imag)]

                    def reals(s, t, x, y):
                        def four(f, c, d, a, b):
                            return [f(a, c) - f(b, d), f(b, c) + f(a, d)]

                        return [*four(jnp.convolve, c, d, s, t),
                                *four(lambda u, w: jnp.correlate(u, w, "same"), c, -d, s, t),
                                *four(conv, k, m, x, y), jnp.tile(s, 2), jnp.tile(t, 2)]

                    operands = [jnp.asarray(p) for p in (*draw(9), *draw(1, 4, 5, 7))]
                    wanted = [actual.double().ravel().tolist() for actual in _outputs(
                        self.verify(reals, *operands, checks=EXECUTE))]
                    checks = COMPARE if source == "integers" else EXECUTE
                    for mode, function in (("eager", complexes), ("jit", jax.jit(complexes))):
                        with self.subTest(dtype=dtype, source=source, mode=mode):
                            report = self.verify(function, *operands, checks=checks)
                            self.assert_passes(report)
                            for n, (actual, expected) in enumerate(
                                    zip(_outputs(report), wanted, strict=True)):
                                _same(self, actual.double().ravel().tolist(), expected, n)
                    self.assertIn("conj", _names(jax.make_jaxpr(
                        lambda v: jnp.correlate(v, taps))(operands[0] + 0j).jaxpr))

        def unknown(s, t, k, m, x, y, p, q):
            v, w, z, u = (lax.complex(a, b) for a, b in ((s, t), (k, m), (x, y), (p, q)))
            out = [jnp.convolve(v, v[:3]), jnp.correlate(v, w, "same"), conv(z, u)]
            return [r for o in out for r in (o.real, o.imag)]

        def parts(s, t, k, m, x, y, p, q):
            def four(f, c, d, a, b):
                return [f(a, c) - f(b, d), f(b, c) + f(a, d)]

            return [*four(jnp.convolve, s[:3], t[:3], s, t),
                    *four(lambda a, c: jnp.correlate(a, c, "same"), k, -m, s, t),
                    *four(conv, p, q, x, y)]

        operands = [jnp.asarray(rng.integers(-3, 4, shape), jnp.float32)
                    for shape in [(9,)] * 2 + [(4,)] * 2 + [(1, 4, 5, 7)] * 2 + [(4, 2, 2, 3)] * 2]
        wanted = [actual.double().ravel().tolist()
                  for actual in _outputs(self.verify(parts, *operands, checks=EXECUTE))]
        report = self.verify(unknown, *operands, checks=COMPARE)
        self.assert_passes(report)
        for n, (actual, expected) in enumerate(zip(_outputs(report), wanted, strict=True)):
            _same(self, actual.double().ravel().tolist(), expected, n)

    def test_complex_scatter_products_multiply_in_index_order(self):
        # scatter-mul of complex values multiplies each slice by its windows of updates in turn,
        # in the row-major order of the index vectors, each step the complex product rounded as
        # mul rounds it: the words of a loop of gathers, products and replacements in that order,
        # with colliding rows and elements, windows dropped outside (mode="drop") or clamped
        # inside (mode="clip"), eager and under jax.jit, at both widths. At parts drawn from
        # small integers, where every product is exact, they are XLA's too. A scatter with
        # batching dimensions, as jax.vmap writes it, is refused.
        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(89)
        rows, outside = [3, 1, 3, 0, 3], [4, 9, -9, 2, 4]
        cells = ([1, 0, 1, 1], [2, 2, 2, 0])

        def operands(s, t, x, y):
            return lax.complex(s, t), lax.complex(x, y), lax.complex(x[:4, 0], y[:4, 1])

        def complexes(s, t, x, y, i, j, k, m):
            z, w, v = operands(s, t, x, y)
            out = [z.at[i].multiply(w), z.at[j].multiply(w, mode="drop"),
                   z.at[j].multiply(w, mode="clip"), z.at[k, m].multiply(v)]
            return [p for o in out for p in (o.real, o.imag)]

        def reals(s, t, x, y):
            z, w, v = operands(s, t, x, y)

            def looped(targets, factors):
                r = z
                for k, target in enumerate(targets):
                    if target is not None:
                        r = r.at[target].set(r[target] * factors[k])
                return r

            n = z.shape[0]
            normal = [q + n if q < 0 else q for q in outside]
            out = [looped(rows, w), looped([q if 0 <= q < n else None for q in normal], w),
                   looped([min(max(q, 0), n - 1) for q in normal], w),
                   looped(list(zip(*cells)), v)]
            return [p for o in out for p in (o.real, o.imag)]

        for dtype, real in (("complex64", "float32"), ("complex128", "float64")):
            with jax.enable_x64(dtype == "complex128"):
                for source in ("integers", "normal"):
                    def draw(*shape):
                        values = (rng.integers(-3, 4, shape) if source == "integers" else
                                  rng.standard_normal(shape))
                        return jnp.asarray(values.astype(real))

                    parts = [draw(5, 3) for _ in range(4)]
                    indices = [jnp.asarray(q, jnp.int32) for q in (rows, outside, *cells)]
                    wanted = [actual.double().ravel().tolist() for actual in _outputs(
                        self.verify(reals, *parts, checks=EXECUTE))]
                    checks = COMPARE if source == "integers" else EXECUTE
                    for mode, function in (("eager", complexes), ("jit", jax.jit(complexes))):
                        with self.subTest(dtype=dtype, source=source, mode=mode):
                            report = self.verify(function, *parts, *indices, checks=checks)
                            self.assert_passes(report)
                            for n, (actual, expected) in enumerate(
                                    zip(_outputs(report), wanted, strict=True)):
                                _same(self, actual.double().ravel().tolist(), expected, n)
        self.assertIn("scatter-mul", _names(jax.make_jaxpr(complexes)(*parts, *indices).jaxpr))

        def batched(s, t, i):
            return jax.vmap(lambda z, q: z.at[q].multiply(z[:2]))(lax.complex(s, t), i).real

        self.assert_refused(self.verify(batched, *[np.ones((2, 5), np.float32)] * 2,
                                        np.zeros((2, 2), np.int32), checks=("structure",)),
                            "batching dimensions")

    def test_complex_inputs_and_outputs_are_their_parts(self):
        # A complex argument or result of shape s is the tensor of its parts' format and shape
        # s + [2] that holds each element's real and imaginary parts on the last axis: the IR's
        # input or output, its declared type, as a complex value inside is declared, and the
        # document's argument or reference output (encode). Exact operations on complex inputs
        # give XLA's words, NaNs, infinities and signed zeros included, eager and under jax.jit,
        # and a product of two complex inputs is each exact part rounded once
        # (_complex_reference).
        import math

        import numpy as np

        from torchlean.jaxpr import bits, encode

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(29)
        inf, nan = math.inf, math.nan
        bitwise = ("structure", "types", "execute", "compare_bitwise")
        for dtype, part in (("complex64", "float32"), ("complex128", "float64")):
            with self.subTest(dtype=dtype), jax.enable_x64(dtype == "complex128"):
                z, w = [jnp.asarray(rng.standard_normal((2, 3)) + 1j * rng.standard_normal((2, 3)),
                                    dtype) for _ in range(2)]
                x = jnp.asarray(rng.standard_normal((2, 3)), part)
                special = jnp.asarray([complex(-0.0, nan), complex(inf, -0.0), complex(-inf, 1),
                                       complex(nan, -inf)], dtype)
                words = bits(np.stack([np.asarray(z).real, np.asarray(z).imag], axis=-1))
                self.assertEqual(encode(z), dict(dtype=part, shape=[2, 3, 2], bits=words))
                document = self.artifact(lambda z, x: (z, x, z + x), z, x)
                declared = document["declared_types"]
                self.assertEqual([(entry["dtype"], entry["shape"]) for entry in declared[:2]],
                                 [(part, [2, 3, 2]), (part, [2, 3])])
                (added,) = [entry for entry in declared if entry["origin"].endswith(" = add")]
                self.assertEqual((added["dtype"], added["shape"]), (part, [2, 3, 2]))
                self.assertEqual(document["arguments"][0], encode(z))
                self.assertEqual(document["reference_outputs"][0], encode(z))
                report = self.verify(lambda z: z, z, checks=bitwise)
                self.assert_passes(report)
                (output,) = report["outputs"]
                self.assertEqual((output["element_type"], output["shape"], output["bits"]),
                                 (part, [2, 3, 2], words))
                cases = [(lambda z: (z, jnp.conj(z), z.real, z.imag), special),
                         (lambda z, w: z + w, z, w), (lambda z, w: jnp.conj(z) - w, z, w),
                         (lambda x: lax.complex(x, -x), x), (lambda z: jnp.sum(z, axis=1), z),
                         (jax.jit(lambda z, w: jnp.where(z.real > 0, z, w)), z, w)]
                for function, *operands in cases:
                    self.assert_passes(self.verify(function, *operands, checks=bitwise))
                report = self.verify(lambda z, w: z * w, z, w)
                self.assert_passes(report)
                parts = [np.asarray(v, np.float64).ravel()
                         for v in (z.real, z.imag, w.real, w.imag)]
                wanted = [_complex_reference("mul", *entry, part) for entry in zip(*parts)]
                _same(self, [float(v) for v in _outputs(report)[0].double().ravel()],
                      [v for entry in wanted for v in entry])
