"""JAX operator captures checked with ``verify ir`` (``JaxGeneralOperatorTests``), the lax
primitives the Jaxpr importer reads (``python/torchlean/jaxpr.py``). Run them through
``scripts/checks/test_ir_capture.py``, which describes the checks.
"""

import json
import unittest

from ir_capture_support import (CHECKER, COMPARE, EXECUTE, FORMATS, HAS_JAX, CheckerCase,
                                _assert_within_ulps, _cube_root, _float_of_word, _float_words,
                                _ieee_maximum, _ieee_minimum, _midpoints, _nearest, _outputs,
                                _records, _registered, _same, _splitmix64, _steps, _tags,
                                _window_fold, _words)
import torchlean


@unittest.skipUnless(CHECKER and HAS_JAX, "needs the built verify executable and JAX")
class JaxGeneralOperatorTests(CheckerCase):
    def setUp(self):
        import jax
        import jax.numpy as jnp

        self.jax, self.jnp = jax, jnp
        self.x = jnp.arange(24.0, dtype=jnp.float32).reshape(2, 3, 4) - 10

    def verify(self, function, *inputs, checks=COMPARE):
        return torchlean.verify(function, *inputs, frontend="jax", checks=checks,
                                checker=CHECKER)

    def artifact(self, function, *inputs):
        from torchlean.jaxpr import export

        return export(function, *inputs)

    def test_reductions_and_softmax(self):
        jax, jnp = self.jax, self.jnp

        def reductions(x):
            return (jnp.max(x, axis=(0, 2)), jnp.min(x, axis=1), jnp.prod(x[:, :, :2], axis=2),
                    jnp.sum(x, axis=(0, 1)))

        from torchlean._table import op_table

        self.assert_passes(self.verify(reductions, self.x))
        document = self.artifact(reductions, self.x)
        self.assertEqual(document["artifact"]["format"], op_table()["format"])
        codes = op_table()["codes"]["reduce"]
        self.assertEqual({record["fields"][1]["nat"] for record in _records(document)
                          if record["tag"] == "reduce"},
                         {codes[name] for name in ("max", "min", "prod", "sum")})
        softmax = lambda x: (jax.nn.softmax(x, axis=-1), jax.nn.log_softmax(x, axis=1))
        self.assert_passes(self.verify(softmax, self.x / 7, checks=EXECUTE))

    def test_data_movement_and_contraction(self):
        jax, jnp = self.jax, self.jnp
        lax = jax.lax

        def unit(x):
            return (lax.pad(x, 0.5, ((0, 0, 0), (-1, 1, 1), (1, 0, 0))),
                    jnp.flip(x, axis=(0, 2)), jnp.concatenate([x, x], axis=1),
                    jnp.split(x, [1], axis=2)[1], jnp.tile(x, (1, 2, 1)),
                    jnp.broadcast_to(x[:, None], (2, 5, 3, 4)),
                    x + lax.broadcasted_iota(jnp.float32, x.shape, 1),
                    jnp.einsum("bij,bkj->bik", x, x))

        movement = lambda x: (x[:, 1:3, ::2], *unit(x))
        document = self.artifact(movement, self.x)
        tags = _tags(document)
        for tag in ("slice", "pad", "flip", "cat", "broadcast", "contract"):
            self.assertIn(tag, tags)
        self.assertNotIn("matmul", tags)
        # A slice record's fields are its operand shape, dimensions, starts, stops and steps.
        steps = [record["fields"][4]["nats"] for record in _records(document)
                 if record["tag"] == "slice"]
        self.assertEqual([step for step in steps if step != [1] * len(step)], [[1, 2]])
        self.assert_passes(self.verify(unit, self.x))

    def test_stack_and_unstack_move_data(self):
        # lax.stack, which jnp.stack of a sequence binds, becomes one cat of its operands each
        # reshaped to an axis of size one, and lax.unstack one slice of width one per output
        # without that axis. Both compare bit for bit with JAX, at every axis and at an axis of
        # size one, of floating and integer operands.
        jax, jnp = self.jax, self.jnp
        lax = jax.lax

        def unit(x):
            return (jnp.stack([x, -x, x * 2]), lax.stack([x, x + 1], axis=1),
                    jnp.stack([x, x], axis=-1), lax.stack([x], axis=2),
                    *lax.unstack(x), *jnp.unstack(x, axis=1),
                    *lax.unstack(x.reshape(2, 3, 4, 1), axis=-1))

        names = {equation.primitive.name for equation in jax.make_jaxpr(unit)(self.x).eqns}
        self.assertLessEqual({"stack", "unstack"}, names)
        tags = _tags(self.artifact(unit, self.x))
        self.assertEqual(tags.count("cat"), 4)
        # One slice for each output of the axes of sizes two and three; an axis of size one is
        # the whole operand.
        self.assertEqual(tags.count("slice"), 2 + 3)
        self.assert_passes(self.verify(unit, self.x))
        n = jnp.arange(6, dtype=jnp.int32).reshape(2, 3)
        self.assert_passes(self.verify(lambda n: (jnp.stack([n, n - 7]), *jnp.unstack(n, axis=1)),
                                       n))

    def test_complex_split_unstack_and_stack_move_pairs(self):
        # A complex split, unstack or stack cuts or joins the paired-real form along the
        # operand's axes, never the pair axis, so the pieces are complex too and the IR's words
        # are XLA's, NaNs, infinities and signed zeros included, eager and under jax.jit, at
        # every axis, an axis of size one and a piece of width zero.
        import math

        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        inf, nan = math.inf, math.nan
        bitwise = ("structure", "types", "execute", "compare_bitwise")
        values = [complex(-0.0, nan), complex(inf, -0.0), complex(-inf, 1), complex(nan, -inf),
                  complex(1.5, -2.25), complex(0.0, -0.0)]

        def moved(z):
            return (*jnp.split(z, [1, 3], axis=2), *lax.split(z, [0, 2, 1], axis=1),
                    *lax.unstack(z), *jnp.unstack(z, axis=2), *lax.unstack(z[:1], axis=0),
                    jnp.stack([z, -z, jnp.conj(z)], axis=1), lax.stack([z[0]], axis=2),
                    jnp.stack(lax.unstack(z, axis=1), axis=-1), *jnp.split(-z, 2, axis=2))

        for dtype in ("complex64", "complex128"):
            with self.subTest(dtype=dtype), jax.enable_x64(dtype == "complex128"):
                z = jnp.asarray(np.resize(values, (2, 3, 4)), dtype)
                names = {equation.primitive.name for equation in jax.make_jaxpr(moved)(z).eqns}
                self.assertLessEqual({"split", "unstack", "stack"}, names)
                for function in (moved, jax.jit(moved)):
                    self.assert_passes(self.verify(function, z, checks=bitwise))

    def test_real_fourier_transforms_within_tolerance(self):
        # A complex value is held in paired-real form. fft of type RFFT and IRFFT along the last
        # axis is Op.rfft and Op.irfft, and every other transform, FFT and IFFT and the other axes
        # of an RFFT or IRFFT, is built from Op.rfft of each part. These are TorchLean's sums and
        # not XLA's FFT schedule, so those values agree within a tolerance, lengths 1 and 2
        # included. real, imag, complex, conj, add, sub, neg, conversions and data movement act
        # on the parts and compare bit for bit, as does a reshape whose dimensions permute the
        # operand's axes. The imaginary parts of frequency 0
        # and, for an even length n, n / 2 are ignored, as numpy.fft.irfft documents, so
        # infinities and NaNs there leave the signal finite. An orthonormal RFFT scales by
        # 1 / sqrt(n), which jnp.fft computes in complex arithmetic: reduce_prod of n + 0i, sqrt,
        # div and mul.
        import numpy as np

        jax, jnp = self.jax, self.jnp
        lax = jax.lax
        rng = np.random.default_rng(7)
        x = rng.standard_normal((4, 6))
        u, v = rng.standard_normal((2, 4, 4))

        def transforms(x, u, v):
            z = lax.complex(u, v)
            f, g = jnp.fft.rfft(x), jnp.fft.rfft(x, n=10, axis=0)
            return (f.real, f.imag, g.real, jnp.imag(jnp.conj(g)), jnp.fft.irfft(f, n=6),
                    jnp.fft.irfft(z), jnp.fft.irfft(z, n=5, axis=0), jnp.fft.irfft(z, n=9),
                    jnp.fft.irfft(z, norm="ortho"), jnp.fft.irfft(x),
                    (jnp.fft.rfft(x, n=4) + z[:, :3] - (-z[:, 1:])).imag)

        def parts(u, v):
            z = lax.complex(u, v)
            w = jnp.conj(z) - z + (-z)
            return (w.real, w.imag, jnp.broadcast_to(z, (2, 4, 4)).reshape(8, 4).T.imag,
                    jnp.squeeze(z[None], 0).real, z[1:3, ::2].imag, lax.complex(u[0], 2.0).imag,
                    jnp.pad(z, ((0, 0), (1, 2))).imag, (z + u.astype(jnp.complex64)).real,
                    lax.reshape(z[:, :3], (12,), dimensions=(1, 0)).real,
                    lax.reshape(z[:, :3], (3, 2, 2), dimensions=(1, 0)).imag)

        for dtype in ("float32", "float64"):
            with self.subTest(dtype=dtype), jax.enable_x64(dtype == "float64"):
                arguments = [jnp.asarray(value, dtype=dtype) for value in (x, u, v)]
                report = self.verify(transforms, *arguments, checks=EXECUTE)
                self.assert_passes(report)
                for actual, wanted in zip(_outputs(report), transforms(*arguments), strict=True):
                    self.assertEqual(str(actual.dtype), f"torch.{dtype}")
                    np.testing.assert_allclose(actual.numpy(), np.asarray(wanted), rtol=1e-4,
                                               atol=1e-5)
                self.assert_passes(self.verify(parts, *arguments[1:]))
        y = rng.standard_normal((2, 4, 6))

        def complexes(x, u, v, y):
            z, w = lax.complex(u, v), lax.complex(y, -y)
            spectra = (jnp.fft.fft(x), jnp.fft.ifft(z), jnp.fft.fft(z, n=5, axis=0),
                       jnp.fft.ifft(z[:, :1]), jnp.fft.fft(z[:, :2]), jnp.fft.fft2(z),
                       jnp.fft.ifft2(z), jnp.fft.ifftn(w, axes=(0, 1, 2)), jnp.fft.rfft2(x),
                       jnp.fft.rfftn(y), lax.rev(z, (0,)), lax.concatenate([z, -z], 1),
                       jnp.fft.rfft(x, norm="ortho"))
            return (*[s.real for s in spectra], *[s.imag for s in spectra], jnp.fft.irfft2(z),
                    jnp.fft.irfft2(z, s=(4, 7)), jnp.fft.irfftn(z, s=(3, 6)),
                    jnp.fft.irfftn(w[..., :4], s=(2, 4, 6)))

        for dtype in ("float32", "float64"):
            with self.subTest(dtype=dtype, complex=True), jax.enable_x64(dtype == "float64"):
                arguments = [jnp.asarray(value, dtype=dtype) for value in (x, u, v, y)]
                report = self.verify(complexes, *arguments, checks=EXECUTE)
                self.assert_passes(report)
                for actual, wanted in zip(_outputs(report), complexes(*arguments), strict=True):
                    np.testing.assert_allclose(actual.numpy(), np.asarray(wanted), rtol=1e-4,
                                               atol=1e-5)
        inf, nan = np.inf, np.nan
        r = rng.standard_normal((2, 4))
        ignored = {6: [[inf, 1.0, -2.0, nan], [-inf, -2.0, 3.0, inf]],
                   7: [[nan, 1.0, -2.0, 0.5], [inf, -2.0, 3.0, 1.5]],
                   4: [[-inf, 1.0, nan, nan], [nan, -2.0, inf, 1.5]]}

        def inverses(r, *parts):
            return tuple(jnp.fft.irfft(lax.complex(r, i), n=n) for n, i in zip(ignored, parts))

        for dtype in ("float32", "float64"):
            with self.subTest(dtype=dtype, ignored=True), jax.enable_x64(dtype == "float64"):
                arguments = [jnp.asarray(value, dtype=dtype) for value in (r, *ignored.values())]
                report = self.verify(inverses, *arguments, checks=EXECUTE)
                self.assert_passes(report)
                for actual, n, i in zip(_outputs(report), ignored, arguments[1:], strict=True):
                    spectrum = np.empty((2, 4), dtype=complex)
                    spectrum.real, spectrum.imag = arguments[0], i
                    np.testing.assert_allclose(actual.numpy(), np.fft.irfft(spectrum, n=n),
                                               rtol=1e-4, atol=1e-5)
        tags = _tags(self.artifact(transforms, *[value.astype(np.float32) for value in (x, u, v)]))
        self.assertEqual((tags.count("rfft"), tags.count("irfft")), (3, 6))
        names = {equation.primitive.name
                 for equation in jax.make_jaxpr(parts)(u.astype(np.float32), v).eqns}
        self.assertLessEqual({"real", "imag", "complex", "conj", "add", "sub", "neg", "reshape",
                              "transpose", "broadcast_in_dim", "squeeze", "slice",
                              "convert_element_type"}, names)
        def primitives(jaxpr):
            for equation in jaxpr.eqns:
                yield equation
                for value in equation.params.values():
                    inner = getattr(value, "jaxpr", value)
                    if hasattr(inner, "eqns"):
                        yield from primitives(inner)

        equations = list(primitives(jax.make_jaxpr(complexes)(x, u, v, y).jaxpr))
        self.assertLessEqual({"rev", "concatenate", "pad"},
                             {equation.primitive.name for equation in equations})
        self.assertEqual({int(equation.params["fft_type"]) for equation in equations
                          if equation.primitive.name == "fft"}, {0, 1, 2, 3})
        x = x.astype(np.float32)
        refused = {"complex logistic": (lambda x: lax.logistic(jnp.fft.rfft(x)).real,
                                        "complex values are imported only through"),
                   "empty transform": (
                       lambda x: lax.fft(lax.complex(x, x)[:, :0], lax.FftType.FFT, (0,)).real,
                       "do not name nonempty trailing axes"),
                   "short spectrum": (
                       lambda x: lax.fft(lax.complex(x, x)[:, :3], lax.FftType.IRFFT, (8,)),
                       "do not name nonempty trailing axes")}
        for name, (function, text) in refused.items():
            with self.subTest(name=name):
                self.assert_refused(self.verify(function, x, checks=("structure",)), text)

    def test_complex_modulus_and_constants(self):
        # JAX documents abs of a + ib as sqrt(a^2 + b^2). It is the IR's hypot of the parts, the
        # exact modulus rounded once, which these words match, and +inf where a part is infinite
        # even if the other is a quiet NaN, as IEEE 754-2019 9.2.1 specifies hypot. XLA's CPU code
        # returns NaN there and is up to two units in the last place away elsewhere, so it is
        # compared within that on normal values. A complex constant, a literal or a captured
        # array, is the pair of its parts.
        import math
        from fractions import Fraction

        import mpmath
        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(15)

        def modulus(a, b, dtype):
            # The exact modulus of a + ib rounded to `dtype`, certified by the midpoints around
            # it (a midpoint has one more bit than the format, and these squares are not ties).
            if math.isinf(a) or math.isinf(b):
                return math.inf
            if a != a or b != b:
                return math.nan
            square = Fraction(a) ** 2 + Fraction(b) ** 2
            precision, _, emax = FORMATS[dtype]
            if square == 0:
                return 0.0
            if square >= (Fraction(2) ** (emax + 1) - Fraction(2) ** (emax - precision)) ** 2:
                return math.inf
            with mpmath.workprec(300):
                _, man, shift, _ = mpmath.sqrt(mpmath.mpf(a) ** 2 + mpmath.mpf(b) ** 2)._mpf_
            candidate = _nearest(Fraction(int(man)) * Fraction(2) ** shift, dtype)
            low, high = _midpoints(candidate, dtype)
            if not (low ** 2 < square and (high is None or square < high ** 2)):
                raise AssertionError(f"uncertified modulus of {a} + {b}i")
            return candidate

        def magnitudes(x, y):
            return jnp.abs(lax.complex(x, y)), lax.abs(lax.complex(-y, x))

        def flat(values):
            return np.asarray(values, np.float64).flatten().tolist()

        inf, nan = math.inf, math.nan
        for dtype, scale in (("float32", 15), ("float64", 150)):
            with self.subTest(dtype=dtype), jax.enable_x64(dtype == "float64"):
                x = rng.standard_normal(96) * 10.0 ** rng.integers(-scale, scale, 96)
                y = x * rng.uniform(0.1, 10, 96)
                y[48:] = rng.standard_normal(48) * 10.0 ** rng.integers(-scale, scale, 48)
                x, y = (jnp.asarray(v.reshape(8, 12), dtype) for v in (x, y))
                report = self.verify(magnitudes, x, y, checks=EXECUTE)
                self.assert_passes(report)
                wanted = [modulus(a, b, dtype) for a, b in zip(flat(x), flat(y))]
                for got, xla in zip(_outputs(report), jax.jit(magnitudes)(x, y)):
                    _same(self, flat(got.double()), wanted)
                    np.testing.assert_allclose(got.double().numpy(), np.asarray(xla, np.float64),
                                               rtol=2 * float(np.finfo(dtype).eps))
                info = np.finfo(dtype)
                big, tiny = float(info.max), float(info.smallest_subnormal)
                special = [(inf, nan), (nan, -inf), (-inf, 0.0), (nan, 0.0), (-0.0, -0.0),
                           (0.0, -0.0), (big, big), (big / 2, big / 2), (tiny, tiny), (3.0, -4.0)]
                x, y = (jnp.asarray([pair[k] for pair in special], dtype) for k in (0, 1))
                report = self.verify(magnitudes, x, y, checks=EXECUTE)
                self.assert_passes(report)
                for got in _outputs(report):
                    _same(self, flat(got.double()), [modulus(a, b, dtype) for a, b in special])
        c = jnp.asarray([1.5 - 2j, -0.25 + 0.5j, 3j], jnp.complex64)

        def shifted(x, y):
            w = lax.complex(x, y) + (2 - 1j) - c
            return w.real, w.imag, jnp.abs(w)

        x, y = (jnp.asarray(rng.integers(-8, 9, 3) / 4, jnp.float32) for _ in range(2))
        self.assertTrue(any(np.iscomplexobj(v) for v in jax.make_jaxpr(shifted)(x, y).consts))
        report = self.verify(shifted, x, y, checks=EXECUTE)
        self.assert_passes(report)
        real, imag = flat(x + 2 - c.real), flat(y - 1 - c.imag)
        _same(self, [w for v in _outputs(report) for w in flat(v.double())],
              real + imag + [modulus(a, b, "float32") for a, b in zip(real, imag)])
        x = rng.standard_normal((3, 8)).astype(np.float32)
        report = self.verify(lambda x: jnp.abs(jnp.fft.rfft(x)), x, checks=EXECUTE)
        self.assert_passes(report)
        np.testing.assert_allclose(_outputs(report)[0].numpy(), np.abs(np.fft.rfft(x)),
                                   rtol=1e-4, atol=1e-5)

    def test_complex_sums_comparisons_and_indexing_act_on_each_part(self):
        # Complex addition acts on each part, so reduce_sum, cumsum and a scatter's sum are those
        # of each part; eq holds where both parts are equal and ne where either differs; a
        # selection, a gather or a scatter moves each part at real predicates and indices, and a
        # FILL_OR_DROP gather's default fill is NaN converted to the complex type, NaN + 0i.
        # Multiples of 1/4 keep every sum exact whatever XLA's order, so those compare with XLA
        # bit for bit; random sums of at most 32 entries are XLA's row-major fold, and the IR's.
        import numpy as np

        jnp, lax = self.jnp, self.jax.lax
        rng = np.random.default_rng(16)

        def parts(*values):
            return [w for v in values for w in (v.real, v.imag)]

        def sums(x, y):
            z = lax.complex(x, y)
            return parts(jnp.sum(z, axis=1), jnp.sum(z, axis=0), jnp.cumsum(z, axis=1),
                         lax.cumsum(z, axis=0, reverse=True), jnp.sum(z))

        x, y = (jnp.asarray(rng.integers(-32, 33, (3, 40)) / 4, jnp.float32) for _ in range(2))
        report = self.verify(sums, x, y)
        self.assert_passes(report)
        a, b = (np.asarray(v, np.float64) for v in (x, y))
        wanted = [np.sum(v, axis=1) for v in (a, b)] + [np.sum(v, axis=0) for v in (a, b)]
        wanted += [np.cumsum(v, axis=1) for v in (a, b)]
        wanted += [np.flip(np.cumsum(np.flip(v, 0), axis=0), 0) for v in (a, b)]
        wanted += [np.sum(a), np.sum(b)]
        for got, want in zip(_outputs(report), wanted, strict=True):
            _same(self, got.double().flatten().tolist(), np.ravel(want).tolist())
        x, y = (jnp.asarray(rng.standard_normal((4, 32)) * 10.0 ** rng.integers(-3, 4, (4, 32)),
                            jnp.float32) for _ in range(2))

        def short(x, y):
            z = lax.complex(x, y)
            return parts(jnp.sum(z, axis=1), jnp.sum(z, axis=0))

        report = self.verify(short, x, y)
        self.assert_passes(report)

        def fold(rows):
            totals = []
            for row in rows:
                total = np.float32(0)
                for value in row:
                    total = np.float32(total + value)
                totals.append(float(total))
            return totals

        a, b = (np.asarray(v) for v in (x, y))
        _same(self, [w for v in _outputs(report) for w in v.double().flatten().tolist()],
              fold(a) + fold(b) + fold(a.T) + fold(b.T))
        nan = float("nan")
        left = [jnp.asarray(v, jnp.float32)
                for v in ([1, 1, 1, nan, 0.0, 3], [2, 2, 5, 0, -0.0, 4])]
        right = [jnp.asarray(v, jnp.float32)
                 for v in ([1, 7, 1, nan, -0.0, 4], [2, 2, 6, 0, 0, 5])]

        def compared(a, b, c, d):
            z, w = lax.complex(a, b), lax.complex(c, d)
            return z == w, z != w, z == 1 + 2j

        report = self.verify(compared, *left, *right)
        self.assert_passes(report)
        equal = [True, False, False, False, True, False]
        self.assertEqual([v.tolist() for v in _outputs(report)],
                         [equal, [not e for e in equal], [True, True, False, False, False, False]])

        def moved(x, y, p, out, dup, unique, start):
            z, w = lax.complex(x, y), lax.complex(-y, x + 1)
            return parts(jnp.where(p, z, w), z[unique], z.at[out].get(mode="fill"),
                         z.at[out].get(mode="fill", fill_value=1 - 2j),
                         lax.dynamic_slice(z, (start,), (2,)),
                         lax.dynamic_update_slice(z, w[:2], (start,)), z.at[unique].set(w[:4]),
                         z.at[dup].add(w[:4]), z.at[dup].subtract(w[:4]), jnp.array(z, copy=True))

        x, y = (jnp.asarray(rng.integers(-32, 33, 8) / 4, jnp.float32) for _ in range(2))
        inputs = (x, y, jnp.asarray(rng.integers(0, 2, 8), bool), jnp.asarray([0, 9, -9, 3]),
                  jnp.asarray([5, 1, 5, 7]), jnp.asarray([6, 0, 2, 3]), jnp.asarray(7, jnp.int32))
        report = self.verify(moved, *inputs)
        self.assert_passes(report)
        got = [v.double().flatten().tolist() for v in _outputs(report)]
        z = np.asarray(x, np.float64) + 1j * np.asarray(y, np.float64)
        w = -np.asarray(y, np.float64) + 1j * (np.asarray(x, np.float64) + 1)
        _same(self, got[0] + got[1], np.where(inputs[2], z, w).real.tolist() +
              np.where(inputs[2], z, w).imag.tolist())
        _same(self, got[4] + got[5], [z[0].real, nan, nan, z[3].real, z[0].imag, 0.0, 0.0,
                                      z[3].imag])
        _same(self, got[6] + got[7], [z[0].real, 1.0, 1.0, z[3].real, z[0].imag, -2.0, -2.0,
                                      z[3].imag])
        added = z.copy()
        np.add.at(added, [5, 1, 5, 7], w[:4])
        _same(self, got[14] + got[15], added.real.tolist() + added.imag.tolist())

    def test_windowed_extreme_transposes_compare(self):
        # The pullback of a windowed maximum or minimum binds select_and_scatter_add: the operand
        # padded with -inf or +inf, in each window the cell a left fold in row-major cell order
        # selects (the first of equal extremes), and each cotangent added to its window's cell in
        # row-major window order. With ties, NaNs, infinities, padding and overlapping windows,
        # it compares bit for bit with JAX.
        import numpy as np

        jax, jnp = self.jax, self.jnp
        lax = jax.lax
        rng = np.random.default_rng(5)
        x = rng.integers(-2, 3, (2, 5, 6)).astype(np.float32)
        x[0, 0, :3] = [np.nan, 1.0, np.nan]
        x[1, 2, 2:4] = [-np.inf, np.inf]

        def pullbacks(x, g, h):
            def pooled(window, strides, padding, combine, init):
                return lambda x: lax.reduce_window(x, init, combine, window, strides, padding)

            maximum = pooled((1, 2, 2), (1, 2, 2), "VALID", lax.max, -jnp.inf)
            overlapping = pooled((1, 3, 3), (1, 1, 1), "SAME", lax.max, -jnp.inf)
            minimum = pooled((2, 2, 3), (1, 2, 1), ((0, 1), (1, 0), (0, 0)), lax.min, jnp.inf)
            return (jax.vjp(maximum, x)[1](g[:, :2, :3])[0], jax.vjp(overlapping, x)[1](h)[0],
                    jax.vjp(minimum, x)[1](g[:, :3, :4])[0])

        names = {equation.primitive.name for equation in jax.make_jaxpr(pullbacks)(
            x, x[:, :3, :4], x).eqns}
        self.assertIn("select_and_scatter_add", names)
        for dtype in ("float32", "float64", "bfloat16", "float16"):
            with self.subTest(dtype=dtype), jax.enable_x64(dtype == "float64"):
                g = rng.standard_normal((2, 3, 4)) * 4
                h = rng.standard_normal(x.shape) * 4
                arguments = [jnp.asarray(value, dtype=dtype) for value in (x, g, h)]
                self.assert_passes(self.verify(pullbacks, *arguments))
        tags = _tags(self.artifact(pullbacks, x, x[:, :3, :4], x))
        self.assertEqual(tags.count("scatter"), 3)

    def test_windowed_extreme_transposes_select_operand_cells(self):
        # select_and_scatter_add with padding selects among a window's operand cells only, as
        # StableHLO's select_and_scatter does, so a padding cell neither ties with an infinite
        # extreme nor follows a NaN, and the cotangent of a window of padding alone is dropped.
        # That is JAX's default lowering, which XLA's CPU code runs too, and here the IR compares
        # bit for bit with it. XLA takes the cells of some windows padded on their last axis in
        # column-major order, so its selected cell can differ where a window's operand cells span
        # two rows and two columns and hold equal extremes or a NaN; no case below has such a
        # window. JAX's CPU and GPU lowering pads the operand with -inf or +inf instead
        # (expand_padding), and its words are checked apart: they differ in every case but the
        # fifth, whose first window holds padding alone.
        import numpy as np
        from jax._src.lax import windowed_reductions

        jax, jnp = self.jax, self.jnp
        lax = jax.lax
        inf, nan = np.inf, np.nan
        cases = [([-inf], (2,), (1,), ((1, 0),), "max"),
                 ([-inf, -inf], (3,), (1,), ((1, 1),), "max"),
                 ([0.0, nan], (2,), (1,), ((0, 1),), "max"),
                 ([inf, 2.0, inf], (2,), (1,), ((1, 1),), "min"),
                 ([3.0], (2,), (1,), ((2, 0),), "max"),
                 ([[-inf, -inf, 1.0], [nan, -inf, -inf]], (2, 2), (1, 2), ((1, 0), (1, 1)),
                  "max")]

        def lowered(expand, window, strides, padding, select):
            return jax.jit(lambda x, g: windowed_reductions._select_and_scatter_add_impl(
                g, x, select_prim=select, window_dimensions=window, window_strides=strides,
                padding=padding, expand_padding=expand))

        for index, (entries, window, strides, padding, extreme) in enumerate(cases):
            combine, init, select = ((lax.max, -jnp.inf, lax.ge_p) if extreme == "max"
                                     else (lax.min, jnp.inf, lax.le_p))
            pool = lambda x: lax.reduce_window(x, init, combine, window, strides, padding)
            pullback = lambda x, g: jax.vjp(pool, x)[1](g)[0]
            for dtype in ("float32", "float64"):
                with self.subTest(x=entries, dtype=dtype), jax.enable_x64(dtype == "float64"):
                    x = np.asarray(entries, dtype=dtype)
                    g = 2.0 ** np.arange(np.prod(pool(x).shape)).reshape(pool(x).shape)
                    g = g.astype(dtype)
                    names = {equation.primitive.name
                             for equation in jax.make_jaxpr(pullback)(x, g).eqns}
                    self.assertIn("select_and_scatter_add", names)
                    report = self.verify(pullback, x, g, checks=EXECUTE)
                    self.assert_passes(report)
                    default, padded = (np.asarray(lowered(expand, window, strides, padding,
                                                          select)(x, g)).flatten().tolist()
                                       for expand in (False, True))
                    _same(self, _outputs(report)[0].flatten().tolist(), default)
                    _same(self, np.asarray(pullback(x, g)).flatten().tolist(), padded)
                    self.assertEqual(index != 4, np.asarray(default).tobytes() !=
                                     np.asarray(padded).tobytes())

    def test_windowed_extreme_tangents(self):
        # The tangent of a windowed maximum or minimum binds select_and_gather_add: a
        # reduce_window over (operand, tangent) pairs from (-inf, 0) or (+inf, 0) that keeps the
        # accumulated pair where select(accumulated, cell) holds. As in reduce_window, padding and
        # the holes of a base dilation are init pairs and each window is folded in row-major cell
        # order: the tangent of the first of equal extremes, and 0 where the extreme is the init
        # value. Windows without padding, with ties, NaNs and infinities, and padded and dilated
        # windows of distinct entries compare bit for bit with JAX at every floating type. Padded
        # windows with ties and NaNs, where XLA's CPU loop skips the padding and takes some cells
        # in another order, are checked against that fold in NumPy.
        import numpy as np

        jax, jnp = self.jax, self.jnp
        lax = jax.lax
        rng = np.random.default_rng(6)
        ties = rng.integers(-2, 3, (2, 5, 6)).astype(np.float32)
        ties[0, 0, :3] = [np.nan, 1.0, np.nan]
        ties[0, 4, 5] = np.nan
        ties[1, :2, :2] = -np.inf
        ties[1, 2, 2:4] = [-np.inf, np.inf]
        distinct = (rng.permutation(60) - 30).reshape(2, 5, 6).astype(np.float32)
        none, unit = ((0, 0),) * 3, (1, 1, 1)
        plain = [((1, 2, 2), (1, 2, 2), none, "max", unit, unit),
                 ((2, 2, 3), unit, none, "min", unit, unit)]
        padded = [((1, 3, 3), unit, ((0, 0), (1, 1), (1, 1)), "max", unit, unit),
                  ((2, 2, 3), (1, 2, 1), ((0, 1), (1, 0), (0, 0)), "min", unit, unit),
                  ((1, 2, 2), (1, 1, 2), ((0, 0), (1, 1), (0, 1)), "max", (1, 3, 1), (1, 1, 2))]

        def tangents(cases):
            def pooled(x, t):
                outputs = []
                for window, strides, padding, extreme, base, dilation in cases:
                    combine, init = (lax.max, -jnp.inf) if extreme == "max" else (lax.min, jnp.inf)
                    pool = lambda x: lax.reduce_window(x, init, combine, window, strides, padding,
                                                       base, dilation)
                    outputs.append(jax.jvp(pool, (x,), (t,))[1])
                return tuple(outputs)
            return pooled

        for cases in (plain, padded):
            names = {equation.primitive.name
                     for equation in jax.make_jaxpr(tangents(cases))(ties, ties).eqns}
            self.assertIn("select_and_gather_add", names)
        for dtype in ("float32", "float64", "bfloat16", "float16"):
            with self.subTest(dtype=dtype), jax.enable_x64(dtype == "float64"):
                t = jnp.asarray(rng.standard_normal(ties.shape) * 4, dtype=dtype)
                for cases, x in ((plain, ties), (padded, distinct)):
                    self.assert_passes(self.verify(tangents(cases), jnp.asarray(x, dtype=dtype),
                                                   t))
        t = (rng.standard_normal(ties.shape) * 4).astype(np.float32)
        report = self.verify(tangents(padded), ties, t, checks=EXECUTE)
        self.assert_passes(report)
        pairs = np.empty(ties.shape, dtype=object)
        for index in np.ndindex(ties.shape):
            pairs[index] = (float(ties[index]), float(t[index]))
        for output, (window, strides, padding, extreme, base, dilation) in zip(_outputs(report),
                                                                               padded):
            if extreme == "max":
                init, keep = (-np.inf, 0.0), lambda a, b: a >= b
            else:
                init, keep = (np.inf, 0.0), lambda a, b: a <= b
            folded = _window_fold(pairs, init, lambda acc, cell: acc if keep(acc[0], cell[0])
                                  else cell, window, strides, padding, base, dilation)
            _same(self, output.flatten().tolist(), [value for _, value in folded])

    def test_elementary_exact_and_sorting_primitives(self):
        # The elementary functions execute, and floor, ceil, round to nearest even, rem and the
        # stable one-operand sort, which keeps -0 and +0 in order and puts NaN last, compare.
        jax, jnp = self.jax, self.jnp
        lax = jax.lax
        x = jnp.linspace(-2.0, 2.0, 24, dtype=jnp.float32).reshape(3, 8)
        x = x.at[0, :3].set(jnp.array([0.0, -0.0, jnp.nan]))

        def elementary(x):
            return (lax.expm1(x), lax.log1p(jnp.abs(x)), lax.tan(x / 4), jnp.sinh(x),
                    jnp.cosh(x), jnp.arcsinh(x), jnp.arccosh(jnp.abs(x) + 1),
                    jnp.arctanh(x / 4), jnp.exp2(x))

        def exact(x):
            return (jnp.floor(x * 3), jnp.ceil(x * 3), jnp.round(x * 2.5),
                    lax.rem(x * 5, jnp.asarray(1.5, x.dtype)), jnp.sort(x, axis=-1),
                    jnp.sort(x, axis=0))

        for dtype in (jnp.float32, jnp.bfloat16):
            with self.subTest(dtype=dtype):
                self.assert_passes(self.verify(elementary, x.astype(dtype), checks=EXECUTE))
        for dtype in (jnp.float32, jnp.float16):
            with self.subTest(dtype=dtype):
                self.assert_passes(self.verify(exact, x.astype(dtype)))
        tags = _tags(self.artifact(elementary, x)) + _tags(self.artifact(exact, x))
        for tag in ("expm1", "log1p", "tan", "sinh", "cosh", "asinh", "acosh", "atanh", "exp2",
                    "floor", "ceil", "round", "fmod", "sort"):
            self.assertIn(tag, tags)

    def test_sort_permutes_every_operand_alike(self):
        # lax.sort of several operands orders each fiber lexicographically by the first num_keys
        # operands, the first primary, and gives every operand that permutation (lax.sort's
        # docstring), as from jnp.argsort (along each axis, and descending, where jnp.argsort's
        # source keeps equal entries in input order), lax.sort_key_val, two keys of two formats
        # with integer and bool operands, jnp.lexsort and the JVP of jnp.sort, and of int32,
        # uint32 and bool keys, compared as they are, alone or before a floating one. The
        # reference is NumPy's stable lexsort of the keys, -0 as +0 and NaN last (first,
        # descending), the IR's words its permutation of the operands' words, and `compare`
        # requires XLA's, in binary32, binary16, bfloat16 and, as keys and as other operands,
        # float8_e5m2 and float8_e4m3fn, and of int64 and uint64 keys under x64, each its high and
        # then its low 32 bits.
        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(22)
        x = rng.integers(-2, 3, (3, 8)).astype(np.float32)
        x[0, :4] = [-0.0, 0.0, np.nan, -np.nan]
        x[2, 5:] = [np.nan, 0.0, -0.0]
        y = rng.integers(0, 2, (3, 8)).astype(np.float32)
        i = rng.integers(-3, 4, (3, 8)).astype(np.int32)
        # Integers that binary32 does not tell apart, out of order.
        i[0, :3] = [2**31 - 1, -2**31, 2**31 - 2]
        u = rng.choice([0, 1, 2**31, 2**32 - 1], (3, 8)).astype(np.uint32)
        u[0, :2] = [2**32 - 1, 2**32 - 2]

        def order(keys, axis):
            keys = [np.asarray(key, np.float64) + 0.0 for key in reversed(keys)]
            return np.lexsort(keys, axis=axis).astype(np.int32)

        def descending(x):
            x = np.asarray(x, np.float64)
            return order([~np.isnan(x), -x], 1)

        def permuted(operands, keys, axis):
            return [np.take_along_axis(np.asarray(a), order(keys, axis), axis) for a in operands]

        def gradient(x):
            # The derivative of sum(sort(x) * ramp) at x: ramp's entries at the sorted positions.
            g = np.zeros_like(x)
            np.put_along_axis(g, order([x], 1), np.arange(8, dtype=x.dtype)[None], 1)
            return g

        def check(function, inputs, expected):
            report = self.verify(function, *inputs)
            self.assert_passes(report)
            for k, array in enumerate(expected):
                words = array.astype(np.uint8) if array.dtype == bool else array.view(
                    f"uint{8 * array.itemsize}")
                got = np.asarray(report["outputs"][k]["bits"], np.uint64).ravel()
                self.assertEqual(got.tolist(), words.ravel().tolist(), k)

        cases = {
            "argsort": (lambda x, y, i, u: (jnp.argsort(x, axis=1), jnp.argsort(x, axis=0),
                                            jnp.argsort(x, axis=1, descending=True)),
                        lambda x, y, i, u: [order([x], 1), order([x], 0), descending(x)]),
            "sort_key_val": (lambda x, y, i, u: lax.sort_key_val(x, y, dimension=0),
                             lambda x, y, i, u: permuted([x, y], [x], 0)),
            "two keys": (lambda x, y, i, u: lax.sort((x, y.astype(jnp.float16), i, x > 0),
                                                     dimension=1, num_keys=2),
                         lambda x, y, i, u: permuted([x, y.astype(np.float16), i, x > 0], [x, y],
                                                     1)),
            "lexsort": (lambda x, y, i, u: jnp.lexsort((y, x), axis=1),
                        lambda x, y, i, u: [order([x, y], 1)]),
            "jvp": (lambda x, y, i, u: jax.grad(lambda x: jnp.sum(
                        jnp.sort(x, axis=1) * jnp.arange(8, dtype=x.dtype)))(x),
                    lambda x, y, i, u: [gradient(x)]),
            "integer keys": (lambda x, y, i, u: (jnp.sort(i, axis=0), jnp.argsort(u, axis=1),
                                                 jnp.sort(x > 0, axis=1),
                                                 *lax.sort((i, x, y), dimension=1, num_keys=2),
                                                 *lax.sort_key_val(u, x, dimension=0)),
                             lambda x, y, i, u: [*permuted([i], [i], 0), order([u], 1),
                                                 *permuted([x > 0], [x > 0], 1),
                                                 *permuted([i, x, y], [i, x], 1),
                                                 *permuted([u, x], [u], 0)]),
        }
        for name, (function, reference) in cases.items():
            for dtype in (jnp.float32, jnp.float16, jnp.bfloat16):
                inputs = (x.astype(dtype), y.astype(dtype), i, u)
                # Comparing bfloat16's NaN with 0 in NumPy warns.
                with self.subTest(case=name, dtype=dtype.__name__), np.errstate(invalid="ignore"):
                    check(function, inputs, reference(*inputs))
        for name in ("float8_e5m2", "float8_e4m3fn"):
            f8 = jnp.dtype(name)
            pairs = {"keys": (x.astype(f8), (y * 3 - 1).astype(f8)), "operand": (y, x.astype(f8))}
            for role, (a, b) in pairs.items():
                with self.subTest(dtype=name, role=role):
                    check(lambda a, b: lax.sort_key_val(a, b, dimension=1), (a, b),
                          permuted([a, b], [a], 1))
        tags = _tags(self.artifact(cases["two keys"][0], x, y, i, u))
        self.assertEqual(tags.count("sort"), 2)
        self.assertIn("gather", tags)
        with jax.enable_x64(True):
            # 64-bit keys: values that binary64 does not tell apart, ones that differ only in
            # their high or their low 32 bits (the top one of those among them), ties, and the
            # extremes.
            for dtype in (np.int64, np.uint64):
                info = np.iinfo(dtype)
                w = rng.integers(info.min, info.max, (4, 8), dtype=dtype, endpoint=True)
                w[0] = np.array([info.max, info.min, info.max - 1, 2**53 + 1, 2**53, 0, 2**32,
                                 2**32 - 1], dtype)
                w[1, :4] = w[1, 4:]
                w[2] = np.array([2**33 + 5, 2**33 + 4, 5, 2**34 + 4, 2**31, 2**31 - 1,
                                 2**32 + 2**31, 2**32 + 1], dtype)
                if info.min < 0:
                    w[3, :4] = [-1, -2**32, -2**32 - 1, -2**33]
                down = np.argsort(w, axis=0, kind="stable")
                with self.subTest(dtype=np.dtype(dtype).name):
                    check(lambda w: (jnp.sort(w, axis=1), jnp.argsort(w, axis=1),
                                     *lax.sort_key_val(w, w[::-1], dimension=0)), (w,),
                          [np.sort(w, axis=1, kind="stable"), np.argsort(w, axis=1, kind="stable"),
                           np.take_along_axis(w, down, 0), np.take_along_axis(w[::-1], down, 0)])

    def test_total_order_comparisons(self):
        # lax.eq_to, lt_to and le_to are StableHLO's compare in TOTALORDER for floating operands
        # and in the plain order for integer and Boolean ones (_compare_lower_hlo). The import
        # takes IEEE 754-2019's totalOrder: the order of each word read as a signed integer, its
        # bits other than the sign flipped where the sign is set, with a fnuz format's NaN, the
        # word 0x80, above every number. They are checked at every pair of float8, int8, uint8
        # and Boolean words, and of extreme, special and random words of the wider formats,
        # against that order and XLA's, which agrees but puts the fnuz NaN below every number,
        # as its sort puts it first. jnp.searchsorted by each method on both sides and
        # jnp.digitize reach them through JAX's comparators, which make -0 +0 and every NaN one
        # NaN, and are checked against NumPy at every integer width, Booleans and the floating
        # formats, as is jnp.histogram.
        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(23)
        compares = ((lax.eq_to_p, np.equal), (lax.lt_to_p, np.less),
                    (lax.le_to_p, np.less_equal))

        def compared(x, y):
            return tuple(primitive.bind(x, y) for primitive, _ in compares)

        def key(words, dtype):
            width = 8 * dtype.itemsize
            signed = words.view(f"int{width}").astype(np.int64)
            if dtype.kind in "biu":
                return signed if dtype.kind == "i" else words.astype(np.uint64)
            flipped = np.where(signed < 0, signed ^ (2 ** (width - 1) - 1), signed)
            return np.where(words == 0x80, 256, flipped) if dtype.name.endswith("fnuz") else flipped

        def words(name):
            dtype = jnp.dtype(name)
            width = 8 * dtype.itemsize
            if width == 8:
                return np.arange(256, dtype=np.uint8)
            sign = 1 << (width - 1)
            special = [0, 1, sign - 1, sign, sign | 1, 2 ** width - 1, 2 ** width - 2]
            if dtype.kind not in "iu":
                exponent, fraction = {"float16": (5, 10), "bfloat16": (8, 7), "float32": (8, 23),
                                      "float64": (11, 52)}[name]
                inf, quiet = ((1 << exponent) - 1) << fraction, 1 << (fraction - 1)
                # Of each sign: the greatest finite value, the infinity, quiet NaNs with payloads
                # 0 and 1, the signaling NaN of payload 1, the least normal value and the greatest
                # subnormal one.
                special += [s | v for s in (0, sign) for v in (
                    inf - 1, inf, inf | quiet, inf | quiet | 1, inf | 1, 1 << fraction,
                    (1 << fraction) - 1)]
            spread = rng.integers(0, 2 ** width - 1, 40, dtype=np.uint64, endpoint=True)
            return np.r_[np.array(special, np.uint64), spread].astype(f"uint{width}")

        for name in ("float16", "bfloat16", "float32", "float64", "float8_e5m2", "float8_e4m3fn",
                     "float8_e4m3fnuz", "float8_e5m2fnuz", "int8", "uint8", "int64", "uint64",
                     "bool"):
            dtype = jnp.dtype(name)
            w = np.array([0, 1], np.uint8) if name == "bool" else words(name)
            a, b = np.repeat(w, len(w)), np.tile(w, len(w))
            nan = (a == 0x80) != (b == 0x80) if name.endswith("fnuz") else np.zeros(len(a), bool)
            with jax.enable_x64(name.endswith("64")):
                x, y = (jnp.asarray(v.astype(bool) if name == "bool" else v.view(dtype))
                        for v in (a, b))
                for mode, run in (("eager", compared), ("jit", jax.jit(compared))):
                    with self.subTest(dtype=name, mode=mode):
                        report = self.verify(run, x, y, checks=EXECUTE)
                        self.assert_passes(report)
                        xla = run(x, y)
                        for k, (primitive, order) in enumerate(compares):
                            expected = order(key(a, dtype), key(b, dtype))
                            got = np.asarray(report["outputs"][k]["bits"]).astype(bool)
                            self.assertEqual(np.flatnonzero(got != expected).tolist(), [],
                                             primitive.name)
                            self.assertEqual((np.asarray(xla[k]) != expected).tolist(),
                                             (nan & (primitive is not lax.eq_to_p)).tolist(),
                                             primitive.name)

        def searches(a, v):
            return (*(jnp.searchsorted(a, v, side=side, method=method)
                      for side in ("left", "right")
                      for method in ("scan", "scan_unrolled", "compare_all", "sort")),
                    jnp.digitize(v, a), jnp.digitize(v, a, right=True))

        def expected(a, v):
            if a.dtype.kind not in "biu":
                a, v = a.astype(np.float64), v.astype(np.float64)
            return [np.searchsorted(a, v, side) for side in ("left", "right")
                    for _ in range(4)] + [np.digitize(v, a), np.digitize(v, a, right=True)]

        inf, nan = float("inf"), float("nan")
        for name in ("float16", "bfloat16", "float32", "float64", "int8", "int16", "int32",
                     "int64", "uint8", "uint16", "uint32", "uint64", "bool"):
            dtype = jnp.dtype(name)
            if dtype.kind == "b":
                a, v = np.array([False, False, True]), np.array([True, False, True, False])
            elif dtype.kind in "iu":
                info = np.iinfo(dtype)
                drawn = rng.integers(info.min, info.max, 6, dtype=dtype, endpoint=True)
                a = np.sort(np.r_[drawn, np.array([info.min, info.max, 0, 0], dtype)])
                v = np.r_[a[::-1], rng.integers(info.min, info.max, 6, dtype=dtype),
                          np.array([info.min + 1, info.max - 1], dtype)]
            else:
                a = np.array([-inf, -2.0, -0.0, 0.0, 0.5, 3.0, 3.0, inf]).astype(dtype)
                v = np.array([-0.0, 0.0, nan, -nan, -inf, inf, 3.0, 2.75, -2.0, 5.0, 0.5,
                              0.25]).astype(dtype)
            with jax.enable_x64(name.endswith("64")), np.errstate(invalid="ignore"):
                for mode, run in (("eager", searches), ("jit", jax.jit(searches))):
                    with self.subTest(case="search", dtype=name, mode=mode):
                        report = self.verify(run, jnp.asarray(a), jnp.asarray(v), checks=EXECUTE)
                        self.assert_passes(report)
                        xla = run(jnp.asarray(a), jnp.asarray(v))
                        for k, reference in enumerate(expected(a, v)):
                            got = np.asarray(report["outputs"][k]["bits"], np.int64)
                            self.assertEqual(got.tolist(), reference.tolist(), k)
                            self.assertEqual(np.asarray(xla[k]).tolist(), reference.tolist(), k)
        v = jnp.asarray(rng.uniform(-1, 4, 40).astype(np.float32))
        histogram = lambda v: jnp.histogram(v, bins=4, range=(0, 3))[0]
        report = self.verify(histogram, v, checks=EXECUTE)
        self.assert_passes(report)
        counts = np.asarray(histogram(v))
        self.assertEqual(report["outputs"][0]["bits"],
                         counts.view(f"uint{8 * counts.itemsize}").ravel().tolist())
        self.assertEqual(counts.tolist(), np.histogram(np.asarray(v), 4, (0, 3))[0].tolist())

    def test_round_away_from_zero_is_exact(self):
        # lax.round's default rounding method, AWAY_FROM_ZERO, takes of two integers at the same
        # distance the one farther from 0 (jax.lax.RoundingMethod). The reference is the integer
        # nearest the exact value, in Python's Fraction, with the operand's sign at a zero and the
        # operand itself at an infinity. The IR's words are the reference's, and so are XLA's,
        # eagerly and under jax.jit, at every word of binary16, bfloat16 and the float8 formats,
        # and in binary32 and binary64 at the halfway values from -40.5 to 40.5, the largest one
        # below 2^(p - 1), integers around 2^(p - 1) and 2^p, the neighbours of each, the specials
        # and 200 values spread over the exponents. A NaN matches any NaN.
        import math
        from fractions import Fraction

        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(29)

        def nearest(value):
            if not math.isfinite(value):
                return value
            return math.copysign(float(math.floor(abs(Fraction(value)) + Fraction(1, 2))), value)

        def words(array, dtype):
            array = np.asarray(array, dtype)
            unsigned = array.view(f"uint{8 * dtype.itemsize}")
            return [None if math.isnan(v) else int(w)
                    for v, w in zip(array.astype(np.float64).tolist(), unsigned.tolist())]

        def samples(dtype):
            info = jnp.finfo(dtype)
            if dtype.itemsize <= 2:
                return np.arange(2 ** info.bits).astype(f"uint{info.bits}").view(dtype)
            p = info.nmant + 1
            values = np.r_[np.arange(-40, 41) + 0.5, 2.0 ** (p - 1) - 0.5, 2.0 ** (p - 1),
                           2.0 ** (p - 1) + 1, 2.0 ** p, 2.0 ** p + 2]
            values = np.r_[values, -values].astype(dtype)
            specials = np.asarray([0.0, -0.0, np.inf, -np.inf, np.nan, info.max, -info.max,
                                   info.tiny, -info.tiny, info.smallest_subnormal,
                                   -info.smallest_subnormal], dtype)
            spread = rng.standard_normal(200) * 2.0 ** rng.integers(-4, p + 2, 200)
            return np.r_[values, np.nextafter(values, dtype.type(np.inf)),
                         np.nextafter(values, dtype.type(-np.inf)), specials,
                         spread.astype(dtype)]

        def check(x, got, expected, label):
            self.assertEqual(len(got), len(expected), label)
            wrong = [i for i, (a, b) in enumerate(zip(got, expected)) if a != b]
            self.assertFalse(wrong, (label, len(wrong), [
                (float(x[i]), got[i], expected[i]) for i in wrong[:4]]))

        for name in ("float16", "bfloat16", "float32", "float64", "float8_e5m2", "float8_e4m3fn",
                     "float8_e4m3fnuz", "float8_e5m2fnuz"):
            dtype = jnp.dtype(name)
            # Converting bfloat16's NaN words to binary64 warns.
            with jax.enable_x64(name == "float64"), np.errstate(invalid="ignore"):
                x = samples(dtype)
                expected = words([nearest(v) for v in x.astype(np.float64).tolist()], dtype)
                (equation,) = jax.make_jaxpr(lax.round)(x).eqns
                self.assertEqual(equation.params["rounding_method"],
                                 lax.RoundingMethod.AWAY_FROM_ZERO)
                for mode, run in (("eager", lax.round), ("jit", jax.jit(lax.round))):
                    with self.subTest(dtype=name, mode=mode):
                        report = self.verify(run, jnp.asarray(x), checks=EXECUTE)
                        self.assert_passes(report)
                        got = np.asarray(report["outputs"][0]["bits"], np.uint64).astype(
                            f"uint{8 * dtype.itemsize}").view(dtype)
                        check(x, words(got, dtype), expected, "IR")
                        check(x, words(run(jnp.asarray(x)), dtype), expected, "XLA")

    def test_argmax_and_argmin_match_eager(self):
        # lax.argmax and lax.argmin are the registered Op.argmax and Op.argmin, their index
        # results encoded in the primitive's index dtype. `compare` requires the eager words, at
        # repeated values, signed zeros and NaNs.
        import numpy as np

        jnp = self.jnp
        nan = np.nan
        x = np.array([[1.0, 3.0, 3.0, -0.0, 0.0], [nan, 2.0, nan, -1.0, 5.0],
                      [0.0, -0.0, 0.0, -0.0, -0.0], [7.0, -2.0, 7.0, nan, 1.0]],
                     dtype=np.float32)

        def extrema(x):
            return (jnp.argmax(x, axis=1), jnp.argmin(x, axis=0), jnp.argmax(x),
                    jnp.argmin(x, axis=1, keepdims=True))

        for dtype in (np.float32, jnp.bfloat16, np.float16):
            with self.subTest(dtype=dtype):
                self.assert_passes(self.verify(extrema, jnp.asarray(x, dtype=dtype)))
        records = _registered(self.artifact(extrema, jnp.asarray(x)))
        self.assertEqual({record["tag"] for record in records}, {"argmax", "argmin"})

    def test_sign_scans_and_top_k_match_eager(self):
        # lax.sign keeps -0 and NaN, and of integers is -1, 0 or 1 (the type's limits included).
        # cumsum and cumprod are the IR's left folds, compared where every partial result of any
        # grouping is exact and elsewhere equal word for word to NumPy's sequential binary32
        # cumsum. cummax and cummin are running IEEE extrema, the words exact at mixed signed
        # zeros and infinities, a NaN propagating. top_k is the stable topk, equal entries in
        # input order, for k = 0, k = n and is_stable=False too, and its execution fails on a
        # fiber holding a NaN or both zeros. `compare` checks the positions of NaNs and ignores
        # their words. Eager and one jax.jit computation.
        import numpy as np
        import torch

        jax, jnp = self.jax, self.jnp
        lax = jax.lax
        nan, inf = np.nan, np.inf
        signs = np.array([[-2.5, -0.0, 0.0, nan, 3.0, -inf], [inf, 1e-30, -1e-30, 7.0, -0.0, 0.0]])
        exact = np.array([[1.0, -2.0, 0.5, 4.0, -0.0, 3.0], [0.0, 2.0, -0.25, -0.0, 8.0, 1.0]])
        zeros = np.array([[0.0, -0.0, 0.0, -1.0, nan, 2.0], [-0.0, 0.0, -0.0, -3.0, -0.0, 0.0],
                          [-inf, nan, 1.0, nan, inf, -inf], [inf, -inf, -0.0, -inf, 0.0, nan]])
        # No fiber holds a NaN or both zeros, where the stable order and XLA's totalOrder agree.
        ranked = np.array([[3.0, 0.0, 3.0, -inf, 0.0, 1.0], [-1.0, -1.0, 2.0, -0.0, 2.0, -0.0]])

        def unit(s, e, z, r):
            return (jnp.sign(s), jnp.cumsum(e, axis=1), lax.cumprod(e, 1, reverse=True),
                    lax.cumsum(e, 0, reverse=True), lax.cummax(z, 1), lax.cummin(z, 1),
                    lax.cummax(z, 0, reverse=True), lax.cummin(z, 1, reverse=True),
                    *lax.top_k(r, 4), *lax.top_k(r, 1, axis=0), *lax.top_k(r, 0),
                    *lax.top_k(r, 6), *lax.top_k(r, 3, is_stable=False))

        samples = (signs, exact, zeros, ranked)
        for dtype in (jnp.float32, jnp.bfloat16, jnp.float16):
            with self.subTest(dtype=str(dtype)):
                arguments = [jnp.asarray(value, dtype) for value in samples]
                self.assert_passes(self.verify(unit, *arguments))
                self.assert_passes(self.verify(jax.jit(unit), *arguments))
        with jax.enable_x64(True):
            arguments = [jnp.asarray(value, jnp.float64) for value in samples]
            self.assert_passes(self.verify(unit, *arguments))
            self.assert_passes(self.verify(jax.jit(unit), *arguments))
        # A prefix of -0 entries sums to +0 from the fold's seed, as XLA's CPU code computes it.
        self.assert_passes(self.verify(lambda e: jnp.cumsum(e, axis=1),
                                       jnp.array([[-0.0, -0.0, 1.0, -0.0]])))
        for dtype in (jnp.int8, jnp.int16, jnp.int32, jnp.uint8, jnp.uint32):
            with self.subTest(dtype=str(dtype)):
                info = np.iinfo(dtype)
                values = np.array([[info.min, 0, info.max, 1], [info.max - 1, 0, info.min + 1, 7]])
                self.assert_passes(self.verify(jnp.sign, jnp.asarray(values, dtype)))
                self.assert_passes(self.verify(jax.jit(jnp.sign), jnp.asarray(values, dtype)))
        # Where a NaN or both zeros could decide the order, execution fails, but k = 0 decides none.
        for fiber in ([-0.0, 0.0, 1.0], [1.0, nan, 2.0], [-nan, 1.0, 0.0]):
            with self.subTest(fiber=fiber):
                self.assert_unsupported_execution(
                    self.verify(lambda r: lax.top_k(r, 1), jnp.array([fiber, [1.0, 2.0, 3.0]])),
                    "outside")
        self.assert_passes(self.verify(lambda r: lax.top_k(r, 0), jnp.array([-0.0, 0.0, nan])))
        # The NaN and zero test reads the operand's type, not that of the primitive before top_k.
        self.assert_passes(self.verify(lambda n, r: (n * 2, *lax.top_k(r, 2)), jnp.int32(3),
                                       jnp.asarray(ranked, jnp.float32)))
        # On general samples the left fold is NumPy's sequential cumsum, word for word.
        x = np.random.default_rng(5).standard_normal((3, 50)).astype(np.float32)
        report = self.verify(lambda x: jnp.cumsum(x, axis=1), x, checks=EXECUTE)
        self.assert_passes(report)
        (actual,) = _outputs(report)
        self.assertTrue(torch.equal(actual.view(torch.int32),
                                    torch.from_numpy(np.cumsum(x, axis=1)).view(torch.int32)))
        tags = _tags(self.artifact(unit, *[jnp.asarray(value, jnp.float32) for value in samples]))
        for tag in ("copysign", "select", "cumsum", "cumprod", "cummax", "cummin", "flip", "topk"):
            self.assertIn(tag, tags)

    def test_integer_and_boolean_top_k(self):
        # lax.top_k of integers at every width and of Booleans, false below true: the first k of
        # the stable ascending sort of each fiber's entries reversed by bitwise or logical not,
        # equal entries in input order. Checked word for word, with the element types, against
        # NumPy's stable argsort of the reversed entries, on fibers full of ties and the type's
        # limits, for k = 0, 1 and n, the first axis, and is_stable=False, where any order of ties
        # is admissible and the import keeps the stable one. XLA gives the same words, except for
        # the indices under is_stable=False, which are not compared. Eager and one jax.jit
        # computation.
        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(29)

        def unit(x):
            return (*lax.top_k(x, 3), *lax.top_k(x, 0), *lax.top_k(x, 1), *lax.top_k(x, 7),
                    *lax.top_k(x, 2, axis=0), *lax.top_k(x, 4, is_stable=False))

        def reference(x):
            def top(k, axis):
                order = np.argsort(~x, axis=axis, kind="stable").astype(np.int32)
                order = np.take(order, np.arange(k), axis=axis)
                return [np.take_along_axis(x, order, axis), order]
            return [*top(3, 1), *top(0, 1), *top(1, 1), *top(7, 1), *top(2, 0), *top(4, 1)]

        def words(array):
            array = np.asarray(array)
            if array.dtype == bool:
                return array.astype(np.uint8).ravel().tolist()
            return array.view(f"uint{8 * array.itemsize}").ravel().tolist()

        for name in ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
                     "bool"):
            dtype = np.dtype(name)
            if name == "bool":
                x = rng.integers(0, 1, (4, 7), endpoint=True).astype(bool)
            else:
                info = np.iinfo(dtype)
                limits = np.array([info.min, info.min + 1, info.max - 1, info.max, 0, 1], dtype)
                drawn = rng.integers(info.min, info.max, 3, dtype=dtype, endpoint=True)
                x = rng.choice(np.r_[limits, drawn], (4, 7))
            expected = reference(x)
            with jax.enable_x64(name.endswith("64")):
                for mode, run in (("eager", unit), ("jit", jax.jit(unit))):
                    with self.subTest(dtype=name, mode=mode):
                        report = self.verify(run, jnp.asarray(x), checks=EXECUTE)
                        self.assert_passes(report)
                        xla = run(jnp.asarray(x))
                        for k, array in enumerate(expected):
                            output = report["outputs"][k]
                            self.assertEqual(output["element_type"], array.dtype.name, k)
                            got = np.asarray(output["bits"], np.uint64)
                            self.assertEqual(got.ravel().tolist(), words(array), k)
                            if k < 10 or k % 2 == 0:
                                self.assertEqual(words(xla[k]), words(array), k)
        tags = _tags(self.artifact(unit, jnp.zeros((4, 7), jnp.int32)))
        for tag in ("bitwiseNot", "sort"):
            self.assertIn(tag, tags)

    def test_approx_top_k_is_the_stable_top_k(self):
        # approx_max_k and approx_min_k are the exact stable topk that XLA's fallback for
        # ApproxTopK defines: a stable sort under > or <, cut to k, or without aggregate_to_topk to
        # the extent JAX's shape rule gives (256 of 1024 entries, 128 with an input size override).
        # They compare with XLA where its rewrites of that sort agree with it: no tie at an
        # extreme kept alone (a reduction keeps a later one there) and no fiber holding both
        # zeros among the largest binary32 entries cut short (its TopK ranks in totalOrder).
        # There execution gives NumPy's stable order, -0 equal to +0. Execution fails on a fiber
        # holding a NaN.
        import numpy as np

        jax, jnp = self.jax, self.jnp
        lax = jax.lax
        rng = np.random.default_rng(7)
        distinct = (rng.permutation(72) - 36).astype(np.float64).reshape(2, 4, 9)
        ties = rng.integers(-3, 4, (2, 4, 9)).astype(np.float64)
        zeros = rng.choice([-0.0, 0.0, 1.0, -1.0], (2, 4, 9))
        wide = rng.permutation(1024).astype(np.float64)[None]

        def cut(x):
            return (*lax.approx_max_k(x, 3), *lax.approx_min_k(x, 2),
                    *lax.approx_max_k(x, 2, reduction_dimension=0),
                    *lax.approx_min_k(x, 3, reduction_dimension=1, aggregate_to_topk=False))

        def single(x):
            return *lax.approx_max_k(x, 1), *lax.approx_min_k(x, 1, reduction_dimension=1)

        def signed(x):
            return *lax.approx_min_k(x, 3), *lax.approx_max_k(x, 2, aggregate_to_topk=False)

        def spread(x):
            return (*lax.approx_max_k(x, 10, aggregate_to_topk=False),
                    *lax.approx_min_k(x, 10, aggregate_to_topk=False,
                                      reduction_input_size_override=4096))

        names = {equation.primitive.name for equation in jax.make_jaxpr(cut)(ties).eqns}
        self.assertEqual(names, {"approx_top_k"})
        self.assertEqual([out.shape for out in jax.eval_shape(spread, wide)],
                         [(1, 256), (1, 256), (1, 128), (1, 128)])
        mixed = [np.signbit(fiber[fiber == 0]) for fiber in zeros.reshape(8, 9)]
        self.assertTrue(any(signs.any() and not signs.all() for signs in mixed))
        for dtype in ("float32", "float64", "bfloat16", "float16"):
            with self.subTest(dtype=dtype), jax.enable_x64(dtype == "float64"):
                for function, x in ((cut, ties), (cut, distinct), (single, distinct),
                                    (signed, zeros), (spread, wide)):
                    self.assert_passes(self.verify(function, jnp.asarray(x, dtype=dtype)))
                self.assert_passes(self.verify(jax.jit(cut), jnp.asarray(ties, dtype=dtype)))

        def kept(x):
            return *lax.approx_max_k(x, 1), *lax.approx_min_k(x, 1), *lax.approx_max_k(x, 2)

        sample = zeros.astype(np.float32)
        report = self.verify(kept, sample, checks=EXECUTE)
        self.assert_passes(report)
        outputs = _outputs(report)
        for (k, sign), values, indices in zip(((1, -1), (1, 1), (2, -1)), outputs[::2],
                                              outputs[1::2]):
            order = np.argsort(sign * sample, axis=-1, kind="stable")[..., :k]
            self.assertEqual(indices.tolist(), order.tolist())
            _same(self, values.flatten().tolist(),
                  np.take_along_axis(sample, order, -1).flatten().tolist())
        for fiber in ([1.0, np.nan, 2.0], [-np.nan, 0.0, 1.0]):
            with self.subTest(fiber=fiber):
                self.assert_unsupported_execution(
                    self.verify(lambda r: lax.approx_min_k(r, 2),
                                jnp.array([fiber, [1.0, 2.0, 3.0]])), "outside")

    def test_cumlogsumexp_is_its_reduce_window(self):
        # cumlogsumexp is the reduce_window of lax's logaddexp that JAX lowers it to, each prefix
        # (each suffix for reverse) folded from -inf in row-major order: its execution is word
        # for word that of the fold written with jnp, at NaNs, infinities and zeros and along 40
        # entries too. Along up to 17 entries XLA's CPU code computes the same fold, its words
        # those of the jnp fold, eager or under jax.jit. The IR's and JAX's results are compared
        # within a tolerance of the format's precision, not word for word: XLA's exp and log1p
        # are not the IR's, and a difference of an ulp in them can grow where the logarithm
        # cancels max(a, b).
        import numpy as np

        jax, jnp = self.jax, self.jnp
        lax = jax.lax
        nan, inf = np.nan, np.inf

        def logaddexp(a, b):
            difference = a - b
            return jnp.where(difference != difference, a + b,
                             jnp.maximum(a, b) + jnp.log1p(jnp.exp(-jnp.abs(difference))))

        def fold(x, axis, reverse):
            x = jnp.moveaxis(x, axis, -1)
            extent = x.shape[-1]
            accumulated = jnp.full(x.shape, -jnp.inf, x.dtype)
            for j in range(extent):
                reached = np.arange(extent) <= j if reverse else np.arange(extent) >= j
                accumulated = jnp.where(reached, logaddexp(accumulated, x[..., j:j + 1]),
                                        accumulated)
            return jnp.moveaxis(accumulated, -1, axis)

        rng = np.random.default_rng(8)
        x = rng.standard_normal((3, 17)) * 3
        x[2, :8] = [nan, 1.0, -inf, inf, -0.0, 2.0, -inf, 0.0]
        y = np.array([[-inf, -inf, -0.0, 0.0, inf, inf], [inf, -inf, 1.0, -inf, 0.0, -0.0]])
        long = rng.standard_normal((2, 40)) * 3
        cases = [(x, 1, False), (x, 1, True), (x, 0, False), (y, 0, True), (y, 1, False),
                 (long, 1, False), (long, 1, True)]
        names = {equation.primitive.name
                 for equation in jax.make_jaxpr(lambda v: lax.cumlogsumexp(v, 1))(x).eqns}
        self.assertEqual(names, {"cumlogsumexp"})
        for dtype in ("float32", "float64", "bfloat16", "float16"):
            for value, axis, reverse in cases:
                with (self.subTest(dtype=dtype, shape=value.shape, axis=axis, reverse=reverse),
                      jax.enable_x64(dtype == "float64")):
                    operand = jnp.asarray(value, dtype=dtype)
                    scanned, folded = (
                        lambda v: lax.cumlogsumexp(v, axis, reverse=reverse),
                        lambda v: fold(v, axis, reverse))
                    reports = [self.verify(f, operand, checks=EXECUTE) for f in (scanned, folded)]
                    for report in reports:
                        self.assert_passes(report)
                    self.assertEqual(*map(_words, reports))
                    tolerance = {"float32": 1e-4, "float64": 1e-10, "bfloat16": 1e-1,
                                 "float16": 2e-2}[dtype]
                    np.testing.assert_allclose(
                        np.asarray(_outputs(reports[0])[0].tolist(), np.float64),
                        np.asarray(scanned(operand), np.float64), rtol=tolerance,
                        atol=tolerance)
                    if value.shape[axis] > 17:
                        continue
                    wanted = np.asarray(scanned(operand), np.float64).ravel().tolist()
                    for reference in (folded, jax.jit(folded)):
                        _same(self, np.asarray(reference(operand), np.float64).ravel().tolist(),
                              wanted)

    def test_rng_uniform_draws_from_torchleans_generator(self):
        # rng_uniform is Op.randUniform under the key of its node's position, as PyTorch's rand
        # is, scaled to the bounds: entry i is a + (b - a) * u_i in the dtype's arithmetic, u_i the
        # 32-bit draw splitmix64(key + i) mod 2^32 over 2^32 rounded to the dtype. XLA's stream
        # plays no part. The keys of two calls are splitmix64(0) plus distinct multiples of 2^32.
        # With a = -1 and b = 3, u_i and a + 4 u_i are integers over 2^32, rounded here exactly.
        import math

        jax, jnp = self.jax, self.jnp
        lax = jax.lax

        def rounded(n, dtype):
            # n / 2^32 for an integer n, rounded to nearest, ties to even, in the format dtype.
            info = jnp.finfo(dtype)
            digits = info.nmant + 1
            shift = max(abs(n).bit_length() - digits, 32 + int(info.minexp) - digits + 1, 0)
            q, r = divmod(abs(n), 1 << shift)
            half = (1 << shift) >> 1
            q += shift > 0 and (r > half or r == half and q % 2 == 1)
            return math.copysign(q * 2.0 ** (shift - 32), n)

        def draws(a, b):
            return lax.rng_uniform(a, b, (3, 4)), lax.rng_uniform(a, b, (5,))

        for dtype in ("float32", "float64", "bfloat16", "float16"):
            with self.subTest(dtype=dtype), jax.enable_x64(dtype == "float64"):
                bounds = jnp.asarray(-1.0, dtype), jnp.asarray(3.0, dtype)
                names = {equation.primitive.name
                         for equation in jax.make_jaxpr(draws)(*bounds).eqns}
                self.assertEqual(names, {"rng_uniform"})
                report = self.verify(draws, *bounds, checks=EXECUTE)
                self.assert_passes(report)
                keys = [record["fields"][1]["nat"]
                        for record in _records(self.artifact(draws, *bounds))
                        if record["tag"] == "randUniform"]
                self.assertEqual(len(keys), 2)
                self.assertEqual({key % 2 ** 32 for key in keys}, {_splitmix64(0) % 2 ** 32})
                self.assertNotEqual(*keys)
                for key, words in zip(keys, _words(report)):
                    units = [rounded(_splitmix64((key + i) % 2 ** 64) % 2 ** 32, dtype)
                             for i in range(len(words))]
                    wanted = [rounded(int((4 * unit - 1) * 2 ** 32), dtype) for unit in units]
                    self.assertEqual(words, _float_words(wanted, dtype))

    def test_rng_uniform_keeps_to_its_bounds(self):
        # Where b - a overflows, rng_uniform is 2 * (a / 2 + (b / 2 - a / 2) * u), whose halvings
        # and doubling are exact at such bounds, and an entry at b or above is the greatest value
        # below b, so every entry lies in [a, b), as JAX documents. In bfloat16, a draw rounds to
        # u = 1 with probability 2^-9, which gives b for a = 0 and b = 3, and 4 for a = -256,
        # where b - a = 259 rounds to 260. Bounds that are not finite or not ordered fail at
        # execution: JAX leaves the result undefined for b <= a.
        import numpy as np

        jax, jnp = self.jax, self.jnp
        lax = jax.lax

        def run(a, b, size):
            # The executed entries as binary64 values, their words, and the 32-bit draws.
            draws = lambda a, b: lax.rng_uniform(a, b, (size,))
            report = self.verify(draws, a, b, checks=EXECUTE)
            self.assert_passes(report)
            key = next(record["fields"][1]["nat"] for record in _records(self.artifact(draws, a, b))
                       if record["tag"] == "randUniform")
            return (_outputs(report)[0].double().numpy(), _words(report)[0],
                    [_splitmix64((key + i) % 2 ** 64) % 2 ** 32 for i in range(size)])

        for dtype, top in (("float32", 127), ("float64", 1023)):
            with self.subTest(dtype=dtype), jax.enable_x64(dtype == "float64"):
                a, b = (np.asarray(sign * 2.0 ** top, dtype=dtype) for sign in (-1, 1))
                values, words, draws = run(a, b, 64)
                units = (np.asarray(draws, dtype=np.float64) / 2 ** 32).astype(dtype)
                half = np.asarray(0.5, dtype=dtype)
                wanted = 2 * (a * half + (b * half - a * half) * units)
                wanted = np.where(wanted >= b, np.nextafter(b, a), wanted)
                self.assertEqual(words, _float_words(wanted.tolist(), dtype))
                self.assertTrue(np.isfinite(values).all() and (a <= values).all()
                                and (values < b).all())
        # A draw of 2^32 - 2^23 or more rounds to 1 in bfloat16, and its entry is then 3 - 2^-6,
        # the greatest bfloat16 value below 3.
        for low in (-256.0, 0.0):
            with self.subTest(low=low):
                values, _, draws = run(jnp.asarray(low, jnp.bfloat16),
                                       jnp.asarray(3.0, jnp.bfloat16), 4096)
                top = [value for value, draw in zip(values, draws) if draw >= 2 ** 32 - 2 ** 23]
                self.assertTrue(top)
                self.assertEqual(set(top), {3 - 2 ** -6})
                self.assertTrue(((low <= values) & (values < 3)).all())
        for a, b in ((3.0, -1.0), (1.0, 1.0), (np.nan, 1.0), (-np.inf, 0.0), (0.0, np.inf)):
            with self.subTest(a=a, b=b):
                self.assert_unsupported_execution(self.verify(
                    lambda a, b: lax.rng_uniform(a, b, (3,)), np.float32(a), np.float32(b),
                    checks=EXECUTE), "outside")

    def test_integer_rng_uniform_is_a_multiply_shift_of_the_draws(self):
        # With integer bounds of at most 32 bits, entry i of rng_uniform is
        # low + floor(d_i * (high - low) / 2^32), where d_i = splitmix64(key + i) mod 2^32 is the
        # word of the binary64 draw d_i / 2^32, so it lies in [low, high) as JAX documents. Bounds
        # with low >= high, which JAX leaves undefined, fail at execution; 64-bit bounds, which
        # one 32-bit word cannot cover, are refused.
        import numpy as np

        jax = self.jax
        lax = jax.lax
        cases = [("int32", -7, 5), ("int32", -2 ** 31, 2 ** 31 - 1), ("uint32", 0, 2 ** 32 - 1),
                 ("uint32", 2 ** 32 - 3, 2 ** 32 - 1), ("int16", -300, 301), ("uint8", 3, 250),
                 ("int8", -128, 127), ("int32", 4, 5)]
        draws = lambda a, b: lax.rng_uniform(a, b, (64,))
        for dtype, low, high in cases:
            with self.subTest(dtype=dtype, low=low, high=high):
                bounds = np.asarray(low, dtype), np.asarray(high, dtype)
                self.assertEqual({equation.primitive.name
                                  for equation in jax.make_jaxpr(draws)(*bounds).eqns},
                                 {"rng_uniform"})
                report = self.verify(draws, *bounds, checks=EXECUTE)
                self.assert_passes(report)
                key = next(record["fields"][1]["nat"]
                           for record in _records(self.artifact(draws, *bounds))
                           if record["tag"] == "randUniform")
                wanted = [low + (_splitmix64((key + i) % 2 ** 64) % 2 ** 32 * (high - low) >> 32)
                          for i in range(64)]
                self.assertTrue(all(low <= value < high for value in wanted))
                width = np.iinfo(dtype).bits
                self.assertEqual(_words(report)[0], [value % 2 ** width for value in wanted])
        for low, high in ((5, 5), (6, -2)):
            with self.subTest(low=low, high=high):
                self.assert_unsupported_execution(self.verify(
                    lambda a, b: lax.rng_uniform(a, b, (3,)), np.int32(low), np.int32(high),
                    checks=EXECUTE), "outside")
        with jax.enable_x64(True):
            self.assert_refused(self.verify(lambda a, b: lax.rng_uniform(a, b, (3,)),
                                            np.int64(0), np.int64(9), checks=("structure",)),
                                "wider than 32 bits")

    def test_runtime_indices_match_eager(self):
        # gather, dynamic_slice, scatter, scatter-add, scatter-sub and dynamic_update_slice at
        # runtime indices are one registered gather or scatter: starts clamped as XLA clamps
        # them, fill_value at a FILL_OR_DROP gather start outside, dropped scatter windows,
        # repeated scatter-add targets summed in index order (1e8 - 1e8 + 1 is 1, where
        # 1e8 + 1 - 1e8 is 0) and the last repeated scatter update kept, overlapping windows and
        # batching dimensions included.
        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        i = jnp.array([2, -1, 0, 5, -4], dtype=jnp.int32)
        j = jnp.array([[0, 3], [1, 1], [2, 0]], dtype=jnp.int32)
        rng = np.random.default_rng(0)
        u = (rng.standard_normal((2, 5, 4)) * 10.0 ** rng.integers(-2, 9, (2, 5, 4)))
        windows = lax.ScatterDimensionNumbers(update_window_dims=(0, 1), inserted_window_dims=(),
                                              scatter_dims_to_operand_dims=(0,))

        def reads(x, i, j, s):
            return (x[:, i], x[1, i[:3]], jnp.take(x, i, axis=2),
                    jnp.take_along_axis(x[0], j, axis=1), x.at[:, i].get(mode="clip"),
                    x.at[:, i].get(mode="fill", fill_value=-1.0),
                    lax.dynamic_slice(x, (s, s, 1), (1, 2, 3)),
                    lax.dynamic_index_in_dim(x, s, axis=2),
                    jax.vmap(lambda row, k: row[k])(x[0], j[:, 0]))

        def writes(x, i, s, u):
            sums = jnp.zeros(3, x.dtype).at[jnp.array([1, 1, 1, 0])].add(
                jnp.array([1e8, -1e8, 1.0, 5.0], x.dtype))
            overlap = lax.scatter_add(u[1, 1:, 2:], jnp.array([[0], [1], [2]]), u[:, :2, :3],
                                      windows)
            return (x.at[:, i].set(7.0), x.at[:, i].add(u), x.at[0, i].add(1e8),
                    x.at[:, i].set(7.0, mode="clip"), x.at[:, i].add(1.0, mode="drop"),
                    x.at[:, i].set(u), x.at[:, i].subtract(u), sums, overlap,
                    lax.dynamic_update_slice(x, u[:1, :2, :3], (s, s, 1)),
                    jax.vmap(lambda row, k: row.at[k].add(3.0))(x[0], j[:, 0]))

        for dtype in (jnp.float32, jnp.bfloat16):
            x = self.x.astype(dtype)
            for s in (1, 7, -9):
                start = jnp.int32(s)
                with self.subTest(dtype=str(dtype), start=s):
                    self.assert_passes(self.verify(reads, x, i, j, start))
                    self.assert_passes(self.verify(writes, x, i, start, jnp.asarray(u, dtype)))
        # A FILL_OR_DROP gather converts int64 indices to int32 first, so 2 ** 32 + 1 reads entry
        # 1 and 2 ** 40 entry 0, as in JAX.
        with jax.enable_x64(True):
            wide = jnp.array([2 ** 32 + 1, -1, 2 ** 40, 4], dtype=jnp.int64)
            self.assert_passes(self.verify(lambda x, k: (jnp.take(x, k, axis=2), x[:, k[1:2]]),
                                           self.x, wide))
        document = self.artifact(writes, self.x, i, jnp.int32(1), jnp.asarray(u, jnp.float32))
        self.assertEqual({record["tag"] for record in _registered(document)}, {"scatter"})

    def test_runtime_indices_of_integers_and_booleans(self):
        # Integer and Boolean gathers, scatters, dynamic_slice and dynamic_update_slice are the
        # registered gather and scatter at their element: sums, differences and products modulo
        # 2^width, the type's maximum and minimum, repeated targets in index order, starts
        # clamped, dropped windows, and a FILL_OR_DROP gather's fill value, which lax.gather
        # binds, by default the least signed value, the greatest unsigned one or True. A Boolean
        # maximum or minimum is a uint8 one of 0 and 1. Every integer width of both signs, eager
        # and one jax.jit computation. A fill value that the type does not hold is refused.
        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        i = jnp.array([2, -1, 0, 5, -4, 2], dtype=jnp.int32)
        rng = np.random.default_rng(5)

        def reads(x, i, s):
            return (x[:, i], jnp.take(x, i, axis=2), x.at[:, i].get(mode="clip"),
                    x.at[:, i].get(mode="fill"), x.at[:, i].get(mode="fill", fill_value=1),
                    lax.dynamic_slice(x, (s, s, s), (1, 2, 3)),
                    jax.vmap(lambda row, k: row[k])(x[0], i[:3]))

        def writes(x, i, s, u):
            return (x.at[:, i].set(u), x.at[:, i].add(u), x.at[:, i].subtract(u),
                    x.at[:, i].multiply(u), x.at[:, i].max(u), x.at[:, i].min(u),
                    x.at[:, i].add(u, mode="drop"), x.at[:, i].multiply(u, mode="drop"),
                    x.at[:, i].min(u, mode="clip"),
                    lax.dynamic_update_slice(x, u[:1, :2, :3], (s, s, s)))

        def choices(x, i, s, u):
            return (x.at[:, i].set(u), x.at[:, i].max(u), x.at[:, i].min(u),
                    x.at[:, i].set(u, mode="drop"), x.at[:, i].max(u, mode="drop"),
                    x.at[:, i].min(u, mode="clip"),
                    lax.dynamic_update_slice(x, u[:1, :2, :3], (s, s, s)))

        def check(dtype, s):
            if dtype == np.bool_:
                x, u, change = rng.random((2, 3, 4)) < 0.5, rng.random((2, 6, 4)) < 0.5, choices
            else:
                info = np.iinfo(dtype)
                x, u = (rng.integers(info.min, info.max, size, dtype=dtype, endpoint=True)
                        for size in ((2, 3, 4), (2, 6, 4)))
                change = writes
            with self.subTest(dtype=np.dtype(dtype).name, start=s):
                self.assert_passes(self.verify(reads, x, i, jnp.int32(s)))
                self.assert_passes(self.verify(change, x, i, jnp.int32(s), u))
                if dtype in (np.int8, np.bool_):
                    self.assert_passes(self.verify(jax.jit(change), x, i, jnp.int32(s), u))
            return x, u

        for dtype, s in zip((np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32,
                             np.bool_), (1, 7, -9, 1, 7, -9, 1)):
            x, u = check(dtype, s)
        with jax.enable_x64(True):
            check(np.int64, 7)
            check(np.uint64, -9)
        element = {"integer": {"signed": False, "width": 8}}
        document = self.artifact(choices, x, i, jnp.int32(1), u)
        self.assertEqual({(record["tag"], json.dumps(record["fields"][-1]["element"]))
                          for record in _registered(document)},
                         {("scatter", json.dumps("bool")), ("scatter", json.dumps(element))})
        self.assert_refused(self.verify(lambda x, i: x.at[:, i].get(mode="fill", fill_value=1.5),
                                        x.astype(np.int32), i, checks=("structure",)),
                            "where the type holds it")

    def test_runtime_scatter_products_match_eager(self):
        # scatter-mul at runtime indices is one registered Op.scatter with the rule prod. It
        # multiplies repeated targets in row-major order of the index vectors, in the clip and
        # drop modes too. Binary16 and bfloat16 products round at every step, as XLA's CPU code
        # does: the largest finite value times 2 times 0.5 overflows to inf. A NaN update
        # propagates. A select step replaces a dropped update by one before the scatter.
        # Binary64 too. Eager and one jax.jit computation.
        import numpy as np
        from torchlean._table import op_table

        jax, jnp = self.jax, self.jnp
        i = jnp.array([2, -1, 0, 5, -4, 2], dtype=jnp.int32)
        rng = np.random.default_rng(3)
        u = rng.uniform(-2.0, 2.0, (2, 6, 4))
        u[1, 0, 0] = np.nan
        u[0, 3, :] = np.inf

        def writes(x, i, u):
            return (x.at[:, i].multiply(u), x.at[0, i].multiply(2.0, mode="drop"),
                    x.at[:, i].multiply(u, mode="drop"), x.at[:, i].multiply(u, mode="clip"),
                    x.at[1, i[:3]].multiply(-u[0, :3]))

        def overflow(x, u):
            return x.at[jnp.array([0, 0, 0])].multiply(u)

        for dtype in (jnp.float32, jnp.bfloat16, jnp.float16):
            x = self.x.astype(dtype)
            with self.subTest(dtype=str(dtype)):
                self.assert_passes(self.verify(writes, x, i, jnp.asarray(u, dtype)))
                big = float(jnp.finfo(dtype).max)
                ones, factors = jnp.ones(2, dtype), jnp.array([big, 2.0, 0.5], dtype)
                self.assert_passes(self.verify(overflow, ones, factors))
                self.assert_passes(self.verify(jax.jit(overflow), ones, factors))
        with jax.enable_x64(True):
            self.assert_passes(self.verify(writes, self.x.astype(jnp.float64), i,
                                           jnp.asarray(u, jnp.float64)))
        steps = _steps(self.artifact(lambda x, i, u: x.at[:, i].multiply(u, mode="drop"), self.x,
                                     i, jnp.asarray(u, jnp.float32)))
        scatters = [step["op"] for step in steps
                    if "op" in step and step["op"]["record"]["tag"] == "scatter"]
        self.assertEqual([op["record"]["fields"][2] for op in scatters],
                         [{"nat": op_table()["codes"]["scatter"]["prod"]}])
        # The scatter's operands are the operand, the decoded indices and the updates.
        updates = steps[scatters[0]["inputs"][2][0]]
        self.assertEqual(updates["op"]["record"]["tag"], "select")

    def test_runtime_scatter_extrema_match_eager(self):
        # scatter-max and scatter-min at runtime indices are one registered Op.scatter with the
        # rules max and min, IEEE maximum and minimum as XLA's CPU code computes them.
        # max(-0, +0) and max(+0, -0) are +0 and min of both is -0. The compared words are
        # exact at zeros, so the sign of a zero counts. A NaN propagates through later updates.
        # The cases cover repeated targets, the clip and drop modes, binary16, bfloat16 and
        # binary64. Eager and one jax.jit computation.
        import numpy as np
        from torchlean._table import op_table

        jax, jnp = self.jax, self.jnp
        i = jnp.array([2, -1, 0, 5, -4, 2], dtype=jnp.int32)
        rng = np.random.default_rng(4)
        u = rng.uniform(-12.0, 12.0, (2, 6, 4))
        u[1, 0, 0] = np.nan

        def writes(x, i, u):
            return (x.at[:, i].max(u), x.at[:, i].min(u), x.at[0, i].max(2.0, mode="drop"),
                    x.at[:, i].min(u, mode="drop"), x.at[:, i].max(u, mode="clip"))

        def zeros(x, u):
            k = jnp.array([0, 1, 2, 2, 3])
            return x.at[k].max(u), x.at[k].min(u)

        signed = jnp.array([-0.0, 0.0, 1.0, np.nan])
        updates = jnp.array([0.0, -0.0, np.nan, 5.0, -0.0])
        for dtype in (jnp.float32, jnp.bfloat16, jnp.float16):
            with self.subTest(dtype=str(dtype)):
                self.assert_passes(self.verify(writes, self.x.astype(dtype), i,
                                               jnp.asarray(u, dtype)))
                self.assert_passes(self.verify(zeros, signed.astype(dtype), updates.astype(dtype)))
                self.assert_passes(self.verify(jax.jit(zeros), signed.astype(dtype),
                                               updates.astype(dtype)))
        with jax.enable_x64(True):
            self.assert_passes(self.verify(writes, self.x.astype(jnp.float64), i,
                                           jnp.asarray(u, jnp.float64)))
            self.assert_passes(self.verify(zeros, signed.astype(jnp.float64),
                                           updates.astype(jnp.float64)))
        rules = op_table()["codes"]["scatter"]
        document = self.artifact(writes, self.x, i, jnp.asarray(u, jnp.float32))
        self.assertEqual({record["fields"][2]["nat"] for record in _records(document)
                          if record["tag"] == "scatter"}, {rules["max"], rules["min"]})

    def test_runtime_index_windows_and_refusals(self):
        # Windows partly outside a length-four operand are dropped whole or clipped. Window rows
        # apply index vector by index vector ([0, 1e8, 0] gets 1 in the middle). Binary16 and
        # bfloat16 updates round at each step (2048 + 1 + 1 stays 2048 on XLA's CPU code).
        # int8 and uint32 indices and int8 starts at -1, -extent and -extent - 1 read as JAX
        # reads them, a uint32 2 ** 31 clipped to the last entry and filled in FILL_OR_DROP.
        # Raw gathers and scatters of an empty index batch or an empty window are registered
        # operators that read and write nothing. The windows and the rounding also run as one
        # jax.jit computation, compared with its compiled execution. .at[].apply, an operand
        # without entries, uint64 indices and a constant its declared type does not hold
        # (beyond an integer type's bounds, or overflowing to an infinity) are refused.
        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        modes = lax.GatherScatterMode
        rows = lax.ScatterDimensionNumbers(update_window_dims=(1,), inserted_window_dims=(),
                                           scatter_dims_to_operand_dims=(0,))
        columns = lax.ScatterDimensionNumbers(update_window_dims=(0,), inserted_window_dims=(),
                                              scatter_dims_to_operand_dims=(0,))
        last = lax.GatherDimensionNumbers(offset_dims=(0, 1), collapsed_slice_dims=(2,),
                                          start_index_map=(2,))
        line = jnp.array([1.0, 2.0, 3.0, 4.0])
        starts = jnp.array([[-1], [3], [1], [2]], dtype=jnp.int32)
        pieces = jnp.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]])

        def edges(line, starts, pieces):
            return tuple(scatter(line, starts, pieces, rows, mode=mode)
                         for scatter in (lax.scatter, lax.scatter_add)
                         for mode in (modes.FILL_OR_DROP, modes.CLIP)) + (
                lax.scatter_add(jnp.array([0.0, 1e8, 0.0]), jnp.array([[0], [1]]),
                                jnp.array([[0.0, 1.0], [-1e8, 0.0]]), columns),)

        def twice(base):
            return base.at[jnp.array([0, 0])].add(jnp.ones(2, base.dtype))

        def narrow(x, small, unsigned, s):
            return (jnp.take(x, small, axis=2), x[:, small[:2]],
                    lax.gather(x, unsigned[:, None], last, (2, 3, 1), mode=modes.CLIP),
                    lax.gather(x, unsigned[:, None], last, (2, 3, 1), mode=modes.FILL_OR_DROP,
                               fill_value=-1.0),
                    lax.dynamic_slice(x, (s, s, s), (1, 2, 3)))

        planes = lax.ScatterDimensionNumbers(update_window_dims=(0, 1), inserted_window_dims=(2,),
                                             scatter_dims_to_operand_dims=(2,))

        def empties(x, unsigned):
            none = unsigned[:0, None]
            return (lax.gather(x, none, last, (2, 3, 1), mode=modes.CLIP),
                    lax.gather(x, unsigned[:, None], last, (2, 0, 1), mode=modes.CLIP),
                    lax.scatter(x, none, jnp.ones((2, 3, 0)), planes),
                    lax.scatter_add(x, unsigned[:, None], jnp.ones((2, 0, 3)), planes))

        self.assert_passes(self.verify(edges, line, starts, pieces))
        self.assert_passes(self.verify(jax.jit(edges), line, starts, pieces))
        self.assertEqual(self.artifact(jax.jit(edges), line, starts, pieces)[
            "reference_execution"], "jit")
        for dtype, value in ((jnp.float16, 2048.0), (jnp.bfloat16, 256.0)):
            with self.subTest(dtype=str(dtype)):
                self.assert_passes(self.verify(twice, jnp.array([value, 0.0], dtype)))
                self.assert_passes(self.verify(jax.jit(twice), jnp.array([value, 0.0], dtype)))
        small = jnp.array([-1, 2, -4, 3], dtype=jnp.int8)
        unsigned = jnp.array([2 ** 31, 1, 3], dtype=jnp.uint32)
        for s in (-1, -4, -5):
            with self.subTest(start=s):
                self.assert_passes(self.verify(narrow, self.x, small, unsigned, jnp.int8(s)))
        self.assert_passes(self.verify(empties, self.x, unsigned))
        records = _registered(self.artifact(empties, self.x, unsigned))
        self.assertEqual(sorted(record["tag"] for record in records),
                         ["gather", "gather", "scatter", "scatter"])
        refused = {"apply": (lambda x, i: x.at[:, i].apply(lambda v: v + 1.0), self.x,
                             "a replacement, a sum, a difference, a product, a maximum"),
                   "empty operand": (lambda x, i: lax.scatter(
                       x[:, :0], i[:, None], jnp.ones((2, 0, 4)),
                       lax.ScatterDimensionNumbers((1, 2), (0,), (0,))), self.x,
                       "without entries")}
        for name, (function, operand, text) in refused.items():
            with self.subTest(name=name):
                self.assert_refused(self.verify(function, operand, jnp.array([1, 0]),
                                                checks=("structure",)), text)
        with jax.enable_x64(True):
            huge = jnp.array([1, 0], dtype=jnp.uint64)
            self.assert_refused(self.verify(
                lambda x, k: lax.gather(x, k[:, None], last, (2, 3, 1), mode=modes.CLIP),
                self.x, huge, checks=("structure",)), "other than uint64")
        from types import SimpleNamespace

        import numpy as np
        from torchlean.jaxpr import Unsupported, _Builder

        def declared(dtype):
            return SimpleNamespace(dtype=np.dtype(dtype), shape=())

        for value, dtype in ((np.int64(-1), np.uint64), (np.uint64(2 ** 63), np.int64),
                             (np.int32(-2 ** 31), np.float16), (np.float64(0.1), np.float32)):
            with self.subTest(constant=f"{value!r} as {np.dtype(dtype).name}"):
                with self.assertRaisesRegex(Unsupported, "declared element type"):
                    _Builder("float32").constant(value, declared(dtype))
        self.assertIsInstance(_Builder("float32").constant(np.int64(-2 ** 31), declared(np.int32)),
                              int)

    def test_sqrt_is_ieee_square_root(self):
        # lax.sqrt imports as Op.sqrt at `.raw`, IEEE squareRoot, with no condition on its operand:
        # -0 at -0, NaN below zero and at a NaN, inf at inf. XLA computes binary16 and bfloat16
        # sqrt in binary32 and rounds once, which for a square root is the correctly rounded
        # result, so every format agrees with XLA.
        import numpy as np

        jax, jnp = self.jax, self.jnp
        x = np.array([-0.0, 0.0, -1.0, np.inf, np.nan, 4.0, 2.0, 3.0])
        for dtype in (np.float32, jnp.bfloat16, np.float16):
            with self.subTest(dtype=dtype):
                self.assert_passes(self.verify(jax.lax.sqrt, jnp.asarray(x, dtype=dtype)))

    def test_log2_within_one_xla_ulp(self):
        # The IR's log2 is the binary dictionary's RN(RN(ln x) / RN(ln 2)), which can differ from
        # the correctly rounded log2, and binary16 and bfloat16 compute it in binary32 and round
        # once. On these samples XLA's CPU code (jax 0.12 on x86-64) is within one ulp of it in
        # each format (its bfloat16 log2 is not its binary32 log2 rounded once), so the words are
        # not compared.
        import numpy as np
        import torch

        jnp = self.jnp
        rng = np.random.default_rng(6)
        x = np.concatenate([np.abs(rng.standard_normal(60)) * 8, 2.0 ** np.arange(-6, 7),
                            [0.0, -0.0, -1.0, np.inf, np.nan, 1e-30, 3.0, 0.75]])
        for dtype in (np.float32, jnp.bfloat16, np.float16):
            with self.subTest(dtype=dtype):
                operand = jnp.asarray(x, dtype=dtype)
                report = self.verify(jnp.log2, operand, checks=EXECUTE)
                self.assert_passes(report)
                (actual,) = _outputs(report)
                wanted = torch.from_numpy(np.asarray(jnp.log2(operand)).astype(np.float32))
                _assert_within_ulps(self, actual, wanted.to(actual.dtype), 1)
        # JAX 0.11.1 has no log2 primitive: its jnp.log2 is log(x) / log(2).
        if hasattr(self.jax.lax, "log2"):
            self.assertIn("log2", _tags(self.artifact(jnp.log2, x.astype(np.float32))))

    def test_inverse_trigonometric_functions_within_xla_ulps(self):
        # The IR rounds asin, acos, atan and atan2 correctly. On these inputs, XLA's CPU code
        # (jax 0.12 on x86-64) is within 2, 2, 3 and 1 binary32 ulps of the IR and, computing
        # binary16 and bfloat16 in binary32, within one ulp in those formats.
        import numpy as np
        import torch

        jnp = self.jnp
        rng = np.random.default_rng(3)
        unit = np.concatenate([np.linspace(-1, 1, 41), [0.0, -0.0, 1.0, -1.0, 2.0, np.nan]])
        x = np.concatenate([rng.standard_normal(41) * 4, [0.0, -0.0, np.inf, -np.inf, 1e-30,
                                                          np.nan]])
        y = np.concatenate([rng.standard_normal(41) * 4, [-0.0, 1.0, np.inf, 2.0, -np.inf, 0.5]])

        def inverse(u, x, y):
            return jnp.arcsin(u), jnp.arccos(u), jnp.arctan(x), jnp.arctan2(y, x)

        for dtype, ulps in ((np.float32, (2, 2, 3, 1)), (jnp.bfloat16, (1, 1, 1, 1)),
                            (np.float16, (1, 1, 1, 1))):
            with self.subTest(dtype=dtype):
                arguments = [jnp.asarray(value, dtype=dtype) for value in (unit, x, y)]
                report = self.verify(inverse, *arguments, checks=EXECUTE)
                self.assert_passes(report)
                expected = [torch.from_numpy(np.asarray(value).astype(np.float32))
                            for value in inverse(*arguments)]
                for actual, wanted, bound in zip(_outputs(report), expected, ulps):
                    _assert_within_ulps(self, actual, wanted.to(actual.dtype), bound)
        tags = _tags(self.artifact(inverse, unit.astype(np.float32), x.astype(np.float32),
                                   y.astype(np.float32)))
        for tag in ("asin", "acos", "atan", "atan2"):
            self.assertIn(tag, tags)

    def test_real_models_import_without_unsupported_nodes(self):
        jax, jnp = self.jax, self.jnp
        lax = jax.lax
        keys = jax.random.split(jax.random.PRNGKey(0), 8)
        normal = lambda key, shape: 0.3 * jax.random.normal(key, shape, dtype=jnp.float32)
        w1, w2 = normal(keys[0], (8, 16)), normal(keys[1], (16, 4))
        k1, k2 = normal(keys[2], (4, 1, 3, 3)), normal(keys[3], (8, 4, 3, 3))
        w = normal(keys[4], (8, 3))
        qkv, proj = normal(keys[5], (8, 24)), normal(keys[6], (8, 8))

        def mlp(x):
            return jax.nn.log_softmax(jax.nn.relu(x @ w1) @ w2, axis=-1)

        def cnn(x):
            numbers = ("NCHW", "OIHW", "NCHW")
            x = jax.nn.relu(lax.conv_general_dilated(x, k1, (1, 1), ((1, 1), (1, 1)),
                                                     dimension_numbers=numbers))
            x = lax.conv_general_dilated(x, k2, (2, 2), "VALID", dimension_numbers=numbers)
            x = lax.reduce_window(jax.nn.relu(x), -jnp.inf, lax.max, (1, 1, 2, 2), (1, 1, 2, 2),
                                  "VALID")
            return jax.nn.softmax(x.reshape(x.shape[0], -1) @ w, axis=-1)

        def block(x):
            mean = jnp.mean(x, axis=-1, keepdims=True)
            normed = (x - mean) * lax.rsqrt(jnp.mean(jnp.square(x - mean), axis=-1,
                                                      keepdims=True) + 1e-5)
            q, k, v = jnp.split(normed @ qkv, 3, axis=-1)
            q, k, v = (y.reshape(4, 2, 4).transpose(1, 0, 2) for y in (q, k, v))
            scores = jnp.einsum("htd,hsd->hts", q, k) / 2.0
            y = jnp.einsum("hts,hsd->htd", jax.nn.softmax(scores, axis=-1), v)
            return x + y.transpose(1, 0, 2).reshape(4, 8) @ proj

        cases = {"mlp": (mlp, jnp.ones((2, 8))), "cnn": (cnn, jnp.ones((2, 1, 6, 6))),
                 "block": (block, jnp.ones((4, 8)))}
        for name, (function, x) in cases.items():
            with self.subTest(model=name):
                self.assert_passes(self.verify(function, x, checks=("structure", "types")))

    def test_operator_models_import_or_name_their_lane(self):
        # A CNN with kernel dilation and feature groups, a model of sorts and extrema, a
        # classifier's argmax, top_k of scores, prefix sums and a causal mask import.
        jax, jnp = self.jax, self.jnp
        lax = jax.lax
        keys = jax.random.split(jax.random.PRNGKey(1), 4)
        normal = lambda key, shape: 0.3 * jax.random.normal(key, shape, dtype=jnp.float32)
        k1, k2 = normal(keys[0], (4, 2, 3, 3)), normal(keys[1], (8, 2, 3, 3))
        w, w6 = normal(keys[2], (8 * 9, 3)), normal(keys[3], (6, 6))
        numbers = ("NCHW", "OIHW", "NCHW")

        def cnn(x):
            x = lax.conv_general_dilated(x, k1, (1, 1), ((2, 2), (2, 2)), rhs_dilation=(2, 2),
                                         dimension_numbers=numbers)
            x = lax.conv_general_dilated(jax.nn.relu(x), k2, (1, 1), ((1, 1), (2, 2)),
                                         rhs_dilation=(1, 2), feature_group_count=2,
                                         dimension_numbers=numbers)
            x = lax.reduce_window(jax.nn.relu(x), -jnp.inf, lax.max, (1, 1, 2, 2), (1, 1, 2, 2),
                                  "VALID")
            return x.reshape(x.shape[0], -1) @ w

        def ranks(x):
            h = jnp.tanh(x @ w6)
            s = jnp.sort(h, axis=-1)
            return (s - jnp.max(h, axis=-1, keepdims=True), jnp.floor(s * 4),
                    lax.expm1(jnp.min(h, axis=0)))

        def causal(x):
            i = jnp.arange(x.shape[0])
            mask = jnp.logical_and(i[None, :] <= i[:, None], i[:, None] >= 0)
            return jax.nn.softmax(jnp.where(mask, x @ x.T, -jnp.inf), axis=-1) @ x

        rows = jnp.linspace(-1.0, 1.0, 24, dtype=jnp.float32).reshape(4, 6)
        self.assert_passes(self.verify(cnn, jnp.ones((2, 2, 6, 6)), checks=("structure", "types")))
        self.assert_passes(self.verify(ranks, rows, checks=EXECUTE))
        self.assert_passes(self.verify(lambda x: jnp.argmax(x @ w6, axis=-1), rows,
                                       checks=EXECUTE))
        self.assert_passes(self.verify(causal, jnp.ones((4, 8)), checks=EXECUTE))
        self.assert_passes(self.verify(lambda x: lax.top_k(x @ w6, 2), rows, checks=EXECUTE))
        self.assert_passes(self.verify(lambda x: jnp.cumsum(x @ w6, axis=-1), rows,
                                       checks=EXECUTE))

    def test_mean_over_an_empty_axis_is_nan(self):
        # jnp.mean is a sum and a division by the count, so over an empty axis it is 0 / 0, NaN,
        # as in JAX. The IR mean, which the PyTorch capture refuses there, is not used.
        jnp = self.jnp
        empty = jnp.ones((3, 0), dtype=jnp.float32)
        self.assert_passes(self.verify(lambda x: jnp.mean(x, axis=1), empty))
        self.assertNotIn("mean", _tags(self.artifact(lambda x: jnp.mean(x, axis=1), empty)))

    def test_integer_reductions_and_scans_wrap_in_the_type(self):
        # reduce_sum, reduce_prod, reduce_max, reduce_min, cumsum, cumprod, cummax and cummin of
        # integers, as jnp and lax bind them, over one axis and two, reversed, over an empty axis,
        # eager and under jax.jit, are trees of the IR's add, mul, max and min in the type
        # (jaxpr._Builder.wrapping), and argmax and argmin of integers are the first position of
        # the extremum, found by such trees. argmax and argmin of Booleans compare binary64
        # values, and reduce_max and reduce_min of Booleans are reduce_or and reduce_and. In int8,
        # uint8, int16, uint16, int32 and uint32, and int64 and uint64 under x64, values drawn from
        # the whole range, with a row of the largest value and four smallest ones so that sums
        # wrap, and extremes next to values that binary32 does not tell apart from them, the IR's
        # words are those of NumPy's arithmetic in the type, whose sums and products wrap modulo
        # 2^width, and of its first extremum, and `compare` requires XLA's. jnp.nanvar, whose
        # count of entries is an int32 sum, matches NumPy's where every quotient is by 4 and
        # exact.
        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(43)

        def functions(v):
            empty = v[:, :0]
            results = (jnp.sum(v, axis=1, dtype=v.dtype), jnp.sum(v, axis=(0, 1), dtype=v.dtype),
                       jnp.prod(v, axis=0, dtype=v.dtype), jnp.max(v, axis=1), jnp.min(v, axis=0),
                       lax.cumsum(v, 1), lax.cumsum(v, 0, reverse=True), lax.cumprod(v, 1),
                       lax.cumprod(v, 1, reverse=True), lax.cummax(v, 1, reverse=True),
                       lax.cummin(v, 0), jnp.prod(empty, axis=1, dtype=v.dtype),
                       lax.reduce_max(empty, (1,)), lax.reduce_min(empty, (1,)))
            return results + (jnp.argmax(v, axis=1), jnp.argmin(v, axis=0),
                              jnp.count_nonzero(v, axis=1))

        def references(v, wide=False):
            def flipped(scan, axis):
                return np.flip(scan(np.flip(v, axis), axis), axis)

            info, rows = np.iinfo(v.dtype), v.shape[0]
            results = (v.sum(1, dtype=v.dtype), v.sum(dtype=v.dtype), v.prod(0, dtype=v.dtype),
                       v.max(1), v.min(0), np.cumsum(v, 1, dtype=v.dtype),
                       flipped(lambda w, axis: np.cumsum(w, axis, dtype=v.dtype), 0),
                       np.cumprod(v, 1, dtype=v.dtype),
                       flipped(lambda w, axis: np.cumprod(w, axis, dtype=v.dtype), 1),
                       flipped(np.maximum.accumulate, 1), np.minimum.accumulate(v, 0),
                       np.ones(rows, v.dtype), np.full(rows, info.min, v.dtype),
                       np.full(rows, info.max, v.dtype))
            index = np.int64 if wide else np.int32
            return results + (v.argmax(1).astype(index), v.argmin(0).astype(index),
                              np.count_nonzero(v, axis=1).astype(index))

        def booleans(p):
            return (jnp.max(p, axis=1), jnp.min(p, axis=0), jnp.argmax(p, axis=1),
                    jnp.argmin(p, axis=0), jnp.sum(p, axis=1))

        def boolean_references(p):
            return (p.max(1), p.min(0), p.argmax(1).astype(np.int32),
                    p.argmin(0).astype(np.int32), p.sum(1).astype(np.int32))

        def check(function, value, expected):
            for mode, run in (("eager", function), ("jit", jax.jit(function))):
                with self.subTest(dtype=value.dtype.name, mode=mode):
                    report = self.verify(run, value)
                    self.assert_passes(report)
                    self.assertEqual(len(report["outputs"]), len(expected))
                    for k, array in enumerate(expected):
                        array = np.asarray(array)
                        words = array.astype(np.uint8) if array.dtype == bool else array.view(
                            f"uint{8 * array.itemsize}")
                        got = np.asarray(report["outputs"][k]["bits"], np.uint64).ravel()
                        self.assertEqual(got.tolist(), words.ravel().tolist(), k)

        for dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64,
                      np.uint64):
            info, wide = np.iinfo(dtype), np.dtype(dtype).itemsize == 8
            v = rng.integers(info.min, info.max, (6, 7), dtype=dtype, endpoint=True)
            v[0], v[1, :4], v[2, :3] = info.max, info.min, 0
            v[3, :2], v[2:4, 5] = [info.max - 1, info.max], [info.min + 1, info.min]
            with jax.enable_x64(wide):
                check(functions, v, references(v, wide))
        p = rng.integers(0, 2, (5, 7)).astype(bool)
        p[0], p[1] = False, True
        check(booleans, p, boolean_references(p))
        x = rng.integers(-8, 9, (3, 5)).astype(np.float32)
        x[np.arange(3), [0, 2, 4]] = np.nan
        check(lambda x: jnp.nanvar(x, axis=1), x, [np.nanvar(x.astype(np.float64), axis=1)
                                                   .astype(np.float32)])

    def test_comparisons_and_selections(self):
        # lax comparisons and is_finite give Boolean steps, the logical operators of Booleans
        # stay logical, select_n is select with its cases in the other order, and
        # lax.clamp(lo, x, hi) is the IR clamp(x, lo, hi), a maximum and then a minimum. Integer
        # comparisons read the values, integer add, sub and mul wrap modulo 2 ** width, and
        # integer max, min and clamp follow the type's order.
        jnp, lax = self.jnp, self.jax.lax
        x = jnp.array([[jnp.nan, -0.0, 0.0, 1.5], [-jnp.inf, jnp.inf, -2.0, 30.0]],
                      dtype=jnp.float32)
        y = jnp.array([[1.0, 0.0, -0.0, 1.5], [-jnp.inf, 2.0, jnp.nan, 30.0]], dtype=jnp.float32)

        def boolean(x, y):
            p, q = x > 0, y < 1
            return (x == y, x != y, x < y, x <= y, x > y, x >= 0.25, jnp.isfinite(x),
                    jnp.logical_and(p, q), p | q, p ^ q, ~p, jnp.where(x > y, x, y),
                    lax.clamp(-1.0, x, 2.0))

        self.assert_passes(self.verify(boolean, x, y))
        # reduce_or, reduce_and and reduce_xor of Booleans: a binary64 count compared with zero,
        # its parity for xor. An empty fiber gives false, true and false.
        flags = jnp.array([[True, False, True], [False, False, False]])
        empty = jnp.zeros((2, 0), dtype=bool)
        self.assert_passes(self.verify(
            lambda b, e: (jnp.any(b, axis=1), jnp.all(b, axis=0), jnp.all(b),
                          lax.reduce_xor(b, (1,)), jnp.any(e, axis=1), jnp.all(e, axis=1),
                          lax.reduce_xor(e, (1,))), flags, empty))
        tags = set(_tags(self.artifact(boolean, x, y)))
        for tag in ("eq", "isFinite", "logicalAnd", "logicalNot", "select", "max", "min"):
            self.assertIn(tag, tags)
        def integers(i, j):
            one = jnp.ones((), dtype=i.dtype)
            return (i == j, i != j, i < j, i <= 2, i > j, i >= j, i + j, i - j, i * j,
                    jnp.maximum(i, j), jnp.minimum(i, j), lax.clamp(one, i, j),
                    jnp.where(i < j, i, j))

        for dtype in (jnp.int8, jnp.uint8, jnp.int32):
            info = jnp.iinfo(dtype)
            i = jnp.array([info.max, info.min, 5, 0, 2, 7], dtype=dtype)
            j = jnp.array([1, info.max, 5, 3, 2, 1], dtype=dtype)
            with self.subTest(dtype=str(dtype)):
                self.assert_passes(self.verify(integers, i, j))
        # Boolean comparisons, false below true as in NumPy, at 0 and 1 in binary64, of every
        # pair of values, and jnp.count_nonzero of Booleans, which compares them with False.
        import numpy as np

        a = np.array([[False, False, True, True], [True, True, False, False]])
        b = np.array([[False, True, False, True], [True, False, True, False]])
        report = self.verify(lambda a, b: (a == b, a != b, a < b, a <= b, a > b, a >= b,
                                           jnp.count_nonzero(b, axis=1)), a, b)
        self.assert_passes(report)
        expected = (a == b, a != b, a < b, a <= b, a > b, a >= b, np.array([2, 2]))
        for k, want in enumerate(expected):
            got = np.asarray(report["outputs"][k]["bits"], np.uint64).ravel().tolist()
            self.assertEqual(got, want.astype(np.uint64).ravel().tolist(), k)

    def test_unsupported_primitives_are_refused(self):
        lax = self.jax.lax
        cases = {"unstable sort": (lambda x: lax.sort((x, x), is_stable=False)[1],
                                   "only a stable sort"),
                 "accuracy request": (lambda x: lax.exp(x, accuracy=lax.Tolerance(
                     atol=1e-3, rtol=0.0, ulps=0)), "accuracy request")}
        for name, (function, text) in cases.items():
            with self.subTest(name=name):
                self.assert_refused(self.verify(function, self.x, checks=("structure",)), text)

    def test_windowed_reductions_fold_the_init_value(self):
        # reduce_window_max, _min, _sum and reduce_window with a mul or a max body: the operand
        # padded with the init value (and base dilation's init values between entries), each
        # window folded from the init value in row-major cell order, max and min the IEEE maximum
        # and minimum (+0 above -0 whatever their order, a NaN propagated, -inf for a window of
        # padding only), integers wrapping. The reference is that fold, entry by entry, in
        # NumPy's binary32 arithmetic and in Python integers modulo 2^32.
        import math

        import numpy as np

        jnp, lax = self.jnp, self.jax.lax
        reference = _window_fold
        f32 = np.float32
        add = lambda a, b: float(f32(a) + f32(b))
        mul = lambda a, b: float(f32(a) * f32(b))
        x = np.arange(60, dtype=np.float32).reshape(2, 5, 6) / 7 - 4
        x[0, 0, :4] = [-0.0, 0.0, 0.0, -0.0]
        x[1, 2, 3], x[1, 4, 0], x[0, 3, 5] = np.nan, -np.inf, np.inf
        window, strides = (1, 2, 3), (1, 2, 1)
        padding, base, dilation = ((0, 0), (1, 2), (2, 0)), (1, 1, 2), (1, 2, 1)

        def windows(x):
            fold = lambda init, body, **k: lax.reduce_window(
                x, jnp.float32(init), body, window, strides, padding, base_dilation=base,
                window_dilation=dilation, **k)
            return (fold(-np.inf, lax.max), fold(np.inf, lax.min), fold(0, lax.add),
                    fold(1, lax.mul), fold(0.5, lax.max),
                    lax.reduce_window(x, -jnp.inf, lax.max, (1, 1, 3), (1, 1, 3),
                                      ((0, 0), (0, 0), (4, 4))))

        document = self.artifact(windows, x)
        self.assertNotIn("pool", _tags(document))
        # jax keeps a base- or window-dilated monoid as the general reduce_window. Without
        # dilation the monoids are reduce_window_max, _min and _sum.
        primitives = str(self.jax.make_jaxpr(windows)(x))
        self.assertIn("reduce_window[", primitives)
        self.assertIn("reduce_window_max", primitives)
        # (an init value that is a Python number, which jax compares with the identity).
        plain = lambda x: (lax.reduce_window(x, np.inf, lax.min, window, strides, padding),
                           lax.reduce_window(x, 0.0, lax.add, window, strides, padding))
        names = str(self.jax.make_jaxpr(plain)(x))
        self.assertIn("reduce_window_min", names)
        self.assertIn("reduce_window_sum", names)
        report = self.verify(plain, x, checks=EXECUTE)
        self.assert_passes(report)
        flat = np.asarray([float(v) for v in x.reshape(-1)], dtype=object).reshape(x.shape)
        for output, (init, combine) in zip(_outputs(report), ((math.inf, _ieee_minimum),
                                                               (0.0, add))):
            _same(self, output.flatten().tolist(),
                  reference(flat, init, combine, window, strides, padding, (1, 1, 1), (1, 1, 1)))
        report = self.verify(windows, x, checks=EXECUTE)
        self.assert_passes(report)
        flat = np.asarray([float(v) for v in x.reshape(-1)], dtype=object).reshape(x.shape)
        general = (window, strides, padding, base, dilation)
        cases = [(-math.inf, _ieee_maximum, general), (math.inf, _ieee_minimum, general),
                 (0.0, add, general), (1.0, mul, general), (0.5, _ieee_maximum, general),
                 (-math.inf, _ieee_maximum, ((1, 1, 3), (1, 1, 3), ((0, 0), (0, 0), (4, 4)),
                                             (1, 1, 1), (1, 1, 1)))]
        for output, (init, combine, shape) in zip(_outputs(report), cases):
            _same(self, output.flatten().tolist(), reference(flat, init, combine, *shape))
        # The first window of each row of the last case holds padding only.
        self.assertEqual(_outputs(report)[5].flatten().tolist()[::4], [-math.inf] * 10)
        zeros = np.array([-0.0, 0.0, 0.0, -0.0], dtype=np.float32)
        pairs = lambda z: (lax.reduce_window(z, -jnp.inf, lax.max, (2,), (2,), "VALID"),
                           lax.reduce_window(z, jnp.inf, lax.min, (2,), (2,), "VALID"))
        report = self.verify(pairs, zeros, checks=EXECUTE)
        self.assert_passes(report)
        _same(self, [v for t in _outputs(report) for v in t.tolist()], [0.0, 0.0, -0.0, -0.0])
        # A negative padding crops the operand, and cropping past it leaves no window.
        crops = lambda z: (lax.reduce_window(z, -jnp.inf, lax.max, (1, 2), (1, 1),
                                             ((0, 0), (-2, 1))),
                           lax.reduce_window(z, 0.0, lax.add, (1, 1), (1, 1), ((0, 0), (-6, 0))))
        report = self.verify(crops, x[0], checks=EXECUTE)
        self.assert_passes(report)
        _same(self, _outputs(report)[0].flatten().tolist(),
              reference(np.asarray([float(v) for v in x[0].reshape(-1)], dtype=object)
                        .reshape(5, 6)[:, 2:], -math.inf, _ieee_maximum, (1, 2), (1, 1),
                        ((0, 0), (0, 1)), (1, 1), (1, 1)))
        self.assertEqual(tuple(_outputs(report)[1].shape), (5, 0))
        # lax.reduce with a lambda body and the identity init is the IR reduce.
        report = self.verify(lambda z: lax.reduce(z, 0.0, lambda a, b: lax.add(a, b), (1,)),
                             x[0], checks=EXECUTE)
        self.assert_passes(report)
        sums = []
        for row in x[0]:
            total = f32(0)
            for value in row:
                total = f32(total + value)
            sums.append(float(total))
        _same(self, _outputs(report)[0].flatten().tolist(), sums)
        # Integers: the type's order and wrapping sums and products.
        i = (np.arange(24, dtype=np.int64).reshape(4, 6) * 0x2345679 - 2 ** 31).astype(np.int32)
        wrap = lambda value: (value + 2 ** 31) % 2 ** 32 - 2 ** 31

        def integers(i):
            return (lax.reduce_window(i, jnp.int32(-2 ** 31), lax.max, (2, 3), (1, 2),
                                      ((1, 0), (0, 1))),
                    lax.reduce_window(i, jnp.int32(0), lax.add, (2, 2), (2, 1), "VALID"),
                    lax.reduce_window(i, jnp.int32(1), lax.mul, (1, 3), (1, 1), "VALID"))

        report = self.verify(integers, i, checks=EXECUTE)
        self.assert_passes(report)
        ints = np.asarray([int(v) for v in i.reshape(-1)], dtype=object).reshape(i.shape)
        expected = [reference(ints, -2 ** 31, max, (2, 3), (1, 2), ((1, 0), (0, 1)), (1, 1),
                              (1, 1)),
                    reference(ints, 0, lambda a, b: wrap(a + b), (2, 2), (2, 1), ((0, 0), (0, 0)),
                              (1, 1), (1, 1)),
                    reference(ints, 1, lambda a, b: wrap(a * b), (1, 3), (1, 1), ((0, 0), (0, 0)),
                              (1, 1), (1, 1))]
        for words, wanted in zip(_words(report), expected):
            self.assertEqual(words, [v % 2 ** 32 for v in wanted])

    def test_reduce_precision_rounds_the_mantissa_and_clamps_the_exponent(self):
        # StableHLO's reduce_precision(x, e, m): the mantissa rounded to m bits, to nearest even,
        # at the operand's own exponent (the operand's subnormal spacing for its subnormals),
        # then an infinity of the sign where the exponent exceeds the range of e bits and a zero of
        # the sign below its normal range, and a NaN word unchanged. The reference rounds exact
        # rationals.
        import math
        from fractions import Fraction

        import numpy as np

        jnp, lax = self.jnp, self.jax.lax
        fields = {"float32": (8, 23, 127), "bfloat16": (8, 7, 127), "float16": (5, 10, 15)}

        def reference(value, name, e, m):
            width, fraction, bias = fields[name]
            if value != value or value == 0 or math.isinf(value):
                return value
            magnitude = Fraction(abs(value))
            exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
            if Fraction(2) ** exponent > magnitude:
                exponent -= 1
            if m < fraction:
                quantum = Fraction(2) ** (max(exponent, 1 - bias) - m)
                q, r = divmod(magnitude, quantum)
                q = int(q) + (1 if r > quantum / 2 or r == quantum / 2 and int(q) % 2 else 0)
                magnitude = q * quantum
                if magnitude >= 2 ** (bias + 1):
                    return math.copysign(math.inf, value)
            if e < width:
                largest = 2 ** (e - 1) - 1
                if magnitude >= Fraction(2) ** (largest + 1):
                    return math.copysign(math.inf, value)
                if magnitude < Fraction(2) ** (1 - largest):
                    return math.copysign(0.0, value)
            return math.copysign(float(magnitude), value)

        rng = np.random.default_rng(11)
        for dtype, settings in ((np.float32, ((5, 10), (8, 7), (4, 3), (1, 2), (11, 30),
                                              (6, 23), (8, 1))),
                                (jnp.bfloat16, ((5, 3), (3, 7), (8, 2))),
                                (np.float16, ((3, 2), (5, 4), (2, 10)))):
            name = np.dtype(dtype).name
            width, fraction, bias = fields[name]
            values = list(rng.standard_normal(40) * 2.0 ** rng.integers(-bias - fraction,
                                                                         bias, 40))
            for e, m in settings:
                largest = 2 ** (min(e, width) - 1) - 1
                for power in (largest + 1, 1 - largest, -bias + 1, -bias - fraction + 2):
                    for step in (-2.0 ** (-m - 2), -2.0 ** (-m - 1), 0.0, 2.0 ** (-m - 1)):
                        values.append(2.0 ** power * (1 + step))
            values += [0.0, -0.0, math.inf, -math.inf, 1.0, -3.5]
            operand = np.asarray(values, dtype=dtype)
            words = _float_words(operand, dtype)
            unsigned = {2: np.uint16, 4: np.uint32}[np.dtype(dtype).itemsize]
            nans = np.asarray({"float32": [0x7fc00001, 0x7f800001, 0xffc12345],
                               "bfloat16": [0x7fc1, 0x7f81, 0xffc3],
                               "float16": [0x7e01, 0x7c01, 0xfe23]}[name],
                              dtype=unsigned).view(dtype)
            operand = jnp.asarray(np.concatenate([operand, nans]))
            function = lambda x: tuple(lax.reduce_precision(x, e, m) for e, m in settings)
            with self.subTest(dtype=name):
                report = self.verify(function, operand, checks=EXECUTE)
                self.assert_passes(report)
                for (e, m), got in zip(settings, _words(report)):
                    expected = _float_words([reference(_float_of_word(w, dtype), name, e, m)
                                             for w in words], dtype)
                    self.assertEqual(got[:len(words)], expected, (e, m))
                    self.assertEqual(got[len(words):], _float_words(nans, dtype), (e, m))
        self.assert_refused(self.verify(lambda x: lax.reduce_precision(x, 8, 0),
                                        self.x, checks=("structure",)), "mantissa_bits = 0")

    def test_cube_root_and_nextafter(self):
        # cbrt is IEEE rootn(x, 3), rounded once in the operand's own format (the reference
        # rounds the exact cube root, certified by cubes of the midpoints), -0 at -0 and the
        # infinities at the infinities. nextafter is C's: a NaN where either operand is one, y
        # where x == y, else the adjacent value toward y (subnormals included), which the
        # reference computes on the words.
        import math

        import numpy as np

        jnp, lax = self.jnp, self.jax.lax
        rng = np.random.default_rng(12)
        for dtype in (np.float32, jnp.bfloat16, np.float16):
            name = np.dtype(dtype).name
            tiny = _float_of_word(1, dtype)
            values = np.concatenate([rng.standard_normal(30) * 4.0 ** rng.integers(-8, 8, 30),
                                     [tiny, -tiny, 3 * tiny, 8.0, -27.0, 0.0, -0.0, math.inf,
                                      -math.inf, math.nan, 1e-3, 1000.0]])
            x = np.asarray(values, dtype=dtype)
            y = np.asarray(np.concatenate([rng.standard_normal(30), [0.0, 0.0, -1.0, 8.0, -27.0,
                                                                     -0.0, 0.0, 0.0, math.inf,
                                                                     1.0, math.nan, 1000.0]]),
                           dtype=dtype)
            with self.subTest(dtype=name):
                report = self.verify(lambda x, y: (lax.cbrt(x), lax.nextafter(x, y)),
                                     jnp.asarray(x), jnp.asarray(y), checks=EXECUTE)
                self.assert_passes(report)
                roots, adjacent = _words(report)
                xs = [_float_of_word(w, dtype) for w in _float_words(x, dtype)]
                ys = [_float_of_word(w, dtype) for w in _float_words(y, dtype)]
                expected = _float_words([_cube_root(v, dtype) for v in xs], dtype)
                for got, want, v in zip(roots, expected, xs):
                    if v != v:
                        self.assertTrue(math.isnan(_float_of_word(got, dtype)))
                    else:
                        self.assertEqual(got, want, v)
                width = np.dtype(dtype).itemsize * 8
                for got, a, b, word in zip(adjacent, xs, ys, _float_words(x, dtype)):
                    value = _float_of_word(got, dtype)
                    if a != a or b != b:
                        self.assertTrue(math.isnan(value))
                    elif a == b:
                        self.assertEqual(got, _float_words([b], dtype)[0])
                    elif a == 0:
                        self.assertEqual(got, 1 | (1 << (width - 1) if b < 0 else 0))
                    else:
                        self.assertEqual(got, word + (1 if (a < b) == (a > 0) else -1), (a, b))
        self.assertIn("cbrt", _tags(self.artifact(lax.cbrt, self.x)))

    def test_integer_bit_operators_act_on_words(self):
        # and, or, xor and not of integers, the shifts (an amount of the width or more, a negative
        # one read as its unsigned word among them, shifts every bit out: zero, or the top bit in
        # every position for the arithmetic shift, even of an unsigned type), population_count,
        # clz and mulhi (the floor of the exact product over 2^width). The reference is Python
        # integer arithmetic on the words.
        import numpy as np

        jnp, lax = self.jnp, self.jax.lax
        rng = np.random.default_rng(13)
        for dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32):
            info, width = np.iinfo(dtype), np.dtype(dtype).itemsize * 8
            x = rng.integers(info.min, int(info.max) + 1, 40, dtype=np.int64)
            x[:4] = [info.min, info.max, 0, -1 if info.min < 0 else 1]
            k = rng.integers(0, width + 3, 40)
            k[:4] = [width, width + 1, -1 if info.min < 0 else 255, 0]
            x, k = x.astype(dtype), k.astype(dtype)
            mask = 1 << width
            word = lambda v: int(v) % mask
            signed = lambda w: w - mask if info.min < 0 and w >= mask // 2 else w

            def bits(x, k):
                return (x & k, x | k, x ^ k, ~x, lax.shift_left(x, k),
                        lax.shift_right_arithmetic(x, k), lax.shift_right_logical(x, k),
                        lax.population_count(x), lax.clz(x), lax.mulhi(x, k))

            expected = [[] for _ in range(10)]
            for a, b in zip(map(word, x), map(word, k)):
                top = (a >> (width - 1)) & 1
                for out, value in zip(expected, (
                        a & b, a | b, a ^ b, ~a, 0 if b >= width else a << b,
                        (mask - 1 if top else 0) if b >= width else
                        ((a - mask if top else a) >> b),
                        0 if b >= width else a >> b, bin(a).count("1"), width - a.bit_length(),
                        (signed(a) * signed(b)) >> width)):
                    out.append(value % mask)
            with self.subTest(dtype=np.dtype(dtype).name):
                report = self.verify(bits, jnp.asarray(x), jnp.asarray(k), checks=EXECUTE)
                self.assert_passes(report)
                for name, got, want in zip(("and", "or", "xor", "not", "shl", "sra", "srl",
                                            "popcnt", "clz", "mulhi"), _words(report), expected):
                    self.assertEqual(got, want, name)
        tags = _tags(self.artifact(lambda x: (x & 3, lax.clz(x)), jnp.arange(4, dtype=np.int32)))
        self.assertIn("bitwiseAnd", tags)
        self.assertIn("countLeadingZeros", tags)

    def test_bitcast_between_equal_widths_keeps_every_bit(self):
        # bitcast_convert_type between types of one width reads each word at the new type, NaN
        # payloads and signaling NaNs included. Other widths are refused.
        import numpy as np

        jnp, lax = self.jnp, self.jax.lax
        words = np.asarray([0x3f800000, 0x80000000, 0x7fc00001, 0x7f800001, 0xffffffff, 1],
                           dtype=np.uint32)
        floats = words.view(np.float32)
        halves = np.asarray([0x3c00, 0x8000, 0x7e01, 0x7c01, 0xffff, 1], dtype=np.uint16)

        def casts(f, h):
            as_int = lax.bitcast_convert_type(f, jnp.int32)
            return (as_int, lax.bitcast_convert_type(f, jnp.uint32),
                    lax.bitcast_convert_type(as_int, jnp.float32),
                    lax.bitcast_convert_type(h, jnp.float16),
                    lax.bitcast_convert_type(lax.bitcast_convert_type(h, jnp.bfloat16),
                                             jnp.int16))

        report = self.verify(casts, jnp.asarray(floats), jnp.asarray(halves), checks=EXECUTE)
        self.assert_passes(report)
        got = _words(report)
        for output in got[:3]:
            self.assertEqual(output, [int(w) for w in words])
        for output in got[3:]:
            self.assertEqual(output, [int(w) for w in halves])
        self.assert_refused(self.verify(lambda f: lax.bitcast_convert_type(f, jnp.int16),
                                        jnp.asarray(floats), checks=("structure",)),
                            "one word width")

    def test_rows_without_an_executable_meaning_are_refused(self):
        # rng_bit_generator, whose bits are a function of its key operand, which TorchLean's
        # random operators do not read, factorizations whose outputs the API leaves to the
        # implementation, lax.reduce with a non-identity init (its documentation calls the
        # result undefined) and a reduce_window body that is not one
        # monoid operation are refused, each by the general unsupported-primitive rule or the
        # rule's own reason.
        import numpy as np

        jnp, lax = self.jnp, self.jax.lax
        x = jnp.asarray(np.linspace(0.1, 0.9, 6, dtype=np.float32).reshape(2, 3))
        m = jnp.asarray(np.eye(3, dtype=np.float32) * 2)
        key = jnp.zeros(4, jnp.uint32)
        cases = {"rng_bit_generator": (
                     lambda x: x + lax.rng_bit_generator(key, x.shape)[1].astype(x.dtype), x),
                 "lu": (lambda m: lax.linalg.lu(m)[0], m), "qr": (lambda m: lax.linalg.qr(m)[0], m),
                 "eigh": (lambda m: lax.linalg.eigh(m)[1], m),
                 "reduce": (lambda x: lax.reduce(x, jnp.float32(0.5), lax.max, (1,)), x),
                 "reduce_window": (lambda x: lax.reduce_window(x, jnp.float32(0), lax.sub, (1, 2),
                                                               (1, 1), "VALID"), x)}
        for name, (function, operand) in cases.items():
            with self.subTest(name=name):
                report = self.verify(function, operand, checks=("structure",))
                self.assert_refused(report, {"rng_bit_generator": "its key operand",
                                             "reduce_window": "reduce_window",
                                             "reduce": "body's identity",
                                             "lu": "multiple primitive outputs",
                                             "qr": "multiple primitive outputs",
                                             "eigh": "multiple primitive outputs"}.get(
                                                 name, "unsupported primitive"))

    def test_calls_placements_and_markers_import(self):
        # The call of a traced function (eval_jaxpr), linear_call, under jax.vjp a custom VJP's
        # forward function with optimize_remat (remat_opt), and the high-level primitives
        # (call_hi_primitive) that checkpointing and custom derivatives bind under the jax_remat3,
        # jax_custom_vjp3 and jax_custom_jvp3 options import their bodies; sharding,
        # resharding and layout constraints and checkpoint_name (name) are their operand, tie is
        # its second operand and dce_sink gives nothing; jnp.empty's lax.empty and the empty2
        # residuals of a cond's gradient are zeros (jaxpr._CALLS, _IDENTITIES). Tokens, the
        # unreachable primitive and values held unreduced are refused.
        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        from jax._src.custom_derivatives import unreachable_p
        from jax._src.lax.lax import tie_p
        from jax.ad_checkpoint import checkpoint_name
        from jax.custom_derivatives import linear_call
        from jax.experimental.layout import Layout, with_layout_constraint

        x = jnp.asarray(np.linspace(-1.5, 2.5, 6, dtype=np.float32).reshape(2, 3))
        traced = jax.jit(lambda v: jnp.sin(v) * 2).trace(x)

        @jax.custom_vjp
        def sine(v):
            return jnp.sin(v)

        sine.defvjp(lambda v: (jnp.sin(v), jnp.cos(v)), lambda r, g: (r * g,),
                    optimize_remat=True)
        mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("d",))
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        branches = lambda v: lax.cond(v.sum() > 0, lambda u: jnp.sin(u) * u, jnp.cos, v)
        cases = {
            "eval_jaxpr": lambda v: traced(v) + 1,
            "linear_call": lambda v: linear_call(lambda r, u: r * u, lambda r, t: r * t, x, v) + 1,
            "remat_opt": lambda v: jax.vjp(sine, v)[0] * 3,
            "sharding_constraint": lambda v: lax.with_sharding_constraint(v * 2, replicated) + 1,
            "layout_constraint": lambda v: with_layout_constraint(v * 2, Layout((0, 1))) + 1,
            "name": lambda v: checkpoint_name(v * 2, "a") + 1,
            "tie": lambda v: tie_p.bind(v * 2, v + 1) * 3,
            "dce_sink": lambda v: (lax.dce_sink(v * 3), v + 1)[1],
            "empty": lambda v: jnp.empty(v.shape).at[:].set(v) + 1,
            "cond": jax.grad(lambda v: branches(v).sum()),
        }
        for name, function in cases.items():
            with self.subTest(name=name):
                equations = jax.make_jaxpr(function)(x).jaxpr.eqns
                self.assertIn(name, {e.primitive.name for e in equations})
                self.assert_passes(self.verify(function, x))
        # The last case's: the forward cond's branches.
        self.assertIn("empty2", {inner.primitive.name for e in equations
                                 if e.primitive.name == "cond" for branch in e.params["branches"]
                                 for inner in branch.jaxpr.eqns})
        from jax._src import config

        with config.remat3(True), config.custom_vjp3(True), config.custom_jvp3(True):
            @jax.custom_vjp
            def high(v):
                return jnp.sin(v)

            @jax.custom_jvp
            def low(v):
                return jnp.cos(v)

            high.defvjp(lambda v: (jnp.sin(v), jnp.cos(v)), lambda r, g: (r * g,))
            low.defjvp(lambda p, t: (jnp.cos(p[0]), -jnp.sin(p[0]) * t[0]))
            remat = jax.checkpoint(lambda v: jnp.sin(v) * v)
            cases = {"name": lambda v: checkpoint_name(v * 2, "a") + 1, "remat": remat,
                     "custom_vjp": lambda v: high(v) * 3, "custom_jvp": lambda v: low(v) * 3,
                     "grad": jax.grad(lambda v: remat(v).sum())}
            for name, function in cases.items():
                with self.subTest(name=name, primitive="call_hi_primitive"):
                    equations = jax.make_jaxpr(function)(x).jaxpr.eqns
                    self.assertIn("call_hi_primitive", {e.primitive.name for e in equations})
                    self.assert_passes(self.verify(function, x))
        stop = lambda v: v + unreachable_p.bind(v, out_avals=(jax.typeof(v),), exc_type=TypeError,
                                                message="unreachable")[0]
        token = lambda v: (lax.after_all(lax.create_token()), v + 1)[1]
        self.assert_refused(self.verify(stop, x, checks=("structure",)), "raises its error")
        self.assert_refused(self.verify(token, x, checks=("structure",)), "a token orders")
        explicit = jax.make_mesh((1,), ("d",), axis_types=(jax.sharding.AxisType.Explicit,))
        P = jax.sharding.PartitionSpec
        with jax.sharding.set_mesh(explicit):
            self.assert_passes(self.verify(lambda v: jax.sharding.reshard(v * 2, P("d")) + 1, x))
            parts = lambda v: jax.sharding.reshard(jax.sharding.reshard(v, P(unreduced={"d"})),
                                                   P())
            self.assert_refused(self.verify(parts, x, checks=("structure",)), "held unreduced")
