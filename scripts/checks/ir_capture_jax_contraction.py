"""JAX's contractions captured and checked with ``verify ir`` (``JaxContractionTests``): the
``ragged_dot_general``, ``scaled_dot`` and ``dot_general`` primitives that the Jaxpr importer reads
(``python/torchlean/_jax_contraction.py``), with float8, binary16, bfloat16, integer and Boolean
operands and preferred element types, and float8 convolutions. Run them through
``scripts/checks/test_ir_capture.py``, which describes the checks.
"""

import unittest

from ir_capture_support import (CHECKER, EXECUTE, HAS_JAX, CheckerCase, _outputs, _same,
                                _steps, _tags)
from ir_capture_jax_operators import JaxGeneralOperatorTests


@unittest.skipUnless(CHECKER and HAS_JAX, "needs the built verify executable and JAX")
class JaxContractionTests(CheckerCase):
    setUp, verify = JaxGeneralOperatorTests.setUp, JaxGeneralOperatorTests.verify
    artifact = JaxGeneralOperatorTests.artifact

    def test_ragged_dot_contracts_each_group(self):
        # lax.ragged_dot, and ragged_dot_general with a ragged free axis, is one contraction with
        # every group and one selection per group. Group j holds the positions from the sum of
        # the sizes before it to that sum plus its own size, and a position in no group is +0.
        # Integral operands make every contraction exact, so those compare bit for bit; random
        # ones agree within a tolerance. An infinity in one group leaves the others finite, where
        # JAX's CPU reference, contracting with every group masked to zero, gives NaN.
        import numpy as np

        jax, jnp = self.jax, self.jnp
        lax = jax.lax
        rng = np.random.default_rng(11)
        x, r = rng.integers(-4, 5, (6, 4)), rng.integers(-4, 5, (3, 4, 5))
        y, s = rng.integers(-4, 5, (2, 6, 4)), rng.integers(-4, 5, (3, 2, 4, 5))
        batched = lax.RaggedDotDimensionNumbers(dot_dimension_numbers=(([2], [2]), ([0], [1])),
                                                lhs_ragged_dimensions=[1],
                                                rhs_group_dimensions=[0])

        def ragged(x, r, y, s, n):
            return lax.ragged_dot(x, r, n), lax.ragged_dot_general(y, s, n, batched)

        integral = [jnp.asarray(value, jnp.float32) for value in (x, r, y, s)]
        for sizes in ([2, 1, 2], [2, 0, 4], [0, 6, 0], [7, 1, 0]):
            with self.subTest(sizes=sizes):
                self.assert_passes(self.verify(ragged, *integral, jnp.asarray(sizes, jnp.int32)))
        n = jnp.asarray([2, 1, 2], jnp.int32)
        normal = [jnp.asarray(rng.standard_normal(value.shape), jnp.float32)
                  for value in (x, r, y, s)]
        report = self.verify(ragged, *normal, n, checks=EXECUTE)
        self.assert_passes(report)
        for actual, wanted in zip(_outputs(report), ragged(*normal, n), strict=True):
            np.testing.assert_allclose(actual.numpy(), np.asarray(wanted), rtol=1e-5, atol=1e-6)
        infinite = r.astype(np.float32)
        infinite[2, 0, 0] = np.inf
        report = self.verify(lambda x, r, n: lax.ragged_dot(x, r, n), integral[0],
                             jnp.asarray(infinite), n, checks=EXECUTE)
        self.assert_passes(report)
        rows = _outputs(report)[0].numpy()
        _same(self, rows[:3].flatten().tolist(),
              np.concatenate([x[:2] @ r[0], x[2:3] @ r[1]]).astype(np.float32).flatten().tolist())
        _same(self, rows[5].tolist(), [0.0] * 5)
        # Sizes 3, -1, 2 give group 0 the positions 0 to 2 and group 2 the positions 2 and 3, so
        # position 2 takes group 0, where JAX's reference adds the two contractions.
        report = self.verify(lambda x, r, n: lax.ragged_dot(x, r, n), *integral[:2],
                             jnp.asarray([3, -1, 2], jnp.int32), checks=EXECUTE)
        self.assert_passes(report)
        _same(self, _outputs(report)[0].numpy()[2:4].flatten().tolist(),
              np.stack([x[2] @ r[0], x[3] @ r[2]]).astype(np.float32).flatten().tolist())
        tags = _tags(self.artifact(ragged, *integral, n))
        self.assertEqual(tags.count("contract"), 2)
        self.assertEqual(tags.count("select"), 2 * 3)
        x = integral[0]
        refused = {"narrower result": (lambda x, n: lax.ragged_dot(
                       x, integral[1], n, preferred_element_type=jnp.bfloat16), n,
                       "not an exact widening")}
        for name, (function, sizes, text) in refused.items():
            with self.subTest(name=name):
                self.assert_refused(self.verify(function, x, sizes, checks=("structure",)), text)

    def test_ragged_contracting_and_batch_axes(self):
        # With a ragged contracting axis, entry j of the new leading axis contracts both operands
        # set to zero outside group j, as JAX's reference implementation does: one contraction
        # per group, so an infinity outside a group leaves it finite and a position in two
        # overlapping groups adds to both. With a ragged batch axis the result is dot_general's,
        # the sizes unread, a batch position in no group included.
        import numpy as np

        jax, jnp = self.jax, self.jnp
        lax = jax.lax
        rng = np.random.default_rng(17)
        x, r = rng.integers(-4, 5, (6, 4)), rng.integers(-4, 5, (4, 5))
        y, s = rng.integers(-4, 5, (2, 4, 6)), rng.integers(-4, 5, (5, 2, 4))
        plain = lax.RaggedDotDimensionNumbers(dot_dimension_numbers=(([1], [0]), ([], [])),
                                              lhs_ragged_dimensions=[1], rhs_group_dimensions=[])
        # Batch axes 0 and 1, contracting axes 1 and 2, the left operand's free axis last.
        batched = lax.RaggedDotDimensionNumbers(dot_dimension_numbers=(([1], [2]), ([0], [1])),
                                                lhs_ragged_dimensions=[1],
                                                rhs_group_dimensions=[])

        def ragged(x, r, y, s, n):
            return (lax.ragged_dot_general(x, r, n, plain),
                    lax.ragged_dot_general(y, s, n, batched))

        integral = [jnp.asarray(value, jnp.float32) for value in (x, r, y, s)]
        for sizes in ([1, 3], [0, 4], [2, 0, 3], [3, -1, 2], [5, 1]):
            with self.subTest(sizes=sizes):
                n = jnp.asarray(sizes, jnp.int32)
                report = self.verify(ragged, *integral, n)
                self.assert_passes(report)
                ends = np.cumsum(sizes)
                wanted = np.stack([x[:, max(e - k, 0):e] @ r[max(e - k, 0):e]
                                   for k, e in zip(sizes, ends)])
                _same(self, _outputs(report)[0].numpy().flatten().tolist(),
                      wanted.astype(np.float32).flatten().tolist())
                self.assertEqual(_tags(self.artifact(ragged, *integral, n)).count("contract"),
                                 2 * len(sizes))
        report = self.verify(ragged, *integral, jnp.zeros((0,), jnp.int32))
        self.assert_passes(report)
        self.assertEqual([tuple(i.shape) for i in _outputs(report)], [(0, 6, 5), (0, 2, 6, 5)])
        infinite = r.astype(np.float32)
        infinite[0, 0] = np.inf
        n = jnp.asarray([1, 3], jnp.int32)
        report = self.verify(lambda x, r, n: lax.ragged_dot_general(x, r, n, plain),
                             integral[0], jnp.asarray(infinite), n)
        self.assert_passes(report)
        _same(self, _outputs(report)[0].numpy()[1].flatten().tolist(),
              (x[:, 1:] @ r[1:]).astype(np.float32).flatten().tolist())
        normal = [jnp.asarray(rng.standard_normal(value.shape), jnp.float32)
                  for value in (x, r, y, s)]
        report = self.verify(ragged, *normal, jnp.asarray([2, 2], jnp.int32), checks=EXECUTE)
        self.assert_passes(report)
        for actual, wanted in zip(_outputs(report),
                                  ragged(*normal, jnp.asarray([2, 2], jnp.int32)), strict=True):
            np.testing.assert_allclose(actual.numpy(), np.asarray(wanted), rtol=1e-5, atol=1e-6)
        batch = lax.RaggedDotDimensionNumbers(dot_dimension_numbers=(([2], [1]), ([0], [0])),
                                              lhs_ragged_dimensions=[0], rhs_group_dimensions=[])
        u, v = rng.integers(-4, 5, (3, 6, 4)), rng.integers(-4, 5, (3, 4, 5))
        for sizes in ([1, 2], [1, 1], [0, 0, 0, 0]):
            with self.subTest(batch=sizes):
                n = jnp.asarray(sizes, jnp.int32)
                operands = [jnp.asarray(u, jnp.float32), jnp.asarray(v, jnp.float32), n]
                report = self.verify(lambda u, v, n: lax.ragged_dot_general(u, v, n, batch),
                                     *operands)
                self.assert_passes(report)
                _same(self, _outputs(report)[0].numpy().flatten().tolist(),
                      np.matmul(u, v).astype(np.float32).flatten().tolist())
                tags = _tags(self.artifact(lambda u, v, n: lax.ragged_dot_general(u, v, n, batch),
                                           *operands))
                self.assertEqual((tags.count("contract"), tags.count("select")), (1, 0))

    def test_ragged_group_sizes_per_position(self):
        # Group sizes of shape p + [g] give each position of the prefix axes its own groups: the
        # batch axes, then the left operand's free axes before the ragged one, or its contracting
        # axes before it, as in JAX's reference implementation. Integral operands compare bit
        # for bit with XLA. The contracting case's prefix, batch axis 1 then contracting axis 0,
        # is not in axis order.
        import numpy as np

        jax, jnp = self.jax, self.jnp
        lax = jax.lax
        rng = np.random.default_rng(23)
        y, s = rng.integers(-4, 5, (2, 6, 4)), rng.integers(-4, 5, (3, 2, 4, 5))
        x, r = rng.integers(-4, 5, (2, 6, 4)), rng.integers(-4, 5, (3, 4, 5))
        c, q = rng.integers(-4, 5, (3, 2, 6)), rng.integers(-4, 5, (2, 3, 6, 5))
        u, v = rng.integers(-4, 5, (2, 3, 6, 4)), rng.integers(-4, 5, (2, 3, 4, 5))
        numbers = lambda dims, ragged, group: lax.RaggedDotDimensionNumbers(
            dot_dimension_numbers=dims, lhs_ragged_dimensions=[ragged],
            rhs_group_dimensions=group)
        batch_free = numbers((([2], [2]), ([0], [1])), 1, [0])
        free_free = numbers((([2], [1]), ([], [])), 1, [0])
        contracting = numbers((([0, 2], [1, 2]), ([1], [0])), 2, [])
        batch = numbers((([3], [2]), ([0, 1], [0, 1])), 1, [])

        def ragged(y, s, x, r, c, q, u, v, m, n, k, b):
            return (lax.ragged_dot_general(y, s, m, batch_free),
                    lax.ragged_dot_general(x, r, n, free_free),
                    lax.ragged_dot_general(c, q, k, contracting),
                    lax.ragged_dot_general(u, v, b, batch))

        def rows(left, groups, sizes):
            # Row i of a [6, 4] operand contracted with its group of [g, 4, 5], +0 in none.
            ends, out = np.cumsum(sizes), np.zeros((6, 5))
            for i in range(6):
                inside = [j for j, e in enumerate(ends) if e - sizes[j] <= i < e]
                if inside:
                    out[i] = left[i] @ groups[inside[0]]
            return out

        integral = [jnp.asarray(value, jnp.float32) for value in (y, s, x, r, c, q, u, v)]
        cases = [([[2, 1, 3], [0, 6, 0]], [[1, 0, 2], [3, 3, 0]], [[[1, 2], [3, 3], [0, 6]],
                                                                  [[6, 0], [2, 0], [4, 1]]],
                  [[1, 1], [2, 0]]),
                 ([[0, 0, 0], [1, 2, 3]], [[6, 0, 0], [0, 0, 4]], [[[0, 0], [5, 1], [2, 2]],
                                                                  [[1, 4], [0, 3], [3, 0]]],
                  [[0, 0], [1, 1]])]
        for m, n, k, b in cases:
            with self.subTest(m=m, n=n, k=k):
                sizes = [jnp.asarray(value, jnp.int32) for value in (m, n, k, b)]
                report = self.verify(ragged, *integral, *sizes)
                self.assert_passes(report)
                wanted = [np.stack([rows(y[i], s[:, i], m[i]) for i in range(2)]),
                          np.stack([rows(x[i], r, n[i]) for i in range(2)]),
                          np.zeros((2, 2, 5)), np.matmul(u, v)]
                for i in range(2):
                    for t in range(3):
                        ends = np.cumsum(k[i][t])
                        for j, e in enumerate(ends):
                            start = e - k[i][t][j]
                            wanted[2][j, i] += c[t, i, start:e] @ q[i, t, start:e]
                for actual, value in zip(_outputs(report), wanted, strict=True):
                    _same(self, actual.numpy().flatten().tolist(),
                          value.astype(np.float32).flatten().tolist())

    def test_scaled_dot_is_its_decomposition(self):
        # scaled_dot is the decomposition of JAX's xla.scaled_dot composite: entry k of a scale
        # along a contracting axis covers the m positions from k * m, each operand and its scale
        # are rounded to bfloat16 and multiplied there, and the products contract as
        # dot_general does. Entries b * (1 + 2^-8), for integers b of magnitude 1, 2 or 4, are
        # halfway between bfloat16 neighbours and round to b, ties to even, and every product
        # and partial sum is then an integer of magnitude at most 256, exact in bfloat16, so
        # XLA's result compares bit for bit.
        import numpy as np

        jax, jnp = self.jax, self.jnp
        lax = jax.lax
        rng = np.random.default_rng(12)

        def entries(shape, values, dtype):
            base = rng.choice(values, shape).astype(np.float64)
            return jnp.asarray(base * (1 + 2.0 ** -8 * rng.integers(0, 2, shape)), dtype)

        def rounded(value, scale, axes):
            # The operand times its repeated scale, both rounded to bfloat16, as binary64.
            scale = np.asarray(scale, np.float64)
            for axis in axes:
                scale = np.repeat(scale, value.shape[axis] // scale.shape[axis], axis)
            return [np.asarray(np.asarray(v).astype(jnp.bfloat16), np.float64)
                    for v in (value, scale)]

        three = ([entries(shape, [-2, -1, 0, 1, 2], "float32") for shape in ((2, 3, 4), (2, 4, 5))]
                 + [entries(shape, [1, 2, 4], "float32") for shape in ((2, 3, 2), (2, 2, 5))])
        two = [entries((4, 3, 6), [-1, 0, 1], "bfloat16"),
               entries((5, 4, 6), [-1, 0, 1], "float16"),
               entries((2, 3, 3), [1, 2], "float32"), entries((5, 2, 3), [1, 2], "float32")]
        numbers = (((0, 2), (1, 2)), ((), ()))
        cases = {"default batched": (three, None, "bik,bkj->bij", ([2], [1])),
                 "two contracting axes": (two, numbers, "kil,jkl->ij", ([0, 2], [1, 2]))}
        for name, (operands, dimensions, spec, (lc, rc)) in cases.items():
            with self.subTest(name=name):
                def scaled(x, y, s, t, dimensions=dimensions):
                    return lax.scaled_dot(x, y, lhs_scale=s, rhs_scale=t,
                                          dimension_numbers=dimensions)

                x, s = rounded(operands[0], operands[2], lc)
                y, t = rounded(operands[1], operands[3], rc)
                wanted = np.einsum(spec, x * s, y * t)
                self.assertLessEqual(np.abs(wanted).max(), 256)
                np.testing.assert_array_equal(
                    np.asarray(jax.jit(scaled)(*operands), np.float64), wanted)
                report = self.verify(scaled, *operands)
                self.assert_passes(report)
                _same(self, _outputs(report)[0].double().flatten().tolist(),
                      wanted.flatten().tolist())
                self.assertEqual(_tags(self.artifact(scaled, *operands)).count("contract"), 1)
        # A binary32 result contracts the bfloat16 products converted to binary32.
        def wider(x, y, s, t):
            return lax.scaled_dot(x, y, lhs_scale=s, rhs_scale=t,
                                  preferred_element_type=jnp.float32)

        report = self.verify(wider, *three)
        self.assert_passes(report)
        x, s = rounded(three[0], three[2], [2])
        y, t = rounded(three[1], three[3], [1])
        _same(self, _outputs(report)[0].double().flatten().tolist(),
              np.einsum("bik,bkj->bij", x * s, y * t).flatten().tolist())
        refused = {"narrower result": (lambda x, y, s, t: lax.scaled_dot(
                       x, y, lhs_scale=s, rhs_scale=t, preferred_element_type=jnp.float16),
                       "does not hold every bfloat16"),
                   "default scales": (lambda x, y, s, t: lax.scaled_dot(x, y),
                                      "float8_e8m0fnu")}
        for name, (function, text) in refused.items():
            with self.subTest(name=name):
                self.assert_refused(self.verify(function, *three, checks=("structure",)), text)

    def test_float8_conversions_round_once_and_feed_scaled_dot(self):
        # The float8 formats of FloatLib's catalog are imported as their binary descriptors, through
        # convert_element_type, as scaled_dot's operands and scales, as operands of a wider
        # contraction (the next test), and in movement, arithmetic, comparisons, selection and the
        # reductions (ir_capture_jax.py, test_float8_arithmetic_rounds_once_in_the_format). Every
        # word of each converts to binary32, bfloat16 and binary16 exactly,
        # and a value converts to it rounded once to nearest, ties to even, as XLA's conversions do
        # word for word: an e4m3fn value above 464, or an infinity, is its NaN, an e5m2 one from
        # 61440 up is infinite, and a fnuz format's -0 is +0. A NaN's sign and payload are left
        # open, and XLA and the IR choose differently. ml_dtypes, NumPy's float8, rounds a binary64
        # value to binary32 first, so the binary64 values below come with the words of the single
        # rounding: 1.0625 + 2^-40 is above the midpoint of 1 and 1.125, which it is in binary32,
        # and 61440 - 2^-20 below e5m2's overflow midpoint.
        import numpy as np

        jax, jnp = self.jax, self.jnp
        lax = jax.lax
        rng = np.random.default_rng(14)
        once = {"float8_e5m2": ([1.125 + 2.0 ** -40, 61440 - 2.0 ** -20, 2.0 ** -17 + 2.0 ** -60],
                                [61, 123, 1]),
                "float8_e4m3fn": ([1.0625 + 2.0 ** -40, -1.0625 - 2.0 ** -40, 1.1875 - 2.0 ** -40,
                                   464 - 2.0 ** -30, 2.0 ** -10 + 2.0 ** -50],
                                  [57, 185, 57, 126, 1]),
                "float8_e4m3fnuz": ([1.0625 + 2.0 ** -40], [65]),
                "float8_e5m2fnuz": ([1.125 + 2.0 ** -40], [65])}
        for name, (doubles, words) in once.items():
            dtype = jnp.dtype(name)
            every = np.arange(256, dtype=np.uint8).view(dtype)
            for target in ("float32", "bfloat16", "float16"):
                with self.subTest(name=name, target=target):
                    report = self.verify(lambda v: v.astype(target), jnp.asarray(every))
                    self.assert_passes(report)
                    _same(self, _outputs(report)[0].double().tolist(),
                          np.asarray(every.astype(target), np.float64).tolist())
            values = np.concatenate([
                np.asarray(every, np.float32),
                rng.standard_normal(256).astype(np.float32) * 2.0 ** rng.integers(-20, 18, 256),
                np.asarray([np.inf, -np.inf, np.nan, -0.0, 464, 479.9, 480, 57344, 61439, 61440,
                            240, 248, 256, 2.0 ** -10, 2.0 ** -17, 1.5 * 2.0 ** -17], np.float32)])
            for source in ("float32", "bfloat16", "float64"):
                with self.subTest(name=name, source=source), jax.enable_x64(source == "float64"):
                    x = np.asarray(jnp.asarray(values, source))
                    if source == "float64":
                        x = np.concatenate([x, np.asarray(doubles)])
                    report = self.verify(lambda v: v.astype(dtype), jnp.asarray(x))
                    self.assert_passes(report)
                    wanted = x[:len(values)].astype(dtype).view(np.uint8).tolist()
                    wanted += words if source == "float64" else []
                    got = np.asarray(report["outputs"][0]["bits"], np.uint8)
                    _same(self, np.asarray(got.view(dtype), np.float64).tolist(), np.asarray(
                        np.asarray(wanted, np.uint8).view(dtype), np.float64).tolist())
        # scaled_dot rounds a float8 operand or scale to bfloat16 exactly. Operands of magnitude
        # at most 4 and scales 2^-2 to 2^2 make every product and partial sum exact in binary32,
        # so the result is the exact contraction rounded once.
        dimensions = (((1,), (0,)), ((), ()))
        kinds = [("float8_e4m3fn", "float8_e4m3fn", "bfloat16", None),
                 ("float8_e5m2", "float8_e4m3fn", "float32", None),
                 ("float8_e4m3fnuz", "float8_e5m2fnuz", "bfloat16", "float32"),
                 ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fn", "float32"),
                 ("float8_e5m2", "bfloat16", "bfloat16", "float32")]
        for lhs, rhs, scales, wider in kinds:
            with self.subTest(lhs=lhs, rhs=rhs, scales=scales, wider=wider):
                x, y = [rng.integers(-4, 5, shape).astype(np.float64)
                        for shape in ((3, 64), (64, 5))]
                s, t = [2.0 ** rng.integers(-2, 3, shape) for shape in ((3, 2), (2, 5))]
                operands = [jnp.asarray(v, d) for v, d in ((x, lhs), (y, rhs), (s, scales),
                                                          (t, scales))]

                def scaled(a, b, c, d, wider=wider):
                    return lax.scaled_dot(a, b, lhs_scale=c, rhs_scale=d,
                                          dimension_numbers=dimensions,
                                          preferred_element_type=wider)

                report = self.verify(scaled, *operands)
                self.assert_passes(report)
                exact = (x * np.repeat(s, 32, 1)) @ (y * np.repeat(t, 32, 0))
                rounded = np.asarray(exact.astype(wider or jnp.bfloat16), np.float64)
                _same(self, _outputs(report)[0].double().flatten().tolist(),
                      rounded.flatten().tolist())

    def test_float8_contractions_convert_their_operands(self):
        # A dot_general, ragged_dot_general or conv_general_dilated of float8 operands with a
        # preferred_element_type that holds them converts them to it exactly and contracts or
        # convolves there, a binary16 or bfloat16 result in binary32 rounded once. XLA's CPU
        # code converts them so too. Integers of magnitude at most 4 keep every product and
        # partial sum exact, so those results compare with XLA word for word. At random finite
        # operands a binary32 product is the left fold of the exact products from +0, and XLA's
        # binary32 contraction and convolution sum them in another order, so the two agree
        # within the bound of two summation orders. JAX leaves the accumulation type of a float8
        # result to the backend, and the importer contracts and convolves in binary32 and rounds
        # once to the format. Operands of magnitude 1/8 to 4 keep every binary32 sum exact in any
        # order, so the IR's words there are the exact sum rounded once. XLA's CPU code rounds the
        # binary32 sum to binary16 and then to the format, and its words are checked apart:
        # they are that model's, and differ from the IR's at some entries, as at the hand-worked
        # e4m3fn sum 1 + 1/16 + 1/4096, whose binary16 value 1.0625 is a midpoint of the format.
        # A ragged_dot_general with a float8 result is that contraction in binary32, its groups
        # selected there, rounded once, and XLA's words are again the binary16 model's, in each of
        # its three modes and at that sum. A float8 kernel holding a NaN gives NaN in the outputs
        # it reaches.
        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(15)
        dimensions = (((1,), (0,)), ((), ()))
        names = ("float8_e5m2", "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2fnuz")
        pairs = [(name, name) for name in names] + [("float8_e4m3fn", "float8_e5m2"),
                                                    ("float8_e4m3fnuz", "float8_e5m2fnuz")]
        for (lhs, rhs), target in [(pair, target) for pair in pairs
                                   for target in ("float32", "bfloat16", "float16", "float64")]:
            with self.subTest(lhs=lhs, rhs=rhs, target=target), \
                    jax.enable_x64(target == "float64"):
                x = jnp.asarray(rng.integers(-4, 5, (3, 8)), lhs)
                y = jnp.asarray(rng.integers(-4, 5, (8, 5)), rhs)

                def product(x, y, target=target):
                    return lax.dot_general(x, y, dimensions,
                                           preferred_element_type=jnp.dtype(target))

                report = self.verify(product, x, y)
                self.assert_passes(report)
                wanted = np.asarray(x, np.float64) @ np.asarray(y, np.float64)
                _same(self, _outputs(report)[0].double().flatten().tolist(),
                      wanted.flatten().tolist())
        x = jnp.asarray(rng.integers(-4, 5, (6, 4)), jnp.float8_e5m2)
        r = jnp.asarray(rng.integers(-4, 5, (2, 4, 3)), jnp.float8_e4m3fn)
        n = jnp.asarray([2, 4], jnp.int32)
        report = self.verify(lambda x, r, n: lax.ragged_dot(
            x, r, n, preferred_element_type=jnp.float32), x, r, n)
        self.assert_passes(report)
        a, b = (np.asarray(v, np.float64) for v in (x, r))
        _same(self, _outputs(report)[0].double().flatten().tolist(),
              np.concatenate([a[:2] @ b[0], a[2:] @ b[1]]).flatten().tolist())
        for name in ("float8_e5m2", "float8_e4m3fn"):
            with self.subTest(random=name):
                words = rng.integers(0, 256, 512).astype(np.uint8).view(jnp.dtype(name))
                finite = words[np.isfinite(np.asarray(words, np.float64))]
                x, y = finite[:96].reshape(6, 16), finite[96:176].reshape(16, 5)

                def product(x, y):
                    return lax.dot_general(x, y, dimensions,
                                           preferred_element_type=jnp.float32)

                report = self.verify(product, jnp.asarray(x), jnp.asarray(y), checks=EXECUTE)
                self.assert_passes(report)
                a, b = (np.asarray(v, np.float64) for v in (x, y))
                total = np.zeros((6, 5), np.float32)
                for p in range(16):
                    total = total + np.outer(a[:, p], b[p]).astype(np.float32)
                actual = _outputs(report)[0].double().numpy()
                _same(self, actual.flatten().tolist(), total.astype(np.float64).flatten().tolist())
                # Two orders of 16 terms differ by at most 30 units of 2^-24 times the sum of
                # the magnitudes, less than 2^-19 times it.
                xla = np.asarray(jax.jit(product)(jnp.asarray(x), jnp.asarray(y)), np.float64)
                self.assertTrue(np.all(np.abs(actual - xla)
                                       <= 2.0 ** -19 * (np.abs(a) @ np.abs(b))))
        for name, target in [(name, target) for name in names
                             for target in ("float32", "bfloat16", "float16", "float64")]:
            with self.subTest(convolution=name, target=target), \
                    jax.enable_x64(target == "float64"):
                x = jnp.asarray(rng.integers(-4, 5, (2, 3, 9)), name)
                w = jnp.asarray(rng.integers(-4, 5, (4, 3, 3)), name)

                def convolved(x, w=w, target=target):
                    return lax.conv_general_dilated(x, w, (1,), "VALID",
                                                    preferred_element_type=jnp.dtype(target))

                a, b = (np.asarray(v, np.float64) for v in (x, w))
                wanted = np.stack([np.einsum("bck,ock->bo", a[:, :, p:p + 3], b)
                                   for p in range(7)], -1)
                report = self.verify(convolved, x)
                self.assert_passes(report)
                _same(self, _outputs(report)[0].double().flatten().tolist(),
                      wanted.flatten().tolist())
        for name in ("float8_e5m2", "float8_e4m3fn"):
            with self.subTest(random_convolution=name):
                words = rng.integers(0, 256, 1024).astype(np.uint8).view(jnp.dtype(name))
                finite = words[np.isfinite(np.asarray(words, np.float64))]
                x, w = finite[:360].reshape(2, 6, 6, 5), finite[360:432].reshape(6, 2, 3, 2)

                def convolved(x, w, target):
                    return lax.conv_general_dilated(
                        x, w, (1, 2), [(1, 0), (0, 1)], rhs_dilation=(1, 2),
                        feature_group_count=3, preferred_element_type=target)

                def wide(x, w=jnp.asarray(w)):
                    return convolved(x, w, jnp.float32)

                report = self.verify(wide, jnp.asarray(x), checks=EXECUTE)
                self.assert_passes(report)
                actual = _outputs(report)[0].double().numpy()
                xla = np.asarray(jax.jit(wide)(jnp.asarray(x)), np.float64)
                with jax.enable_x64(True):
                    magnitudes = np.asarray(convolved(
                        *(jnp.asarray(np.abs(np.asarray(v, np.float64))) for v in (x, w)),
                        jnp.float64))
                # Each binary32 sum of the 12 exact products is within about 11 units of 2^-24
                # times the sum of their magnitudes of the exact sum, so two orders differ by
                # less than 2^-19 times it.
                self.assertTrue(np.all(np.abs(actual - xla) <= 2.0 ** -19 * magnitudes))

        def canonical(array):
            # The words of a float8 array, each NaN as -1.
            array = np.asarray(array)
            words = array.view(np.uint8).astype(np.int64)
            words[np.isnan(array.astype(np.float32))] = -1
            return words.ravel().tolist()

        def report_words(report, dtype, k=0):
            return canonical(np.asarray(report["outputs"][k]["bits"], np.uint8).view(dtype))

        for name in ("float8_e4m3fn", "float8_e4m3fnuz"):
            dtype = jnp.dtype(name)
            x = jnp.asarray([[1, 1 / 16, 1 / 64]], dtype)
            y = jnp.asarray([[1], [1], [1 / 64]], dtype)
            for label, function, operands in (
                    ("dot_general", lambda x, y: lax.dot_general(x, y, dimensions), (x, y)),
                    ("ragged_dot", lambda x, r, n: lax.ragged_dot(x, r, n),
                     (x, y[None], jnp.asarray([1], jnp.int32)))):
                with self.subTest(float8_result=name, midpoint=True, function=label):
                    report = self.verify(function, *operands, checks=EXECUTE)
                    self.assert_passes(report)
                    self.assertEqual(report_words(report, dtype),
                                     canonical(np.asarray([1.125], dtype)))
                    self.assertEqual(canonical(function(*operands)),
                                     canonical(np.asarray([1.0], dtype)))
        draws, differ = np.random.default_rng(17), 0
        modes = [lax.RaggedDotDimensionNumbers(
            dot_dimension_numbers=dims, lhs_ragged_dimensions=[axis], rhs_group_dimensions=group)
            for dims, axis, group in (((([1], [1]), ([], [])), 0, [0]),
                                      ((([1], [0]), ([], [])), 1, []),
                                      ((([2], [1]), ([0], [0])), 0, []))]

        def ragged(x, r, c, q, u, v, n, k, b):
            return tuple(lax.ragged_dot_general(*operands, sizes, mode) for operands, sizes, mode
                         in zip(((x, r), (c, q), (u, v)), (n, k, b), modes))

        for name in names:
            dtype = jnp.dtype(name)
            words = draws.integers(0, 256, 8192).astype(np.uint8).view(dtype)
            values = np.asarray(words, np.float64)
            grid = words[(values == 0) | (np.abs(values) >= 0.125) & (np.abs(values) <= 4)]
            x, y = grid[:256].reshape(16, 16), grid[256:512].reshape(16, 16)
            v, w = grid[512:752].reshape(2, 3, 40), grid[752:788].reshape(4, 3, 3)
            a, b, c, d = (np.asarray(t, np.float64) for t in (x, y, v, w))
            exact = [a @ b, np.stack([np.einsum("bck,ock->bo", c[:, :, p:p + 3], d)
                                      for p in range(38)], -1)]
            once = [canonical(e.astype(np.float32).astype(dtype)) for e in exact]
            twice = [canonical(e.astype(np.float16).astype(np.float32).astype(dtype))
                     for e in exact]
            differ += sum(np.sum(np.asarray(o) != np.asarray(t)) for o, t in zip(once, twice))

            def product(x, y):
                return lax.dot_general(x, y, dimensions)

            def convolved(v, w=jnp.asarray(w)):
                return lax.conv_general_dilated(v, w, (1,), "VALID")

            for k, (function, operands) in enumerate(((product, (x, y)), (convolved, (v,)))):
                for mode, run in (("eager", function), ("jit", jax.jit(function))):
                    with self.subTest(float8_result=name, function=function.__name__, mode=mode):
                        operands = [jnp.asarray(t) for t in operands]
                        report = self.verify(run, *operands, checks=EXECUTE)
                        self.assert_passes(report)
                        self.assertEqual(report_words(report, dtype), once[k])
                        self.assertEqual(canonical(run(*operands)), twice[k])
            with self.subTest(float8_result=name, random=True):
                finite = words[np.isfinite(values)]
                x, y = finite[:96].reshape(6, 16), finite[96:176].reshape(16, 5)
                a, b = (np.asarray(t, np.float64) for t in (x, y))
                total = np.zeros((6, 5), np.float32)
                for p in range(16):
                    total = total + np.outer(a[:, p], b[p]).astype(np.float32)
                report = self.verify(product, jnp.asarray(x), jnp.asarray(y), checks=EXECUTE)
                self.assert_passes(report)
                with np.errstate(over="ignore"):
                    self.assertEqual(report_words(report, dtype), canonical(total.astype(dtype)))
                    self.assertEqual(
                        canonical(jax.jit(product)(jnp.asarray(x), jnp.asarray(y))),
                        canonical(total.astype(np.float16).astype(np.float32).astype(dtype)))
            # A ragged free axis with an empty group and a position in none, a ragged contracting
            # axis and a ragged batch axis, whose sizes are not read.
            parts = grid[788:1172]
            x, r = parts[:48].reshape(6, 8), parts[48:168].reshape(3, 8, 5)
            c, q = parts[168:216].reshape(6, 8), parts[216:256].reshape(8, 5)
            u, v = parts[256:304].reshape(2, 3, 8), parts[304:384].reshape(2, 8, 5)
            a = [np.asarray(t, np.float64) for t in (x, r, c, q, u, v)]
            exact = [np.concatenate([a[0][:2] @ a[1][0], a[0][2:5] @ a[1][2], np.zeros((1, 5))]),
                     np.stack([a[2][:, :3] @ a[3][:3], a[2][:, 3:] @ a[3][3:]]), a[4] @ a[5]]
            once = [canonical(e.astype(np.float32).astype(dtype)) for e in exact]
            twice = [canonical(e.astype(np.float16).astype(np.float32).astype(dtype))
                     for e in exact]
            differ += sum(np.sum(np.asarray(o) != np.asarray(t)) for o, t in zip(once, twice))
            operands = [jnp.asarray(t) for t in (x, r, c, q, u, v)] + [
                jnp.asarray(sizes, jnp.int32) for sizes in ([2, 0, 3], [3, 5], [1, 1])]
            for mode, run in (("eager", ragged), ("jit", jax.jit(ragged))):
                with self.subTest(float8_result=name, function="ragged", mode=mode):
                    report = self.verify(run, *operands, checks=EXECUTE)
                    self.assert_passes(report)
                    self.assertEqual([report_words(report, dtype, k) for k in range(3)], once)
                    self.assertEqual([canonical(t) for t in run(*operands)], twice)
        self.assertGreater(differ, 0)
        x8, w8 = (jnp.asarray(rng.integers(-4, 5, shape), jnp.float8_e4m3fn)
                  for shape in ((2, 3, 9), (4, 3, 3)))
        nan = w8.at[0, 1, 2].set(jnp.nan)

        def unpadded(x):
            return lax.conv_general_dilated(x, nan, (1,), "VALID",
                                            preferred_element_type=jnp.float32)

        report = self.verify(unpadded, x8, checks=EXECUTE)
        self.assert_passes(report)
        rows = _outputs(report)[0].double().numpy()
        _same(self, rows.flatten().tolist(),
              np.asarray(unpadded(x8), np.float64).flatten().tolist())
        self.assertTrue(np.isnan(rows[:, 0]).all() and np.isfinite(rows[:, 1:]).all())

    def test_convolutions_take_a_kernel_not_known_at_capture(self):
        # A conv_general_dilated kernel that is an input, or computed from one, is the second
        # operand of the IR's convolution: in one to three spatial dimensions with strides,
        # padding, dilation and groups, in binary32, binary64, binary16 and bfloat16, of binary16
        # and float8 operands with a binary32 preferred_element_type, and through jnp.convolve,
        # jnp.correlate and jax.scipy.signal.convolve2d, which flip, scale or broadcast it first.
        # Integers of magnitude at most 3 keep every product and partial sum exact, so those
        # results compare with XLA word for word. A padding cell is a zero times its kernel
        # entry, as StableHLO pads the input before its dot products, so an infinite entry over
        # padding gives NaN. XLA's CPU code gives that NaN for a 3x3 kernel over a 4x4 input,
        # and for a 5x5 kernel over a 3x3 input it skips the padding and the sum stays finite.
        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(17)
        cases = {"1d": ((2, 3, 9), (4, 3, 3), dict(window_strides=(2,), padding=((1, 2),),
                                                   rhs_dilation=(2,))),
                 "2d, groups": ((1, 4, 5, 7), (6, 2, 3, 3), dict(
                     window_strides=(1, 2), padding=((1, 1), (2, 0)), feature_group_count=2)),
                 "3d": ((1, 2, 4, 4, 4), (2, 2, 2, 2, 2), dict(window_strides=(1, 1, 1),
                                                               padding="SAME"))}
        types = [("float32", None), ("float64", None), ("float16", None), ("bfloat16", None),
                 ("float16", "float32"), ("float8_e4m3fn", "float32")]
        for (name, (lhs, rhs, options)), (dtype, preferred) in [
                (case, kind) for case in cases.items() for kind in types]:
            with self.subTest(case=name, dtype=dtype, preferred=preferred), \
                    jax.enable_x64(dtype == "float64"):
                x, w = (jnp.asarray(rng.integers(-3, 4, shape), dtype) for shape in (lhs, rhs))

                def convolution(x, w, options=options, preferred=preferred):
                    return lax.conv_general_dilated(x, w, **options, preferred_element_type=(
                        preferred and jnp.dtype(preferred)))

                self.assert_passes(self.verify(convolution, x, w))
        v, taps = (jnp.asarray(rng.integers(-3, 4, n), jnp.float32) for n in (9, 4))
        x, stencil = (jnp.asarray(rng.integers(-3, 4, shape), jnp.float32)
                      for shape in ((5, 6), (2, 3)))
        modes = ("full", "same", "valid")
        for name, (function, operands) in {
                "convolve": (lambda v, t: [jnp.convolve(v, t, mode) for mode in modes],
                             (v, taps)),
                "correlate": (lambda v, t: [jnp.correlate(v, t * 2.0, mode) for mode in modes],
                              (v, taps)),
                "convolve2d": (jax.scipy.signal.convolve2d, (x, stencil))}.items():
            with self.subTest(function=name):
                self.assert_passes(self.verify(function, *operands))
        for size, extent, skipped in ((4, 3, False), (3, 5, True)):
            with self.subTest(input=size, kernel=extent):
                pad = extent // 2
                x = jnp.asarray(rng.integers(-3, 4, (1, 2, size, size)), jnp.float32)
                w = rng.integers(-3, 4, (2, 2, extent, extent)).astype(np.float32)
                w[0, 0, 0, 0] = np.inf

                def padded(x, w, pad=pad):
                    return lax.conv_general_dilated(x, w, (1, 1), ((pad, pad), (pad, pad)))

                report = self.verify(padded, x, jnp.asarray(w), checks=EXECUTE)
                self.assert_passes(report)
                cells = np.lib.stride_tricks.sliding_window_view(np.pad(
                    np.asarray(x, np.float64), [(0, 0), (0, 0), (pad, pad), (pad, pad)]),
                    (extent, extent), axis=(2, 3))
                with np.errstate(invalid="ignore"):
                    wanted = np.einsum("bihwkl,oikl->bohw", cells, w.astype(np.float64))
                actual = _outputs(report)[0].double().numpy()
                _same(self, actual.flatten().tolist(), wanted.flatten().tolist())
                self.assertTrue(np.isnan(actual[0, 0, 0]).all())
                xla = np.asarray(padded(x, jnp.asarray(w)), np.float64)
                if skipped:
                    self.assertTrue(np.isfinite(xla[0, 0, 0]).any())
                else:
                    _same(self, xla.flatten().tolist(), wanted.flatten().tolist())

    def test_reduced_float_contractions_round_once(self):
        # JAX leaves the accumulation type of a contraction to the backend, and XLA's CPU code
        # contracts and convolves binary16 and bfloat16 operands in binary32 and rounds the result
        # once. So does the importer: each entry is the binary32 fold of the products from +0,
        # exact here, rounded to the operands' format. Rounding at each step in that format
        # instead differs in about half the entries already at k = 2.
        import numpy as np

        jax, jnp = self.jax, self.jnp
        rng = np.random.default_rng(13)
        for dtype in ("bfloat16", "float16"):
            for k in (2, 32):
                with self.subTest(dtype=dtype, k=k):
                    x = jnp.asarray(rng.standard_normal((6, k)), dtype)
                    y = jnp.asarray(rng.standard_normal((k, 5)), dtype)
                    report = self.verify(jnp.matmul, x, y)
                    self.assert_passes(report)
                    a, b = (np.asarray(v, np.float32) for v in (x, y))
                    total = np.zeros((6, 5), np.float32)
                    for p in range(k):
                        total = total + np.outer(a[:, p], b[p])
                    _same(self, _outputs(report)[0].double().flatten().tolist(),
                          total.astype(jnp.dtype(dtype)).astype(np.float64).flatten().tolist())
            for channels in (1, 8):
                with self.subTest(dtype=dtype, channels=channels):
                    x = jnp.asarray(rng.standard_normal((2, channels, 12)), dtype)
                    w = jnp.asarray(rng.standard_normal((4, channels, 4)), dtype)
                    report = self.verify(
                        lambda x: jax.lax.conv_general_dilated(x, w, (1,), "VALID"), x)
                    self.assert_passes(report)
                    a, b = (np.asarray(v, np.float64) for v in (x, w))
                    exact = np.stack([np.einsum("bck,ock->bo", a[:, :, p:p + 4], b)
                                      for p in range(9)], -1)
                    _same(self, _outputs(report)[0].double().flatten().tolist(),
                          exact.astype(jnp.dtype(dtype)).astype(np.float64).flatten().tolist())

    def test_preferred_element_type_converts_the_operands(self):
        # JAX documents a preferred_element_type as the type a contraction accumulates in and
        # returns. One that holds every operand value is reached by converting the operands
        # exactly, as XLA's CPU code does, and the contraction or convolution runs in it.
        # Integers of magnitude at most 4 keep every product and partial sum exact, so those
        # results compare with XLA bit for bit. Random bfloat16 operands with a binary32 result
        # are the binary32 left fold of their exact products; XLA contracts them in another
        # binary32 order, within a tolerance. A type that does not hold the operands is refused.
        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(14)
        pairs = (("bfloat16", "float32"), ("float16", "float32"), ("float32", "float64"),
                 ("bfloat16", "float64"), ("int8", "float32"), ("uint8", "bfloat16"))
        for source, target in pairs:
            with self.subTest(source=source, target=target), \
                    jax.enable_x64("64" in source + target):
                low = 0 if source == "uint8" else -4
                x, y = (jnp.asarray(rng.integers(low, 5, shape), source)
                        for shape in ((3, 4), (4, 5)))

                def product(x, y, target=target):
                    return lax.dot(x, y, preferred_element_type=jnp.dtype(target))

                wanted = (np.asarray(x, np.float64) @ np.asarray(y, np.float64)).flatten()
                _same(self, np.asarray(jax.jit(product)(x, y), np.float64).flatten().tolist(),
                      wanted.tolist())
                report = self.verify(product, x, y)
                self.assert_passes(report)
                _same(self, _outputs(report)[0].double().flatten().tolist(), wanted.tolist())
                # One conversion per operand; a bfloat16 result is then contracted in binary32,
                # two conversions there and one rounding back.
                steps = _steps(self.artifact(product, x, y))
                self.assertEqual(sum("cast" in step for step in steps),
                                 5 if target == "bfloat16" else 2)
        x = jnp.asarray(rng.standard_normal((6, 32)), jnp.bfloat16)
        y = jnp.asarray(rng.standard_normal((32, 5)), jnp.bfloat16)

        def product(x, y):
            return lax.dot(x, y, preferred_element_type=jnp.float32)

        report = self.verify(product, x, y, checks=EXECUTE)
        self.assert_passes(report)
        a, b = (np.asarray(v, np.float32) for v in (x, y))
        total = np.zeros((6, 5), np.float32)
        for p in range(32):
            total = total + np.outer(a[:, p], b[p])
        actual = _outputs(report)[0].double().numpy()
        _same(self, actual.flatten().tolist(), total.astype(np.float64).flatten().tolist())
        np.testing.assert_allclose(actual, np.asarray(jax.jit(product)(x, y)), rtol=1e-5,
                                   atol=1e-5)
        for source, target in (("bfloat16", "float32"), ("float16", "float32"),
                               ("float32", "float64")):
            with self.subTest(convolution=source, target=target), \
                    jax.enable_x64(target == "float64"):
                x = jnp.asarray(rng.integers(-4, 5, (2, 3, 9)), source)
                w = jnp.asarray(rng.integers(-4, 5, (4, 3, 3)), source)

                def convolved(x, w=w, target=target):
                    return lax.conv_general_dilated(x, w, (1,), "VALID",
                                                    preferred_element_type=jnp.dtype(target))

                a, b = (np.asarray(v, np.float64) for v in (x, w))
                wanted = np.stack([np.einsum("bck,ock->bo", a[:, :, p:p + 3], b)
                                   for p in range(7)], -1).flatten().tolist()
                report = self.verify(convolved, x)
                self.assert_passes(report)
                _same(self, _outputs(report)[0].double().flatten().tolist(), wanted)
        x = jnp.asarray(rng.integers(-4, 5, (6, 4)), jnp.bfloat16)
        r = jnp.asarray(rng.integers(-4, 5, (3, 4, 5)), jnp.bfloat16)
        sizes = jnp.asarray([2, 1, 3], jnp.int32)
        report = self.verify(lambda x, r, n: lax.ragged_dot(
            x, r, n, preferred_element_type=jnp.float32), x, r, sizes, checks=EXECUTE)
        self.assert_passes(report)
        a, b = np.asarray(x, np.float64), np.asarray(r, np.float64)
        _same(self, _outputs(report)[0].double().flatten().tolist(),
              np.concatenate([a[:2] @ b[0], a[2:3] @ b[1], a[3:] @ b[2]]).flatten().tolist())
        x32 = jnp.asarray(rng.standard_normal((3, 4)), jnp.float32)
        w = jnp.asarray(rng.integers(-4, 5, (2, 3, 2)), jnp.float32)
        refused = {"narrower format": (lambda x: lax.dot(
                       x, x.T, preferred_element_type=jnp.bfloat16), x32),
                   "bfloat16 in binary16": (lambda x: lax.dot(
                       x, x.T, preferred_element_type=jnp.float16), x32.astype(jnp.bfloat16)),
                   "int32 in binary32": (lambda x: lax.dot(
                       x, x.T, preferred_element_type=jnp.float32), x32.astype(jnp.int32)),
                   "int32 convolution in binary32": (lambda x: lax.conv_general_dilated(
                       x[None], w.astype(jnp.int32), (1,), "VALID",
                       preferred_element_type=jnp.float32), x32.astype(jnp.int32))}
        for name, (function, operand) in refused.items():
            with self.subTest(name=name):
                self.assert_refused(self.verify(function, operand, checks=("structure",)),
                                    "not an exact widening")

    def test_integer_and_boolean_contractions(self):
        # dot_general of integers and Booleans as StableHLO defines it (its spec, "dot_general"):
        # each entry the sum of the products of its slices, wrapping modulo 2^width, or the
        # logical or of logical ands (its "add" and "multiply"). Batch axes, two contracted axes,
        # free axes on both sides, a vector product and an empty contraction, at full-range
        # operands of each width, compared word for word with XLA and with Python's integers,
        # eager and under jax.jit; an int8 or Boolean contraction with an int32 result converts
        # the operands first. The Boolean maximum and minimum, also of windows, are logical or
        # and and.
        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(23)
        functions = {
            "batched matmul": lambda x, y: jnp.matmul(x, y),
            "two contracted axes": lambda x, y: jnp.einsum("bij,bjk->ik", x, y),
            "tensordot": lambda x, y: jnp.tensordot(x, y, axes=([2], [1])),
            "vector": lambda x, y: jnp.dot(x[0, 1], y[1, :, 2]),
            "empty": lambda x, y: lax.dot_general(x[:, :, :0], y[:, :0], (((2,), (1,)),
                                                                          ((0,), (0,)))),
        }
        exact = {"batched matmul": "bij,bjk->bik", "two contracted axes": "bij,bjk->ik",
                 "tensordot": "bij,cjk->bick", "vector": None, "empty": None}
        for dtype in ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
                      "bool"):
            with self.subTest(dtype=dtype), jax.enable_x64("64" in dtype):
                if dtype == "bool":
                    x, y = rng.random((2, 3, 5)) < 0.3, rng.random((2, 5, 4)) < 0.3
                else:
                    info = np.iinfo(dtype)
                    x, y = (rng.integers(info.min, info.max, shape, dtype, endpoint=True)
                            for shape in ((2, 3, 5), (2, 5, 4)))
                for name, function in functions.items():
                    report = self.verify(function, x, y)
                    self.assert_passes(report)
                    if exact[name] is None:
                        continue
                    # Python's integers, reduced modulo 2^width to the type, or for Booleans
                    # whether some product is true.
                    total = np.einsum(exact[name], x.astype(object), y.astype(object))
                    wanted = total > 0 if dtype == "bool" else np.vectorize(
                        lambda v: (v - int(info.min)) % 2 ** info.bits + int(info.min))(
                        total).astype(dtype)
                    np.testing.assert_array_equal(np.asarray(function(x, y)), wanted, name)
                if dtype in ("int8", "uint32", "bool"):
                    self.assert_passes(self.verify(jax.jit(functions["batched matmul"]), x, y))
        for dtype in ("int8", "bool"):
            with self.subTest(dtype=dtype, result="int32"):
                x = rng.integers(-128, 128, (3, 6), dtype) if dtype == "int8" else \
                    rng.random((3, 6)) < 0.5
                report = self.verify(lambda x: lax.dot(x, x.T, preferred_element_type=jnp.int32),
                                     x)
                self.assert_passes(report)
                np.testing.assert_array_equal(
                    _outputs(report)[0].numpy(), x.astype(np.int64) @ x.T.astype(np.int64))
        b, c = rng.random((2, 9)) < 0.5
        for name, function in {"max": lax.max, "min": lax.min, "maximum": jnp.maximum,
                               "minimum": jnp.minimum,
                               "window max": lambda b, c: lax.reduce_window(
                                   b, False, lax.max, (3,), (1,), "VALID"),
                               "window min": lambda b, c: lax.reduce_window(
                                   c, True, lax.min, (2,), (2,), "SAME")}.items():
            with self.subTest(boolean=name):
                self.assert_passes(self.verify(function, b, c))
