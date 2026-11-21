"""JAX's linear algebra captured and checked with ``verify ir`` (``JaxLinalgTests``): the
``lax.linalg`` primitives the Jaxpr importer expands into IR operators
(``python/torchlean/_linalg.py``), and the reverse mode of those expansions. Run them through
``scripts/checks/test_ir_capture.py``, which describes the checks.
"""

import unittest

from ir_capture_support import (CHECKER, EXECUTE, HAS_JAX, CheckerCase, _outputs, _reflect32,
                                _same)
import torchlean
from ir_capture_jax_operators import JaxGeneralOperatorTests


def _graph_function(artifact, target, dtype):
    """The value of the imported graph's node ``target`` as a JAX function of the graph's
    inputs, read node by node for the operators the linear algebra expansions use before their
    refusal (``KeyError`` for any other). JAX's derivative of it is the reverse mode of the graph
    itself, each operator's pullback in turn (``where`` gives the branch not taken a zero
    cotangent), not JAX's rule for the primitive."""
    import jax.numpy as jnp
    import numpy as np

    nodes, constants = artifact["nodes"], artifact["constants"]
    words = {"float32": np.uint32, "float64": np.uint64}[dtype]
    elementwise = {"abs": jnp.abs, "neg": jnp.negative, "sqrt": jnp.sqrt, "add": jnp.add,
                   "sub": jnp.subtract, "mul": jnp.multiply, "div": jnp.divide,
                   "gt": jnp.greater, "ge": jnp.greater_equal, "eq": jnp.equal,
                   "select": jnp.where}

    def function(*inputs):
        values = {}

        def value(k):
            if k not in values:
                values[k] = evaluate(k, nodes[k])
            return values[k]

        def evaluate(k, step):
            if "input" in step:
                return inputs[step["input"]["index"]]
            if "const" in step:
                bits = np.asarray(constants[k], words)
                return jnp.asarray(bits.view(dtype).reshape(step["const"]["value"]))
            record = step["op"]["record"]
            tag, fields = record["tag"], [next(iter(f.values())) for f in record["fields"]]
            operands = [value(source) for source, _ in step["op"]["inputs"]]
            if tag in elementwise:
                return elementwise[tag](*operands)
            if tag == "reshape":
                return operands[0].reshape(fields[1])
            if tag == "broadcast":
                source, shape, dims = fields[:3]
                placed = [1] * len(shape)
                for extent, axis in zip(source, dims):
                    placed[axis] = extent
                return jnp.broadcast_to(operands[0].reshape(placed), shape)
            if tag == "slice":
                index = [slice(None)] * operands[0].ndim
                for axis, first, end, stride in zip(*fields[1:5]):
                    index[axis] = slice(first, end, stride)
                return operands[0][tuple(index)]
            if tag == "cat":
                return jnp.concatenate(operands, axis=fields[1])
            raise KeyError(tag)

        return value(target)

    return function


@unittest.skipUnless(CHECKER and HAS_JAX, "needs the built verify executable and JAX")
class JaxLinalgTests(CheckerCase):
    setUp, verify = JaxGeneralOperatorTests.setUp, JaxGeneralOperatorTests.verify

    def test_triangular_solve_cholesky_and_symmetric_product(self):
        # triangular_solve is substitution on op(a) x = b or x op(a) = b, reading only the used
        # triangle (and no diagonal for unit_diagonal): a lower system forward, an upper one
        # backward, row i (b_i - sum_j op(a)_ij x_j) / op(a)_ii with the sum a left fold from +0
        # over the rows solved before it, in substitution order. A right-side equation is solved
        # transposed. cholesky of the symmetrized operand is the left-looking column algorithm,
        # NaN where a pivot is not positive, under lax.linalg.cholesky's zeros above the diagonal.
        # symmetric_product is alpha (a a^T) + beta c. The reference is that arithmetic in NumPy's
        # binary32, operation by operation. Each result is also within 1e-5 of JAX's own, whose
        # LAPACK and XLA kernels round in another order, so that a reading the reference shares
        # with the importer fails too.
        import itertools
        import math

        import numpy as np

        jnp, lax = self.jnp, self.jax.lax
        f32 = np.float32
        rng = np.random.default_rng(14)
        n = 4
        a = rng.standard_normal((2, n, n)).astype(f32)
        a[:, range(n), range(n)] += 3

        def close(got, want):
            np.testing.assert_allclose(np.asarray(got, np.float64), np.asarray(want, np.float64),
                                       rtol=1e-5, atol=1e-5)
        b_left = rng.standard_normal((2, n, 3)).astype(f32)
        b_right = rng.standard_normal((2, 3, n)).astype(f32)

        def fold(terms):
            total = f32(0)
            for term in terms:
                total = f32(total + term)
            return total

        def solve_left(m, rhs, lower, unit):
            # m is op(a), read in its lower or upper triangle, row by row in substitution order.
            size = m.shape[0]
            order = range(size) if lower else range(size - 1, -1, -1)
            x = np.zeros(rhs.shape, dtype=f32)
            done = []
            for i in order:
                for col in range(rhs.shape[1]):
                    s = fold(f32(m[i, j] * x[j, col]) for j in (done if lower else done))
                    r = f32(rhs[i, col] - s) if done else rhs[i, col]
                    x[i, col] = r if unit else f32(r / m[i, i])
                done.append(i)
            return x

        def reference(a, b, left_side, lower, transpose_a, unit):
            m = a.T if transpose_a else a
            low = lower != transpose_a
            if left_side:
                return solve_left(m, b, low, unit)
            return solve_left(m.T, b.T, not low, unit).T

        cases = list(itertools.product((True, False), repeat=4))

        def solves(a, b_left, b_right):
            return tuple(lax.linalg.triangular_solve(
                a, b_left if left else b_right, left_side=left, lower=lower,
                transpose_a=transpose, unit_diagonal=unit) for left, lower, transpose, unit
                in cases)

        report = self.verify(solves, jnp.asarray(a), jnp.asarray(b_left), jnp.asarray(b_right),
                             checks=EXECUTE)
        self.assert_passes(report)
        jax_solves = solves(jnp.asarray(a), jnp.asarray(b_left), jnp.asarray(b_right))
        for got, jax_got, (left, lower, transpose, unit) in zip(_outputs(report), jax_solves,
                                                                 cases):
            want = np.stack([reference(a[k], (b_left if left else b_right)[k], left, lower,
                                       transpose, unit) for k in range(2)])
            _same(self, got.flatten().tolist(), [float(v) for v in want.reshape(-1)],
                  (left, lower, transpose, unit))
            close(got, jax_got)
        # Cholesky of a positive definite batch member and of one with a negative pivot.
        spd = np.stack([a[0] @ a[0].T + np.eye(n, dtype=f32),
                        np.diag([1.0, -2.0, 1.0, 1.0]) + 0.25]).astype(f32)

        def factor(m, symmetrize=True):
            s = np.array(m, dtype=f32)
            for i, j in itertools.product(range(n), repeat=2):
                if symmetrize:
                    s[i, j] = f32(f32(m[i, j] + m[j, i]) / f32(2))
            low = np.zeros((n, n), dtype=f32)
            valid = True
            for j in range(n):
                d = f32(s[j, j] - fold(f32(low[j, k] * low[j, k]) for k in range(j)))
                valid = valid and bool(d > 0)
                root = np.sqrt(f32(max(d, f32(0))))
                low[j, j] = root
                for i in range(j + 1, n):
                    low[i, j] = f32(f32(s[i, j] - fold(f32(low[i, k] * low[j, k])
                                                       for k in range(j))) / root)
            out = low if valid else np.full((n, n), np.nan, dtype=f32)
            return np.where(np.tril(np.ones((n, n), dtype=bool)), out, f32(0))

        report = self.verify(jnp.linalg.cholesky, jnp.asarray(spd), checks=EXECUTE)
        self.assert_passes(report)
        (got,) = _outputs(report)
        want = np.stack([factor(spd[0]), factor(spd[1])])
        _same(self, got.flatten().tolist(), [float(v) for v in want.reshape(-1)])
        self.assertTrue(math.isnan(want[1, 0, 0]) and want[1, 0, 1] == 0)
        # JAX's factor of the second member is NaN in its lower triangle too.
        close(np.tril(got), np.tril(np.asarray(jnp.linalg.cholesky(jnp.asarray(spd)))))
        # cholesky_update: JAX's Givens rotations, row by row. A zero update component keeps the
        # row, so a factor whose squares under- and overflow binary32 is returned unchanged.
        def rotations(r, z):
            r, z = r.copy(), z.copy()
            for k in range(n):
                x, y = r[k, k], z[k]
                if y == 0:
                    c, s = f32(1), f32(0)
                else:
                    d = abs(x) if abs(x) > abs(y) else abs(y)
                    xn, yn = f32(x / d), f32(y / d)
                    rh = f32(f32(1) / np.sqrt(f32(f32(xn * xn) + f32(yn * yn))))
                    c, s = f32(xn * rh), f32(f32(-yn) * rh)
                row = r[k].copy()
                r[k] = [f32(f32(c * a) - f32(s * b)) for a, b in zip(row, z)]
                z = np.asarray([f32(f32(c * b) + f32(s * a)) for a, b in zip(row, z)], dtype=f32)
            return r
        r = np.ascontiguousarray(want[0].T)
        w = rng.standard_normal(n).astype(f32)
        extreme = np.diag(np.asarray([2.0 ** -100, 2.0 ** 100, 1.0, 3.0], dtype=f32))
        extreme[0, 3] = 5.0
        for factor_r, update in ((r, w), (extreme, np.zeros(n, dtype=f32))):
            report = self.verify(lax.linalg.cholesky_update, jnp.asarray(factor_r),
                                 jnp.asarray(update), checks=EXECUTE)
            self.assert_passes(report)
            _same(self, _outputs(report)[0].flatten().tolist(),
                  [float(v) for v in rotations(factor_r, update).reshape(-1)])
            close(_outputs(report)[0], lax.linalg.cholesky_update(jnp.asarray(factor_r),
                                                                  jnp.asarray(update)))
        self.assertEqual(rotations(extreme, np.zeros(n, dtype=f32)).tolist(), extreme.tolist())
        # symmetric_product and custom_linear_solve with a diagonal solve.
        c = rng.standard_normal((2, n, n)).astype(f32)
        report = self.verify(lambda a, c: lax.linalg.symmetric_product(a, c, alpha=0.3,
                                                                       beta=-1.5),
                             jnp.asarray(a[:, :, :3]), jnp.asarray(c), checks=EXECUTE)
        self.assert_passes(report)
        (got,) = _outputs(report)
        want = np.empty((2, n, n), dtype=f32)
        for k, i, j in itertools.product(range(2), range(n), range(n)):
            product = fold(f32(a[k, i, t] * a[k, j, t]) for t in range(3))
            want[k, i, j] = f32(f32(f32(0.3) * product) + f32(f32(-1.5) * c[k, i, j]))
        _same(self, got.flatten().tolist(), [float(v) for v in want.reshape(-1)])
        # JAX guarantees only the lower triangle of its result.
        close(np.tril(got), np.tril(np.asarray(lax.linalg.symmetric_product(
            jnp.asarray(a[:, :, :3]), jnp.asarray(c), alpha=0.3, beta=-1.5))))
        d = np.asarray([2.0, 4.0, 0.5], dtype=f32)
        solve = lambda b: lax.custom_linear_solve(lambda x: d * x, b, lambda mv, r: r / d)
        rhs = np.asarray([1.0, 3.0, -2.0], dtype=f32)
        report = self.verify(solve, jnp.asarray(rhs), checks=EXECUTE)
        self.assert_passes(report)
        _same(self, _outputs(report)[0].tolist(), [float(f32(r / s)) for r, s in zip(rhs, d)])

    def test_householder_products_and_pivot_permutations(self):
        # ormqr and householder_product apply the reflectors H_i = 1 - tau_i v_i v_i^T of a's
        # columns one at a time in the order of Q = H_0 ... H_{k-1}, and householder_product is Q
        # times the identity's first n columns. The reference is that arithmetic in NumPy's
        # binary32 (_reflect32); JAX's LAPACK results are within 1e-5 of it.
        # lu_pivots_to_permutation fails at a pivot outside the permutation (none is documented).
        import itertools

        import numpy as np

        jnp, lax = self.jnp, self.jax.lax
        f32 = np.float32
        rng = np.random.default_rng(15)
        m, k = 5, 3
        h, taus = jnp.linalg.qr(jnp.asarray(rng.standard_normal((2, m, k)).astype(f32)),
                                mode="raw")
        a, taus = np.asarray(h).transpose(0, 2, 1).copy(), np.asarray(taus)
        cases = list(itertools.product((True, False), repeat=2))
        c_left = rng.standard_normal((2, m, 4)).astype(f32)
        c_right = rng.standard_normal((2, 4, m)).astype(f32)

        def products(a, taus, c_left, c_right):
            return (lax.linalg.householder_product(a, taus),
                    *(lax.linalg.ormqr(a, taus, c_left if left else c_right, left=left,
                                       transpose=transpose) for left, transpose in cases))

        inputs = [jnp.asarray(value) for value in (a, taus, c_left, c_right)]
        report = self.verify(products, *inputs, checks=EXECUTE)
        self.assert_passes(report)
        wants = [np.stack([_reflect32(a[b], taus[b], np.eye(m, k, dtype=f32), True, False)
                           for b in range(2)])]
        wants += [np.stack([_reflect32(a[b], taus[b], (c_left if left else c_right)[b], left,
                                       transpose) for b in range(2)]) for left, transpose in cases]
        for got, want, lapack, case in zip(_outputs(report), wants, products(*inputs),
                                           [None] + cases):
            _same(self, got.flatten().tolist(), [float(v) for v in want.reshape(-1)], case)
            self.assertTrue(np.allclose(want, np.asarray(lapack), atol=1e-5), case)
        self.assert_refused(self.verify(lax.linalg.householder_product,
                                        jnp.asarray(a, jnp.complex64),
                                        jnp.asarray(taus, jnp.complex64)), "complex values")
        # Row swaps: iota with entries i and pivots[i] swapped in turn (Factorization.pivotPerm).
        def swapped(row):
            permutation = list(range(m))
            for i, j in enumerate(row):
                permutation[i], permutation[j] = permutation[j], permutation[i]
            return permutation

        pivots = np.array([[2, 2, 4], [0, 3, 3]], dtype=np.int32)
        convert = lambda p: lax.linalg.lu_pivots_to_permutation(p, m)
        report = self.verify(convert, jnp.asarray(pivots))
        self.assert_passes(report)
        self.assertEqual(_outputs(report)[0].tolist(), [swapped(row) for row in pivots])
        for bad in ([[2, 5, 4], [0, 3, 3]], [[2, 2, 4], [0, -1, 3]]):
            with self.subTest(pivots=bad):
                self.assert_unsupported_execution(
                    self.verify(convert, jnp.asarray(bad, jnp.int32)), "outside")

    def test_tridiagonal_solve_by_partial_pivoting(self):
        # tridiagonal_solve is LAPACK gtsv's elimination, rows i and i + 1 kept where the pivot
        # |u_i| >= |dl[i + 1]| and swapped otherwise, then back substitution. The reference is
        # that arithmetic in NumPy's binary32; its residual and JAX's LAPACK result are within
        # 1e-4. dl[0] and du[m - 1] are NaN, as neither is read. A singular matrix (a zero pivot)
        # fails at execution, as no result is documented for it.
        import numpy as np

        jnp, lax = self.jnp, self.jax.lax
        f32 = np.float32
        rng = np.random.default_rng(16)

        def gtsv(dl, d, du, b):
            m, d, du, b = len(d), d.copy(), du.copy(), b.copy()
            lower, fill = dl[1:], np.zeros(max(m - 2, 0), dtype=f32)
            for i in range(m - 1):
                if abs(d[i]) >= abs(lower[i]):
                    f = lower[i] / d[i]
                    d[i + 1] = d[i + 1] - f * du[i]
                    b[i + 1] = b[i + 1] - f * b[i]
                else:
                    f = d[i] / lower[i]
                    d[i], d[i + 1], du[i] = lower[i], du[i] - f * d[i + 1], d[i + 1]
                    if i + 2 < m:
                        fill[i], du[i + 1] = du[i + 1], -f * du[i + 1]
                    b[i], b[i + 1] = b[i + 1].copy(), b[i] - f * b[i + 1]
            x = np.empty_like(b)
            for i in reversed(range(m)):
                row = b[i]
                if i + 1 < m:
                    row = row - du[i] * x[i + 1]
                if i + 2 < m:
                    row = row - fill[i] * x[i + 2]
                x[i] = row / d[i]
            return x

        swaps = 0
        for m in (1, 2, 5):
            dl = rng.standard_normal((2, m)).astype(f32)
            du = rng.standard_normal((2, m)).astype(f32)
            d = (rng.standard_normal((2, m)) * [[0.3], [1.0]] + [[0.0], [4.0]]).astype(f32)
            d[0, 0] = 0 if m > 1 else d[0, 0]
            b = rng.standard_normal((2, m, 3)).astype(f32)
            matrices = np.stack([np.diag(d[k].astype(float)) + np.diag(dl[k, 1:].astype(float), -1)
                                 + np.diag(du[k, :-1].astype(float), 1) for k in range(2)])
            swaps += sum(abs(d[0, i]) < abs(dl[0, i + 1]) for i in range(m - 1))
            dl[:, 0], du[:, -1] = np.nan, np.nan
            inputs = [jnp.asarray(value) for value in (dl, d, du, b)]
            report = self.verify(lax.linalg.tridiagonal_solve, *inputs, checks=EXECUTE)
            self.assert_passes(report)
            want = np.stack([gtsv(dl[k], d[k], du[k], b[k]) for k in range(2)])
            _same(self, _outputs(report)[0].flatten().tolist(),
                  [float(v) for v in want.reshape(-1)], m)
            self.assertTrue(np.allclose(matrices @ want, b, atol=1e-4), m)
            lapack = np.asarray(lax.linalg.tridiagonal_solve(*inputs))
            self.assertTrue(np.allclose(want, lapack, atol=1e-4), (m, want, lapack))
        self.assertGreater(swaps, 0)
        # [[0, 1], [0, 0]] stops at its first pivot and [[1, 1], [1, 1]] at its last.
        for dl, d, du in (([0, 0], [0, 0], [1, 0]), ([0, 1], [1, 1], [1, 0])):
            with self.subTest(d=d, du=du):
                singular = [jnp.asarray([v], jnp.float32) for v in (dl, d, du)]
                self.assert_unsupported_execution(self.verify(
                    lax.linalg.tridiagonal_solve, *singular, jnp.ones((1, 2, 1), jnp.float32)),
                    "outside")
        self.assert_refused(self.verify(
            lambda *operands: lax.linalg.tridiagonal_solve(*operands, perturb_singular=True),
            *inputs), "perturb_singular")
        self.assert_refused(self.verify(lax.linalg.tridiagonal_solve,
                                        *(jnp.asarray(v, jnp.complex64) for v in inputs)),
                            "complex values")

    def test_expansions_stay_finite_in_reverse_mode(self):
        # The expansions compute both branches of each pivot or rotation choice and select one;
        # the selection's pullback gives the branch not taken a zero cotangent, which a quotient
        # by zero there (the identity's infinite multiplier 1 / 0) turns into 0 * inf = NaN.
        # Every value of the imported graph, executed with each operator's result an output, is
        # finite at identity, triangular and pivot-swapping tridiagonal systems and at a zero
        # rotation of cholesky_update (R[0, 0] = z[0] = 0), in binary32 and binary64. The
        # graph's reverse mode (`_graph_function`, its result the checker's word for word) is
        # finite too, and for tridiagonal_solve within rounding of JAX's vjp, which
        # differentiates the system implicitly; JAX has no derivative rule for cholesky_update.
        import numpy as np

        jax, lax = self.jax, self.jax.lax
        systems = {"identity": ([0, 0, 0], [1, 1, 1], [0, 0, 0]),
                   "lower": ([0, 2, 3], [1, 1, 1], [0, 0, 0]),
                   "upper": ([0, 0, 0], [1, 1, 1], [2, 3, 0]),
                   "swap": ([0, 4, 0], [1, 1, 1], [0, 1, 0])}
        for dtype in ("float32", "float64"):
            with jax.enable_x64(dtype == "float64"):
                jnp = self.jnp
                cases = {name: (lax.linalg.tridiagonal_solve,
                                [jnp.asarray(v, dtype) for v in diagonals]
                                + [jnp.ones((3, 2), dtype)])
                         for name, diagonals in systems.items()}
                cases["cholesky_update"] = (lax.linalg.cholesky_update,
                                            [jnp.asarray([[0, 1], [0, 2]], dtype),
                                             jnp.asarray([0, 1], dtype)])
                for name, (function, inputs) in cases.items():
                    with self.subTest(dtype=dtype, name=name):
                        captured = torchlean._capture(function, tuple(inputs), "jax", EXECUTE)
                        artifact = captured["artifact"]
                        steps = [k for k, step in enumerate(artifact["nodes"]) if "op" in step]
                        report = torchlean.verify(
                            {"artifact": dict(artifact, outputs=[[k, 0] for k in steps]),
                             "arguments": captured["arguments"]}, checks=EXECUTE, checker=CHECKER)
                        self.assert_passes(report)
                        values = dict(zip(steps, _outputs(report)))
                        self.assertEqual([k for k, v in values.items() if v.dtype.is_floating_point
                                          and not bool(v.isfinite().all())], [])
                        ((target, _),) = artifact["outputs"]
                        if function is lax.linalg.tridiagonal_solve:
                            # the solution, which the singular-matrix refusal selects
                            refusal = artifact["nodes"][target]["op"]
                            self.assertEqual(refusal["record"]["tag"], "select")
                            target = refusal["inputs"][2][0]
                        result, pullback = jax.vjp(_graph_function(artifact, target, dtype),
                                                   *inputs)
                        # word for word, so a zero of the other sign differs
                        got, want = np.asarray(result), values[target].numpy()
                        self.assertEqual(got.dtype, want.dtype)
                        self.assertTrue(np.array_equal(got.view(f"u{got.itemsize}"),
                                                       want.view(f"u{want.itemsize}")))
                        cotangent = jnp.arange(result.size, dtype=dtype).reshape(result.shape) - 2
                        cotangents = [np.asarray(v) for v in pullback(cotangent)]
                        self.assertTrue(all(np.isfinite(v).all() for v in cotangents), cotangents)
                        if function is lax.linalg.tridiagonal_solve:
                            for got, want in zip(cotangents,
                                                 jax.vjp(function, *inputs)[1](cotangent)):
                                self.assertTrue(np.allclose(got, want, rtol=1e-6), (got, want))
