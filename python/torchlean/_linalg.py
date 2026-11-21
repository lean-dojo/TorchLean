"""The captures' linear algebra, in the IR's movement, arithmetic and ``matmul``.

``LinalgImport`` builds ``triangular_solve``, ``cholesky``, ``cholesky_update``,
``householder_product``, ``ormqr``, ``lu_pivots_to_permutation`` and ``tridiagonal_solve`` each
as a fixed sequence of IR operations whose products are the IR ``matmul``, a left fold from
``+0``. A capture's builder subclasses it and supplies ``nodes``, ``node``, ``flags``, ``scalar``,
``current_dtype`` (the dtype of the nodes it makes) and ``refusal`` (the exception of its
refusals). The Jaxpr importer (``torchlean.jaxpr``) imports all seven; ``pivot_permutation``
also uses its ``start``, ``iota``, ``bounded`` and ``refusing``, and ``tridiagonal_solve`` its
``refusing``. The PyTorch capture (``torchlean._emit``) imports
``householder_product`` and ``ormqr``. The methods that build NumPy masks import NumPy
themselves, so the PyTorch capture of other operators does not need it.

The docstrings below name the definitions and relations of
``TorchLean.Tensor.Factorization.Decomposition`` (``householderOp``, ``householderProduct``,
``pivotPerm``, ``IsTridiagonalSolve``) as the statements each program is read against. No theorem
connects an imported program with them: the tests compare its values with JAX and PyTorch.
"""

from __future__ import annotations

import math
from typing import Any


class LinalgImport:
    def shape(self, index: int) -> list[int]:
        return self.nodes[index]["shape"]

    def reshape(self, index: int, shape: list[int]) -> int:
        source = self.shape(index)
        if source == shape:
            return index
        if math.prod(source) != math.prod(shape):
            raise self.refusal(f"reshape changes element count: {source} -> {shape}")
        return self.node("reshape", [index], shape, dtype=self.nodes[index]["dtype"])

    def permute(self, index: int, axes: list[int]) -> int:
        source = self.shape(index)
        if sorted(axes) != list(range(len(source))):
            raise self.refusal(f"invalid axis permutation {axes} for {source}")
        if axes == list(range(len(source))):
            return index
        return self.node("permute", [index], [source[i] for i in axes],
                         dtype=self.nodes[index]["dtype"], dims=axes)

    def broadcast(self, index: int, shape: list[int], dims: list[int] | None = None) -> int:
        """Place source axis ``i`` at target axis ``dims[i]`` (by default, the trailing axes)."""
        source = self.shape(index)
        if dims is None:
            dims = list(range(len(shape) - len(source), len(shape)))
        if source == shape and dims == list(range(len(shape))):
            return index
        if (len(dims) != len(source) or any(a >= b for a, b in zip(dims, dims[1:]))
                or any(d < 0 or d >= len(shape) for d in dims)
                or any(source[i] not in (1, shape[d]) for i, d in enumerate(dims))):
            raise self.refusal(f"invalid broadcast {source} -> {shape} along {dims}")
        return self.node("broadcast", [index], shape, dtype=self.nodes[index]["dtype"],
                         dims=dims)

    def slice(self, index: int, starts: list[int], stops: list[int],
              steps: list[int]) -> int:
        """One slice node over every axis whose window is not the whole axis."""
        source = self.shape(index)
        dims = [d for d in range(len(source))
                if (starts[d], stops[d], steps[d]) != (0, source[d], 1)]
        if not dims:
            return index
        if any(steps[d] < 1 or not 0 <= starts[d] <= stops[d] <= source[d] for d in dims):
            raise self.refusal(f"slice: invalid window {starts}:{stops}:{steps} of {source}")
        shape = [len(range(starts[d], stops[d], steps[d])) for d in range(len(source))]
        return self.node("slice", [index], shape, dtype=self.nodes[index]["dtype"], dims=dims,
                         starts=[starts[d] for d in dims], stops=[stops[d] for d in dims],
                         steps=[steps[d] for d in dims])

    def binary(self, name: str) -> None:
        """Refuse the operation ``name`` unless its nodes are binary32 or binary64."""
        if self.current_dtype not in ("float32", "float64"):
            raise self.refusal(f"{name}: only binary32 and binary64 operands are imported")

    def block(self, index: int, rows: tuple[int, int], cols: tuple[int, int]) -> int:
        """Rows ``rows[0]:rows[1]`` and columns ``cols[0]:cols[1]`` of each matrix in the last two
        axes of ``index``."""
        extents = self.shape(index)
        rank = len(extents)
        return self.slice(index, [0] * (rank - 2) + [rows[0], cols[0]],
                          extents[:-2] + [rows[1], cols[1]], [1] * rank)

    def triangular_solve(self, a: int, b: int, params: dict[str, Any], shape: list[int]) -> int:
        """op(a) x = b, or x op(a) = b, by forward substitution on the equivalent lower, left-side
        system. A transposed op(a) is a with its last two axes swapped, an upper op(a) turns lower
        by reversing both of its axes (and the right-hand side's rows), and a right-side equation
        is transposed. Row i is (b_i - L[i, :i] x[:i]) / L_ii, with the product the IR matmul (a
        left fold from +0) and no division for a unit diagonal. Only the used triangle is read."""
        self.binary("triangular_solve")
        rank = len(shape)
        swap = list(range(rank - 2)) + [rank - 1, rank - 2]
        lower = bool(params["lower"]) != bool(params["transpose_a"])
        if params["transpose_a"]:
            a = self.permute(a, swap)
        if not params["left_side"]:
            a, b, lower = self.permute(a, swap), self.permute(b, swap), not lower
        if not lower:
            a = self.node("flip", [a], self.shape(a), dims=[rank - 2, rank - 1])
            b = self.node("flip", [b], self.shape(b), dims=[rank - 2])
        size, lead, width = self.shape(a)[-1], self.shape(b)[:-2], self.shape(b)[-1]
        solved: list[int] = []
        for i in range(size):
            row = self.block(b, (i, i + 1), (0, width))
            if i:
                known = self.node("cat", list(solved), lead + [i, width], dim=rank - 2) \
                    if i > 1 else solved[0]
                product = self.node("matmul", [self.block(a, (i, i + 1), (0, i)), known],
                                    lead + [1, width])
                row = self.node("sub", [row, product], lead + [1, width])
            if not params["unit_diagonal"]:
                pivot = self.broadcast(self.block(a, (i, i + 1), (i, i + 1)), lead + [1, width])
                row = self.node("div", [row, pivot], lead + [1, width])
            solved.append(row)
        x = solved[0] if size == 1 else self.node("cat", solved, lead + [size, width],
                                                  dim=rank - 2) if size else b
        if not lower:
            x = self.node("flip", [x], self.shape(x), dims=[rank - 2])
        return x if params["left_side"] else self.permute(x, swap)

    def cholesky_update(self, r: int, z: int, shape: list[int], numpy: Any) -> int:
        """The upper factor R' with R'^T R' = R^T R + z z^T (the documented result) as JAX's own
        computation of it: for k = 0, ..., n - 1 the Givens rotation of (R[k, k], z[k]), c = 1 and
        s = 0 where z[k] = 0, otherwise x = R[k, k] / d, y = z[k] / d with d the larger magnitude,
        rh = 1 / sqrt(x^2 + y^2), c = x rh, s = -y rh, applied to row k and z:
        R[k, :] = c R[k, :] - s z and z = c z + s R[k, :]. JAX's ``cond`` evaluates one branch; the
        graph evaluates both and selects, so where z[k] = 0 it rotates (1, 0) in the branch not
        taken, keeping that branch and its zero cotangent finite where R[k, k] = 0 too
        (0 / 0 otherwise)."""
        self.binary("cholesky_update")
        size = shape[-1]
        if len(shape) != 2 or self.shape(z) != [size]:
            raise self.refusal("cholesky_update: one square factor and one vector are imported")
        rows = [self.reshape(self.slice(r, [k, 0], [k + 1, size], [1, 1]), [size])
                for k in range(size)]
        for k in range(size):
            y = self.slice(z, [k], [k + 1], [1])
            zero = self.node("eq", [y, self.scalar(0.0, numpy, [1])], [1], dtype="bool")
            x = self.node("where", [zero, self.scalar(1.0, numpy, [1]),
                                    self.slice(rows[k], [k], [k + 1], [1])], [1])
            ax, ay = self.node("abs", [x], [1]), self.node("abs", [y], [1])
            larger = self.node("where", [self.node("gt", [ax, ay], [1], dtype="bool"), ax, ay],
                               [1])
            xn, yn = self.node("div", [x, larger], [1]), self.node("div", [y, larger], [1])
            norm = self.node("sqrt", [self.node("add", [self.node("mul_elem", [xn, xn], [1]),
                                                        self.node("mul_elem", [yn, yn], [1])],
                                                [1])], [1])
            rh = self.node("div", [self.scalar(1.0, numpy, [1]), norm], [1])
            c = self.node("where", [zero, self.scalar(1.0, numpy, [1]),
                                    self.node("mul_elem", [xn, rh], [1])], [1])
            s = self.node("where", [zero, self.scalar(0.0, numpy, [1]), self.node(
                "mul_elem", [self.node("neg", [yn], [1]), rh], [1])], [1])
            # Rotate row k and z together.
            c, s = self.broadcast(c, [size]), self.broadcast(s, [size])
            row = rows[k]
            rows[k] = self.node("sub", [self.node("mul_elem", [c, row], [size]),
                                        self.node("mul_elem", [s, z], [size])], [size])
            z = self.node("add", [self.node("mul_elem", [c, z], [size]),
                                  self.node("mul_elem", [s, row], [size])], [size])
        if size == 0:
            return r
        return self.node("cat", [self.reshape(row, [1, size]) for row in rows], shape, dim=0) \
            if size > 1 else self.reshape(rows[0], shape)

    def cholesky(self, a: int, shape: list[int]) -> int:
        """The lower Cholesky factor by the left-looking column algorithm, reading the lower
        triangle. Column j is zero above the diagonal, sqrt(d_j) on it with the pivot
        d_j = a_jj - L[j, :j] L[j, :j]^T, and (a_ij - L[i, :j] L[j, :j]^T) / sqrt(d_j) below it,
        each product the IR matmul. Every entry is NaN unless every pivot is positive, LAPACK
        potrf's failure test (a NaN pivot fails: ``A( 1, 1 ).LE.ZERO.OR.DISNAN( A( 1, 1 ) )`` in
        LAPACK 3.12.0's dpotrf2.f, which dpotrf.f calls on each diagonal block), as
        lax.linalg.cholesky documents ("a matrix full of NaNs")."""
        self.binary("cholesky")
        dtype = self.current_dtype
        rank, size = len(shape), shape[-1]
        lead = shape[:-2]
        if size == 0:
            return a
        swap = list(range(rank - 2)) + [rank - 1, rank - 2]
        # `columns` holds the columns of L computed so far, and `valid` whether every pivot so
        # far is positive.
        columns: list[int] = []
        valid = None
        for j in range(size):
            known = (columns[0] if j == 1 else self.node("cat", list(columns), lead + [size, j],
                                                         dim=rank - 1)) if j else None
            pivot = self.block(a, (j, j + 1), (j, j + 1))
            if known is not None:
                row = self.block(known, (j, j + 1), (0, j))
                across = self.permute(row, swap)
                pivot = self.node("sub", [pivot, self.node("matmul", [row, across],
                                                           lead + [1, 1])], lead + [1, 1])
            test = self.node("gt", [pivot, self.scalar(0.0, dtype, lead + [1, 1])], lead + [1, 1],
                             dtype="bool")
            valid = test if valid is None else self.node("logical_and", [valid, test],
                                                         lead + [1, 1], dtype="bool")
            # A nonpositive pivot's root is taken at 0, and `valid` makes the factor NaN.
            root = self.node("sqrt", [self.node("max_elem", [pivot, self.scalar(
                0.0, dtype, lead + [1, 1])], lead + [1, 1])], lead + [1, 1])
            pieces = [self.scalar(0.0, dtype, lead + [j, 1])] if j else []
            pieces.append(root)
            if j + 1 < size:
                below = self.block(a, (j + 1, size), (j, j + 1))
                if known is not None:
                    product = self.node("matmul", [self.block(known, (j + 1, size), (0, j)),
                                                   across], lead + [size - j - 1, 1])
                    below = self.node("sub", [below, product], lead + [size - j - 1, 1])
                pieces.append(self.node("div", [below, self.broadcast(
                    root, lead + [size - j - 1, 1])], lead + [size - j - 1, 1]))
            columns.append(pieces[0] if len(pieces) == 1 else self.node(
                "cat", pieces, lead + [size, 1], dim=rank - 2))
        factor = columns[0] if size == 1 else self.node("cat", columns, shape, dim=rank - 1)
        return self.node("where", [self.broadcast(valid, shape),
                                   factor, self.scalar(math.nan, dtype, shape)], shape)

    def reflected(self, name: str, a: int, taus: int, c: int, left: bool,
                  transpose: bool) -> int:
        """``op(Q) c`` where ``left`` is set and ``c op(Q)`` otherwise, with
        ``Q = H_0 H_1 ... H_{k-1}`` and ``op(Q)`` its transpose where ``transpose`` is set:
        ``householderOp`` of ``TorchLean.Tensor.Factorization.Decomposition``, the product that
        ``lax.linalg.ormqr`` and ``torch.ormqr`` document. ``H_i = 1 - tau_i v_i v_i^T`` is the
        reflector of ``taus[i]`` and of the vector ``v_i`` that column ``i`` of ``a`` stores: zero
        above row ``i``, one at row ``i`` and the column below it. The reflectors are applied to
        ``c`` one at a time in the order of the product, the last first for ``Q c`` and
        ``c Q^T`` and the first first for ``Q^T c`` and ``c Q``: ``c - tau_i (v_i (v_i^T c))`` on
        the left and ``c - tau_i ((c v_i) v_i^T)`` on the right, ``v_i^T c`` and ``c v_i`` the IR
        matmul and each entry of the outer product, of its multiple and of the difference rounded
        once.
        The reflector of a column past the last row is the identity (``v_i`` is zero), and is
        skipped. JAX's shape rule and ``torch.ormqr``'s docstring (``mn``) give ``c`` as many
        rows (on the left) or columns (on the right) as ``a`` has rows."""
        import numpy as np

        self.binary(name)
        extents, shape, k = self.shape(a), self.shape(c), self.shape(taus)[-1]
        rank, rows, lead = len(extents), extents[-2], extents[:-2]
        if k > extents[-1] or self.shape(taus)[:-1] != lead or shape[:-2] != lead:
            raise self.refusal(f"{name}: at most one scalar factor per column of reflectors, "
                               "and one batch shape, are imported")
        swap = list(range(rank - 2)) + [rank - 1, rank - 2]
        dtype, square = self.current_dtype, lead + [rows, k]
        mask = lambda flags: self.broadcast(self.flags(flags), square)
        # The reflector vectors: `a` below the diagonal, ones on it and zeros above.
        vectors = self.node("where", [
            mask(np.tri(rows, k, -1, dtype=bool)), self.block(a, (0, rows), (0, k)),
            self.node("where", [mask(np.eye(rows, k, dtype=bool)), self.scalar(1, dtype, square),
                                self.scalar(0, dtype, square)], square)], square)
        # The last reflector first for `Q c` and `c Q^T`, the first first otherwise.
        order = range(min(k, rows))
        for i in reversed(order) if left != transpose else order:
            v = self.block(vectors, (0, rows), (i, i + 1))
            tau = self.reshape(self.slice(taus, [0] * (rank - 2) + [i], lead + [i + 1],
                                          [1] * (rank - 1)), lead + [1, 1])
            if left:
                product = self.node("matmul", [self.permute(v, swap), c], lead + [1, shape[-1]])
                outer = [self.broadcast(v, shape), self.broadcast(product, shape)]
            else:
                product = self.node("matmul", [c, v], lead + [shape[-2], 1])
                outer = [self.broadcast(product, shape),
                         self.broadcast(self.permute(v, swap), shape)]
            update = self.node("mul_elem", [self.broadcast(tau, shape),
                                            self.node("mul_elem", outer, shape)], shape)
            c = self.node("sub", [c, update], shape)
        return c

    def householder_product(self, a: int, taus: int, shape: list[int]) -> int:
        """The first ``n`` columns of ``Q = H_0 H_1 ... H_{k-1}`` for an ``a`` of shape
        ``[..., m, n]``: ``householderProduct`` of ``TorchLean.Tensor.Factorization.
        Decomposition``, the product that ``lax.linalg.householder_product`` and
        ``torch.linalg.householder_product`` document, computed as ``Q`` applied to the first
        ``n`` columns of the identity (``reflected``)."""
        import numpy as np

        self.binary("householder_product")
        dtype = self.current_dtype
        identity = self.node("where", [
            self.broadcast(self.flags(np.eye(*shape[-2:], dtype=bool)), shape),
            self.scalar(1, dtype, shape), self.scalar(0, dtype, shape)], shape)
        return self.reflected("householder_product", a, taus, identity, True, False)

    def pivot_permutation(self, pivots: int, shape: list[int]) -> int:
        """The permutation of the row swaps ``pivots`` of shape ``[..., k]``: from ``iota`` of
        ``shape``, the entries at ``i`` and ``pivots[i]`` swapped for ``i = 0, ..., k - 1`` in
        turn, ``pivotPerm`` of ``TorchLean.Tensor.Factorization.Decomposition``, whose docstring
        cites ``lax.linalg.lu_pivots_to_permutation``. The entries are held in binary64, exact
        below ``2 ** 53``, and the entry at ``pivots[i]`` is the sum of the entries that ``eq``
        selects there. The documentation names no permutation for a swap with a row outside
        ``[0, permutation_size)``, and execution fails at such a pivot (``refusing``)."""
        import numpy as np

        lead, size, k = shape[:-1], shape[-1], self.shape(pivots)[-1]
        swaps = self.start(pivots)
        position = self.iota(shape, len(lead))
        entries = self.node("custom", [position], shape, dtype="float64", primitive="cast",
                            rounding="nearestEven")
        zero = self.scalar(0.0, np.float64, shape)
        for i in range(k):
            row = self.slice(swaps, [0] * len(lead) + [i], lead + [i + 1], [1] * len(shape))
            hit = self.node("eq", [position, self.broadcast(row, shape)], shape, dtype="bool")
            read = self.node("reduce", [self.node("where", [hit, entries, zero], shape,
                                                  dtype="float64")],
                             lead, dtype="float64", op="sum", dims=[len(lead)], keepdim=False)
            mine = self.slice(entries, [0] * len(lead) + [i], lead + [i + 1], [1] * len(shape))
            here = self.broadcast(self.flags(np.arange(size) == i), shape)
            entries = self.node("where", [here, self.broadcast(read, shape, list(range(len(lead)))),
                                          entries], shape, dtype="float64")
            entries = self.node("where", [hit, self.broadcast(mine, shape), entries], shape,
                                dtype="float64")
        if k:
            outside = self.node("logical_not", [self.bounded(swaps, size - 1, False)],
                                lead + [k], dtype="bool")
            count = self.node("reduce", [self.node(
                "custom", [outside], lead + [k], dtype="float64", primitive="cast",
                rounding="nearestEven")], [], dtype="float64", op="sum",
                dims=list(range(len(lead) + 1)), keepdim=False)
            entries = self.refusing(self.node("ne", [count, self.scalar(0.0, np.float64, [])],
                                              [], dtype="bool"), entries)
        return self.node("custom", [entries], shape, primitive="cast", rounding="towardZero")

    def tridiagonal_solve(self, dl: int, d: int, du: int, b: int, shape: list[int]) -> int:
        """The solution ``x`` of ``A x = b`` for ``b`` of shape ``[..., m, n]`` and the tridiagonal
        ``A`` with ``A[i, i - 1] = dl[i]``, ``A[i, i] = d[i]`` and ``A[i, i + 1] = du[i]``
        (``dl[0]`` and ``du[m - 1]`` are not read): the system that ``lax.linalg.tridiagonal_solve``
        documents and ``IsTridiagonalSolve`` of ``TorchLean.Tensor.Factorization.Decomposition``
        states, by the LU factorization with partial pivoting that the docstring names (under
        ``perturb_singular``) and its CPU lowering calls, LAPACK ``gtsv`` (``gtsv_ffi``; we read
        dgtsv.f and sgtsv.f of LAPACK 3.12.0, whose test is ``ABS( D( I ) ).GE.ABS( DL( I ) )``,
        false at a NaN). Step ``i`` keeps rows ``i`` and ``i + 1`` where ``|u_i| >= |l_i|``, ``u_i``
        the current diagonal entry of row ``i`` and ``l_i = dl[i + 1]`` the one below it,
        subtracting ``(l_i / u_i)`` times row ``i`` from row ``i + 1``; otherwise (a NaN included)
        it swaps them first and subtracts ``(u_i / l_i)`` times the new row ``i``, which fills the
        second superdiagonal. Back substitution is
        ``x_i = ((b_i - c_i x_{i+1}) - e_i x_{i+2}) / u_i``, with ``c`` and ``e`` the first and
        second superdiagonals; each operation is rounded once. Both quotients are built before the
        selection, so the one of the branch not taken divides by 1 instead: it stays finite at
        finite operands (``1 / 0`` at the identity's ``l_i = 0`` otherwise), and so does its
        pullback, the zero cotangent ``where`` gives it (``0 * inf`` is NaN). The values selected
        are unchanged. Execution fails (``refusing``) at a zero pivot, a kept ``u_i = 0`` or a last
        ``u_{m-1} = 0``, gtsv's test of the rounded pivots (``INFO = I`` at a kept ``D( I ) = 0``
        and ``INFO = N`` at ``D( N ) = 0``): in exact arithmetic that happens exactly where ``A`` is
        singular, for which the documentation names no result."""
        self.binary("tridiagonal_solve")
        dtype, rank, lead = self.current_dtype, len(shape), shape[:-2]
        size, width = shape[-2:]
        if size == 0:
            return b
        one, wide = lead + [1, 1], lead + [1, width]
        pair = lambda name, x, y, extents=one, kind=None: self.node(
            name, [x, y], extents, **({"dtype": kind} if kind else {}))
        scaled = lambda factor, row: pair("mul_elem", self.broadcast(factor, wide), row, wide)

        def entries(v: int, indices: range) -> list[int]:
            column = self.reshape(v, lead + [size, 1]) if indices else v
            return [self.block(column, (i, i + 1), (0, 1)) for i in indices]

        lower, diag, upper = (entries(dl, range(1, size)), entries(d, range(size)),
                              entries(du, range(size - 1)))
        rows = [self.block(b, (i, i + 1), (0, width)) for i in range(size)]
        zero, unit = self.scalar(0.0, dtype, one), self.scalar(1.0, dtype, one)
        fill: list[int] = []
        failing = []
        for i in range(size - 1):
            keep = pair("ge", self.node("abs", [diag[i]], one), self.node("abs", [lower[i]], one),
                        kind="bool")
            failing.append(pair("logical_and", keep, pair("eq", diag[i], zero, kind="bool"),
                                kind="bool"))
            choose = lambda kept, swapped, extents=one: self.node(
                "where", [self.broadcast(keep, extents), kept, swapped], extents)
            kept = pair("div", lower[i], choose(diag[i], unit))
            swapped = pair("div", diag[i], choose(unit, lower[i]))
            following = choose(pair("sub", diag[i + 1], pair("mul_elem", kept, upper[i])),
                               pair("sub", upper[i], pair("mul_elem", swapped, diag[i + 1])))
            diag[i], upper[i], diag[i + 1] = (choose(diag[i], lower[i]),
                                              choose(upper[i], diag[i + 1]), following)
            if i + 2 < size:
                fill.append(choose(zero, upper[i + 1]))
                upper[i + 1] = choose(upper[i + 1], pair(
                    "mul_elem", self.node("neg", [swapped], one), upper[i + 1]))
            rows[i], rows[i + 1] = (
                choose(rows[i], rows[i + 1], wide),
                choose(pair("sub", rows[i + 1], scaled(kept, rows[i]), wide),
                       pair("sub", rows[i], scaled(swapped, rows[i + 1]), wide), wide))
        failing.append(pair("eq", diag[-1], zero, kind="bool"))
        solved: list[int] = [0] * size
        for i in reversed(range(size)):
            row = rows[i]
            for offset, factors in ((1, upper), (2, fill)):
                if i + offset < size:
                    row = pair("sub", row, scaled(factors[i], solved[i + offset]), wide)
            solved[i] = pair("div", row, self.broadcast(diag[i], wide), wide)
        x = solved[0] if size == 1 else self.node("cat", solved, shape, dim=rank - 2)
        singular = failing[0]
        for test in failing[1:]:
            singular = pair("logical_or", singular, test, kind="bool")
        count = self.node("reduce", [self.node(
            "custom", [singular], one, dtype="float64", primitive="cast",
            rounding="nearestEven")], [], dtype="float64", op="sum", dims=list(range(rank)),
            keepdim=False)
        return self.refusing(self.node("ne", [count, self.scalar(0.0, "float64", [])], [],
                                       dtype="bool"), x)
