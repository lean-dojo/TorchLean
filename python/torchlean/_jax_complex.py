"""The importer's complex values, held in paired-real form.

``torchlean.jaxpr``'s builder holds a complex value of shape ``s`` as a tensor of shape
``s + [2]`` in the parts' format, the real part at 0 and the imaginary part at 1 of the last axis,
as ``Op.rfft`` writes and ``Op.irfft`` reads it. ``ComplexImport`` is the part of that builder
that imports a primitive with a complex operand or result: its Fourier transforms and its
arithmetic on the parts. The importer's errors and helpers are read from ``torchlean.jaxpr`` when
a primitive is imported, since that module imports this one.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from ._table import op_table


def _wide(component: str, precision: int | None = None) -> dict[str, Any]:
    """The binary format in which a complex product, quotient or phase (``phase``) with parts of
    the format ``component`` is computed: of ``precision`` significand bits, by default the
    precision at which each part of a product or quotient is its exact value rounded once.

    Finite parts of precision ``p`` and exponents ``emin`` to ``emax`` are integer multiples of
    ``s = 2^(emin - p + 1)`` below ``2^(emax + 1)``, so a sum of two of their products is a
    multiple of ``s^2`` below ``2^(2 emax + 3)``: it has at most ``2 (emax - emin + p) + 1``
    significant bits, and the products and the sums are exact here. A point where rounding to
    the parts' format changes, a midpoint between two neighbours or the overflow threshold, is an
    odd multiple ``m`` of ``2^k`` for some ``k`` from ``emin - p`` to ``emax - p``, below
    ``2^(k + p + 1)``. If a quotient ``x = N / D`` of two such sums is not ``m`` but rounds to
    ``m`` at precision ``P``, then ``|x - m| <= 2^-P |x|``, so ``|x| < 2^(k + p + 2)``. But
    ``N - m D`` is a nonzero multiple of ``s^2 2^min(k, 0)`` and ``|N| = D |x|`` is below
    ``2^(2 emax + 3)`` and below ``2^(2 emax + k + p + 5)``, so
    ``|x - m| / |x| = |N - m D| / |N| > 2^-P`` for ``P = 2 (emax - emin) + 3 p + 3``. At that
    precision a quotient therefore rounds to a value on the same side of every such point as
    itself, or to itself where it is one, and rounding that again to the parts' format gives the
    quotient rounded once. The quotients lie between ``2^-(2 (emax - emin + p) + 1)`` and
    ``2^(2 (emax - emin + p) + 1)``, so an exponent bias above ``2 (emax - emin + p) + 1`` keeps
    every value normal. Binary32 parts take 581 significand bits and 11 exponent bits (bias
    1023), and binary64 parts 4252 and 14 (bias 8191)."""
    from .jaxpr import _PRECISIONS
    p, emin, emax = _PRECISIONS[component]
    bits = (2 * (emax - emin + p) + 2).bit_length() + 1
    return dict(kind="binary", exponent_bits=bits,
                fraction_bits=(precision or 2 * (emax - emin) + 3 * p + 3) - 1,
                bias=2 ** (bits - 1) - 1, encoding="ieee")


# The lax primitives that TorchLean.Op.complex applies to a complex operand, the names of its
# functions (ComplexFn.jaxName).
_FUNCTIONS = ("exp", "expm1", "log", "log1p", "sqrt", "rsqrt", "sin", "cos", "tan", "sinh", "cosh",
              "tanh", "asin", "acos", "atan", "asinh", "acosh", "atanh")


class ComplexImport:
    """The complex primitives of ``torchlean.jaxpr``'s builder, on the paired-real form."""

    def part(self, index: int, k: int) -> int:
        """The real (``k = 0``) or imaginary (``k = 1``) part of the paired node ``index``."""
        source = self.shape(index)
        starts, stops = [0] * len(source), list(source)
        starts[-1], stops[-1] = k, k + 1
        return self.reshape(self.slice(index, starts, stops, [1] * len(source)), source[:-1])

    def hermitian(self, index: int, n: int) -> int:
        """The paired spectrum ``index`` of an inverse real transform of length ``n`` along its
        last complex axis, with ``+0`` for the imaginary parts of frequency ``0`` and, for an even
        ``n``, ``n / 2``. A real signal's spectrum is real there, and NumPy's ``irfft``, which
        ``jnp.fft.irfft`` follows, treats those entries as real. ``Op.irfft`` multiplies these
        parts by a zero coefficient, which a NaN or an infinity would turn into a NaN."""
        source = self.shape(index)
        ignored = np.zeros(source[-2:], dtype=bool)
        ignored[0, 1] = True
        if n % 2 == 0 and n // 2 < source[-2]:
            ignored[n // 2, 1] = True
        mask = self.broadcast(self.flags(ignored), source, [len(source) - 2, len(source) - 1])
        dtype = self.nodes[index]["dtype"]
        return self.node("where", [mask, self.scalar(0, dtype, source), index], source,
                         dtype=dtype)

    def spectrum(self, x: int, axis: int, n: int) -> tuple[int, int]:
        """The real and imaginary parts of the discrete Fourier transform along ``axis`` of length
        ``n`` of the real node ``x``: ``Op.rfft``'s frequencies ``0`` through ``n / 2``, then the
        conjugates of frequencies ``n - n / 2 - 1`` down to ``1``, since frequency ``n - k`` of a
        real signal is the conjugate of frequency ``k``."""
        source, dtype = self.shape(x), self.nodes[x]["dtype"]
        half = source[:axis] + [n // 2 + 1] + source[axis + 1:]
        index = self.registered("rfft", [{"shape": source}, {"nat": axis}, {"nat": n}], [x],
                                half + [2], dtype, dtype=dtype)
        parts = [self.part(index, k) for k in (0, 1)]
        if n - n // 2 == 1:
            return parts[0], parts[1]
        starts, stops = [0] * len(half), list(half)
        starts[axis], stops[axis] = 1, n - n // 2
        tails = [self.slice(i, starts, stops, [1] * len(half)) for i in parts]
        tails = [self.node("flip", [i], self.shape(i), dtype=dtype, dims=[axis]) for i in tails]
        tails[1] = self.node("neg", [tails[1]], self.shape(tails[1]), dtype=dtype)
        real, imag = [self.node("cat", [head, tail], source, dtype=dtype, dim=axis)
                      for head, tail in zip(parts, tails)]
        return real, imag

    def fourier(self, index: int, axis: int, inverse: bool) -> int:
        """The discrete Fourier transform along ``axis`` of the paired node ``index``, or with
        ``inverse`` its inverse, which divides by the length ``n``. With ``p + iq`` the operand
        and ``a + ib`` and ``c + id`` the transforms of ``p`` and ``q`` (``spectrum``), the
        transform is ``(a - d) + i(b + c)``. The inverse transform of a real signal is the
        conjugate of its transform over ``n``, so the inverse is ``((a + d) + i(c - b)) / n``."""
        shape = self.shape(index)[:-1]
        dtype, n = self.nodes[index]["dtype"], shape[axis]
        (a, b), (c, d) = [self.spectrum(self.part(index, k), axis, n) for k in (0, 1)]
        if not inverse:
            return self.pair(self.node("sub", [a, d], shape, dtype=dtype),
                             self.node("add", [b, c], shape, dtype=dtype))
        scale = self.scalar(n, dtype, shape)
        return self.pair(*[self.node("div", [self.node(op, terms, shape, dtype=dtype), scale],
                                     shape, dtype=dtype)
                           for op, terms in (("add", [a, d]), ("sub", [c, b]))])

    def pair(self, real: int, imag: int) -> int:
        """The paired node of the complex value whose parts are the nodes ``real`` and ``imag``."""
        shape = self.shape(real)
        index = self.node("cat", [self.reshape(i, shape + [1]) for i in (real, imag)], shape + [2],
                          dtype=self.nodes[real]["dtype"], dim=len(shape))
        self.paired.add(index)
        return index

    def operand(self, index: int, shape: list[int], component: str) -> list[int]:
        """The parts of an operand of a primitive whose result has parts of the format
        ``component``, converted to that format and broadcast to ``shape``: a real operand is the
        real part, with imaginary part ``+0``. JAX 0.11.1 lowers ``lax.mul`` with an
        ``out_dtype`` so: ``_nary_lower_hlo`` (``jax/_src/lax/lax.py``) applies StableHLO's
        ``convert`` to each operand before multiplying. The documentation of ``lax.mul`` says only
        that this is typically how the conversion is done and that some backends may use a custom
        kernel."""
        parts = ([self.part(index, k) for k in (0, 1)] if index in self.paired
                 else [index, self.scalar(0, component, self.shape(index))])
        return [self.broadcast(self.converted(i, component), shape) for i in parts]

    def converted(self, index: int, dtype: Any) -> int:
        """The node ``index`` rounded once to ``dtype`` (IEEE 754-2019 5.4.1, 5.4.2)."""
        if self.nodes[index]["dtype"] == dtype:
            return index
        return self.node("custom", [index], self.shape(index), dtype=dtype, primitive="cast",
                         rounding="nearestEven")

    def product(self, a: int, b: int, c: int, d: int, shape: list[int],
                component: str) -> tuple[int, int]:
        """The real and imaginary parts of ``(a + ib)(c + id)``, written by hand from the
        definition of the product, ``ac - bd`` and ``ad + bc`` (Mathlib's ``Complex.mul_re`` and
        ``Complex.mul_im``). The parts are converted exactly to ``_wide(component)``, where the
        four products and the two sums are exact, and each result is rounded once to
        ``component``: each part of finite operands is the exact part rounded to nearest with ties
        to even (IEEE 754-2019 4.3.1), an infinity only where that overflows, and a part stays
        finite where the other overflows. JAX documents ``lax.mul`` as ``x × y`` and names no
        rounding of the parts; XLA's CPU code fuses products, so its last places can differ. An
        infinite or NaN part gives what IEEE 754-2019's operations give the formula
        without rounding, where a product of finite parts is finite, so that
        ``(inf + 2^600 i)(1 + 2^600 i)`` is ``inf + inf i`` in binary64, and without the recovery
        of C's Annex G: ``(inf + inf i)(inf + 0i)`` is ``NaN + NaN i``."""
        wide = _wide(component)
        a, b, c, d = [self.converted(i, wide) for i in (a, b, c, d)]

        def term(kind: str, x: tuple[int, int], y: tuple[int, int]) -> int:
            products = [self.node("mul_elem", list(p), shape, dtype=wide) for p in (x, y)]
            return self.converted(self.node(kind, products, shape, dtype=wide), component)

        return term("sub", (a, c), (b, d)), term("add", (a, d), (b, c))

    def quotient(self, a: int, b: int, c: int, d: int, shape: list[int],
                 component: str) -> tuple[int, int]:
        """The real and imaginary parts of ``(a + ib) / (c + id)``, written by hand from the
        definition of the quotient, ``(ac + bd) / (c^2 + d^2)`` and ``(bc - ad) / (c^2 + d^2)``
        (Mathlib's ``Complex.div_re`` and ``Complex.div_im``, over one denominator), computed in
        ``_wide(component)`` as ``product`` computes. There the products, the numerators and the
        denominator are exact, and each quotient is rounded once to a precision at which, by the
        bound of ``_wide``, rounding it again to ``component`` gives the exact quotient rounded
        once: each part of finite operands with a nonzero divisor is the exact part rounded to
        nearest with ties to even, an infinity only where that overflows, and
        ``(2^127 + 2^127 i) / (2^127 + 2^127 i)`` is ``1 + 0i`` in binary32. JAX documents
        ``lax.div`` as ``x ÷ y`` and names no algorithm; XLA's CPU code divides by Smith's
        algorithm, whose last places differ at most operands and which loses that quotient. A zero
        or infinite divisor, or an infinite or NaN part, gives what IEEE
        754-2019's operations give the formula without rounding, and without the recovery of C's
        Annex G: ``(1 - i) / 0`` and ``(1 - i) / (inf + inf i)`` are ``NaN + NaN i``."""
        wide = _wide(component)
        a, b, c, d = [self.converted(i, wide) for i in (a, b, c, d)]

        def op(kind: str, *terms: int) -> int:
            return self.node(kind, list(terms), shape, dtype=wide)

        def term(kind: str, x: tuple[int, int], y: tuple[int, int]) -> int:
            return op(kind, op("mul_elem", *x), op("mul_elem", *y))

        norm = term("add", (c, c), (d, d))
        return tuple(self.converted(self.divided(term(kind, x, y), norm, shape, wide), component)
                     for kind, x, y in (("add", (a, c), (b, d)), ("sub", (b, c), (a, d))))

    def phase(self, a: int, b: int, shape: list[int], component: str) -> tuple[int, int]:
        """The real and imaginary parts of the phase of ``a + ib``, ``(a + ib) / |a + ib|``, which
        JAX documents as ``lax.sign`` of a complex value (jax/_src/lax/lax.py:304 in JAX 0.11.1),
        written by hand from that definition: ``a / sqrt(a^2 + b^2)`` and ``b / sqrt(a^2 + b^2)``,
        and ``a + ib`` itself where both parts are zero, as ``lax.sign`` gives a real zero its own
        sign. They are computed in ``_wide(component, P)`` for ``P = 2 emax - 3 emin + 4 p + 6``,
        where ``N = a^2 + b^2`` is exact (``_wide``), ``sqrt(N)`` and the quotient are each
        rounded once, and the result rounded again to ``component``: each part of finite
        operands is the exact part rounded to nearest with ties to even.

        The exact part ``u = a / sqrt(N)`` lies in ``[-1, 1]`` and the computed one is ``u (1 +
        t)`` with ``|t| < 2^(2 - P)``. A point where rounding to ``component`` changes, near
        ``u``, is a midpoint ``m``, an odd multiple of ``2^k`` with ``emin - p <= k < 0`` and
        ``|m| < 2^(k + p + 1)``, and ``u`` is never one: ``u = m = M 2^-j`` with ``M`` odd would
        make ``b / sqrt(N) = b m / a`` a rational ``r / q`` in lowest terms with ``m^2 + (r / q)^2
        = 1``, so that ``2^j`` divides ``q``, ``q = 2^j`` and ``r`` is odd, and ``M^2 + r^2 = 4^j``,
        which is 2 modulo 4. So ``a^2 - m^2 N``, a multiple of ``s^2 2^(2k)`` for ``s = 2^(emin -
        p + 1)``, is at least that in magnitude, and ``|u - m| = |a^2 - m^2 N| / (N |u + m|)`` is
        at least ``s^2 2^(2k) / (2 N)`` for ``m`` of the sign of ``u``, with ``N < 2^(2 emax +
        3)``. Were ``m`` between ``u`` and the computed part or equal to it, ``|u - m| <= |t| |u|``
        and ``|u| < 2^(k + p + 2)``, which that bound excludes at this ``P``. Binary32 parts take
        734 significand bits and binary64 parts 5330, with ``_wide``'s exponents, between whose
        bounds ``N``, ``sqrt(N)`` and the quotients, at least ``2^(emin - p - emax - 1)`` in
        magnitude, lie. An infinite or NaN part gives what IEEE 754-2019's operations give the
        formula without rounding, so that ``inf + 1i`` gives ``NaN + 0i``."""
        from .jaxpr import _PRECISIONS
        p, emin, emax = _PRECISIONS[component]
        wide = _wide(component, 2 * emax - 3 * emin + 4 * p + 6)
        x, y = [self.converted(i, wide) for i in (a, b)]

        def op(kind: str, *terms: int) -> int:
            return self.node(kind, list(terms), shape, dtype=wide)

        norm = self.rooted(op("add", op("mul_elem", x, x), op("mul_elem", y, y)), shape, wide)
        zero = self.scalar(0, component, shape)
        origin = self.node("logical_and", [self.boolean("eq", [i, zero], shape) for i in (a, b)],
                           shape, dtype="bool")
        return tuple(self.node("where", [origin, i, self.converted(
            self.divided(w, norm, shape, wide), component)], shape, dtype=component)
            for i, w in ((a, x), (b, y)))

    def divided(self, x: int, y: int, shape: list[int], wide: dict[str, Any]) -> int:
        """``x / y`` at the format ``wide``, rounded once: TorchLean.Op.div, whose IR kind ``div``
        has forms only at the named formats of the op table."""
        fields = [{"shape": shape}, {"nat": op_table(self.checker)["codes"]["div"]["none"]},
                  {"element": "scalar"}]
        return self.registered("div", fields, [x, y], shape, wide, dtype=wide)

    def rooted(self, x: int, shape: list[int], wide: dict[str, Any]) -> int:
        """``sqrt(x)`` at the format ``wide``, rounded once (TorchLean.Op.sqrt)."""
        fields = [{"shape": shape}, {"nat": op_table(self.checker)["codes"]["sqrt"]["raw"]}]
        return self.registered("sqrt", fields, [x], shape, wide, dtype=wide)

    def selected(self, x: tuple[int, int], y: tuple[int, int], name: str, shape: list[int],
                 component: str) -> tuple[int, int]:
        """The parts of ``lax.max`` (``name`` ``max``) or ``lax.min`` of the pairs ``x`` and ``y``:
        ``x`` where the real parts compare so, or are equal and the imaginary parts do, by IEEE
        comparisons, and ``y`` elsewhere (``complex``)."""
        (a, b), (c, d) = x, y
        order = "gt" if name == "max" else "lt"
        first = self.node("where", [self.boolean("eq", [a, c], shape),
                                    self.boolean(order, [b, d], shape),
                                    self.boolean(order, [a, c], shape)], shape, dtype="bool")
        return tuple(self.node("where", [first, p, q], shape, dtype=component)
                     for p, q in ((a, c), (b, d)))

    def folded(self, a: int, b: int, axes: list[int], prefixes: bool, reverse: bool,
               step: Callable[[tuple[int, int], tuple[int, int], list[int]], tuple[int, int]],
               empty: tuple[float, float], component: str) -> tuple[int, int]:
        """The real and imaginary parts of the fold of ``step`` over ``axes`` of the entries of
        ``a + ib``, or with ``prefixes`` over each prefix along the one axis of ``axes``, each
        suffix with ``reverse``: the left fold from the first entry, the accumulator ``step``'s
        first operand, over the entries in row-major order of the reduced coordinates, last to
        first with ``reverse``, and ``empty`` over no entry. lax documents no order, and the fold
        is TorchLean's, as for the real folds (``_Builder.scan``); ``complex`` says how XLA's
        CPU code folds instead."""
        shape = self.shape(a)
        rank = len(shape)
        kept = [d for d in range(rank) if d not in axes]
        n = math.prod(shape[d] for d in axes)
        if prefixes:
            (axis,) = axes
            narrow = shape[:axis] + [1] + shape[axis + 1:]

            def entry(x: int, j: int) -> int:
                return self.slice(x, [j if d == axis else 0 for d in range(rank)],
                                  [j + 1 if d == axis else shape[d] for d in range(rank)],
                                  [1] * rank)
        else:
            narrow = [shape[d] for d in kept]
            a, b = [self.reshape(self.permute(x, kept + axes), narrow + [n]) for x in (a, b)]

            def entry(x: int, j: int) -> int:
                return self.reshape(self.slice(x, [0] * len(narrow) + [j], narrow + [j + 1],
                                               [1] * (len(narrow) + 1)), narrow)
        if 0 in narrow or n == 0:
            return (a, b) if prefixes else tuple(self.scalar(v, component, narrow) for v in empty)
        order = list(reversed(range(n))) if reverse else list(range(n))
        z = (entry(a, order[0]), entry(b, order[0]))
        steps = [z]
        for j in order[1:]:
            z = step(z, (entry(a, j), entry(b, j)), narrow)
            steps.append(z)
        if not prefixes:
            return z
        steps = steps[::-1] if reverse else steps
        return tuple(self.node("cat", [s[k] for s in steps], shape, dtype=component, dim=axis)
                     if n > 1 else steps[0][k] for k in (0, 1))

    def multiplied(self, parents: list[int], params: dict[str, Any], updates: Any,
                   shape: list[int], component: str) -> int:
        """``scatter-mul`` of complex values of shape ``shape``: each window of the updates, whose
        abstract value is ``updates``, multiplies its slice in turn, in the row-major order of the
        index vectors in which the IR applies a real scatter's updates (``_Builder.indexed``). A
        step gathers the slice at the scatter's dimension numbers, multiplies it by the window as
        ``mul`` does (``product``) and scatters the result back as a replacement in the scatter's
        mode, so a window that the scatter drops writes nothing. StableHLO applies the updates in
        an implementation-defined order (spec.md:6010), and JAX's documentation of ``.at``
        (jax/_src/numpy/array_methods.py) calls the order of conflicting updates
        "implementation-defined and may be nondeterministic", so this is one admissible order, the
        real import's. The node count
        grows with the number of index vectors. A scatter with batching dimensions, which
        ``jax.vmap`` writes, is refused."""
        import jax

        from .jaxpr import Unsupported, _plain_update
        numbers = params["dimension_numbers"]
        if not _plain_update("scatter-mul", params):
            raise Unsupported("scatter-mul: only a product of the updates is imported")
        if numbers.operand_batching_dims:
            raise Unsupported("scatter-mul: a complex scatter with batching dimensions is not "
                              "imported")
        operand, indices, values = parents
        extents = [int(d) for d in updates.shape]
        windows = [int(d) for d in numbers.update_window_dims]
        inserted = [int(d) for d in numbers.inserted_window_dims]
        lead = [d for d in range(len(extents)) if d not in windows]
        sizes = [1] * len(shape)
        for axis, along in zip([d for d in range(len(shape)) if d not in inserted], windows):
            sizes[axis] = extents[along]
        gathered = dict(dimension_numbers=jax.lax.GatherDimensionNumbers(
            offset_dims=tuple(windows), collapsed_slice_dims=tuple(inserted),
            start_index_map=tuple(numbers.scatter_dims_to_operand_dims)),
            slice_sizes=tuple(sizes), mode=params["mode"])
        replaced = dict(params, update_jaxpr=None, update_consts=())
        window = [1 if d in lead else extents[d] for d in range(len(extents))]
        source, dtype = self.shape(indices), np.dtype(component)
        target = [self.part(operand, k) for k in (0, 1)]
        factors = [self.part(values, k) for k in (0, 1)]
        for position in np.ndindex(*[extents[d] for d in lead]):
            # The index vector's batch axes are the updates' leading axes, in order.
            vector = self.slice(indices, [*position, 0], [q + 1 for q in position] + source[-1:],
                                [1] * len(source))
            at = dict(zip(lead, position))
            factor = [self.slice(x, [at.get(d, 0) for d in range(len(extents))],
                                 [at[d] + 1 if d in at else extents[d]
                                  for d in range(len(extents))], [1] * len(extents))
                      for x in factors]
            slices = [self.indexed("gather", [t, vector], gathered, window, dtype) for t in target]
            parts = self.product(*slices, *factor, window, component)
            target = [self.indexed("scatter", [t, vector, q], replaced, shape, dtype)
                      for t, q in zip(target, parts)]
        return self.pair(*target)

    def complex(self, equation: Any, parents: list[int]) -> int:
        """A primitive with a complex operand or result, on the paired-real form (``paired``).

        ``fft`` transforms the trailing axes that its lengths name. Along the last one, type RFFT
        is ``Op.rfft`` and type IRFFT ``Op.irfft``, the transforms PyTorch's ``rfft`` and
        ``irfft`` import as. Types FFT and IFFT, and RFFT and IRFFT along the other axes, are the
        complex transforms of ``fourier``, built from ``Op.rfft`` of each part; an IRFFT
        transforms those axes first and its last axis after, as ``numpy.fft.irfftn``, which
        ``jnp.fft.irfftn`` follows, does. These are TorchLean's sums and not XLA's FFT schedule,
        so their rounding differs. ``real`` and ``imag`` read a part, ``complex`` pairs two,
        ``conj`` negates the imaginary part, and ``add``, ``sub`` and ``neg`` act on each part,
        which is complex addition, subtraction and negation exactly; so are ``reduce_sum`` and
        ``cumsum``, a sum of each part in the IR's order, as XLA's CPU code sums each part.
        ``mul`` and ``div`` are the definitions, ``product`` and ``quotient``: each part is the
        exact part rounded once. ``lax.mul`` with an ``out_dtype`` converts its operands to that
        type first, as JAX 0.11.1 lowers it (``operand``), and a real operand is the real part,
        with imaginary part ``+0``. XLA's CPU code computes neither so, and its last places differ
        (``TorchLean/NN/IR/README.md``). ``dot_general`` is the product extended by linearity: four
        real contractions of the parts, ``re = a c - b d`` and
        ``im = b c + a d``, each summed in the IR's order as a real ``dot_general`` is, so a
        contraction of one term rounds ``a c`` and ``b d`` apart where ``mul`` rounds once. A
        real operand is converted with imaginary part ``+0``, so an infinite ``d`` makes ``re``
        NaN through ``0 inf``. XLA's CPU code (Eigen's complex GEBP) also keeps the four
        contractions apart: it sums two matrices in that order up to 7 terms, and a vector
        operand so in binary64 but not in binary32, groups longer sums by the output's place in
        its packing, and contracts one term as its ``mul``, so the last places can differ there.
        ``conv_general_dilated`` is four real convolutions of the parts in the same way, each
        the IR's (``convolution``), with the parts of a kernel known at capture (``fixed``) as
        their payloads. XLA's CPU code sums in another order, so its words are these where every
        sum is exact and differ in most last places elsewhere.
        ``eq`` holds where both parts are equal and ``ne`` where either differs, as StableHLO
        compares complex values. ``abs`` is the IR's ``hypot`` of the parts:
        JAX documents the modulus ``sqrt(a^2 + b^2)``, and ``hypot`` is that exact value rounded
        once, ``+inf`` when a part is infinite even if the other is a quiet NaN, as IEEE 754-2019
        9.2.1 specifies it, and a quiet NaN when a part is a signaling NaN. XLA's CPU code returned
        NaN there and was up to two units in the last place away at the random pairs sampled.
        ``sign`` is the phase ``z / |z|`` (``phase``), each part the exact part rounded once, and
        ``z`` itself at zero. XLA's CPU code returned ``z`` where both parts are subnormal and
        ``1 + 0i`` at ``2^-126 + 2^-127 i``, the results of zero parts (how it loses them is our
        inference, not read from its code), and ``0`` at the greatest finite pair.
        ``exp``, ``expm1``, ``log``, ``log1p``, ``sqrt``, ``rsqrt``, ``sin``, ``cos``, ``tan``,
        ``sinh``, ``cosh``, ``tanh``, ``asin``, ``acos``, ``atan``, ``asinh``, ``acosh`` and
        ``atanh`` are ``Op.complex`` of the function of that name (``_FUNCTIONS``), FloatLib's
        ``ExecComplex`` function, whose parts at a finite argument are the parts of the exact
        result rounded once (``TorchLean.Tensor.ConfiguredBinary.Complex``), with the special
        values and branch cuts of C23 7.3 and Annex G, the sign of a zero part selecting the side
        of a cut. ``rsqrt``, which C does not have, is ``1 / √z``, as JAX documents ``lax.rsqrt``
        (``jax/_src/lax/lax.py`` in 0.11.1); its values at ``0`` and ``inf``, which JAX does not
        name, are C23's ``(1 + 0i) / (0 + 0i)``, ``+inf + NaN i``, and the limit
        ``+0 - i copysign(0, b)``. XLA's CPU code gave ``0 + 2i`` for ``sqrt(-4 - 0i)`` and the
        root of the conjugate at a negative subnormal ``b``, the results of a ``+0`` part (again
        our inference), ``0`` for a subnormal part of the root, ``NaN + NaN i`` where C's results
        are infinite beside a NaN, and a unit or two in the last place away at about a quarter of
        the parts of 500 random pairs. For ``rsqrt`` it gave ``inf + NaN i`` and ``0 - 0i`` at
        ``0`` and ``inf``, a nonzero real part at ``-4 ∓ 0i``, where ``1 / √z`` is ``±0.5i``, and
        a part a unit in the last place from the correctly rounded one at ``3 + 4i``.
        ``pow`` is ``Op.complexPow``, FloatLib's ``ExecComplex.pow``, C23's ``cpow``
        ``exp (w log z)``: at a finite base other than zero and a finite exponent, each part of a
        result it certifies is the exact part rounded once, the sign of the base's zero imaginary
        part selecting the side of the cut of ``log``. ``z^0`` is ``1 ± 0i``, at a NaN too, with
        the sign of the base's imaginary part, and a zero base gives ``+0 + 0i``, ``inf + NaN i``
        or ``NaN + NaN i`` as the real part of the exponent is positive, negative or zero. An
        integer exponent is converted to the complex type, its value rounded once and its
        imaginary part ``+0``, as JAX documents for ``lax.pow`` and lowers it (StableHLO's
        ``convert`` in ``_pow_lower``, ``jax/_src/lax/lax.py`` in 0.11.1). XLA's CPU code gave
        ``NaN + NaN i`` at a zero base, ``inf + 0i`` for ``inf^1``, a nonzero real part for
        ``(-1 ± 0i)^0.5`` and ``(1 + i)^2``, and parts 4 and 2 units in the last place from the
        correctly rounded ones for ``(1 + 2i)^(3 - i)`` in binary32. ``exp2`` is ``2^z``, as JAX
        documents ``lax.exp2``, the power of the base ``2 + 0i``; JAX lowers it as ``exp`` of
        ``z`` times ``log 2`` rounded (``_exp2_lower``).
        ``integer_pow`` multiplies in the order of JAX's ``lax._integer_pow`` (``integer_pow``),
        each product ``product`` and the reciprocal of a negative power ``quotient``, so that each
        part of ``z^2`` and ``z^-1`` is the exact part rounded once and the other powers round at
        each step; ``z^0`` is ``1 + 0i``, at a NaN too, as ``lax._integer_pow`` gives. ``square``
        is ``z^2``. ``max`` and ``min`` (``selected``) compare the pairs lexicographically, which is
        all that JAX documents for complex operands (the docstrings of ``lax.max`` and ``lax.min``,
        jax/_src/lax/lax.py in 0.11.1): of pairs without a NaN part, ``x`` where the real parts
        compare so, or are equal and the imaginary parts do, and ``y`` elsewhere. Where a NaN part
        decides the comparison JAX documents nothing, and IEEE 754-2019 has no complex maximum, so
        there the import follows, as our convention, JAX 0.11.1's lowering (``_minmax_hlo`` in
        ``jax/_src/interpreters/mlir.py``), whose comparisons (IEEE 754-2019 5.11) are false at a
        NaN: ``y`` where the real parts are unordered, or equal beside unordered imaginary parts. So
        ``max(NaN + 0i, 1 + 0i)`` is ``1 + 0i`` and ``max(1 + 0i, NaN + 0i)`` is ``NaN + 0i``, while
        ``max(2 + NaN i, 1 + 0i)`` is ``2 + NaN i``, which the real parts decide.
        ``reduce_prod``, ``reduce_max`` and ``reduce_min``, and ``cumprod``, ``cummax`` and
        ``cummin`` over each prefix or suffix, fold ``product`` or ``selected`` from the first
        entry (``folded``), as JAX documents the product, maximum or minimum of the elements; over
        no entry they are ``1 + 0i``, ``-inf + 0i`` and ``inf + 0i``, the initial values of the
        ``reduce`` and ``reduce_window`` JAX 0.11.1 lowers them to. Those values are not
        identities here: ``(1 + 0i)(inf + 0i)`` is ``inf + NaN i``, ``max(-inf + 0i, -inf - i)``
        is ``-inf + 0i``, and ``selected`` is not associative at a NaN, where the docstring of
        ``lax.reduce`` (jax/_src/lax/lax.py) leaves XLA's result undefined. XLA's CPU words
        were those of the left fold from the initial value, the accumulator first, along 2 to 32
        entries of a reduction and 1 to 8 of a cumulative one, a reverse one folding each suffix
        first to last; it returns a lone entry of a reduction, and longer extrema follow neither
        fold. ``reduce_window_sum``, ``reduce_window_max`` and ``reduce_window_min``, and
        ``reduce_window`` with one add, mul, max or min as its body, fold each window from the
        init value in row-major order of its cells, as the real window reductions do
        (``windows``), by each part's sum, ``product`` or ``selected``, on the operand padded
        with the init value's parts. The init values of the first three are those JAX 0.11.1
        lowers them with (``_reduce_window_lower`` in ``jax/_src/lax/windowed_reductions.py``):
        ``0``, and ``_get_max_identity``'s ``-inf + 0i`` and ``_get_min_identity``'s
        ``inf + 0i``. ``sort`` reaches ``OrderImport.sort``, for real and complex operands
        alike: it orders a complex key by its real and then its imaginary part, each taken as
        JAX's comparator takes a real key, ``-0`` as ``+0`` and a NaN as one that follows
        ``+inf``, equal entries in their order, and moves a complex operand part by part. XLA's
        CPU words were the same at parts drawn from a few values, NaN, signed zeros and
        infinities among them.
        ``convert_element_type`` to a complex type converts the parts, and a real value becomes the
        real part, with imaginary part ``+0``. ``reshape``, ``squeeze``, ``transpose``,
        ``broadcast_in_dim``, ``slice``, ``rev``, ``concatenate``, ``stack``, ``tile`` and
        ``copy`` move the pairs, as ``split`` and ``unstack`` do (``_Builder.equation``), and
        ``pad`` pads each part with that part of the padding value.
        ``select_n``, ``gather``, ``dynamic_slice``, ``dynamic_update_slice`` and ``scatter`` move
        each part at the same real predicate or indices, ``scatter-add`` and ``scatter-sub`` add
        or subtract each part of the updates, and ``scatter-mul`` multiplies by the updates in turn
        (``multiplied``). Every other primitive with a complex operand or result is refused, so a
        complex value never reaches anything but these."""
        from .jaxpr import Unsupported, _WINDOWED, _component, _dtype, _shape
        name, params = equation.primitive.name, equation.params
        if len(equation.outvars) != 1:
            raise Unsupported(f"{name}: multiple primitive outputs are not supported")
        aval = equation.outvars[0].aval
        shape, component = _shape(aval), _component(aval)
        self.current_dtype = component or _dtype(aval)
        paired = shape + [2]
        if name == "fft":
            # FftType: FFT 0, IFFT 1, RFFT 2, IRFFT 3. The signal is the real operand of an RFFT
            # and the real result of an IRFFT.
            kind, lengths = int(params["fft_type"]), [int(n) for n in params["fft_lengths"]]
            if self.current_dtype not in ("float32", "float64"):
                raise Unsupported("fft: only binary32 and binary64 operands are imported")
            signal = self.shape(parents[0]) if kind == 2 else shape
            rank, count = len(signal), len(lengths)
            axes = list(range(rank - count, rank))
            if not 0 < count <= rank or signal[rank - count:] != lengths or 0 in lengths or \
                    kind == 3 and self.shape(parents[0]) != shape[:-1] + [lengths[-1] // 2 + 1, 2]:
                raise Unsupported(f"fft: lengths {lengths} do not name nonempty trailing axes of "
                                  f"{signal}")
            index = parents[0]
            if kind == 3:
                for axis in axes[:-1]:
                    index = self.fourier(index, axis, inverse=True)
                return self.registered("irfft", [{"shape": shape}, {"nat": rank - 1},
                                                 {"nat": lengths[-1]}],
                                       [self.hermitian(index, lengths[-1])], shape,
                                       self.current_dtype)
            if kind == 2:
                half = signal[:-1] + [lengths[-1] // 2 + 1, 2]
                index = self.registered("rfft", [{"shape": signal}, {"nat": rank - 1},
                                                 {"nat": lengths[-1]}], [index], half, component)
                self.paired.add(index)
                axes = axes[:-1]
            for axis in axes:
                index = self.fourier(index, axis, inverse=kind == 1)
        elif name in ("real", "imag"):
            return self.part(parents[0], 0 if name == "real" else 1)
        elif name == "complex":
            return self.pair(*[self.broadcast(i, shape) for i in parents])
        elif name == "abs":
            return self.node("hypot", [self.part(parents[0], k) for k in (0, 1)], shape)
        elif name == "sign":
            return self.pair(*self.phase(*[self.part(parents[0], k) for k in (0, 1)], shape,
                                         component))
        elif name in _FUNCTIONS:
            index = self.registered("complex", [{"name": name}, {"shape": shape}], parents, paired,
                                    component)
        elif name in ("pow", "exp2"):
            # The base is complex; a scalar or an integer exponent is broadcast and converted.
            if name == "exp2":
                parents = [self.pair(*[self.scalar(v, component, shape) for v in (2, 0)]), *parents]
            operands = [i if i in self.paired and self.shape(i) == paired else
                        self.pair(*self.operand(i, shape, component)) for i in parents]
            index = self.registered("complex_pow", [{"shape": shape}], operands, paired,
                                    component)
        elif name in ("max", "min"):
            x, y = [self.operand(i, shape, component) for i in parents]
            return self.pair(*self.selected(x, y, name, shape, component))
        elif name in ("square", "integer_pow"):
            return self.pair(*self.integer_pow(
                tuple(self.part(parents[0], k) for k in (0, 1)),
                2 if name == "square" else int(params["y"]),
                lambda: (self.scalar(1, component, shape), self.scalar(0, component, shape)),
                lambda z, w: self.product(*z, *w, shape, component),
                lambda z, w: self.quotient(*z, *w, shape, component)))
        elif name == "conj" and parents[0] in self.paired:
            index = parents[0]
            return self.pair(self.part(index, 0), self.node("neg", [self.part(index, 1)], shape))
        elif name in ("add", "add_any", "sub", "neg"):
            index = self.operation(name, parents, params, paired, aval.dtype)
        elif name in ("mul", "div"):
            if not component:
                raise Unsupported(f"{name}: a complex operand of a real result is not imported")
            (a, b), (c, d) = [self.operand(i, shape, component) for i in parents]
            method = self.product if name == "mul" else self.quotient
            return self.pair(*method(a, b, c, d, shape, component))
        elif name == "dot_general":
            if any(self.nodes[i]["dtype"] != component for i in parents):
                raise Unsupported("dot_general: a complex contraction of operands of other "
                                  "component types than its result needs an operand/accumulator "
                                  "type rule")
            (a, b), (c, d) = [[self.part(i, k) for k in (0, 1)] if i in self.paired
                              else [i, self.scalar(0, component, self.shape(i))] for i in parents]
            ac, bd, bc, ad = [self.dot(x, y, params, shape)
                              for x, y in ((a, c), (b, d), (b, c), (a, d))]
            return self.pair(self.node("sub", [ac, bd], shape), self.node("add", [bc, ad], shape))
        elif name == "conv_general_dilated":
            # JAX convolves operands of one type only, so both are complex here. The parts of a
            # kernel known at capture are constants, the payloads of the convolutions.
            value = self.fixed(parents[1])
            a, b = [self.part(parents[0], k) for k in (0, 1)]
            if value is None:
                c, d = [self.part(parents[1], k) for k in (0, 1)]
            else:
                part = equation.invars[1].aval.update(dtype=np.dtype(component))
                c, d = [self.constant(np.asarray(p), part) for p in (value.real, value.imag)]
            ac, bd, bc, ad = [self.convolution([x, y], params, shape)
                              for x, y in ((a, c), (b, d), (b, c), (a, d))]
            return self.pair(self.node("sub", [ac, bd], shape), self.node("add", [bc, ad], shape))
        elif name == "convert_element_type" and component:
            if parents[0] not in self.paired:
                real = self.operation(name, parents, params, shape, aval.dtype)
                return self.pair(real, self.scalar(0, component, shape))
            index = self.operation(name, parents, params, paired, aval.dtype)
        elif name in ("reshape", "squeeze"):
            # A reshape's dimensions, unlike a squeeze's, permute the operand's axes first; the
            # pair axis is the paired operand's last, after them.
            index = parents[0]
            if name == "reshape" and params.get("dimensions") is not None:
                dimensions = [int(d) for d in params["dimensions"]]
                index = self.permute(index, [*dimensions, len(dimensions)])
            index = self.reshape(index, paired)
        elif name == "transpose":
            index = self.permute(parents[0], [*params["permutation"], len(shape)])
        elif name == "tile":
            # The pair axis is repeated once.
            index = self.operation(name, parents, dict(params, reps=[*params["reps"], 1]), paired,
                                   aval.dtype)
        elif name == "broadcast_in_dim":
            dims = [*params["broadcast_dimensions"], len(shape)]
            index = self.broadcast(parents[0], paired, dims)
        elif name == "slice":
            steps = [*(params["strides"] or [1] * len(shape)), 1]
            index = self.slice(parents[0], [*params["start_indices"], 0],
                               [*params["limit_indices"], 2], steps)
        elif name in ("rev", "concatenate", "stack", "reduce_sum", "cumsum", "copy"):
            # The pair axis is last and none of these reaches it: a sum or prefix sum along the
            # operand's axes is that of each part, and ``stack`` puts its new axis before it.
            index = self.operation(name, parents, params, paired, aval.dtype)
        elif name == "pad":
            return self.pair(*[self.operation(name, [self.part(i, k) for i in parents], params,
                                              shape, aval.dtype) for k in (0, 1)])
        elif name in ("reduce_prod", "cumprod", "reduce_max", "reduce_min", "cummax", "cummin"):
            axes = (sorted(int(d) for d in params["axes"]) if name.startswith("reduce")
                    else [int(params["axis"])])
            kind = name.removeprefix("reduce_").removeprefix("cum")

            def step(z: tuple[int, int], w: tuple[int, int], s: list[int]) -> tuple[int, int]:
                return (self.product(*z, *w, s, component) if kind == "prod"
                        else self.selected(z, w, kind, s, component))
            # Over no entry, the initial value of JAX's reduce (_get_prod_identity,
            # _get_max_identity and _get_min_identity in jax/_src/lax/lax.py).
            empty = {"prod": (1.0, 0.0), "max": (-math.inf, 0.0), "min": (math.inf, 0.0)}[kind]
            return self.pair(*self.folded(*[self.part(parents[0], k) for k in (0, 1)], axes,
                                          name.startswith("cum"), bool(params.get("reverse")),
                                          step, empty, component))
        elif name in _WINDOWED or name == "reduce_window":
            kind, swapped = (_WINDOWED[name], False) if name in _WINDOWED else \
                self.monoid(name, params)
            if len(parents) != (1 if name in _WINDOWED else 2):
                raise Unsupported(f"{name}: only one operand is imported")
            init = ([self.part(parents[1], k) for k in (0, 1)] if name == "reduce_window" else
                    [self.scalar(v, component, []) for v in (
                        {"add": 0.0, "max_elem": -math.inf, "min_elem": math.inf}[kind], 0.0)])

            def combined(z: tuple[int, int], w: tuple[int, int]) -> tuple[int, int]:
                z, w = (w, z) if swapped else (z, w)
                if kind == "add":
                    return tuple(self.node("add", [p, q], shape, dtype=component)
                                 for p, q in zip(z, w))
                if kind == "mul_elem":
                    return self.product(*z, *w, shape, component)
                return self.selected(z, w, kind.removesuffix("_elem"), shape, component)

            z = tuple(self.broadcast(i, shape, []) for i in init)
            if 0 not in shape:
                for cells in zip(*[self.cells(self.dilated(self.part(parents[0], k), init[k],
                                                           params), params, shape)
                                   for k in (0, 1)]):
                    z = combined(z, cells)
            return self.pair(*z)
        elif name in ("eq", "ne"):
            # Equal when both parts are, unequal when either is.
            parts = [self.boolean(name, [self.part(i, k) for i in parents], shape) for k in (0, 1)]
            return self.node("logical_and" if name == "eq" else "logical_or", parts, shape,
                             dtype="bool")
        elif name == "select_n":
            return self.pair(*[self.boolean(name, [parents[0], *[self.part(i, k) for i in
                                                                 parents[1:]]], shape)
                               for k in (0, 1)])
        elif name == "scatter-mul":
            return self.multiplied(parents, params, equation.invars[2].aval, shape, component)
        elif name in ("gather", "dynamic_slice", "dynamic_update_slice", "scatter",
                      "scatter-add", "scatter-sub"):
            # The indices are real; a FILL_OR_DROP gather fills each part with that part of its
            # fill value, which by default is NaN converted to the complex type, NaN + 0i.
            fill = np.asarray(np.nan if params.get("fill_value") is None else
                              params["fill_value"]).astype(aval.dtype)
            return self.pair(*[self.indexed(name, [self.part(i, k) if i in self.paired else i
                                                   for i in parents],
                                            dict(params, fill_value=(fill.real, fill.imag)[k]),
                                            shape, np.dtype(component)) for k in (0, 1)])
        else:
            raise Unsupported(f"{name}: complex values are imported only through fft, real, imag, "
                              "complex, conj, abs, sign, " + ", ".join(_FUNCTIONS) +
                              ", pow, exp2, eq, ne, max, min, add, sub, neg, mul, div, square, "
                              "integer_pow, dot_general, conv_general_dilated, reduce_sum, "
                              "reduce_prod, reduce_max, reduce_min, cumsum, cumprod, cummax, "
                              "cummin, the window reductions, a stable sort, select_n, "
                              "convert_element_type, data movement and the gathers and scatters "
                              "that replace, add, subtract or multiply")
        self.paired.add(index)
        return index
