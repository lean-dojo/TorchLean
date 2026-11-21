"""The importer's orderings: ``lax.sort``, ``lax.top_k``, ``lax.approx_top_k``, integer
``argmax`` and ``argmin``, and the total-order comparisons ``eq_to``, ``lt_to`` and ``le_to``.

``OrderImport`` is the part of ``torchlean.jaxpr``'s builder that imports a primitive ordering the
entries of each fiber along an axis, with the IR's stable ``sort`` and ``topk`` (``Op.sort`` and
``Op.topk``, ``TorchLean/Tensor/Sort.lean``) and registered gathers. The importer's errors and
helpers are read from ``torchlean.jaxpr`` when a primitive is imported, since that module imports
this one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from . import _artifact
from ._table import op_table

if TYPE_CHECKING:
    from .jaxpr import DType


def _float8(dtype: Any) -> bool:
    return isinstance(dtype, dict) and dtype.get("kind") == "binary"


class OrderImport:
    """The ordering primitives of ``torchlean.jaxpr``'s builder."""

    def sort(self, parents: list[int], params: dict[str, Any]) -> list[int]:
        """``lax.sort`` along ``dimension``: the fibers of every operand permuted alike, into the
        lexicographic order of the first ``num_keys`` operands, the first key primary, as the
        docstring of ``lax.sort`` in JAX 0.11.1 states, and with ``is_stable`` entries whose keys
        compare equal keep their order. Its comparator (``_sort_lt_comparator`` and
        ``_operands_to_keys`` in ``jax/_src/lax/lax.py``) compares a floating key canonicalized,
        ``-0`` as ``+0`` and every NaN as one NaN after ``+inf``
        (``_canonicalize_float_for_sort``). That is the order of the IR's stable ``sort``, which
        places a NaN after every other entry and neither of ``-0`` and ``+0``, or of two NaNs,
        before the other (``Tensor.sort``). A complex key is two such keys, its real and then its
        imaginary part, which is the order the docstring states for complex operands,
        lexicographic over the parts with the real part primary. NumPy's order differs:
        ``numpy.sort`` documents ``[R + Rj, R + nanj, nan + Rj, nan + nanj]``, every pair with a
        NaN part after the pairs without one, so ``1 + 0i`` before ``0 + NaN i``, which this order
        reverses. An integer or Boolean key is compared as it is.

        One real operand is the IR's ``sort``. Otherwise a stable sort by each key in turn, from
        the last to the first, each ordering the key as the passes before it left it, gives the
        lexicographic order: two entries end in the order of the first key that tells them apart,
        and in their input order where none does. ``Op.sort``'s second output, the
        position each sorted entry came from, composes the passes into one permutation of each
        fiber, at which every operand is read (``permuted``), a complex one part by part, its words
        unchanged, so a payload of any type, such as the iota of ``jnp.argsort`` or of ``sort``'s
        JVP, comes out as ``lax.sort`` gives it.

        A float8 key is ordered in binary32, which holds each of its values and keeps their order,
        zero signs and NaNs, and the one operand's sorted values are converted back, exactly; with
        more operands a float8 one is gathered word for word as any other. XLA's CPU code sorts
        the NaN of a fnuz format first, and the five values 2^-15 to 2^-14 of each sign of
        float8_e5m2fnuz as equal. An integer or Boolean key is sorted at its own words, by their
        values, false below true (``Op.sort`` at ``Op.Element``). An unstable sort is refused:
        ``is_stable=False`` leaves the order of equal keys open."""
        from .jaxpr import Unsupported, _INT64
        if not params["is_stable"]:
            raise Unsupported("sort: only a stable sort is imported")
        axis, dtype = int(params["dimension"]), self.nodes[parents[0]]["dtype"]
        # A complex operand is its real and imaginary parts.
        parts = [[self.part(i, k) for k in (0, 1)] if i in self.paired else [i] for i in parents]
        keys = [self.converted(i, "float32") if _float8(self.nodes[i]["dtype"]) else i
                for key in parts[:int(params["num_keys"])] for i in key]
        shape = self.shape(keys[0])
        if len(parents) == len(keys) == 1:
            values = self.node("sort", keys, shape, dtype=self.nodes[keys[0]]["dtype"], dim=axis,
                               descending=False)
            return [self.converted(values, dtype)]
        if 0 in shape:
            return list(parents)
        # Stable passes from the last key to the first; `order` maps each sorted position to its
        # source position so far.
        order, table = None, op_table(self.checker)
        for key in reversed(keys):
            arranged = key if order is None else self.permuted(key, order, axis)
            scalar = self.nodes[arranged]["dtype"]
            element = _artifact._element(scalar, table)
            step = self.registered("sort", [{"shape": shape}, {"nat": axis}, {"bool": False},
                                            {"element": element}], [arranged], shape,
                                   scalar if element == "scalar" else "float64", output=1,
                                   dtype=_INT64)
            order = step if order is None else self.permuted(order, step, axis)
        moved = [[self.permuted(i, order, axis) for i in operand] for operand in parts]
        return [self.pair(*operand) if len(operand) == 2 else operand[0] for operand in moved]

    def held(self, x: int) -> int | None:
        """``x`` in a binary format that holds each of its values, exactly and in their order: a
        floating ``x`` itself, a float8 one in binary32, with its zero signs and NaNs, and a
        Boolean or integer one of at most 32 bits in binary64. None for a wider integer, which
        binary64 does not hold."""
        from .jaxpr import _FLOATING, _is_integer
        dtype = self.nodes[x]["dtype"]
        if dtype in _FLOATING:
            return x
        if _float8(dtype):
            return self.converted(x, "float32")
        if dtype == "bool" or _is_integer(dtype) and dtype["width"] <= 32:
            return self.converted(x, "float64")
        return None

    def total(self, name: str, parents: list[int], shape: list[int]) -> int:
        """``eq_to``, ``lt_to`` or ``le_to`` (``name``) of ``parents`` broadcast to ``shape``:
        StableHLO's ``compare`` with ``compare_type`` TOTALORDER for floating operands, and SIGNED
        or UNSIGNED, the plain order, for integer and Boolean ones, as JAX 0.11.1 lowers them
        (``_compare_lower_hlo`` in jax/_src/lax/lax.py). JAX's sort and search comparators apply
        them to keys whose ``-0`` is made ``+0`` and whose NaNs are made one NaN
        (``_sort_lt_comparator``), but the primitives take any word. StableHLO's spec ("compare")
        says only that TOTALORDER "uses the combination of totalOrder and compareQuietEqual
        operations from IEEE-754". The import takes the relation totalOrder of IEEE 754-2019
        (section 5.10): ``le_to`` is totalOrder(x, y), ``lt_to`` is that without
        totalOrder(y, x), and ``eq_to`` is both, the same word. totalOrder puts ``-0`` below
        ``+0``, a NaN below every number if its sign is negative and above if positive, and among
        NaNs of one sign a signaling NaN nearer the numbers than a quiet one. It leaves the rest
        of the order among NaNs to the implementation. In a format of sign, exponent and
        fraction fields (binary16 to binary64, bfloat16, float8_e5m2 and float8_e4m3fn) the
        import takes the order of signed integers: the words read as integers of their width,
        with the bits other than the sign flipped where the sign is set, which orders payloads by
        their bits. XLA's CPU code agrees at every pair of words tested. A fnuz format has no
        ``-0``, and its one NaN is the word that would be ``-0``, which IEEE 754 does not order.
        That NaN is placed above every number, as the IR's ``sort`` places it and as JAX's
        comparator states its order ("-infinity < ... < 0 < ... < infinity < NaN"); the numbers
        are compared in binary32. XLA's CPU code places it below every number, as its ``sort``
        puts it first."""
        from .jaxpr import Unsupported, _FLOATING, _is_integer
        dtype = self.nodes[parents[0]]["dtype"]
        kind = {"eq_to": "eq", "lt_to": "lt", "le_to": "le"}[name]
        if any(self.nodes[i]["dtype"] != dtype for i in parents):
            raise Unsupported(f"{name}: operands of two element types are not imported")
        if dtype == "bool" or _is_integer(dtype):
            return self.boolean(kind, parents, shape)
        if dtype not in _FLOATING and not _float8(dtype):
            raise Unsupported(f"{name}: only floating, integer and Boolean operands are imported")
        operands = [self.broadcast(i, shape) for i in parents]

        def flag(kind: str, flags: list[int]) -> int:
            return self.node(kind, flags, shape, dtype="bool")

        if _float8(dtype) and dtype["encoding"] == "finiteUnsignedZero":
            x, y = [self.converted(i, "float32") for i in operands]
            nan_x, nan_y = [flag("ne", [v, v]) for v in (x, y)]
            if name == "eq_to":
                return flag("logical_or", [flag("eq", [x, y]), flag("logical_and", [nan_x, nan_y])])
            if name == "lt_to":
                nan_y = flag("logical_and", [flag("logical_not", [nan_x]), nan_y])
            return flag("logical_or", [flag(kind, [x, y]), nan_y])
        width = 8 if _float8(dtype) else 8 * np.dtype(jax.numpy.dtype(dtype)).itemsize
        signed, word = dict(kind="integer", signed=True, width=width), np.dtype(f"int{width}")

        def key(x: int) -> int:
            w = self.node("custom", [x], shape, dtype=signed, primitive="bitcast")
            sign = self.node("shift_right_arithmetic", [w, self.scalar(width - 1, word, shape)],
                             shape, dtype=signed)
            low = self.node("bitwise_and", [sign, self.scalar(2 ** (width - 1) - 1, word, shape)],
                            shape, dtype=signed)
            return self.node("bitwise_xor", [w, low], shape, dtype=signed)

        return flag(kind, [key(i) for i in operands])

    def first(self, name: str, x: int, axis: int) -> int:
        """``argmax`` or ``argmin`` (``name``) of the integer ``x`` along its nonempty ``axis``,
        kept at length 1, in the index type: the least position whose entry equals the greatest
        entry, or the least. Both reductions are ``wrapping``'s trees in their types, so the
        result is exact at every width and is the first extremum, as XLA's is."""
        from .jaxpr import _INT64
        source = self.shape(x)
        n, every = source[axis], list(range(len(source)))
        extreme = self.wrapping("reduce_max" if name == "argmax" else "reduce_min", x, [axis])
        hit = self.node("eq", [x, self.broadcast(extreme, source, every)], source, dtype="bool")
        positions = self.broadcast(self.int64_constant(np.arange(n), [n]), source, [axis])
        past = self.broadcast(self.int64_constant(n, []), source, [])
        found = self.wrapping("reduce_min", self.node("where", [hit, positions, past], source,
                                                      dtype=_INT64), [axis])
        if self.current_dtype == _INT64:
            return found
        # An int32 index, exact since the axis is at most 2^31 long.
        return self.node("custom", [found], self.shape(found), primitive="cast",
                         rounding="towardZero")

    def permuted(self, x: int, order: int, axis: int) -> int:
        """``x`` with each fiber along ``axis`` read at the int64 positions ``order``, of ``x``'s
        shape: a registered gather of ``x``'s words, booked at ``x``'s floating format, float8
        included."""
        from .jaxpr import Unsupported
        source, dtype = self.shape(x), self.nodes[x]["dtype"]
        element = _artifact._element(dtype, op_table(self.checker))
        if element is None:
            raise Unsupported(f"sort: an operand of element type {dtype} is not imported")
        terms = [[order if d == axis else self.iota(source, d)] for d in range(len(source))]
        fields = [{"shape": source}, {"shape": source}, {"nats": list(range(len(source)))},
                  {"pairs": []}, {"element": element}]
        return self.registered("gather", fields, [x, self.coordinates(source, terms)], source,
                               dtype if element == "scalar" else "float64", dtype=dtype)

    def top_k(self, name: str, operand: int, params: dict[str, Any], shapes: list[list[int]],
              index: DType) -> list[int]:
        """``lax.top_k`` and ``approx_top_k``: the IR's stable ``topk`` for the values and
        ``Op.topk``'s second output for the indices, in the primitive's index dtype.

        Equal entries keep their input order, which is the order ``lax.top_k`` documents for
        ``is_stable`` (``is_stable=False`` leaves it open). XLA's CPU code ranks in IEEE
        totalOrder, ``+0`` above ``-0`` and a NaN by its sign and payload, and ``lax.top_k``
        documents nothing for NaN, so for ``k > 0`` execution fails where a fiber holds a NaN, or
        both a ``-0`` and a ``+0``: elsewhere the stable order is totalOrder's.

        ``approx_top_k`` (``lax.approx_max_k`` and ``lax.approx_min_k``) documents an approximate
        result at ``recall_target``, which "does not guarantee stability", and whose subset of
        indices is "implementation-defined" where a tie crosses the boundary. The import is one
        admissible refinement of that specification, the exact top or bottom of the fiber, which
        meets every recall target, so ``recall_target`` and ``reduction_input_size_override`` are
        not read. It is also what JAX's lowering off TPU asks for: XLA's ``ApproxTopK`` with
        ``is_fallback``, which XLA computes as a stable sort of the fiber with its indices under
        ``>`` (``is_max_k``) or ``<``, cut to the result's extent along ``reduction_dimension``:
        ``k`` with ``aggregate_to_topk``, else the extent JAX's shape rule gives. That is the exact
        top or bottom of the fiber, equal entries (``-0`` and ``+0`` among them) in input order,
        which ``topk`` with ``largest = is_max_k`` computes. Under ``>`` and ``<`` a NaN is
        equivalent to every entry, so its place is the sorting algorithm's, and execution fails
        where a fiber holds a NaN. XLA's CPU code rewrites the sort: into a reduction keeping a
        later one of equal extremes where the result keeps one entry, and into its totalOrder
        ``TopK`` for the largest binary32 entries where it keeps more but not all, so its indices of
        ties, and its order of ``-0`` and ``+0``, can differ there.

        A float8 operand of ``top_k`` or ``approx_top_k`` is ranked in binary32, which holds each
        of its values and keeps their order, zero signs and NaNs (``binary32``), and the values are
        converted back, exactly. In ``top_k`` XLA's CPU code ranks the five values 2^-15 to 2^-14
        of each sign of float8_e5m2fnuz as equal, in their input order, and in ``approx_top_k`` in
        their order.

        Integer and Boolean entries, false below true, are the first ``k`` of ``sort``'s stable
        ascending order of each fiber, taken for the largest of the entries reversed by bitwise
        or logical not, which maps each type onto itself in the opposite order. Equal entries
        keep their input order, at every width."""
        from .jaxpr import Unsupported, _FLOATING, _INT64, _is_integer
        source, scalar = self.shape(operand), self.nodes[operand]["dtype"]
        if isinstance(scalar, dict) and scalar.get("kind") == "binary":
            values, indices = self.top_k(name, self.converted(operand, "float32"), params, shapes,
                                         index)
            return [self.converted(values, scalar), indices]
        exact = scalar == "bool" or _is_integer(scalar)
        if scalar not in _FLOATING and not exact:
            raise Unsupported(f"{name}: only floating, integer and Boolean operands are imported")
        if not source:
            raise Unsupported(f"{name}: a 0-d operand has no axis")
        approximate = name == "approx_top_k"
        axis = int(params["reduction_dimension"] if approximate else params.get("axis", -1))
        axis = axis + len(source) if axis < 0 else axis
        k, largest = shapes[0][axis], bool(params.get("is_max_k", True))
        if index not in (dict(kind="integer", signed=True, width=32), _INT64):
            raise Unsupported(f"{name}: only int32 and int64 indices are imported")
        if source[axis] > 2 ** (index["width"] - 1):
            raise Unsupported(f"{name}: an axis longer than the index dtype's positive range")
        if exact:
            def reversed_(x: int) -> int:
                if not largest:
                    return x
                kind = "logical_not" if scalar == "bool" else "bitwise_not"
                return self.node(kind, [x], self.shape(x), dtype=scalar)

            keys, order = self.sort([reversed_(operand), self.iota(source, axis)],
                                    dict(dimension=axis, is_stable=True, num_keys=1))
            stop = [k if a == axis else extent for a, extent in enumerate(source)]
            starts, steps = [0] * len(source), [1] * len(source)
            indices = self.slice(order, starts, stop, steps)
            if index != _INT64:
                # An int32 index, exact since the axis is at most 2^31 long.
                indices = self.node("custom", [indices], shapes[1], dtype=index,
                                    primitive="cast", rounding="towardZero")
            return [reversed_(self.slice(keys, starts, stop, steps)), indices]
        if k > 0:
            operand = self.decided(operand, axis, mixed_zeros=not approximate)
        values = self.node("topk", [operand], shapes[0], dtype=scalar, dim=axis, k=k,
                           largest=largest)
        indices = self.registered("topk", [{"shape": source}, {"nat": axis}, {"nat": k},
                                           {"bool": largest}], [operand], shapes[1], scalar,
                                  output=1, dtype=index)
        return [values, indices]

    def decided(self, x: int, axis: int, mixed_zeros: bool = True) -> int:
        """``x``, or a read at index 1 of a one-entry tensor, whose range check fails at execution,
        where a fiber along ``axis`` holds a NaN, or with ``mixed_zeros`` both a ``-0`` and a
        ``+0``."""
        source, scalar = self.shape(x), self.nodes[x]["dtype"]
        dtype = jax.numpy.dtype(scalar)
        rest = [extent for a, extent in enumerate(source) if a != axis]
        def present(flag: int) -> int:
            # Whether the fiber holds a flagged entry: a binary64 count, exact below 2 ** 53.
            counted = self.node("custom", [flag], source, dtype="float64", primitive="cast",
                                rounding="nearestEven")
            total = self.node("reduce", [counted], rest, dtype="float64", op="sum", dims=[axis],
                              keepdim=False)
            return self.node("ne", [total, self.scalar(0.0, np.float64, rest)], rest,
                             dtype="bool")
        bad = present(self.node("ne", [x, x], source, dtype="bool"))
        if mixed_zeros:
            zero, one = self.scalar(0, dtype, source), self.scalar(1, dtype, source)
            positive = self.node("gt", [self.node("copysign", [one, x], source, dtype=scalar),
                                        zero], source, dtype="bool")
            zeros = self.node("eq", [x, zero], source, dtype="bool")
            mixed = self.node("logical_and", [
                present(self.node("logical_and", [zeros, positive], source, dtype="bool")),
                present(self.node("logical_and", [zeros, self.node("logical_not", [positive],
                                                                   source, dtype="bool")],
                                  source, dtype="bool"))], rest, dtype="bool")
            bad = self.node("logical_or", [bad, mixed], rest, dtype="bool")
        counted = self.node("custom", [bad], rest, dtype="float64", primitive="cast",
                            rounding="nearestEven")
        total = self.node("reduce", [counted], [], dtype="float64", op="sum",
                          dims=list(range(len(rest))), keepdim=False) if rest else counted
        return self.refusing(self.node("ne", [total, self.scalar(0.0, np.float64, [])], [],
                                       dtype="bool"), x)
