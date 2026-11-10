"""Lower static tensor Jaxpr computations to TorchLean's typed artifact, inside a capture document.

Use ``export(function, *examples, checker=checker)`` or ``lower(closed_jaxpr, checker)``, and
``references`` for the reference outputs that ``compare`` checks. The document's ``artifact`` is
in the format ``op_table(checker)["format"]`` and holds its constants, so Lean loads it without a
Python interpreter. The adapter neither certifies the translation nor prescribes XLA's
floating-point reduction order.

Each primitive keeps its JAX meaning: ``lax.div`` is IEEE division, ``rsqrt`` is the IR's
``1 / sqrt x`` rounded twice (in binary64 ``rsqrt 3`` is one ulp above the correctly rounded
value), and results underflow gradually. XLA's compiled CPU code departs from that meaning: it
multiplies by the reciprocal of a broadcast divisor, approximates ``rsqrt`` and flushes subnormal
values to zero. Under ``jax.jit`` it also multiplies by the reciprocal of a constant divisor and
rewrites divisions and constant powers. The adapter does not follow it, but records what the
checker's ``compare`` needs to emulate it. ``reference_execution`` is ``"jit"`` when the function
is one ``jax.jit`` call, so that calling it runs one compiled program, and ``"eager"`` otherwise.
The document's ``computation`` and ``primitive_index`` have one entry per artifact step. In an
eager function, a step's ``computation`` is the primitive, or the nested ``jax.jit`` call, that
XLA compiles its equation in, and ``jit_computations`` lists the computations that are nested
calls. A step's ``primitive_index`` is the primitive that produced it, counted inside nested calls
too, for a reference computed one primitive at a time. An entry is null where there is no such
equation or primitive, as for an input. ``TorchLean/NN/IR/README.md`` gives the measured
differences.
"""

from __future__ import annotations

from collections.abc import Sequence
import itertools
import math
from pathlib import Path
from typing import Any, Callable

import jax
import numpy as np

from . import _artifact
from ._table import op_table

DType = str | dict[str, Any]
_REDUCED_FLOATS = ("bfloat16", "float16")
# Call primitives, whose bodies the importer inlines, with the parameter holding each body.
_CALLS = {"jit": "jaxpr", "pjit": "jaxpr", "custom_jvp_call": "call_jaxpr",
          "custom_vjp_call": "call_jaxpr", "remat2": "jaxpr", "composite": "call_jaxpr"}
# Calls that run their body one primitive at a time outside jax.jit, not as one computation.
_TRANSPARENT = ("custom_jvp_call", "custom_vjp_call", "remat2", "composite")
# Primitives that return their operands unchanged.
_IDENTITIES = ("copy", "device_put", "optimization_barrier")
_REDUCTIONS = {"reduce_sum": "sum", "reduce_prod": "prod", "reduce_max": "max",
               "reduce_min": "min"}
# Elementary functions computed, for binary16 and bfloat16, in binary32 and rounded once.
_WIDENED_UNARY = ("expm1", "log1p", "exp2", "log2", "tan", "sinh", "cosh", "asinh", "acosh",
                  "atanh", "asin", "acos", "atan")
# Functions whose results are exact in the operand's floating format.
_EXACT_UNARY = ("floor", "ceil")
_FLOATING = ("float64", "float32", "bfloat16", "float16")
# Comparisons of floating or integer operands, is_finite of floating operands, and the logical
# operators of Boolean operands, with their IR kinds. Each gives a Boolean.
_BOOLEAN = {"eq": "eq", "ne": "ne", "lt": "lt", "le": "le", "gt": "gt", "ge": "ge",
            "is_finite": "isfinite", "and": "logical_and", "or": "logical_or",
            "xor": "logical_xor", "not": "logical_not"}
_INT64 = dict(kind="integer", signed=True, width=64)
# Primitives that read or write their operand at runtime indices.
_INDEXED = ("gather", "dynamic_slice", "scatter", "scatter-add", "scatter-sub", "scatter-mul",
            "scatter-max", "scatter-min", "dynamic_update_slice")
# The rule of `Op.scatter` each writing primitive uses, by its name in the op table's scatter
# codes.
_SCATTER_RULES = {"scatter": "replace", "dynamic_update_slice": "replace", "scatter-add": "sum",
                  "scatter-sub": "sum", "scatter-mul": "prod", "scatter-max": "max",
                  "scatter-min": "min"}
_UINT64 = dict(kind="integer", signed=False, width=64)
# Integer bit operators (StableHLO's, on the two's-complement word) and their IR operators.
# For integer operands, lax's and, or, xor and not are the bitwise ones.
_BITS = {"shift_left": "bitwise_left_shift", "shift_right_arithmetic": "shift_right_arithmetic",
         "shift_right_logical": "shift_right_logical", "population_count": "population_count",
         "clz": "count_leading_zeros", "mulhi": "mulhi", "and": "bitwise_and",
         "or": "bitwise_or", "xor": "bitwise_xor", "not": "bitwise_not"}
# The windowed and general reductions whose body is one of these monoids, with its IR kind.
_MONOIDS = {"add": "add", "mul": "mul_elem", "max": "max_elem", "min": "min_elem"}
_WINDOWED = {"reduce_window_sum": "add", "reduce_window_max": "max_elem",
             "reduce_window_min": "min_elem"}


def _plain_update(name: str, params: dict[str, Any]) -> bool:
    """Whether a scatter's update computation is its name's: none for ``scatter`` (a
    replacement), one ``add``, ``sub``, ``mul``, ``max`` or ``min`` of the target and the update
    otherwise.
    ``.at[...].apply`` reaches ``scatter`` with its own function and dummy updates."""
    jaxpr = params.get("update_jaxpr")
    if params.get("update_consts"):
        return False
    if name == "scatter":
        return jaxpr is None
    jaxpr = getattr(jaxpr, "jaxpr", jaxpr)
    if jaxpr is None or getattr(jaxpr, "constvars", None) or len(jaxpr.eqns) != 1:
        return False
    equation = jaxpr.eqns[0]
    return (equation.primitive.name == name.removeprefix("scatter-")
            and list(equation.invars) == list(jaxpr.invars)
            and list(jaxpr.outvars) == list(equation.outvars))
# Scans along one axis, each one IR scan.
_SCANS = ("cumsum", "cumprod", "cummax", "cummin")


class Unsupported(ValueError):
    """Raised when a captured computation has no translation into TorchLean IR."""


def _shape(aval: Any) -> list[int]:
    try:
        shape = [int(d) for d in aval.shape]
    except (AttributeError, TypeError, ValueError) as error:
        raise Unsupported("Jaxpr import requires static tensor shapes") from error
    if any(d < 0 for d in shape):
        raise Unsupported("Jaxpr import requires nonnegative dimensions")
    return shape


def _dtype(aval: Any) -> DType:
    raw = getattr(aval, "dtype", None)
    if raw is None:
        raise Unsupported("Jaxpr import: a tensor value must declare its element type")
    try:
        dtype = np.dtype(raw)
    except (TypeError, ValueError) as error:
        raise Unsupported(f"Jaxpr import: unsupported element type {raw}") from error
    if dtype.name in {"float16", "bfloat16", "float32", "float64"}:
        return dtype.name
    if dtype.kind == "b":
        return "bool"
    if dtype.kind in {"i", "u"}:
        return dict(kind="integer", signed=dtype.kind == "i", width=8 * dtype.itemsize)
    if dtype.kind == "c" and dtype.itemsize in (8, 16):
        return dict(kind="complex", component="float32" if dtype.itemsize == 8 else "float64")
    raise Unsupported(f"Jaxpr import: unsupported element type {dtype}")


def bits(value: Any) -> list[int]:
    """Exact element words: Boolean 0/1, or consecutive complex real/imaginary words."""
    array = np.asarray(value)
    if array.dtype.kind == "b":
        return [int(value) for value in array.reshape(-1)]
    width = array.dtype.itemsize // 2 if array.dtype.kind == "c" else array.dtype.itemsize
    unsigned = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}.get(width)
    supported = (array.dtype.name in {"float16", "bfloat16", "float32", "float64"}
                 or array.dtype.kind in {"i", "u"}
                 or array.dtype.kind == "c" and array.dtype.itemsize in (8, 16))
    if unsigned is None or not supported:
        raise Unsupported(f"no binary word encoding for {array.dtype}")
    words = array.reshape(-1).view(np.dtype(unsigned).newbyteorder(array.dtype.byteorder))
    return [int(word) for word in words]


def _is_integer(dtype: DType) -> bool:
    return isinstance(dtype, dict) and dtype.get("kind") == "integer"


def encode(value: Any) -> dict[str, Any]:
    """Serialize a tensor with its canonical element descriptor, shape and exact words."""
    return dict(dtype=_dtype(value), shape=_shape(value), bits=bits(value))


class _Builder:
    def __init__(self, dtype: DType, checker: str | Path | Sequence[str] | None = None):
        self.current_dtype = dtype
        # The caller's checker: the records follow its op table.
        self.checker = checker
        self.nodes: list[dict[str, Any]] = []
        # The computation XLA compiles new nodes in when the function runs eagerly: each
        # primitive is one, and so is each nested jax.jit call.
        self.computation: int | None = None
        self.computations = 0
        self.jit_computations: list[int] = []
        # The primitive that produces new nodes, counted inside nested calls too: each one is a
        # computation of its own when the function is evaluated one primitive at a time.
        self.primitive_index: int | None = None
        self.primitives = 0

    def node(self, kind: str, parents: list[int], shape: list[int],
             dtype: DType | None = None, **attrs: Any) -> int:
        index = len(self.nodes)
        if self.computation is not None:
            attrs["computation"] = self.computation
        if self.primitive_index is not None:
            attrs["primitive_index"] = self.primitive_index
        self.nodes.append(dict(id=index, kind=kind, parents=parents, shape=shape,
                               dtype=dtype or self.current_dtype, **attrs))
        return index

    def registered(self, tag: str, fields: list[dict[str, Any]], parents: list[int],
                   shape: list[int], scalar: DType, output: int = 0,
                   dtype: DType | None = None) -> int:
        # The one constructor of an IR node holding a registered operator of TorchLean.Op: output
        # `output` of the operator whose codec record is (tag, fields), its value ports at
        # `scalar`. Lean's `Node.check` refuses a parent whose port, shape included, the
        # operator's input port does not accept.
        record = {"version": op_table(self.checker)["version"], "tag": tag, "fields": fields}
        return self.node("custom", parents, shape, dtype, name=tag, primitive="op", op=record,
                         scalar=scalar, output=output)

    def shape(self, index: int) -> list[int]:
        return self.nodes[index]["shape"]

    def constant(self, value: Any, aval: Any) -> int:
        dtype = _dtype(aval)
        array = np.asarray(value)
        if array.dtype.name != np.dtype(aval.dtype).name:
            # A typed literal such as lax.pad's padding value or a weakly typed integer may be
            # stored as a Python number. It is accepted when its declared type holds the stored
            # value exactly: an integer within the declared integer type's bounds, otherwise a
            # value other than a NaN that converts back unchanged and does not overflow to an
            # infinity.
            target = np.dtype(aval.dtype)
            with np.errstate(all="ignore"):
                converted = array.astype(target)
            if array.dtype.kind in "iu" and target.kind in "iu":
                exact = array.size == 0 or (np.iinfo(target).min <= int(array.min()) and
                                            int(array.max()) <= np.iinfo(target).max)
            else:
                with np.errstate(all="ignore"):
                    exact = array.dtype.kind in "fiu" and np.array_equal(
                        converted.astype(array.dtype), array) and not np.any(
                        np.isinf(converted.astype(np.float64)) &
                        ~np.isinf(array.astype(np.float64)))
            if not exact:
                raise Unsupported("Jaxpr import: constant storage differs from its declared "
                                  "element type")
            array = value = converted
        attrs = dict(bits=bits(value))
        # The homogeneous diagnostic loader also accepts finite native float values.
        # Other types retain only their exact words, so no large integer is rounded through Float.
        if isinstance(dtype, str) and dtype != "bool":
            values = [float(x) for x in array.reshape(-1)]
            if all(math.isfinite(x) for x in values):
                attrs["values"] = values
        return self.node("const", [], _shape(aval), dtype=dtype, **attrs)

    def reshape(self, index: int, shape: list[int]) -> int:
        source = self.shape(index)
        if source == shape:
            return index
        if math.prod(source) != math.prod(shape):
            raise Unsupported(f"reshape changes element count: {source} -> {shape}")
        return self.node("reshape", [index], shape, dtype=self.nodes[index]["dtype"])

    def permute(self, index: int, axes: list[int]) -> int:
        source = self.shape(index)
        if sorted(axes) != list(range(len(source))):
            raise Unsupported(f"invalid axis permutation {axes} for {source}")
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
            raise Unsupported(f"invalid broadcast {source} -> {shape} along {dims}")
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
            raise Unsupported(f"slice: invalid window {starts}:{stops}:{steps} of {source}")
        shape = [len(range(starts[d], stops[d], steps[d])) for d in range(len(source))]
        return self.node("slice", [index], shape, dtype=self.nodes[index]["dtype"], dims=dims,
                         starts=[starts[d] for d in dims], stops=[stops[d] for d in dims],
                         steps=[steps[d] for d in dims])

    def int64_constant(self, value: Any, shape: list[int]) -> int:
        array = np.ascontiguousarray(np.broadcast_to(np.asarray(value, dtype=np.int64), shape))
        return self.node("const", [], list(shape), dtype=_INT64, bits=bits(array))

    def iota(self, shape: list[int], axis: int) -> int:
        """The int64 constant of ``shape`` whose entry at ``c`` is ``c[axis]``."""
        view = [-1 if a == axis else 1 for a in range(len(shape))]
        return self.int64_constant(np.arange(shape[axis]).reshape(view), shape)

    def start(self, indices: int, entry: int | None = None, through: DType = _INT64) -> int:
        """Entry ``entry`` of each index vector (the trailing axis), or a 0-d start, in int64,
        converted to ``through`` first."""
        self.index_type(indices)
        if entry is not None:
            source = self.shape(indices)
            starts, stops = [0] * len(source), list(source)
            starts[-1], stops[-1] = entry, entry + 1
            indices = self.reshape(self.slice(indices, starts, stops, [1] * len(source)),
                                   source[:-1])
        for target in (through, _INT64):
            if self.nodes[indices]["dtype"] != target:
                indices = self.node("custom", [indices], self.shape(indices), dtype=target,
                                    primitive="cast", rounding="towardZero")
        return indices

    def index_type(self, indices: int) -> None:
        dtype = self.nodes[indices]["dtype"]
        if not _is_integer(dtype) or dtype == _UINT64:
            raise Unsupported("runtime indices: only integer indices other than uint64 are "
                              "imported")

    def bounded(self, start: int, high: int, clamp: bool) -> int:
        """Int64 starts clamped to ``[0, high]``, or whether each lies there."""
        shape = self.shape(start)
        low, high = [self.broadcast(self.int64_constant(bound, []), shape, [])
                     for bound in (0, high)]
        if clamp:
            above = self.node("max_elem", [start, low], shape, dtype=_INT64)
            return self.node("min_elem", [above, high], shape, dtype=_INT64)
        above = self.node("ge", [start, low], shape, dtype="bool")
        below = self.node("le", [start, high], shape, dtype="bool")
        return self.node("logical_and", [above, below], shape, dtype="bool")

    def coordinates(self, shape: list[int], terms: list[list[int]], inside: int | None = None,
                    spare: int = 0) -> int:
        """Index tuples of ``shape``: along operand axis ``a`` the sum of ``terms[a]``, and along
        axis 0 ``spare`` where ``inside`` is false."""
        components = []
        for parts in terms:
            total = parts[0] if parts else self.int64_constant(0, shape)
            for part in parts[1:]:
                total = self.node("add", [total, part], shape, dtype=_INT64)
            components.append(total)
        if inside is not None:
            components[0] = self.node("where", [inside, components[0],
                                                self.int64_constant(spare, shape)],
                                      shape, dtype=_INT64)
        if len(components) == 1:
            return components[0]
        pieces = [self.reshape(component, shape + [1]) for component in components]
        return self.node("cat", pieces, shape + [len(pieces)], dtype=_INT64, dim=len(shape))

    def indexed(self, name: str, parents: list[int], params: dict[str, Any], shape: list[int],
                dtype: Any) -> int:
        # Reads and writes at runtime indices: one registered gather or scatter at complete int64
        # coordinates of the operand. Starts are clamped so that the window lies inside, as XLA
        # clamps them: always for gather, dynamic_slice and dynamic_update_slice, in the CLIP
        # mode for scatter. A FILL_OR_DROP gather gives fill_value where a start lies outside,
        # and a scatter in another mode drops the whole window of such a start.
        operand = parents[0]
        if name.startswith("scatter") and not _plain_update(name, params):
            raise Unsupported(f"{name}: only a replacement, a sum, a difference, a product, a "
                              "maximum or a minimum of the updates is imported")
        values = [operand] + ([parents[1]] if name == "dynamic_update_slice" else
                              [parents[2]] if name.startswith("scatter") else [])
        if self.current_dtype not in _FLOATING or any(
                self.nodes[value]["dtype"] != self.current_dtype for value in values):
            raise Unsupported(f"{name}: only floating operands, and updates of their type, are "
                              "imported")
        source, scalar = self.shape(operand), self.current_dtype
        if not source:
            raise Unsupported(f"{name}: a 0-d operand has no index")
        if 0 in source:
            raise Unsupported(f"{name}: an operand without entries is not imported")
        modes = jax.lax.GatherScatterMode
        mode = params.get("mode")
        if mode is not None and mode not in (modes.CLIP, modes.FILL_OR_DROP,
                                             modes.PROMISE_IN_BOUNDS):
            raise Unsupported(f"{name}: the {mode} mode is not imported")
        every = list(range(len(source)))
        if name in ("dynamic_slice", "dynamic_update_slice"):
            update = parents[1] if name == "dynamic_update_slice" else None
            starts = parents[1 if update is None else 2:]
            window = shape if update is None else self.shape(update)
            if len(starts) != len(source):
                raise Unsupported(f"{name}: one start per operand axis is imported")
            terms = [[self.broadcast(self.bounded(self.start(begin), extent - size, True),
                                     window, []), self.iota(window, axis)]
                     for axis, (begin, extent, size) in enumerate(zip(starts, source, window))]
            tuples = self.coordinates(window, terms)
            if update is None:
                return self.registered("gather", [{"shape": source}, {"shape": window},
                                                  {"nats": every}, {"pairs": []},
                                                  _artifact._SCALAR],
                                       [operand, tuples], shape, scalar)
            code = op_table(self.checker)["codes"]["scatter"][_SCATTER_RULES[name]]
            return self.registered("scatter", [{"shape": source}, {"shape": window},
                                               {"nat": code}, {"nats": every}, _artifact._SCALAR],
                                   [operand, tuples, update], shape, scalar)
        numbers, indices = params["dimension_numbers"], parents[1]
        self.index_type(indices)
        terms, inside = [[] for _ in source], None
        if name == "gather":
            # Output axes outside offset_dims follow the index vectors, the others the window.
            sizes = [int(size) for size in params["slice_sizes"]]
            windows, positions = [int(axis) for axis in numbers.offset_dims], shape
            batch = [axis for axis in range(len(shape)) if axis not in windows]
            skipped = set(numbers.collapsed_slice_dims) | set(numbers.operand_batching_dims)
            pairs = zip(numbers.operand_batching_dims, numbers.start_indices_batching_dims)
            mapped, checked = numbers.start_index_map, mode == modes.FILL_OR_DROP
            # A FILL_OR_DROP gather converts its indices to int32 when every axis fits, as
            # `_gather_fill` does, so a wider index wraps there.
            through = (dict(kind="integer", signed=True, width=32)
                       if checked and max(source) < 2 ** 31 else _INT64)
        else:
            # With the index axes first, the IR applies the updates index vector by index vector
            # in row-major order, each window whole.
            extents = self.shape(parents[2])
            skipped = set(numbers.inserted_window_dims) | set(numbers.operand_batching_dims)
            lead = [axis for axis in range(len(extents))
                    if axis not in numbers.update_window_dims]
            updates = self.permute(parents[2], lead + list(numbers.update_window_dims))
            positions, batch = self.shape(updates), list(range(len(lead)))
            if name == "scatter-sub":
                # x - u is x + (-u) in IEEE arithmetic.
                updates = self.node("neg", [updates], positions)
            windows = list(range(len(lead), len(extents)))
            sizes = [1] * len(source)
            for axis, along in zip([a for a in every if a not in skipped], windows):
                sizes[axis] = positions[along]
            pairs = zip(numbers.operand_batching_dims, numbers.scatter_indices_batching_dims)
            mapped, checked = numbers.scatter_dims_to_operand_dims, mode != modes.CLIP
            through = _INT64
        for entry, axis in enumerate(int(axis) for axis in mapped):
            begin = self.start(indices, entry, through)
            if checked:
                flag = self.bounded(begin, source[axis] - sizes[axis], False)
                inside = flag if inside is None else self.node(
                    "logical_and", [inside, flag], self.shape(flag), dtype="bool")
            begin = self.bounded(begin, source[axis] - sizes[axis], True)
            terms[axis].append(self.broadcast(begin, positions, batch))
        for axis, along in pairs:
            terms[int(axis)].append(self.iota(positions, batch[int(along)]))
        for axis, along in zip([a for a in every if a not in skipped], windows):
            terms[axis].append(self.iota(positions, along))
        if inside is not None:
            inside = self.broadcast(inside, positions, batch)
        if name == "gather":
            result = self.registered("gather", [{"shape": source}, {"shape": shape},
                                                {"nats": every}, {"pairs": []}, _artifact._SCALAR],
                                     [operand, self.coordinates(shape, terms)], shape, scalar)
            if inside is None:
                return result
            value = params.get("fill_value")
            word = np.asarray(np.nan if value is None else value).astype(dtype)
            attrs = dict(values=[float(word)]) if math.isfinite(float(word)) else {}
            fill = self.node("const", [], [], dtype=scalar, bits=bits(word), **attrs)
            return self.node("where", [inside, result, self.broadcast(fill, shape, [])], shape)
        code, base = op_table(self.checker)["codes"]["scatter"][_SCATTER_RULES[name]], operand
        if inside is not None and name == "scatter-mul":
            # A dropped update becomes the product's identity, so the padded slice's zero entries
            # and its zero cotangent meet finite factors only (0 * inf would give a NaN pullback).
            one = self.node("const", [], [], dtype=scalar, bits=bits(np.ones((), dtype=dtype)),
                            values=[1.0])
            updates = self.node("where", [inside, updates, self.broadcast(one, positions, [])],
                                positions)
        if inside is not None:
            # A dropped window writes into a slice appended along axis 0, removed afterwards.
            zero = np.zeros((), dtype=dtype)
            padded = [source[0] + 1] + source[1:]
            base = self.node("pad", [operand, self.node("const", [], [], dtype=scalar,
                                                        bits=bits(zero), values=[0.0])],
                             padded, low=[0] * len(source), high=[1] + [0] * (len(source) - 1),
                             interior=[0] * len(source))
        extents = self.shape(base)
        written = self.registered("scatter", [{"shape": extents}, {"shape": positions},
                                              {"nat": code}, {"nats": every}, _artifact._SCALAR],
                                  [base, self.coordinates(positions, terms, inside, source[0]),
                                   updates], extents, scalar)
        return self.slice(written, [0] * len(source), source, [1] * len(source))

    def scalar(self, value: Any, dtype: Any, shape: list[int]) -> int:
        """A constant ``value`` of the NumPy type ``dtype``, broadcast to ``shape``."""
        word = np.asarray(value, dtype=dtype)
        return self.broadcast(self.constant(word, jax.core.ShapedArray((), word.dtype)), shape, [])

    def sign(self, x: int, shape: list[int], dtype: Any) -> int:
        """``lax.sign`` as documented: ``x`` itself where it is a NaN or a zero, so that ``-0``
        stays ``-0`` and a NaN stays that NaN word (which JAX does not promise), and
        ``copysign(1, x)`` elsewhere. Integers give ``-1``, ``0`` or ``1`` by the type's order."""
        if _is_integer(self.current_dtype):
            zero = self.scalar(0, dtype, shape)
            result = self.node("where", [self.node("gt", [x, zero], shape, dtype="bool"),
                                         self.scalar(1, dtype, shape), x], shape)
            if not self.current_dtype["signed"]:
                return result
            return self.node("where", [self.node("lt", [x, zero], shape, dtype="bool"),
                                       self.scalar(-1, dtype, shape), result], shape)
        if self.current_dtype not in _FLOATING:
            raise Unsupported("sign: only floating and integer operands are imported")
        nan = self.node("ne", [x, x], shape, dtype="bool")
        zero = self.node("eq", [x, self.scalar(0, dtype, shape)], shape, dtype="bool")
        kept = self.node("logical_or", [nan, zero], shape, dtype="bool")
        unit = self.node("copysign", [self.scalar(1, dtype, shape), x], shape)
        return self.node("where", [kept, x, unit], shape)

    def scan(self, name: str, x: int, params: dict[str, Any], shape: list[int],
             dtype: Any) -> int:
        """IR scans along ``axis``, between two flips for ``reverse``.

        ``cumsum`` and ``cumprod`` are the IR's left folds from ``+0`` and ``1``, rounding each step
        in the operand's format. The order is TorchLean's choice: lax documents none, and XLA's CPU
        code combines prefixes associatively, so the two agree where every partial result of either
        grouping is exact. A prefix of ``-0`` entries sums to ``+0`` from the seed, as on XLA's and
        PyTorch's CPU code. ``cummax`` and ``cummin`` give the value and zero sign of the running
        IEEE maximum and minimum of ``lax.max`` and ``lax.min``, a NaN propagating as the latest
        NaN's word. The IR's ``cummax`` and ``cummin`` take a later equal entry, as ATen does, which
        on numbers differs from the IEEE extrema only on mixed signed zeros. A second scan of a
        ``+0`` flag turns a zero result into ``+0`` where a ``+0`` entry precedes it (``-0`` and a
        ``-0`` entry for ``cummin``)."""
        if self.current_dtype not in _FLOATING:
            raise Unsupported(f"{name}: only floating operands are imported")
        axis = int(params["axis"])
        if params.get("reverse"):
            x = self.node("flip", [x], shape, dims=[axis])
        result = self.node(name, [x], shape, dim=axis)
        if name in ("cummax", "cummin"):
            zero, one = self.scalar(0, dtype, shape), self.scalar(1, dtype, shape)
            positive = self.node("gt", [self.node("copysign", [one, x], shape), zero], shape,
                                 dtype="bool")
            signed = positive if name == "cummax" else self.node("logical_not", [positive],
                                                                  shape, dtype="bool")
            flag = self.node("logical_and", [self.node("eq", [x, zero], shape, dtype="bool"),
                                             signed], shape, dtype="bool")
            seen = self.node("cummax", [self.node("where", [flag, one, zero], shape)], shape,
                             dim=axis)
            fixed = self.node("logical_and", [
                self.node("eq", [result, zero], shape, dtype="bool"),
                self.node("gt", [seen, zero], shape, dtype="bool")], shape, dtype="bool")
            target = zero if name == "cummax" else self.scalar(-0.0, dtype, shape)
            result = self.node("where", [fixed, target, result], shape)
        if params.get("reverse"):
            result = self.node("flip", [result], shape, dims=[axis])
        return result

    def top_k(self, operand: int, params: dict[str, Any], shapes: list[list[int]],
              index: DType) -> list[int]:
        """``lax.top_k``: the IR's stable ``topk`` for the values and ``Op.topk``'s second output
        for the indices, in the primitive's index dtype.

        Equal entries keep their input order, which is the order ``lax.top_k`` documents for
        ``is_stable`` (``is_stable=False`` leaves it open). XLA's CPU code ranks in IEEE
        totalOrder, ``+0`` above ``-0`` and a NaN by its sign and payload, and ``lax.top_k``
        documents nothing for NaN, so for ``k > 0`` execution fails where a fiber holds a NaN, or
        both a ``-0`` and a ``+0``: elsewhere the stable order is totalOrder's."""
        source, scalar = self.shape(operand), self.nodes[operand]["dtype"]
        if scalar not in _FLOATING:
            raise Unsupported("top_k: only floating operands are imported")
        if not source:
            raise Unsupported("top_k: a 0-d operand has no axis")
        axis, k = int(params.get("axis", -1)), int(params["k"])
        axis = axis + len(source) if axis < 0 else axis
        if index not in (dict(kind="integer", signed=True, width=32), _INT64):
            raise Unsupported("top_k: only int32 and int64 indices are imported")
        if source[axis] > 2 ** (index["width"] - 1):
            raise Unsupported("top_k: an axis longer than the index dtype's positive range")
        if k > 0:
            operand = self.decided(operand, axis)
        values = self.node("topk", [operand], shapes[0], dtype=scalar, dim=axis, k=k,
                           largest=True)
        indices = self.registered("topk", [{"shape": source}, {"nat": axis}, {"nat": k},
                                           {"bool": True}], [operand], shapes[1], scalar,
                                  output=1, dtype=index)
        return [values, indices]

    def decided(self, x: int, axis: int) -> int:
        """``x``, or a read at index 1 of a one-entry tensor, whose range check fails at execution,
        where a fiber along ``axis`` holds a NaN, or both a ``-0`` and a ``+0``."""
        source, scalar = self.shape(x), self.nodes[x]["dtype"]
        dtype = jax.numpy.dtype(scalar)
        zero, one = self.scalar(0, dtype, source), self.scalar(1, dtype, source)
        positive = self.node("gt", [self.node("copysign", [one, x], source), zero], source,
                             dtype="bool")
        zeros = self.node("eq", [x, zero], source, dtype="bool")
        rest = [extent for a, extent in enumerate(source) if a != axis]
        def present(flag: int) -> int:
            # Whether the fiber holds a flagged entry: a binary64 count, exact below 2 ** 53.
            counted = self.node("custom", [flag], source, dtype="float64", primitive="cast",
                                rounding="nearestEven")
            total = self.node("reduce", [counted], rest, dtype="float64", op="sum", dims=[axis],
                              keepdim=False)
            return self.node("ne", [total, self.scalar(0.0, np.float64, rest)], rest,
                             dtype="bool")
        mixed = self.node("logical_and", [
            present(self.node("logical_and", [zeros, positive], source, dtype="bool")),
            present(self.node("logical_and", [zeros, self.node("logical_not", [positive], source,
                                                               dtype="bool")],
                              source, dtype="bool"))], rest, dtype="bool")
        bad = self.node("logical_or", [present(self.node("ne", [x, x], source, dtype="bool")),
                                       mixed], rest, dtype="bool")
        counted = self.node("custom", [bad], rest, dtype="float64", primitive="cast",
                            rounding="nearestEven")
        total = self.node("reduce", [counted], [], dtype="float64", op="sum",
                          dims=list(range(len(rest))), keepdim=False) if rest else counted
        failing = self.node("ne", [total, self.scalar(0.0, np.float64, [])], [], dtype="bool")
        position = self.node("where", [self.reshape(failing, [1]),
                                       self.int64_constant(1, [1]), self.int64_constant(0, [1])],
                             [1], dtype=_INT64)
        read = self.registered("gather", [{"shape": [1]}, {"shape": [1]}, {"nats": [0]},
                                          {"pairs": []}, _artifact._SCALAR],
                               [self.scalar(0, dtype, [1]), position], [1], scalar, dtype=scalar)
        return self.node("where", [self.broadcast(failing, source, []),
                                   self.broadcast(self.reshape(read, []), source, []), x], source,
                         dtype=scalar)

    def integer_pow(self, base: int, exponent: int, shape: list[int], dtype: Any) -> int:
        """Multiply in the order XLA does: square-and-multiply, then 1 / product if negative.

        The schedule is JAX's ``lax._integer_pow``, which its lowering emits for every exponent.
        The products round at each step, so the result need not be the correctly rounded power.
        """
        if exponent == 0:
            one = self.constant(np.array(1, dtype=dtype), jax.core.ShapedArray((), dtype))
            return self.broadcast(one, shape)
        power, result = abs(exponent), None
        while power > 0:
            if power & 1:
                result = base if result is None else self.node("mul_elem", [result, base], shape)
            power >>= 1
            if power > 0:
                base = self.node("mul_elem", [base, base], shape)
        if exponent < 0:
            one = self.constant(np.array(1, dtype=dtype), jax.core.ShapedArray((), dtype))
            result = self.node("div", [self.broadcast(one, shape), result], shape)
        return result

    def widened(self, kind: str, parents: list[int], shape: list[int]) -> int:
        """Compute ``kind`` in binary32 and round once, as XLA's CPU backend computes ``pow``,
        ``rsqrt`` and ``erf`` on binary16 and bfloat16 operands."""
        target = self.current_dtype
        if target not in _REDUCED_FLOATS:
            return self.node(kind, parents, shape)
        wide = [self.node("custom", [i], self.shape(i), dtype="float32", primitive="cast",
                          rounding="nearestEven") for i in parents]
        result = self.node(kind, wide, shape, dtype="float32")
        return self.node("custom", [result], shape, dtype=target, primitive="cast",
                         rounding="nearestEven")

    def dot(self, left: int, right: int, params: dict[str, Any], shape: list[int]) -> int:
        """One contraction over the paired axes, keeping JAX's output order: batch axes, then
        the free axes of the left and of the right operand."""
        precision = params.get("precision")
        defaults = (None, (jax.lax.Precision.DEFAULT, jax.lax.Precision.DEFAULT),
                    jax.lax.DotAlgorithmPreset.DEFAULT)
        if precision not in defaults:
            raise Unsupported("dot_general: an explicit precision or dot algorithm needs an "
                              "operand/accumulator arithmetic rule; the IR contraction does not "
                              f"implement {precision}")
        (lc, rc), (lb, rb) = params["dimension_numbers"]
        lc, rc, lb, rb = map(list, (lc, rc, lb, rb))
        ls, rs = self.shape(left), self.shape(right)
        for axes, rank in ((lb + lc, len(ls)), (rb + rc, len(rs))):
            if len(set(axes)) != len(axes) or any(i < 0 or i >= rank for i in axes):
                raise Unsupported("dot_general: invalid batch or contraction axes")
        if [ls[i] for i in lb] != [rs[i] for i in rb]:
            raise Unsupported("dot_general: batch dimensions differ")
        if [ls[i] for i in lc] != [rs[i] for i in rc]:
            raise Unsupported("dot_general: contraction dimensions differ")
        lf = [i for i in range(len(ls)) if i not in lb + lc]
        rf = [i for i in range(len(rs)) if i not in rb + rc]
        expected = [ls[i] for i in lb] + [ls[i] for i in lf] + [rs[i] for i in rf]
        if expected != shape:
            raise Unsupported(f"dot_general: expected output {expected}, got {shape}")
        return self.node("contract", [left, right], shape, dims=[list(p) for p in zip(lc, rc)],
                         batch_dims=[list(p) for p in zip(lb, rb)])

    def split(self, parents: list[int], params: dict[str, Any],
              shapes: list[list[int]]) -> list[int]:
        """One slice along ``axis`` for each output of ``lax.split``."""
        index, axis = parents[0], int(params["axis"])
        source = self.shape(index)
        outputs, start = [], 0
        for size, shape in zip(params["sizes"], shapes):
            stop = start + int(size)
            starts, stops = [0] * len(source), list(source)
            starts[axis], stops[axis] = start, stop
            outputs.append(self.slice(index, starts, stops, [1] * len(source)))
            start = stop
        if start != source[axis]:
            raise Unsupported("split: sizes do not cover the axis")
        return outputs

    def windows(self, operand: int, init: int, params: dict[str, Any], shape: list[int],
                kind: str, swapped: bool = False) -> int:
        """StableHLO's reduce_window with the body ``kind``: the operand padded with the init value,
        and each window folded from the init value in row-major order of its cells. The operand gets
        ``base_dilation - 1`` init values between entries, then the low and high padding, a negative
        amount cropping. The fold is one strided slice per cell, combined entrywise with the
        accumulated value as the body's first operand (its second where the body names them the
        other way). StableHLO leaves the order open. For the IEEE maximum and minimum, the order
        changes no value but a NaN's word."""
        if 0 in shape:
            return self.broadcast(init, shape, [])
        source = self.shape(operand)
        padding = [(int(a), int(b)) for a, b in params["padding"]]
        low, high = [max(0, a) for a, _ in padding], [max(0, b) for _, b in padding]
        interior = [int(d) - 1 for d in params["base_dilation"]]
        if any(d < 0 for d in interior):
            raise Unsupported("reduce_window: base dilation must be positive")
        padded = [lo + hi + n + max(0, n - 1) * inner
                  for n, lo, hi, inner in zip(source, low, high, interior)]
        index = operand
        if any(low + high + interior):
            index = self.node("pad", [operand, init], padded, low=low, high=high,
                              interior=interior)
        index = self.slice(index, [max(0, -a) for a, _ in padding],
                           [n - max(0, -b) for n, (_, b) in zip(padded, padding)],
                           [1] * len(source))
        result = self.broadcast(init, shape, [])
        strides = [int(s) for s in params["window_strides"]]
        dilation = [int(d) for d in params["window_dilation"]]
        for cell in itertools.product(*[range(int(w)) for w in params["window_dimensions"]]):
            begin = [c * d for c, d in zip(cell, dilation)]
            piece = self.slice(index, begin, [b + (o - 1) * s + 1
                                              for b, o, s in zip(begin, shape, strides)], strides)
            result = self.node(kind, [piece, result] if swapped else [result, piece], shape)
        return result

    def monoid(self, name: str, params: dict[str, Any]) -> tuple[str, bool]:
        """The IR kind of a reduction body that is one add, mul, max or min of its two arguments
        and nothing else, and whether it takes them in the other order."""
        body = params["jaxpr"]
        jaxpr = getattr(body, "jaxpr", body)
        constants = list(params.get("consts", ())) + list(getattr(body, "consts", ()))
        equations = list(jaxpr.eqns)
        if (len(jaxpr.invars) != 2 or constants or jaxpr.constvars or len(equations) != 1
                or equations[0].primitive.name not in _MONOIDS
                or set(equations[0].invars) != set(jaxpr.invars) or len(set(jaxpr.invars)) != 2
                or list(jaxpr.outvars) != list(equations[0].outvars)):
            raise Unsupported(f"{name}: only a body that is one add, mul, max or min of its two "
                              "arguments is imported")
        return (_MONOIDS[equations[0].primitive.name],
                list(equations[0].invars) != list(jaxpr.invars))

    def reduce_precision(self, x: int, params: dict[str, Any], shape: list[int],
                         numpy: Any) -> int:
        """StableHLO's reduce_precision: the mantissa rounded to ``mantissa_bits`` bits to nearest
        even at the operand's exponent, overflow to an infinity rounding there too (two casts
        through the FloatLib format with the operand's exponent field and ``mantissa_bits``
        fraction bits). The range step then gives an infinity of the original sign where the
        exponent exceeds the range of ``exponent_bits`` bits, and a zero of the original sign
        where it is below its normal range. Where ``exponent_bits`` is narrower than the
        operand's, the result has no subnormals, and otherwise the operand's own subnormals stay.
        A NaN is returned unchanged, the policy XLA's CPU code follows (measured, payloads and
        signaling NaNs kept)."""
        dtype = self.current_dtype
        if dtype not in _FLOATING:
            raise Unsupported("reduce_precision: only floating operands are imported")
        exponent, mantissa = int(params["exponent_bits"]), int(params["mantissa_bits"])
        binary = op_table(self.checker)["dtypes"][dtype]
        width, fraction = binary["exponent_bits"], binary["fraction_bits"]
        if mantissa < 1 or exponent < 1:
            # A width of zero has no last kept bit for "ties to even" (XLA's code rounds such a
            # tie to the even exponent field), and a FloatLib format needs a fraction bit.
            raise Unsupported("reduce_precision: mantissa_bits = 0 is not imported (ties to even "
                              "has no last mantissa bit, and FloatLib formats need one)")
        y = x
        if mantissa < fraction:
            narrow = dict(binary, fraction_bits=mantissa)
            y = self.node("custom", [x], shape, dtype=narrow, primitive="cast",
                          rounding="nearestEven")
            y = self.node("custom", [y], shape, primitive="cast", rounding="nearestEven")
        if exponent < width:
            largest = 2 ** (exponent - 1) - 1
            magnitude = self.node("abs", [y], shape)
            def flag(kind: str, power: int) -> int:
                return self.node(kind, [magnitude, self.scalar(2.0 ** power, numpy, shape)],
                                 shape, dtype="bool")
            def signed(value: float) -> int:
                return self.node("copysign", [self.scalar(value, numpy, shape), y], shape)
            y = self.node("where", [flag("ge", largest + 1), signed(math.inf),
                                    self.node("where", [flag("lt", 1 - largest), signed(0.0), y],
                                              shape)], shape)
        if y == x:
            return x
        return self.node("where", [self.node("isnan", [x], shape, dtype="bool"), x, y], shape)

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
        if self.current_dtype not in ("float32", "float64"):
            raise Unsupported("triangular_solve: only binary32 and binary64 operands are imported")
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
        R[k, :] = c R[k, :] - s z and z = c z + s R[k, :]."""
        if self.current_dtype not in ("float32", "float64"):
            raise Unsupported("cholesky_update: only binary32 and binary64 operands are imported")
        size = shape[-1]
        if len(shape) != 2 or self.shape(z) != [size]:
            raise Unsupported("cholesky_update: one square factor and one vector are imported")
        rows = [self.reshape(self.slice(r, [k, 0], [k + 1, size], [1, 1]), [size])
                for k in range(size)]
        for k in range(size):
            x = self.slice(rows[k], [k], [k + 1], [1])
            y = self.slice(z, [k], [k + 1], [1])
            ax, ay = self.node("abs", [x], [1]), self.node("abs", [y], [1])
            larger = self.node("where", [self.node("gt", [ax, ay], [1], dtype="bool"), ax, ay],
                               [1])
            xn, yn = self.node("div", [x, larger], [1]), self.node("div", [y, larger], [1])
            norm = self.node("sqrt", [self.node("add", [self.node("mul_elem", [xn, xn], [1]),
                                                        self.node("mul_elem", [yn, yn], [1])],
                                                [1])], [1])
            rh = self.node("div", [self.scalar(1.0, numpy, [1]), norm], [1])
            zero = self.node("eq", [y, self.scalar(0.0, numpy, [1])], [1], dtype="bool")
            c = self.node("where", [zero, self.scalar(1.0, numpy, [1]),
                                    self.node("mul_elem", [xn, rh], [1])], [1])
            s = self.node("where", [zero, self.scalar(0.0, numpy, [1]), self.node(
                "mul_elem", [self.node("neg", [yn], [1]), rh], [1])], [1])
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
        potrf's failure test (a NaN pivot fails), as lax.linalg.cholesky documents
        ("a matrix full of NaNs")."""
        dtype = self.current_dtype
        if dtype not in ("float32", "float64"):
            raise Unsupported("cholesky: only binary32 and binary64 operands are imported")
        rank, size = len(shape), shape[-1]
        lead = shape[:-2]
        if size == 0:
            return a
        swap = list(range(rank - 2)) + [rank - 1, rank - 2]
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

    def convolution(self, parents: list[int], params: dict[str, Any], shape: list[int]) -> int:
        """An IR convolution of a constant kernel, batch and feature axes leading.

        The IR node keeps the kernel as its payload. Other layouts, input dilation and batch
        groups need the general convolution."""
        source, kernel = parents
        numbers = params["dimension_numbers"]
        rank = len(self.shape(source)) - 2
        identity = tuple(range(rank + 2))
        if (tuple(numbers.lhs_spec), tuple(numbers.rhs_spec), tuple(numbers.out_spec)) != \
                (identity, identity, identity):
            raise Unsupported("conv_general_dilated: only the NC..., OI... and NC... layouts "
                              "are imported")
        if (any(int(d) != 1 for d in params["lhs_dilation"])
                or int(params["batch_group_count"]) != 1):
            raise Unsupported("conv_general_dilated: input dilation and batch groups need the "
                              "general convolution")
        if params.get("precision") is not None or params.get("preferred_element_type") is not None:
            raise Unsupported("conv_general_dilated: explicit precision needs an arithmetic rule")
        weights = self.nodes[kernel]
        if weights["kind"] != "const" or "values" not in weights:
            raise Unsupported("conv_general_dilated: the kernel must be a finite constant")
        padding = [(int(a), int(b)) for a, b in params["padding"]]
        if any(a < 0 or b < 0 for a, b in padding):
            raise Unsupported("conv_general_dilated: negative padding needs a slice first")
        groups = int(params["feature_group_count"])
        out_channels, per_group = weights["shape"][0], weights["shape"][1]
        values = np.asarray(weights["values"]).reshape(weights["shape"])
        words = np.asarray(weights["bits"], dtype=object).reshape(weights["shape"])
        dense = np.zeros([out_channels, per_group * groups, *weights["shape"][2:]])
        # Word 0 is +0 in every binary format, the value of the cells outside a group.
        dense_words = np.zeros(dense.shape, dtype=object)
        step = out_channels // groups
        for group in range(groups):
            rows = slice(group * step, (group + 1) * step)
            dense[rows, group * per_group:(group + 1) * per_group] = values[rows]
            dense_words[rows, group * per_group:(group + 1) * per_group] = words[rows]
        return self.node("conv", [source], shape, spatial_rank=rank,
                         kernel=list(weights["shape"][2:]),
                         stride=[int(s) for s in params["window_strides"]],
                         padding=[a for a, _ in padding], padding_after=[b for _, b in padding],
                         dilation=[int(d) for d in params["rhs_dilation"]], groups=groups,
                         channel_axis=1, input_spatial=self.shape(source)[2:],
                         in_channels=per_group * groups, out_channels=out_channels,
                         weight=dense.tolist(), bias=[0.0] * out_channels,
                         weight_bits=[int(word) for word in dense_words.reshape(-1)],
                         weight_dtype=weights["dtype"], bias_bits=[0] * out_channels,
                         bias_dtype=weights["dtype"])

    def body(self, closed: Any, inputs: list[int]) -> list[int]:
        jaxpr = getattr(closed, "jaxpr", closed)
        constants = getattr(closed, "consts", ())
        if getattr(jaxpr, "effects", ()):
            raise Unsupported("Jaxpr import: effects and mutation are not supported")
        if len(jaxpr.invars) != len(inputs) or len(jaxpr.constvars) != len(constants):
            raise Unsupported("Jaxpr import: call input or constant count mismatch")
        env = dict(zip(jaxpr.invars, inputs))
        for var, value in zip(jaxpr.constvars, constants):
            env[var] = self.constant(value, var.aval)

        def read(atom: Any) -> int:
            if hasattr(atom, "val"):
                return self.constant(atom.val, atom.aval)
            if atom not in env:
                raise Unsupported(f"Jaxpr import: unbound variable {atom}")
            return env[atom]

        for equation in jaxpr.eqns:
            name = equation.primitive.name
            if getattr(equation, "effects", ()):
                raise Unsupported(f"{name}: effects are not supported")
            values = [*equation.outvars, *(a for a in equation.invars if hasattr(a, "aval"))]
            for var in values:
                kind = _dtype(var.aval)
                if isinstance(kind, dict) and kind.get("kind") == "complex":
                    # The IR's interpreters implement binary, Boolean and integer elements only.
                    raise Unsupported(f"{name}: complex values have no element type in the IR's "
                                      "interpreters")
            enclosing, primitive = self.computation, self.primitive_index
            if enclosing is None and name not in _TRANSPARENT:
                # A custom JVP wrapper runs the primitives of its body one by one.
                self.computation, self.computations = self.computations, self.computations + 1
                if name in {"jit", "pjit"}:
                    self.jit_computations.append(self.computation)
            if name not in _CALLS:
                self.primitive_index, self.primitives = self.primitives, self.primitives + 1
            try:
                outputs = self.equation(equation, [read(atom) for atom in equation.invars])
            finally:
                self.computation, self.primitive_index = enclosing, primitive
            if len(outputs) != len(equation.outvars):
                raise Unsupported(f"{name}: call output count mismatch")
            for var, index in zip(equation.outvars, outputs):
                if self.shape(index) != _shape(var.aval):
                    raise Unsupported(f"{name}: lowering produced an inconsistent output shape")
                if self.nodes[index]["dtype"] != _dtype(var.aval):
                    raise Unsupported(f"{name}: lowering produced an inconsistent output dtype")
                env[var] = index
        return [read(atom) for atom in jaxpr.outvars]

    def equation(self, equation: Any, parents: list[int]) -> list[int]:
        name, params = equation.primitive.name, equation.params
        if name in _CALLS:
            return self.body(params[_CALLS[name]], parents)
        if name in _IDENTITIES and name != "copy":
            return list(parents)
        if name == "split":
            return self.split(parents, params, [_shape(var.aval) for var in equation.outvars])
        if name == "custom_linear_solve":
            # The forward value: the solve jaxpr applied to its constants and the right-hand side,
            # as JAX evaluates it. The matvec, the transposed solve and the implicit derivative
            # rule are not used, and the solution is not checked against matvec.
            lengths = params["const_lengths"]
            start = lengths.matvec + lengths.vecmat
            return self.body(params["jaxprs"].solve,
                             parents[start:start + lengths.solve] + parents[sum(lengths):])
        if params.get("accuracy") is not None:
            raise Unsupported(f"{name}: an accuracy request is not imported; the IR computes one "
                              "fixed approximation")
        if name == "top_k":
            return self.top_k(parents[0], params, [_shape(var.aval) for var in equation.outvars],
                              _dtype(equation.outvars[1].aval))
        if name == "sort" and len(equation.outvars) != 1:
            raise Unsupported("sort: several operands (argsort) are refused: their payload is "
                              "not checked to be the sorting axis's index")
        if len(equation.outvars) != 1:
            raise Unsupported(f"{name}: multiple primitive outputs are not supported")
        shape = _shape(equation.outvars[0].aval)
        self.current_dtype = _dtype(equation.outvars[0].aval)
        if name in ("and", "or", "xor", "not") and _is_integer(self.current_dtype):
            return [self.operation(name, parents, params, shape, equation.outvars[0].aval.dtype)]
        if name in _BOOLEAN or name == "select_n":
            return [self.boolean(name, parents, shape)]
        if name == "bitcast_convert_type":
            # Each element's word read at the new type, for types of one width (StableHLO's
            # bitcast_convert). Other widths add or remove an axis of parts in the platform's
            # byte order, which the API does not fix.
            source = equation.invars[0].aval.dtype
            if np.dtype(source).itemsize != np.dtype(params["new_dtype"]).itemsize or \
                    "bool" in (np.dtype(source).name, np.dtype(params["new_dtype"]).name):
                raise Unsupported("bitcast_convert_type: only types of one word width other than "
                                  "bool are imported")
            return [self.node("custom", parents, shape, primitive="bitcast")]
        if name in _INDEXED:
            return [self.indexed(name, parents, params, shape, equation.outvars[0].aval.dtype)]
        # argmax and argmin take a floating operand and return indices.
        if name not in ("convert_element_type", "argmax", "argmin") and any(
                self.nodes[i]["dtype"] != self.current_dtype for i in parents):
            raise Unsupported(f"{name}: heterogeneous arithmetic needs an explicit "
                              "operand/accumulator type rule")
        return [self.operation(name, parents, params, shape, equation.outvars[0].aval.dtype)]

    def boolean(self, name: str, parents: list[int], shape: list[int]) -> int:
        # Comparisons and logical operators take operands of one element type. With a Boolean
        # predicate and two cases, select_n(p, false_case, true_case) is the IR
        # where(p, true_case, false_case).
        dtypes = [self.nodes[i]["dtype"] for i in parents]
        operands = [self.broadcast(i, shape) for i in parents]
        if name == "select_n":
            if len(parents) != 3 or dtypes[0] != "bool":
                raise Unsupported("select_n: only a Boolean predicate and two cases are imported")
            if dtypes[1] != self.current_dtype or dtypes[2] != self.current_dtype:
                raise Unsupported("select_n: heterogeneous cases need an explicit type rule")
            predicate, false_case, true_case = operands
            return self.node("where", [predicate, true_case, false_case], shape)
        if any(dtype != dtypes[0] for dtype in dtypes):
            raise Unsupported(f"{name}: operands of two element types are not imported")
        if name in ("and", "or", "xor", "not"):
            if dtypes[0] != "bool":
                raise Unsupported(f"{name}: lax.{name} of integer operands is bitwise; only "
                                  "Boolean operands are imported")
        elif dtypes[0] not in _FLOATING and (name == "is_finite" or not _is_integer(dtypes[0])):
            raise Unsupported(f"{name}: only floating operands are imported")
        return self.node(_BOOLEAN[name], operands, shape, dtype="bool")

    def operation(self, name: str, parents: list[int], params: dict[str, Any],
                  shape: list[int], dtype: Any) -> int:
        unary = {"abs": "abs", "exp": "exp", "log": "log",
                 "sin": "sin", "cos": "cos", "tanh": "tanh", "logistic": "sigmoid",
                 "stop_gradient": "detach", "neg": "neg"}
        # `div` is IEEE division, also for a broadcast or constant divisor, where XLA's CPU code
        # multiplies by the rounded reciprocal.
        binary = {"add": "add", "add_any": "add", "sub": "sub", "mul": "mul_elem",
                  "max": "max_elem", "min": "min_elem", "div": "div"}
        if name in unary:
            return self.node(unary[name], parents, shape)
        if name in _WIDENED_UNARY + _EXACT_UNARY + ("round", "rem", "sort") and \
                self.current_dtype not in _FLOATING:
            raise Unsupported(f"{name}: only floating operands are imported")
        if name in _WIDENED_UNARY:
            return self.widened(name, parents, shape)
        if name in _EXACT_UNARY:
            return self.node(name, parents, shape)
        if name == "round":
            if params["rounding_method"] != jax.lax.RoundingMethod.TO_NEAREST_EVEN:
                raise Unsupported("round: only TO_NEAREST_EVEN is the IR's round")
            return self.node("round", parents, shape)
        if name == "rem":
            # lax.rem of floating operands is C's fmod, exact in the operand format.
            return self.node("fmod", [self.broadcast(i, shape) for i in parents], shape)
        if name == "sort":
            # One operand sorted ascending: XLA's stable sort places NaN last and keeps equal
            # entries, -0 and +0 among them, in order, as the IR sort does.
            if len(parents) != 1 or int(params["num_keys"]) != 1 or not params["is_stable"]:
                raise Unsupported("sort: only a stable sort of one operand is imported")
            return self.node("sort", parents, shape, dim=int(params["dimension"]),
                             descending=False)
        if name in ("argmax", "argmin"):
            # One axis, removed: Op.argmax and Op.argmin, the first extremum and the first NaN,
            # as a custom node holding the registered operator, in the primitive's index dtype.
            (axis,) = [int(a) for a in params["axes"]]
            source, scalar = self.shape(parents[0]), self.nodes[parents[0]]["dtype"]
            if scalar not in _FLOATING:
                raise Unsupported(f"{name}: only floating operands are imported")
            if source[axis] == 0:
                raise Unsupported(f"{name}: an empty axis has no index")
            # XLA breaks ties by comparing iota in the index dtype, so a narrower or unsigned
            # one can order positions differently from Op.argmax's natural indices.
            if self.current_dtype not in (dict(kind="integer", signed=True, width=32),
                                          dict(kind="integer", signed=True, width=64)):
                raise Unsupported(f"{name}: only int32 and int64 indices are imported")
            # A signed iota wraps past 2^(width - 1) - 1, so a longer axis is refused.
            if source[axis] > 2 ** (self.current_dtype["width"] - 1):
                raise Unsupported(f"{name}: an axis longer than the index dtype's positive range")
            return self.registered(name, [{"shape": source}, {"nat": axis}, {"bool": False}],
                                   parents, shape, scalar)
        if name in ("rsqrt", "erf"):
            return self.widened(name, parents, shape)
        if name in ("cbrt", "nextafter"):
            # cbrt is IEEE rootn(x, 3), correctly rounded in the operand's own format, and
            # nextafter is C's, exact.
            if self.current_dtype not in _FLOATING:
                raise Unsupported(f"{name}: only floating operands are imported")
            return self.node(name, [self.broadcast(i, shape) for i in parents], shape)
        if name in _BITS:
            if not _is_integer(self.current_dtype):
                raise Unsupported(f"{name}: only integer operands are imported")
            return self.node(_BITS[name], [self.broadcast(i, shape) for i in parents], shape)
        if name in _WINDOWED or name == "reduce_window":
            kind, swapped = (_WINDOWED[name], False) if name in _WINDOWED else \
                self.monoid(name, params)
            if len(parents) != (1 if name in _WINDOWED else 2):
                raise Unsupported(f"{name}: only one operand is imported")
            if name == "reduce_window":
                init = parents[1]
            else:
                # The identities jax gives these primitives: 0, and the type's least and greatest
                # values (infinities for floating types).
                bounds = np.iinfo(dtype) if _is_integer(self.current_dtype) else None
                least = -math.inf if bounds is None else bounds.min
                greatest = math.inf if bounds is None else bounds.max
                init = self.scalar({"add": 0, "max_elem": least, "min_elem": greatest}[kind],
                                   dtype, [])
            return self.windows(parents[0], init, params, shape, kind, swapped)
        if name == "reduce":
            # lax.reduce with a body jax does not recognize (a lambda around lax.add, say). Its
            # documentation requires an identity init and an associative body and leaves any other
            # result undefined, so the init must be the identity constant (+0, 1, -inf, +inf). The
            # result is then the IR reduce, from that identity in row-major order.
            kind, _ = self.monoid(name, params)
            if len(parents) != 2 or self.current_dtype not in _FLOATING:
                raise Unsupported("reduce: only one floating operand is imported")
            operand, init = parents
            identity = {"add": 0.0, "mul_elem": 1.0, "max_elem": -math.inf,
                        "min_elem": math.inf}[kind]
            node = self.nodes[init]
            if node["kind"] != "const" or node["bits"] != bits(np.asarray(identity, dtype=dtype)):
                raise Unsupported("reduce: the documentation leaves a result undefined unless the "
                                  "init value is the body's identity")
            axes = sorted(int(a) for a in params["dimensions"])
            if not axes:
                return operand
            if kind in ("max_elem", "min_elem") and any(self.shape(operand)[a] == 0 for a in axes):
                return self.broadcast(init, shape, [])
            op = {"add": "sum", "mul_elem": "prod", "max_elem": "max", "min_elem": "min"}[kind]
            return self.node("reduce", [operand], shape, op=op, dims=axes, keepdim=False)
        if name == "symmetric_product":
            # alpha (a a^T) + beta c, the formula lax.linalg.symmetric_product documents and its
            # CPU lowering computes, on the whole matrix (the documentation guarantees the lower
            # triangle), alpha and beta rounded once to the dtype as jax converts them.
            if self.current_dtype not in ("float32", "float64"):
                raise Unsupported("symmetric_product: only binary32 and binary64 operands are "
                                  "imported")
            a, c = parents
            rank = len(shape)
            swap = list(range(rank - 2)) + [rank - 1, rank - 2]
            product = self.node("matmul", [a, self.permute(a, swap)], shape)
            return self.node("add", [
                self.node("mul_elem", [self.scalar(params["alpha"], dtype, shape), product], shape),
                self.node("mul_elem", [self.scalar(params["beta"], dtype, shape), c], shape)],
                shape)
        if name == "reduce_precision":
            return self.reduce_precision(parents[0], params, shape, dtype)
        if name == "triangular_solve":
            return self.triangular_solve(parents[0], parents[1], params, shape)
        if name == "cholesky":
            return self.cholesky(parents[0], shape)
        if name == "cholesky_update":
            return self.cholesky_update(parents[0], parents[1], shape, dtype)
        if name == "sign":
            return self.sign(parents[0], shape, dtype)
        if name in _SCANS:
            return self.scan(name, parents[0], params, shape, dtype)
        if name == "atan2":
            if self.current_dtype not in _FLOATING:
                raise Unsupported("atan2: only floating operands are imported")
            return self.widened("atan2", [self.broadcast(i, shape) for i in parents], shape)
        if name == "clamp":
            # lax.clamp(lo, x, hi) is the IR clamp(x, lo, hi): minimum (maximum x lo) hi, and of
            # integers the IR maximum and minimum in the type's order.
            low, x, high = [self.broadcast(i, shape) for i in parents]
            if _is_integer(self.current_dtype):
                return self.node("min_elem", [self.node("max_elem", [x, low], shape), high],
                                 shape)
            if self.current_dtype not in _FLOATING:
                raise Unsupported("clamp: only floating operands are imported")
            return self.node("clamp", [x, low, high], shape)
        if name == "pow":
            return self.widened("pow", [self.broadcast(i, shape) for i in parents], shape)
        if name == "copy":
            return parents[0]
        if name == "sqrt":
            # lax.sqrt is IEEE squareRoot, correctly rounded in each format: Op.sqrt at `.raw`.
            scalar = self.nodes[parents[0]]["dtype"]
            if scalar not in _FLOATING:
                raise Unsupported("sqrt: only floating operands are imported")
            raw = op_table(self.checker)["codes"]["sqrt"]["raw"]
            return self.registered("sqrt", [{"shape": shape}, {"nat": raw}], parents, shape, scalar)
        if name == "square":
            return self.node("mul_elem", [parents[0], parents[0]], shape)
        if name in binary:
            left, right = [self.broadcast(i, shape) for i in parents]
            if name == "div" and _is_integer(self.current_dtype):
                # Integer lax.div rounds toward zero, as C does.
                return self.node("div", [left, right], shape, rounding_mode="trunc")
            return self.node(binary[name], [left, right], shape)
        if name == "integer_pow":
            return self.integer_pow(parents[0], int(params["y"]), shape, dtype)
        if name == "reshape":
            index = parents[0]
            if params.get("dimensions") is not None:
                index = self.permute(index, list(params["dimensions"]))
            return self.reshape(index, shape)
        if name == "squeeze":
            return self.reshape(parents[0], shape)
        if name == "transpose":
            return self.permute(parents[0], list(params["permutation"]))
        if name == "broadcast_in_dim":
            return self.broadcast(parents[0], shape, list(params["broadcast_dimensions"]))
        if name in ("reduce_or", "reduce_and", "reduce_xor"):
            # Boolean operands (lax applies these bitwise to integers). Each counts the true
            # entries, the false ones for reduce_and, by a binary64 sum and compares the count, or
            # for reduce_xor its parity, with zero. The sum is exact below 2^53 terms, and an empty
            # fiber gives true for reduce_and and false for the others.
            source = parents[0]
            if self.nodes[source]["dtype"] != "bool":
                raise Unsupported(f"{name}: lax.{name} of integer operands is bitwise; only "
                                  "Boolean operands are imported")
            axes = sorted(int(a) for a in params["axes"])
            if not axes:
                return source
            fiber = math.prod(self.shape(source)[axis] for axis in axes)
            if name == "reduce_xor" and fiber >= 2 ** 53:
                raise Unsupported("reduce_xor: a fiber of 2^53 or more entries")
            if name == "reduce_and":
                source = self.node("logical_not", [source], self.shape(source), dtype="bool")
            counted = self.node("custom", [source], self.shape(source), dtype="float64",
                                primitive="cast", rounding="nearestEven")
            total = self.node("reduce", [counted], shape, dtype="float64", op="sum", dims=axes,
                              keepdim=False)
            def scalar(value: float) -> int:
                word = self.node("const", [], [], dtype="float64", bits=bits(np.float64(value)),
                                 values=[value])
                return self.broadcast(word, shape, [])
            if name == "reduce_xor":
                total = self.node("fmod", [total, scalar(2.0)], shape, dtype="float64")
            return self.node("eq" if name == "reduce_and" else "ne", [total, scalar(0.0)], shape,
                             dtype="bool")
        if name in _REDUCTIONS:
            source, axes = self.shape(parents[0]), sorted(int(a) for a in params["axes"])
            if not axes:
                return parents[0]
            if _REDUCTIONS[name] in ("max", "min") and any(source[i] == 0 for i in axes):
                raise Unsupported(f"{name}: the IR maximum and minimum need nonempty axes")
            return self.node("reduce", parents, shape, op=_REDUCTIONS[name], dims=axes,
                             keepdim=False)
        if name == "concatenate":
            return self.node("cat", parents, shape, dim=int(params["dimension"]))
        if name == "slice":
            source = self.shape(parents[0])
            strides = list(params["strides"] or [1] * len(source))
            return self.slice(parents[0], list(params["start_indices"]),
                              list(params["limit_indices"]), strides)
        if name == "pad":
            # lax.pad pads first and then crops the negative amounts, interior padding included.
            index, value = parents
            config = [(int(lo), int(hi), int(inner)) for lo, hi, inner in params["padding_config"]]
            low = [max(0, lo) for lo, _, _ in config]
            high = [max(0, hi) for _, hi, _ in config]
            interior = [inner for _, _, inner in config]
            if any(inner < 0 for inner in interior):
                raise Unsupported("pad: interior padding must be nonnegative")
            source = self.shape(index)
            padded = [lo + hi + n + max(0, n - 1) * inner
                      for n, lo, hi, inner in zip(source, low, high, interior)]
            if any(low + high + interior):
                index = self.node("pad", [index, value], padded, low=low, high=high,
                                  interior=interior)
            starts = [max(0, -lo) for lo, _, _ in config]
            stops = [n - max(0, -hi) for n, (_, hi, _) in zip(padded, config)]
            return self.slice(index, starts, stops, [1] * len(source))
        if name == "rev":
            dims = sorted(int(d) for d in params["dimensions"])
            return self.node("flip", parents, shape, dims=dims) if dims else parents[0]
        if name == "tile":
            index, reps = parents[0], [int(r) for r in params["reps"]]
            source = self.shape(index)
            source = [1] * (len(reps) - len(source)) + source
            interleaved = [extent for pair in zip([1] * len(source), source) for extent in pair]
            tiled = [extent for pair in zip(reps, source) for extent in pair]
            return self.reshape(self.broadcast(self.reshape(index, interleaved), tiled), shape)
        if name == "iota":
            # An input-independent index tensor: the value lax.broadcasted_iota computes.
            value = jax.lax.broadcasted_iota(params["dtype"], tuple(shape),
                                             int(params["dimension"]))
            return self.constant(np.asarray(value), jax.core.ShapedArray(tuple(shape), dtype))
        if name == "conv_general_dilated":
            return self.convolution(parents, params, shape)
        if name == "dot_general":
            return self.dot(*parents, params, shape)
        if name == "convert_element_type":
            # One cast, rounding the exact value once (IEEE 754-2019 5.4.1, 5.4.2), also from an
            # integer or binary64 source to binary16 or bfloat16, which XLA's CPU backend rounds
            # twice, through binary32. To an integer type a floating value truncates toward zero,
            # as the documented static_cast does.
            index, source = parents[0], self.nodes[parents[0]]["dtype"]
            if source == self.current_dtype:
                return index
            rounding = "towardZero" if _is_integer(self.current_dtype) else "nearestEven"
            return self.node("custom", [index], shape, primitive="cast", rounding=rounding)
        raise Unsupported(f"Jaxpr import: unsupported primitive {name}")


def lower(closed_jaxpr: Any,
          checker: str | Path | Sequence[str] | None = None) -> dict[str, Any]:
    """Lower a closed Jaxpr to a capture document, rejecting unsupported semantics.

    Every artifact step keeps its own dtype. A conversion is one cast step that rounds once, also
    from an integer or binary64 value to binary16 or bfloat16, which XLA's CPU backend rounds twice,
    through binary32. Binary, Boolean and integer constants keep their exact words. ``div``,
    ``pow``, ``neg``, ``rsqrt`` and ``erf`` become IR pointwise operators with their JAX meaning,
    not XLA's rewrites of them. Integer ``div`` rounds toward zero, integer ``add``, ``sub`` and
    ``mul`` wrap modulo ``2 ** width``, and integer comparisons, ``max``, ``min`` and ``clamp``
    follow the type's order. Binary16 and bfloat16 ``pow``, ``rsqrt`` and ``erf`` are computed in
    binary32 and rounded once, as on XLA's CPU backend. ``integer_pow`` becomes the products XLA
    computes. ``sqrt`` is IEEE squareRoot, ``Op.sqrt`` at ``.raw``. Complex values and control flow
    are rejected. Custom JVP wrappers contribute their forward body only, not a proof of their
    derivative rule, and custom VJP, checkpointing and composite calls are inlined the same way.
    Nondefault dot precision and algorithm choices are rejected rather than erased.

    Several primitives become one general operator with their axes as arguments. ``reduce_sum``,
    ``reduce_prod``, ``reduce_max`` and ``reduce_min`` become one ``reduce``, ``dot_general`` one
    ``contract`` over its ordered axis pairs, ``broadcast_in_dim`` one ``broadcast``, ``slice`` one
    ``slice`` with its strides, ``rev`` one ``flip`` and ``concatenate`` one ``cat``. ``pad`` pads
    by the nonnegative amounts, interior entries included, and then crops the negative ones, as
    ``lax.pad`` does. ``split`` is one slice per output and ``iota`` a constant. A
    ``conv_general_dilated`` with a constant kernel in the NC..., OI... and NC... layouts becomes an
    IR convolution holding that kernel. ``reduce_window_sum``, ``reduce_window_max``,
    ``reduce_window_min`` and ``reduce_window`` with one add, mul, max or min as its body (any init
    value, floating or integer operands) pad the operand with the init value. They fold one strided
    slice per window cell into the init value, in row-major cell order, with the IR's add, mul_elem
    or IEEE max_elem and min_elem. ``jax.nn.softmax`` and ``jax.nn.log_softmax`` import as JAX's own
    primitives. ``cbrt`` is IEEE ``rootn(x, 3)`` and ``nextafter`` C's. ``reduce_precision`` is
    StableHLO's (round to the mantissa width, overflow to an infinity and flush below the normal
    range of the exponent width, a NaN unchanged), two casts through a FloatLib format and a range
    step. The bitwise operators of integers, the shifts, ``population_count``, ``clz`` and ``mulhi``
    are the IR's integer operators (a shift by the width or more shifts every bit out).
    ``bitcast_convert_type`` between types of one width reads each word at the new type.
    ``triangular_solve`` is forward substitution on the equivalent lower left-side system and
    ``cholesky`` the left-looking column algorithm, NaN everywhere unless every pivot is positive.
    ``cholesky_update`` becomes JAX's own Givens rotations, ``symmetric_product`` its formula
    ``alpha (a a^T) + beta c`` (binary32 and binary64), and ``custom_linear_solve`` the forward
    value of its solve jaxpr. ``sign`` keeps ``-0`` and NaN, as ``lax.sign`` documents (a ``where``
    around ``copysign``). ``cumsum``, ``cumprod``, ``cummax`` and ``cummin`` are IR scans: left
    folds for the sum and product, running IEEE maximum and minimum for the extrema. ``top_k`` is
    the stable ``topk``, equal entries in input order, its execution failing where a NaN or both
    zeros could decide the order (XLA ranks in totalOrder). An ``accuracy`` request of a
    transcendental primitive is refused.

    ``gather``, ``dynamic_slice``, ``scatter``, ``scatter-add``, ``scatter-sub``, ``scatter-mul``,
    ``scatter-max``, ``scatter-min`` and ``dynamic_update_slice`` of floating operands are each one
    registered gather or scatter at complete int64 coordinates. ``scatter-sub`` sums the negated
    updates. As in XLA's CPU code, ``scatter-mul`` rounds every step in the dtype, and
    ``scatter-max`` and ``scatter-min`` are the IEEE maximum and minimum: a NaN propagates and -0 is
    below +0. Starts are clamped to keep the window inside, as XLA clamps them, except that a
    ``FILL_OR_DROP`` gather gives ``fill_value`` for a start outside and a scatter in a mode other
    than ``CLIP`` drops that start's window. A ``FILL_OR_DROP`` gather converts its indices to int32
    first when every operand axis is shorter than 2^31, as JAX does. Updates are applied in
    row-major order of the index vectors, each window whole. JAX leaves the order of colliding
    updates open, and the tests compare this one with CPU execution on their inputs. A scatter whose
    update computation is not its name's (as from ``.at[...].apply``) and an operand without entries
    are refused.

    The document's ``reference_execution``, ``computation``, ``jit_computations`` and
    ``primitive_index`` record how XLA compiles the function, for the checker's ``compare``. The
    artifact's records follow the op table of ``checker``, a checker as in ``torchlean.verify``.
    """
    jaxpr = closed_jaxpr.jaxpr
    if not jaxpr.invars:
        raise Unsupported("Jaxpr import needs at least one tensor input")
    for var in [*jaxpr.invars, *(v for v in jaxpr.outvars if hasattr(v, "aval"))]:
        kind = _dtype(var.aval)
        if isinstance(kind, dict) and kind.get("kind") == "complex":
            raise Unsupported("Jaxpr import: complex values have no element type in the IR's "
                              "interpreters")
    builder = _Builder(_dtype(jaxpr.invars[0].aval), checker)
    input_ids = [builder.node("input", [], _shape(var.aval), dtype=_dtype(var.aval))
                 for var in jaxpr.invars]
    output_ids = builder.body(closed_jaxpr, input_ids)
    if not output_ids:
        raise Unsupported("Jaxpr import needs at least one tensor output")
    whole = (len(jaxpr.eqns) == 1 and jaxpr.eqns[0].primitive.name in {"jit", "pjit"}
             and list(jaxpr.eqns[0].outvars) == list(jaxpr.outvars))
    artifact, _, origin = _artifact.lower(builder.nodes, input_ids, output_ids, op_table(checker))

    def per_step(key: str) -> list[int | None]:
        # The metadata of the internal node whose lowering emitted each artifact step.
        return [None if node is None else builder.nodes[node].get(key) for node in origin]

    return dict(artifact=artifact, frontend="jaxpr",
                reference_execution="jit" if whole else "eager",
                jit_computations=builder.jit_computations, computation=per_step("computation"),
                primitive_index=per_step("primitive_index"))


def export(function: Callable[..., Any], *examples: Any,
           checker: str | Path | Sequence[str] | None = None) -> dict[str, Any]:
    """Capture positional tensor arguments at their example shapes and lower the Jaxpr.

    Input order is preserved, including unused arguments. Every argument must be a tensor, so a
    nested pytree is refused rather than silently flattened. ``reference_execution`` describes the
    call ``function(*examples)``, and ``references(function, *examples)`` computes the matching
    reference fields. A reference computed another way must set these fields to match it.
    ``checker`` is passed to ``lower``.
    """
    if not examples or any(not hasattr(value, "shape") or not hasattr(value, "dtype")
                           for value in examples):
        raise Unsupported("Jaxpr capture requires one or more positional tensor arguments")
    return lower(jax.make_jaxpr(function)(*examples), checker)


def _jaxpr(body: Any) -> Any:
    """The open Jaxpr of a call body, closed or not."""
    return getattr(body, "jaxpr", body)


def _closed(body: Any) -> Any:
    """A call body as a closed Jaxpr (an open one is closed with no constants)."""
    from jax.extend.core import ClosedJaxpr
    return body if hasattr(body, "consts") else ClosedJaxpr(body, ())


def _calls_jit(jaxpr: Any) -> bool:
    for equation in jaxpr.eqns:
        name = equation.primitive.name
        if name in {"jit", "pjit"}:
            return True
        if name in _TRANSPARENT and _calls_jit(_jaxpr(equation.params[_CALLS[name]])):
            return True
    return False


def evaluate(closed_jaxpr: Any, *arguments: Any) -> list[Any]:
    """Evaluate a closed Jaxpr one primitive at a time, inside nested ``jax.jit`` calls too.

    Each primitive is dispatched on its own, so XLA compiles it as a computation of its own, as
    eager JAX does outside ``jax.jit``, and no rewrite spans two primitives.
    """
    jaxpr = closed_jaxpr.jaxpr
    env: dict[Any, Any] = dict(zip(jaxpr.constvars, closed_jaxpr.consts))
    env.update(zip(jaxpr.invars, arguments))

    def read(atom: Any) -> Any:
        return atom.val if hasattr(atom, "val") else env[atom]

    for equation in jaxpr.eqns:
        name = equation.primitive.name
        operands = [read(atom) for atom in equation.invars]
        if name in _CALLS:
            outputs = evaluate(_closed(equation.params[_CALLS[name]]), *operands)
        else:
            outputs = equation.primitive.bind(*operands, **equation.params)
            if not equation.primitive.multiple_results:
                outputs = [outputs]
        env.update(zip(equation.outvars, outputs))
    return [read(atom) for atom in jaxpr.outvars]


def _platform(values: list[Any]) -> str:
    platforms = {device.platform for value in values if hasattr(value, "devices")
                 for device in value.devices()}
    return ",".join(sorted(platforms)) if platforms else "unknown"


def references(function: Callable[..., Any], *examples: Any) -> dict[str, Any]:
    """The reference outputs of ``function(*examples)``, with how they were computed.

    ``reference_outputs`` are the outputs' words, ``reference_platform`` the platform of the devices
    that computed them (``"cpu"`` for XLA's CPU backend) and ``jaxlib_version`` the version of
    jaxlib. When ``jax.jit`` compiles any part of the function, the result also holds
    ``eager_reference_outputs``: the outputs of ``evaluate``, one primitive at a time, which XLA
    computes without ``jax.jit``'s rewrites. Their platform is in ``reference_platform`` too. Merge
    the result into the document that ``export`` writes for the same function and examples.
    """
    import jaxlib

    outputs = jax.tree.leaves(function(*examples))
    record: dict[str, Any] = dict(reference_outputs=[encode(value) for value in outputs],
                                  jaxlib_version=jaxlib.__version__)
    closed = jax.make_jaxpr(function)(*examples)
    computed = list(outputs)
    if _calls_jit(closed.jaxpr):
        eager = evaluate(closed, *examples)
        record["eager_reference_outputs"] = [encode(value) for value in eager]
        computed += eager
    record["reference_platform"] = _platform(computed)
    return record
