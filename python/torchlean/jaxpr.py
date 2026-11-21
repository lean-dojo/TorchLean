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
For an artifact without control flow, the document's ``computation`` and ``primitive_index``
have one entry per artifact step. In an eager function, a step's ``computation`` is the
primitive, the nested ``jax.jit`` call, the ``cond``, the ``scan`` or the ``while`` loop that XLA
compiles its equation in, and ``jit_computations`` lists the computations that are nested calls,
conds, scans or loops. A step's ``primitive_index`` is the primitive that produced it, counted
inside nested calls too, for a reference computed one primitive at a time. An entry is null
where there is no such equation or primitive, as for an input.
``TorchLean/NN/IR/README.md`` gives the measured differences.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
import itertools
import math
from pathlib import Path
from typing import Any, Callable

import jax
import numpy as np

from . import _artifact, _flow
from ._jax_complex import ComplexImport
from ._jax_contraction import ContractionImport
from ._jax_order import OrderImport
from ._linalg import LinalgImport
from ._jax_reference import (_WHILE_TRIPS, call_body, evaluate as evaluate, foldable, known_value,
                             moved, references as references, tested, trips, unknowns)
from ._table import op_table

DType = str | dict[str, Any]
_REDUCED_FLOATS = ("bfloat16", "float16")
# Call primitives, whose bodies the importer inlines, with the parameter holding each body
# (call_body): eval_jaxpr is the call of a traced function (jax.stages.Traced), linear_call's body
# is its function, not its transpose, remat_opt's is a custom VJP's forward function, whose outputs
# are the residuals and then the primal outputs, and call_hi_primitive's is its primitive's
# expand, as with the jax_remat3, jax_custom_vjp3 and jax_custom_jvp3 options.
_CALLS = {"jit": "jaxpr", "pjit": "jaxpr", "custom_jvp_call": "call_jaxpr",
          "custom_vjp_call": "call_jaxpr", "remat2": "jaxpr", "composite": "call_jaxpr",
          "eval_jaxpr": "call_jaxpr", "linear_call": "callee", "remat_opt": "fwd_jaxpr",
          "call_hi_primitive": "_prim"}
# Calls that run their body one primitive at a time outside jax.jit, not as one computation.
_TRANSPARENT = ("custom_jvp_call", "custom_vjp_call", "remat2", "composite", "eval_jaxpr",
                "linear_call", "remat_opt", "call_hi_primitive")
# Primitives that XLA compiles as one computation when they run eagerly, with their bodies.
_COMPILED = ("jit", "pjit", "cond", "scan", "while")
# The JAX platforms that the IR's executions stand for: its CPU interpreters and its CUDA path.
_TARGETS = ("cpu", "cuda")
# Primitives that return their operands unchanged: copies, placements on devices, shardings and
# layouts, and checkpoint_name's name, which only checkpoint policies read.
_IDENTITIES = ("copy", "device_put", "optimization_barrier", "sharding_constraint", "reshard",
               "layout_constraint", "name")
# Primitives refused, with the reason.
_REFUSED = {
    "rng_bit_generator": "its bits are a function of its key operand, and TorchLean's random "
                         "operators take their key at capture",
    "unreachable": "it raises its error when evaluated",
    "relayout": "JAX 0.11.1 neither evaluates nor lowers it (both raise NotImplementedError)",
    "create_token": "a token orders effects and holds no value, and effects are not imported",
    "after_all": "a token orders effects and holds no value, and effects are not imported",
    "to_edtype": "extended element types, such as typed random keys, are not IR element types",
    "from_edtype": "extended element types, such as typed random keys, are not IR element types",
}
# Why a value held unreduced (_unreduced) is refused.
_UNREDUCED = ("a value held unreduced, as parts on several devices that a collective reduces, is "
              "not imported: it belongs with the collectives (DistLean)")
_REDUCTIONS = {"reduce_sum": "sum", "reduce_prod": "prod", "reduce_max": "max",
               "reduce_min": "min"}
# Elementary functions computed, for binary16 and bfloat16, in binary32 and rounded once.
_WIDENED_UNARY = ("expm1", "log1p", "exp2", "log2", "tan", "sinh", "cosh", "asinh", "acosh",
                  "atanh", "asin", "acos", "atan")
# Functions whose results are exact in the operand's floating format.
_EXACT_UNARY = ("floor", "ceil")
# Special functions that are members of TorchLean's SpecialFn, or their derivatives, with each
# member's torch.special name (SpecialFn.torchName, betainc for lax.betainc) and its orders in its
# arguments, the fields of Op.special's codec record. igamma_grad_a is orders [1, 0] of igamma
# (SpecialFn.deriv_igamma).
_SPECIAL = {"erfc": ("erfc", [0]), "erf_inv": ("erfinv", [0]), "lgamma": ("gammaln", [0]),
            "digamma": ("digamma", [0]), "bessel_i0e": ("i0e", [0]), "bessel_i1e": ("i1e", [0]),
            "igamma": ("gammainc", [0, 0]), "igammac": ("gammaincc", [0, 0]),
            "zeta": ("zeta", [0, 0]), "igamma_grad_a": ("gammainc", [1, 0]),
            "regularized_incomplete_beta": ("betainc", [0, 0, 0])}
_FLOATING = ("float64", "float32", "bfloat16", "float16")
# Comparisons of floating, integer or Boolean operands, is_finite of floating operands, and the
# logical operators of Boolean operands, with their IR kinds. Each gives a Boolean.
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
# The IR kind of each step of an integer reduction or scan (``_Builder.wrapping``).
_INTEGER_FOLDS = {"reduce_sum": "add", "cumsum": "add", "reduce_prod": "mul_elem",
                  "cumprod": "mul_elem", "reduce_max": "max_elem", "cummax": "max_elem",
                  "reduce_min": "min_elem", "cummin": "min_elem"}
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


def _effects(holder: Any) -> list[Any]:
    """The effects of a Jaxpr or an equation but ``dce_sink``'s (``NoDCEEffect``), which only
    keeps its operand from being removed as dead code."""
    return [effect for effect in getattr(holder, "effects", ())
            if type(effect).__name__ != "NoDCEEffect"]


def _unreduced(var: Any) -> bool:
    """Whether a variable is held unreduced (``jax.sharding.PartitionSpec``'s ``unreduced``
    axes): as parts on the devices of a mesh axis, whose sum, maximum or minimum is its value."""
    spec = getattr(getattr(var.aval, "sharding", None), "spec", None)
    return bool(getattr(spec, "unreduced", None))


# The 8-bit formats of FloatLib's catalog, by their NumPy names (ml_dtypes, _artifact._FLOAT8).
_FLOAT8 = _artifact._FLOAT8
# The unary primitives that a float8 operand reaches through binary32 (_Builder.binary32).
_FLOAT8_UNARY = ("sign", "sqrt", "floor", "ceil", "round")
# The elementary functions that a float8 operand reaches through binary32 too: binary32's function
# of the converted operand, rounded once to the format (_Builder.binary32).
_FLOAT8_ELEMENTARY = ("exp", "exp2", "expm1", "log", "log1p", "sin", "cos", "tan", "tanh",
                      "logistic", "sinh", "cosh", "asinh", "acosh", "atanh", "asin", "acos",
                      "atan", "rsqrt", "cbrt")
# The functions of two operands that float8 operands reach through binary64: binary64's function
# of the converted operands, rounded once to the format (_Builder.binary64).
_FLOAT8_PAIRS = ("atan2", "pow")
# The indexed reads and writes of float8 values, through binary32 (_Builder.indexed); those that
# sum or multiply are _FLOAT8_SCATTERS.
_FLOAT8_INDEXED = ("gather", "dynamic_slice", "dynamic_update_slice", "scatter", "scatter-max",
                   "scatter-min")
# The scatters that sum or multiply, which the IR's scatter computes in the format, each update
# applied in turn, in the row-major order of the index vectors, and rounded once
# (_Builder.indexed). JAX documents the order of colliding updates as implementation-defined, and
# the scatter as possibly nondeterministic (jax.numpy.ndarray.at), so this order is one admissible
# refinement, TorchLean's choice; XLA's CPU code follows it in the test below. A float8
# select_and_scatter_add selects its cells in binary32, which holds each value and keeps their
# order, and sums the source values so, in the format (_Builder.select_scatter).
_FLOAT8_SCATTERS = ("scatter-add", "scatter-sub", "scatter-mul", "select_and_scatter_add")
# The reductions and scans that select an entry, which a float8 operand reaches through binary32
# (_Builder.binary32); those that sum or multiply are _FLOAT8_FOLDS.
_FLOAT8_SELECTIONS = ("reduce_max", "reduce_min", "argmax", "argmin", "cummax", "cummin")
# rem, which a float8 operand reaches through binary32 too (_Builder.binary32): C's fmod is exact
# in every format.
_FLOAT8_EXACT = ("rem",)
# The window reductions, folds of the IR's add, mul_elem, max_elem or min_elem in the format from
# the init value (_Builder.windows).
_FLOAT8_WINDOWS = ("reduce_window_sum", "reduce_window_max", "reduce_window_min", "reduce_window")
# The primitives of a format's words: nextafter, the IR's in the format, bitcast_convert_type to and
# from the 8-bit integers and the other float8 formats, and iota, each index rounded once.
_FLOAT8_WORDS = ("nextafter", "bitcast_convert_type", "iota")
# The reductions and scans that sum or multiply, and integer_pow, which a float8 operand reaches as
# the IR's reduce, scans and mul_elem in the format, each partial result rounded once; the
# reciprocal of a negative power is binary32's, rounded once.
_FLOAT8_FOLDS = ("reduce_sum", "reduce_prod", "cumsum", "cumprod", "integer_pow")
# Other float8 operators and movement, together with the groups above. Operations without a
# float8 IR form convert their operands exactly to binary32 and a floating result back once
# (binary32). Reductions, scans and colliding scatter updates keep the IR's evaluation order;
# contractions widen to binary32 first. These choices can differ from XLA's CPU kernels,
# including their intermediate rounding, NaN ordering and subnormal handling. The IR guide
# records those differences; the float8 cases in ir_capture_jax.py check their exact words.
_FLOAT8_PRIMITIVES = ("convert_element_type", "scaled_dot", "reshape", "squeeze", "transpose",
                      "broadcast_in_dim", "concatenate", "stack", "slice", "pad", "rev", "copy",
                      "split", "unstack", "abs", "add", "add_any", "sub", "mul", "square", "div",
                      "max", "min", "neg", "select_n", "eq", "ne", "lt", "le", "gt", "ge",
                      "eq_to", "lt_to", "le_to", "is_finite", "clamp", "sort", "top_k",
                      "approx_top_k", "reduce_precision",
                      *_FLOAT8_UNARY, *_FLOAT8_ELEMENTARY, *_FLOAT8_PAIRS, *_FLOAT8_INDEXED,
                      *_FLOAT8_SELECTIONS, *_FLOAT8_EXACT,
                      *_FLOAT8_WINDOWS, *_FLOAT8_WORDS, *_FLOAT8_FOLDS, *_FLOAT8_SCATTERS)


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
    if dtype.name in _FLOAT8:
        return _artifact._float8(dtype.name)
    if dtype.kind == "b":
        return "bool"
    if dtype.kind in {"i", "u"}:
        return dict(kind="integer", signed=dtype.kind == "i", width=8 * dtype.itemsize)
    if dtype.kind == "c" and dtype.itemsize in (8, 16):
        return dict(kind="complex", component="float32" if dtype.itemsize == 8 else "float64")
    raise Unsupported(f"Jaxpr import: unsupported element type {dtype}")


def _component(aval: Any) -> str | None:
    """The format of the real and imaginary parts of a complex ``aval``, None for other types."""
    kind = _dtype(aval)
    return kind["component"] if isinstance(kind, dict) and kind.get("kind") == "complex" else None


def bits(value: Any) -> list[int]:
    """Exact element words: Boolean 0/1, or consecutive complex real/imaginary words."""
    array = np.asarray(value)
    if array.dtype.kind == "b":
        return [int(value) for value in array.reshape(-1)]
    width = array.dtype.itemsize // 2 if array.dtype.kind == "c" else array.dtype.itemsize
    unsigned = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}.get(width)
    supported = (array.dtype.name in {"float16", "bfloat16", "float32", "float64", *_FLOAT8}
                 or array.dtype.kind in {"i", "u"}
                 or array.dtype.kind == "c" and array.dtype.itemsize in (8, 16))
    if unsigned is None or not supported:
        raise Unsupported(f"no binary word encoding for {array.dtype}")
    words = array.reshape(-1).view(np.dtype(unsigned).newbyteorder(array.dtype.byteorder))
    return [int(word) for word in words]


def _is_integer(dtype: DType) -> bool:
    return isinstance(dtype, dict) and dtype.get("kind") == "integer"


# The significand precision and the exponent range of each binary format (IEEE 754-2019 clause
# 3; bfloat16 has binary32's exponents and 8 significand bits).
_PRECISIONS = {"float16": (11, -14, 15), "bfloat16": (8, -126, 127), "float32": (24, -126, 127),
               "float64": (53, -1022, 1023)}


def _holds_flow(closed: Any) -> bool:
    """Whether the jaxpr ``closed``, or one nested in its equations' parameters, has a ``cond``
    or a ``while``, which the import may write as a flow."""
    pending = [getattr(closed, "jaxpr", closed)]
    while pending:
        for equation in pending.pop().eqns:
            if equation.primitive.name in ("cond", "while"):
                return True
            for value in equation.params.values():
                for inner in value if isinstance(value, (tuple, list)) else [value]:
                    inner = getattr(inner, "jaxpr", inner)
                    if hasattr(inner, "eqns"):
                        pending.append(inner)
    return False


def _binary(dtype: DType) -> bool:
    return isinstance(dtype, str) and dtype in _PRECISIONS


def _precision(dtype: DType) -> tuple[int, int, int] | None:
    """The significand precision and the least and greatest exponents of the normal values of a
    binary format: a named format's (``_PRECISIONS``), or a float8 descriptor's, from ``1 - bias``
    to the largest exponent field that is finite at fraction zero less the bias (FloatLib's
    ``FloatFormat.Encoding.maxFiniteExponent``: all ones but one in the IEEE encoding, all ones in
    the others). None for another type."""
    if _binary(dtype):
        return _PRECISIONS[dtype]
    if isinstance(dtype, dict) and dtype.get("kind") == "binary":
        largest = 2 ** dtype["exponent_bits"] - (2 if dtype["encoding"] == "ieee" else 1)
        return dtype["fraction_bits"] + 1, 1 - dtype["bias"], largest - dtype["bias"]
    return None


def _holds(source: DType, target: DType) -> bool:
    """Whether every value of the element type ``source`` is a value of ``target``, so that
    converting to it is exact: a named format whose precision and greatest exponent are at least
    the source format's (``_precision``) and whose smallest subnormal, ``2^(emin - p + 1)``, is no
    larger, so that a source value below its normal range is one of its subnormals; an integer type
    with at least the range; and a format whose precision covers an integer type's magnitude bits.
    Booleans are ``0`` and ``1``. A float8 target holds only itself, as no float8 format holds
    another's values (e5m2's infinities and ``-0`` are not e5m2fnuz's, for one)."""
    if source == target:
        return True
    if source == "bool":
        return _binary(target) or _is_integer(target)
    if _is_integer(source):
        magnitude = source["width"] - source["signed"]
        if _is_integer(target):
            return source["signed"] <= target["signed"] and \
                target["width"] - target["signed"] >= magnitude
        return _binary(target) and magnitude <= _PRECISIONS[target][0]
    if _binary(target) and _precision(source) is not None:
        (p, low, high), (q, bottom, top) = _precision(source), _PRECISIONS[target]
        return p <= q and high <= top and low - p >= bottom - q
    return False


def encode(value: Any) -> dict[str, Any]:
    """Serialize a tensor as JAX reads it (``canonical``), with its canonical element descriptor,
    shape and exact words. With 64-bit mode off, a NumPy float64 array is written as the float32
    array that a traced call computes on. A complex tensor of shape ``s`` is written in the
    importer's paired-real form, the tensor of its parts' format and shape ``s + [2]`` whose last
    axis holds each element's real and imaginary parts (``bits``)."""
    (value,) = canonical([value])
    component = _component(value)
    return dict(dtype=component or _dtype(value), shape=_shape(value) + ([2] if component else []),
                bits=bits(value))


def canonical(values: Sequence[Any]) -> list[Any]:
    """The values with every pytree leaf as a JAX array (``jnp.asarray``): the values a traced call
    computes on. A container keeps its structure.

    With 64-bit mode off, JAX converts a float64, int64 or uint64 array to its 32-bit type (an
    integer modulo ``2 ** 32``), and with it on it keeps the array. A capture traces, serializes
    and runs its reference on these values, so all three see the same words: ``export`` traces
    them (``jax.make_jaxpr`` converts its examples so), ``encode`` writes them and ``references``
    computes on them.
    """
    import jax.numpy as jnp
    return [jax.tree.map(jnp.asarray, value) for value in values]


class _Builder(ComplexImport, ContractionImport, LinalgImport, OrderImport):
    refusal = Unsupported

    def __init__(self, dtype: DType, checker: str | Path | Sequence[str] | None = None):
        self.current_dtype = dtype
        # The caller's checker: the records follow its op table.
        self.checker = checker
        self.nodes: list[dict[str, Any]] = []
        # The computation XLA compiles new nodes in when the function runs eagerly: each
        # primitive is one, and so is each nested jax.jit call, cond, scan and while loop.
        self.computation: int | None = None
        self.computations = 0
        self.jit_computations: list[int] = []
        # The primitive that produces new nodes, counted inside nested calls too: each one is a
        # computation of its own when the function is evaluated one primitive at a time.
        self.primitive_index: int | None = None
        self.primitives = 0
        # The type JAX gives each traced variable, by the internal node holding its value.
        self.declared: dict[int, dict[str, Any]] = {}
        # The nodes holding a complex value of shape s in paired-real form: a tensor of shape
        # s + [2] in the parts' format, the real part at 0 and the imaginary part at 1 of the last
        # axis, as Op.rfft writes and Op.irfft reads it.
        self.paired: set[int] = set()
        # The region being built: the indices of its nodes and its flows (``_flow``), in order.
        self.items: list = []
        # The value of each constant node, by its index, for the indices of conds and the
        # kernels of convolutions, and the equation and operands of each
        # node that ``known_value`` may fold: ``foldable``'s and, for a kernel only, ``moved``'s.
        self.known: dict[int, np.ndarray] = {}
        self.producers: dict[int, tuple[Any, list[int]]] = {}
        self.moved: dict[int, tuple[Any, list[int]]] = {}
        # The iterations of while loops that the import may still count and unroll.
        self.iterations = _WHILE_TRIPS
        # In a scan's trial (``looped``), each time the lowering found a value unknown at capture:
        # the nodes at which its fold stopped (``unknowns``), and whether a nested trial's region
        # read it. None outside a trial.
        self.queries: list[tuple[frozenset[int], bool]] | None = None
        # The variables that the Jaxpr being imported reads, its outputs too where it is nested:
        # ``trial`` looks for a carry known at capture there after a scan.
        self.reads: set[Any] | None = None

    def declare(self, index: int, origin: str, aval: Any) -> None:
        """Declare the node ``index`` to hold the value of ``aval``'s type, a complex one in
        paired-real form: the parts' format at its shape with an appended axis of 2."""
        component = _component(aval)
        self.declared.setdefault(index, dict(
            node=index, origin=origin, source="capture", dtype=component or _dtype(aval),
            shape=_shape(aval) + ([2] if component else [])))

    def node(self, kind: str, parents: list[int], shape: list[int],
             dtype: DType | None = None, **attrs: Any) -> int:
        index = len(self.nodes)
        if self.computation is not None:
            attrs["computation"] = self.computation
        if self.primitive_index is not None:
            attrs["primitive_index"] = self.primitive_index
        self.nodes.append(dict(id=index, kind=kind, parents=parents, shape=shape,
                               dtype=dtype or self.current_dtype, **attrs))
        self.items.append(index)
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

    def constant(self, value: Any, aval: Any) -> int:
        dtype = _dtype(aval)
        complex_type = isinstance(dtype, dict) and dtype["kind"] == "complex"
        floating = dtype in _FLOATING or isinstance(dtype, dict) and dtype["kind"] == "binary"
        python = isinstance(value, (float, complex)) and not isinstance(value, np.generic)
        if python and (isinstance(value, float) and floating or complex_type):
            # A Python float or complex number, as jnp.clip's bounds or the NaN of jnp.var and
            # jnp.median, is NumPy's conversion to its declared floating or complex type, which
            # JAX 0.11.1 lowers it to (_python_scalar_handler in jax/_src/interpreters/mlir.py:
            # np.array(val, aval.dtype)). A NumPy scalar such as np.float64, a subclass of float,
            # has its own handler there, ahead of float's in its MRO, and takes the rule below.
            # It is refused where a part changes its class, a finite one overflowing or an
            # infinity becoming a NaN in a format without one.
            with np.errstate(all="ignore"):
                converted = np.array(value, aval.dtype)
                parts = ([(np.real(value), converted.real.astype(np.float64)),
                          (np.imag(value), converted.imag.astype(np.float64))] if complex_type
                         else [(value, converted.astype(np.float64))])
            for before, after in parts:
                if not (math.isnan(before) and np.isnan(after) or math.isinf(before) and
                        before == after or math.isfinite(before) and np.isfinite(after)):
                    raise Unsupported("Jaxpr import: a Python number overflows or loses its "
                                      "infinity at its declared element type")
            value = converted
        component = _component(aval)
        if component:
            # A complex constant is the pair of its parts, each a constant of the parts' format.
            array, part = np.asarray(value), aval.update(dtype=np.dtype(component))
            index = self.pair(self.constant(array.real, part), self.constant(array.imag, part))
            self.known[index] = array.astype(aval.dtype)
            return index
        array = np.asarray(value)
        if array.dtype.name != np.dtype(aval.dtype).name:
            # Another typed literal, such as a weakly typed integer, may be stored as a Python
            # number or an array of another type. It is accepted when its declared type holds the
            # stored value exactly: an integer within the declared integer type's bounds, otherwise
            # a value other than a NaN that converts back unchanged and does not overflow to an
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
        index = self.node("const", [], _shape(aval), dtype=dtype, **attrs)
        self.known[index] = np.asarray(array, dtype=np.dtype(aval.dtype))
        return index

    def uniform_natural(self, index: int) -> int | None:
        # The natural number at every entry of node `index`, a constant or a broadcast or reshape
        # of one, and None for any other node.
        node = self.nodes[index]
        while node["kind"] in ("broadcast", "reshape"):
            node = self.nodes[node["parents"][0]]
        values = node.get("values", []) if node["kind"] == "const" else []
        if len(set(values)) != 1 or values[0] < 0 or not float(values[0]).is_integer():
            return None
        return int(values[0])

    def int64_constant(self, value: Any, shape: list[int]) -> int:
        array = np.ascontiguousarray(np.broadcast_to(np.asarray(value, dtype=np.int64), shape))
        return self.node("const", [], list(shape), dtype=_INT64, bits=bits(array))

    def flags(self, value: Any) -> int:
        """The Boolean constant of the NumPy array ``value``."""
        array = np.asarray(value, dtype=bool)
        return self.node("const", [], list(array.shape), dtype="bool", bits=bits(array))

    def iota(self, shape: list[int], axis: int, dtype: Any = np.int64) -> int:
        """A constant whose entry at ``c`` is ``c[axis]`` in the recorded dtype."""
        dtype = np.dtype(dtype)
        if dtype.kind in "iu" or dtype.name in _FLOAT8:
            indices = np.arange(shape[axis])
            # Float8 rounds each index once. An index too large for binary32 to
            # represent exactly is already beyond the range of every float8 format.
            if dtype.name in _FLOAT8:
                indices = indices.astype(np.float32)
            view = [shape[axis] if i == axis else 1 for i in range(len(shape))]
            value = np.broadcast_to(indices.astype(dtype).reshape(view), shape)
        else:
            # The importer can run inside jit and after the trace's x64 scope has ended.
            with jax.enable_x64(), jax.ensure_compile_time_eval():
                value = np.asarray(jax.lax.broadcasted_iota(dtype, tuple(shape), axis))
        return self.constant(value, jax.core.ShapedArray(tuple(shape), dtype))

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
        # and a scatter in another mode drops the whole window of such a start. Integer and
        # Boolean values move at their element (Op.Element); an integer scatter sums and multiplies
        # modulo 2^width and takes the maximum and minimum in the type's order.
        operand = parents[0]
        format8 = isinstance(self.current_dtype, dict) and \
            self.current_dtype.get("kind") == "binary"
        if format8 and name not in _FLOAT8_SCATTERS:
            # Float8 values are read and written in binary32, which holds each of them, and
            # converted back, so the words move exactly. A fill value is imported only where the
            # format holds it: JAX rounded 1.125 + 2^-30 once from binary64, to e5m2's 1.25, with
            # and without the x64 flag, and NumPy's conversion (ml_dtypes) through binary32, to 1.
            format8, value = self.current_dtype, params.get("fill_value")
            if value is not None:
                held = float(np.asarray(value).astype(dtype))
                if held != float(value) and not (math.isnan(held) and math.isnan(float(value))):
                    raise Unsupported(f"{name}: a float8 fill value is imported only where the "
                                      "format holds it")
            wide = [self.converted(i, "float32") if self.nodes[i]["dtype"] == format8 else i
                    for i in parents]
            self.current_dtype = "float32"
            result = self.indexed(name, wide, params, shape, np.dtype(np.float32))
            self.current_dtype = format8
            return self.converted(result, format8)
        # A float8 sum or product is the IR's in the format, each update applied in turn and
        # rounded once (_FLOAT8_SCATTERS).
        if name.startswith("scatter") and not _plain_update(name, params):
            raise Unsupported(f"{name}: only a replacement, a sum, a difference, a product, a "
                              "maximum or a minimum of the updates is imported")
        values = [operand] + ([parents[1]] if name == "dynamic_update_slice" else
                              [parents[2]] if name.startswith("scatter") else [])
        source, scalar = self.shape(operand), self.current_dtype
        element = _artifact._element(scalar, op_table(self.checker))
        if element is None or any(self.nodes[value]["dtype"] != scalar for value in values):
            raise Unsupported(f"{name}: only floating, integer and Boolean operands, and updates "
                              "of their type, are imported")
        if element == "bool" and name in ("scatter-max", "scatter-min"):
            # JAX orders Booleans False below True and has no other Boolean scatter but the
            # replacement; the IR's Boolean scatter only replaces. So the maximum and minimum run
            # on the uint8 words 0 and 1, in the type's order, and come back as those that are not
            # 0.
            word, byte = np.dtype(np.uint8), dict(kind="integer", signed=False, width=8)
            wide = [self.node("where", [i, self.scalar(1, word, self.shape(i)),
                                        self.scalar(0, word, self.shape(i))], self.shape(i),
                              dtype=byte) if self.nodes[i]["dtype"] == "bool" else i
                    for i in parents]
            self.current_dtype = byte
            result = self.indexed(name, wide, params, shape, word)
            self.current_dtype = scalar
            return self.node("ne", [result, self.scalar(0, word, shape)], shape, dtype="bool")
        # A record at integer words or Booleans books no scalar, so any binary format serves.
        booked = scalar if element == "scalar" else "float64"
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
                                                  {"element": element}],
                                       [operand, tuples], shape, booked, dtype=scalar)
            code = op_table(self.checker)["codes"]["scatter"][_SCATTER_RULES[name]]
            return self.registered("scatter", [{"shape": source}, {"shape": window},
                                               {"nat": code}, {"nats": every},
                                               {"element": element}],
                                   [operand, tuples, update], shape, booked, dtype=scalar)
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
                # x - u is x + (-u) in IEEE arithmetic and modulo 2^width; the IR has no float8
                # neg, and binary32's is exact.
                updates = (self.binary32("neg", [updates], positions) if format8 else
                           self.node("neg", [updates], positions))
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
                                                {"nats": every}, {"pairs": []},
                                                {"element": element}],
                                     [operand, self.coordinates(shape, terms)], shape, booked,
                                     dtype=scalar)
            if inside is None:
                return result
            # lax.gather binds its default fill value (jax/_src/lax/slicing.py, gather): NaN, the
            # least signed value, the greatest unsigned one or True.
            value = params.get("fill_value")
            value = np.nan if value is None else value
            with np.errstate(invalid="ignore"):
                word = np.asarray(value).astype(dtype)
            if element != "scalar" and word != value:
                raise Unsupported(f"{name}: an integer or Boolean fill value is imported only "
                                  "where the type holds it")
            return self.node("where", [inside, result, self.scalar(word, dtype, shape)], shape)
        code, base = op_table(self.checker)["codes"]["scatter"][_SCATTER_RULES[name]], operand
        if inside is not None and name == "scatter-mul":
            # A dropped update becomes the product's identity, so the padded slice's zero entries
            # and its zero cotangent meet finite factors only (0 * inf would give a NaN pullback).
            kept = [inside, updates, self.scalar(1, dtype, positions)]
            updates = (self.binary32("where", kept, positions) if format8 else
                       self.node("where", kept, positions))
        if inside is not None:
            # A dropped window writes into a slice appended along axis 0, removed afterwards.
            padded = [source[0] + 1] + source[1:]
            base = self.node("pad", [operand, self.scalar(0, dtype, [])], padded,
                             low=[0] * len(source), high=[1] + [0] * (len(source) - 1),
                             interior=[0] * len(source))
        extents = self.shape(base)
        written = self.registered("scatter", [{"shape": extents}, {"shape": positions},
                                              {"nat": code}, {"nats": every},
                                              {"element": element}],
                                  [base, self.coordinates(positions, terms, inside, source[0]),
                                   updates], extents, booked, dtype=scalar)
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

    def round_away(self, x: int, shape: list[int], dtype: Any) -> int:
        """``lax.round``'s default rounding method, ``AWAY_FROM_ZERO``, which takes of two integers
        at the same distance from ``x`` the one farther from 0 (``jax.lax.RoundingMethod``). The
        IR's ``round`` takes the even one, so this is ``t + copysign(1, x)`` where
        ``|x - t| >= 1/2`` and ``t`` elsewhere, for ``t`` the IR's ``trunc`` of ``x``, and each step
        whose value reaches the result is exact. For the format's precision p, ``x`` is an integer
        where ``|x| >= 2^(p - 1)``, and there ``t = x`` and ``where`` takes ``t``: the sum beside it
        can round (``2^24 + 1`` in binary32) but is not taken. Elsewhere ``x - t`` is a multiple of
        ``x``'s spacing no larger than ``|x|`` in magnitude, so a value of the format, and ``t + 1``
        and ``t - 1`` are integers of magnitude at most ``2^(p - 1)``. A zero result has ``x``'s
        sign, as ``t`` has, and at an infinity or a NaN, where ``x - t`` is a NaN, the result is
        ``t``: the infinity, or a NaN."""
        t = self.node("trunc", [x], shape)
        fraction = self.node("abs", [self.node("sub", [x, t], shape)], shape)
        away = self.node("ge", [fraction, self.scalar(0.5, dtype, shape)], shape, dtype="bool")
        step = self.node("copysign", [self.scalar(1, dtype, shape), x], shape)
        return self.node("where", [away, self.node("add", [t, step], shape), t], shape)

    def wrapping(self, name: str, x: int, axes: list[int], reverse: bool = False) -> int:
        """The integer reduction or scan ``name`` of ``x`` along ``axes``, in ``x``'s type: each
        step is the IR's ``add``, ``mul_elem``, ``max_elem`` or ``min_elem`` (``_INTEGER_FOLDS``),
        which wrap modulo ``2 ^ width`` or follow the type's order. Each is associative and
        commutative, so every grouping and order of the steps gives the left fold's value. StableHLO
        leaves integer overflow implementation-defined (its "Errors" section), so wrapping is the
        importer's choice, as for ``add``, and XLA's CPU code at JAX 0.11.1 wraps in every case the
        tests check. A reduction halves each axis in turn, combining its first half with its second
        and carrying an odd last entry, down to one entry, which it keeps as an axis of length 1; an
        empty axis gives the identity, 0, 1 or the type's least or greatest value. A scan, along one
        axis, combines in round ``k`` each entry with the one ``2^k`` before it, or after it for
        ``reverse``, the identity past the end (Hillis and Steele's scan), so that after
        ``ceil(log2 n)`` rounds each entry holds its prefix's fold, or its suffix's."""
        dtype = self.nodes[x]["dtype"]
        word = np.dtype(f"{'' if dtype['signed'] else 'u'}int{dtype['width']}")
        op = _INTEGER_FOLDS[name]
        identity = {"add": 0, "mul_elem": 1, "max_elem": np.iinfo(word).min,
                    "min_elem": np.iinfo(word).max}[op]

        def part(x: int, axis: int, start: int, stop: int) -> int:
            extents = self.shape(x)
            return self.slice(x, [start if d == axis else 0 for d in range(len(extents))],
                              [stop if d == axis else e for d, e in enumerate(extents)],
                              [1] * len(extents))

        def joined(pieces: list[int], axis: int) -> int:
            extents = list(self.shape(pieces[0]))
            extents[axis] = sum(self.shape(piece)[axis] for piece in pieces)
            return self.node("cat", pieces, extents, dtype=dtype, dim=axis)

        for axis in axes:
            extents = list(self.shape(x))
            n = extents[axis]
            if name.startswith("cum"):
                step = 1
                while step < n:
                    seed = self.scalar(identity, word, extents[:axis] + [step] + extents[axis + 1:])
                    shifted = joined([part(x, axis, step, n), seed] if reverse else
                                     [seed, part(x, axis, 0, n - step)], axis)
                    x = self.node(op, [x, shifted], extents, dtype=dtype)
                    step *= 2
                continue
            if n == 0:
                x = self.scalar(identity, word, extents[:axis] + [1] + extents[axis + 1:])
            while n > 1:
                half = n // 2
                pair = self.node(op, [part(x, axis, 0, half), part(x, axis, half, 2 * half)],
                                 extents[:axis] + [half] + extents[axis + 1:], dtype=dtype)
                x = pair if n % 2 == 0 else joined([pair, part(x, axis, 2 * half, n)], axis)
                n = half + n % 2
        return x

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
        ``-0`` entry for ``cummin``).

        A float8 ``cumsum`` or ``cumprod`` (``_FLOAT8_FOLDS``) is the IR's scan in the format, each
        partial result rounded once. Along up to 16 entries of every format XLA's CPU code gives
        those words forward; reversed, it folds each suffix first to last, where this scan, between
        two flips, folds it last to first, as in the other formats
        (scripts/checks/ir_capture_jax.py, test_float8_arithmetic_rounds_once_in_the_format). A
        float8 ``cummax`` or ``cummin`` (``_FLOAT8_SELECTIONS``) is this scan of the operand's
        binary32 values, converted back exactly (``binary32``). An integer scan is ``wrapping``'s,
        in the operand's type."""
        target = self.current_dtype
        axis = int(params["axis"])
        if _is_integer(target):
            return self.wrapping(name, x, [axis], bool(params.get("reverse")))
        float8 = isinstance(target, dict) and target.get("kind") == "binary"
        if target not in _FLOATING and not (float8 and name in _FLOAT8_FOLDS):
            raise Unsupported(f"{name}: only floating operands are imported")
        if params.get("reverse"):
            x = self.node("flip", [x], shape, dims=[axis])
        result = self.node(name, [x], shape, dim=axis)
        if name in ("cummax", "cummin"):
            # `flag` marks a `+0` entry (`-0` for cummin), `seen` whether one came at or before
            # each position, and `fixed` the zero results to replace with that zero.
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

    def cumlogsumexp(self, x: int, params: dict[str, Any], shape: list[int], dtype: Any) -> int:
        """``cumlogsumexp`` as JAX lowers it: the ``reduce_window`` along ``axis`` over the prefix
        of each entry (its suffix for ``reverse``), the window the axis's extent and the low (high)
        padding one less, from ``-inf``, with lax's ``logaddexp`` as the body:
        ``max(a, b) + log1p(exp(-|a - b|))``, and ``a + b`` where ``a - b`` is NaN. As in
        ``windows``, each window is folded from ``-inf`` in row-major order. The body never
        returns ``-0``, its logarithm being at least ``+0``, and leaves every other value
        unchanged at a ``-inf`` cell, so the result is the left fold of the prefix, or of the
        suffix from its first entry. XLA's CPU code folds in that order along an axis of up to 17
        entries; along a longer one it combines blocks of 16 entries, which can round
        differently."""
        if self.current_dtype not in _FLOATING:
            raise Unsupported("cumlogsumexp: only floating operands are imported")
        axis, rank = int(params["axis"]), len(shape)
        extent = shape[axis]
        reach = (0, extent - 1) if params.get("reverse") else (extent - 1, 0)
        window = dict(window_dimensions=[extent if a == axis else 1 for a in range(rank)],
                      window_strides=[1] * rank, window_dilation=[1] * rank,
                      base_dilation=[1] * rank,
                      padding=[reach if a == axis else (0, 0) for a in range(rank)])

        def logaddexp(a: int, b: int) -> int:
            difference = self.node("sub", [a, b], shape)
            undecided = self.node("ne", [difference, difference], shape, dtype="bool")
            spread = self.node("neg", [self.node("abs", [difference], shape)], shape)
            tail = self.operation("log1p", [self.operation("exp", [spread], {}, shape, dtype)],
                                  {}, shape, dtype)
            return self.node("where", [undecided, self.node("add", [a, b], shape),
                                       self.node("add", [self.node("max_elem", [a, b], shape),
                                                         tail], shape)], shape)

        return self.windows(x, self.scalar(-np.inf, dtype, []), window, shape, logaddexp)

    def uniform(self, low: int, high: int, shape: list[int]) -> int:
        """``rng_uniform`` with TorchLean's generator, not XLA's stateful one, whose stream the API
        leaves to the implementation: ``Op.randUniform`` under the key of its node's position, as
        the PyTorch capture keys ``rand`` (``_artifact._random_key``), each entry ``u`` in
        ``[0, 1)`` rounded once to the dtype, then ``low + (high - low) * u`` in the dtype's
        arithmetic. Where ``high - low`` overflows it is ``2 * (low / 2 + (high / 2 - low / 2) *
        u)``, whose halvings and doubling are exact at such bounds. Rounding can reach ``high``:
        ``u`` rounds to 1 with probability ``2^-25`` in binary32, and ``high - low`` can round up,
        as for the bounds ``-2^25`` and ``3``. An entry at ``high`` or above is therefore
        ``nextafter(high, low)``, the greatest value below ``high``, so that every entry lies in
        ``[low, high)``, as JAX documents (jax/_src/lax/lax.py:9201 in JAX 0.11.1). Execution
        fails unless both bounds are finite and ``low < high`` (JAX leaves ``high <= low``
        undefined; an infinite bound has no uniform law). Integer bounds: ``uniform_integer``."""
        if _is_integer(self.current_dtype):
            return self.uniform_integer(low, high, shape)
        if self.current_dtype not in _FLOATING:
            raise Unsupported("rng_uniform: only floating and integer bounds are imported")
        try:
            key = _artifact._random_key(len(self.nodes), math.prod(shape))
        except NotImplementedError as error:
            raise Unsupported(f"rng_uniform: {error}") from error
        draw = self.registered("randUniform", [{"shape": shape}, {"nat": key}], [], shape,
                               self.current_dtype)
        dtype = jax.numpy.dtype(self.current_dtype)
        def scaled(low: int, high: int) -> int:
            spread = self.broadcast(self.node("sub", [high, low], []), shape, [])
            return self.node("add", [self.broadcast(low, shape, []),
                                     self.node("mul_elem", [spread, draw], shape)], shape)
        half = self.scalar(0.5, dtype, [])
        halved = scaled(*[self.node("mul_elem", [bound, half], []) for bound in (low, high)])
        spread = self.node("sub", [high, low], [])
        value = self.node("where", [self.broadcast(self.node("isfinite", [spread], [],
                                                             dtype="bool"), shape, []),
                                    scaled(low, high),
                                    self.node("add", [halved, halved], shape)], shape)
        top = self.broadcast(high, shape, [])
        below = self.node("nextafter", [top, self.broadcast(low, shape, [])], shape)
        value = self.node("where", [self.node("ge", [value, top], shape, dtype="bool"), below,
                                    value], shape)
        valid = self.node("logical_and", [
            self.node("logical_and", [self.node("isfinite", [bound], [], dtype="bool")
                                      for bound in (low, high)], [], dtype="bool"),
            self.node("lt", [low, high], [], dtype="bool")], [], dtype="bool")
        return self.refusing(self.node("logical_not", [valid], [], dtype="bool"), value)

    def uniform_integer(self, low: int, high: int, shape: list[int]) -> int:
        """``rng_uniform`` with integer bounds of at most 32 bits, in ``[low, high)`` as JAX
        documents, from the same generator: the binary64 draw ``u`` is ``d / 2^32`` exactly for the
        32-bit word ``d`` of ``Op.randUniform``, and each entry is ``low + mulhi(d, n)`` for
        ``n = high - low``, the high half of the 64-bit product ``d * n``, which is
        ``floor(d * n / 2^32)``. That multiply-shift gives each of the ``n`` values ``floor(2^32 /
        n)`` or ``ceil(2^32 / n)`` of the ``2^32`` words, so the law is uniform when ``n`` divides
        ``2^32`` and within one word in ``2^32 / n`` otherwise; ``low <= entry < high`` always.
        Execution fails unless ``low < high``, which JAX leaves undefined. Wider bounds are
        refused: one 32-bit word cannot cover their range."""
        if self.current_dtype["width"] > 32:
            raise Unsupported("rng_uniform: integer bounds wider than 32 bits are refused")
        try:
            key = _artifact._random_key(len(self.nodes), math.prod(shape))
        except NotImplementedError as error:
            raise Unsupported(f"rng_uniform: {error}") from error
        word = dict(kind="integer", signed=False, width=32)
        draw = self.registered("randUniform", [{"shape": shape}, {"nat": key}], [], shape,
                               "float64", dtype="float64")
        valid = self.node("lt", [low, high], [], dtype="bool")
        draw = self.refusing(self.node("logical_not", [valid], [], dtype="bool"), draw)

        def cast(index: int, dtype: DType) -> int:
            return self.node("custom", [index], self.shape(index), dtype=dtype, primitive="cast",
                             rounding="towardZero")

        words = cast(self.node("mul_elem", [draw, self.scalar(2.0 ** 32, np.float64, shape)],
                               shape, dtype="float64"), word)
        low, high = cast(low, _INT64), cast(high, _INT64)
        spread = self.broadcast(cast(self.node("sub", [high, low], [], dtype=_INT64), word),
                                shape, [])
        offset = cast(self.node("mulhi", [words, spread], shape, dtype=word), _INT64)
        return cast(self.node("add", [self.broadcast(low, shape, []), offset], shape,
                              dtype=_INT64), self.current_dtype)

    def refusing(self, failing: int, x: int) -> int:
        """``x``, or where the Boolean scalar ``failing`` holds, a read at index 1 of a one-entry
        tensor, whose range check fails at execution."""
        source, scalar = self.shape(x), self.nodes[x]["dtype"]
        position = self.node("where", [self.reshape(failing, [1]),
                                       self.int64_constant(1, [1]), self.int64_constant(0, [1])],
                             [1], dtype=_INT64)
        read = self.registered("gather", [{"shape": [1]}, {"shape": [1]}, {"nats": [0]},
                                          {"pairs": []}, _artifact._SCALAR],
                               [self.scalar(0, jax.numpy.dtype(scalar), [1]), position], [1],
                               scalar, dtype=scalar)
        return self.node("where", [self.broadcast(failing, source, []),
                                   self.broadcast(self.reshape(read, []), source, []), x], source,
                         dtype=scalar)

    def integer_pow(self, base: Any, exponent: int, one: Callable[[], Any],
                    multiply: Callable[[Any, Any], Any], divide: Callable[[Any, Any], Any]) -> Any:
        """``base`` to the power ``exponent``, multiplied in the order XLA multiplies: ``one()``
        for exponent 0, else square-and-multiply, then ``divide(one(), product)`` if negative.

        The schedule is JAX's ``lax._integer_pow``, which its lowering emits for every exponent but
        1, 2, 3 and -1, whose direct forms ``x``, ``x * x``, ``(x * x) * x`` and ``1 / x`` are the
        same products, as IEEE multiplication commutes, and ``lax.integer_pow`` documents a sequence
        of O(log2 y) multiplications. ``base`` is a node, or the pair of a complex value's parts,
        and ``multiply`` and ``divide`` round at each step, so the result need not be the correctly
        rounded power.
        """
        if exponent == 0:
            return one()
        power, result = abs(exponent), None
        while power > 0:
            if power & 1:
                result = base if result is None else multiply(result, base)
            power >>= 1
            if power > 0:
                base = multiply(base, base)
        return divide(one(), result) if exponent < 0 else result

    def widened(self, kind: str, parents: list[int], shape: list[int], **attrs: Any) -> int:
        """Compute ``kind`` in binary32 and round once, as XLA's CPU backend computes ``pow``,
        ``rsqrt``, ``erf``, contractions and convolutions on binary16 and bfloat16 operands. A
        float8 result, which only a contraction or a convolution reaches here with, is rounded
        once from binary32 to its format too. JAX's ``dot_general`` documentation leaves the
        accumulation type without ``preferred_element_type`` to the backend and names no
        intermediate format. XLA's CPU code rounds a float8 result twice, through binary16,
        which can move a sum that binary32 holds exactly off its correctly rounded value, and
        the importer does not copy that."""
        target = self.current_dtype
        float8 = isinstance(target, dict) and target.get("kind") == "binary"
        if target not in _REDUCED_FLOATS and not float8:
            return self.node(kind, parents, shape, **attrs)
        wide = [self.node("custom", [i], self.shape(i), dtype="float32", primitive="cast",
                          rounding="nearestEven") for i in parents]
        result = self.node(kind, wide, shape, dtype="float32", **attrs)
        return self.node("custom", [result], shape, dtype=target, primitive="cast",
                         rounding="nearestEven")

    def promoted(self, index: int, name: str) -> int:
        """The operand ``index`` of a contraction or convolution in its result type. JAX documents a
        convolution's ``preferred_element_type`` as the type the results are accumulated in and
        returned as, and a ``dot_general``'s as the type it returns and "a hint to the compiler" for
        accumulation. A type that holds every operand value (``_holds``) is reached by one exact
        conversion, after which the contraction runs in it. XLA's CPU code converts the operands so
        too, word for word at the operands we sampled, float8 ones among them, except that it
        contracts bfloat16 operands with a binary32 result unconverted, in another binary32 order. A
        type that does not hold the operands, a narrower format or binary32 for int32, is
        refused."""
        source, target = self.nodes[index]["dtype"], self.current_dtype
        if source == target:
            return index
        if not _holds(source, target):
            raise Unsupported(f"{name}: a {_artifact._describe(target)} result of "
                              f"{_artifact._describe(source)} operands is not an exact "
                              "widening; only exact widenings are imported")
        rounding = "towardZero" if _is_integer(target) else "nearestEven"
        return self.node("custom", [index], self.shape(index), primitive="cast", rounding=rounding)

    def select_scatter(self, parents: list[int], params: dict[str, Any], shape: list[int],
                       dtype: Any) -> int:
        """``select_and_scatter_add``, the transpose of a windowed maximum (``select_prim`` ge) or
        minimum (le), as JAX's default lowering defines it: StableHLO's ``select_and_scatter`` with
        the primitive's padding, whose selection only takes cells of the operand. In each window the
        selected cell is the one that a left fold in row-major order of the window's operand cells
        selects, keeping the selected cell where ``select(selected, cell)`` holds, so the first of
        equal extremes. Each source value is added to the cell its window selects, by one scatter
        from ``+0`` in row-major window order, and the value of a window that holds padding alone is
        dropped. StableHLO selects by ``reduce``'s schedule without init values, a binary tree whose
        leaves, read in order, are the window's operand cells in ascending lexicographic (row-major)
        order. As ``ge`` and ``le`` keep their left argument at a tie, every such tree selects the
        first of equal extremes, and its shape decides only the cell of a window holding a NaN. The
        additions are a ``reduce`` over the source values in row-major window order with ``+0`` init
        values, whose tree decides only their rounding and a zero's sign. XLA's CPU
        select-and-scatter departs from that order at some windows padded on their last axis, whose
        cells it takes in column-major order, so that a tie can select another cell: a 2 x 2 window
        over ``[[0, 1], [1, 0]]`` and one padding column takes ``(1, 0)``, where StableHLO's order
        selects ``(0, 1)``. For CPU and for GPU, both of the importer's targets (``_TARGETS``), JAX
        registers another lowering as a workaround for XLA crashes (``expand_padding=True``,
        jax/_src/lax/windowed_reductions.py): it pads the operand with ``-inf`` for ge and ``+inf``
        for le and gives XLA no padding, so there a padding cell can tie with an infinite extreme or
        follow a NaN and take the source value, which is lost, and JAX's words differ from these on
        both. The importer keeps StableHLO's documented operation rather than that workaround.

        A float8 operand's cells are compared in binary32, which holds each of its values and
        keeps their order, so each window selects the same cell, and the source values are added
        in the format, each sum rounded once, as the IR's float8 scatters do. In the formats
        without infinities JAX's workaround pads with the format's least or greatest finite
        value, which a cell can equal."""
        source, operand = parents
        select = getattr(params["select_prim"], "name", None)
        if select not in ("ge", "le"):
            raise Unsupported("select_and_scatter_add: only the selections ge and le are imported")
        format8 = isinstance(self.current_dtype, dict) and \
            self.current_dtype.get("kind") == "binary"
        if self.current_dtype not in _FLOATING and not format8 or \
                self.nodes[source]["dtype"] != self.current_dtype:
            raise Unsupported("select_and_scatter_add: only floating operands, and a source of "
                              "their type, are imported")
        padding = [(int(lo), int(hi)) for lo, hi in params["padding"]]
        extents = [int(d) for d in params["window_dimensions"]]
        strides = [int(d) for d in params["window_strides"]]
        windows, rank = self.shape(source), len(shape)
        if not shape or 0 in extents or any(lo < 0 or hi < 0 for lo, hi in padding):
            raise Unsupported("select_and_scatter_add: only nonempty windows of a padded operand "
                              "with at least one axis are imported")
        if 0 in windows:
            return self.scalar(0, dtype, shape)
        compared = dtype
        if format8:
            operand, compared = self.converted(operand, "float32"), np.dtype(np.float32)
        identity = np.asarray(-np.inf if select == "ge" else np.inf, dtype=compared)
        padded = [lo + n + hi for (lo, hi), n in zip(padding, shape)]
        cells = self.operation("pad", [operand, self.constant(identity, jax.core.ShapedArray(
            (), identity.dtype))], dict(padding_config=[(lo, hi, 0) for lo, hi in padding]),
            padded, compared)
        chosen, offsets, seen = None, [], np.zeros(windows, dtype=bool)
        for cell in itertools.product(*[range(extent) for extent in extents]):
            # Which windows hold this cell inside the operand rather than in its padding: those
            # take part in the selection, and the others keep what they hold.
            inside = np.ones(windows, dtype=bool)
            for axis, (c, w, s, (lo, _), n) in enumerate(zip(cell, windows, strides, padding,
                                                             shape)):
                position = (np.arange(w) * s + c).reshape([-1 if a == axis else 1
                                                           for a in range(rank)])
                inside = inside & (lo <= position) & (position < lo + n)
            if not inside.any():
                continue
            stops = [c + (w - 1) * s + 1 for c, w, s in zip(cell, windows, strides)]
            value = self.slice(cells, list(cell), stops, strides)
            offset = [self.broadcast(self.int64_constant(c, []), windows, []) for c in cell]
            if chosen is None:
                chosen, offsets, seen = value, offset, inside
                continue
            keep = self.node(select, [chosen, value], windows, dtype="bool")
            if not (seen & inside).all():
                # Keep where the cell is padding, take the first operand cell of a window.
                keep = self.node("logical_or", [
                    self.node("logical_and", [keep, self.flags(seen & inside)], windows,
                              dtype="bool"), self.flags(~inside)], windows, dtype="bool")
            chosen = self.node("where", [keep, chosen, value], windows,
                               dtype=self.nodes[value]["dtype"])
            offsets = [self.node("where", [keep, a, b], windows, dtype=_INT64)
                       for a, b in zip(offsets, offset)]
            seen = seen | inside
        if chosen is None:
            return self.scalar(0, dtype, shape)
        # A window that never saw an operand cell still points at a padding cell, so its value
        # is added there and cropped.
        # Window w's cell at offset c lies at w * stride + c of the padded operand.
        terms = [[self.broadcast(self.int64_constant(np.arange(w) * s, [w]), windows, [axis]),
                  offset] for axis, (w, s, offset) in enumerate(zip(windows, strides, offsets))]
        code = op_table(self.checker)["codes"]["scatter"]["sum"]
        written = self.registered("scatter", [{"shape": padded}, {"shape": windows}, {"nat": code},
                                              {"nats": list(range(rank))}, _artifact._SCALAR],
                                  [self.scalar(0, dtype, padded),
                                   self.coordinates(windows, terms), source],
                                  padded, self.current_dtype)
        return self.slice(written, [lo for lo, _ in padding],
                          [lo + n for (lo, _), n in zip(padding, shape)], [1] * rank)

    def select_gather(self, parents: list[int], params: dict[str, Any], shape: list[int],
                      dtype: Any) -> int:
        """``select_and_gather_add``, the tangent of a windowed maximum (``select_prim`` ge) or
        minimum (le), as JAX lowers it on CPU: one reduce_window over the pairs of operand and
        tangent entries, its init pair ``(-inf, 0)`` for ge and ``(+inf, 0)`` for le, whose body
        keeps the accumulated pair where ``select(accumulated, cell)`` holds and takes the cell
        otherwise. As in ``windows``, the operand's padding and the holes of its base dilation are
        init pairs, and each window is folded from the init pair in row-major cell order, one
        schedule StableHLO allows. The result is the tangent of the first of equal extremes, and
        ``0`` where the extreme is the init value. Every schedule StableHLO allows takes the
        window's entries in this order, with init pairs at positions it chooses, so it gives the
        same tangent except in a window holding a NaN or whose extreme is the init's infinity. XLA's
        CPU loop skips the padding and the holes and, for some padded windows, takes the cells in
        another order, neither of which StableHLO allows, so its result can differ there."""
        tangents, operand = parents
        select = getattr(params["select_prim"], "name", None)
        if select not in ("ge", "le"):
            raise Unsupported("select_and_gather_add: only the selections ge and le are imported")
        if self.current_dtype not in _FLOATING or \
                self.nodes[operand]["dtype"] != self.current_dtype:
            raise Unsupported("select_and_gather_add: only floating tangents, and an operand of "
                              "their type, are imported")
        bound = -np.inf if select == "ge" else np.inf
        chosen, gathered = self.scalar(bound, dtype, shape), self.scalar(0, dtype, shape)
        if 0 in shape:
            return gathered
        keys = self.cells(self.dilated(operand, self.scalar(bound, dtype, []), params), params,
                          shape)
        values = self.cells(self.dilated(tangents, self.scalar(0, dtype, []), params), params,
                            shape)
        for key, value in zip(keys, values):
            keep = self.node(select, [chosen, key], shape, dtype="bool")
            chosen = self.node("where", [keep, chosen, key], shape)
            gathered = self.node("where", [keep, gathered, value], shape)
        return gathered

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
                combine: Callable[[int, int], int]) -> int:
        """StableHLO's reduce_window with the body ``combine``: the operand padded with the init
        value, and each window folded from the init value in row-major order of its cells. The
        operand gets ``base_dilation - 1`` init values between entries, then the low and high
        padding, a negative amount cropping. The fold is one strided slice per cell, combined
        entrywise as ``combine(accumulated, cell)``. StableHLO leaves the fold's tree open, its
        leaves the window's entries in row-major order with init values interspersed. For the IEEE
        maximum and minimum, the tree changes no value but a NaN's word. XLA's CPU loop skips the
        padding and takes the cells of some windows padded on a later axis in another order, which
        can round a sum or product differently."""
        result = self.broadcast(init, shape, [])
        if 0 in shape:
            return result
        for piece in self.cells(self.dilated(operand, init, params), params, shape):
            result = combine(result, piece)
        return result

    def padded(self, operand: int, fill: int, config: list[tuple[int, int, int]]) -> int:
        """``operand`` padded as ``lax.pad`` pads it, ``config`` giving each axis's
        ``(low, high, interior)``: ``interior`` copies of ``fill`` between entries and ``low`` and
        ``high`` of ``fill`` at the ends, then a negative ``low`` or ``high`` cropping."""
        source = self.shape(operand)
        low = [max(0, lo) for lo, _, _ in config]
        high = [max(0, hi) for _, hi, _ in config]
        interior = [inner for _, _, inner in config]
        padded = [lo + hi + n + max(0, n - 1) * inner
                  for n, lo, hi, inner in zip(source, low, high, interior)]
        index = operand
        if any(low + high + interior):
            index = self.node("pad", [operand, fill], padded, dtype=self.nodes[operand]["dtype"],
                              low=low, high=high, interior=interior)
        return self.slice(index, [max(0, -lo) for lo, _, _ in config],
                          [n - max(0, -hi) for n, (_, hi, _) in zip(padded, config)],
                          [1] * len(source))

    def dilated(self, operand: int, fill: int, params: dict[str, Any]) -> int:
        """A reduce_window operand as its windows read it: ``base_dilation - 1`` copies of ``fill``
        between entries, then the low and high padding of ``fill``, a negative amount cropping."""
        interior = [int(d) - 1 for d in params["base_dilation"]]
        if any(d < 0 for d in interior):
            raise Unsupported("reduce_window: base dilation must be positive")
        return self.padded(operand, fill, [(int(lo), int(hi), inner) for (lo, hi), inner
                                           in zip(params["padding"], interior)])

    def cells(self, index: int, params: dict[str, Any], shape: list[int]) -> Iterator[int]:
        """For each window cell in row-major order, the strided slice of the ``dilated`` operand
        ``index`` holding that cell of every window of the result ``shape``."""
        strides = [int(s) for s in params["window_strides"]]
        dilation = [int(d) for d in params["window_dilation"]]
        for cell in itertools.product(*[range(int(w)) for w in params["window_dimensions"]]):
            begin = [c * d for c, d in zip(cell, dilation)]
            yield self.slice(index, begin, [b + (o - 1) * s + 1
                                            for b, o, s in zip(begin, shape, strides)], strides)

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
        signaling NaNs kept), but a float8 one that is reduced is quieted (below).

        A float8 operand is converted to binary32, which holds its values, and rounded there at
        the float8 format's exponents: through the format of one more exponent bit, its bias and
        ``mantissa_bits`` fraction bits in IEEE's encoding, whose subnormals have the float8
        format's spacing at that width and which no rounding overflows. The range step applies
        where ``exponent_bits`` is narrower than the float8 format's exponent field, and the
        result is converted back once. That is exact for a value of the format; a value past its
        largest finite one and an infinity give its infinity or, in a format without one, its
        NaN, and ``-0`` gives ``0`` in a format without it. So ``exponent_bits`` and
        ``mantissa_bits`` at least the format's give the operand. Otherwise a NaN keeps only its
        sign and payload: the IR has no float8 selection, and FloatLib's conversions give a quiet
        NaN (IEEE 754-2019 6.2.3, FloatLib's ``Conversion/Cast/Runtime.lean``), so e5m2's
        signaling ``0x7d`` and ``0xfd`` give ``0x7f`` and ``0xff``. XLA's CPU code instead
        reduces a float8 operand converted to binary16 and converts the result back, so its
        words differ where binary16's subnormal spacing or its range step at ``exponent_bits``
        below 5 applies (``TorchLean/NN/IR/README.md``)."""
        dtype = self.current_dtype
        float8 = isinstance(dtype, dict) and dtype.get("kind") == "binary"
        if dtype not in _FLOATING and not float8:
            raise Unsupported("reduce_precision: only floating operands are imported")
        exponent, mantissa = int(params["exponent_bits"]), int(params["mantissa_bits"])
        binary = dtype if float8 else op_table(self.checker)["dtypes"][dtype]
        width, fraction = binary["exponent_bits"], binary["fraction_bits"]
        if mantissa < 1 or exponent < 1:
            # A width of zero has no last kept bit for "ties to even" (XLA's code rounds such a
            # tie to the even exponent field), and a FloatLib format needs a fraction bit.
            raise Unsupported("reduce_precision: mantissa_bits = 0 is not imported (ties to even "
                              "has no last mantissa bit, and FloatLib formats need one)")
        if mantissa >= fraction and exponent >= width:
            return x
        if float8:
            x, self.current_dtype, numpy = self.converted(x, "float32"), "float32", np.float32
        y = x
        if mantissa < fraction:
            narrow = dict(binary, fraction_bits=mantissa)
            if float8:
                narrow.update(exponent_bits=width + 1, encoding="ieee")
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
        y = self.node("where", [self.node("isnan", [x], shape, dtype="bool"), x, y], shape)
        if float8:
            self.current_dtype = dtype
            return self.converted(y, dtype)
        return y

    def fixed(self, index: int) -> np.ndarray | None:
        """The value at capture of the node ``index``, or None where it is not known: a
        constant's, or one that ``known_value`` folds from constants through the equations that
        ``foldable`` or ``moved`` admits, as ``jnp.convolve`` flips its kernel and
        ``jnp.correlate`` conjugates a complex one."""
        producers = {**self.producers, **self.moved}
        value = known_value(index, self.known, producers)
        if value is None:
            self.asked([index], producers)
        return value

    def asked(self, indices: list[int], producers: dict[int, tuple[Any, list[int]]]) -> None:
        """Note, in a scan's trial, that the lowering found the nodes ``indices`` not all known at
        capture, folded through ``producers``: the nodes at which their folds stop, for ``trial``
        to look back on (``queries``)."""
        if self.queries is not None:
            stops = frozenset().union(*[unknowns(index, self.known, producers)
                                        for index in indices])
            if stops:
                self.queries.append((stops, False))

    def body(self, closed: Any, inputs: list[int]) -> list[int]:
        jaxpr = getattr(closed, "jaxpr", closed)
        constants = getattr(closed, "consts", ())
        if _effects(jaxpr):
            raise Unsupported("Jaxpr import: effects and mutation are not supported")
        if len(jaxpr.invars) != len(inputs) or len(jaxpr.constvars) != len(constants):
            raise Unsupported("Jaxpr import: call input or constant count mismatch")
        if any(_unreduced(var) for var in jaxpr.invars):
            raise Unsupported(_UNREDUCED)
        env = dict(zip(jaxpr.invars, inputs))
        for var, value in zip(jaxpr.constvars, constants):
            env[var] = self.constant(value, var.aval)
        # ``_restore`` restores the enclosing Jaxpr's reads where a trial fails.
        outer, self.reads = self.reads, {atom for equation in jaxpr.eqns
                                         for atom in equation.invars if not hasattr(atom, "val")}
        if outer is not None:
            self.reads.update(atom for atom in jaxpr.outvars if not hasattr(atom, "val"))

        def read(atom: Any) -> int:
            if hasattr(atom, "val"):
                return self.constant(atom.val, atom.aval)
            if atom not in env:
                raise Unsupported(f"Jaxpr import: unbound variable {atom}")
            return env[atom]

        for equation in jaxpr.eqns:
            name = equation.primitive.name
            if _effects(equation):
                raise Unsupported(f"{name}: effects are not supported")
            if any(_unreduced(var) for var in equation.outvars):
                raise Unsupported(f"{name}: {_UNREDUCED}")
            enclosing, primitive = self.computation, self.primitive_index
            if enclosing is None and name not in _TRANSPARENT:
                # A custom JVP wrapper runs the primitives of its body one by one. XLA compiles
                # an eager cond or scan whole, as it compiles a jax.jit call.
                self.computation, self.computations = self.computations, self.computations + 1
                if name in _COMPILED:
                    self.jit_computations.append(self.computation)
            if name not in _CALLS and name != "call_hi_primitive":
                self.primitive_index, self.primitives = self.primitives, self.primitives + 1
            try:
                parents = [read(atom) for atom in equation.invars]
                outputs = self.equation(equation, parents)
            finally:
                self.computation, self.primitive_index = enclosing, primitive
            if len(outputs) != len(equation.outvars):
                raise Unsupported(f"{name}: call output count mismatch")
            if foldable(equation) and outputs[0] not in parents:
                self.producers.setdefault(outputs[0], (equation, parents))
            elif moved(equation) and outputs[0] not in parents:
                self.moved.setdefault(outputs[0], (equation, parents))
            for var, index in zip(equation.outvars, outputs):
                # A complex value is held, and declared, in paired-real form.
                component = _component(var.aval)
                shape = _shape(var.aval) + ([2] if component else [])
                if self.shape(index) != shape or (index in self.paired) != bool(component):
                    raise Unsupported(f"{name}: lowering produced an inconsistent output shape")
                if self.nodes[index]["dtype"] != (component or _dtype(var.aval)):
                    raise Unsupported(f"{name}: lowering produced an inconsistent output dtype")
                self.declare(index, f"{var} = {name}", var.aval)
                env[var] = index
        results = [read(atom) for atom in jaxpr.outvars]
        self.reads = outer
        return results

    def slot(self, like: int) -> int:
        """A node of node ``like``'s type that no block computes: a carried value of a loop inside
        it, or a result of a flow after it. The blocks that read it have it as an input."""
        index = len(self.nodes)
        self.nodes.append(dict(id=index, kind="input", parents=[], shape=self.shape(like),
                               dtype=self.nodes[like]["dtype"]))
        if like in self.paired:
            self.paired.add(index)
        return index

    def region(self, build: Callable[[], list[int]]) -> tuple[list, list[int]]:
        """The nodes and flows that ``build`` adds, as a region of their own, and its results."""
        outer, self.items = self.items, []
        try:
            results = build()
            return self.items, results
        finally:
            self.items = outer

    def cond(self, equation: Any, parents: list[int]) -> list[int]:
        """``cond``: Lean's ``Flow.cond``, which runs only the branch that its index selects, as
        StableHLO's ``case`` selects it: branch ``index`` for ``0 <= index < n`` and branch
        ``n - 1`` otherwise. ``lax.cond`` binds the index ``0`` or ``1`` of its predicate, and
        ``lax.switch`` clamps its index first. Each branch is a region on the operands, so an
        unselected one never runs, and never fails. An index known at capture (``known_value``),
        such as ``platform_index``'s, selects its branch there, and only that branch is
        imported."""
        branches, (index, *operands) = equation.params["branches"], parents
        if self.shape(index) or not _is_integer(self.nodes[index]["dtype"]):
            raise Unsupported("cond: only a scalar integer index is imported")
        known = known_value(index, self.known, self.producers)
        if known is not None:
            k = int(known)
            return self.body(branches[k if 0 <= k < len(branches) else -1], operands)
        self.asked([index], self.producers)
        regions, results = zip(*[self.region(lambda branch=branch: self.body(branch, operands))
                                 for branch in branches])
        outs = [self.slot(j) for j in results[-1]]
        self.items.append(_flow.Cond(index, list(regions), list(results), outs))
        return outs

    def platform(self, equation: Any) -> int:
        """``platform_index``, which ``lax.platform_dependent`` binds for the ``cond`` on its
        branches (jax/_src/lax/control_flow/conditionals.py:1252 in JAX 0.11.1), as
        ``jnp.diagonal`` does for a square matrix: the position in ``platforms`` of the tuple
        that names the platform the computation is lowered for, or else of the default, ``None``,
        as its lowering rule and ``mlir.lower_per_platform`` select it (conditionals.py:1274,
        mlir.py:2516). The IR stands for the computation on each platform of ``_TARGETS``, so
        the index is a constant, which the ``cond`` reads at capture, where they all select one
        branch. Where they select different branches, or one of them none (JAX's lowering then
        fails), it is refused."""
        platforms, rules = equation.params["platforms"], {}
        for k, names in enumerate(platforms):
            for name in (None,) if names is None else names:
                rules[name] = k
        selected = {rules.get(target, rules.get(None)) for target in _TARGETS}
        if len(selected) != 1 or None in selected:
            raise Unsupported(f"platform_index: the platforms {', '.join(_TARGETS)}, on which "
                              f"the IR runs, select different branches of {platforms} or none")
        return self.constant(np.int32(selected.pop()), equation.outvars[0].aval)

    def unrolled(self, equation: Any, parents: list[int]) -> list[int]:
        """``scan``: its body imported once per iteration, as ``lax.scan``'s reference
        implementation loops over it, so that each iteration's counter is a constant, which the
        conds and loops of its body read at capture. Iteration ``i`` reads the constants, the
        carry and entry ``i`` of each scanned operand along its leading axis, ``length - 1 - i``
        when ``reverse`` holds; its new carry is the next iteration's, and its other outputs are
        entry ``i`` of the stacked outputs. The outputs are the last carry and the stacked
        outputs, of ``length`` times the body's size. A complex operand or output is cut and
        stacked in its paired-real form (``paired``), along its leading axis, which is never the
        pair axis. A scan of length zero never runs its body: its outputs are the carry and
        stacked outputs of zeros, without entries. A scan whose iterations would all import
        alike, with a flow in each, is one ``Flow.scan`` instead (``looped``)."""
        params = equation.params
        length = int(params["length"])

        def like(index: int, source: int) -> int:
            # A complex value's pieces, and the stack of its pieces, are complex too.
            if source in self.paired:
                self.paired.add(index)
            return index
        # ft_in groups the operands as constants, carry and scanned operands, and ft_out the
        # body's outputs as carry and stacked outputs, as JAX's _scan_impl reads them.
        fixed, carry, scanned = map(list, params["ft_in"].update(parents).unpack())
        looped = self.looped(equation, fixed, carry, scanned) if length > 1 else None
        if looped is not None:
            return looped
        if length == 0:
            stacked = []
            for var in equation.outvars[len(carry):]:
                zero = self.constant(np.zeros((), var.aval.dtype), var.aval.update(shape=()))
                stacked.append(like(self.broadcast(zero, [*var.aval.shape, *self.shape(zero)]),
                                    zero))
            return [*carry, *stacked]
        pieces: list[list[int]] = [[] for _ in equation.outvars[len(carry):]]
        for i in reversed(range(length)) if params["reverse"] else range(length):
            entries = []
            for x in scanned:
                shape = self.shape(x)
                entry = self.slice(x, [i] + [0] * (len(shape) - 1), [i + 1] + shape[1:],
                                   [1] * len(shape))
                entries.append(like(self.reshape(like(entry, x), shape[1:]), x))
            carry, ys = map(list, params["ft_out"].update(
                self.body(params["jaxpr"], [*fixed, *carry, *entries])).unpack())
            for stacked, y in zip(pieces, ys, strict=True):
                stacked.append(like(self.reshape(y, [1] + self.shape(y)), y))
        if params["reverse"]:
            for stacked in pieces:
                stacked.reverse()
        return [*carry, *[stacked[0] if length == 1 else like(self.node(
            "cat", stacked, [length] + self.shape(stacked[0])[1:],
            dtype=self.nodes[stacked[0]]["dtype"], dim=0), stacked[0]) for stacked in pieces]]

    def looped(self, equation: Any, fixed: list[int], carry: list[int],
               scanned: list[int]) -> list[int] | None:
        """The ``scan`` ``equation`` on the constants ``fixed``, the carry ``carry`` and the scanned
        operands ``scanned`` as Lean's ``Flow.scan`` of ``length`` iterations of one region of its
        body, or None where it is unrolled. That region reads the carry as slots and, where there
        are scanned operands or stacked outputs, a counter ``i`` from 0, which it adds one to: it
        reads entry ``i`` of each scanned operand, ``length - 1 - i`` when ``reverse`` holds, and
        writes its outputs there, into stacked outputs that start at zeros, as ``lax.scan``'s
        reference implementation loops, by the gather and scatter that PyTorch's ``map`` reads and
        writes by (``_flow.entry``). Each unrolled iteration would import as this region does where
        no value known at capture that it reads, carry known before the first iteration or computed
        from such carry, decides how the import lowers what reads it: a cond, a loop, a
        convolution's kernel or a nested scan (``trial``). So the scan is a ``Flow.scan`` where
        that holds and the region holds a flow, of no more nodes than one iteration, as
        ``lax.fori_loop``'s is where its counter decides no cond. Elsewhere, where an iteration's
        counter, or what else it reads, decides its conds and loops at capture, as in a loop over
        the columns of a matrix, or where the capture decides every one, as in each of the decided
        programs of the tests, the scan is unrolled, and the region's nodes are dropped
        (``_restore``). So it is where a scanned operand is known at capture, or where the region
        draws from ``rng_uniform``, whose unrolled iterations draw under keys of their own. A
        complex or float8 scanned operand or stacked output is unrolled too, as is one with no
        element, and a scan of a zero-sized carry, scanned operand or stacked output. The region is
        a trial: where anything in it is refused or fails while it is built, it is dropped and the
        scan is unrolled, so that building it adds no refusal to the unrolled scan's; a region that
        is built and that ``_flow.lower`` then refuses is not retried (no such region is known)."""
        params = equation.params
        stacks = equation.outvars[len(carry):]

        def float8(dtype: DType) -> bool:
            return isinstance(dtype, dict) and dtype.get("kind") == "binary"
        if not _holds_flow(params["jaxpr"]) or any(index in self.paired for index in carry) or any(
                self.fixed(x) is not None or x in self.paired or float8(self.nodes[x]["dtype"])
                for x in scanned) or any(
                _component(var.aval) or float8(_dtype(var.aval)) for var in stacks) or any(
                0 in self.shape(index) for index in [*carry, *scanned]) or any(
                0 in _shape(var.aval) for var in stacks):
            return None
        mark, outer = self._mark(), self.queries
        self.queries = []
        try:
            outs = self.trial(equation, fixed, carry, scanned)
        except Exception:
            outs = None
        queries, self.queries = self.queries, outer
        if outs is None:
            self._restore(mark)
        if outer is not None:
            # Where an enclosing trial's carry is known, a nested trial may decide otherwise
            # wherever it read that carry, so the enclosing trial looks back on its reads too, at
            # the nodes that both hold.
            for stops, _ in queries:
                if earlier := frozenset(index for index in stops if index < mark[0]):
                    outer.append((earlier, True))
        return outs

    def trial(self, equation: Any, fixed: list[int], carry: list[int],
              scanned: list[int]) -> list[int] | None:
        """``looped``'s ``Flow.scan`` of the ``scan`` ``equation``, or None where its region holds
        no flow, draws from ``rng_uniform``, or may import otherwise in an unrolled iteration: where
        the carry known before the first iteration, or carry that each iteration computes from
        such carry and constants alone, would decide a value that the region found unknown at
        capture (``queries``), or one that the import reads after the scan. ``looped`` drops what
        it adds where it gives None or raises."""
        params = equation.params
        length, reverse = int(params["length"]), params["reverse"]
        stacks = equation.outvars[len(carry):]
        counted = bool(scanned or stacks)
        init = ([self.int64_constant(0, [])] if counted else []) + carry + [
            self.scalar(0, var.aval.dtype, _shape(var.aval)) for var in stacks]
        slots = [self.slot(j) for j in init]

        def at(x: int, i: int, update: int | None = None) -> int:
            # Entry i of x along its leading axis, or x with update written there, by the gather
            # and the replacing scatter of PyTorch's map (_flow.entry). A record at integer words
            # or Booleans books no scalar, as in indexed.
            shape, scalar, table = self.shape(x), self.nodes[x]["dtype"], op_table(self.checker)
            element = _artifact._element(scalar, table)
            if element is None:
                raise Unsupported("scan: a row of a type with no element of the IR")
            booked = scalar if element == "scalar" else "float64"
            if update is None:
                return self.registered("gather", _flow.entry(shape, element), [x, i], shape[1:],
                                       booked, dtype=scalar)
            code = table["codes"]["scatter"]["replace"]
            return self.registered("scatter", _flow.entry(shape, element, code), [x, i, update],
                                   shape, booked, dtype=scalar)

        def iteration() -> list[int]:
            counter, state = (slots[0], slots[1:]) if counted else (None, slots)
            i = counter
            if counted and reverse:
                last = self.int64_constant(length - 1, [])
                i = self.node("sub", [last, counter], [], dtype=_INT64)
            new, ys = map(list, params["ft_out"].update(self.body(
                params["jaxpr"], [*fixed, *state[:len(carry)],
                                  *[at(x, i) for x in scanned]])).unpack())
            written = [at(buffer, i, y) for buffer, y in zip(state[len(carry):], ys, strict=True)]
            step = [self.node("add", [counter, self.int64_constant(1, [])], [],
                              dtype=_INT64)] if counted else []
            return [*step, *new, *written]

        carried = slots[counted:counted + len(carry)]
        knowable = {slot for slot, index in zip(carried, carry) if self.fixed(index) is not None}
        first = len(self.nodes)
        body, results = self.region(iteration)
        # The carry that an unrolled iteration may know at capture: the carry known before the
        # first one, and the carry whose new value folds from such carry and constants alone, to a
        # fixed point. An unrolled iteration may lower otherwise wherever the region found a value
        # unknown whose fold stops only at such carry, or where a nested trial's read stops at it.
        producers = {**self.producers, **self.moved}
        stops = [unknowns(index, self.known, producers)
                 for index in results[counted:counted + len(carry)]]
        while wider := {slot for slot, stop in zip(carried, stops) if stop <= knowable} - knowable:
            knowable |= wider
        decided = any(stop & knowable if nested else stop <= knowable
                      for stop, nested in self.queries)
        self.queries.extend((stop, True) for stop in stops if stop)
        drawn = any(node.get("name") == "randUniform" for node in self.nodes[first:])
        # The last carry is known after an unrolled scan where its new value folds so.
        last = [var for var, stop in zip(equation.outvars, stops) if stop <= knowable]
        if all(isinstance(item, int) for item in body) or decided or drawn or any(
                var in self.reads for var in last):
            return None
        outs = [self.slot(j) for j in init]
        self.items.append(_flow.Loop(slots, init, body, results, outs, length=length))
        return outs[1:] if counted else outs

    def _mark(self) -> tuple:
        """What ``_restore`` returns the import to: its nodes and the region's items so far."""
        return (len(self.nodes), len(self.items), self.computations, len(self.jit_computations),
                self.primitives, self.iterations, self.current_dtype, self.reads)

    def _restore(self, mark: tuple) -> None:
        """The import as it was at ``mark``, without the nodes added since, nor what it holds of
        them."""
        size, items, computations, calls, primitives, iterations, dtype, self.reads = mark
        del self.nodes[size:], self.items[items:], self.jit_computations[calls:]
        for table in (self.declared, self.known, self.producers, self.moved):
            for index in [index for index in table if index >= size]:
                del table[index]
        self.paired.difference_update([index for index in self.paired if index >= size])
        self.computations, self.primitives, self.iterations = computations, primitives, iterations
        self.current_dtype = dtype

    def loop(self, equation: Any, parents: list[int]) -> list[int]:
        """``while``. When its predicate reads only values known at capture, constants of the IR
        or integer and Boolean values folded from them (``known_value``), ``trips`` counts its
        iterations by running the predicate and the parts of the body that compute what it reads
        with JAX, one primitive at a time, each an exact integer or Boolean one, within the
        import's share of ``_WHILE_TRIPS``. The body is then imported once per iteration, each
        iteration reading the constants and the previous carry, as the Python loop that gives
        ``lax.while_loop``'s semantics runs it (its docstring,
        jax/_src/lax/control_flow/loops.py:1620 in JAX 0.11.1); the predicate leaves no node,
        and a loop of no iteration, as ``lax.fori_loop(0, 0, ...)``, imports neither. Otherwise
        it is Lean's ``Flow.whileLoop``, which runs its body while its test holds, and fails
        where the test still holds after the artifact's fuel of iterations. The test and the
        body are regions on their constants and the carried values. A loop that runs more than
        the import's share spends it, so that the import's later loops that run an iteration
        are flows too, and the import's document holds at most ``_WHILE_TRIPS`` unrolled
        iterations in all. A dropped trial (``_restore``) gives back what it counted, so the
        counting may run more."""
        params = equation.params
        split = params["cond_nconsts"]
        start = split + params["body_nconsts"]
        init = parents[start:]
        count = trips(equation, [known_value(index, self.known, self.producers)
                                 for index in parents], self.iterations)
        if count is None:
            self.asked([index for index, need in zip(parents, tested(equation)[0]) if need],
                       self.producers)
        if count is not None and count > self.iterations:
            self.iterations = 0
        elif count is not None:
            self.iterations -= count
            carry = list(init)
            for _ in range(count):
                carry = self.body(params["body_jaxpr"], [*parents[split:start], *carry])
            return carry
        slots = [self.slot(j) for j in init]
        test, (holds,) = self.region(
            lambda: self.body(params["cond_jaxpr"], [*parents[:split], *slots]))
        body, results = self.region(
            lambda: self.body(params["body_jaxpr"], [*parents[split:start], *slots]))
        outs = [self.slot(j) for j in init]
        self.items.append(_flow.Loop(slots, init, body, results, outs, test, holds))
        return outs

    def equation(self, equation: Any, parents: list[int]) -> list[int]:
        name, params = equation.primitive.name, equation.params
        if name in _CALLS:
            return self.body(call_body(equation), parents)
        if name == "cond":
            return self.cond(equation, parents)
        if name == "scan":
            return self.unrolled(equation, parents)
        if name == "while":
            return self.loop(equation, parents)
        if name == "platform_index":
            return [self.platform(equation)]
        if name in _IDENTITIES and name != "copy":
            return list(parents)
        if name == "tie":
            # Its second operand, as JAX evaluates and lowers it.
            return [parents[1]]
        if name == "dce_sink":
            # No output: it only keeps its operand from being removed as dead code.
            return []
        if name in ("empty", "empty2"):
            # Zeros: JAX documents lax.empty's values as possibly uninitialized and backend
            # dependent, so zeros are values it may hold. empty2 is its scalar form, which partial
            # evaluation of a cond binds for the residuals of the branches not taken.
            aval = equation.outvars[0].aval
            return [self.constant(np.zeros(_shape(aval), aval.dtype), aval)]
        if name in _REFUSED:
            raise Unsupported(f"{name}: {_REFUSED[name]}")
        if params.get("accuracy") is not None:
            raise Unsupported(f"{name}: an accuracy request is not imported; the IR computes one "
                              "fixed approximation")
        if name == "sort":
            # Real and complex operands alike, a complex one by its parts.
            return self.sort(parents, params)
        if name in ("split", "unstack"):
            # ``unstack`` is a split into pieces of width one along the axis, each without that
            # axis, as JAX lowers it. A complex operand is cut in its paired-real form
            # (``paired``) along its own axes, never the pair axis, into complex pieces.
            shapes = [_shape(var.aval) for var in equation.outvars]
            pair = [2] if parents[0] in self.paired else []
            if name == "split":
                pieces = self.split(parents, params, shapes)
            else:
                pieces = [self.reshape(piece, shape + pair) for piece, shape in zip(self.split(
                    parents, dict(axis=params["axis"], sizes=[1] * len(shapes)), shapes), shapes)]
            if pair:
                self.paired.update(pieces)
            return pieces
        if any(_component(atom.aval) for atom in [*equation.outvars, *equation.invars]):
            return [self.complex(equation, parents)]
        # A contraction or convolution converts float8 operands to a wider result type
        # (``promoted``) or contracts them for a float8 result in binary32 rounded once
        # (``widened``), a ragged one selecting its groups there too (``ragged_dot``).
        contraction = name in ("dot_general", "conv_general_dilated", "ragged_dot_general")
        if name not in _FLOAT8_PRIMITIVES and not contraction and any(
                np.dtype(atom.aval.dtype).name in _FLOAT8
                for atom in [*equation.outvars, *equation.invars]):
            raise Unsupported(f"{name}: float8 values are imported only through "
                              "convert_element_type, scaled_dot, dot_general, "
                              "conv_general_dilated, ragged_dot_general, "
                              "movement, abs, neg, add, sub, mul, square, div, max, min, sign, "
                              "sqrt, floor, ceil, round, exp, exp2, expm1, log, log1p, the "
                              "trigonometric and hyperbolic functions and their inverses, "
                              "logistic, rsqrt, cbrt, atan2, pow, the comparisons, select_n, "
                              "gathers, dynamic slices, scatters, select_and_scatter_add, "
                              "reduce_sum, reduce_prod, reduce_max, reduce_min, argmax, argmin, "
                              "cumsum, cumprod, cummax, cummin, integer_pow, clamp, rem, sort, "
                              "top_k, approx_top_k, the window reductions, nextafter, "
                              "bitcast_convert_type, iota and reduce_precision")
        if name == "custom_linear_solve":
            # The forward value: the solve jaxpr applied to its constants and the right-hand side,
            # as JAX evaluates it. The matvec, the transposed solve and the implicit derivative
            # rule are not used, and the solution is not checked against matvec.
            lengths = params["const_lengths"]
            start = lengths.matvec + lengths.vecmat
            return self.body(params["jaxprs"].solve,
                             parents[start:start + lengths.solve] + parents[sum(lengths):])
        if name in ("top_k", "approx_top_k"):
            return self.top_k(name, parents[0], params,
                              [_shape(var.aval) for var in equation.outvars],
                              _dtype(equation.outvars[1].aval))
        if len(equation.outvars) != 1:
            raise Unsupported(f"{name}: multiple primitive outputs are not supported")
        shape = _shape(equation.outvars[0].aval)
        self.current_dtype = _dtype(equation.outvars[0].aval)
        if name in ("and", "or", "xor", "not") and _is_integer(self.current_dtype):
            return [self.operation(name, parents, params, shape, equation.outvars[0].aval.dtype)]
        if name in _BOOLEAN or name == "select_n":
            return [self.boolean(name, parents, shape)]
        if name in ("eq_to", "lt_to", "le_to"):
            return [self.total(name, parents, shape)]
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
        if name == "ragged_dot_general":
            return [self.ragged_dot(parents, params, shape, equation.outvars[0].aval.dtype,
                                    equation.invars[2].aval.dtype)]
        if name == "scaled_dot":
            return [self.scaled_dot(parents, params, shape)]
        if name == "dot_general":
            parents = [self.promoted(i, name) for i in parents]
        # argmax and argmin return indices of another type than their operand; a convolution
        # converts its input and kernel itself (``convolution``).
        exempt = ("convert_element_type", "argmax", "argmin", "conv_general_dilated")
        if name not in exempt and any(
                self.nodes[i]["dtype"] != self.current_dtype for i in parents):
            raise Unsupported(f"{name}: heterogeneous arithmetic needs an explicit "
                              "operand/accumulator type rule")
        return [self.operation(name, parents, params, shape, equation.outvars[0].aval.dtype)]

    def binary32(self, kind: str, operands: list[int], shape: list[int],
                 params: dict[str, Any] | None = None) -> int:
        """The IR's ``kind`` of float8 ``operands`` computed in binary32, for neg, the
        comparisons, where and div, of which the IR has no float8 form, and the primitive
        ``kind`` of ``_FLOAT8_UNARY``, ``_FLOAT8_SELECTIONS`` or ``_FLOAT8_EXACT`` with its
        ``params``, imported as for a binary32 operand: each operand converted to binary32, which
        holds every value of the four float8 formats, and a result of the operands' type converted
        back. The conversions are exact but for a quotient's and a square root's, so the result is
        ``kind``'s in the float8 format, a NaN's sign and payload aside, and a quotient or a
        square root is the exact one rounded once to the format. The conversion keeps each
        value's order, zero sign and NaN, so a maximum or a minimum is the format's, and so is
        each running one of ``cummax`` and ``cummin`` (``scan``), and ``argmax`` and ``argmin``,
        whose rule reads only the order, the equalities and the NaNs, give the position they give
        in the format. That rule is the IR's, the first NaN or else the first extremum
        (TorchLean/Tensor/ArgMax.lean): PyTorch's documented tie rule and the NaN rule measured on
        torch 2.8.0, not a rule taken from JAX's or XLA's documentation.
        That XLA's CPU ``argmax`` and ``argmin`` follow it on float8 operands is measured, at
        every pair of words of each format and at every word but the NaNs in one row, and so is
        that its ``cummax`` and ``cummin`` give the running IEEE maximum and minimum, at every
        pair of words and in rows of 32 words, NaNs among them, forward and reversed
        (scripts/checks/ir_capture_jax.py, test_float8_arithmetic_rounds_once_in_the_format).
        ``sign`` gives -1, 1, a zero or a NaN, and ``floor``, ``ceil`` and ``round`` give x
        itself where |x| >= 2^(p - 1), for the format's precision p, and otherwise an integer of
        magnitude at most 2^(p - 1), each a value of the format, but a -0 of a format without
        one, which converts to 0. So does ``rem``, C's ``fmod``: x where y is infinite and x
        finite, a NaN where x is infinite, y is 0 or either is a NaN, and otherwise x - n y for
        the integer n of x / y rounded toward zero, of magnitude below |y| and at most |x|, and a
        multiple of the smaller q of the spacings of the format's values at x and at y, so below
        2^p q in magnitude and a value of the format.

        Binary32 holds every quotient of float8 values in its normal range, from 2^-33 to 2^33
        in magnitude, and rounds it within a relative 2^-24, and a quotient is never within a
        relative 2^-23 of a midpoint m = M 2^k of the format (M an odd integer below 2^(p + 1),
        for the format's precision p <= 4, the overflow threshold among them) unless it is m:
        for x = X 2^a and y = Y 2^b, with integers |X| and |Y| below 2^p, x - m y is a multiple
        of 2^min(a, k + b), |m y| < 2^(k + b + 2 p + 1), and |m y| < 2^(a + p + 1) where x / y is
        within a relative 2^-23 of m, so a nonzero x - m y is above a relative 2^-(2 p + 1) of
        m y there. Binary32's quotient is therefore on the exact quotient's side of every
        midpoint, or on it where the exact quotient is. For a square root of x = X 2^a > 0, with
        an integer X below 2^p, a nonzero x - m^2 is a multiple of 2^min(a, 2 k), m^2 is below
        2^(2 k + 2 p + 2), and m^2 < 2 x where x is within a relative 2^-(2 p + 2) of m^2, so a
        nonzero x - m^2 is above a relative 2^-(2 p + 2) of m^2, and sqrt x is m or more than a
        relative 2^-(2 p + 4) from m. Binary32 holds sqrt x, from 2^-9 to 2^8, in its normal
        range and rounds it within a relative 2^-24, so its root too is on the exact root's side
        of every midpoint, or on it where the exact root is.

        An elementary function of ``_FLOAT8_ELEMENTARY`` is the IR's binary32 function of the
        converted operand, rounded once to the format. No argument covers every function here, so
        the claim is checked at each of the 256 words of each format, with mpmath at 200 bits: an
        exact value below twice the format's largest finite value is a value of the format to 150
        bits, more than a relative 2^-17 from every midpoint of the format and from its overflow
        threshold (2^-16.54 at the least, ``logistic`` of -0.0625 in the e4m3 formats), or one of
        two exact midpoints, ``exp2(-10)`` in e4m3fn and ``exp2(-11)`` in e4m3fnuz, between 0 and
        the least subnormal, which binary32 holds; above it every rounding overflows. A binary32
        result within a relative 2^-17 of the exact value, or within 2^-149 of it below binary32's
        normal range, far below every format's least midpoint (2^-18 or more), therefore rounds
        to the exact value rounded once, ties to even. In the reference runtime's binary32, ``exp``,
        ``expm1``, ``log``, ``log1p`` and ``cbrt``, IEEE ``rootn(x, 3)``, are FloatLib's correctly
        rounded functions (TorchLean/Tensor/ConfiguredBinary.lean), ``exp2``, ``tan`` and the
        inverse hyperbolic functions are defined to round the real value once
        (ConfiguredBinary/Elementary.lean), and so are ``asin``, ``acos`` and ``atan``
        (ConfiguredBinary/InverseTrigonometric.lean); ``rsqrt`` is ``1 / sqrt x``, two roundings,
        within a relative 2^-22 (Context.rsqrt in TorchLean/Tensor/Context.lean). ``sin``, ``cos``,
        ``sinh``, ``cosh``, ``tanh`` and ``logistic`` carry no accuracy theorem. The IR's words of
        all of them are checked at every word (scripts/checks/ir_capture_jax.py,
        test_float8_elementary_functions_round_once_through_binary32)."""
        # The last operand is of the float8 type (where's first is its predicate), and every
        # operand of that type is widened.
        dtype = self.nodes[operands[-1]]["dtype"]
        wide = [self.converted(i, "float32") if self.nodes[i]["dtype"] == dtype else i
                for i in operands]
        # Indices and comparisons are not converted back.
        if kind in ("argmax", "argmin"):
            return self.operation(kind, wide, params or {}, shape, None)
        if kind in (*_FLOAT8_UNARY, *_FLOAT8_ELEMENTARY, *_FLOAT8_SELECTIONS, *_FLOAT8_EXACT):
            self.current_dtype = "float32"
            result = self.operation(kind, wide, params or {}, shape, np.dtype(np.float32))
            self.current_dtype = dtype
            return self.converted(result, dtype)
        if kind in ("neg", "where", "div"):
            return self.converted(self.node(kind, wide, shape, dtype="float32"), dtype)
        return self.node(kind, wide, shape, dtype="bool")

    def binary64(self, kind: str, operands: list[int], shape: list[int]) -> int:
        """``atan2`` or ``pow`` (``_FLOAT8_PAIRS``) of float8 ``operands``: the IR's binary64
        function of the operands converted to binary64, which holds every float8 value, rounded
        once to the format. At every pair of words of each format, with mpmath at 200 bits, the
        exact value below twice the format's largest finite value is a value of the format to 150
        bits, more than a relative 2^-38 from every midpoint of the format and from its overflow
        threshold (2^-37.58 at the least, ``atan2(-0.21875, 57344)`` in e5m2fnuz, and 2^-23.82
        for ``pow``, at ``(104, -0.0068359375)`` in e4m3fnuz), or a midpoint, which only powers
        reach; above it every rounding overflows. A binary64 result within a relative 2^-38 of
        the exact value, and exact at a midpoint, therefore rounds to the exact value rounded
        once, ties to even. Binary32 does not suffice: it holds the e5m2 midpoint ``3 * 2^-17``,
        within a relative 2^-32 of ``atan2(3 * 2^-16, 2)``, and rounds that onto it. The IR's
        binary64 ``atan2`` rounds the polar angle once (ConfiguredBinary/InverseTrigonometric.lean).
        Its ``pow`` is FloatLib's ``Model.pow``, correctly rounded for an integral exponent with a
        finite result and where its certified search settles, and otherwise binary64's
        ``exp(y * log x)``, which carries no accuracy theorem. The IR's words are the exact value
        rounded once at every pair (scripts/checks/ir_capture_jax.py,
        test_float8_atan2_and_pow_round_once_through_binary64)."""
        dtype = self.current_dtype
        wide = [self.converted(i, "float64") for i in operands]
        return self.converted(self.node(kind, wide, shape, dtype="float64"), dtype)

    def boolean(self, name: str, parents: list[int], shape: list[int]) -> int:
        # Comparisons and logical operators take operands of one element type. With a Boolean
        # predicate and two cases, select_n(p, false_case, true_case) is the IR
        # where(p, true_case, false_case). Float8 operands are compared and selected in
        # binary32 (``binary32``).
        dtypes = [self.nodes[i]["dtype"] for i in parents]
        operands = [self.broadcast(i, shape) for i in parents]
        float8 = isinstance(dtypes[-1], dict) and dtypes[-1].get("kind") == "binary"
        if name == "select_n":
            if len(parents) != 3 or dtypes[0] != "bool":
                raise Unsupported("select_n: only a Boolean predicate and two cases are imported")
            if dtypes[1] != self.current_dtype or dtypes[2] != self.current_dtype:
                raise Unsupported("select_n: heterogeneous cases need an explicit type rule")
            predicate, false_case, true_case = operands
            if float8:
                return self.binary32("where", [predicate, true_case, false_case], shape)
            return self.node("where", [predicate, true_case, false_case], shape)
        if any(dtype != dtypes[0] for dtype in dtypes):
            raise Unsupported(f"{name}: operands of two element types are not imported")
        if float8:
            return self.binary32(_BOOLEAN[name], operands, shape)
        if name in ("and", "or", "xor", "not"):
            if dtypes[0] != "bool":
                raise Unsupported(f"{name}: lax.{name} of integer operands is bitwise; only "
                                  "Boolean operands are imported")
        elif dtypes[0] == "bool" and name != "is_finite":
            # Booleans, false below true as in NumPy, are compared at 0 and 1 in binary64
            # (``held``).
            operands = [self.held(i) for i in operands]
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
        operand = self.nodes[parents[0]]["dtype"] if parents else None
        if name in ("neg", *_FLOAT8_UNARY, *_FLOAT8_ELEMENTARY, *_FLOAT8_SELECTIONS,
                    *_FLOAT8_EXACT) and \
                isinstance(operand, dict) and operand.get("kind") == "binary":
            return self.binary32(name, parents, shape, params)
        if name == "abs" and _is_integer(self.current_dtype):
            # StableHLO's integer modulus (its abs), -x below 0 modulo 2^width, so the least
            # signed value is its own magnitude, as XLA's CPU code gives it.
            x = parents[0]
            below = self.node("lt", [x, self.scalar(0, dtype, shape)], shape, dtype="bool")
            return self.node("where", [below, self.node("neg", [x], shape), x], shape)
        if name in unary:
            return self.node(unary[name], parents, shape)
        if name == "rem" and _is_integer(self.current_dtype):
            # lax.rem of integers is C's %, x - trunc(x / y) * y modulo 2^width (StableHLO's
            # remainder): 0 at the least signed value and -1, where StableHLO leaves the result
            # implementation-defined, as XLA's CPU code gives it. The IR's integer division fails
            # at a zero divisor, where StableHLO leaves the result implementation-defined too.
            left, right = [self.broadcast(i, shape) for i in parents]
            quotient = self.node("div", [left, right], shape, rounding_mode="trunc")
            return self.node("sub", [left, self.node("mul_elem", [quotient, right], shape)],
                             shape)
        if name in _WIDENED_UNARY + _EXACT_UNARY + ("round", "rem") and \
                self.current_dtype not in _FLOATING:
            raise Unsupported(f"{name}: only floating operands are imported")
        if name in _WIDENED_UNARY:
            return self.widened(name, parents, shape)
        if name in _EXACT_UNARY:
            return self.node(name, parents, shape)
        if name == "round":
            if params["rounding_method"] == jax.lax.RoundingMethod.TO_NEAREST_EVEN:
                return self.node("round", parents, shape)
            return self.round_away(parents[0], shape, dtype)
        if name == "rem":
            # lax.rem of floating operands is C's fmod, exact in the operand format.
            return self.node("fmod", [self.broadcast(i, shape) for i in parents], shape)
        if name in ("argmax", "argmin"):
            # One axis, removed. Of a floating or Boolean operand, Op.argmax and Op.argmin, the
            # first extremum and the first NaN, as a custom node holding the registered operator,
            # in the primitive's index dtype.
            (axis,) = [int(a) for a in params["axes"]]
            source, scalar = self.shape(parents[0]), self.nodes[parents[0]]["dtype"]
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
            if _is_integer(scalar):
                # The first position of the extremum, in trees in the types (``first``).
                return self.reshape(self.first(name, parents[0], axis), shape)
            if scalar == "bool":
                # Booleans at 0 and 1 in binary64, which has no NaN (``held``).
                parents, scalar = [self.held(parents[0])], "float64"
            return self.registered(name, [{"shape": source}, {"nat": axis}, {"bool": False}],
                                   parents, shape, scalar)
        if name in ("rsqrt", "erf"):
            return self.widened(name, parents, shape)
        if name in _SPECIAL or name == "polygamma":
            # One Op.special node. Its meaning is the real function, and no binary format has a
            # rule for it, so a run at one refuses the step, as for erf.
            # How XLA rounds these functions on binary16 and bfloat16 operands is not modeled, so
            # those are refused.
            if self.current_dtype not in ("float32", "float64"):
                raise Unsupported(f"{name}: only binary32 and binary64 operands are imported")
            if name == "polygamma":
                # polygamma(m, x) is the derivative of order m of digamma: digamma at that order,
                # when m holds one constant natural number at every entry, of x broadcast to the
                # result's shape (m may have more entries than x).
                order = self.uniform_natural(parents[0])
                if order is None:
                    raise Unsupported(f"{name}: the order must be one constant natural number")
                special, orders = "digamma", [order]
                operands = [self.broadcast(parents[1], shape)]
            else:
                # At the orders of _SPECIAL, of the operands at the result's shape: lax.betainc's
                # are a, b and x, in that order.
                special, orders = _SPECIAL[name]
                operands = [self.broadcast(i, shape) for i in parents]
            return self.registered("special",
                                   [{"name": special}, {"nats": orders}, {"shape": shape}],
                                   operands, shape, self.current_dtype)
        if name in ("cbrt", "nextafter"):
            # cbrt is IEEE rootn(x, 3), correctly rounded in the operand's own format (a float8
            # one's in binary32, _FLOAT8_ELEMENTARY), and nextafter is C's, exact, also in a float8
            # format, which the op table's forms do not name: there it is Op.nextafter itself.
            operands = [self.broadcast(i, shape) for i in parents]
            if name == "nextafter" and isinstance(self.current_dtype, dict) and \
                    self.current_dtype.get("kind") == "binary":
                return self.registered("nextafter", [{"shape": shape}], operands, shape,
                                       self.current_dtype)
            if self.current_dtype not in _FLOATING:
                raise Unsupported(f"{name}: only floating operands are imported")
            return self.node(name, operands, shape)
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
                # values (false and true for Booleans, infinities for floating types that have
                # them, and otherwise, as for float8_e4m3fn and the fnuz formats, the largest
                # finite magnitude of each sign).
                finite = _FLOAT8.get(np.dtype(dtype).name, (None, None, None, "ieee"))[3] != "ieee"
                bounds = np.iinfo(dtype) if _is_integer(self.current_dtype) else \
                    jax.numpy.finfo(dtype) if finite else None
                least = -math.inf if bounds is None else bounds.min
                greatest = math.inf if bounds is None else bounds.max
                if self.current_dtype == "bool":
                    least, greatest = False, True
                init = self.scalar({"add": 0, "max_elem": least, "min_elem": greatest}[kind],
                                   dtype, [])
            return self.windows(parents[0], init, params, shape, lambda accumulated, cell:
                                self.node(kind, [cell, accumulated] if swapped
                                          else [accumulated, cell], shape))
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
        if name == "householder_product":
            return self.householder_product(parents[0], parents[1], shape)
        if name == "ormqr":
            return self.reflected(name, *parents, bool(params["left"]), bool(params["transpose"]))
        if name == "lu_pivots_to_permutation":
            return self.pivot_permutation(parents[0], shape)
        if name == "tridiagonal_solve":
            if params["perturb_singular"]:
                raise Unsupported("tridiagonal_solve: perturb_singular=True is refused, as its "
                                  "documentation gives the perturbation only to an order")
            return self.tridiagonal_solve(*parents, shape)
        if name == "sign":
            return self.sign(parents[0], shape, dtype)
        if name in _SCANS:
            return self.scan(name, parents[0], params, shape, dtype)
        if name == "cumlogsumexp":
            return self.cumlogsumexp(parents[0], params, shape, dtype)
        if name == "rng_uniform":
            return self.uniform(parents[0], parents[1], shape)
        if name in _FLOAT8_PAIRS and isinstance(self.current_dtype, dict) and \
                self.current_dtype.get("kind") == "binary":
            return self.binary64(name, [self.broadcast(i, shape) for i in parents], shape)
        if name == "atan2":
            if self.current_dtype not in _FLOATING:
                raise Unsupported("atan2: only floating operands are imported")
            return self.widened("atan2", [self.broadcast(i, shape) for i in parents], shape)
        if name == "clamp":
            # lax.clamp(lo, x, hi) is the IR clamp(x, lo, hi): minimum (maximum x lo) hi, the
            # formula of StableHLO's clamp and of XLA's semantics, where lax.clamp's docstring
            # gives cases that differ at a NaN bound, at x = -0 and where lo > hi. Of integers it
            # is the IR maximum and minimum in the type's order, and of float8 values the IR's
            # maximum and minimum in the format.
            low, x, high = [self.broadcast(i, shape) for i in parents]
            if _is_integer(self.current_dtype) or isinstance(self.current_dtype, dict) and \
                    self.current_dtype.get("kind") == "binary":
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
            if name == "div" and isinstance(self.current_dtype, dict) and \
                    self.current_dtype.get("kind") == "binary":
                return self.binary32("div", [left, right], shape)
            return self.node(binary[name], [left, right], shape)
        if name == "integer_pow":
            # The IR has no float8 div, so a float8 reciprocal is binary32's, rounded once.
            float8 = isinstance(self.current_dtype, dict) and \
                self.current_dtype.get("kind") == "binary"
            return self.integer_pow(parents[0], int(params["y"]),
                                    lambda: self.scalar(1, dtype, shape),
                                    lambda x, y: self.node("mul_elem", [x, y], shape),
                                    lambda x, y: self.binary32("div", [x, y], shape) if float8
                                    else self.node("div", [x, y], shape))
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
        if name in ("reduce_max", "reduce_min") and self.nodes[parents[0]]["dtype"] == "bool":
            # The Boolean maximum and minimum, false below true, with the identities false and
            # true of an empty fiber, are reduce_or and reduce_and.
            name = "reduce_or" if name == "reduce_max" else "reduce_and"
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
            if _is_integer(self.current_dtype):
                return self.reshape(self.wrapping(name, parents[0], axes), shape)
            if _REDUCTIONS[name] in ("max", "min") and any(source[i] == 0 for i in axes):
                raise Unsupported(f"{name}: the IR maximum and minimum need nonempty axes")
            return self.node("reduce", parents, shape, op=_REDUCTIONS[name], dims=axes,
                             keepdim=False)
        if name == "concatenate":
            return self.node("cat", parents, shape, dim=int(params["dimension"]))
        if name == "stack":
            # Each operand given an axis of size one at the new axis, then all joined along it, as
            # JAX lowers it.
            axis = int(params["axis"])
            expanded = shape[:axis] + [1] + shape[axis + 1:]
            return self.node("cat", [self.reshape(i, expanded) for i in parents], shape, dim=axis)
        if name == "slice":
            source = self.shape(parents[0])
            strides = list(params["strides"] or [1] * len(source))
            return self.slice(parents[0], list(params["start_indices"]),
                              list(params["limit_indices"]), strides)
        if name == "pad":
            config = [(int(lo), int(hi), int(inner)) for lo, hi, inner in params["padding_config"]]
            if any(inner < 0 for _, _, inner in config):
                raise Unsupported("pad: interior padding must be nonnegative")
            return self.padded(*parents, config)
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
            return self.iota(shape, int(params["dimension"]), dtype)
        if name == "select_and_scatter_add":
            return self.select_scatter(parents, params, shape, dtype)
        if name == "select_and_gather_add":
            return self.select_gather(parents, params, shape, dtype)
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


def lower(closed_jaxpr: Any, checker: str | Path | Sequence[str] | None = None,
          fuel: int = _flow.FUEL) -> dict[str, Any]:
    """Lower a closed Jaxpr to a typed capture document.

    Every step keeps its own dtype and shape. Constants retain their exact words, and conversions
    round once to the declared target type. Integer arithmetic wraps at its word width, division
    rounds toward zero, and Boolean arithmetic uses the IR's Boolean rules. A complex tensor of
    shape ``s`` is represented by its real and imaginary parts at shape ``s + [2]``.

    Floating execution follows the IR's arithmetic and evaluation order. Binary16 and bfloat16
    contractions accumulate in binary32 and round once; a ``preferred_element_type`` is accepted
    only when it holds every operand value. Complex arithmetic uses the paired-real operations
    of ``ComplexImport``. Float8 conversions and intermediate formats follow
    ``_FLOAT8_PRIMITIVES`` and the corresponding builder methods. These choices do not assert
    agreement with every XLA kernel: reduction order, fused operations, subnormal flushing and
    approximations can change its results. The operator coverage and measured differences are
    documented in ``TorchLean/NN/IR/README.md``. Special-function nodes may be imported for
    real-valued reasoning even when their floating-point execution has no rule.

    Capture-time partial evaluation uses constants and exact integer or Boolean operations.
    A known ``cond`` index selects only that branch. A decided ``while`` loop is unrolled;
    otherwise it becomes ``Flow.whileLoop``. A ``scan`` is unrolled when its iterations need
    distinct imports, or represented by one ``Flow.scan`` when its body imports uniformly.
    Each executed while loop has its own budget of ``fuel`` iterations each time it is entered;
    it fails if its test still holds after that budget. A scan runs its declared length, and
    only the selected branch of a condition runs.
    The import itself shares a separate unrolling budget of ``_WHILE_TRIPS`` iterations.

    Custom JVP, custom VJP, checkpoint and composite calls contribute their forward bodies.
    Importing one does not prove its custom derivative rule. Effects, unsupported primitives,
    nondefault contraction precision and algorithms without an arithmetic rule are refused.
    Shape and value checks of the resulting IR remain separate from translation correctness.

    The records use the operator table returned by ``checker``, as in ``torchlean.verify``.
    Straight-line documents include ``computation``, ``jit_computations`` and
    ``primitive_index`` for the checker's comparison with XLA's CPU execution; flow documents
    include only ``reference_execution`` and block-qualified ``declared_types``. Input and
    output order are preserved, including unused inputs.
    """
    fuel = _flow.natural(fuel)
    jaxpr = closed_jaxpr.jaxpr
    if not jaxpr.invars:
        raise Unsupported("Jaxpr import needs at least one tensor input")
    first = jaxpr.invars[0].aval
    builder = _Builder(_component(first) or _dtype(first), checker)
    input_ids = []
    for var in jaxpr.invars:
        component = _component(var.aval)
        index = builder.node("input", [], _shape(var.aval) + ([2] if component else []),
                             dtype=component or _dtype(var.aval))
        if component:
            builder.paired.add(index)
        builder.declare(index, f"input {var}", var.aval)
        input_ids.append(index)
    builder.items.clear()  # the region holds the nodes after the inputs
    output_ids = builder.body(closed_jaxpr, input_ids)
    if not output_ids:
        raise Unsupported("Jaxpr import needs at least one tensor output")
    whole = (len(jaxpr.eqns) == 1 and jaxpr.eqns[0].primitive.name in {"jit", "pjit"}
             and list(jaxpr.eqns[0].outvars) == list(jaxpr.outvars))
    reference_execution = "jit" if whole else "eager"
    flat = _flow.flat(builder.nodes, builder.items, input_ids, output_ids)
    if flat is None:
        artifact, where = _flow.lower(builder.nodes, input_ids, output_ids, builder.items,
                                      op_table(checker), fuel)
        return dict(artifact=artifact, frontend="jaxpr", reference_execution=reference_execution,
                    declared_types=[dict(entry, block=where[index][0], ref=where[index][1])
                                    for index, entry in sorted(builder.declared.items())
                                    if index in where])
    nodes, number = flat
    artifact, refs, origin = _artifact.lower(nodes, [number[j] for j in input_ids],
                                             [number[j] for j in output_ids], op_table(checker))

    def per_step(key: str) -> list[int | None]:
        # The metadata of the internal node whose lowering emitted each artifact step.
        return [None if node is None else nodes[node].get(key) for node in origin]

    return dict(artifact=artifact, frontend="jaxpr", reference_execution=reference_execution,
                jit_computations=builder.jit_computations, computation=per_step("computation"),
                primitive_index=per_step("primitive_index"),
                declared_types=[dict(entry, ref=refs[number[index]])
                                for index, entry in sorted(builder.declared.items())
                                if index in number])


def export(function: Callable[..., Any], *examples: Any,
           checker: str | Path | Sequence[str] | None = None,
           fuel: int = _flow.FUEL) -> dict[str, Any]:
    """Capture positional tensor arguments at their example shapes and lower the Jaxpr.

    Input order is preserved, including unused arguments. Every argument must be a tensor, so a
    nested pytree is refused rather than silently flattened. ``reference_execution`` describes the
    call ``function(*examples)``, and ``references(function, *examples)`` computes the matching
    reference fields. A reference computed another way must set these fields to match it.
    ``checker`` and ``fuel`` are passed to ``lower``, and a ``fuel`` that ``_flow.natural``
    refuses is refused before the capture.
    """
    fuel = _flow.natural(fuel)
    if not examples or any(not hasattr(value, "shape") or not hasattr(value, "dtype")
                           for value in examples):
        raise Unsupported("Jaxpr capture requires one or more positional tensor arguments")
    return lower(jax.make_jaxpr(function)(*examples), checker, fuel)
