"""Lower static tensor Jaxpr computations to TorchLean's checked graph format.

Use ``export(function, *examples)`` or ``lower(closed_jaxpr)``, and ``references`` for the
reference outputs that ``compare`` checks. Constants are included in the artifact; no Python
interpreter is needed to load it in Lean. This adapter does not certify the translation or
prescribe XLA's floating-point reduction order.

Each primitive keeps its JAX meaning: ``lax.div`` is IEEE division, ``rsqrt`` is the IR's
``1 / sqrt x`` rounded twice, and results underflow gradually. XLA's compiled CPU code departs
from that meaning: it multiplies by the reciprocal of a broadcast divisor, approximates ``rsqrt``
and flushes subnormal values to zero, and under ``jax.jit`` it also multiplies by the reciprocal
of a constant divisor and rewrites divisions and constant powers. The adapter does not follow it,
but records what the checker's ``compare`` needs to emulate it. ``reference_execution`` is
``"jit"`` when the function is one ``jax.jit`` call, so that calling it runs one compiled program,
and ``"eager"`` otherwise. In an eager function, each node's ``computation`` is the primitive, or
the nested ``jax.jit`` call, that XLA compiles it in, and ``jit_computations`` lists the nested
calls. Each node's ``primitive_index`` is the primitive that produced it, counted inside nested
calls too, for a reference computed one primitive at a time. ``NN/IR/README.md`` gives the
measured differences.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import jax
import numpy as np

DType = str | dict[str, Any]
_REDUCED_FLOATS = ("bfloat16", "float16")
# Call primitives, whose bodies the importer inlines.
_CALLS = ("jit", "pjit", "custom_jvp_call")


class Unsupported(ValueError):
    """The captured computation needs semantics absent from the current IR."""


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
    def __init__(self, dtype: DType):
        self.dtype = dtype
        self.current_dtype = dtype
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

    def shape(self, index: int) -> list[int]:
        return self.nodes[index]["shape"]

    def constant(self, value: Any, aval: Any) -> int:
        dtype = _dtype(aval)
        array = np.asarray(value)
        if array.dtype.name != np.dtype(aval.dtype).name:
            raise Unsupported("Jaxpr import: constant storage differs from its declared element type")
        attrs = dict(bits=bits(value))
        # The homogeneous diagnostic loader also accepts finite native float values.
        # Other types retain only their exact words; never round large integers through Float.
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
        return self.node("reshape", [index], shape, dtype=self.nodes[index]["dtype"],
                         in_shape=source, out_shape=shape)

    def permute(self, index: int, axes: list[int]) -> int:
        source = self.shape(index)
        if sorted(axes) != list(range(len(source))):
            raise Unsupported(f"invalid axis permutation {axes} for {source}")
        if axes == list(range(len(source))):
            return index
        return self.node("permute", [index], [source[i] for i in axes],
                         dtype=self.nodes[index]["dtype"], perm=axes)

    def broadcast(self, index: int, shape: list[int]) -> int:
        source = self.shape(index)
        if source == shape:
            return index
        padded = [1] * (len(shape) - len(source)) + source
        if len(padded) != len(shape) or any(a not in (1, b) for a, b in zip(padded, shape)):
            raise Unsupported(f"invalid broadcast {source} -> {shape}")
        return self.node("broadcast_to", [index], shape, dtype=self.nodes[index]["dtype"],
                         from_shape=source, to_shape=shape)

    def nonnegative(self, index: int) -> bool:
        """Recognize expressions without negative values or negative zero.

        This uses the expression, not the capture inputs. NaNs are not treated as
        finite witnesses; they propagate through the reference's maximum and sqrt.
        """
        node = self.nodes[index]
        kind, parents = node["kind"], node["parents"]
        if kind == "const":
            values = node.get("values")
            return values is not None and all(
                value >= 0 and math.copysign(1, value) > 0 for value in values)
        if kind in {"abs", "exp", "relu", "sigmoid", "softplus", "sqrt"}:
            return True
        if kind in {"reshape", "permute", "broadcast_to", "detach", "sum", "reduce_sum"}:
            return self.nonnegative(parents[0])
        if kind == "add":
            return all(self.nonnegative(parent) for parent in parents)
        if kind == "mul_elem":
            return parents[0] == parents[1] or all(self.nonnegative(parent) for parent in parents)
        if kind == "max_elem":
            return any(self.nonnegative(parent) for parent in parents)
        return False

    def integer_pow(self, base: int, exponent: int, shape: list[int], dtype: Any) -> int:
        """Multiply in the order XLA does: square-and-multiply, then 1 / product if negative.

        This is JAX's ``lax._integer_pow`` schedule, which its lowering emits for every exponent;
        the products round at each step, so the result is not a correctly rounded power.
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
        """Expose batch, free, and contracted axes before a shared IR matmul."""
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
        batch = [ls[i] for i in lb]
        m, k, n = (math.prod(ls[i] for i in lf), math.prod(ls[i] for i in lc),
                   math.prod(rs[i] for i in rf))
        left = self.reshape(self.permute(left, lb + lf + lc), batch + [m, k])
        right = self.reshape(self.permute(right, rb + rc + rf), batch + [k, n])
        product = self.node("matmul", [left, right], batch + [m, n])
        expected = batch + [ls[i] for i in lf] + [rs[i] for i in rf]
        if expected != shape:
            raise Unsupported(f"dot_general: expected output {expected}, got {shape}")
        return self.reshape(product, shape)

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
            for var in equation.outvars:
                _dtype(var.aval)
            enclosing, primitive = self.computation, self.primitive_index
            if enclosing is None and name != "custom_jvp_call":
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
            key = "call_jaxpr" if name == "custom_jvp_call" else "jaxpr"
            return self.body(params[key], parents)
        if len(equation.outvars) != 1:
            raise Unsupported(f"{name}: multiple primitive outputs are not supported")
        shape = _shape(equation.outvars[0].aval)
        self.current_dtype = _dtype(equation.outvars[0].aval)
        if name != "convert_element_type" and any(
                self.nodes[i]["dtype"] != self.current_dtype for i in parents):
            raise Unsupported(f"{name}: heterogeneous arithmetic needs an explicit "
                              "operand/accumulator type rule")
        return [self.operation(name, parents, params, shape, equation.outvars[0].aval.dtype)]

    def operation(self, name: str, parents: list[int], params: dict[str, Any],
                  shape: list[int], dtype: Any) -> int:
        unary = {"abs": "abs", "exp": "exp", "log": "log",
                 "sin": "sin", "cos": "cos", "tanh": "tanh", "logistic": "sigmoid",
                 "stop_gradient": "detach", "neg": "neg"}
        # `div` is IEEE division, also for a broadcast or constant divisor, which XLA's CPU code
        # multiplies by the rounded reciprocal instead.
        binary = {"add": "add", "add_any": "add", "sub": "sub", "mul": "mul_elem",
                  "max": "max_elem", "min": "min_elem", "div": "div"}
        if name in unary:
            return self.node(unary[name], parents, shape)
        if name in ("rsqrt", "erf"):
            return self.widened(name, parents, shape)
        if name == "pow":
            return self.widened("pow", [self.broadcast(i, shape) for i in parents], shape)
        if name == "copy":
            return parents[0]
        if name == "sqrt":
            if not self.nonnegative(parents[0]):
                raise Unsupported("sqrt: TorchLean's square root clamps negative inputs; "
                                  "the source argument needs a structurally nonnegative expression")
            return self.node("sqrt", parents, shape)
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
            index = parents[0]
            source, dims = self.shape(index), list(params["broadcast_dimensions"])
            if (len(dims) != len(source) or sorted(set(dims)) != dims
                    or any(i < 0 or i >= len(shape) for i in dims)):
                raise Unsupported("broadcast_in_dim: invalid axis mapping")
            expanded = [1] * len(shape)
            for extent, axis in zip(source, dims):
                expanded[axis] = extent
            return self.broadcast(self.reshape(index, expanded), shape)
        if name == "reduce_sum":
            source = self.shape(parents[0])
            axes = list(params["axes"])
            if sorted(axes) == list(range(len(source))):
                return self.node("sum", parents, [])
            if not axes:
                return parents[0]
            if any(source[i] == 0 for i in axes):
                raise Unsupported("reduce_sum: the IR axis reduction requires a nonempty axis")
            return self.node("reduce_sum", parents, shape, axes=axes, keepdim=False)
        if name == "concatenate":
            return self.node("concat", parents, shape, axis=int(params["dimension"]))
        if name == "dot_general":
            return self.dot(*parents, params, shape)
        if name == "convert_element_type":
            index, source = parents[0], self.nodes[parents[0]]["dtype"]
            if source == self.current_dtype:
                return index
            if self.current_dtype in _REDUCED_FLOATS and (
                    source in ("bool", "float64") or _is_integer(source)):
                # XLA's CPU backend makes binary16 and bfloat16 from binary32, so an integer,
                # Boolean or binary64 source rounds twice.
                index = self.node("custom", [index], shape, dtype="float32", primitive="cast",
                                  rounding="nearestEven")
            return self.node("custom", [index], shape, primitive="cast", rounding="nearestEven")
        raise Unsupported(f"Jaxpr import: unsupported primitive {name}")


def lower(closed_jaxpr: Any) -> dict[str, Any]:
    """Lower a captured tensor computation, rejecting unsupported semantics.

    Every node retains its own dtype; casts become explicit IR operations. Binary,
    Boolean, integer and complex tensors have lossless descriptors and constant words.
    Accepting a descriptor does not supply its execution semantics. Converting an
    integer, Boolean or binary64 value to binary16 or bfloat16 takes two casts, through
    binary32, as XLA's CPU backend converts. ``div``, ``pow``, ``neg``, ``rsqrt`` and
    ``erf`` become IR pointwise operators with their JAX meaning, not XLA's rewrites of
    them; integer ``div`` rounds toward zero. Binary16 and bfloat16 ``pow``, ``rsqrt``
    and ``erf`` are computed in binary32 and rounded once, as on XLA's CPU backend.
    ``integer_pow`` becomes the products XLA computes.
    Square roots need structurally nonnegative operands, since the existing IR clamps
    negative inputs. Control flow is rejected. Custom JVP wrappers contribute their
    forward body only, not a proof of their derivative rule.

    The artifact records how XLA compiles the function for the checker's ``compare``:
    ``reference_execution`` is ``"jit"`` when the Jaxpr is one ``jax.jit`` call, otherwise
    ``"eager"``, with each node's ``computation`` and the nested ``jit_computations``, and each
    node's ``primitive_index`` for a reference computed one primitive at a time.
    """
    jaxpr = closed_jaxpr.jaxpr
    if not jaxpr.invars:
        raise Unsupported("Jaxpr import needs at least one tensor input")
    builder = _Builder(_dtype(jaxpr.invars[0].aval))
    input_ids = [builder.node("input", [], _shape(var.aval), dtype=_dtype(var.aval))
                 for var in jaxpr.invars]
    output_ids = builder.body(closed_jaxpr, input_ids)
    if not output_ids:
        raise Unsupported("Jaxpr import needs at least one tensor output")
    whole = (len(jaxpr.eqns) == 1 and jaxpr.eqns[0].primitive.name in {"jit", "pjit"}
             and list(jaxpr.eqns[0].outvars) == list(jaxpr.outvars))
    return dict(format="torchlean.ir.v2", frontend="jaxpr", dtype=builder.dtype,
                input_ids=input_ids, output_ids=output_ids, nodes=builder.nodes,
                reference_execution="jit" if whole else "eager",
                jit_computations=builder.jit_computations)


def export(function: Callable[..., Any], *examples: Any) -> dict[str, Any]:
    """Capture positional tensor arguments at their example shapes and lower the Jaxpr.

    Input order is preserved, including unused arguments. A nested pytree needs an
    explicit container interface rather than silently flattening the user's call.
    ``reference_execution`` describes ``function(*examples)``: ``"jit"`` for a
    ``jax.jit``-compiled function, ``"eager"`` otherwise. ``references(function, *examples)``
    computes the matching reference fields; a reference computed another way must set them
    accordingly.
    """
    if not examples or any(not hasattr(value, "shape") or not hasattr(value, "dtype")
                           for value in examples):
        raise Unsupported("Jaxpr capture requires one or more positional tensor arguments")
    return lower(jax.make_jaxpr(function)(*examples))


def _calls_jit(jaxpr: Any) -> bool:
    for equation in jaxpr.eqns:
        name = equation.primitive.name
        if name in {"jit", "pjit"}:
            return True
        if name == "custom_jvp_call" and _calls_jit(equation.params["call_jaxpr"].jaxpr):
            return True
    return False


def evaluate(closed_jaxpr: Any, *arguments: Any) -> list[Any]:
    """Evaluate a closed Jaxpr one primitive at a time, inside nested ``jax.jit`` calls too.

    Each primitive is dispatched on its own, so XLA compiles it as a computation of its own, as
    eager JAX does outside ``jax.jit``; no rewrite spans two primitives.
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
            key = "call_jaxpr" if name == "custom_jvp_call" else "jaxpr"
            outputs = evaluate(equation.params[key], *operands)
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

    ``reference_outputs`` are the outputs' words, ``reference_platform`` the platform of the
    devices that computed them (``"cpu"`` for XLA's CPU backend) and ``jaxlib_version`` the
    version of jaxlib. When ``jax.jit`` compiles any part of the function,
    ``eager_reference_outputs`` are also the outputs of ``evaluate``, one primitive at a time,
    which XLA computes without ``jax.jit``'s rewrites; its platform is included in
    ``reference_platform``. Merge the result into the artifact that ``export`` writes for the same
    function and examples.
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
