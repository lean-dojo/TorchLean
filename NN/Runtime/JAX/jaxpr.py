"""Lower static tensor Jaxpr computations to TorchLean's checked graph format.

Use ``export(function, *examples)`` or ``lower(closed_jaxpr)``. Constants are included
in the artifact; no Python interpreter is needed to load it in Lean. This adapter
does not certify the translation or prescribe XLA's floating-point reduction order.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import jax
import numpy as np

DType = str | dict[str, Any]


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


def encode(value: Any) -> dict[str, Any]:
    """Serialize a tensor with its canonical element descriptor, shape and exact words."""
    return dict(dtype=_dtype(value), shape=_shape(value), bits=bits(value))


class _Builder:
    def __init__(self, dtype: DType):
        self.dtype = dtype
        self.current_dtype = dtype
        self.nodes: list[dict[str, Any]] = []

    def node(self, kind: str, parents: list[int], shape: list[int],
             dtype: DType | None = None, **attrs: Any) -> int:
        index = len(self.nodes)
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
            name, params = equation.primitive.name, equation.params
            if getattr(equation, "effects", ()):
                raise Unsupported(f"{name}: effects are not supported")
            for var in equation.outvars:
                _dtype(var.aval)
            parents = [read(atom) for atom in equation.invars]
            if name in {"jit", "pjit", "custom_jvp_call"}:
                key = "call_jaxpr" if name == "custom_jvp_call" else "jaxpr"
                outputs = self.body(params[key], parents)
            else:
                if len(equation.outvars) != 1:
                    raise Unsupported(f"{name}: multiple primitive outputs are not supported")
                shape = _shape(equation.outvars[0].aval)
                self.current_dtype = _dtype(equation.outvars[0].aval)
                if name != "convert_element_type" and any(
                        self.nodes[i]["dtype"] != self.current_dtype for i in parents):
                    raise Unsupported(f"{name}: heterogeneous arithmetic needs an explicit "
                                      "operand/accumulator type rule")
                outputs = [self.operation(name, parents, params, shape,
                                          equation.outvars[0].aval.dtype)]
            if len(outputs) != len(equation.outvars):
                raise Unsupported(f"{name}: call output count mismatch")
            for var, index in zip(equation.outvars, outputs):
                if self.shape(index) != _shape(var.aval):
                    raise Unsupported(f"{name}: lowering produced an inconsistent output shape")
                if self.nodes[index]["dtype"] != _dtype(var.aval):
                    raise Unsupported(f"{name}: lowering produced an inconsistent output dtype")
                env[var] = index
        return [read(atom) for atom in jaxpr.outvars]

    def operation(self, name: str, parents: list[int], params: dict[str, Any],
                  shape: list[int], dtype: Any) -> int:
        unary = {"abs": "abs", "exp": "exp", "log": "log",
                 "sin": "sin", "cos": "cos", "tanh": "tanh", "logistic": "sigmoid",
                 "stop_gradient": "detach"}
        binary = {"add": "add", "add_any": "add", "sub": "sub", "mul": "mul_elem",
                  "max": "max_elem", "min": "min_elem"}
        if name in unary:
            return self.node(unary[name], parents, shape)
        if name == "sqrt":
            if not self.nonnegative(parents[0]):
                raise Unsupported("sqrt: TorchLean's square root clamps negative inputs; "
                                  "the source argument needs a structurally nonnegative expression")
            return self.node("sqrt", parents, shape)
        if name == "square":
            return self.node("mul_elem", [parents[0], parents[0]], shape)
        if name in binary or name == "div":
            left, right = [self.broadcast(i, shape) for i in parents]
            if name == "div":
                raise Unsupported("div: a division IR operation is required; reciprocal followed "
                                  "by multiplication changes rounding and exceptional values")
            return self.node(binary[name], [left, right], shape)
        if name == "neg":
            raise Unsupported("neg: an explicit negation IR operation is required; "
                              "zero minus the input changes signed-zero behavior")
        if name == "integer_pow":
            exponent = int(params["y"])
            if exponent == 1:
                return parents[0]
            if exponent == 2:
                return self.node("mul_elem", [parents[0], parents[0]], shape)
            if exponent != 0:
                raise Unsupported("integer_pow: this exponent needs an explicit power operation; "
                                  "repeated-squaring lowering changes the rounding schedule")
            one = self.constant(np.array(1, dtype=dtype), jax.core.ShapedArray((), dtype))
            return self.broadcast(one, shape)
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
            if self.nodes[parents[0]]["dtype"] == self.current_dtype:
                return parents[0]
            return self.node("custom", parents, shape, primitive="cast", rounding="nearestEven")
        raise Unsupported(f"Jaxpr import: unsupported primitive {name}")


def lower(closed_jaxpr: Any) -> dict[str, Any]:
    """Lower a captured tensor computation, rejecting unsupported semantics.

    Every node retains its own dtype; casts become explicit IR operations. Binary,
    Boolean, integer and complex tensors have lossless descriptors and constant words.
    Accepting a descriptor does not supply its execution semantics. Division is
    rejected rather than replaced by reciprocal multiplication. Square roots need
    structurally nonnegative operands, since the existing IR clamps negative inputs.
    Negation and general integer powers also need explicit operators; they are not
    rewritten into floating-point subtraction or reassociated products. Control flow is
    rejected. Custom JVP wrappers contribute their forward body only, not a proof
    of their derivative rule.
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
    return dict(format="torchlean.ir.v2", frontend="jaxpr", dtype=builder.dtype,
                input_ids=input_ids, output_ids=output_ids, nodes=builder.nodes)


def export(function: Callable[..., Any], *examples: Any) -> dict[str, Any]:
    """Capture positional tensor arguments at their example shapes and lower the Jaxpr.

    Input order is preserved, including unused arguments. A nested pytree needs an
    explicit container interface rather than silently flattening the user's call.
    """
    if not examples or any(not hasattr(value, "shape") or not hasattr(value, "dtype")
                           for value in examples):
        raise Unsupported("Jaxpr capture requires one or more positional tensor arguments")
    return lower(jax.make_jaxpr(function)(*examples))
