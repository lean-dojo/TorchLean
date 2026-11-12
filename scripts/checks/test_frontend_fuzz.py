"""Differential fuzzing of the PyTorch and JAX frontends against their own frameworks.

A seeded generator draws small programs, one to three calls over the operators of the PyTorch and
JAX frontends, with the argument spellings their frameworks accept (`dim`/`axis`,
`keepdim`/`keepdims`, positional and keyword, Tensor methods, `torch.*` functions and Python
operators), negative dimensions, every dtype, broadcasting, Python scalars and 0-d tensors, empty
and size-1 dimensions, and for JAX NumPy and jnp inputs with 64-bit mode on and off. Every program
returns every value it computes. The framework runs it first, PyTorch and JAX eagerly, with JAX
inputs converted as its trace reads them; a program the framework rejects is redrawn. The capture
then goes through one path, in turn torch.export, FX's symbolic trace and
Dynamo (`torchlean.compile`) for PyTorch, and `verify(..., frontend="jax")` for JAX, and `verify`
executes it. Every executed value must have the framework's shape and dtype, and its words: exactly
for a value whose operation and operands are exact; for the others within 4 units in the last place,
or within 64 units of the format's precision times the largest finite magnitude among the program's
values (a sum or a softmax may cancel). NaNs agree whatever their sign and payload, and for JAX so
do zeros of either sign: JAX gives `0.0` for `x * -1.25` at a Boolean `False`, eagerly and under
`jax.jit`, where IEEE multiplication of the converted `0.0` gives `-0.0`, which the IR computes.

A capture the frontend refuses (`NotImplementedError`, also when Dynamo wraps it, a Dynamo graph
break, or `unsupported` checks) is counted, not a failure. Any other outcome fails the test with the
seed, the program and the difference: a mismatch, a failed check, or an exception inside torchlean.
Behaviours that are documented as different from the framework or undefined in it (XLA's subnormal
flushing, floor division of an infinity or a NaN, a conversion of a NaN, an infinity or an
out-of-range value to an integer type, and a zero of PyTorch's floating `remainder`, whose sign
`Context.remainder` takes from Python's documented `%` and PyTorch's CPU kernel from `fmod`) are not
drawn. Every input is used, and the inputs are listed in the order of their first use, which is the
order of Dynamo's graph inputs.

The seed is `TORCHLEAN_FUZZ_SEED` (default 20261111) and the number of programs per framework
`TORCHLEAN_FUZZ_COUNT` (default 40). The test skips without the built `verify` executable or a
framework, and fails like `test_ir_capture.py` under `TORCHLEAN_REQUIRE_INTEROP=1` or
`TORCHLEAN_REQUIRE_JAX=1`.

An exact JAX mismatch is counted separately as a documented numerical difference only when the
checker's existing reciprocal-division emulation reproduces every differing result. It remains
an unsuccessful comparison, not a matched program. This covers eager XLA division by a broadcast
scalar; every other exact mismatch fails.
"""

import importlib.util
import math
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np  # noqa: E402

import torchlean  # noqa: E402


def _checker():
    path = os.environ.get("TORCHLEAN_VERIFY") or str(ROOT / ".lake/build/bin/verify")
    return path if Path(path).is_file() else None


CHECKER = _checker()
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_JAX = importlib.util.find_spec("jax") is not None
if os.environ.get("TORCHLEAN_REQUIRE_INTEROP") == "1" and not (CHECKER and HAS_TORCH):
    raise SystemExit("test_frontend_fuzz: TORCHLEAN_REQUIRE_INTEROP=1 needs PyTorch and the built "
                     "verify executable")
if os.environ.get("TORCHLEAN_REQUIRE_JAX") == "1" and not (CHECKER and HAS_JAX):
    raise SystemExit("test_frontend_fuzz: TORCHLEAN_REQUIRE_JAX=1 needs JAX and the built verify "
                     "executable")
SEED = int(os.environ.get("TORCHLEAN_FUZZ_SEED", "20261111"))
COUNT = int(os.environ.get("TORCHLEAN_FUZZ_COUNT", "40"))
EXECUTE = ("structure", "types", "execute")
WIDTHS = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}

# The operations whose results the frameworks round once, or compute exactly. The other floating
# results may differ by 4 units in the last place (transcendental functions and reductions, whose
# order of summation is the framework's).
EXACT = {"neg", "abs", "floor", "ceil", "trunc", "round", "sign", "relu", "square", "sqrt",
         "add", "sub", "mul", "div", "maximum", "minimum", "max", "min", "amax", "amin", "where",
         "clamp", "eq", "ne", "lt", "le", "gt", "ge", "reshape", "transpose", "permute", "flip",
         "squeeze", "unsqueeze", "cat", "slice", "flatten", "to", "argmax", "argmin", "all",
         "any", "logical_not", "logical_and", "logical_or", "bitwise_not", "bitwise_and",
         "bitwise_or", "bitwise_xor", "floor_divide", "trunc_divide", "remainder", "isnan",
         "isfinite", "expand_dims", "concatenate", "astype", "negative", "T", "count_nonzero"}


def _words(array):
    # The interchange words of an array: unsigned integers of its width, Booleans as 0 and 1.
    array = np.ascontiguousarray(array)
    if array.dtype == np.bool_:
        return array.reshape(-1).astype(np.uint8).astype(np.uint64)
    return array.astype(array.dtype.newbyteorder("=")).view(WIDTHS[array.dtype.itemsize]) \
        .reshape(-1).astype(np.uint64)


def _ulps(a, b, width):
    # The distance in units in the last place between two words of one binary format.
    sign = 1 << (8 * width - 1)
    def ordered(w):
        return -(w & (sign - 1)) if w & sign else w
    return abs(ordered(a) - ordered(b))


def _value(word, dtype_name):
    # The number a word of a binary format encodes.
    if dtype_name == "bfloat16":
        return float(np.array([word << 16], dtype=np.uint32).view(np.float32)[0])
    width = {"float16": np.uint16, "float32": np.uint32, "float64": np.uint64}[dtype_name]
    return float(np.array([word], dtype=width).view(dtype_name)[0])


def _difference(output, expected, exact, dtype_name, scale, zeros=False):
    """None when the executed `output` is the framework's `expected` value, else the difference.
    `scale` is the largest finite magnitude among the program's values; with `zeros` a zero
    matches a zero of either sign."""
    shape = list(expected.shape)
    if list(output["shape"]) != shape:
        return f"shape {output['shape']}, framework {shape}"
    kind = output["element_type"]
    if kind != dtype_name and not (kind == "index" and dtype_name in ("int64", "int32")):
        return f"dtype {kind}, framework {dtype_name}"
    got = [int(w) for w in np.asarray(output["bits"], dtype=object).reshape(-1)]
    if kind == "index":
        want = [int(v) for v in expected.reshape(-1)]
    else:
        want = [int(w) for w in _words(expected)]
    if len(got) != len(want):
        return f"{len(got)} words, framework {len(want)}"
    floating = expected.dtype.kind == "f" or dtype_name == "bfloat16"
    for index, (g, w) in enumerate(zip(got, want)):
        if g == w:
            continue
        if floating:
            width = 2 if dtype_name in ("float16", "bfloat16") else expected.dtype.itemsize
            nan = _is_nan(g, dtype_name) and _is_nan(w, dtype_name)
            sign = 1 << (8 * width - 1)
            if nan or (zeros and g & ~sign == 0 and w & ~sign == 0):
                continue
            if not exact:
                actual, reference = _value(g, dtype_name), _value(w, dtype_name)
                # The largest finite word neighbors infinity, and signaling NaNs neighbor it
                # too. Neither is a small numerical error across that boundary.
                if math.isfinite(actual) and math.isfinite(reference):
                    if _ulps(g, w, width) <= 4 or \
                            abs(actual - reference) <= \
                            64 * 2.0 ** -FORMATS[dtype_name][1] * scale:
                        continue
        return f"element {index}: word {g:#x}, framework {w:#x}"
    return None


FORMATS = {"float16": (5, 10), "bfloat16": (8, 7), "float32": (8, 23), "float64": (11, 52)}


def _is_nan(word, dtype_name):
    exponent, fraction = FORMATS[dtype_name]
    return (word >> fraction) & ((1 << exponent) - 1) == (1 << exponent) - 1 and \
        word & ((1 << fraction) - 1) != 0


class ComparisonTests(unittest.TestCase):
    def test_tolerances_do_not_cross_nonfinite_boundaries(self):
        for dtype_name, (exponent, fraction) in FORMATS.items():
            width = (1 + exponent + fraction) // 8
            infinity = ((1 << exponent) - 1) << fraction
            sign = 1 << (exponent + fraction)

            def compare(actual, expected, *, exact=False, zeros=False):
                values = np.array([expected], dtype=WIDTHS[width])
                if dtype_name != "bfloat16":
                    values = values.view(dtype_name)
                output = dict(shape=[1], element_type=dtype_name, bits=[actual])
                return _difference(output, values, exact, dtype_name, 1.0, zeros)

            with self.subTest(dtype=dtype_name), np.errstate(invalid="ignore"):
                for negative in (0, sign):
                    finite, inf, nan = [word | negative for word in
                                        (infinity - 1, infinity, infinity + 1)]
                    for a, b in ((finite, inf), (inf, finite), (inf, nan), (nan, inf),
                                 (nan, finite), (finite, nan)):
                        self.assertIsNotNone(compare(a, b))
                    self.assertIsNone(compare(inf, inf))
                    self.assertIsNone(compare(nan, (infinity + 3) | negative))
                self.assertIsNotNone(compare(infinity, infinity | sign))
                self.assertIsNone(compare(infinity - 2, infinity - 1))
                self.assertIsNotNone(compare(infinity - 2, infinity - 1, exact=True))
                self.assertIsNotNone(compare(0, sign, exact=True))
                self.assertIsNone(compare(0, sign, exact=True, zeros=True))


def _statuses(report):
    return {check["name"]: check["status"] for check in report["checks"]}


# ---------------------------------------------------------------- the generator

def _convertible(values, dtype_name):
    # Whether converting the floating `values` to `dtype_name` is defined in the frameworks: a
    # floating or Boolean target, or finite values whose truncations lie in the integer range.
    if dtype_name in ("bool", "float16", "bfloat16", "float32", "float64"):
        return True
    values = np.asarray(values, dtype=np.float64)
    info = np.iinfo(dtype_name)
    return bool(np.all(np.isfinite(values))) and \
        bool(np.all((np.trunc(values) >= info.min) & (np.trunc(values) <= info.max)))


def _used(program, name):
    return any(re.search(rf"\b{name}\b", line.split(" = ", 1)[1]) for line in program.lines)


def _extend(program, unit, evaluate, source=None):
    # Add one line drawn by `unit` that `evaluate` runs; False after 20 rejected draws.
    for _attempt in range(20):
        line = unit(program, source)
        try:
            value = evaluate(line, program)
        except Exception:
            program.operations.pop()
            continue
        if value is None:
            program.operations.pop()
            continue
        # A value is exact when its operation is and so is every value it reads.
        exact = dict(program.inputs and [(n, True) for n, _ in program.inputs])
        exact.update((f"v{k}", e) for k, e in enumerate(program.exact))
        reads = [n for n in exact if re.search(rf"\b{n}\b", line)]
        name = f"v{len(program.lines)}"
        program.lines.append(f"{name} = {line}")
        program.values.append((name, value))
        program.exact.append(program.operations[-1] in EXACT and all(exact[n] for n in reads))
        return True
    return False


def _draw(rng, inputs, unit, evaluate):
    # A program of one to three lines over `inputs` that uses every input, in first-use order.
    while True:
        program = Program(inputs())
        for _ in range(rng.randint(1, 3)):
            if not _extend(program, unit, evaluate):
                break
        for name, value in list(program.inputs):
            if program.lines and not _used(program, name):
                _extend(program, unit, evaluate, (name, value))
        if not program.lines or not all(_used(program, name) for name, _ in program.inputs):
            continue
        text = " ".join(line.split(" = ", 1)[1] for line in program.lines)
        program.inputs.sort(key=lambda item: re.search(rf"\b{item[0]}\b", text).start())
        return program


class Program:
    """A drawn program: its source, the names of its inputs, and the operations it calls."""

    def __init__(self, inputs):
        self.inputs = inputs          # [(name, value)]
        self.lines = []
        self.values = list(inputs)    # [(name, eager value)]
        self.exact = []               # whether each computed value is exact
        self.operations = []

    def source(self, prefix):
        names = ", ".join(name for name, _ in self.inputs)
        body = "".join(f"    {line}\n" for line in self.lines)
        results = ", ".join(f"v{k}" for k in range(len(self.lines)))
        return f"def {prefix}({names}):\n{body}    return ({results},)\n"


def _pick_dims(rng, rank, many=True):
    # A dimension argument: one dimension, negative or not, or (when `many`) a tuple of them.
    if rank == 0:
        return rng.choice(["0", "-1"]) if rng.random() < 0.3 else None
    dims = list(range(rank))
    if many and rank > 1 and rng.random() < 0.3:
        chosen = rng.sample(dims, rng.randint(1, rank))
        return "(" + ", ".join(str(d - rank if rng.random() < 0.5 else d) for d in chosen) + ",)"
    d = rng.choice(dims)
    return str(d - rank if rng.random() < 0.5 else d)


class Torch:
    """The PyTorch half of the generator."""

    DTYPES = ["float32", "float64", "float16", "bfloat16", "int64", "int32", "int16", "int8",
              "uint8", "bool"]
    SHAPES = [(2, 3), (3,), (), (1, 3), (2, 1, 2), (0, 3), (3, 1)]
    FLOATS = [0.0, -0.0, 1.0, -1.5, 2.5, 3.0, 0.1, -7.25, 1000.0, 1e-40, float("inf"),
              -float("inf"), float("nan")]
    # Forms that every path refuses (`logsumexp`, `std`, `nansum`, `count_nonzero`, a
    # `torch.tensor` call in the program) are left to the curated refusal tests.
    REDUCTIONS = ["sum", "mean", "amax", "amin", "prod", "argmax", "argmin", "all", "any",
                  "var", "max", "min", "cumsum", "softmax", "log_softmax"]
    UNARY = ["neg", "abs", "exp", "log", "sqrt", "floor", "ceil", "round", "trunc", "sign", "relu",
             "sigmoid", "tanh", "sin", "reciprocal", "square", "logical_not", "bitwise_not",
             "isnan", "isfinite", "expm1", "log1p"]
    BINARY = {"add": "+", "sub": "-", "mul": "*", "div": "/", "floor_divide": "//",
              "trunc_divide": None, "remainder": "%", "maximum": None, "minimum": None, "eq": "==", "ne": "!=",
              "lt": "<", "le": "<=", "gt": ">", "ge": ">=", "logical_and": None,
              "logical_or": None, "bitwise_and": "&", "bitwise_or": "|", "bitwise_xor": "^"}
    ALIASES = {"sub": "subtract", "mul": "multiply", "div": "true_divide", "gt": "greater",
               "lt": "less", "ge": "greater_equal", "le": "less_equal", "ne": "not_equal"}

    def __init__(self, rng):
        import torch
        self.torch = torch
        self.rng = rng

    def tensor(self, shape, dtype):
        torch, rng = self.torch, self.rng
        size = math.prod(shape)
        if dtype == "bool":
            values = [rng.random() < 0.5 for _ in range(size)]
        elif dtype.startswith("float") or dtype == "bfloat16":
            values = [rng.choice(self.FLOATS) if rng.random() < 0.4 else rng.randint(-12, 12) / 4
                      for _ in range(size)]
        else:
            info = torch.iinfo(getattr(torch, dtype))
            values = [rng.choice([info.min, info.max]) if rng.random() < 0.1
                      else max(info.min, rng.randint(-5, 9)) for _ in range(size)]
        return torch.tensor(values, dtype=getattr(torch, dtype)).reshape(shape)

    def inputs(self):
        rng = self.rng
        shape = rng.choice(self.SHAPES)
        first = ("x", self.tensor(shape, rng.choice(self.DTYPES)))
        if rng.random() < 0.5:
            return [first]
        # A second input of a broadcastable shape and maybe another dtype.
        other = rng.choice([shape, shape[1:], (1,) * len(shape), shape[:-1] + (1,) if shape else ()])
        return [first, ("y", self.tensor(other, rng.choice(self.DTYPES)))]

    def scalar(self):
        return repr(self.rng.choice([2, 3, -2, 0.5, -1.25, True, 7, 1e3]))

    def unit(self, program, source=None):
        rng = self.rng
        name, value = source or rng.choice(program.values)
        rank = value.dim()
        kind = rng.choice(["reduce", "unary", "binary", "binary", "structure", "convert"])
        if kind == "reduce":
            op = rng.choice(self.REDUCTIONS)
            single = op in ("prod", "argmax", "argmin", "all", "any", "cumsum", "softmax",
                            "log_softmax", "max", "min")
            dims = _pick_dims(rng, rank, many=not single)
            if op in ("cumsum", "softmax", "log_softmax", "max", "min") and dims is None:
                dims = "0" if rank <= 1 else "-1"
            keep = rng.choice([None, "True", "False"])
            arguments = []
            if dims is not None:
                arguments.append(rng.choice([f"dim={dims}", f"axis={dims}", dims]))
            if keep is not None and dims is not None and op not in (
                    "cumsum", "softmax", "log_softmax"):
                positional = not arguments[0].startswith(("dim", "axis"))
                arguments.append(rng.choice([f"keepdim={keep}", f"keepdims={keep}"]
                                            + ([keep] if positional else [])))
            tail = "".join(", " + a for a in arguments)
            call = rng.choice([f"torch.{op}({name}{tail})", f"{name}.{op}({tail[2:]})"])
            if op in ("max", "min"):
                call += rng.choice([".values", "[0]", ".indices", "[1]"])
            program.operations.append(op)
            return call
        if kind == "unary":
            op = rng.choice(self.UNARY)
            forms = [f"torch.{op}({name})", f"{name}.{op}()"]
            if op == "neg":
                forms.append(f"-{name}")
            if op == "abs":
                forms.append(f"abs({name})")
            if op == "bitwise_not":
                forms.append(f"~{name}")
            if op == "relu":
                forms = [f"torch.relu({name})", f"torch.nn.functional.relu({name})"]
            program.operations.append(op)
            return rng.choice(forms)
        if kind == "binary":
            op = rng.choice(list(self.BINARY))
            other = rng.choice([self.scalar(), rng.choice(program.values)[0]])
            if op in ("floor_divide", "remainder", "trunc_divide"):
                # Integer division by zero raises; floor division of an infinity is documented.
                other = rng.choice(["2", "3", "-2", "0.5", "0.1"])
            forms = [f"torch.{op}({name}, {other})", f"{name}.{op}({other})",
                     f"torch.{op}({name}, other={other})"]
            if op in ("floor_divide", "trunc_divide"):
                mode = op[:-len("_divide")]
                forms = [f"torch.div({name}, {other}, rounding_mode={mode!r})",
                         f"{name}.div({other}, rounding_mode={mode!r})"] + forms[:1] * (
                             op == "floor_divide")
            if self.BINARY[op]:
                forms.append(f"{name} {self.BINARY[op]} {other}")
            if op in self.ALIASES:
                forms.append(f"torch.{self.ALIASES[op]}({name}, {other})")
            program.operations.append(op)
            return rng.choice(forms)
        if kind == "structure":
            choices = ["reshape", "flip", "unsqueeze", "cat", "flatten"]
            if rank >= 1:
                choices += ["squeeze", "slice"]
            if rank >= 2:
                choices += ["transpose", "permute"]
            op = rng.choice(choices)
            program.operations.append(op)
            d = _pick_dims(rng, rank, many=False) or "0"
            if op == "reshape":
                return rng.choice([f"{name}.reshape(-1)", f"torch.reshape({name}, (-1, 1))",
                                   f"{name}.reshape({tuple(reversed(value.shape))})"])
            if op == "flip":
                return rng.choice([f"torch.flip({name}, dims=({d},))", f"{name}.flip({d})"]) \
                    if rank else f"torch.flip({name}, dims=())"
            if op == "unsqueeze":
                return rng.choice([f"{name}.unsqueeze({rng.randint(-rank - 1, rank)})",
                                   f"torch.unsqueeze({name}, dim={rng.randint(-rank - 1, rank)})"])
            if op == "cat":
                return rng.choice([f"torch.cat([{name}, {name}], dim={d})",
                                   f"torch.cat(({name}, {name}), axis={d})",
                                   f"torch.concat([{name}, {name}], {d})"])
            if op == "flatten":
                return rng.choice([f"{name}.flatten()", f"torch.flatten({name}, start_dim=0)"])
            if op == "squeeze":
                return rng.choice([f"{name}.squeeze()", f"{name}.squeeze({d})",
                                   f"torch.squeeze({name}, dim={d})"])
            if op == "slice":
                return rng.choice([f"{name}[1:]", f"{name}[..., :1]", f"{name}[-1]"])
            if op == "transpose":
                return rng.choice([f"{name}.transpose(0, -1)", f"torch.transpose({name}, -2, -1)",
                                   f"{name}.swapaxes(0, 1)"])
            perm = list(range(rank))
            rng.shuffle(perm)
            return rng.choice([f"{name}.permute({', '.join(map(str, perm))})",
                               f"torch.permute({name}, dims={tuple(perm)})"])
        dtype = rng.choice(self.DTYPES)
        if value.is_floating_point() and not _convertible(value.double().numpy(), dtype):
            dtype = rng.choice(["float32", "float64", "float16", "bfloat16", "bool"])
        program.operations.append("to")
        return rng.choice([f"{name}.to(torch.{dtype})", f"{name}.type(torch.{dtype})",
                           {"float32": f"{name}.float()", "float64": f"{name}.double()",
                            "float16": f"{name}.half()", "int64": f"{name}.long()",
                            "int32": f"{name}.int()", "bool": f"{name}.bool()"}
                           .get(dtype, f"{name}.to(dtype=torch.{dtype})")])

    def draw(self):
        """A program the framework runs, and its eager values."""
        torch = self.torch

        def evaluate(line, program):
            with torch.no_grad():
                value = eval(line, dict({"torch": torch}, **dict(program.values)))
            if not isinstance(value, torch.Tensor):
                return None
            if program.operations[-1] == "remainder" and value.is_floating_point() and \
                    bool((value == 0).any()):
                return None
            if program.operations[-1] == "floor_divide" and value.is_floating_point() and \
                    not bool(torch.isfinite(value).all()):
                return None
            return value

        return _draw(self.rng, self.inputs, self.unit, evaluate)


class Jax:
    """The JAX half of the generator."""

    DTYPES = ["float32", "float64", "float16", "bfloat16", "int64", "int32", "int16", "int8",
              "uint8", "uint32", "uint64", "bool"]
    SHAPES = [(2, 3), (3,), (), (1, 3), (2, 1, 2), (0, 3)]
    # No subnormals: XLA's CPU code flushes them, which is documented, not drawn.
    FLOATS = [0.0, -0.0, 1.0, -1.5, 2.5, 3.0, 0.1, -7.25, 1000.0, float("inf"), -float("inf"),
              float("nan")]
    REDUCTIONS = ["sum", "mean", "max", "min", "prod", "argmax", "argmin", "all", "any"]
    UNARY = ["negative", "abs", "floor", "ceil", "sign", "exp", "log", "sqrt", "logical_not",
             "isnan", "isfinite", "square"]
    BINARY = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/", "floor_divide": "//",
              "remainder": "%", "maximum": None, "minimum": None, "equal": "==", "less": "<",
              "greater_equal": ">=", "logical_and": None, "bitwise_xor": "^"}
    NAMES = {"subtract": "sub", "multiply": "mul", "divide": "div", "negative": "neg",
             "equal": "eq", "less": "lt", "greater_equal": "ge"}

    def __init__(self, rng):
        import jax
        import jax.numpy as jnp
        self.jax, self.jnp, self.rng = jax, jnp, rng

    def array(self, shape, dtype):
        rng = self.rng
        size = math.prod(shape)
        if dtype == "bool":
            return np.array([rng.random() < 0.5 for _ in range(size)]).reshape(shape)
        if dtype.startswith("float") or dtype == "bfloat16":
            values = [rng.choice(self.FLOATS) if rng.random() < 0.4 else rng.randint(-12, 12) / 4
                      for _ in range(size)]
            if dtype == "bfloat16":
                return self.jnp.array(np.array(values, dtype=np.float32).reshape(shape),
                                      dtype=self.jnp.bfloat16)
            return np.array(values, dtype=dtype).reshape(shape)
        info = np.iinfo(dtype)
        values = [rng.choice([info.min, info.max]) if rng.random() < 0.1
                  else max(int(info.min), rng.randint(-5, 9)) for _ in range(size)]
        return np.array(values, dtype=dtype).reshape(shape)

    def inputs(self):
        rng = self.rng
        shape = rng.choice(self.SHAPES)
        first = self.array(shape, rng.choice(self.DTYPES))
        if rng.random() < 0.3 and not isinstance(first, self.jax.Array):
            first = self.jnp.asarray(first)
        if rng.random() < 0.5:
            return [("x", first)]
        other = rng.choice([shape, shape[1:], (1,) * len(shape)])
        return [("x", first), ("y", self.array(other, rng.choice(self.DTYPES)))]

    def unit(self, program, source=None):
        rng, jnp = self.rng, self.jnp
        name, value = source or rng.choice(program.values)
        rank = np.ndim(value)
        kind = rng.choice(["reduce", "unary", "binary", "binary", "structure", "convert"])
        if kind == "reduce":
            op = rng.choice(self.REDUCTIONS)
            dims = _pick_dims(rng, rank, many=op not in ("argmax", "argmin"))
            arguments = []
            if dims is not None:
                arguments.append(rng.choice([f"axis={dims}", dims]))
            if rng.random() < 0.6:
                arguments.append(f"keepdims={rng.choice(['True', 'False'])}")
            tail = "".join(", " + a for a in arguments)
            call = rng.choice([f"jnp.{op}({name}{tail})", f"{name}.{op}({tail[2:]})"])
            program.operations.append({"max": "amax", "min": "amin"}.get(op, op))
            return call
        if kind == "unary":
            op = rng.choice(self.UNARY)
            forms = [f"jnp.{op}({name})"]
            if op == "negative":
                forms.append(f"-{name}")
            if op == "abs":
                forms.append(f"abs({name})")
            program.operations.append(self.NAMES.get(op, op))
            return rng.choice(forms)
        if kind == "binary":
            op = rng.choice(list(self.BINARY))
            other = rng.choice([repr(rng.choice([2, 3, -2, 0.5, -1.25, True, 7])),
                                rng.choice(program.values)[0]])
            if op in ("floor_divide", "remainder"):
                other = rng.choice(["2", "3", "-2", "0.5"])
            forms = [f"jnp.{op}({name}, {other})"]
            if self.BINARY[op]:
                forms.append(f"{name} {self.BINARY[op]} {other}")
            program.operations.append(self.NAMES.get(op, op))
            return rng.choice(forms)
        if kind == "structure":
            choices = ["reshape", "flip", "expand_dims", "concatenate"]
            if rank >= 1:
                choices += ["squeeze", "slice"]
            if rank >= 2:
                choices += ["transpose", "T"]
            op = rng.choice(choices)
            program.operations.append(op)
            d = _pick_dims(rng, rank, many=False) or "0"
            if op == "reshape":
                return rng.choice([f"{name}.reshape(-1)", f"jnp.reshape({name}, (-1, 1))"])
            if op == "flip":
                return f"jnp.flip({name}, axis={d})" if rank else f"jnp.flip({name})"
            if op == "expand_dims":
                return f"jnp.expand_dims({name}, {rng.randint(-rank - 1, rank)})"
            if op == "concatenate":
                return f"jnp.concatenate([{name}, {name}], axis={d})" if rank \
                    else f"jnp.stack([{name}, {name}])"
            if op == "squeeze":
                return rng.choice([f"jnp.squeeze({name})", f"{name}.squeeze()"])
            if op == "slice":
                return rng.choice([f"{name}[1:]", f"{name}[..., :1]", f"{name}[-1]"])
            if op == "transpose":
                return f"jnp.transpose({name})"
            return f"{name}.T"
        dtype = rng.choice(self.DTYPES)
        array = np.asarray(value)
        if (array.dtype.kind == "f" or str(array.dtype) == "bfloat16") and \
                not _convertible(array.astype(np.float64), dtype):
            dtype = rng.choice(["float32", "float64", "float16", "bfloat16", "bool"])
        program.operations.append("astype")
        return rng.choice([f"{name}.astype(jnp.{dtype})", f"jnp.asarray({name}, dtype=jnp.{dtype})"])

    def draw(self):
        """A program the framework runs, and its eager values."""
        jnp = self.jnp

        def evaluate(line, program):
            value = eval(line, dict({"jnp": jnp}, **dict(program.values)))
            array = np.asarray(value)
            if program.operations[-1] in ("floor_divide", "remainder") and \
                    array.dtype.kind == "f" and not bool(np.isfinite(array).all()):
                return None
            return value

        return _draw(self.rng, self.inputs, self.unit, evaluate)


# ---------------------------------------------------------------- the paths

def _torch_paths(function, inputs):
    """The captures of `function` at `inputs` through torch.export, FX and Dynamo, by name."""
    import torch
    from torchlean.export import export_torchlean_graph_json
    from torchlean._fx import _tensor_value

    # FX traces the parameters of `forward`, so it names one per input.
    names = ", ".join(f"a{k}" for k in range(len(inputs)))
    scope = {"function": function}
    exec(f"def forward(self, {names}):\n    return function({names})\n", scope)
    Call = type("Call", (torch.nn.Module,), {"forward": scope["forward"]})

    def through_export():
        return torchlean.verify(function, *inputs, checks=EXECUTE, checker=CHECKER)

    def through_fx():
        with tempfile.TemporaryDirectory() as directory:
            document = export_torchlean_graph_json(Call(), tuple(inputs),
                                                   str(Path(directory) / "g.json"),
                                                   prefer_torch_export=False)
        document["arguments"] = [_tensor_value(v) for v in inputs]
        return torchlean.verify(document, checks=EXECUTE, checker=CHECKER)

    def through_dynamo():
        compiled = torchlean.compile(checker=CHECKER)(function)
        compiled(*inputs)
        graphs = compiled.torchlean.graphs
        if len(graphs) != 1:
            raise NotImplementedError(f"Dynamo split the program into {len(graphs)} graphs")
        # The graph's inputs are the program's, in the order of their first use (`_draw`).
        document = graphs[0]["artifact"]
        document["arguments"] = [_tensor_value(v) for v in inputs]
        return torchlean.verify(document, checks=EXECUTE, checker=CHECKER)

    return {"export": through_export, "fx": through_fx, "dynamo": through_dynamo}


def _dtype_name(value):
    text = str(value.dtype)
    return text.replace("torch.", "")


def _refusal(error):
    """The reason of a refusal that `error` is or wraps (Dynamo wraps a backend's error), else None."""
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, NotImplementedError):
            return str(error)
        if type(error).__name__ == "Unsupported" and "_dynamo" in type(error).__module__:
            return f"Dynamo: {error}"
        error = getattr(error, "inner_exception", None) or error.__cause__ or error.__context__
    return None



class Outcomes:
    """The outcomes of one fuzzing run: refusals counted by reason, failures listed in full."""

    def __init__(self, framework):
        self.framework, self.accepted, self.refused, self.failures = framework, 0, {}, []
        self.documented_differences = 0

    def refuse(self, reason):
        key = " ".join(reason.split())[:110]
        self.refused[key] = self.refused.get(key, 0) + 1

    def summary(self):
        reasons = "".join(f"\n  refused {n}: {reason}" for reason, n in
                          sorted(self.refused.items(), key=lambda item: -item[1]))
        return (f"{self.framework}: {self.accepted} accepted and matched, "
                f"{sum(self.refused.values())} refused, "
                f"{self.documented_differences} documented numerical differences, "
                f"{len(self.failures)} failed"
                f"{reasons}")


def _scale(arrays):
    # The largest finite magnitude among the floating `arrays`, at least 1.
    finite = [np.abs(a[np.isfinite(a)]).max() for a in arrays if a.size and np.isfinite(a).any()]
    return float(max(finite, default=1.0))


def _judge(outcomes, label, program, source, report, expected, scale):
    # Count a refusal, or check every output against the framework's value.
    statuses = _statuses(report)
    if all(s == "unsupported" for s in statuses.values()):
        outcomes.refuse(next((c["message"] for c in report["checks"]), "unsupported"))
        return
    if not report["ok"]:
        messages = {c["name"]: c["message"][:300] for c in report["checks"]
                    if c["status"] != "checked"}
        outcomes.failures.append(f"{label}: checks {statuses} {messages}\n{source}")
        return
    outputs = report["outputs"]
    if outputs is None or len(outputs) != len(expected):
        outcomes.failures.append(f"{label}: {None if outputs is None else len(outputs)} outputs, "
                                 f"framework {len(expected)}\n{source}")
        return
    for k, (output, (value, dtype_name)) in enumerate(zip(outputs, expected)):
        difference = _difference(output, value, program.exact[k], dtype_name, scale,
                                 zeros=outcomes.framework == "JAX")
        if difference is not None:
            outcomes.failures.append(f"{label}: v{k} = {program.lines[k].split(' = ', 1)[1]}: "
                                     f"{difference}\n{source}")
            return
    outcomes.accepted += 1


@unittest.skipUnless(CHECKER and HAS_TORCH, "needs PyTorch and the built verify executable")
class PyTorchFuzz(unittest.TestCase):
    maxDiff = None

    def test_programs_match_eager(self):
        import torch
        torch._dynamo.config.cache_size_limit = 4 * COUNT
        rng = random.Random(SEED)
        generator = Torch(rng)
        outcomes = Outcomes("PyTorch")
        for index in range(COUNT):
            program = generator.draw()
            path = ("export", "fx", "dynamo")[index % 3]
            source = program.source(f"program{index}")
            namespace = {"torch": torch}
            exec(source, namespace)
            function = namespace[f"program{index}"]
            inputs = [value for _, value in program.inputs]
            expected = [(value.detach().numpy() if value.dtype != torch.bfloat16
                         else value.view(torch.int16).numpy().view(np.uint16), _dtype_name(value))
                        for _, value in program.values[len(program.inputs):]]
            label = f"seed {SEED} program {index} via {path}, inputs " + ", ".join(
                f"{n}: {_dtype_name(v)}{list(v.shape)}" for n, v in program.inputs)
            scale = _scale([v.detach().double().numpy() for _, v in program.values
                            if v.is_floating_point()])
            try:
                report = _torch_paths(function, inputs)[path]()
            except Exception as error:  # noqa: BLE001 - a crash inside torchlean is a failure
                reason = _refusal(error)
                if reason is not None:
                    outcomes.refuse(reason)
                else:
                    outcomes.failures.append(f"{label}: {type(error).__name__}: {error}\n{source}")
                continue
            _judge(outcomes, label, program, source, report, expected, scale)
        print("\n" + outcomes.summary(), file=sys.stderr)
        self.assertEqual([], outcomes.failures, "\n\n".join(outcomes.failures))
        self.assertGreater(outcomes.accepted, COUNT // 4, outcomes.summary())


@unittest.skipUnless(CHECKER and HAS_JAX, "needs JAX and the built verify executable")
class JaxFuzz(unittest.TestCase):
    maxDiff = None

    def test_programs_match_eager(self):
        import jax
        import jax.numpy as jnp
        from torchlean import jaxpr

        rng = random.Random(SEED + 1)
        outcomes = Outcomes("JAX")
        for index in range(COUNT):
            x64 = index % 2 == 1
            with jax.enable_x64(x64):
                generator = Jax(rng)
                while True:
                    # NumPy inputs must follow JAX's types. Compiling the whole function here
                    # would change the computation boundaries recorded by the eager capture.
                    program = generator.draw()
                    source = program.source(f"program{index}")
                    namespace = {"jnp": jnp}
                    exec(source, namespace)
                    function = namespace[f"program{index}"]
                    inputs = [value for _, value in program.inputs]
                    try:
                        values = [np.asarray(v) for v in function(*jaxpr.canonical(inputs))]
                    except Exception:  # noqa: BLE001 - JAX rejects the program
                        continue
                    break
                expected = [(array.view(np.uint16) if str(array.dtype) == "bfloat16" else array,
                             str(array.dtype)) for array in values]
                scale = _scale([np.asarray(a, dtype=np.float64)
                                for a in [np.asarray(v) for _, v in program.inputs] + values
                                if a.dtype.kind == "f" or str(a.dtype) == "bfloat16"])
                label = f"seed {SEED + 1} program {index} with x64 {'on' if x64 else 'off'}, " \
                    "inputs " + ", ".join(f"{n}: {type(v).__name__} {np.asarray(v).dtype}"
                                          f"{list(np.shape(v))}" for n, v in program.inputs)
                try:
                    report = torchlean.verify(function, *inputs, frontend="jax", checks=EXECUTE,
                                              checker=CHECKER)
                except Exception as error:  # noqa: BLE001
                    reason = _refusal(error)
                    if reason is not None:
                        outcomes.refuse(reason)
                    else:
                        outcomes.failures.append(
                            f"{label}: {type(error).__name__}: {error}\n{source}")
                    continue
                failures = len(outcomes.failures)
                _judge(outcomes, label, program, source, report, expected, scale)
                if len(outcomes.failures) > failures and report["ok"]:
                    compared = torchlean.verify(function, *inputs, frontend="jax",
                                                checks=(*EXECUTE, "compare"), checker=CHECKER)
                    checks = {check["name"]: check for check in compared["checks"]}
                    comparison = checks.get("compare", {})
                    results = comparison.get("results", [])
                    if (all(checks.get(name, {}).get("status") == _statuses(report)[name]
                            for name in EXECUTE)
                            and compared.get("outputs") == report["outputs"]
                            and comparison.get("status") == "failed"
                            and comparison.get("consistent_with") == ["divisor_reciprocal"]
                            and results and all(result.get("not_reproduced") == 0
                                and result.get("differing") == result.get("reproduced")
                                and result.get("window") is None for result in results)):
                        outcomes.failures.pop()
                        outcomes.documented_differences += 1
        print("\n" + outcomes.summary(), file=sys.stderr)
        self.assertEqual([], outcomes.failures, "\n\n".join(outcomes.failures))
        self.assertGreater(outcomes.accepted, COUNT // 4, outcomes.summary())


if __name__ == "__main__":
    unittest.main()
