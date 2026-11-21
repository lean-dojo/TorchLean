"""Shared helpers of the capture tests that ``scripts/checks/test_ir_capture.py`` runs: the
built checker, the check names, report readers, reference arithmetic and ``CheckerCase``.
Import it through that script, which puts this directory on ``sys.path``.
"""

import importlib.util
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

# The JAX cases compare against XLA's CPU code, so select the CPU before JAX is imported.
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _checker():
    path = os.environ.get("TORCHLEAN_VERIFY") or str(ROOT / ".lake/build/bin/verify")
    return path if Path(path).is_file() else None


CHECKER = _checker()
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_JAX = importlib.util.find_spec("jax") is not None
# A strict interop run (`TORCHLEAN_REQUIRE_INTEROP=1`) fails when it cannot exercise the PyTorch
# captures, and under `TORCHLEAN_REQUIRE_JAX=1` the same holds for the JAX ones.
if os.environ.get("TORCHLEAN_REQUIRE_INTEROP") == "1" and not (CHECKER and HAS_TORCH):
    raise SystemExit("test_ir_capture: TORCHLEAN_REQUIRE_INTEROP=1 needs PyTorch and the built "
                     "verify executable (scripts/lake.sh build verify)")
if os.environ.get("TORCHLEAN_REQUIRE_JAX") == "1" and not (CHECKER and HAS_JAX):
    raise SystemExit("test_ir_capture: TORCHLEAN_REQUIRE_JAX=1 needs JAX and the built verify "
                     "executable")
EXECUTE = ("structure", "types", "execute")
COMPARE = ("structure", "types", "execute", "compare")


def _statuses(report):
    return {check["name"]: (check["status"], check["message"]) for check in report["checks"]}


def _outputs(report):
    # The executed results as tensors: a binary or integer result from its words, a Boolean one
    # from its 0s and 1s, and a result of index tuples as the int64 tensor of their components.
    import torch

    tensors = []
    for output in report["outputs"]:
        if output["element_type"] in ("index", "bool"):
            dtype = torch.int64 if output["element_type"] == "index" else torch.bool
            tensors.append(torch.tensor(output["bits"], dtype=dtype).reshape(output["shape"]))
            continue
        dtype = getattr(torch, output["element_type"])
        width = torch.finfo(dtype).bits if dtype.is_floating_point else torch.iinfo(dtype).bits
        integer = {8: torch.int8, 16: torch.int16, 32: torch.int32, 64: torch.int64}[width]
        words = [word - (1 << width) if word >= 1 << (width - 1) else word
                 for word in output["bits"]]
        tensors.append(torch.tensor(words, dtype=integer).view(dtype).reshape(output["shape"]))
    return tensors


FORMATS = {"float16": (11, -14, 15), "bfloat16": (8, -126, 127), "float32": (24, -126, 127),
           "float64": (53, -1022, 1023)}

# IEEE 754-2019 binary128 (3.6, Table 3.5): its precision and exponent range, for `_nearest`.
BINARY128 = (113, -16382, 16383)


def _nearest(value, dtype):
    # The exact rational `value` (an int, a float or a Fraction) rounded to the format named
    # `dtype`, or given as its (precision, emin, emax), to nearest with ties to even, an infinity
    # from the overflow threshold on, as a Python float, or as a Fraction for a format wider than
    # binary64. A float zero, infinity or NaN passes through with its sign.
    import math
    from fractions import Fraction

    if isinstance(value, float) and (value != value or math.isinf(value) or value == 0):
        return value
    precision, emin, emax = FORMATS[dtype] if isinstance(dtype, str) else dtype
    exact = Fraction(value)
    if exact == 0:
        return 0.0
    magnitude = abs(exact)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if Fraction(2) ** exponent > magnitude:
        exponent -= 1
    quantum = Fraction(2) ** (max(exponent, emin) - precision + 1)
    rounded = round(magnitude / quantum) * quantum
    if rounded >= 2 ** (emax + 1):
        rounded = math.inf
    elif precision <= 53 and emin >= -1022:
        rounded = float(rounded)
    return -rounded if exact < 0 else rounded


def _splitmix64(value):
    # One output of SplitMix64 for the 64-bit word `value`: the tests' reference for the
    # keyed draws of TorchLean's generator.
    z = (value + 0x9e3779b97f4a7c15) % 2 ** 64
    z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) % 2 ** 64
    z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) % 2 ** 64
    return z ^ (z >> 31)


def _uniform(key, count, dtype):
    # The first `count` draws of TorchLean's generator under `key`, an Op.randUniform record's:
    # entry i is splitmix64(key + i) mod 2^32 over 2^32, rounded once to the torch dtype `dtype`.
    import torch

    draws = [_splitmix64((key + i) % 2 ** 64) % 2 ** 32 for i in range(count)]
    return torch.tensor(draws, dtype=torch.float64).div(2 ** 32).to(dtype)


def _power(base, exponent):
    # base ** exponent for Python floats, exactly where it is rational and otherwise to 300 bits
    # (mpmath), as a Fraction, and NaN for a negative base and a non-integral exponent. Finite
    # nonzero bases only.
    from fractions import Fraction

    import mpmath

    if base < 0 and not float(exponent).is_integer():
        return float("nan")
    if float(exponent).is_integer():
        return Fraction(base) ** int(exponent)
    with mpmath.workprec(300):
        sign, man, shift, _ = mpmath.power(mpmath.mpf(base), mpmath.mpf(exponent))._mpf_
    return (-1 if sign else 1) * Fraction(int(man)) * Fraction(2) ** shift


def _ieee_maximum(a, b):
    # IEEE 754-2019 maximum: a NaN operand gives NaN, and -0 is below +0.
    import math

    if a != a or b != b:
        return math.nan
    if a == b == 0:
        return 0.0 if math.copysign(1.0, a) > 0 or math.copysign(1.0, b) > 0 else -0.0
    return max(a, b)


def _ieee_minimum(a, b):
    # IEEE 754-2019 minimum: a NaN operand gives NaN, and -0 is below +0.
    import math

    if a != a or b != b:
        return math.nan
    if a == b == 0:
        return -0.0 if math.copysign(1.0, a) < 0 or math.copysign(1.0, b) < 0 else 0.0
    return min(a, b)


def _same(case, actual, expected, message=None):
    # Equal values with equal zero signs, or both NaN, entry by entry.
    import math

    case.assertEqual(len(actual), len(expected), message)
    for a, b in zip(actual, expected):
        if a != a or b != b:
            case.assertTrue(a != a and b != b, (a, b, message))
        else:
            case.assertEqual((a, math.copysign(1.0, a)), (b, math.copysign(1.0, b)), message)


def _reflect32(a, taus, c, left, transpose):
    # LinalgImport.reflected's arithmetic in NumPy's binary32 for one matrix a of shape (m, n):
    # the reflectors H_i = 1 - tau_i v_i v_i^T of a's first k = len(taus) columns (one on the
    # diagonal, zero above it) applied to c one at a time in the order of op(Q) with
    # Q = H_0 ... H_{k-1}: c - tau_i (v_i (v_i^T c)) on the left, c - tau_i ((c v_i) v_i^T) on
    # the right, each product a left fold from +0 and every other entry rounded once.
    import numpy as np

    f32 = np.float32

    def fold(terms):
        total = f32(0)
        for term in terms:
            total = f32(total + term)
        return total

    m, k = a.shape[0], len(taus)
    v = np.where(np.tri(m, k, -1, dtype=bool), a[:, :k], np.eye(m, k, dtype=f32))
    for i in reversed(range(k)) if left != transpose else range(k):
        if left:
            w = np.array([fold(f32(v[r, i] * c[r, j]) for r in range(m))
                          for j in range(c.shape[1])], dtype=f32)
            outer = np.outer(v[:, i], w)
        else:
            w = np.array([fold(f32(c[r, s] * v[s, i]) for s in range(m))
                          for r in range(c.shape[0])], dtype=f32)
            outer = np.outer(w, v[:, i])
        c = c - taus[i] * outer
    return c


def _window_fold(x, init, combine, window, strides, padding, base, dilation):
    # StableHLO's reduce_window by one schedule it allows: the object array x with base - 1 init
    # values between entries and its padding of init values, each window folded from init with
    # combine in row-major cell order. The windows' results in row-major order.
    import itertools

    import numpy as np

    def filled(shape):
        grid = np.empty(shape, dtype=object)
        grid.fill(init)
        return grid

    x = np.asarray(x)
    dilated = [(n - 1) * b + 1 if n else 0 for n, b in zip(x.shape, base)]
    grid = filled(dilated)
    grid[tuple(slice(None, None, b) for b in base)] = x
    padded = filled([n + lo + hi for n, (lo, hi) in zip(dilated, padding)])
    padded[tuple(slice(lo, lo + n) for n, (lo, _) in zip(dilated, padding))] = grid
    extents = [(p - (w - 1) * d - 1) // s + 1 if p >= (w - 1) * d + 1 else 0
               for p, w, s, d in zip(padded.shape, window, strides, dilation)]
    result = []
    for out in itertools.product(*map(range, extents)):
        total = init
        for cell in itertools.product(*map(range, window)):
            total = combine(total, padded[tuple(o * s + c * d for o, c, s, d in
                                                zip(out, cell, strides, dilation))])
        result.append(total)
    return result



class CheckerCase(unittest.TestCase):
    def assert_passes(self, report):
        self.assertTrue(report["ok"], _statuses(report))

    def assert_unsupported_execution(self, report, text):
        statuses = _statuses(report)
        self.assertEqual(statuses["structure"][0], "checked", statuses)
        self.assertEqual(statuses["types"][0], "checked", statuses)
        self.assertEqual(statuses["execute"][0], "failed", statuses)
        self.assertIn(text, statuses["execute"][1])

    def assert_refused(self, report, text):
        statuses = _statuses(report)
        self.assertFalse(report["ok"], statuses)
        self.assertEqual(statuses["structure"][0], "unsupported", statuses)
        self.assertIn(text, statuses["structure"][1])

    def assert_special(self, document, dtype, shape, steps, outputs):
        # The capture reads one input of `shape`, then applies Op.special once per entry
        # (name, order, at) of `steps`, in order: the derivative of that order of the member with
        # that torch.special name, to the value of step `at`. Every step and declared type is at
        # `dtype` and `shape`, and the program returns the steps `outputs`.
        artifact = document["artifact"]
        nodes = artifact["nodes"]
        self.assertEqual(nodes[0], {"input": {"index": 0, "port": {"value": shape}}})
        self.assertEqual([(node["op"]["record"]["tag"], node["op"]["record"]["fields"],
                           node["op"]["inputs"]) if "op" in node else node for node in nodes[1:]],
                         [("special", [{"name": name}, {"nats": [order]}, {"shape": shape}],
                           [[at, 0]]) for name, order, at in steps])
        self.assertEqual([_scalar_name(scalar) for scalar in artifact["scalars"]],
                         [dtype] * len(nodes))
        self.assertEqual(artifact["outputs"], [[at, 0] for at in outputs])
        self.assertEqual({entry["node"]: (entry["dtype"], entry["shape"])
                          for entry in document["declared_types"]},
                         {at: (dtype, shape) for at in range(len(nodes))})

    def assert_broadcast_step(self, document, dtype, tag, fields, shapes, shape):
        # The capture reads one input of each shape in `shapes`, broadcasts each operand whose
        # shape is not the result's shape `shape` to it, in operand order, and applies one step
        # to the operands, with the record tagged `tag` whose fields are `fields` and then
        # `shape`. Every step is at `dtype`, the program returns that step, and the types
        # declared are those of the inputs and of that step (a broadcast is the capture's own
        # step, not one of the graph's).
        artifact = document["artifact"]
        nodes = artifact["nodes"]
        arity = len(shapes)
        self.assertEqual(nodes[:arity], [{"input": {"index": k, "port": {"value": extents}}}
                                         for k, extents in enumerate(shapes)])
        broadcast = [k for k, extents in enumerate(shapes) if extents != shape]
        self.assertEqual([node["op"]["record"]["tag"] for node in nodes[arity:]],
                         ["broadcast"] * len(broadcast) + [tag])
        operands = list(range(arity))
        for at, k in enumerate(broadcast, arity):
            self.assertEqual(nodes[at]["op"]["inputs"], [[k, 0]])
            operands[k] = at
        self.assertEqual((nodes[-1]["op"]["record"]["fields"], nodes[-1]["op"]["inputs"]),
                         (fields + [{"shape": shape}], [[at, 0] for at in operands]))
        self.assertEqual([_scalar_name(scalar) for scalar in artifact["scalars"]],
                         [dtype] * len(nodes))
        self.assertEqual(artifact["outputs"], [[len(nodes) - 1, 0]])
        self.assertEqual({entry["node"]: (entry["dtype"], entry["shape"])
                          for entry in document["declared_types"]},
                         {at: (dtype, shapes[at] if at < arity else shape)
                          for at in (*range(arity), len(nodes) - 1)})



def _assert_within_ulps(case, actual, expected, ulps):
    # The NaNs, the zeros and the infinities of both tensors are at the same positions, the zeros
    # and infinities with the same signs (NaN signs are not compared), and every other pair of
    # entries lies within `ulps` units in the last place, on the monotone integer line of the
    # format's words, measured with Python integers.
    import torch

    case.assertEqual(actual.dtype, expected.dtype)
    for kind in (torch.isnan, lambda t: t == 0, torch.isinf):
        case.assertTrue(torch.equal(kind(actual), kind(expected)), (actual, expected))
    special = (actual == 0) | actual.isinf()
    case.assertTrue(torch.equal(actual[special].signbit(), expected[special].signbit()),
                    (actual, expected))
    width = actual.element_size() * 8
    integer = {16: torch.int16, 32: torch.int32, 64: torch.int64}[width]

    def line(tensor):
        words = tensor.contiguous().view(integer).flatten().tolist()
        magnitude = [word & ((1 << (width - 1)) - 1) for word in words]
        return [-m if word < 0 else m for word, m in zip(words, magnitude)]

    finite = (~(actual.isnan() | special)).flatten().tolist()
    distance = max((abs(a - b) for a, b, keep in zip(line(actual), line(expected), finite)
                    if keep), default=0)
    case.assertLessEqual(distance, ulps, (actual, expected))


def _blocks(flow):
    # The artifacts of the blocks of a flow artifact's flow, in the order of Lean's Flow.blocks.
    tag, *parts = flow
    if tag == "block":
        return parts
    return [block for part in parts if isinstance(part, list) for block in _blocks(part)]


def _flows(document):
    # The cond, scan and while nodes of the flow of a capture document's artifact, by their
    # tags, sorted: none for an artifact without control flow.
    found, pending = [], [document["artifact"].get("flow", ["skip"])]
    while pending:
        tag, *parts = pending.pop()
        if tag != "block":
            found += [tag] if tag in ("cond", "scan", "while") else []
            pending += [part for part in parts if isinstance(part, list)]
    return sorted(found)


def _steps(document):
    # The steps of the artifact of a capture document, those of every block of a flow in order.
    artifact = document["artifact"]
    if "flow" not in artifact:
        return artifact["nodes"]
    return [step for block in _blocks(artifact["flow"]) for step in block["nodes"]]


def _records(document):
    # The operator records of the artifact's op steps, in step order.
    return [step["op"]["record"] for step in _steps(document) if "op" in step]


def _tags(document):
    return [record["tag"] for record in _records(document)]


def _names(jaxpr):
    # The primitive names of a Jaxpr's equations and, recursively, of the Jaxprs among their
    # parameters, a cond's branches included.
    names = set()
    for equation in jaxpr.eqns:
        names.add(equation.primitive.name)
        for value in equation.params.values():
            for inner in value if isinstance(value, (tuple, list)) else (value,):
                inner = getattr(inner, "jaxpr", inner)
                if hasattr(inner, "eqns"):
                    names |= _names(inner)
    return names


def _scalar_name(scalar):
    # The FORMATS name of a step's scalar (a binary format's DType JSON), or None without a scalar.
    if scalar is None:
        return None
    (name,) = [name for name, (precision, _, emax) in FORMATS.items()
               if scalar == {"kind": "binary", "exponent_bits": (emax + 1).bit_length(),
                             "fraction_bits": precision - 1, "bias": emax, "encoding": "ieee"}]
    return name


def _ops(document):
    # The records of the artifact's op steps with the names of their scalars, in step order.
    artifact = document["artifact"]
    return [(step["op"]["record"], _scalar_name(scalar))
            for step, scalar in zip(artifact["nodes"], artifact["scalars"]) if "op" in step]


def _registered(document):
    # The records of the registered operators with index ports, in step order: the op steps that
    # read a decodeIndex step or that an encodeIndex step reads.
    steps = _steps(document)
    tags = [step["op"]["record"]["tag"] if "op" in step else None for step in steps]
    encoded = {steps[at]["op"]["inputs"][0][0] for at, tag in enumerate(tags)
               if tag == "encodeIndex"}
    return [step["op"]["record"] for at, step in enumerate(steps) if "op" in step
            and (at in encoded or "decodeIndex" in [tags[j] for j, _ in step["op"]["inputs"]])]


def _words(report):
    # The executed result words of every output, unsigned.
    return [list(output["bits"]) for output in report["outputs"]]


def _float_words(values, dtype):
    # The words of Python floats rounded to the NumPy floating type `dtype`.
    import numpy as np

    array = np.asarray(values, dtype=dtype)
    unsigned = {2: np.uint16, 4: np.uint32, 8: np.uint64}[array.dtype.itemsize]
    return [int(word) for word in array.reshape(-1).view(unsigned)]


def _float_of_word(word, dtype):
    import numpy as np

    width = np.dtype(dtype).itemsize
    unsigned = {2: np.uint16, 4: np.uint32, 8: np.uint64}[width]
    return float(np.asarray([word], dtype=unsigned).view(dtype)[0])


def _midpoints(value, dtype):
    # The midpoints between the positive finite `value` of the format `dtype` and its two
    # neighbours, as exact rationals. Zero below the least subnormal, None above the largest
    # finite value.
    import math
    from fractions import Fraction

    word = _float_words([value], dtype)[0]
    below, above = _float_of_word(word - 1, dtype), _float_of_word(word + 1, dtype)
    low = (Fraction(below) + Fraction(value)) / 2 if word > 0 else Fraction(0)
    high = None if math.isinf(above) else (Fraction(value) + Fraction(above)) / 2
    return low, high


def _cube_root(value, dtype):
    # The value of the format `dtype` nearest the real cube root of the finite Python float
    # `value`, certified exactly: the candidate from mpmath at 200 bits is accepted when the cubes
    # of the midpoints around it bracket |value| strictly (no tie can occur: a midpoint has one
    # more significant bit than the format, so its cube is never a value of the format).
    import math
    from fractions import Fraction

    import mpmath
    import numpy as np

    if value == 0 or math.isinf(value) or value != value:
        return value
    magnitude = Fraction(abs(value))
    with mpmath.workprec(200):
        sign, man, shift, _ = mpmath.cbrt(mpmath.mpf(abs(value)))._mpf_
    candidate = _nearest(Fraction(int(man)) * Fraction(2) ** shift, np.dtype(dtype).name)
    low, high = _midpoints(candidate, dtype)
    if not (low ** 3 < magnitude and (high is None or magnitude < high ** 3)):
        raise AssertionError(f"uncertified cube root of {value}")
    return -candidate if value < 0 else candidate


def _exp(value, dtype):
    # The value of the format `dtype` nearest exp(value), certified: mpmath's interval arithmetic
    # (`mpmath.iv`, outward-rounded bounds) encloses exp(value), and the enclosure lies strictly
    # between the rounding boundaries around the candidate: the midpoints with its neighbours,
    # half the least subnormal above zero, and the overflow threshold below an infinity.
    import math
    from fractions import Fraction

    import mpmath
    import numpy as np

    if value != value:
        return value
    if math.isinf(value):
        return 0.0 if value < 0 else value
    exact = lambda raw: (-1 if raw[0] else 1) * Fraction(int(raw[1])) * Fraction(2) ** raw[2]
    precision, emin, emax = FORMATS[np.dtype(dtype).name]
    least = Fraction(2) ** (emin - precision + 1)
    threshold = (2 - Fraction(2) ** -precision) * Fraction(2) ** emax
    for bits in (120, 400, 2000):
        mpmath.iv.prec = bits
        low, high = (exact(raw) for raw in mpmath.iv.exp(mpmath.iv.mpf(value))._mpi_)
        candidate = _nearest(low, np.dtype(dtype).name)
        if candidate == 0:
            below, above = Fraction(0), least / 2
        elif math.isinf(candidate):
            below, above = threshold, None
        else:
            below, above = _midpoints(candidate, dtype)
            if above is None:
                above = threshold
        if below < low and (above is None or high < above):
            return candidate
    raise AssertionError(f"uncertified exp of {value}")
