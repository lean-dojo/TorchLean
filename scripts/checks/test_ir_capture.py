"""Capture framework computations and check them with ``verify ir``.

Each case captures a tiny computation, then requests structure, type and execution checks
from the built ``verify`` executable. Operations whose IR arithmetic is PyTorch's or XLA's also
compare interchange words with the frontend, in binary32, bfloat16, binary16 and integers. Where
XLA's compiled CPU code departs from JAX's meaning (reciprocals of broadcast and constant
divisors, jax.jit's rewrites, subnormal flushing, its own rsqrt and power), the cases pin the IR's
IEEE result and check that ``compare`` fails and lists the behaviour in ``consistent_with`` only
when its emulation of XLA's code reproduces the reference, and lists nothing for importer bugs
placed next to those operations, including those that coincide with jax.jit's rewrites.
Cases skip when the checker has not been built or a frontend is not installed.
"""

import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

import torchlean  # noqa: E402

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


def _nearest(value, dtype):
    # The exact rational `value` (an int, a float or a Fraction) rounded to the format named
    # `dtype`, to nearest with ties to even, an infinity from the overflow threshold on, as a
    # Python float. A float zero, infinity or NaN passes through with its sign.
    import math
    from fractions import Fraction

    if isinstance(value, float) and (value != value or math.isinf(value) or value == 0):
        return value
    precision, emin, emax = FORMATS[dtype]
    exact = Fraction(value)
    if exact == 0:
        return 0.0
    magnitude = abs(exact)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if Fraction(2) ** exponent > magnitude:
        exponent -= 1
    quantum = Fraction(2) ** (max(exponent, emin) - precision + 1)
    rounded = round(magnitude / quantum) * quantum
    result = math.inf if rounded >= 2 ** (emax + 1) else float(rounded)
    return -result if exact < 0 else result


def _splitmix64(value):
    # One output of SplitMix64 for the 64-bit word `value`: the tests' reference for the
    # keyed draws of TorchLean's generator.
    z = (value + 0x9e3779b97f4a7c15) % 2 ** 64
    z = ((z ^ (z >> 30)) * 0xbf58476d1ce4e5b9) % 2 ** 64
    z = ((z ^ (z >> 27)) * 0x94d049bb133111eb) % 2 ** 64
    return z ^ (z >> 31)


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


class OperatorTableTests(unittest.TestCase):
    def test_environment_overrides_select_their_own_table(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from torchlean import _table

        def table(source):
            return {"format": source, "version": 1, "dtypes": {}, "codes": {}, "pointwise": {}}

        def checker(source):
            source = source if source is not None else os.environ.get("TORCHLEAN_VERIFY")
            return SimpleNamespace(op_table=lambda *_: json.dumps(table(source)))

        # Changing the default checker must not reuse the previous checker's codec records.
        with patch.dict(_table._TABLES, {}, clear=True), patch.dict(os.environ), \
                patch.object(_table, "Checker", checker):
            os.environ.pop("TORCHLEAN_OP_TABLE", None)
            for source in ("checker-A", "checker-B", "checker-A"):
                os.environ["TORCHLEAN_VERIFY"] = source
                self.assertEqual(_table.op_table()["format"], source)
            self.assertEqual(_table.op_table("explicit-checker")["format"], "explicit-checker")

            # A path appends `ir`, but a command sequence is executed exactly as supplied.
            def command_checker(source):
                command = [str(source), "ir"] if isinstance(source, (str, Path)) else source
                return SimpleNamespace(op_table=lambda *_: json.dumps(table(" ".join(command))))

            with patch.object(_table, "Checker", command_checker):
                self.assertEqual(_table.op_table("command-checker")["format"], "command-checker ir")
                self.assertEqual(_table.op_table(["command-checker"])["format"], "command-checker")
                self.assertEqual(_table.op_table(["command-checker", "ir"])["format"],
                                 "command-checker ir")

        # A table file overrides that choice, and switching files changes the codec records too.
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(_table._TABLES, {}, clear=True), patch.dict(os.environ):
            for source in ("table-A", "table-B", "table-A"):
                path = Path(temporary) / source
                path.write_text(json.dumps(table(source)), encoding="utf-8")
                os.environ["TORCHLEAN_OP_TABLE"] = str(path)
                self.assertEqual(_table.op_table()["format"], source)

            # Rebuilding a checker or rewriting a table in place also changes its meaning.
            modified = path.stat().st_mtime_ns + 1_000_000
            path.write_text(json.dumps(table("table-C")), encoding="utf-8")
            os.utime(path, ns=(path.stat().st_atime_ns, modified))
            self.assertEqual(_table.op_table()["format"], "table-C")
            os.environ.pop("TORCHLEAN_OP_TABLE")

            def file_checker(source):
                return SimpleNamespace(op_table=lambda *_: path.read_text(encoding="utf-8"))

            with patch.object(_table, "Checker", file_checker):
                self.assertEqual(_table.op_table(path)["format"], "table-C")
                modified = path.stat().st_mtime_ns + 1_000_000
                path.write_text(json.dumps(table("table-D")), encoding="utf-8")
                os.utime(path, ns=(path.stat().st_atime_ns, modified))
                self.assertEqual(_table.op_table(path)["format"], "table-D")


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


@unittest.skipUnless(CHECKER and HAS_TORCH, "needs the built verify executable and PyTorch")
class PyTorchCaptureTests(CheckerCase):
    def setUp(self):
        import torch

        self.torch = torch
        generator = torch.Generator().manual_seed(0)
        self.x = torch.randn(3, 4, generator=generator)
        self.y = torch.randn(3, 4, generator=generator)

    def verify(self, function, *inputs, checks=EXECUTE):
        return torchlean.verify(function, *inputs, checks=checks, checker=CHECKER)

    def compare_fx(self, directory, module, *inputs):
        # The FX capture of `module`, torch.export skipped, compared with its eager outputs.
        from torchlean._fx import _tensor_value
        from torchlean.export import export_torchlean_graph_json

        artifact = export_torchlean_graph_json(module, inputs, str(Path(directory) / "graph.json"),
                                               prefer_torch_export=False)
        artifact["arguments"] = [_tensor_value(x) for x in inputs]
        outputs = module(*inputs)
        outputs = outputs if isinstance(outputs, tuple) else (outputs,)
        artifact["reference_outputs"] = [_tensor_value(y) for y in outputs]
        return artifact, torchlean.verify(artifact, checks=COMPARE, checker=CHECKER)

    def test_exact_operations_match_eager_words(self):
        # Quotients, negation and the powers PyTorch computes exactly rounded too: x * x for the
        # exponent 2, 1 / x for -1, ones and a copy.
        torch = self.torch

        def exact(x, y):
            return (torch.div(x, y), torch.div(x, y, rounding_mode="trunc"),
                    torch.div(x, y, rounding_mode="floor"), x / 2.5, x // 0.75,
                    torch.neg(x), torch.pow(x, 2), torch.pow(x, -1), torch.pow(x, 0),
                    torch.pow(x, 1))

        self.assert_passes(self.verify(exact, self.x, self.y, checks=COMPARE))

    def test_integer_division_negation_and_power(self):
        torch = self.torch
        a = torch.tensor([7, -7, 9, -9], dtype=torch.int32)
        b = torch.tensor([2, 2, -4, -4], dtype=torch.int32)

        def integers(a, b):
            return (torch.div(a, b, rounding_mode="trunc"), torch.div(a, b, rounding_mode="floor"),
                    -a, torch.pow(a, 3), torch.div(a, b))

        self.assert_passes(self.verify(integers, a, b, checks=COMPARE))

    def test_top_k_capture_keeps_unchecked_cutoff_assumption(self):
        # Tied scores can change which experts PyTorch selects. Preserve the lowering's
        # assumption in both artifact and report, even if this example happens to agree.
        # Empty and full masks are independent of the tie rule.
        torch = self.torch

        def masks(x):
            return tuple(torch.zeros_like(x).scatter(
                -1, torch.topk(x, k, dim=-1).indices, 1.0) for k in (0, 2, 4))

        with tempfile.TemporaryDirectory() as directory:
            for values in ([1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 1.0, 0.0]):
                with self.subTest(values=values):
                    path = Path(directory) / ("tied.json" if values[0] == 1 else "gap.json")
                    report = torchlean.verify(masks, torch.tensor([values]), checks=EXECUTE,
                                              checker=CHECKER, export=path)
                    self.assert_passes(report)
                    artifact = json.loads(path.read_text(encoding="utf-8"))
                    constraints = artifact["capture_constraints"]
                    self.assertEqual(len(constraints), 1)
                    constraint = constraints[0]
                    self.assertEqual(constraint["kind"], "top_k_mask_agreement")
                    self.assertEqual(constraint["k"], 2)
                    self.assertEqual(constraint["axis"], 1)
                    self.assertEqual(constraint["status"], "not_checked")
                    self.assertEqual(constraint["assumptions"],
                                     ["finite_scores", "strict_cutoff_gap"])
                    op = _steps(artifact)[constraint["ref"][0]]["op"]
                    self.assertEqual(op["record"]["tag"], "topKMask")
                    self.assertEqual(op["inputs"], [constraint["value_ref"]])
                    self.assertEqual(report["capture_constraints"], constraints)
                    self.assertEqual(report["capture_constraints_status"], "not_checked")

    def test_reduced_division_and_product_match_eager_words(self):
        # Reduced-precision division and multiplication run in binary32 when the right operand is
        # one element, with that element's own value: a Python scalar, a binary32 rank-zero
        # tensor, a one-element tensor, as in PyTorch's kernels. (A one-element left operand is
        # test_scalar_operands_keep_their_value_on_either_side.)
        torch = self.torch
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                def division(x, y, scalar, single):
                    return (torch.div(x, y, rounding_mode="floor"), x / 0.3, x // 0.3,
                            torch.div(x, scalar, rounding_mode="trunc"),
                            torch.div(x, single, rounding_mode="floor"), x * scalar, y * single)

                self.assert_passes(self.verify(
                    division, (self.x * 100).to(dtype), self.y.to(dtype), torch.tensor(0.3),
                    torch.tensor([0.3], dtype=dtype), checks=COMPARE))

    def test_scalar_powers_keep_their_value(self):
        # x ** s is FloatLib's pow of x and the exponent's own value, both in the computation type
        # (binary32 for binary16, bfloat16 and binary32, binary64 for binary64), rounded to the
        # dtype. A Python scalar base keeps its value the same way. FloatLib's pow is correctly
        # rounded on these inputs, so the reference is the power of the converted operands
        # rounded to the computation type, then to the dtype: x ** 3 is the rounded cube, x ** -2
        # and x ** -0.5 the rounded powers, and bfloat16 1000 ** 2.007 is 1048576 (PyTorch rounds
        # the exponent to 2 and multiplies: 999424). A tensor exponent of another dtype converts
        # to the common dtype first.
        import math

        torch = self.torch
        generator = torch.Generator().manual_seed(4)
        x = torch.cat([torch.tensor([1000.0, 200.0, 50.0, 3.0, 2.25, 0.7, -1.5, 9.5]),
                       torch.randn(24, generator=generator) * 3])
        k = torch.tensor([0.0, 1.0, 2.0, 3.0, -1.0, -2.0, 4.0, 5.0, 0.5, -0.5, 2.5, 7.0])
        exponents = (2.007, 1.003, 0.501, 2.001, 3, -2, -0.5, 0.5, -1, 2.5, -3, 2.1, 3.005,
                     -2.001)
        bases = (1.3, 0.1)
        tie = 1 + 2 ** -8 + 2 ** -30

        def powers(x, k, e):
            return tuple(x ** s for s in exponents) + tuple(torch.pow(b, k) for b in bases) + (
                x ** e,)

        for dtype in ("bfloat16", "float16", "float32", "float64"):
            with self.subTest(dtype=dtype):
                at = getattr(torch, dtype)
                wide = "float64" if dtype == "float64" else "float32"
                xs, ks = x.to(at), k.to(at)
                report = self.verify(powers, xs, ks, torch.tensor(tie, dtype=torch.float64))
                self.assert_passes(report)
                outputs = [output.tolist() for output in _outputs(report)]

                def rounded(base, exponent):
                    exact = _power(base, exponent)
                    return _nearest(_nearest(exact, wide) if exact == exact else exact, dtype)

                self.assertEqual(len(outputs), len(exponents) + len(bases) + 1)
                for s, got in zip(exponents, outputs):
                    own = _nearest(s, wide)
                    _same(self, got, [rounded(v, own) for v in xs.tolist()], (dtype, s))
                for b, got in zip(bases, outputs[len(exponents):]):
                    _same(self, got, [rounded(_nearest(b, wide), e) for e in ks.tolist()],
                          (dtype, b))
                common = _nearest(tie, dtype)
                _same(self, outputs[-1], [rounded(v, common) for v in xs.tolist()], dtype)
        reduced = torch.tensor([1000.0, 200.0, 50.0], dtype=torch.bfloat16)
        self.assertEqual((reduced ** 2.007).tolist()[0], 999424.0)
        report = self.verify(lambda v: (v ** 2.007, v ** 1.003, v ** 0.501), reduced)
        self.assertEqual([output.tolist() for output in _outputs(report)],
                         [[1048576.0, 41472.0, 2576.0], [1020.0, 203.0, 50.5],
                          [31.875, 14.1875, 7.09375]])
        # The exponents 0.5, -0.5, -1 and 3 at -inf, -0, +0, +inf, -4 and 4: IEEE 754-2019 9.2.1's
        # pow, where PyTorch's sqrt and rsqrt give pow(-0, 0.5) = -0, pow(-inf, 0.5) = NaN,
        # pow(-0, -0.5) = -inf and pow(-inf, -0.5) = NaN.
        inf, nan = math.inf, math.nan
        special = [-inf, -0.0, 0.0, inf, -4.0, 4.0]
        expected = [[inf, 0.0, 0.0, inf, nan, 2.0], [0.0, inf, inf, 0.0, nan, 0.5],
                    [-0.0, -inf, inf, 0.0, -0.25, 0.25], [-inf, -0.0, 0.0, inf, -64.0, 64.0]]
        self.assertTrue(math.isnan(torch.pow(torch.tensor(-inf), 0.5).item()))
        for dtype in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype, special=True):
                report = self.verify(lambda v: (v ** 0.5, v ** -0.5, v ** -1, v ** 3),
                                     torch.tensor(special, dtype=dtype))
                self.assert_passes(report)
                self.assertEqual(len(_outputs(report)), len(expected))
                for got, want in zip(_outputs(report), expected):
                    _same(self, got.tolist(), want, dtype)
        # A signaling NaN through pow: x ** 0, x ** 1 and 1 ** x are quiet NaNs in binary32 and
        # binary64, as one pow computes them (PyTorch gives 1, the signaling NaN, and 1). A
        # bfloat16 operand is widened to binary32 first, which quiets it, so its x ** 0 is 1.
        for dtype, signaling, quiet in ((torch.float32, 0x7fa00000, 1 << 22),
                                        (torch.float64, 0x7ff4000000000000, 1 << 51)):
            with self.subTest(signaling=dtype):
                integer = torch.int32 if dtype == torch.float32 else torch.int64
                operand = torch.tensor([signaling], dtype=integer).view(dtype)
                report = self.verify(lambda v: (v ** 0, v ** 1, torch.pow(1, v)), operand)
                self.assert_passes(report)
                self.assertEqual(len(report["outputs"]), 3)
                for output in report["outputs"]:
                    word = output["bits"][0]
                    exponent = (0x7f800000 if dtype == torch.float32 else 0x7ff0000000000000)
                    self.assertEqual(word & exponent, exponent, hex(word))
                    self.assertTrue(word & quiet, hex(word))
        self.assertEqual(torch.tensor([0x7fa00000], dtype=torch.int32).view(torch.float32)
                         .pow(0).item(), 1.0)
        signaling = torch.tensor([0x7fa0], dtype=torch.int16).view(torch.bfloat16)
        report = self.verify(lambda v: v ** 0, signaling)
        self.assert_passes(report)
        self.assertEqual(_outputs(report)[0].tolist(), [1.0])
        # Large exponents that binary32 holds exactly are kept, and 2 ** 53 + 1 is refused for a
        # binary64 tensor, where it would become 2 ** 53.
        bases = torch.tensor([1.0, -1.0, 0.5, 2.0])
        report = self.verify(lambda v: (v ** (2 ** 25), v ** (2 ** 24 + 2)), bases)
        self.assert_passes(report)
        self.assertEqual(len(_outputs(report)), 2)
        for got in _outputs(report):
            _same(self, got.tolist(), [1.0, 1.0, 0.0, math.inf])
        self.assert_refused(self.verify(lambda v: v ** (2 ** 53 + 1), bases.double()),
                            "another integer")
        kinds = _tags(self.artifact(lambda v: (v ** 3, v ** 0.5, v ** -0.5, v ** -1, v ** 0,
                                               v ** 1, v ** 2, torch.pow(1, v)), x))
        self.assertEqual(kinds.count("pow"), 8)
        for removed in ("mul", "sqrt", "reciprocal", "rsqrt"):
            self.assertNotIn(removed, kinds)
        # An exponent whose binary32 rounding is another integer, or an integer from a fraction,
        # is refused: its parity or domain would change.
        for exponent in (2 ** 24 + 1, 2.0000000001):
            with self.subTest(exponent=exponent):
                self.assert_refused(self.verify(lambda v: v ** exponent, xs.float()),
                                    "another integer")

    def test_reduced_conversions_round_once(self):
        # Tensor.to, type promotion, Python constants and fill values round the exact value once
        # to bfloat16 and binary16 (IEEE 754-2019 5.4.1, 5.4.2): 16842753 becomes bfloat16
        # 16908288, where c10::convert rounds through binary32 and gives 16777216. Integers
        # convert to binary16 the same either way: binary32 holds every integer below binary16's
        # overflow threshold.
        from fractions import Fraction

        torch = self.torch
        a = torch.tensor([16842753, -16842753, 3, 65519, 65520], dtype=torch.int32)
        b = torch.tensor([16842753, 2 ** 40 + 2 ** 32 + 1, -(2 ** 40 + 2 ** 32 + 1),
                          2 ** 62 + 2 ** 54 + 1], dtype=torch.int64)
        c = torch.tensor([1 + 2 ** -8 + 2 ** -30, -(1 + 2 ** -8 + 2 ** -30),
                          1 + 2 ** -11 + 2 ** -40, 0.1], dtype=torch.float64)
        self.assertEqual(int(a[:1].to(torch.bfloat16).view(torch.int16)), 0x4B80)

        fill = 1 + 2 ** -8 + 2 ** -40

        def conversions(a, b, c, zero, one):
            return (a.to(torch.bfloat16), a.to(torch.float16), b.to(torch.bfloat16),
                    b.to(torch.float16), c.to(torch.bfloat16), c.to(torch.float16),
                    a + zero, b * one, zero + fill, torch.full_like(zero, fill),
                    torch.full((2,), fill, dtype=torch.bfloat16) + one[:2])

        report = self.verify(conversions, a, b, c, torch.zeros(5, dtype=torch.bfloat16),
                             torch.ones(4, dtype=torch.bfloat16))
        self.assert_passes(report)
        outputs = [output.tolist() for output in _outputs(report)]
        self.assertEqual(len(outputs), 11)
        once = [[_nearest(Fraction(v), dtype) for v in source.tolist()]
                for source in (a, b, c) for dtype in ("bfloat16", "float16")]
        for got, want in zip(outputs, once + [once[0], once[2]]):
            _same(self, got, want)
        self.assertEqual(outputs[0][0], 16908288.0)
        _same(self, outputs[8], [_nearest(fill, "bfloat16")] * 5)
        _same(self, outputs[9], [_nearest(fill, "bfloat16")] * 5)
        _same(self, outputs[10], [_nearest(Fraction(_nearest(fill, "bfloat16")) + 1,
                                           "bfloat16")] * 2)
        self.assertEqual(_nearest(fill, "bfloat16"), 1.0078125)
        self.assertEqual(torch.full((1,), fill, dtype=torch.bfloat16).item(), 1.0)
        artifact = self.artifact(conversions, a, b, c, torch.zeros(5, dtype=torch.bfloat16),
                                 torch.ones(4, dtype=torch.bfloat16))
        self.assertNotIn("float32", [_scalar_name(scalar)
                                     for scalar in artifact["artifact"]["scalars"]])
        # An operand a product keeps in binary32 is rounded there first: the int 16842753 is
        # binary32 16842752, which bfloat16 rounds to 16777216 (a tie), while a tensor of two
        # int32 entries converts once, to 16908288.
        ones = torch.ones(3, dtype=torch.bfloat16)
        report = self.verify(lambda o, single, pair: (torch.mul(16842753, o), single * o,
                                                      pair * o[:2]),
                             ones, a[:1], torch.tensor([16842753, 3], dtype=torch.int32))
        self.assert_passes(report)
        self.assertEqual([output.tolist() for output in _outputs(report)],
                         [[16777216.0] * 3, [16777216.0] * 3, [16908288.0, 3.0]])

    def test_comparison_rejects_a_conversion_pytorch_rounds_twice(self):
        # Binary64 1 + 2^-8 + 2^-40 (0x3FF0100000001000) lies just above the bfloat16 midpoint
        # 1 + 2^-8. The IR rounds it once, to 0x3F81 (1.0078125). PyTorch rounds it to binary32
        # first, which gives that midpoint (0x3F808000), and then to even, 0x3F80 (1.0). So the
        # comparison with PyTorch rejects the conversion, by design.
        torch = self.torch
        x = torch.tensor([1 + 2 ** -8 + 2 ** -40, 1.5], dtype=torch.float64)
        self.assertEqual(int(x[0].view(torch.int64)), 0x3FF0100000001000)
        self.assertEqual(int(x[0].to(torch.float32).view(torch.int32)), 0x3F808000)
        self.assertEqual(int(x[0].to(torch.bfloat16).view(torch.int16)), 0x3F80)
        report = self.verify(lambda v: v.to(torch.bfloat16), x, checks=COMPARE)
        statuses = _statuses(report)
        self.assertEqual(statuses["execute"][0], "executed", statuses)
        self.assertEqual(report["outputs"][0]["bits"], [0x3F81, 0x3FC0])
        self.assertEqual(statuses["compare"], ("failed", "IR comparison: results differ in their "
                                               "interchange words: result 0: 1 of 2 elements "
                                               "differ"))

    def test_python_constants_round_once_at_format_boundaries(self):
        # Python constants and fill values become bfloat16 and binary16 words in one rounding,
        # ties to even: signed zeros, the smallest subnormal and the tie below it, the
        # normal/subnormal boundary's midpoint, the overflow threshold, uint64 and int64 limits,
        # an int that binary32 cannot hold, and a binary64 tie neighbour. The words are written
        # out (checked independently by NumPy's float64-to-float16 conversion and a round-to-odd
        # bfloat16 rounding). A fill goes through full_like, new_full, fill and scalar_tensor where
        # PyTorch accepts the value, and every nonzero value also through `zeros + value`.
        torch = self.torch
        cases = [  # value, bfloat16 word, binary16 word
            (0.0, 0x0000, 0x0000), (-0.0, 0x8000, 0x8000),
            (2.0 ** -133, 0x0001, 0x0000), (2.0 ** -134, 0x0000, 0x0000),
            (2.0 ** -134 + 2.0 ** -160, 0x0001, 0x0000),
            (-(2.0 ** -134 + 2.0 ** -160), 0x8001, 0x8000),
            (2.0 ** -24, 0x3380, 0x0001), (2.0 ** -25, 0x3300, 0x0000),
            (2.0 ** -25 + 2.0 ** -40, 0x3300, 0x0001),
            (2.0 ** -126 - 2.0 ** -134, 0x0080, 0x0000), (2.0 ** -14 - 2.0 ** -25, 0x3880, 0x0400),
            (2.0 ** -126 - 2.0 ** -134 - 2.0 ** -150, 0x007f, 0x0000),
            (2.0 ** -126 - 2.0 ** -134 + 2.0 ** -150, 0x0080, 0x0000),
            (-(2.0 ** -126 - 2.0 ** -134), 0x8080, 0x8000),
            (2.0 ** -14 - 2.0 ** -25 - 2.0 ** -40, 0x3880, 0x03ff),
            (2.0 ** -14 - 2.0 ** -25 + 2.0 ** -40, 0x3880, 0x0400),
            (-(2.0 ** -14 - 2.0 ** -25), 0xb880, 0x8400),
            (65520.0, 0x4780, 0x7c00), (65520.0 - 2.0 ** -10, 0x4780, 0x7bff),
            (-65520.0, 0xc780, 0xfc00), (2.0 ** 128 - 2.0 ** 119, 0x7f80, 0x7c00),
            (2.0 ** 128 - 2.0 ** 119 - 2.0 ** 100, 0x7f7f, 0x7c00),
            (-(2.0 ** 128 - 2.0 ** 119), 0xff80, 0xfc00), (2 ** 64 - 1, 0x5f80, 0x7c00),
            (2 ** 63, 0x5f00, 0x7c00), (-2 ** 63, 0xdf00, 0xfc00), (16842753, 0x4b81, 0x7c00),
            (1 + 2.0 ** -8 + 2.0 ** -40, 0x3f81, 0x3c04), (0.1, 0x3dcd, 0x2e66),
            (True, 0x3f80, 0x3c00)]
        def fits(value, dtype):
            # PyTorch refuses a fill value its dtype overflows.
            try:
                torch.full((1,), value, dtype=dtype)
                return True
            except RuntimeError:
                return False

        for dtype, column in ((torch.bfloat16, 1), (torch.float16, 2)):
            zero = torch.zeros(2, dtype=dtype)
            filled = [(case[0], case[column]) for case in cases if fits(case[0], dtype)]
            added = [(case[0], case[column]) for case in cases if case[0] != 0]
            paths = {"full_like": lambda z: tuple(torch.full_like(z, v) for v, _ in filled),
                     "new_full": lambda z: tuple(z.new_full((2,), v) for v, _ in filled),
                     "fill": lambda z: tuple(torch.fill(z, v) for v, _ in filled),
                     "scalar_tensor": lambda z: tuple(torch.scalar_tensor(v, dtype=z.dtype) * 1
                                                      for v, _ in filled),
                     # +0 plus the constant: its word, a signed zero becoming +0.
                     "add": lambda z: tuple(z + v for v, _ in added)}
            for name, function in paths.items():
                with self.subTest(dtype=dtype, path=name):
                    report = self.verify(function, zero)
                    self.assert_passes(report)
                    expected = ([word for _, word in filled] if name != "add" else
                                [word if word & 0x7fff else 0 for _, word in added])
                    self.assertEqual(len(report["outputs"]), len(expected))
                    for output, word in zip(report["outputs"], expected):
                        self.assertEqual(output["bits"],
                                         [word] * (1 if name == "scalar_tensor" else 2),
                                         (name, hex(word)))

    def test_fx_creation_constants_are_pytorchs_tensors(self):
        # torch.export records torch.full, whose bfloat16 fill the capture rounds once
        # (1 + 2^-8 + 2^-40 is 1.0078125). FX evaluates the input-independent creation while
        # tracing and keeps PyTorch's tensor as a constant, rounded through binary32 (1.0).
        torch = self.torch
        fill = 1 + 2 ** -8 + 2 ** -40

        class Filled(torch.nn.Module):
            def forward(self, z):
                return z + torch.full((2,), fill, dtype=torch.bfloat16)

        zero = torch.zeros(2, dtype=torch.bfloat16)
        report = self.verify(Filled(), zero)
        self.assert_passes(report)
        self.assertEqual(report["outputs"][0]["bits"], [0x3f81, 0x3f81])
        with tempfile.TemporaryDirectory() as directory:
            report = self.execute_fx(directory, Filled(), zero)[1]
        self.assert_passes(report)
        self.assertEqual(report["outputs"][0]["bits"], [0x3f80, 0x3f80])

    def test_python_scalar_operands(self):
        # A scalar is a constant: add and sub round it to the tensor dtype, and reduced-precision
        # mul multiplies by its binary32 value. An int beyond 2 ** 53 rounds once to binary32,
        # from int64.
        torch = self.torch
        large = 2 ** 60 + 2 ** 36 + 1
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                def scalars(x):
                    return x * 0.3, 1.7 * x, x + 0.1, x - 0.3, x * 3

                self.assert_passes(self.verify(scalars, self.x.to(dtype), checks=COMPARE))
        self.assert_passes(self.verify(lambda x: (x + large, x * large, x / large), self.x,
                                       checks=COMPARE))

    def execute_fx(self, directory, module, *inputs):
        # The FX capture of `module`, torch.export skipped, and its execution, without a reference.
        from torchlean._fx import _tensor_value
        from torchlean.export import export_torchlean_graph_json

        artifact = export_torchlean_graph_json(module, inputs, str(Path(directory) / "graph.json"),
                                               prefer_torch_export=False)
        artifact["arguments"] = [_tensor_value(x) for x in inputs]
        return artifact, torchlean.verify(artifact, checks=EXECUTE, checker=CHECKER)

    def test_scalar_operands_keep_their_value_on_either_side(self):
        # Python's s / x is true division, the quotient of the converted operands rounded once,
        # where PyTorch's Tensor.__rtruediv__ computes reciprocal(x) * s. torch.export records
        # that as a reciprocal and a product tagged with one __rdiv__ call, FX as
        # operator.truediv(s, x), and both become one div. A binary16 or bfloat16 product or
        # quotient with an operand that supplies one element keeps that element's own value in
        # binary32 on either side, computes there and rounds once to the dtype, so torch.mul(s, x)
        # is x * s and torch.div(s, x) is s / x (PyTorch converts a left scalar to the tensor's
        # dtype: its bfloat16 torch.mul(0.1, x) is bfloat16(0.1) * x). A written-out
        # x.reciprocal() * s stays a reciprocal and a product.
        import math
        from fractions import Fraction

        from torchlean._fx import _tensor_value
        from torchlean._pointwise import _reflected_division
        from torchlean.export import export_torchlean_graph_json

        torch = self.torch
        x = torch.randn(64, generator=torch.Generator().manual_seed(1)) * 4
        third = torch.tensor(0.3)

        def scalars(x, third):
            return (0.1 / x, torch.div(0.1, x), 3 / x, 1000 // x, torch.mul(0.1, x),
                    torch.mul(257, x), 0.1 * x, torch.mul(third, x), x * third)

        class Reflected(torch.nn.Module):
            def forward(self, x):
                return 0.1 / x, torch.mul(0.1, x), x.reciprocal() * 0.1

        with tempfile.TemporaryDirectory() as directory:
            for dtype in ("bfloat16", "float16", "float32", "float64"):
                with self.subTest(dtype=dtype):
                    values = x.to(getattr(torch, dtype))
                    wide = dtype if dtype in ("float32", "float64") else "float32"
                    report = self.verify(scalars, values, third)
                    self.assert_passes(report)
                    outputs = [output.tolist() for output in _outputs(report)]
                    points = values.tolist()

                    def rounded(exact):
                        return _nearest(_nearest(exact, wide), dtype)

                    def scalar(s):
                        return Fraction(_nearest(s, wide))

                    expected = [[rounded(scalar(s) / Fraction(v)) for v in points]
                                for s in (0.1, 0.1, 3)]
                    expected.append([rounded(math.floor(Fraction(1000) / Fraction(v)))
                                     for v in points])
                    expected += [[rounded(scalar(s) * Fraction(v)) for v in points]
                                 for s in (0.1, 257, 0.1)]
                    one = Fraction(third.item()) if wide == "float32" else Fraction(
                        _nearest(third.item(), dtype))
                    expected += [[rounded(one * Fraction(v)) for v in points]] * 2
                    self.assertEqual(len(outputs), len(expected))
                    for got, want in zip(outputs, expected):
                        _same(self, got, want, dtype)
                    artifact, report = self.execute_fx(directory, Reflected(), values)
                    self.assert_passes(report)
                    kinds = _tags(artifact)
                    self.assertEqual(kinds.count("div"), 1)
                    self.assertEqual(kinds.count("reciprocal"), 1)
                    self.assertEqual(len(_outputs(report)), 3)
                    reflected, product, reciprocal = (o.tolist() for o in _outputs(report))
                    _same(self, reflected, expected[0], dtype)
                    _same(self, product, expected[4], dtype)
                    # reciprocal(x) in the dtype, then the product with binary32 0.1 (binary16
                    # and bfloat16) or the dtype's 0.1, rounded once.
                    _same(self, reciprocal,
                          [rounded(scalar(0.1) * Fraction(_nearest(Fraction(1) / Fraction(v),
                                                                     dtype)))
                           for v in points], dtype)
            exported = self.artifact(lambda v: (0.1 / v, v.reciprocal() * 0.1), x)
            self.assertEqual(_tags(exported).count("div"), 1)
            self.assertEqual(_tags(exported).count("reciprocal"), 1)
        reduced = x.to(torch.bfloat16)
        self.assertNotEqual(torch.mul(0.1, reduced).view(torch.int16).tolist(),
                            (0.1 * reduced).view(torch.int16).tolist())
        # The flooring schedule floors the exact quotient, so 1 // 0.1 is 9 because the binary32
        # and bfloat16 values of 0.1 exceed 0.1, where the floor of the rounded quotient would be
        # 10. A zero result takes the quotient's sign.
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(floor=dtype):
                report = self.verify(lambda v, w: (1 // v, -0.0 // w),
                                     torch.tensor([0.1, 0.3, -0.1], dtype=dtype),
                                     torch.tensor([1.0, -1.0], dtype=dtype))
                self.assert_passes(report)
                self.assertEqual(len(_outputs(report)), 2)
                _same(self, _outputs(report)[0].tolist(), [9.0, 3.0, -10.0])
                _same(self, _outputs(report)[1].tolist(), [-0.0, 0.0])

        for left, right in ((lambda v: torch.mul(0.1, v), lambda v: v * 0.1),
                            (lambda v, t: torch.mul(t, v), lambda v, t: v * t)):
            arguments = (reduced,) if left.__code__.co_argcount == 1 else (reduced, third)
            graph = self.artifact(left, *arguments)
            self.assertEqual(graph["artifact"], self.artifact(right, *arguments)["artifact"])
            self.assertEqual([scalar for record, scalar in _ops(graph) if record["tag"] == "mul"],
                             ["float32"])

        class Quotient(torch.nn.Module):
            def forward(self, v):
                return 0.1 / v

        with tempfile.TemporaryDirectory() as directory:
            graph = torch.export.export(Quotient(), (x,)).graph
            product = next(node for node in graph.nodes if node.op == "call_function"
                           and str(node.target) == "aten.mul.Tensor")
            reciprocal = product.args[0]
            self.assertEqual(_reflected_division(product), (0.1, reciprocal.args[0]))
            for tagged in (reciprocal, product):
                saved = tagged.meta.pop("torch_fn")
                self.assertIsNone(_reflected_division(product))
                tagged.meta["torch_fn"] = saved
            with graph.inserting_after(reciprocal):
                graph.call_function(torch.ops.aten.neg.default, (reciprocal,))
            self.assertIsNone(_reflected_division(product))
            # A reciprocal that is also a program output keeps the reciprocal and the product.
            shared = torch.export.export(Quotient(), (x,)).graph
            product = next(node for node in shared.nodes if node.op == "call_function"
                           and str(node.target) == "aten.mul.Tensor")
            output = next(node for node in shared.nodes if node.op == "output")
            output.args = ((product, product.args[0]),)
            placeholder = next(node.name for node in shared.nodes if node.op == "placeholder")
            artifact = torchlean._lower_pytorch_graph(Quotient(), shared, [placeholder])[0]
            self.assertEqual((_tags(artifact).count("div"), _tags(artifact).count("reciprocal")),
                             (0, 1))
            artifact["arguments"] = [_tensor_value(x)]
            report = torchlean.verify(artifact, checks=EXECUTE, checker=CHECKER)
            self.assert_passes(report)
            self.assertEqual(len(_outputs(report)), 2)
            inverse = [_nearest(Fraction(1) / Fraction(v), "float32") for v in x.tolist()]
            _same(self, _outputs(report)[1].tolist(), inverse)
            _same(self, _outputs(report)[0].tolist(),
                  [_nearest(Fraction(_nearest(0.1, "float32")) * Fraction(v), "float32")
                   for v in inverse])
            artifact = export_torchlean_graph_json(
                Quotient(), (x,), str(Path(directory) / "quotient.json"),
                require_torch_export=True)
            self.assertEqual(_tags(artifact).count("div"), 1)
            self.assertNotIn("reciprocal", _tags(artifact))
            artifact["arguments"] = [_tensor_value(x)]
            report = torchlean.verify(artifact, checks=EXECUTE, checker=CHECKER)
            self.assert_passes(report)
            _same(self, _outputs(report)[0].tolist(),
                  [_nearest(Fraction(_nearest(0.1, "float32")) / Fraction(v), "float32")
                   for v in x.tolist()])

    def artifact(self, function, *inputs):
        # The document the capture writes, before the checker reads it.
        return torchlean._capture(function, inputs, "pytorch", ("structure",))

    def test_clamp_min_bound_converts_once(self):
        # A Python int bound converts from the int64 PyTorch wraps it in, rounded once:
        # 2 ** 60 + 2 ** 36 + 1 becomes binary32 2 ** 60 + 2 ** 37, where binary64 gives 2 ** 60.
        torch = self.torch
        large = 2 ** 60 + 2 ** 36 + 1

        class Clamped(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0, -3.0, 2.0 ** 61]))

            def forward(self, x):
                return torch.clamp_min(self.weight, large) + x

        zero = torch.zeros(3)
        with torch.no_grad():
            self.assertEqual(Clamped()(zero)[0].item(), 2 ** 60 + 2 ** 37)
        report = self.verify(Clamped(), zero)
        self.assert_passes(report)
        _same(self, _outputs(report)[0].tolist(), [2.0 ** 60 + 2.0 ** 37] * 2 + [2.0 ** 61])

    def test_clamp_min_bound_converts_to_the_result_type(self):
        # clamp_min of an integer tensor with a float bound computes in the promoted binary32:
        # an int32 tensor clamped at 0.5 gives binary32 0.5 where the tensor is below it. A
        # buffer is captured by torch.export. FX folds a computation on buffers alone into a
        # constant, so an integer input exercises its lowering.
        torch = self.torch

        class Clamped(torch.nn.Module):
            def __init__(self, buffer, bound):
                super().__init__()
                self.register_buffer("buffer", buffer)
                self.bound = bound

            def forward(self, x):
                return torch.clamp_min(self.buffer, self.bound) + x

        class ClampedInput(torch.nn.Module):
            def __init__(self, bound):
                super().__init__()
                self.bound = bound

            def forward(self, n):
                return torch.clamp_min(n, self.bound), n.clamp_min(self.bound)

        zero = torch.zeros(64)
        with tempfile.TemporaryDirectory() as directory:
            for dtype in (torch.int32, torch.int64):
                integers = torch.arange(-32, 32, dtype=dtype)
                for bound in (0.5, 0.1, 0.3):
                    expected = [_ieee_maximum(float(n), _nearest(bound, "float32"))
                                for n in integers.tolist()]
                    with self.subTest(dtype=dtype, bound=bound):
                        model = Clamped(integers, bound)
                        report = self.verify(model, zero)
                        self.assert_passes(report)
                        _same(self, _outputs(report)[0].tolist(), expected)
                        model = ClampedInput(bound)
                        self.assertEqual(model(integers)[0].dtype, torch.float32)
                        for report in (self.verify(model, integers),
                                       self.execute_fx(directory, model, integers)[1]):
                            self.assert_passes(report)
                            outputs = _outputs(report)
                            self.assertEqual(len(outputs), 2)
                            for got in outputs:
                                _same(self, got.tolist(), expected)

    def test_clamp_min_of_any_tensor(self):
        # The bound's dtype comes from the result, so clamp_min applies to inputs and
        # intermediate values as well as to parameters and buffers.
        torch = self.torch
        x = torch.randn(256, generator=torch.Generator().manual_seed(3))

        class Clamped(torch.nn.Module):
            def forward(self, x):
                return torch.clamp_min(x, 0.1), x.clamp_min(-0.3), (x * 2).clamp_min(0.3)

        with tempfile.TemporaryDirectory() as directory:
            for dtype in ("bfloat16", "float16", "float32"):
                values = x.to(getattr(torch, dtype))
                points = values.tolist()
                expected = [[_ieee_maximum(v, _nearest(0.1, dtype)) for v in points],
                            [_ieee_maximum(v, _nearest(-0.3, dtype)) for v in points],
                            [_ieee_maximum(2 * v, _nearest(0.3, dtype)) for v in points]]
                with self.subTest(dtype=dtype):
                    for report in (self.verify(Clamped(), values),
                                   self.execute_fx(directory, Clamped(), values)[1]):
                        self.assert_passes(report)
                        outputs = [output.tolist() for output in _outputs(report)]
                        self.assertEqual(len(outputs), len(expected))
                        for got, want in zip(outputs, expected):
                            _same(self, got, want, dtype)

    def test_clamp_min_is_ieee_maximum(self):
        # clamp_min(x, min) is max(x, min), IEEE 754-2019 maximum: -0 below +0, a NaN operand or
        # bound gives a NaN, so clamp_min(-0.0, 0) is +0.0, where PyTorch's scalar-bound kernel
        # keeps -0.0. clamp(x, min=s) is the same, and a tensor bound too, promoted with x as
        # torch.maximum promotes, in both the torch.export and the FX capture.
        import math

        torch = self.torch
        values = [-0.0, 0.0, float("nan"), -float("inf"), float("inf"), -1.0, 1.0, 0.5]
        self.assertEqual(torch.clamp_min(torch.tensor([-0.0]), 0).view(torch.int32).item(),
                         -2 ** 31)

        maximum, minimum = _ieee_maximum, _ieee_minimum
        bounds = (0, 0.0, -0.0, float("nan"))

        class Clamped(torch.nn.Module):
            # For each scalar bound: three spellings of max(x, bound), three of min(x, bound).
            def forward(self, x):
                return tuple(y for bound in bounds for y in (
                    torch.clamp_min(x, bound), x.clamp_min(bound), torch.clamp(x, min=bound),
                    torch.clamp_max(x, bound), x.clamp_max(bound), torch.clamp(x, max=bound)))

        class Bounded(torch.nn.Module):
            # The maximum forms, then the minimum forms.
            def forward(self, x, bound):
                return (torch.clamp_min(x, bound), x.clamp_min(bound), torch.clamp_min(bound, x),
                        torch.clamp(x, min=bound), torch.clamp_max(x, bound),
                        x.clamp_max(bound), torch.clamp(x, max=bound))

        with tempfile.TemporaryDirectory() as directory:
            for dtype in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
                x = torch.tensor(values, dtype=dtype)
                expected = [[extremum(v, float(bound)) for v in x.tolist()]
                            for bound in bounds for extremum in (maximum,) * 3 + (minimum,) * 3]
                with self.subTest(dtype=dtype):
                    for report in (self.verify(Clamped(), x),
                                   self.execute_fx(directory, Clamped(), x)[1]):
                        self.assert_passes(report)
                        self.assertEqual(len(_outputs(report)), len(expected))
                        for got, want in zip(_outputs(report), expected):
                            _same(self, got.tolist(), want)
                limits = torch.tensor(values[::-1], dtype=dtype)
                with self.subTest(dtype=dtype, tensor_bound=True):
                    for report in (self.verify(Bounded(), x, limits),
                                   self.execute_fx(directory, Bounded(), x, limits)[1]):
                        self.assert_passes(report)
                        pairs = list(zip(x.tolist(), limits.tolist()))
                        outputs = _outputs(report)
                        self.assertEqual(len(outputs), 7)
                        for got in outputs[:4]:
                            _same(self, got.tolist(), [maximum(a, b) for a, b in pairs])
                        for got in outputs[4:]:
                            _same(self, got.tolist(), [minimum(a, b) for a, b in pairs])
        integers = torch.tensor([-3, 0, 2, 7], dtype=torch.int32)
        report = self.verify(lambda n, b: torch.clamp_min(n, b), integers,
                             torch.tensor([0.5, -0.0, 2.5, float("nan")]))
        self.assert_passes(report)
        _same(self, _outputs(report)[0].tolist(), [0.5, 0.0, 2.5, math.nan])
        kinds = _tags(self.artifact(Bounded(), x, limits)) + _tags(self.artifact(Clamped(), x))
        self.assertEqual((kinds.count("max"), kinds.count("min")), (16, 15))

    def test_positional_variance_flags_and_flip_dims(self):
        # FX and Dynamo keep a call's arguments as written: the unbiased flag of x.var(1, False),
        # torch.var(x, 1, False) and torch.var(x, False) stays positional, and so do the dims of
        # x.flip(0, 1). The variance is the sum of the squared deviations from the mean divided by
        # n for unbiased=False and by n - 1 for unbiased=True. Every partial result here is exact,
        # so each output is that quotient rounded once to binary32. The importer refuses a list
        # among several flip arguments, on a constructed graph: PyTorch can reject
        # x.flip((0,), 1) before the importer sees it.
        from fractions import Fraction

        from torchlean._fx import _constant

        torch = self.torch

        class Positional(torch.nn.Module):
            def forward(self, x):
                return (x.var(1, False), torch.var(x, 1, False), torch.var(x, False),
                        torch.var(x.reshape(-1), False), x.var(-1, False, True),
                        torch.var(x, 1, True), x.flip(0, 1), x.flip(0, -1))

        x = torch.tensor([[1., 3., 5., 7.], [2., 2., 4., 8.]])
        rows = [[Fraction(v) for v in row] for row in x.tolist()]

        def variance(values, correction):
            mean = sum(values) / len(values)
            squares = sum((v - mean) ** 2 for v in values)
            return _nearest(squares / (len(values) - correction), "float32")

        biased, every = [variance(row, 0) for row in rows], variance(rows[0] + rows[1], 0)
        self.assertEqual(biased + [every], [5.0, 6.0, 5.5])
        flipped = [v for row in x.tolist()[::-1] for v in row[::-1]]
        expected = [([2], biased), ([2], biased), ([], [every]), ([], [every]),
                    ([2, 1], biased), ([2], [variance(row, 1) for row in rows]),
                    ([2, 4], flipped), ([2, 4], flipped)]
        with tempfile.TemporaryDirectory() as directory:
            model = torchlean.compile(Positional(), checker=CHECKER)
            model(x)
            self.assert_passes(model.torchlean.report)
            compiled = model.torchlean.artifact
            compiled["arguments"] = [_constant(x)]
            reports = {"torch.export": self.verify(Positional(), x),
                       "fx": self.execute_fx(directory, Positional(), x)[1],
                       "compile": torchlean.verify(compiled, checks=EXECUTE, checker=CHECKER)}
            for path, report in reports.items():
                with self.subTest(path=path):
                    self.assert_passes(report)
                    self.assertEqual(len(report["outputs"]), len(expected))
                    for output, got, (shape, want) in zip(report["outputs"], _outputs(report),
                                                          expected):
                        self.assertEqual(output["shape"], shape)
                        _same(self, got.reshape(-1).tolist(), want)
        graph = torch.fx.Graph()
        source = graph.placeholder("x")
        source.meta["val"] = x
        nested = graph.call_method("flip", (source, (0,), 1))
        nested.meta["val"] = x
        graph.output(nested)
        with self.assertRaisesRegex(NotImplementedError, "flip reads one int per dim"):
            torchlean._lower_pytorch_graph(torch.nn.Module(), graph, ["x"])

    def test_maximum_and_minimum_promote_once(self):
        # torch.maximum and torch.minimum convert an int32 operand to bfloat16 in one rounding, as
        # add and mul do: 16842753 becomes 16908288, where PyTorch rounds through binary32 and
        # gives 16777216.
        torch = self.torch
        a = torch.tensor([16842753, -16842753, 3, 65519, 65520], dtype=torch.int32)

        def extremes(a, low, high, low16, high16):
            return (torch.maximum(a, low), torch.minimum(a, high), torch.maximum(a, low16),
                    torch.minimum(a, high16))

        low, high = torch.full((5,), -1e38), torch.full((5,), 1e38)
        self.assertEqual(torch.maximum(a, low.to(torch.bfloat16))[0].item(), 16777216)
        report = self.verify(extremes, a, low.to(torch.bfloat16), high.to(torch.bfloat16),
                             torch.full((5,), -6e4).half(), torch.full((5,), 6e4).half())
        self.assert_passes(report)
        # -6e4 and 6e4 are binary16 values, and the int32 entries round once to each format.
        bf16 = [_nearest(v, "bfloat16") for v in a.tolist()]
        f16 = [_nearest(v, "float16") for v in a.tolist()]
        outputs = [output.tolist() for output in _outputs(report)]
        _same(self, outputs[0], bf16)
        _same(self, outputs[1], bf16)
        _same(self, outputs[2], [max(v, -6e4) for v in f16])
        _same(self, outputs[3], [min(v, 6e4) for v in f16])
        self.assertEqual(outputs[0][0], 16908288.0)

    def test_nan_results_agree_whatever_their_sign_and_payload(self):
        # On x86, PyTorch's 0 / 0 is the negative default NaN and FloatLib's a positive quiet NaN.
        # The IR leaves both unspecified. compare_bitwise compares the words.
        torch = self.torch
        zeros = torch.zeros(4)
        self.assertTrue(bool(torch.isnan(torch.div(zeros, zeros)).all()))
        self.assert_passes(self.verify(torch.div, zeros, zeros, checks=COMPARE))
        bitwise = self.verify(torch.div, zeros, zeros,
                              checks=("structure", "types", "compare_bitwise"))
        self.assertFalse(bitwise["ok"])
        self.assertEqual(_statuses(bitwise)["compare_bitwise"][0], "failed", _statuses(bitwise))

    def test_rounded_operations_execute(self):
        # Non-integral powers, the reciprocal square root and the tanh GELU round differently
        # from PyTorch's kernels in some last bits, so these captures are executed only.
        torch = self.torch
        functional = torch.nn.functional

        def rounded(x, y):
            return (torch.pow(x.abs(), y), torch.pow(x.abs(), 2.5), torch.rsqrt(x.abs()),
                    functional.gelu(x, approximate="tanh"))

        self.assert_passes(self.verify(rounded, self.x, self.y))
        reduced = self.x.to(torch.bfloat16)
        self.assert_passes(self.verify(rounded, reduced, self.y.to(torch.bfloat16)))

    def test_gelu_module_executes(self):
        torch = self.torch
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.GELU(approximate="tanh"))
        self.assert_passes(self.verify(model.eval(), self.x))

    def test_capture_preserves_non_dense_layout_and_argument_aliases(self):
        # A compact clone changes the branch and lets comparison validate the
        # wrong capture. The input is also a non-leaf view with a nonzero offset.
        torch = self.torch
        base = torch.arange(12., requires_grad=True)
        view = base.reshape(3, 4)[:, 1::2]

        def layout(a, b):
            same = a is b and a.stride() == (4, 2) and a.storage_offset() == 1
            return a + 1 if same else a + 2

        report = self.verify(layout, view, view, checks=COMPARE)
        self.assert_passes(report)
        expected = layout(view, view).detach().flatten().view(torch.int32).tolist()
        self.assertEqual(report["outputs"][0]["bits"], expected)
        torch.testing.assert_close(base, torch.arange(12.))

    def test_error_function_needs_an_implementation(self):
        torch = self.torch
        for function in (torch.erf, torch.nn.functional.gelu):
            with self.subTest(function=function.__name__):
                # The typed runtime has no float32 rule for erf or for gelu without an
                # approximation: the types check passes and execution refuses the step.
                report = self.verify(lambda x, f=function: f(x), self.x)
                self.assert_unsupported_execution(report, "no rule at float32")

    def test_convolutions_execute_with_their_payload(self):
        # A plain convolution, one with dilation and groups, and one with stride, groups and no
        # bias import and execute through torch.export, its core decomposition and FX. Integer
        # weights and inputs make every summation order exact, so `compare` holds.
        torch = self.torch
        nn = torch.nn
        generator = torch.Generator().manual_seed(0)
        cases = {"plain": (nn.Conv2d(2, 4, 3), (2, 2, 6, 6)),
                 "dilation and groups": (nn.Conv2d(4, 8, 3, groups=2, dilation=(1, 2),
                                                   padding=(1, 2)), (1, 4, 5, 7)),
                 "stride, groups, no bias": (nn.Conv1d(3, 6, 2, stride=2, groups=3, bias=False),
                                             (2, 3, 9))}
        original = torch.export.export
        with tempfile.TemporaryDirectory() as directory:
            for name, (model, shape) in cases.items():
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.copy_(torch.randint(-3, 4, parameter.shape, generator=generator))
                model.eval()
                x = torch.randint(-4, 5, shape, generator=generator).float()
                with self.subTest(case=name, capture="export"):
                    self.assert_passes(self.verify(model, x, checks=COMPARE))
                with self.subTest(case=name, capture="core"):
                    torch.export.export = lambda *a, **k: original(*a, **k).run_decompositions()
                    try:
                        self.assert_passes(self.verify(model, x, checks=COMPARE))
                    finally:
                        torch.export.export = original
                with self.subTest(case=name, capture="fx"):
                    report = self.compare_fx(directory, model, x)[1]
                    self.assert_passes(report)

    def test_in_place_activations_are_refused(self):
        # An in-place nn.ELU or nn.Hardtanh whose result is unused mutates its input. FX records
        # the call, and the capture refuses it, since dropping it would lose the mutation.
        torch = self.torch
        nn = torch.nn

        class InPlace(nn.Module):
            def __init__(self, activation):
                super().__init__()
                self.activation = activation

            def forward(self, x):
                self.activation(x)
                return x

        with tempfile.TemporaryDirectory() as directory:
            for activation in (nn.ELU(inplace=True), nn.Hardtanh(inplace=True)):
                with self.subTest(activation=type(activation).__name__):
                    with self.assertRaisesRegex(NotImplementedError, "in-place"):
                        self.execute_fx(directory, InPlace(activation), self.x)

    def test_normalizations_execute_with_their_payload(self):
        # nn.LayerNorm with and without its affine parameters and bias, and nn.BatchNorm1d, 2d and
        # 3d in eval mode, affine or not, with momentum 0.3 or None, import through torch.export,
        # its core decomposition and FX with their weights, biases, eps and running statistics, in
        # binary64, binary32, binary16 and bfloat16. On the rows [a, a, a, a + 4] and with running
        # variances plus eps of 1, 4 and 16 every operation of both computations is exact, so
        # every word agrees with PyTorch's although its kernels compute (x * rstd - mean * rstd) *
        # gamma + beta and x * alpha + (beta - mean * alpha). On random binary32 data, with the
        # default eps, the fixture accepts values within 1e-5 relative and 1e-6 absolute. A
        # negative running variance, where PyTorch takes the square root of var + eps and the IR
        # clamps the variance, is refused.
        torch = self.torch
        nn = torch.nn
        generator = torch.Generator().manual_seed(0)

        def layer_norm(**options):
            module = nn.LayerNorm(4, eps=1.0, **options)
            with torch.no_grad():
                if module.weight is not None:
                    module.weight.copy_(torch.tensor([2.0, 3.0, 4.0, 5.0]))
                if module.bias is not None:
                    module.bias.copy_(torch.tensor([1.0, -1.0, 0.5, 0.25]))
            return module

        def batch_norm(kind, **options):
            module = kind(3, eps=1.0, **options)
            with torch.no_grad():
                module.running_mean.copy_(torch.tensor([1.0, -2.0, 0.5]))
                module.running_var.copy_(torch.tensor([3.0, 15.0, 0.0]))
                if module.weight is not None:
                    module.weight.copy_(torch.tensor([2.0, -1.0, 4.0]))
                    module.bias.copy_(torch.tensor([0.5, 3.0, -1.0]))
            return module

        rows = torch.tensor([[1.0, 1.0, 1.0, 5.0], [-3.0, -3.0, -3.0, 1.0], [5.0, 5.0, 5.0, 5.0]])
        cases = {"layer norm": (layer_norm(), rows),
                 "layer norm without affine parameters": (layer_norm(elementwise_affine=False),
                                                          rows),
                 "layer norm without bias": (layer_norm(bias=False), rows),
                 "batch norm 1d": (batch_norm(nn.BatchNorm1d, momentum=0.3),
                                   torch.randint(-4, 5, (2, 3, 5), generator=generator).float()),
                 "batch norm 2d without affine parameters": (
                     batch_norm(nn.BatchNorm2d, affine=False, momentum=None),
                     torch.randint(-4, 5, (2, 3, 2, 2), generator=generator).float()),
                 "batch norm 3d": (batch_norm(nn.BatchNorm3d),
                                   torch.randint(-4, 5, (1, 3, 2, 1, 2),
                                                 generator=generator).float())}
        original = torch.export.export
        with tempfile.TemporaryDirectory() as directory:
            for name, (model, x) in cases.items():
                model.eval()
                for dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16):
                    typed, inputs = copy.deepcopy(model).to(dtype), x.to(dtype)
                    with self.subTest(case=name, dtype=dtype, capture="export"):
                        self.assert_passes(self.verify(typed, inputs, checks=COMPARE))
                    with self.subTest(case=name, dtype=dtype, capture="core"):
                        torch.export.export = lambda *a, **k: original(*a, **k).run_decompositions()
                        try:
                            self.assert_passes(self.verify(typed, inputs, checks=COMPARE))
                        finally:
                            torch.export.export = original
                    with self.subTest(case=name, dtype=dtype, capture="fx"):
                        # A container, so that FX records the module as one call.
                        module = nn.Sequential(typed)
                        self.assert_passes(self.compare_fx(directory, module, inputs)[1])
        functional = torch.nn.functional
        weight = torch.randn(8, generator=generator)
        bias = torch.randn(8, generator=generator)
        mean, var = torch.randn(3, generator=generator), torch.rand(3, generator=generator) + 0.5
        scale, shift = torch.randn(3, generator=generator), torch.randn(3, generator=generator)
        functions = {"layer_norm": (lambda x: functional.layer_norm(x, (8,), weight, bias),
                                    torch.randn(3, 8, generator=generator)),
                     "batch_norm": (lambda x: functional.batch_norm(x, mean, var, scale, shift),
                                    torch.randn(4, 3, 5, generator=generator))}
        for name, (function, x) in functions.items():
            with self.subTest(case=name, data="random"):
                report = self.verify(function, x, checks=EXECUTE)
                self.assert_passes(report)
                torch.testing.assert_close(_outputs(report)[0], function(x), rtol=1e-5, atol=1e-6)
        negative = nn.BatchNorm1d(3).eval()
        with torch.no_grad():
            negative.running_var.copy_(torch.tensor([1.0, -0.5, 1.0]))
        # Without running statistics the batch norm normalizes by the batch's moments, within a
        # tolerance.
        batch = nn.BatchNorm1d(3, track_running_stats=False).eval()
        inputs = torch.randn(4, 3, generator=generator)
        report = self.verify(batch, inputs, checks=EXECUTE)
        self.assert_passes(report)
        with torch.no_grad():
            torch.testing.assert_close(_outputs(report)[0], batch(inputs), rtol=1e-5, atol=1e-5)
        refused = {"negative running variance": negative}
        for name, model in refused.items():
            with self.subTest(case=name):
                report = self.verify(model, torch.randn(2, 3, generator=generator),
                                     checks=("structure",))
                self.assertFalse(report["ok"])

    def test_argument_aliases_bind_once_on_every_path(self):
        # PyTorch's parser reads NumPy's `axis` and `keepdims` as `dim` and `keepdim`, so every
        # spelling of a reduction keeps its axis through torch.export, FX and Dynamo alike.
        from torchlean._fx import _tensor_value

        torch = self.torch

        class Call(torch.nn.Module):
            def __init__(self, function):
                super().__init__()
                self.function = function

            def forward(self, x):
                return self.function(x)

        x = torch.arange(6.).reshape(2, 3) - 2
        spellings = {
            "torch.sum axis keepdims": lambda v: torch.sum(v, axis=1, keepdims=True),
            "torch.sum dim keepdim": lambda v: torch.sum(v, dim=1, keepdim=True),
            "torch.sum axis -1": lambda v: torch.sum(v, axis=-1, keepdims=True),
            "Tensor.sum axis keepdims": lambda v: v.sum(axis=1, keepdims=True),
            "torch.mean axis keepdims": lambda v: torch.mean(v, axis=1, keepdims=True),
            "torch.mean dim -1 keepdim": lambda v: torch.mean(v, dim=-1, keepdim=True),
            "Tensor.mean dim -1 keepdims": lambda v: v.mean(dim=-1, keepdims=True),
            # The fuzzer's seed 5: FX lowered `axis` before binding it (TypeError).
            "torch.cumsum axis": lambda v: torch.cumsum(v, axis=1),
            "Tensor.cumsum axis -2": lambda v: v.cumsum(axis=-2),
        }
        for name, function in spellings.items():
            expected = function(x)
            self.assertEqual(tuple(expected.shape), (2, 3) if "cumsum" in name else (2, 1))
            compiled = torchlean.compile(checker=CHECKER)(function)
            torch.testing.assert_close(compiled(x), expected)
            dynamo = compiled.torchlean.graphs[-1]["artifact"]
            dynamo["arguments"] = [_tensor_value(x)]
            dynamo["reference_outputs"] = [_tensor_value(expected)]
            with tempfile.TemporaryDirectory() as directory:
                reports = {"export": self.verify(function, x, checks=COMPARE),
                           "fx": self.compare_fx(directory, Call(function), x)[1],
                           "dynamo": torchlean.verify(dynamo, checks=COMPARE, checker=CHECKER)}
            for path, report in reports.items():
                with self.subTest(case=name, path=path):
                    self.assert_passes(report)
                    self.assertEqual(report["outputs"][0]["shape"], list(expected.shape))

    def test_records_without_an_integer_or_boolean_form_are_refused(self):
        # TLFIX-8x: a record Lean's decoder has only at a binary format (a reduction, abs, relu, a
        # product) is refused at Booleans and integer types with that reason, not exported to fail
        # at the import.
        torch = self.torch
        m = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
        cases = {
            "sum": (lambda v: torch.sum(v, axis=1, keepdims=True), "reduce runs at"),
            "prod": (lambda v: torch.prod(v, dim=1), "runs at a binary format only"),
            "Boolean sum": (lambda v: (v > 2).sum(dim=1), "runs at a binary format only"),
            "abs": (torch.abs, "abs runs at a binary format only"),
            "relu": (torch.relu, "runs at a binary format only"),
            "matmul": (lambda v: v @ v.transpose(0, 1), "runs at a binary format only"),
        }
        for name, (function, text) in cases.items():
            with self.subTest(case=name):
                self.assert_refused(self.verify(function, m), text)

    def test_floating_trunc_division_truncates_the_rounded_quotient(self):
        # Fable hunt H1: PyTorch truncates the rounded quotient (`10` at `(1, 0.1)`), where the
        # IR's `trunc` mode truncates the exact one (`9`). Binary32, and bfloat16 with a scalar
        # divisor, which ATen divides in binary32.
        torch = self.torch
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=str(dtype)):
                v = torch.tensor([1.0, 0.3, 7.0, -1.0], dtype=dtype)
                for function in (lambda v: torch.div(v, 0.1, rounding_mode="trunc"),
                                 lambda v: v.div(torch.full_like(v, 0.1), rounding_mode="trunc")):
                    report = self.verify(function, v, checks=COMPARE)
                    self.assert_passes(report)
                    self.assertEqual(_outputs(report)[0].tolist(), function(v).tolist())

    def test_one_parameter_given_twice_is_refused(self):
        # A call that gives one parameter under two spellings, or a keyword its target does not
        # take, binds to no overload, and the binder names the call.
        import torch.fx
        from torchlean._bind import bind_arguments, canonical

        torch = self.torch
        self.assertEqual(canonical("call_function", torch.sum, ("x",),
                                   {"axis": 1, "keepdims": True}), {"dim": 1, "keepdim": True})
        self.assertEqual(canonical("call_method", "mean", ("x",), {"axis": 1}), {"dim": 1})
        cases = {
            "dim and axis": ("call_function", torch.sum, {"dim": 1, "axis": 1},
                             "torch.sum: the parameter 'dim' is given as 'dim' and 'axis'"),
            "keepdim and keepdims": ("call_method", "sum", {"dim": 1, "keepdim": True,
                                                            "keepdims": True},
                                     "Tensor.sum: the parameter 'keepdim' is given as 'keepdim' "
                                     "and 'keepdims'"),
            "axis for dims": ("call_function", torch.flip, {"axis": 0}, "no overload binds them"),
        }
        for name, (op, target, kwargs, message) in cases.items():
            with self.subTest(case=name):
                graph = torch.fx.Graph()
                x = graph.placeholder("x")
                call = graph.call_function if op == "call_function" else graph.call_method
                graph.output(call(target, (x,), kwargs))
                with self.assertRaises(NotImplementedError) as refused:
                    bind_arguments(graph)
                self.assertIn(message, str(refused.exception))

    def test_declared_types_must_match_the_program(self):
        # The capture declares each lowered value's element type and shape from PyTorch's own
        # metadata, and the checker refuses a program that gives a value another type. Without
        # the binder the importer reads `torch.sum(v, axis=1, keepdims=True)` as a sum of every
        # entry, and this check refuses it.
        from unittest import mock

        from torchlean._fx import _tensor_value
        from torchlean.export import export_torchlean_graph_json

        torch = self.torch

        class Reduced(torch.nn.Module):
            def forward(self, v):
                return torch.sum(v, axis=1, keepdims=True)

        x = torch.arange(6.).reshape(2, 3) - 2
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "graph.json")
            document = export_torchlean_graph_json(Reduced(), (x,), path,
                                                   prefer_torch_export=False)
            with mock.patch("torchlean._graph.bind_arguments", lambda graph: None):
                unbound = export_torchlean_graph_json(Reduced(), (x,), path,
                                                      prefer_torch_export=False)
        for value in (document, unbound):
            value["arguments"] = [_tensor_value(x)]
        self.assert_passes(torchlean.verify(document, checks=EXECUTE, checker=CHECKER))
        float64 = copy.deepcopy(document)
        (entry,) = [entry for entry in float64["declared_types"] if entry["shape"] == [2, 1]]
        entry["dtype"] = "float64"
        undeclared = copy.deepcopy(document)
        del undeclared["declared_types"]
        cases = {
            "aliases left unbound": (unbound, "the IR computes float32 [], the capture declares "
                                              "float32 [2, 1]"),
            "another element type": (float64, "the IR computes float32 [2, 1], the capture "
                                              "declares float64 [2, 1]"),
            "no declared types": (undeclared, "IR types: the capture declares no value types "
                                              "(declared_types)"),
        }
        for name, (value, message) in cases.items():
            with self.subTest(case=name):
                report = torchlean.verify(value, checks=EXECUTE, checker=CHECKER)
                statuses = _statuses(report)
                self.assertFalse(report["ok"])
                self.assertEqual(statuses["types"][0], "failed", statuses)
                self.assertTrue(statuses["types"][1].startswith("IR types: "), statuses)
                self.assertIn(message, statuses["types"][1])
                self.assertEqual(statuses["execute"], statuses["types"])

    def test_report_gives_the_complete_conversion_status(self):
        # int16 256 and -1 wrap to uint8 0 and 255, and the report gives all seven flags of the
        # run's conversion status.
        torch = self.torch
        report = self.verify(lambda n: n.to(torch.uint8),
                             torch.tensor([256, -1, 7], dtype=torch.int16), checks=COMPARE)
        self.assert_passes(report)
        self.assertEqual(report["outputs"][0]["bits"], [0, 255, 7])
        self.assertEqual(report["conversion_flags"], {
            "inexact": False, "overflow": False, "underflow": False, "saturated": False,
            "wrapped": True, "mapped_special": False, "invalid": False})

    def test_index_results_are_reported(self):
        # A raw IR program may return index tuples, here argmax's: the report gives them with
        # their port, and comparing them with a reference tensor fails with a reason.
        from torchlean._fx import _tensor_value

        torch = self.torch
        x = torch.tensor([[0., 3, 1], [5, 2, 4]])
        captured = torchlean._capture(lambda v: torch.argmax(v, 1), (x,), "pytorch", EXECUTE)
        artifact = captured["artifact"]
        (k,) = [k for k, step in enumerate(artifact["nodes"])
                if "op" in step and step["op"]["record"]["tag"] == "argmax"]
        for key in ("nodes", "scalars", "constants"):
            artifact[key] = artifact[key][:k + 1]
        artifact["outputs"] = [[k, 0]]
        document = {"artifact": artifact, "arguments": captured["arguments"]}
        report = torchlean.verify(document, checks=EXECUTE, checker=CHECKER)
        self.assert_passes(report)
        self.assertEqual(report["outputs"], [{
            "element_type": "index", "dtype": None,
            "port": {"index": {"bounds": [3], "shape": [2]}}, "shape": [2], "bits": [1, 0]}])
        self.assertTrue(torch.equal(_outputs(report)[0], torch.argmax(x, 1)))
        document["reference_outputs"] = [_tensor_value(torch.argmax(x, 1))]
        compare = _statuses(torchlean.verify(document, checks=COMPARE, checker=CHECKER))["compare"]
        self.assertEqual(compare[0], "failed")
        self.assertIn("result 0 has index tuples below [3], and a reference tensor holds no "
                      "index tuples", compare[1])

    def test_a_document_without_utf8_json_text_fails_every_check(self):
        # TLFIX-H14: a lone surrogate, a NaN or a value JSON lacks fails every request and
        # writes no export, where `verify` used to raise.
        document = torchlean._capture(lambda v, w: v + w, (self.x, self.y), "pytorch", EXECUTE)
        self.assert_passes(torchlean.verify(document, checks=EXECUTE, checker=CHECKER))
        with tempfile.TemporaryDirectory() as directory:
            for name, value in (("surrogate", "\ud800"), ("nan", float("nan")),
                                ("object", object())):
                with self.subTest(name=name):
                    export = Path(directory) / f"{name}.json"
                    report = torchlean.verify(dict(document, zzz=value), checks=EXECUTE,
                                              checker=CHECKER, export=export)
                    self.assertFalse(report["ok"])
                    self.assertEqual(report["checker"], "not_run")
                    self.assertEqual([c["status"] for c in report["checks"]], ["failed"] * 3)
                    self.assertIn("no UTF-8 JSON text", report["checks"][0]["message"])
                    self.assertFalse(export.exists())


@unittest.skipUnless(CHECKER and HAS_JAX, "needs the built verify executable and JAX")
class JaxCaptureTests(CheckerCase):
    def setUp(self):
        import jax
        import jax.numpy as jnp
        import numpy as np

        self.jax, self.jnp, self.np = jax, jnp, np
        keys = jax.random.split(jax.random.PRNGKey(0), 2)
        self.x = jax.random.normal(keys[0], (3, 4))
        self.y = jax.random.normal(keys[1], (3, 4))

    def verify(self, function, *inputs, checks=EXECUTE):
        return torchlean.verify(function, *inputs, frontend="jax", checks=checks,
                                checker=CHECKER)

    def test_records_without_an_integer_or_boolean_form_are_refused(self):
        # TLFIX-8x: integer abs and product records, and Boolean arithmetic, which Lean's decoder
        # does not have, are refused with the reason.
        jnp, np = self.jnp, self.np
        m = np.array([[1, 2], [3, 4]], dtype=np.int32)
        b = np.array([True, False])
        cases = {
            "abs": (jnp.abs, m, "abs runs at a binary format only"),
            "product": (lambda v: v @ v, m, "runs at a binary format only"),
            "Boolean minimum": (lambda v: jnp.minimum(v, v), b, "min has no Boolean form"),
        }
        for name, (function, value, text) in cases.items():
            with self.subTest(case=name):
                self.assert_refused(self.verify(function, value), text)

    def test_an_uncopyable_model_is_refused(self):
        # Fable hunt H11: the capture copies the model, and a model holding a lock is refused.
        import threading

        class Locked:
            def __init__(self):
                self.lock = threading.Lock()

            def __call__(self, v):
                return v * 2

        self.assert_refused(self.verify(Locked(), self.x), "the JAX capture copies the model")

    def advice(self, report):
        # Every mismatch fails, and the behaviours whose emulation reproduces it are advice.
        entry = report["checks"][-1]
        self.assertEqual(entry["status"], "failed", entry["message"])
        self.assertFalse(report["ok"])
        return sorted(entry.get("consistent_with", []))

    def emulated(self, report, result=0):
        (found,) = [item for item in report["checks"][-1]["results"] if item["result"] == result]
        window = found["window"]
        return sorted([item["kind"] for item in found["emulated"]] +
                      ([window["kind"]] if window and window["elements"] else []))

    def artifact(self, function, *inputs, reference=None):
        # The capture document torchlean.verify writes for a JAX function, before the checker
        # runs. The reference may come from another function, as an importer bug would make it.
        from torchlean.jaxpr import encode, export, references

        document = export(function, *inputs)
        document["arguments"] = [encode(value) for value in inputs]
        document.update(references(reference or function, *inputs))
        return document

    @staticmethod
    def tagged(document, tag):
        # The positions of the artifact's op steps whose record has the tag `tag`, in step order.
        return [k for k, step in enumerate(_steps(document))
                if "op" in step and step["op"]["record"]["tag"] == tag]

    def swapped(self, document, tag, which=0):
        # An importer bug: the operands of one op step with the tag `tag` in the wrong order.
        document = copy.deepcopy(document)
        step = _steps(document)[self.tagged(document, tag)[which]]
        step["op"]["inputs"] = step["op"]["inputs"][::-1]
        return document

    def with_constant(self, document, old, new):
        # An importer bug: the first binary constant step holding only `old` written as `new`.
        from torchlean.jaxpr import encode

        document = copy.deepcopy(document)
        artifact = document["artifact"]
        for step, scalar, words in zip(artifact["nodes"], artifact["scalars"],
                                       artifact["constants"]):
            if "const" in step and scalar is not None and words:
                dtype = self.jnp.dtype(_scalar_name(scalar))
                values = self.np.array(words, dtype=f"uint{8 * dtype.itemsize}").view(dtype)
                if (values == old).all():
                    replaced = self.np.full(len(words), new, dtype)
                    words[:] = encode(self.jnp.asarray(replaced))["bits"]
                    return document
        raise AssertionError(f"no constant {old}")

    def assert_fails(self, document):
        # A failure with no documented XLA behaviour listed.
        report = torchlean.verify(document, checks=COMPARE, checker=CHECKER)
        entry = report["checks"][-1]
        self.assertFalse(report["ok"])
        self.assertEqual(entry["status"], "failed", entry["message"])
        self.assertNotIn("consistent_with", entry, entry["message"])
        return entry

    def test_exact_operations_match_xla_words(self):
        # XLA multiplies by 0.5 for x / 2.0, which rounds as the quotient does.
        def exact(x, y):
            return x / y, -x, x ** 3, x ** -2, x / 2.0

        self.assert_passes(self.verify(exact, self.x, self.y, checks=COMPARE))

    def test_numpy_inputs_are_captured_as_jax_reads_them(self):
        # With 64-bit mode off JAX reads NumPy float64, int64 and uint64 values as float32, int32
        # and uint32 (2 ** 33 + 7 is 7), and with it on as they are. The capture traces,
        # serializes and compares those same values, and a NumPy output leaf too, in
        # torchlean.verify and in a document from export, encode and references. A container
        # argument of references keeps its structure, with each leaf read so.
        from torchlean.jaxpr import references

        jax, np = self.jax, self.np
        inputs = (np.array([1 + 2 ** -30, -3.5]), np.array([1, -2], dtype=np.int64),
                  np.array([1, 2 ** 33 + 7], dtype=np.uint64), np.float64(0.1))

        def function(x, n, u, s):
            return x * s, n + 1, u * 2, np.arange(2.0)

        for x64, width in ((False, 32), (True, 64)):
            with self.subTest(x64=x64), jax.enable_x64(x64):
                document = torchlean._capture(function, inputs, "jax", COMPARE)
                floating = f"float{width}"
                self.assertEqual([argument["dtype"] for argument in document["arguments"]], [
                    floating, {"kind": "integer", "signed": True, "width": width},
                    {"kind": "integer", "signed": False, "width": width}, floating])
                self.assertEqual(document["arguments"][2]["bits"], [1, 2 ** 33 + 7 if x64 else 7])
                self.assertEqual(document["reference_outputs"][3]["dtype"], floating)
                self.assert_passes(self.verify(function, *inputs, checks=COMPARE))
                exported = self.artifact(function, *inputs)
                self.assertEqual(exported["arguments"], document["arguments"])
                self.assert_passes(torchlean.verify(exported, checks=COMPARE, checker=CHECKER))
        with jax.enable_x64(False):
            record = references(lambda p: p["w"] * p["v"][1],
                                {"w": np.float64(3.0), "v": [np.zeros(2), np.ones(3)]})
        self.assertEqual(record["reference_outputs"],
                         [dict(dtype="float32", shape=[3], bits=[0x40400000] * 3)])

    def test_dot_algorithms_are_not_erased(self):
        # All operands and the result may be FP32 even when the selected algorithm multiplies
        # in BF16. Shape/dtype checks cannot recover that lost arithmetic choice.
        jax = self.jax
        x = self.jnp.array([[1.001, 1.003]], dtype=self.jnp.float32)
        y = self.jnp.array([[1.005], [1.007]], dtype=self.jnp.float32)

        def dot(precision):
            return lambda a, b: jax.lax.dot_general(
                a, b, (((1,), (0,)), ((), ())), precision=precision)

        for precision in (None, jax.lax.Precision.DEFAULT, jax.lax.DotAlgorithmPreset.DEFAULT):
            with self.subTest(precision=precision):
                self.assert_passes(self.verify(dot(precision), x, y))
        for precision in (jax.lax.Precision.HIGH, jax.lax.Precision.HIGHEST,
                          jax.lax.DotAlgorithmPreset.BF16_BF16_F32,
                          jax.lax.DotAlgorithmPreset.TF32_TF32_F32):
            with self.subTest(precision=precision):
                report = self.verify(dot(precision), x, y)
                self.assertFalse(report["ok"])
                self.assertEqual(report["checker"], "not_run")
                for check in report["checks"]:
                    self.assertEqual(check["status"], "unsupported")
                    self.assertIn("operand/accumulator arithmetic rule", check["message"])

    def test_division_by_a_constant_keeps_the_ieee_quotient(self):
        # The importer gives lax.div its IEEE meaning. XLA's CPU code multiplies by the binary32
        # value of 1/3, so its x / 3.0 differs from the quotient: compare fails and lists the
        # reciprocal as the documented behaviour the mismatch is consistent with.
        jnp, np = self.jnp, self.np
        values = (np.random.default_rng(1).standard_normal(4096) * 3).astype(np.float32)
        x = jnp.asarray(values)
        report = self.verify(lambda v: v / 3.0, x, checks=COMPARE)
        words = np.array(report["outputs"][0]["bits"], dtype=np.uint32)
        self.assertTrue((words == (values / np.float32(3)).view(np.uint32)).all())
        xla = np.asarray(x / 3.0)
        self.assertTrue((xla == values * (np.float32(1) / np.float32(3))).all())
        self.assertGreater(int((xla.view(np.uint32) != words).sum()), 0)
        self.assertEqual(self.advice(report), ["divisor_reciprocal"])
        # Eager JAX divides by a closed-over array of threes as written, so the IR agrees.
        threes = jnp.full(4096, 3.0, dtype=jnp.float32)
        self.assert_passes(self.verify(lambda v: v / threes, x, checks=COMPARE))
        # jax.jit folds the array and multiplies by its reciprocal. Primitive by primitive, XLA
        # divides as the IR does.
        report = self.verify(self.jax.jit(lambda v: v / threes), x, checks=COMPARE)
        self.assertEqual(self.advice(report), ["divisor_reciprocal"])
        self.assertEqual(report["checks"][-1]["eager_reference"]["status"], "agrees")

    def test_mean_multiplies_by_the_reciprocal_of_the_count(self):
        # jnp.mean divides its row sums by the broadcast count 768, which XLA multiplies by
        # binary32 1/768. Small integers keep every sum exact, whatever the summation order.
        jnp, np = self.jnp, self.np
        rows = jnp.asarray(np.random.default_rng(2).integers(-8, 9, (64, 768)).astype(np.float32))
        mean = np.asarray(jnp.mean(rows, axis=-1))
        sums = np.asarray(rows).sum(axis=-1)
        self.assertTrue((mean == sums * (np.float32(1) / np.float32(768))).all())
        self.assertFalse((mean == sums / np.float32(768)).all())
        report = self.verify(lambda v: jnp.mean(v, axis=-1), rows, checks=COMPARE)
        self.assertEqual(self.advice(report), ["divisor_reciprocal"])

    def test_reference_execution_is_recorded(self):
        # The importer records whether calling the function runs one jax.jit computation, and
        # in an eager function the computation of each artifact step: x / 3.0 broadcasts its
        # divisor inside the division, jnp.broadcast_to in a computation of its own. Each step
        # also records its primitive, inside a nested jax.jit call too. The reference records its
        # platform and jaxlib's version, and, when jax.jit compiles part of the function, the
        # outputs of evaluating it one primitive at a time.
        import jaxlib

        jax, jnp = self.jax, self.jnp
        three = jnp.float32(3)
        eager = self.artifact(lambda v: v / 3.0, self.x)
        self.assertEqual(eager["reference_execution"], "eager")
        self.assertEqual(eager["reference_platform"], "cpu")
        self.assertEqual(eager["jaxlib_version"], jaxlib.__version__)
        self.assertNotIn("eager_reference_outputs", eager)
        compiled = self.artifact(jax.jit(lambda v: v / 3.0), self.x)
        self.assertEqual(compiled["reference_execution"], "jit")
        self.assertEqual(len(compiled["eager_reference_outputs"]), 1)
        apart = self.artifact(lambda v: v / jnp.broadcast_to(three, v.shape), self.x)
        (division,), (broadcast,) = self.tagged(apart, "div"), self.tagged(apart, "broadcast")
        self.assertNotEqual(apart["computation"][division], apart["computation"][broadcast])
        for key in ("computation", "primitive_index"):
            self.assertEqual(len(eager[key]), len(_steps(eager)))
        (division,), (broadcast,) = self.tagged(eager, "div"), self.tagged(eager, "broadcast")
        self.assertEqual(eager["computation"][division], eager["computation"][broadcast])
        self.assertEqual(eager["primitive_index"][division], eager["primitive_index"][broadcast])
        nested = self.artifact(jax.jit(lambda u, v, w: (u / v) / w), self.x, self.y, self.x)
        first, second = self.tagged(nested, "div")
        self.assertEqual(nested["computation"][first], nested["computation"][second])
        self.assertNotEqual(nested["primitive_index"][first], nested["primitive_index"][second])

    def test_undocumented_mismatches_fail(self):
        # Importer bugs next to operations where XLA differs, and FloatLib's transcendentals,
        # fail with no documented behaviour listed: the emulation of XLA's CPU code does not
        # reproduce the reference.
        jax, jnp, np = self.jax, self.jnp, self.np
        rng = np.random.default_rng(1)
        x, y = (jnp.asarray(rng.standard_normal(4096).astype(np.float32)) for _ in range(2))
        z = jnp.asarray((np.abs(rng.standard_normal(4096)) * 3 + 0.5).astype(np.float32))
        cases = {
            "(x - y) / 3.0, sub swapped":
                self.swapped(self.artifact(lambda a, b: (a - b) / 3.0, x, y), "sub"),
            "x - y - mean(x), sub swapped":
                self.swapped(self.artifact(lambda a, b: a - b - jnp.mean(a), x, y), "sub"),
            "eager (x / z) / z, inner division swapped":
                self.swapped(self.artifact(lambda a, b: (a / b) / b, x, z), "div"),
            "jax.jit (x / z) / z, inner division swapped":
                self.swapped(self.artifact(jax.jit(lambda a, b: (a / b) / b), x, z), "div"),
        }
        tiny = np.asarray(x).copy()
        tiny[0] = np.float32(1e-40)
        cases["x - y with one subnormal operand, sub swapped"] = self.swapped(
            self.artifact(lambda a, b: a - b, jnp.asarray(tiny), y), "sub")
        for name, function in (("tanh", jnp.tanh), ("exp", jnp.exp),
                               ("sigmoid", jax.nn.sigmoid)):
            cases[f"{name}(2x) / 3.0"] = self.artifact(lambda a, f=function: f(a) / 3.0, x * 2)
        threes = jnp.full(4096, 3.0, dtype=jnp.float32)
        cases["eager tanh(x) / a closed-over array"] = self.artifact(
            lambda a: jnp.tanh(a) / threes, x)
        negation = self.artifact(lambda a: -a, jnp.asarray(np.array([1e-40, 1.0], np.float32)))
        negation["reference_outputs"][0]["bits"][0] = 0x80000000
        cases["negation of a subnormal against a flushed reference"] = negation
        for name, document in cases.items():
            with self.subTest(case=name):
                self.assert_fails(document)

    def test_sqrt_record_names_its_policy(self):
        # lax.sqrt is the registered sqrt with the policy code of `.raw`, IEEE squareRoot. The
        # same document naming the clamped root (`.clampNegative`), as an importer bug would
        # write it, fails at -1 and -0, where the clamped root is +0.
        from torchlean._table import op_table

        jax, jnp = self.jax, self.jnp
        policies = op_table()["codes"]["sqrt"]
        x = jnp.asarray([-1.0, -0.0, 4.0, 2.0], dtype=jnp.float32)
        document = self.artifact(jax.lax.sqrt, x)
        (k,) = self.tagged(document, "sqrt")
        record = _steps(document)[k]["op"]["record"]
        self.assertEqual(record["fields"], [{"shape": [4]}, {"nat": policies["raw"]}])
        self.assert_passes(torchlean.verify(document, checks=COMPARE, checker=CHECKER))
        record["fields"][1] = {"nat": policies["clampNegative"]}
        self.assert_fails(document)

    def test_xla_functions_are_reproduced_within_their_windows(self):
        # XLA's rsqrt and power are its own approximations: a result they compute counts as
        # reproduced within 2 and 1 binary32 ulps, the largest distances measured between them
        # and the IR's. A power qualifies only when both operands are arguments, so a constant
        # exponent gets no window.
        lax, jnp, np = self.jax.lax, self.jnp, self.np
        rng = np.random.default_rng(1)
        x = jnp.asarray((rng.standard_normal(4096) * 3).astype(np.float32))
        positive = jnp.asarray((np.abs(rng.standard_normal(4096)) * 3 + 0.5).astype(np.float32))
        threes = jnp.full(4096, 3.0, dtype=jnp.float32)
        self.assertEqual(self.advice(self.verify(lax.rsqrt, positive, checks=COMPARE)), ["rsqrt"])
        self.assertEqual(self.advice(self.verify(lax.pow, x, threes, checks=COMPARE)), ["pow"])
        self.assert_fails(self.artifact(lambda v: v ** 3.0, x))
        self.assert_fails(self.artifact(lambda v: jnp.abs(v) ** 2.5, x))
        self.assert_fails(self.artifact(lambda v: v * lax.rsqrt(positive), x))

    def test_windows_are_measured_in_the_computing_format(self):
        # XLA computes binary16 and bfloat16 rsqrt in binary32 and rounds once, as the IR does, so
        # the window applies to the binary32 value. A binary64 rsqrt converted to binary32 has a
        # window of binary64 ulps. An importer that loses a factor or an epsilon one ulp of the
        # reduced format wide is not consistent with XLA's rsqrt.
        jax, lax, jnp, np = self.jax, self.jax.lax, self.jnp, self.np
        positive = np.random.default_rng(3).uniform(1.0, 4.0, 4096).astype(np.float32)
        bfloat16 = jnp.asarray(positive).astype(jnp.bfloat16)
        float16 = jnp.asarray(positive).astype(jnp.float16)
        wider = jnp.asarray(1 + 2 ** -7, jnp.bfloat16)
        epsilon = jnp.asarray(0.01, jnp.bfloat16)
        half = jnp.asarray(1 + 2 ** -10, jnp.float16)
        cases = {
            "bfloat16 rsqrt(x * (1 + 2^-7)) written as rsqrt(x)":
                self.artifact(lax.rsqrt, bfloat16, reference=lambda v: lax.rsqrt(v * wider)),
            "bfloat16 rsqrt(x + 0.01) written with epsilon 0":
                self.artifact(lax.rsqrt, bfloat16, reference=lambda v: lax.rsqrt(v + epsilon)),
            "binary16 rsqrt(x * (1 + 2^-10)) written as rsqrt(x)":
                self.artifact(lax.rsqrt, float16, reference=lambda v: lax.rsqrt(v * half)),
        }
        with jax.enable_x64(True):
            wide = jnp.asarray(positive.astype(np.float64))
            correct = self.artifact(lambda v: lax.rsqrt(v * 1.0000002).astype(jnp.float32), wide)
            cases["binary64 rsqrt(x * 1.0000002) to binary32, factor dropped"] = (
                self.with_constant(correct, 1.0000002, 1.0))
            for name, document in cases.items():
                with self.subTest(case=name):
                    self.assert_fails(document)
            self.assert_passes(torchlean.verify(correct, checks=COMPARE, checker=CHECKER))
        factor = jnp.float32(1.0000003)
        self.assertEqual(self.advice(self.verify(
            lambda v: lax.rsqrt(v * factor), jnp.asarray(positive), checks=COMPARE)), ["rsqrt"])

    def test_power_windows_need_argument_operands(self):
        # A changed exponent, or a factor dropped before a small exponent, moves the power by
        # about one ulp, inside a window. With a constant operand the window does not apply.
        jnp, lax, np = self.jnp, self.jax.lax, self.np
        rng = np.random.default_rng(3)
        positive = jnp.asarray(rng.uniform(1.0, 4.0, 4096).astype(np.float32))
        x = jnp.asarray(rng.standard_normal(4096).astype(np.float32))
        factor = jnp.float32(1 + 100 * 2 ** -23)
        cases = {
            "exponent 0.5 written as 0.50000006": self.with_constant(
                self.artifact(lambda v: lax.pow(v, jnp.float32(0.5)), positive), 0.5, 0.50000006),
            "(x * (1 + 100 eps)) ** 0.01 written as x ** 0.01": self.artifact(
                lambda v: v ** 0.01, positive, reference=lambda v: (v * factor) ** 0.01),
            "integer_pow(x, 3) written as pow(x, 3.0)": self.artifact(
                lambda v: v ** 3.0, x, reference=lambda v: lax.integer_pow(v, 3)),
        }
        for name, document in cases.items():
            with self.subTest(case=name):
                self.assert_fails(document)

    def test_jit_advice_needs_the_primitive_by_primitive_reference(self):
        # Some importer bugs coincide with jax.jit's rewrites on every element. Evaluated one
        # primitive at a time, XLA computes each operation as written, so the IR must agree with
        # those outputs, or differ only as XLA's documented behaviour reproduces, before the
        # jax.jit behaviour is listed.
        jax, lax, jnp, np = self.jax, self.jax.lax, self.jnp, self.np
        rng = np.random.default_rng(3)
        x = jnp.asarray(rng.standard_normal(4096).astype(np.float32))
        y, z = (jnp.asarray((np.abs(rng.standard_normal(4096)) * 3 + 0.5).astype(np.float32))
                for _ in range(2))
        quotients = self.artifact(jax.jit(lambda a, b, c: (a / b) / c), x, y, z)
        swapped = copy.deepcopy(quotients)
        inner, outer = (_steps(swapped)[k]["op"]["inputs"] for k in self.tagged(swapped, "div"))
        inner[1], outer[1] = outer[1], inner[1]
        nested = copy.deepcopy(self.artifact(jax.jit(lambda a, b, c: a / (b / c)), x, y, z))
        first, second = (_steps(nested)[k]["op"]["inputs"] for k in self.tagged(nested, "div"))
        # The input steps come first, so the dividend of x / (y / z) is step 0.
        inner, outer = (first, second) if first[0] != [0, 0] else (second, first)
        outer[0], inner[1] = inner[1], outer[0]
        cases = {
            "(x / y) / z with y and z swapped": swapped,
            "x / (y / z) with x and z swapped": nested,
            "x / (y * z) written as (x / y) / z": self.artifact(
                jax.jit(lambda a, b, c: (a / b) / c), x, y, z,
                reference=jax.jit(lambda a, b, c: a / (b * c))),
            "x * sqrt(y) written as x / rsqrt(y)": self.artifact(
                jax.jit(lambda a, b: a / lax.rsqrt(b)), x, y,
                reference=jax.jit(lambda a, b: a * jnp.sqrt(b))),
            "integer_pow(x, 3) written as pow(x, 3.0)": self.artifact(
                jax.jit(lambda a: a ** 3.0), x, reference=jax.jit(lambda a: lax.integer_pow(a, 3))),
        }
        for name, document in cases.items():
            with self.subTest(case=name):
                entry = self.assert_fails(document)
                self.assertEqual(entry["eager_reference"]["status"], "not_reproduced")
        report = torchlean.verify(quotients, checks=COMPARE, checker=CHECKER)
        self.assertEqual(self.advice(report), ["rewritten_division"])
        self.assertEqual(report["checks"][-1]["eager_reference"]["status"], "agrees")

    def test_emulation_needs_a_cpu_reference(self):
        # XLA's CPU behaviour is emulated only for a reference that XLA's CPU backend computed.
        x = self.jnp.asarray((self.np.random.default_rng(1).standard_normal(4096) * 3)
                             .astype(self.np.float32))
        document = self.artifact(lambda v: v / 3.0, x)
        self.assertEqual(self.advice(torchlean.verify(document, checks=COMPARE, checker=CHECKER)),
                         ["divisor_reciprocal"])
        for platform in ("gpu", None):
            with self.subTest(platform=platform):
                changed = copy.deepcopy(document)
                if platform is None:
                    del changed["reference_platform"]
                else:
                    changed["reference_platform"] = platform
                self.assert_fails(changed)

    def test_rewritten_operations_are_consistent_with_jit(self):
        # Compiled with jax.jit, XLA computes (x / y) / z as x / (y * z) and x ** 3.0 as
        # x * (x * x). The IR keeps the two quotients and the correctly rounded cube.
        # Primitive by primitive, XLA divides as the IR does.
        jnp, np = self.jnp, self.np
        rng = np.random.default_rng(1)
        x = jnp.asarray((rng.standard_normal(4096) * 3).astype(np.float32))
        y, z = (jnp.asarray((np.abs(rng.standard_normal(4096)) * 3 + 0.5).astype(np.float32))
                for _ in range(2))
        compiled = self.jax.jit(lambda a, b, c: ((a / b) / c, a / (b / c)))
        report = self.verify(compiled, x, y, z, checks=COMPARE)
        self.assertEqual(self.advice(report), ["rewritten_division"])
        self.assertEqual(self.emulated(report, 0), ["rewritten_division"])
        self.assertEqual(self.emulated(report, 1), ["rewritten_division"])
        self.assertEqual(report["checks"][-1]["eager_reference"]["status"], "agrees")
        # Eager JAX computes each division as written.
        self.assert_passes(self.verify(lambda a, b, c: (a / b) / c, x, y, z, checks=COMPARE))
        # The products reproduce jax.jit's cube. Primitive by primitive, XLA's power differs from
        # the correctly rounded cube on 2 of these 4096 values, and with a constant exponent it
        # has no window, so nothing is listed. On the first 64 it returns the IR's words.
        cube = self.jax.jit(lambda a: a ** 3.0)
        report = self.verify(cube, x, checks=COMPARE)
        self.assertEqual(self.advice(report), [])
        self.assertEqual(self.emulated(report), ["pow_products"])
        self.assertEqual(report["checks"][-1]["results"][0]["not_reproduced"], 0)
        self.assertEqual(report["checks"][-1]["eager_reference"]["status"], "not_reproduced")
        report = self.verify(cube, x[:64], checks=COMPARE)
        self.assertEqual(self.advice(report), ["pow_products"])

    def test_binary16_jit_powers_round_each_product(self):
        # Under jax.jit XLA computes binary16 x ** 3.0 as x * (x * x) with each product rounded
        # to binary16. The IR's power is binary32's, rounded once.
        jnp, np = self.jnp, self.np
        x = jnp.asarray((np.random.default_rng(11).standard_normal(4096) * 3).astype(np.float16))
        cube = self.jax.jit(lambda v: v ** 3.0)
        f32 = np.asarray(x).astype(np.float32)
        steps = (f32 * (f32 * f32).astype(np.float16).astype(np.float32)).astype(np.float16)
        self.assertTrue((np.asarray(cube(x)).view(np.uint16) == steps.view(np.uint16)).all())
        self.assertEqual(self.advice(self.verify(cube, x, checks=COMPARE)), ["pow_products"])

    def test_flushed_subnormals_are_consistent_with_flushing(self):
        # XLA's CPU code flushes subnormal operands and results to zero. The IR underflows
        # gradually, as IEEE arithmetic does. rsqrt of a flushed operand is infinite.
        lax, jnp, np = self.jax.lax, self.jnp, self.np
        dividend = np.array([1e-30, 1e-38], dtype=np.float32)
        divisor = np.array([1e10, 100.0], dtype=np.float32)
        report = self.verify(lax.div, jnp.asarray(dividend), jnp.asarray(divisor), checks=COMPARE)
        words = np.array(report["outputs"][0]["bits"], dtype=np.uint32)
        self.assertTrue((words == (dividend / divisor).view(np.uint32)).all())
        self.assertEqual(np.asarray(lax.div(jnp.asarray(dividend), jnp.asarray(divisor))).tolist(),
                         [0.0, 0.0])
        self.assertEqual(self.advice(report), ["subnormal"])
        tiny = jnp.asarray(np.array([1e-40, 1e-39, 1e-38], dtype=np.float32))
        self.assertEqual(self.advice(self.verify(lax.rsqrt, tiny, checks=COMPARE)), ["subnormal"])
        square = self.verify(lambda v: v ** 2, jnp.asarray(np.array([1e-20], np.float32)),
                             checks=COMPARE)
        self.assertEqual(self.advice(square), ["subnormal"])

    def test_integer_division_rounds_toward_zero(self):
        jnp = self.jnp
        a = jnp.array([7, -7, 9, -9], dtype=jnp.int32)
        b = jnp.array([2, 2, -4, -4], dtype=jnp.int32)
        self.assert_passes(self.verify(self.jax.lax.div, a, b, checks=COMPARE))

    def test_reduced_operations_match_xla_words(self):
        # XLA computes binary16 and bfloat16 pow and rsqrt in binary32 and rounds once.
        lax, jnp = self.jax.lax, self.jnp
        for dtype in (jnp.bfloat16, jnp.float16):
            with self.subTest(dtype=dtype):
                def reduced(x, y):
                    return lax.pow(x, y), lax.rsqrt(x), x / y

                self.assert_passes(self.verify(
                    reduced, (jnp.abs(self.x) + 0.5).astype(dtype), self.y.astype(dtype),
                    checks=COMPARE))

    def test_reduced_conversions_round_once(self):
        # convert_element_type rounds the exact value once (IEEE 754-2019 5.4.1, 5.4.2): int32
        # 16842753 becomes bfloat16 16908288, where XLA's CPU backend rounds through binary32 and
        # gives 16777216.
        from fractions import Fraction

        jax, jnp, np = self.jax, self.jnp, self.np
        with jax.enable_x64(True):
            a = jnp.asarray(np.array([16842753, -16842753, 3, 65519, 65520], dtype=np.int32))
            b = jnp.asarray(np.array([16842753, 2 ** 40 + 2 ** 32 + 1, -(2 ** 40 + 2 ** 32 + 1),
                                      2 ** 62 + 2 ** 54 + 1], dtype=np.int64))
            c = jnp.asarray(np.array([1 + 2 ** -8 + 2 ** -30, -(1 + 2 ** -8 + 2 ** -30),
                                      1 + 2 ** -11 + 2 ** -40, 0.1], dtype=np.float64))
            zero = jnp.zeros(5, dtype=jnp.bfloat16)
            self.assertEqual(int(np.asarray(a[:1].astype(jnp.bfloat16)).view(np.uint16)[0]),
                             0x4B80)

            def conversions(a, b, c, zero):
                return (a.astype(jnp.bfloat16), a.astype(jnp.float16), b.astype(jnp.bfloat16),
                        b.astype(jnp.float16), c.astype(jnp.bfloat16), c.astype(jnp.float16),
                        a + zero)

            for function in (conversions, jax.jit(conversions)):
                report = self.verify(function, a, b, c, zero, checks=EXECUTE)
                self.assert_passes(report)
                once = [[_nearest(Fraction(int(v) if source is not c else float(v)), dtype)
                         for v in np.asarray(source).tolist()]
                        for source in (a, b, c) for dtype in ("bfloat16", "float16")]
                self.assertEqual(len(report["outputs"]), len(once) + 1)
                for output, want in zip(report["outputs"], once + [once[0]]):
                    got = np.array(output["bits"], dtype=np.uint16).view(
                        jnp.bfloat16 if output["element_type"] == "bfloat16" else np.float16)
                    _same(self, [float(v) for v in got.astype(np.float64)], want)
        # To an integer type a floating value truncates toward zero, as the documented
        # static_cast does, and integers wrap modulo 2 ** width. A truncated value the type
        # cannot hold is an execution error.
        floats = jnp.array([1.7, -1.7, 2.5, 3.5, -0.5], jnp.float32)
        report = self.verify(lambda x, n: (x.astype(jnp.int32), n.astype(jnp.uint8)), floats,
                             jnp.array([300, -1, 7], jnp.int32), checks=EXECUTE)
        self.assert_passes(report)
        self.assertEqual([output["bits"] for output in report["outputs"]],
                         [[1, 2 ** 32 - 1, 2, 3, 0], [44, 255, 7]])
        # The report gives the run's complete conversion status: truncating 1.7 is inexact, and
        # 300 and -1 wrap.
        self.assertEqual(report["conversion_flags"], {
            "inexact": True, "overflow": False, "underflow": False, "saturated": False,
            "wrapped": True, "mapped_special": False, "invalid": False})
        self.assert_unsupported_execution(self.verify(
            lambda x: x.astype(jnp.int32), jnp.array([3e9], jnp.float32), checks=EXECUTE),
            "does not fit")

    def test_rounded_operations_execute(self):
        lax, jnp = self.jax.lax, self.jnp

        def rounded(x, y):
            mean = jnp.mean(x, axis=-1, keepdims=True)
            variance = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
            normalized = (x - mean) * lax.rsqrt(variance + 1e-5)
            return lax.pow(jnp.abs(x) + 1, y), self.jax.nn.gelu(normalized)

        self.assert_passes(self.verify(rounded, self.x, self.y))

    def test_error_function_needs_an_implementation(self):
        # The typed runtime has no float32 rule for erf: the types check passes and execution
        # refuses the step.
        report = self.verify(self.jax.lax.erf, self.x)
        self.assert_unsupported_execution(report, "no rule at float32")

    def test_convolutions_execute_with_their_payload(self):
        # The kernel operand of conv_general_dilated becomes the convolution payload, plain and
        # with dilation and groups. Integer kernels and inputs make every summation order exact.
        jnp, lax, np = self.jnp, self.jax.lax, self.np
        rng = np.random.default_rng(0)

        def integers(shape):
            return jnp.asarray(rng.integers(-3, 4, shape), dtype=jnp.float32)

        numbers = ("NCHW", "OIHW", "NCHW")
        plain, grouped = integers((4, 2, 3, 3)), integers((8, 2, 3, 3))
        cases = {
            "plain": (lambda x: lax.conv_general_dilated(x, plain, (1, 1), "VALID",
                                                         dimension_numbers=numbers),
                      integers((2, 2, 6, 6))),
            "dilation and groups": (lambda x: lax.conv_general_dilated(
                x, grouped, (1, 1), ((1, 1), (2, 2)), rhs_dilation=(1, 2),
                dimension_numbers=numbers, feature_group_count=2), integers((1, 4, 5, 7)))}
        for name, (function, x) in cases.items():
            with self.subTest(case=name):
                self.assert_passes(self.verify(function, x, checks=COMPARE))


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


def _steps(document):
    # The steps of the artifact of a capture document.
    return document["artifact"]["nodes"]


def _records(document):
    # The operator records of the artifact's op steps, in step order.
    return [step["op"]["record"] for step in _steps(document) if "op" in step]


def _tags(document):
    return [record["tag"] for record in _records(document)]


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


@unittest.skipUnless(CHECKER and HAS_TORCH, "needs the built verify executable and PyTorch")
class PyTorchGeneralOperatorTests(CheckerCase):
    # One capture per family of core ATen operators onto the general IR operators, and the
    # inputs the capture must refuse. Integer-valued entries make every summation order exact,
    # so `compare` holds against PyTorch's own reduction order.
    def setUp(self):
        import torch

        self.torch = torch
        self.x = torch.arange(24.).reshape(2, 3, 4) - 10

    def verify(self, function, *inputs, checks=COMPARE, core=False):
        torch = self.torch
        original = torch.export.export
        if core:
            torch.export.export = lambda *a, **k: original(*a, **k).run_decompositions()
        try:
            return torchlean.verify(function, *inputs, checks=checks, checker=CHECKER)
        finally:
            torch.export.export = original

    def artifact(self, function, *inputs, core=False):
        # The document the capture writes, before the checker reads it.
        torch = self.torch
        original = torch.export.export
        if core:
            torch.export.export = lambda *a, **k: original(*a, **k).run_decompositions()
        try:
            return torchlean._capture(function, inputs, "pytorch", ("structure",))
        finally:
            torch.export.export = original

    def test_reductions_are_one_node(self):
        from torchlean._table import op_table

        torch = self.torch

        def reductions(x):
            return (torch.amax(x, dim=(0, 2)), torch.amin(x, dim=-1, keepdim=True),
                    x.sum(dim=(0, 2)), x.prod(dim=1), x.mean(dim=(1, 2)), x.sum(dim=[]))

        for core in (False, True):
            with self.subTest(core=core):
                self.assert_passes(self.verify(reductions, self.x, core=core))
        artifact = self.artifact(reductions, self.x)
        self.assertEqual(artifact["artifact"]["format"], op_table()["format"])
        reduce = [record["fields"][1:] for record in _records(artifact)
                  if record["tag"] == "reduce"]
        self.assertIn([{"nat": 0}, {"nats": [0, 2]}, {"bool": False}], reduce)
        # PyTorch reads an empty dim list as every axis.
        self.assertIn([{"nat": 0}, {"nats": [0, 1, 2]}, {"bool": False}], reduce)
        # The variance is TorchLean's two-pass schedule. PyTorch's Welford loop may round
        # differently, so it is executed only.
        variance = lambda x: torch.var(x, dim=(0, 2), correction=1, keepdim=True)
        self.assert_passes(self.verify(variance, self.x, checks=EXECUTE, core=True))

    def test_softmax_family_and_products(self):
        torch = self.torch
        functional = torch.nn.functional
        weight = torch.arange(12.).reshape(4, 3) / 4

        def products(x):
            y = torch.bmm(x, x.transpose(1, 2))
            z = torch.addmm(torch.ones(3), x[0], weight)
            return (functional.log_softmax(y, dim=-1), torch.softmax(z, dim=0),
                    torch.matmul(x, weight))

        for core in (False, True):
            with self.subTest(core=core):
                self.assert_passes(self.verify(products, self.x, checks=EXECUTE, core=core))
        kinds = _tags(self.artifact(products, self.x, core=True))
        self.assertIn("logSoftmax", kinds)
        self.assertIn("softmax", kinds)

    def test_comparisons_logical_operators_and_selections_match_eager_words(self):
        # Comparisons, class predicates and logical and bitwise operators give Boolean nodes,
        # where and masked_fill select with them, clamp, clip and hardtanh are the IR clamp, and
        # elu and softplus select between their kernels' branches. In binary32 and binary16 the
        # words agree with PyTorch's on entries with NaNs, zeros of both signs and infinities, the
        # bounds away from zero, elu's expm1 and softplus's log1p and exp exact. Bitwise
        # operators of integers and operands of two dtypes are refused.
        torch = self.torch
        functional = torch.nn.functional
        inf, nan = float("inf"), float("nan")
        x = torch.tensor([[nan, -0.0, 0.0, 1.5], [-inf, inf, -2.0, 30.0], [0.25, -1.0, 25.0, 3.0]])
        y = torch.tensor([[1.0, 0.0, -0.0, 1.5], [-inf, 2.0, nan, 30.0], [0.5, -1.0, 3.0, -3.0]])
        exact = torch.tensor([nan, -0.0, 0.0, 1.5, -inf, inf, 30.0, 25.0])
        large = torch.tensor([nan, -inf, inf, 30.0, 25.0, 21.0])

        def comparisons(x, y):
            return (x == y, x != y, x < y, x <= y, torch.gt(x, y), torch.ge(x, 0.25), x > 0,
                    torch.isnan(x), torch.isinf(x), torch.isfinite(x))

        def logical(x, y):
            p, q = x > 0, y < 1
            return (torch.logical_and(x, y), torch.logical_or(p, q), torch.logical_xor(x, y),
                    torch.logical_not(x), p & q, p | q, p ^ q, ~p)

        def selections(x, y):
            return (torch.where(x > y, x, y), torch.where(x > 0, x, 0.5),
                    x.masked_fill(x < 0, -1.0), torch.clamp(x, -1.0, 2.0),
                    torch.clamp(x, y - 4, y + 4), torch.clip(x, -2.0, 1.0),
                    functional.hardtanh(x))

        cases = {"comparisons": (comparisons, x, y), "logical": (logical, x, y),
                 "selections": (selections, x, y),
                 "elu": (lambda e: (functional.elu(e), functional.elu(e, alpha=0.5)), exact),
                 "softplus": (functional.softplus, large)}
        for name, (function, *inputs) in cases.items():
            for dtype in (torch.float32, torch.float16):
                for core in (False, True):
                    with self.subTest(case=name, dtype=dtype, core=core):
                        typed = [value.to(dtype) for value in inputs]
                        self.assert_passes(self.verify(function, *typed, core=core))
        integers = torch.arange(6)
        refused = {"integer predicate": (lambda i: torch.isnan(i), (integers,), "no element rule"),
                   "two dtypes": (lambda a, b: a < b, (x, x.double()), "of two dtypes")}
        for name, (function, inputs, text) in refused.items():
            with self.subTest(case=name):
                self.assert_refused(self.verify(function, *inputs, checks=("structure",)), text)

    def test_integer_comparisons_and_arithmetic_match_eager(self):
        # Integer comparisons and logical operators read the values, add, sub and mul wrap
        # modulo 2 ** width and maximum and minimum follow the type's order, as ATen's integer
        # kernels do, and where selects integers. A Python int converts to the tensor's type
        # (300 wraps to 44 in int8) and a float promotes the comparison to binary32.
        torch = self.torch
        values = {torch.int8: [127, -128, 5, 0, -3, 2], torch.uint8: [255, 0, 5, 7, 2, 128],
                  torch.int32: [2 ** 31 - 1, -2 ** 31, 5, 0, -3, 2],
                  torch.int64: [2 ** 63 - 1, -2 ** 63, 5, 0, -3, 2]}

        def integers(a, b):
            return (a == b, a != b, a < b, a <= b, a > b, a >= b, a < 2, a >= 2.5, a < 300,
                    torch.logical_and(a, b), torch.logical_not(a), a + b, a - b, a * b, a + 3,
                    torch.maximum(a, b), torch.minimum(a, b), torch.where(a > b, a, b))

        for dtype, entries in values.items():
            a = torch.tensor(entries, dtype=dtype)
            b = torch.tensor([5, 1, 5, -1 if dtype != torch.uint8 else 1, 2, 2], dtype=dtype)
            for core in (False, True):
                with self.subTest(dtype=dtype, core=core):
                    self.assert_passes(self.verify(integers, a, b, core=core))

    def test_sqrt_is_ieee_square_root(self):
        # torch.sqrt imports as Op.sqrt at `.raw`, IEEE squareRoot, with no condition on its
        # operand: -0 at -0, NaN below zero and at a NaN, inf at inf, in torch.export's default and
        # core forms and in every floating dtype. The words are the correctly rounded roots
        # (math.sqrt's, rounded again to a narrower format, which keeps a root correctly rounded).
        # Eager PyTorch's binary64 and binary32 roots can be one ulp off (2.13's CPU kernel gives
        # 0x3FF6A09E667F3BCC at 2, IEEE 0x3FF6A09E667F3BCD), so there `compare` passes exactly
        # where PyTorch's words are IEEE's. An integer operand, which torch.sqrt converts to
        # binary32, is refused.
        import math

        torch = self.torch
        values = [-0.0, 0.0, -1.0, float("inf"), float("nan"), 4.0, 2.0, 0.5]
        roots = [math.nan if v < 0 else math.sqrt(v) for v in values]
        for dtype in (torch.float64, torch.float32, torch.bfloat16, torch.float16):
            expected = [_nearest(root, str(dtype).removeprefix("torch.")) for root in roots]
            for core in (False, True):
                with self.subTest(dtype=dtype, core=core):
                    x = torch.tensor(values, dtype=dtype)
                    report = self.verify(torch.sqrt, x, core=core)
                    _same(self, _outputs(report)[0].tolist(), expected, dtype)
                    eager = torch.sqrt(x).tolist()
                    if all(a != a and b != b or (a, math.copysign(1.0, a)) ==
                           (b, math.copysign(1.0, b)) for a, b in zip(eager, expected)):
                        self.assert_passes(report)
                    else:
                        statuses = _statuses(report)
                        self.assertEqual({name: status for name, (status, _) in statuses.items()},
                                         {"structure": "checked", "types": "checked",
                                          "execute": "executed", "compare": "failed"})
        self.assert_refused(self.verify(torch.sqrt, torch.arange(4), checks=("structure",)),
                            "sqrt of a")

    def test_logarithms_at_nonfinite_values_and_empty_means(self):
        # log at zero, below zero and at NaN, and log_softmax of rows with infinite and NaN entries
        # give PyTorch's words (two NaNs agree), including a row whose only +inf is its maximum,
        # [inf, 1, 2, 3], which is four NaNs. A mean over an empty fiber is refused. The variance
        # there is PyTorch's 0 / 0.
        torch = self.torch
        values = torch.tensor([0.0, -0.0, -1.0, float("nan"), 1.0, float("inf")])
        inf = float("inf")
        rows = torch.tensor([[-inf, 0.0, -inf, -inf], [inf, inf, 0.0, 0.0],
                             [-inf, -inf, -inf, -inf], [float("nan"), 0.0, 0.0, 0.0],
                             [inf, 1.0, 2.0, 3.0]])
        empty = torch.ones(3, 0)
        for core in (False, True):
            with self.subTest(core=core):
                self.assert_passes(self.verify(torch.log, values, core=core))
                self.assert_passes(self.verify(lambda x: torch.log_softmax(x, 1), rows,
                                               core=core))
                self.assert_passes(self.verify(lambda x: torch.var(x, 1), empty, core=core))
                self.assert_refused(self.verify(lambda x: x.mean(1), empty, checks=("structure",),
                                                core=core), "a mean over an empty fiber")

    def test_data_movement(self):
        torch = self.torch
        functional = torch.nn.functional

        def movement(x):
            pieces = torch.split(x, [1, 3], dim=-1)
            return (functional.pad(x, (1, -1, 0, 2), value=3.0), torch.flip(x, dims=(0, 2)),
                    x.unsqueeze(1).expand(2, 5, 3, 4), x[:, 1], x.repeat(1, 2, 1),
                    torch.cat([pieces[1], pieces[0]], dim=-1), x.permute(2, 0, 1).squeeze(),
                    torch.chunk(x, 2, dim=1)[1])

        def edges(x):
            return (functional.pad(x, (2, 1), mode="reflect"),
                    functional.pad(x, (1, 2), mode="replicate"))

        for core in (False, True):
            with self.subTest(core=core):
                self.assert_passes(self.verify(movement, self.x, core=core))
        kinds = _tags(self.artifact(movement, self.x, core=True))
        for kind in ("pad", "flip", "slice", "cat", "broadcast", "permute", "reshape"):
            self.assertIn(kind, kinds)
        # Reflection and replication padding are flipped or repeated edge slices. Core ATen
        # writes them as index.Tensor with indices computed from constants, which fold at
        # capture into a registered gather.
        self.assert_passes(self.verify(edges, self.x))
        self.assertIn("flip", _tags(self.artifact(edges, self.x)))
        self.assert_passes(self.verify(edges, self.x, core=True))
        records = _registered(self.artifact(edges, self.x, core=True))
        self.assertIn("gather", [record["tag"] for record in records])

    def test_strided_slices_keep_their_step(self):
        artifact = self.artifact(lambda x: x[:, ::2, 1::2], self.x)
        (record,) = [record for record in _records(artifact) if record["tag"] == "slice"][-1:]
        self.assertEqual(record["fields"][4], {"nats": [2]})

    def test_exact_roundings_match_eager_words(self):
        torch = self.torch
        x = torch.tensor([[-2.5, -1.5, -0.5, -0.0], [0.0, 0.5, 1.5, 2.5],
                          [-7.25, 7.75, float("inf"), float("nan")]])

        def roundings(x):
            return (torch.floor(x), torch.ceil(x), torch.round(x), torch.trunc(x), torch.fix(x),
                    torch.frac(x), torch.sign(x))

        class Roundings(torch.nn.Module):
            def forward(self, x):
                return roundings(x)

        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for core in (False, True):
                with self.subTest(dtype=dtype, core=core):
                    values = x.to(dtype)
                    if core:
                        # PyTorch's frac decomposition can change the sign of zero. The import
                        # must match that captured graph, including its zero signs.
                        graph = torch.export.export(Roundings(), (values,)).run_decompositions()
                        expected = graph.module()(values)
                        report = self.verify(roundings, values, core=True, checks=EXECUTE)
                        self.assert_passes(report)
                        outputs = _outputs(report)
                        self.assertEqual(len(outputs), len(expected))
                        for got, want in zip(outputs, expected):
                            _same(self, got.flatten().tolist(), want.flatten().tolist())
                    else:
                        self.assert_passes(self.verify(roundings, values))
        kinds = _tags(self.artifact(roundings, x))
        # torch.sign is captured as comparisons and selections: it maps a NaN to 0 and -0 to +0,
        # where the IR's sign gives the NaN and keeps -0.
        for kind in ("floor", "ceil", "round", "trunc", "frac", "select"):
            self.assertIn(kind, kinds)
        self.assertNotIn("sign", kinds)

    def test_binary_functions_match_eager_words(self):
        # copysign, fmod, fmax and fmin are exact, so every element compares. hypot is computed in
        # binary32 for the reduced formats, so it compares in bfloat16 and binary16.
        torch = self.torch
        nan, inf = float("nan"), float("inf")
        x = torch.tensor([[2.5, -1.5, 0.0, -0.0], [7.25, nan, inf, -3.0], [1.0, -2.0, 0.5, 4.0]])
        y = torch.tensor([[-1.0, 0.75, -0.0, 2.0], [2.0, 1.0, nan, -inf], [nan, 3.0, -0.5, 1.5]])

        def exact(x, y):
            # PyTorch's fmax and fmin of mixed signed zeros depend on the dtype's vectorized
            # loop, so `y + 0.0` turns the -0.0 paired with 0.0 into 0.0 for them.
            w = y + 0.0
            return (torch.copysign(x, y), torch.fmod(x, y), torch.fmax(x, w), torch.fmin(x, w),
                    torch.copysign(x, -1.0), torch.fmod(x, 0.75))

        def opmath(x, y):
            return torch.hypot(x, y)

        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                self.assert_passes(self.verify(exact, x.to(dtype), y.to(dtype)))
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                self.assert_passes(self.verify(opmath, x.to(dtype), y.to(dtype)))
        self.assert_passes(self.verify(opmath, x, y, checks=EXECUTE))
        kinds = _tags(self.artifact(exact, x, y)) + _tags(self.artifact(opmath, x, y))
        for kind in ("copysign", "fmod", "fmax", "fmin", "hypot"):
            self.assertIn(kind, kinds)

    def test_elementary_functions_execute(self):
        # FloatLib's trigonometric and hyperbolic kernels are approximations, and PyTorch's
        # vectorized functions round differently in some last bits, so these are executed only.
        torch = self.torch
        x = torch.linspace(-2, 2, 12).reshape(3, 4)
        unit = torch.linspace(-0.9, 0.9, 12).reshape(3, 4)

        def elementary(x, u):
            return (torch.expm1(x), torch.log1p(x.abs()), torch.exp2(x), torch.log2(x.abs()),
                    torch.log10(x.abs()), torch.tan(u), torch.sinh(x), torch.cosh(x),
                    torch.asinh(x), torch.acosh(x.abs() + 1), torch.atanh(u))

        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.assert_passes(self.verify(elementary, x.to(dtype), unit.to(dtype),
                                               checks=EXECUTE))
        kinds = _tags(self.artifact(elementary, x, unit))
        for kind in ("expm1", "log1p", "exp2", "log2", "log10", "tan", "sinh", "cosh", "asinh",
                     "acosh", "atanh"):
            self.assertIn(kind, kinds)
        # A binary16 operand is computed in binary32 and rounded once.
        half = self.artifact(torch.expm1, x.to(torch.float16))
        (scalar,) = [scalar for record, scalar in _ops(half) if record["tag"] == "expm1"]
        self.assertEqual(scalar, "float32")

    def test_inverse_trigonometric_functions_within_one_ulp_of_eager(self):
        # The IR's asin, acos, atan and atan2 round the real function once. On these inputs,
        # eager PyTorch 2.13 on x86-64 with AVX-512 is within one ulp of the IR in binary32 and
        # binary64, and its binary16 and bfloat16 kernels compute in binary32 and round once, as
        # the IR does. The NaNs, zeros and infinities agree in position, the zeros and infinities
        # in sign, every pair of the special values of atan2 included.
        torch = self.torch
        special = [0.0, -0.0, 1.0, -1.0, 2.0, -3.0, float("inf"), -float("inf"), float("nan")]
        pairs = [(a, b) for a in special for b in special]
        for dtype in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
            # Samples drawn in binary64 and converted, so binary64 gets binary64 samples.
            generator = torch.Generator().manual_seed(1)
            unit = torch.cat([torch.rand(41, generator=generator, dtype=torch.float64) * 2 - 1,
                              torch.tensor(special, dtype=torch.float64)]).to(dtype)
            x = torch.cat([torch.randn(41, generator=generator, dtype=torch.float64) * 4,
                           torch.tensor([b for _, b in pairs], dtype=torch.float64)]).to(dtype)
            y = torch.cat([torch.randn(41, generator=generator, dtype=torch.float64) * 4,
                           torch.tensor([a for a, _ in pairs], dtype=torch.float64)]).to(dtype)

            def inverse(u, x, y):
                return torch.asin(u), torch.acos(u), torch.atan(x), torch.atan2(y, x)

            with self.subTest(dtype=dtype):
                report = self.verify(inverse, unit, x, y, checks=EXECUTE)
                self.assert_passes(report)
                for actual, expected in zip(_outputs(report), inverse(unit, x, y)):
                    _assert_within_ulps(self, actual, expected, 1)
                artifact = self.artifact(inverse, unit, x, y)
                ops = [(record["tag"], scalar) for record, scalar in _ops(artifact)
                       if record["tag"] in ("asin", "acos", "atan", "atan2")]
                self.assertEqual(len(ops), 4)
                # binary16 and bfloat16 compute in binary32 between two casts, R16(R32(f(x))).
                computed = "float32" if dtype in (torch.bfloat16, torch.float16) else \
                    str(dtype).split(".")[1]
                self.assertTrue(all(scalar == computed for _, scalar in ops), ops)

    def test_ulp_helper_rejects_far_values(self):
        # Opposite signs, a zero against a subnormal and a finite value against an infinity.
        torch = self.torch
        cases = [(-2.0, 2.0, torch.float64), (0.0, 5e-324, torch.float64),
                 (3.4028234663852886e38, float("inf"), torch.float32)]
        for actual, expected, dtype in cases:
            with self.subTest(actual=actual, expected=expected):
                with self.assertRaises(AssertionError):
                    _assert_within_ulps(self, torch.tensor([actual], dtype=dtype),
                                        torch.tensor([expected], dtype=dtype), 1)

    def test_remainder_is_python_modulus(self):
        # torch.remainder documents Python's modulus, "the same sign as the divisor other", and the
        # IR computes it: fmod, the divisor added to a nonzero remainder of the other sign, and a
        # zero result with the divisor's sign. Binary16 and bfloat16 add in binary32 and round
        # once. The expected words are Python's `%` of the binary64 values on CPU (NumPy's NaN at
        # a zero divisor, where Python raises), rounded to binary32 and then to the dtype. Eager
        # PyTorch gives a zero result the dividend's sign, so it agrees at every other result.
        torch = self.torch
        generator = torch.Generator().manual_seed(2)
        special = [0.0, -0.0, 1.0, -5.0, float("inf"), -float("inf"), float("nan"), 7.5, -2.0,
                   0.0]
        x = torch.cat([torch.randn(40, generator=generator) * 10, torch.tensor(special)])
        y = torch.cat([torch.randn(40, generator=generator) * 3,
                       torch.tensor([0.0, 2.0, -0.0, float("inf"), 3.0, -2.0, 1.0, -0.25, 1.0,
                                     -3.0])])

        def remainder(x, y):
            return torch.remainder(x, y), x % y, torch.remainder(x, -2.5), 2.5 % y

        def python(a, b):
            try:
                return a % b
            except ZeroDivisionError:
                return float("nan")

        for dtype in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                xs, ys = x.to(dtype), y.to(dtype)
                report = self.verify(remainder, xs, ys, checks=EXECUTE)
                self.assert_passes(report)
                operands = [(xs, ys), (xs, ys), (xs, torch.full_like(xs, -2.5)),
                            (torch.full_like(ys, 2.5), ys)]
                outputs = _outputs(report)
                self.assertEqual(len(outputs), 4)
                for actual, eager, (a, b) in zip(outputs, remainder(xs, ys), operands):
                    expected = torch.tensor(
                        [python(u, v) for u, v in zip(a.double().tolist(), b.double().tolist())],
                        dtype=torch.float64)
                    if dtype != torch.float64:
                        expected = expected.to(torch.float32)
                    expected = expected.to(dtype)
                    nan = expected.isnan()
                    self.assertTrue(torch.equal(actual.isnan(), nan))
                    width = torch.int64 if dtype == torch.float64 else \
                        torch.int32 if dtype == torch.float32 else torch.int16
                    self.assertEqual(actual[~nan].view(width).tolist(),
                                     expected[~nan].view(width).tolist())
                    zero = expected == 0
                    self.assertTrue(bool(zero.any()))
                    self.assertEqual(eager[~nan & ~zero].view(width).tolist(),
                                     expected[~nan & ~zero].view(width).tolist())
                    self.assertTrue(bool((eager[zero] == 0).all()))
        self.assertIn("remainder", _tags(self.artifact(remainder, x, y)))

    def test_index_outputs_match_eager(self):
        # argmax and argmin, and the indices of max.dim, min.dim, cummax, cummin and the stable
        # sort, are the registered operators Op.argmax, Op.argmin, Op.cummax, Op.cummin and
        # Op.sort with int64 results. `compare` requires the eager words, NaN results of the value
        # outputs compared by class. The rows repeat values, signed
        # zeros and NaNs: argmax and argmin keep the first extremum and the first NaN, cummax and
        # cummin the latest, and the stable sort keeps equal entries in order.
        torch = self.torch
        nan = float("nan")
        x = torch.tensor([[1.0, 3.0, 3.0, -0.0, 0.0], [nan, 2.0, nan, -1.0, 5.0],
                          [0.0, -0.0, 0.0, -0.0, -0.0], [7.0, -2.0, 7.0, nan, 1.0]])

        def indices(x):
            return (torch.argmax(x, dim=1), torch.argmin(x, dim=0, keepdim=True),
                    torch.argmax(x), torch.argmin(x, keepdim=True),
                    torch.max(x, dim=1).indices, torch.min(x, dim=1, keepdim=True)[1],
                    torch.cummax(x, dim=1).indices, torch.cummin(x, dim=0)[1],
                    torch.sort(x, dim=1, stable=True).indices,
                    torch.sort(x, dim=0, descending=True, stable=True)[1])

        def both(x):
            values, positions = torch.max(x, dim=1)
            running, places = torch.cummin(x, dim=1)
            return values, positions, running, places, positions.reshape(2, 2)

        for dtype in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                self.assert_passes(self.verify(indices, x.to(dtype)))
                self.assert_passes(self.verify(both, x.to(dtype)))
        scalar = torch.tensor(2.5)
        self.assert_passes(self.verify(lambda s: (torch.argmax(s), torch.argmin(s, dim=0),
                                                  torch.argmax(s, dim=-1, keepdim=True)), scalar))
        empty = torch.zeros(0, 3)
        self.assert_passes(self.verify(lambda e: torch.argmax(e, dim=1), empty))
        artifact = self.artifact(indices, x)
        records = _registered(artifact)
        self.assertEqual(sorted({record["tag"] for record in records}),
                         ["argmax", "argmin", "cummax", "cummin", "sort"])
        encoded = [record for record in _records(artifact) if record["tag"] == "encodeIndex"]
        self.assertTrue(all(record["fields"][2:] == [{"bool": True}, {"nat": 64}]
                            for record in encoded), encoded)

    def test_unsupported_components_and_arguments_are_refused(self):
        torch = self.torch
        functional = torch.nn.functional
        cases = {
            "unstable sort indices": lambda x: torch.sort(x, dim=1).indices,
            "unsorted topk indices": lambda x: torch.topk(x, 2, dim=1, sorted=False).indices,
            "integer argmax": lambda x: torch.argmax(x.to(torch.int32), dim=1),
            "fractional correction": lambda x: torch.var(x, dim=1, correction=0.5),
            "addmm with beta": lambda x: torch.addmm(x[0, 0, :3], x[0], x[1].t(), beta=2.0),
            "round with decimals": lambda x: torch.round(x, decimals=1),
            "integer floor": lambda x: torch.floor(x.to(torch.int32)),
            "integer fmod": lambda x: torch.fmod(x.to(torch.int32), 3),
            "lerp without a capture rule": lambda x: torch.lerp(x, x * 2, 0.25),
        }
        for name, function in cases.items():
            with self.subTest(name=name):
                report = self.verify(function, self.x, checks=("structure",))
                self.assertFalse(report["ok"])
                self.assertEqual(report["checks"][0]["status"], "unsupported", report)
        # native_layer_norm's mean is not imported, but its normalized output is.
        report = self.verify(lambda x: torch.native_layer_norm(x, (4,), None, None, 1e-5)[1],
                             self.x, checks=("structure",))
        self.assertFalse(report["ok"])
        self.assert_passes(self.verify(lambda x: functional.layer_norm(x, (4,)), self.x,
                                       checks=EXECUTE, core=True))

    def test_scans_sorts_and_extrema_match_eager_words(self):
        # cumsum and cumprod accumulate in PyTorch's accumulation type and round each prefix. The
        # stable sort and the first extremum of max.dim and min.dim are the IR's stable topk.
        torch = self.torch
        generator = torch.Generator().manual_seed(0)
        x = torch.randn(3, 20, generator=generator) * 10
        x[0, :4] = torch.tensor([0.0, -0.0, float("nan"), 1.0])

        def orders(x):
            return (torch.cumsum(x, -1), torch.cumprod(x * 0.1 + 1, 1), torch.cummax(x, 1).values,
                    torch.cummin(x, 0)[0], torch.sort(x, dim=1, stable=True, descending=True)[0],
                    torch.max(x, 1)[0], torch.min(x, -1, keepdim=True).values)

        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for core in (False, True):
                with self.subTest(dtype=dtype, core=core):
                    self.assert_passes(self.verify(orders, x.to(dtype), core=core))
        self.assert_passes(self.verify(lambda x: torch.cumsum(x, 0), x.double()))
        kinds = _tags(self.artifact(orders, x))
        for kind in ("cumsum", "cumprod", "cummax", "cummin", "sort", "topk", "neg"):
            self.assertIn(kind, kinds)
        # A binary16 prefix sum is accumulated in binary32, a binary32 one in binary64.
        for dtype, accumulator in ((torch.float16, "float32"), (torch.float32, "float64")):
            artifact = self.artifact(lambda x: torch.cumsum(x, 1), x.to(dtype))
            (scalar,) = [scalar for record, scalar in _ops(artifact) if record["tag"] == "cumsum"]
            self.assertEqual(scalar, accumulator)

    def test_leaky_relu_and_diagonal_match_eager_words(self):
        # leaky_relu is maximum(x, x * slope) for slopes up to one and minimum(x, x * slope) above
        # one, and diagonal is a permutation, a reshape and a strided slice.
        torch = self.torch
        functional = torch.nn.functional
        generator = torch.Generator().manual_seed(1)
        x = torch.randn(3, 20, generator=generator) * 10
        x[0, :5] = torch.tensor([0.0, -0.0, float("nan"), float("inf"), -float("inf")])

        def leaky(x):
            return (functional.leaky_relu(x, 0.2), functional.leaky_relu(x, 3.0),
                    torch.nn.LeakyReLU()(x))

        def diagonals(x):
            return (torch.diagonal(x, 1, 1, 2), torch.diagonal(x, -1, 2, 0),
                    torch.diagonal(x, 5), x.diagonal(0, -1, -2) * 2)

        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for core in (False, True):
                with self.subTest(dtype=dtype, core=core):
                    self.assert_passes(self.verify(leaky, x.to(dtype), core=core))
        for core in (False, True):
            with self.subTest(core=core):
                self.assert_passes(self.verify(diagonals, self.x, core=core))
        kinds = _tags(self.artifact(leaky, x))
        self.assertIn("max", kinds)
        self.assertIn("min", kinds)
        steps = [record["fields"][4] for record in _records(self.artifact(diagonals, self.x))
                 if record["tag"] == "slice"]
        self.assertIn({"nats": [5]}, steps)

    def test_constant_index_gather_and_scatter_match_eager_words(self):
        # Indices held by buffers or created by arange are stored as complete coordinates.
        torch = self.torch
        nn = torch.nn

        class Gather(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("index", torch.tensor([2, 0, 3, 3]))
                self.register_buffer("rows", torch.tensor([[0, 1, 2, 0], [2, 2, 1, 0]]))

            def forward(self, x):
                return (torch.index_select(x, 2, self.index), torch.gather(x[0], 1, self.rows),
                        torch.index_select(x, 1, torch.arange(2)))

        class Scatter(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("index", torch.tensor([[0, 1, 2], [2, 0, 1]]))

            def forward(self, x, u):
                return (x.scatter(0, self.index, u), torch.scatter_add(x, 0, self.index, u),
                        x.scatter(1, self.index, 2.5))

        generator = torch.Generator().manual_seed(2)
        x, u = torch.randn(3, 3, generator=generator), torch.randn(2, 3, generator=generator)
        for core in (False, True):
            with self.subTest(core=core):
                self.assert_passes(self.verify(Gather(), self.x, core=core))
                self.assert_passes(self.verify(Scatter(), x, u, core=core))
        artifact = self.artifact(Gather(), self.x)
        steps = _steps(artifact)
        gathers = [step["op"] for step in steps
                   if "op" in step and step["op"]["record"]["tag"] == "gather"]
        self.assertEqual(gathers[0]["record"]["fields"][2], {"nats": [0, 1, 2]})
        index, _ = gathers[0]["inputs"][1]
        self.assertEqual(steps[index]["const"]["index"]["shape"], [2, 3, 4])
        self.assertEqual(len(artifact["artifact"]["constants"][index]), 2 * 3 * 4 * 3)
        scatters = [record for record in _records(self.artifact(Scatter(), x, u))
                    if record["tag"] == "scatter"]
        self.assertEqual(sorted(record["fields"][2]["nat"] for record in scatters), [0, 0, 1])

    def test_runtime_indices_match_eager(self):
        # Runtime integer indices are index operands of the registered Op.embedding, Op.gather
        # and Op.scatter. index.Tensor and index_put read a negative index as index + extent,
        # take_along_dim wraps its index modulo the extent, and scatter and index_put fold
        # repeated targets in row-major order of the index tensor, which these small cases share
        # with ATen's CPU kernels (PyTorch documents those orders as unspecified).
        torch = self.torch
        functional = torch.nn.functional
        generator = torch.Generator().manual_seed(3)
        x = torch.randn(3, 4, generator=generator)
        cube = torch.randn(2, 3, 4, generator=generator)
        table = torch.randn(5, 3, generator=generator)
        rows = torch.tensor([[0, 2, 1, 0], [1, 1, 0, 2]])
        signed = torch.tensor([[-1, 2, -3], [0, -2, 1]])
        repeated = torch.tensor([[0, 0, 0, 1], [2, 2, 1, 1], [3, 3, 3, 3]])
        heavy = torch.tensor([[2.0 ** 24, 1.0, -2.0 ** 24, 3.0]]).expand(3, 4).contiguous()

        def reads(x, i, n, t):
            return (functional.embedding(t, table), torch.gather(x, 0, i),
                    torch.gather(x, 1, i[:, :2]), torch.index_select(x, 1, i[0]),
                    torch.take_along_dim(x, n[:, :2].repeat(2, 1)[:3], 1), x[n], x[:, n],
                    x[n, n % 4], cube[:, i[0, :2], i[1, :2]], cube[i[:, :2] % 2, :, i[:, 1:3]],
                    x[torch.tensor(1)], x.gather(1, x.argmax(1, keepdim=True)),
                    torch.index_select(x, 1, (x[0] > 0).long()))

        def writes(x, i, r, u, h):
            return (x.scatter(0, i[:, :3], u[:2, :3]), torch.scatter_add(x, 0, i, u[:2]),
                    x.scatter(1, i[:, :2], 2.5), torch.scatter_add(x, 1, r, h), x.scatter(1, r, u),
                    torch.index_put(x, (i[0, :3],), u[:3]),
                    torch.index_put(x, (i[0], i[1]), h[0], accumulate=True),
                    torch.zeros(4).scatter_add(0, r[2], h[0]))

        def assignments(x, i, u):
            # Functionalized to index_put in the core graph. The default graph keeps index_put_,
            # read as index_put of its fresh destination.
            y = x.clone()
            y[:, i[1, :2]] = u[:, :2]
            z = cube * 1.0
            z[:, i[0, :2], i[1, :2]] = u[:2, :1]
            return y, z

        def read_then_write(x, i, u):
            # An earlier reader of the destination allocates its result. The write follows it.
            y = x.clone()
            total = y.sum(0)
            y[:, i[1, :2]] = u[:, :2]
            return y, total

        def view_write(x, i, u):
            # The write reaches y through the view. The default graph's index_put_ is refused.
            y = x.clone()
            row = y[0]
            row[i[1, :2]] = u[0, :2]
            return y

        def stale_view(x, i, u):
            y = x.clone()
            row = y[0]
            y[:, i[1, :2]] = u[:, :2]
            return row * 1.0, y

        u = torch.randn(3, 4, generator=generator)
        for core in (False, True):
            with self.subTest(core=core):
                self.assert_passes(self.verify(reads, x, rows, signed, torch.tensor([[1, 4, 0],
                                                                                     [3, 3, 2]]),
                                               core=core))
                self.assert_passes(self.verify(writes, x, rows, repeated, u, heavy, core=core))
                self.assert_passes(self.verify(assignments, x, rows, u, core=core))
                self.assert_passes(self.verify(read_then_write, x, rows, u, core=core))
                self.assert_passes(self.verify(lambda x, n: x[:, n], x, signed.int(), core=core))
        for dtype in (torch.bfloat16, torch.float16, torch.float64):
            with self.subTest(dtype=dtype):
                self.assert_passes(self.verify(
                    lambda x, i: (torch.gather(x, 0, i), x[i[0]], torch.scatter_add(x, 0, i, x)),
                    x.to(dtype), rows))
        for function in (view_write, stale_view):
            with self.subTest(function=function.__name__):
                self.assert_refused(self.verify(function, x, rows, u), "view")
        artifact = self.artifact(reads, x, rows, signed, torch.tensor([[1, 4, 0], [3, 3, 2]]))
        records = _registered(artifact)
        self.assertEqual(sorted({record["tag"] for record in records}),
                         ["argmax", "embedding", "gather"])
        steps = _steps(artifact)
        # The index operand of the registered gather over cube's axes 1 and 2.
        operands = [steps[step["op"]["inputs"][1][0]] for step in steps if "op" in step
                    and step["op"]["record"]["tag"] == "gather"
                    and step["op"]["record"]["fields"][2] == {"nats": [1, 2]}]
        (decoded, *_) = [step["op"]["record"] for step in operands if "op" in step]
        self.assertEqual((decoded["tag"], decoded["fields"][:2]),
                         ("decodeIndex", [{"shape": [2]}, {"nats": [3, 4]}]))

    def test_runtime_index_outside_its_axis_fails_execution(self):
        # A program captured on valid indices, run on an index outside its axis: the conversion
        # back to int64 or the registered operator's range check fails, as PyTorch raises.
        torch = self.torch
        functional = torch.nn.functional
        x = torch.randn(3, 4)
        cases = {"index": (lambda x, n: x[n], [[-4, 0]], "outside"),
                 "extent": (lambda x, n: x[n], [[3, 0]], "outside"),
                 "huge index": (lambda x, n: x[n], [[-(2 ** 53 + 1), 0]], "outside"),
                 "int64 minimum": (lambda x, n: x[n], [[-(2 ** 63), 0]], "outside"),
                 # i + extent wraps for this positive index. The unselected sum cannot fail.
                 "wrapping sum": (lambda x, n: x[n], [[2 ** 63 - 1, 0]], "outside"),
                 "gather": (lambda x, n: torch.gather(x, 1, n), [[4, 0]], "outside"),
                 "embedding": (lambda x, n: functional.embedding(n, x), [[-1, 2]], "outside")}
        for name, (function, bad, text) in cases.items():
            with self.subTest(name=name):
                artifact = torchlean._capture(function, (x, torch.tensor([[1, 0]])), "pytorch",
                                              EXECUTE)
                argument = dict(artifact["arguments"][1])
                argument["bits"] = [value % 2 ** 64 for row in bad for value in row]
                artifact["arguments"][1] = argument
                artifact.pop("reference_outputs", None)
                self.assert_unsupported_execution(
                    torchlean.verify(artifact, checks=EXECUTE, checker=CHECKER), text)
        edges = torch.tensor([[-3, -1], [0, 2]])
        for index in (edges, edges.int()):
            with self.subTest(dtype=index.dtype):
                self.assert_passes(self.verify(lambda x, n: (x[n], x[:, n]), x, index))

    def test_integer_comparisons_promote_as_result_type(self):
        # Comparisons take one dtype: a uint8 tensor against an int8 tensor is refused, the same
        # values cast to a common type compare, and a Python float promotes an integer tensor
        # to the default floating dtype at capture.
        torch = self.torch
        small = torch.tensor([0, 1, 200, 255], dtype=torch.uint8)
        signed = torch.tensor([-1, 1, 100, -128], dtype=torch.int8)
        self.assert_refused(self.verify(lambda a, b: a < b, small, signed), "two dtypes")
        self.assert_passes(self.verify(lambda a, b: (a.long() < b.long(), torch.where(
            a.long() > 1, a.long(), b.long())), small, signed))
        values = torch.tensor([0, 1, 2 ** 24 + 1, -3])
        default = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            self.assert_passes(self.verify(lambda n: n > 16777216.5, values))
        finally:
            torch.set_default_dtype(default)

    def test_topk_pool_indices_and_scatter_folds_match_eager(self):
        # topk is the IR's stable topk, and execution fails when two of the first k + 1 sorted
        # entries tie. Max pooling indices are a window's first maximum or last NaN, as
        # cpu_max_pool takes them. scatter_reduce with prod, amax and amin is one scatter with
        # the rule prod, max or min, folding in index order as scatter_reduce_two does, NaN and
        # signed zeros included where ATen's tie rule agrees with IEEE maximum and minimum (a
        # +0 met first for amax). Between -0 and +0 met in the other order the IR gives IEEE's
        # +0 (-0 for amin) and ATen the first one.
        torch = self.torch
        functional = torch.nn.functional
        nan, inf = float("nan"), float("inf")
        generator = torch.Generator().manual_seed(6)
        x = torch.randn(3, 7, generator=generator)
        x[0, 2], x[1, 4] = nan, -0.0

        def tops(x):
            return (torch.topk(x, 3)[0], torch.topk(x, 3)[1], torch.topk(x, 2, 0, False)[0],
                    torch.topk(x, 2, dim=0, largest=False).indices,
                    torch.topk(x, 1, sorted=False).indices, torch.topk(x, 0).values,
                    torch.topk(x, 7, dim=1).indices)

        plane = torch.tensor([[1.0, 5.0, 5.0, 0.0, 2.0], [nan, 2.0, 5.0, nan, 1.0],
                              [nan, 0.0, -inf, -inf, -0.0], [3.0, nan, -inf, -inf, 0.0]])
        image = torch.stack([plane, -plane]).reshape(1, 2, 4, 5)
        volume = torch.randn(1, 2, 3, 4, 4, generator=generator)
        volume[0, 0, 1, 1, 1] = nan

        def pools(image, volume):
            return (functional.max_pool2d(image, 2, return_indices=True)[1],
                    functional.max_pool2d(image, 3, stride=2, padding=1, return_indices=True)[1],
                    functional.max_pool2d(image[0], (2, 3), stride=(1, 2),
                                          return_indices=True)[1],
                    functional.max_pool3d(volume, 2, stride=1, padding=1, return_indices=True)[1],
                    functional.max_pool2d(image[:, :, 2:, 2:4], 2, stride=1, padding=1,
                                          return_indices=True)[1])

        base = torch.tensor([[0.0, -0.0, 1.0, 2.0], [nan, 3.0, -1.0, 0.5]])
        source = torch.tensor([[0.0, 0.0, nan, 2.0], [1.0, -2.0, 4.0, -0.0],
                               [5.0, nan, 0.5, 2.5]])
        index = torch.tensor([[0, 0, 1, 1], [1, 3, 3, 0], [2, 2, 0, 0]])

        def folds(base, index, source):
            return tuple(base.scatter_reduce(1, index[:2], source, reduce, include_self=include)
                         for reduce in ("prod", "amax", "amin") for include in (True, False)) + (
                base.scatter_reduce(0, index[:, :2] % 2, source[:, :3], "amin"),)

        ties = torch.tensor([[3.0, 1.0, 3.0, 0.0], [2.0, -0.0, 0.0, 1.0]])
        nans = torch.tensor([[1.0, nan, 0.0, nan], [4.0, 2.0, 3.0, 1.0]])
        wide = torch.randn(2, 200, generator=generator)

        def first(x):
            return torch.topk(x, 1).indices

        def three(x):
            return torch.topk(x, 3).values

        def smallest(x):
            return torch.topk(x, 2, largest=False).indices

        def expanded(base, index, source, reduce="prod"):
            return base.scatter_reduce(0, index[0, :2].unsqueeze(1).expand(2, 4), source[:2],
                                       reduce)

        for core in (False, True):
            with self.subTest(core=core):
                self.assert_passes(self.verify(tops, x, core=core))
                self.assert_passes(self.verify(pools, image, volume, core=core))
                self.assert_passes(self.verify(lambda x: torch.topk(x, 2), ties[1:], core=core))
                self.assert_passes(self.verify(lambda x: torch.topk(x, 2), wide, core=core))
                self.assert_refused(self.verify(expanded, base, index, source, core=core),
                                    "expanded index")
                # amax along an expanded index: the IEEE maximum, compared without signed zeros.
                self.assert_passes(self.verify(lambda b, i, s: expanded(b, i, s, "amax"),
                                               base + 3, index, source + 3, core=core))
                for dtype in (torch.float32, torch.bfloat16, torch.float64):
                    self.assert_passes(self.verify(folds, base.to(dtype), index, source.to(dtype),
                                                   core=core))
                for function, operand in ((first, ties), (three, ties), (first, nans),
                                          (smallest, ties[1:])):
                    self.assert_unsupported_execution(
                        self.verify(function, operand, core=core), "outside")
        self.assert_passes(self.verify(tops, x.double()))
        self.assert_passes(self.verify(pools, image.bfloat16(), volume.half()))
        # The IR's amax of a -0 base and a +0 update is +0 and its amin of +0 and -0 is -0. ATen
        # returns the base.
        def zeros(base, index, source):
            return (base.scatter_reduce(0, index, source, "amax"),
                    (-base).scatter_reduce(0, index, -source, "amin"))

        # One registered scatter per fold, with the rules prod (2), max (3) and min (4).
        codes = sorted({record["fields"][2]["nat"] for record in
                        _registered(self.artifact(folds, base, index, source))
                        if record["tag"] == "scatter"})
        self.assertEqual([code for code in codes if code > 1], [2, 3, 4])
        # Without include_self a maximum takes two scatters (the first updates, then the rest).
        def excluded(base, index, source):
            return base.scatter_reduce(1, index[:2], source[:2], "amax", include_self=False)

        scatters = [record for record in _registered(self.artifact(excluded, base, index, source))
                    if record["tag"] == "scatter"]
        self.assertEqual([record["fields"][2]["nat"] for record in scatters], [0, 3])
        # An empty index leaves the base, and an index equal to the extent fails, as ATen's does.
        empty = torch.zeros(0, dtype=torch.long)
        self.assert_passes(self.verify(lambda b, i, s: b.scatter_reduce(
            0, i, s, "amax", include_self=False), torch.tensor([5.0]), empty, torch.zeros(0)))
        artifact = torchlean._capture(lambda b, i, s: b.scatter_reduce(
            0, i, s, "amax", include_self=False), (torch.tensor([5.0]), torch.tensor([0]),
            torch.tensor([7.0])), "pytorch", EXECUTE)
        argument = dict(artifact["arguments"][1])
        argument["bits"] = [1]
        artifact["arguments"][1] = argument
        artifact.pop("reference_outputs", None)
        self.assert_unsupported_execution(
            torchlean.verify(artifact, checks=EXECUTE, checker=CHECKER), "outside")
        report = self.verify(zeros, torch.tensor([-0.0]), torch.tensor([0]), torch.tensor([0.0]),
                             checks=EXECUTE)
        self.assert_passes(report)
        self.assertEqual([output["bits"] for output in report["outputs"]],
                         [[0], [1 << 31]])

    def test_reflected_division_in_the_core_graph(self):
        # run_decompositions() keeps Tensor.__rtruediv__'s reciprocal and product with their
        # torch_fn tags, so the core graph's s / x is one division too.
        from fractions import Fraction

        torch = self.torch
        x = torch.randn(16, generator=torch.Generator().manual_seed(2)) * 3
        for core in (False, True):
            with self.subTest(core=core):
                kinds = _tags(self.artifact(lambda v: 0.1 / v, x, core=core))
                self.assertEqual((kinds.count("div"), kinds.count("reciprocal")), (1, 0))
                report = self.verify(lambda v: 0.1 / v, x, checks=EXECUTE, core=core)
                self.assert_passes(report)
                _same(self, _outputs(report)[0].tolist(),
                      [_nearest(Fraction(_nearest(0.1, "float32")) / Fraction(v), "float32")
                       for v in x.tolist()])

    def test_scatter_folds_on_generated_indices(self):
        # scatter_reduce prod, amax and amin along an interior dim of a rank-3 tensor and along
        # negative dims, on random colliding indices, against an exact Python fold: IEEE maximum
        # and minimum (-0 below +0, a NaN wins), products of small dyadic values (exact), the base
        # entry a participant only with include_self. With include_self the capture writes one
        # registered scatter with the rule, and without it a replacing scatter (code 0) first. An
        # index with an empty extent leaves the base, and an index at the extent fails execution.
        import functools
        import itertools
        import math

        torch = self.torch
        generator = torch.Generator().manual_seed(12)
        nan = math.nan
        choices = torch.tensor([-2.0, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 2.0, nan])
        base = choices[torch.randint(0, 9, (3, 4, 5), generator=generator)]
        codes = {"prod": 2, "amax": 3, "amin": 4}

        def fold(values, reduce):
            if reduce == "prod":
                return math.prod(values)
            return functools.reduce(_ieee_maximum if reduce == "amax" else _ieee_minimum, values)

        def reference(base, dim, index, source, reduce, include):
            out = base.tolist()
            rank = base.dim()
            dim = dim % rank
            landed = {}
            for position in itertools.product(*(range(n) for n in index.shape)):
                target = list(position)
                target[dim] = int(index[position])
                landed.setdefault(tuple(target), []).append(float(source[position]))
            for target, updates in landed.items():
                cell = out
                for axis in target[:-1]:
                    cell = cell[axis]
                own = [cell[target[-1]]] if include else []
                cell[target[-1]] = fold(own + updates, reduce)
            return [v for plane in out for row in plane for v in row]

        for dim, shape in ((1, (3, 6, 5)), (-1, (3, 4, 7)), (-2, (2, 6, 3)), (0, (4, 4, 5))):
            extent = base.shape[dim]
            index = torch.randint(0, extent, shape, generator=generator)
            source = choices[torch.randint(0, 9, shape, generator=generator)]
            folds = list(itertools.product(codes, (True, False)))

            def scattered(b, i, s):
                return tuple(b.scatter_reduce(dim, i, s, reduce, include_self=include)
                             for reduce, include in folds)

            with self.subTest(dim=dim):
                report = self.verify(scattered, base, index, source, checks=EXECUTE)
                self.assert_passes(report)
                self.assertEqual(len(_outputs(report)), len(folds))
                for got, (reduce, include) in zip(_outputs(report), folds):
                    _same(self, got.reshape(-1).tolist(),
                          reference(base, dim, index, source, reduce, include), (reduce, include))
                rules = [record["fields"][2]["nat"]
                         for record in _registered(self.artifact(scattered, base, index, source))
                         if record["tag"] == "scatter"]
                self.assertEqual(rules, [code for reduce, include in folds
                                         for code in ([codes[reduce]] if include
                                                      else [0, codes[reduce]])])
        for shape in ((3, 0, 5), (0, 4, 5), (3, 4, 0)):
            with self.subTest(empty=shape):
                empty = torch.zeros(shape, dtype=torch.long)
                report = self.verify(lambda b, i, s: b.scatter_reduce(
                    1, i, s, "amax", include_self=False), base, empty, torch.zeros(shape),
                    checks=EXECUTE)
                self.assert_passes(report)
                _same(self, _outputs(report)[0].reshape(-1).tolist(), base.reshape(-1).tolist())
                self.assertEqual(_registered(self.artifact(
                    lambda b, i, s: b.scatter_reduce(1, i, s, "amax", include_self=False),
                    base, empty, torch.zeros(shape))), [])
        index = torch.zeros(3, 2, 5, dtype=torch.long)
        for reduce, include in (("amax", False), ("amax", True), ("prod", False)):
            with self.subTest(extent=True, reduce=reduce, include=include):
                artifact = torchlean._capture(
                    lambda b, i, s: b.scatter_reduce(1, i, s, reduce, include_self=include),
                    (base, index, torch.ones(3, 2, 5)), "pytorch", EXECUTE)
                argument = dict(artifact["arguments"][1])
                argument["bits"] = [4] + [0] * 29
                artifact["arguments"][1] = argument
                artifact.pop("reference_outputs", None)
                self.assert_unsupported_execution(
                    torchlean.verify(artifact, checks=EXECUTE, checker=CHECKER), "outside")

    def test_reduced_scatter_sums_and_topk_fibers_match_eager(self):
        # scatter_add and scatter_reduce sums and products of binary16 and bfloat16 accumulate
        # in binary32 and round once, and a mean divides the rounded sum by the rounded count,
        # as the CPU kernels do: in binary16 2048 + 1 + 1 is 2050, where rounding each step
        # gives 2048, and 65504 * 2 * 0.5 stays finite.
        # index_put with accumulate rounds each step, as its kernel does. An earlier reader of an
        # index_put_ destination may be its values. The topk tie check covers every fiber, a
        # fiber of one entry has no tie, and max pooling indices of an empty batch are refused.
        torch = self.torch
        generator = torch.Generator().manual_seed(8)
        base = torch.randn(3, 4, generator=generator) * 100
        source = torch.randn(3, 4, generator=generator) * 100
        index = torch.randint(0, 2, (3, 4), generator=generator)
        small = torch.tensor([2048.0, 1.0, 1.0, 65504.0, 2.0, 0.5])
        at = torch.tensor([0, 0, 0, 1, 1, 1])

        def sums(base, index, source, small, at):
            zero = torch.zeros(2, dtype=small.dtype)
            return tuple(base.scatter_reduce(dim, index, source, reduce, include_self=include)
                         for dim in (0, 1) for reduce in ("sum", "mean", "prod")
                         for include in (True, False)) + (
                base.scatter_add(0, index, source), base.scatter_add(1, index, source),
                zero.scatter_add(0, at, small),
                zero.scatter_reduce(0, at, small, "prod", include_self=False),
                zero.scatter_reduce(0, at, small, "mean", include_self=False),
                torch.index_put(zero, (at,), small, accumulate=True))

        def snapshot(x):
            y = torch.zeros(4)
            values = y + x
            y[torch.sort(x, stable=True).indices] = values
            return y

        def pool_indices(x):
            return torch.nn.functional.max_pool2d(x, 2, return_indices=True)[1]

        later = torch.tensor([[3.0, 1.0, 2.0], [2.0, 0.0, 2.0]])
        single = torch.tensor([[float("nan")], [1.0]])
        x = torch.randn(4, generator=generator)
        for core in (False, True):
            with self.subTest(core=core):
                for dtype in (torch.float16, torch.bfloat16, torch.float32):
                    self.assert_passes(self.verify(sums, base.to(dtype), index, source.to(dtype),
                                                   small.to(dtype), at, core=core))
                self.assert_passes(self.verify(snapshot, x, core=core))
                self.assert_passes(self.verify(lambda x: torch.topk(x, 1), single, core=core))
                self.assert_unsupported_execution(
                    self.verify(lambda x: torch.topk(x, 1).indices, later, core=core), "outside")
                self.assert_refused(self.verify(pool_indices, torch.zeros(0, 2, 4, 4), core=core),
                                    "nonempty planes")

    def test_boolean_reductions_and_constant_scatters_match_eager(self):
        # any and all are a sum of 0/1 terms compared with zero. An empty fiber gives false and
        # true, and dim=[] reduces nothing. slice_scatter and select_scatter are constant-index
        # scatters, and fold (col2im) is a scatter into a zero canvas in col2im's loop order.
        # scatter_reduce with sum and mean and integer remainder match eager PyTorch too.
        # Constant integer and Boolean values such as a causal mask fold at capture.
        torch = self.torch
        functional = torch.nn.functional
        generator = torch.Generator().manual_seed(4)
        x = torch.randn(3, 4, generator=generator)
        x[0, 0] = float("nan")
        b = torch.tensor([[True, False, False], [False, False, False]])

        def booleans(b, x, e):
            return (torch.any(b, 1), torch.all(b, 0, keepdim=True), b.any(), torch.all(x > -5),
                    torch.any(x, dim=(0, 1)), torch.any(e, 1), torch.all(e, 1),
                    torch.ops.aten.any.dims(b, []), torch.all(x, 1))

        def scatters(x, i, s):
            return (torch.slice_scatter(x, x[:, ::2] * 2, 1, 0, None, 2),
                    torch.select_scatter(x, x[0] * 3, 0, -1),
                    x.scatter_reduce(0, i, s, 'mean', include_self=False),
                    x.scatter_reduce(0, i, s, 'sum'), x.scatter_reduce(0, i, s, 'mean'),
                    x.scatter_reduce(0, i, s, 'sum', include_self=False))

        def folds(c, d):
            return (functional.fold(c, (5, 6), (2, 3), padding=1, stride=2),
                    functional.fold(d, (4, 4), (1, 1), stride=2))

        def remainders(n):
            return torch.remainder(n, 3), torch.remainder(n, -2), n % torch.tensor([2, -3, 4])

        index = torch.tensor([[0, 1, 2, 0], [0, 0, 1, 1]])
        for core in (False, True):
            with self.subTest(core=core):
                self.assert_passes(self.verify(booleans, b, x, torch.zeros(2, 0, dtype=torch.bool),
                                               core=core))
                self.assert_passes(self.verify(scatters, x, index,
                                               torch.randn(2, 4, generator=generator), core=core))
                self.assert_passes(self.verify(folds, torch.randn(2, 12, 9, generator=generator),
                                               torch.randn(1, 4, generator=generator), core=core))
                self.assert_passes(self.verify(remainders, torch.tensor([[-7, 7, 0], [-1, 5, -9]]),
                                               core=core))
                # A conversion to uint8 truncates toward zero. A value uint8 cannot hold is an
                # execution error (c10::convert wraps it through int64).
                inside = torch.tensor([0.0, 0.7, 1.5, 254.9, 255.0, -0.5])
                report = self.verify(lambda x: x.to(torch.uint8), inside, checks=EXECUTE,
                                     core=core)
                self.assert_passes(report)
                self.assertEqual(report["outputs"][0]["bits"], [0, 0, 1, 254, 255, 0])
                self.assert_unsupported_execution(self.verify(
                    lambda x: x.to(torch.uint8), torch.tensor([-1.5, 256.0]), checks=EXECUTE,
                    core=core), "does not fit")

    def test_token_model_matches_eager(self):
        # A token model: token and position embeddings (positions from a folded arange), a
        # layer norm, a causal attention whose mask folds to a Boolean constant, and a head.
        torch = self.torch
        nn = torch.nn
        functional = nn.functional
        torch.manual_seed(5)

        class Tokens(nn.Module):
            def __init__(self, vocab=11, width=8, length=6):
                super().__init__()
                self.token, self.position = nn.Embedding(vocab, width), nn.Embedding(length, width)
                self.ln, self.qkv = nn.LayerNorm(width), nn.Linear(width, 3 * width)
                self.head = nn.Linear(width, vocab)

            def forward(self, tokens):
                t = tokens.shape[1]
                h = self.token(tokens) + self.position(torch.arange(t))
                q, k, v = self.qkv(self.ln(h)).split(h.shape[-1], dim=-1)
                i = torch.arange(t)
                scores = (q @ k.transpose(-2, -1)) / h.shape[-1] ** 0.5
                scores = torch.where(i[None, :] <= i[:, None], scores, float("-inf"))
                return self.head(h + functional.softmax(scores, dim=-1) @ v)

        model = Tokens().eval()
        tokens = torch.tensor([[1, 4, 0, 10, 3, 3], [2, 2, 9, 0, 5, 7]])
        for core in (False, True):
            with self.subTest(core=core):
                report = self.verify(model, tokens, checks=EXECUTE, core=core)
                self.assert_passes(report)
                with torch.no_grad():
                    torch.testing.assert_close(_outputs(report)[0], model(tokens), rtol=1e-5,
                                               atol=1e-5)

    def test_adaptive_pools_and_nearest_upsampling_match_eager(self):
        # Adaptive average pools are the CPU kernel's windows, summed in row-major order in the
        # opmath type and divided by each window extent. A global pool of the composite is the
        # mean. Nearest upsampling gathers the source indices PyTorch's nearest_idx computes.
        torch = self.torch
        functional = torch.nn.functional
        generator = torch.Generator().manual_seed(6)
        x = torch.randn(2, 3, 7, 9, generator=generator) * 3
        cube = torch.randn(1, 2, 5, 4, 6, generator=generator)

        def pools(x, cube):
            return (functional.adaptive_avg_pool2d(x, (3, 4)), functional.adaptive_avg_pool2d(
                x[:, :, :6, :6], (3, 2)), functional.adaptive_avg_pool1d(x[0], 4),
                functional.adaptive_avg_pool3d(cube, (2, 3, 4)),
                functional.adaptive_avg_pool2d(x, (None, 5)))

        def resized(x):
            return (functional.interpolate(x, scale_factor=2, mode="nearest"),
                    functional.interpolate(x, size=(4, 13), mode="nearest"),
                    functional.interpolate(x[0], scale_factor=1.5, mode="nearest"))

        for dtype in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
            for core in (False, True):
                with self.subTest(dtype=dtype, core=core):
                    self.assert_passes(self.verify(pools, x.to(dtype), cube.to(dtype), core=core))
                    self.assert_passes(self.verify(resized, x.to(dtype), core=core))
        for core in (False, True):
            with self.subTest(core=core):
                report = self.verify(lambda x: functional.adaptive_avg_pool2d(x, 1), x,
                                     checks=EXECUTE, core=core)
                self.assert_passes(report)
                torch.testing.assert_close(_outputs(report)[0],
                                           functional.adaptive_avg_pool2d(x, 1))

    def test_real_fourier_transforms_within_tolerance(self):
        # rfft and irfft, through _fft_r2c and _fft_c2r in the core graph, are Op.rfft and
        # Op.irfft on the paired-real form, scaled for the normalization. TorchLean's DFT is
        # not pocketfft's schedule, so the values agree within a tolerance.
        torch = self.torch
        generator = torch.Generator().manual_seed(7)
        x = torch.randn(4, 6, generator=generator)
        y = torch.randn(4, 4, 2, generator=generator)

        def transforms(x, y):
            return (torch.view_as_real(torch.fft.rfft(x)),
                    torch.view_as_real(torch.fft.rfft(x, n=10, dim=0, norm="ortho")),
                    torch.view_as_real(torch.fft.rfft(x, n=4, norm="forward")),
                    torch.fft.irfft(torch.view_as_complex(y), n=6),
                    torch.fft.irfft(torch.view_as_complex(y), dim=0, norm="ortho"),
                    torch.fft.irfft(torch.view_as_complex(y), n=5, norm="forward"))

        for dtype in (torch.float32, torch.float64):
            for core in (False, True):
                with self.subTest(dtype=dtype, core=core):
                    report = self.verify(transforms, x.to(dtype), y.to(dtype), checks=EXECUTE,
                                         core=core)
                    self.assert_passes(report)
                    for got, want in zip(_outputs(report), transforms(x.to(dtype), y.to(dtype))):
                        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
        report = self.verify(lambda x: torch.fft.rfft(x).abs(), x, checks=("structure",))
        self.assert_refused(report, "paired-real form")

    def test_group_norm_and_training_batch_norm_within_tolerance(self):
        # group_norm and training-mode batch norm without running statistics: the IR's
        # two-pass moments and PyTorch's affine formula, within a tolerance of the CPU kernels'
        # vectorized moments. The means and reciprocal standard deviations are outputs too.
        torch = self.torch
        nn = torch.nn
        functional = nn.functional
        torch.manual_seed(8)
        x = torch.randn(2, 6, 3, 4) * 2 + 1
        norm = nn.GroupNorm(3, 6).eval()
        with torch.no_grad():
            norm.weight.copy_(torch.randn(6))
            norm.bias.copy_(torch.randn(6))
        batch = nn.BatchNorm2d(6, track_running_stats=False).eval()

        def statistics(x):
            out, mean, rstd = torch.native_group_norm(x, None, None, 2, 6, 12, 2, 1e-5)
            values, means, inverse = torch.ops.aten._native_batch_norm_legit.no_stats(
                x, None, None, True, 0.1, 1e-5)
            return out, mean, rstd, values, means, inverse

        cases = {"group norm": (norm, (x,)), "batch norm": (batch, (x,)),
                 "functional": (lambda x: functional.group_norm(x, 2), (x,)),
                 "statistics": (statistics, (x,))}
        for name, (model, inputs) in cases.items():
            for dtype in (torch.float32, torch.float64):
                for core in (False, True):
                    with self.subTest(name=name, dtype=dtype, core=core):
                        if isinstance(model, nn.Module): model.to(dtype)
                        arguments = tuple(value.to(dtype) for value in inputs)
                        report = self.verify(model, *arguments, checks=EXECUTE, core=core)
                        self.assert_passes(report)
                        with torch.no_grad():
                            expected = model(*arguments)
                        expected = expected if isinstance(expected, tuple) else (expected,)
                        for got, want in zip(_outputs(report), expected):
                            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)

    def test_training_batch_norm_updates_running_statistics(self):
        # Training-mode batch norm with running statistics: the program returns the normalized
        # values, then the updated running_mean, running_var and num_batches_tracked in buffer
        # order, with no mutation in the IR. The moments agree within a tolerance and the counter
        # exactly. The default and core graphs import one fixture to the same words.
        torch = self.torch
        nn = torch.nn
        functional = nn.functional
        torch.manual_seed(9)
        cases = {"1d": (nn.BatchNorm1d(4, momentum=0.3), (5, 4)),
                 "1d two values per channel": (nn.BatchNorm1d(3), (2, 3)),
                 "1d sequence": (nn.BatchNorm1d(4), (3, 4, 6)),
                 "2d without affine": (nn.BatchNorm2d(3, affine=False), (2, 3, 4, 5)),
                 "3d": (nn.BatchNorm3d(2, momentum=0.05), (2, 2, 3, 2, 2))}
        for name, (template, size) in cases.items():
            for dtype in (torch.float32, torch.float64):
                model = copy.deepcopy(template).to(dtype).train()
                channels = model.num_features
                with torch.no_grad():
                    model.running_mean.copy_(torch.randn(channels))
                    model.running_var.copy_(torch.rand(channels) + 0.5)
                    model.num_batches_tracked.fill_(7)
                    if model.affine:
                        model.weight.copy_(torch.randn(channels))
                        model.bias.copy_(torch.randn(channels))
                x = (torch.randn(*size) * 2 + 1).to(dtype)
                kept = {key: value.clone() for key, value in model.state_dict().items()}
                reference = copy.deepcopy(model)
                with torch.no_grad():
                    expected = (reference(x), reference.running_mean, reference.running_var)
                words = {}
                for core in (False, True):
                    with self.subTest(name=name, dtype=dtype, core=core):
                        report = self.verify(model, x, checks=EXECUTE, core=core)
                        self.assert_passes(report)
                        outputs = report["outputs"]
                        self.assertEqual(len(outputs), 4)
                        for got, want in zip(_outputs({"outputs": outputs[:3]}), expected):
                            self.assertEqual(got.dtype, dtype)
                            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
                        self.assertEqual((outputs[3]["element_type"], outputs[3]["shape"],
                                          outputs[3]["bits"]), ("int64", [], [8]))
                        for key, value in model.state_dict().items():
                            self.assertTrue(torch.equal(value, kept[key]), key)
                        words[core] = [output["bits"] for output in outputs]
                with self.subTest(name=name, dtype=dtype, words="default = core"):
                    self.assertEqual(sorted(words), [False, True])
                    self.assertEqual(words[False], words[True])
        # Running statistics that are inputs (their update mutates a user input) and half
        # precision are refused.
        inputs = (torch.randn(4, 3), torch.zeros(3), torch.ones(3))
        refused = {
            "statistics as inputs": (
                lambda x, m, v: functional.batch_norm(x, m, v, training=True), inputs,
                ("must be two buffers", "other than buffer updates")),
            "binary16": (nn.BatchNorm1d(3).half().train(), (torch.randn(4, 3).half(),),
                         ("binary16 or bfloat16",) * 2)}
        for name, (model, arguments, texts) in refused.items():
            for core, text in zip((False, True), texts):
                with self.subTest(case=name, core=core):
                    report = self.verify(model, *arguments, checks=("structure",), core=core)
                    self.assert_refused(report, text)

        class Two(nn.Module):
            # Two modules whose execution order reverses their registration order.
            def __init__(self):
                super().__init__()
                self.first, self.second = nn.BatchNorm1d(3), nn.BatchNorm1d(3, momentum=0.5)

            def forward(self, x):
                return self.first(self.second(x) * 2)

        two = Two().train()
        with torch.no_grad():
            for buffer in two.buffers():
                buffer.copy_(torch.randint(1, 5, buffer.shape))
        x = torch.randn(6, 3)
        reference = copy.deepcopy(two)
        with torch.no_grad():
            expected = [reference(x)] + list(reference.buffers())
        for core in (False, True):
            with self.subTest(case="two modules", core=core):
                report = self.verify(two, x, checks=EXECUTE, core=core)
                self.assert_passes(report)
                outputs = report["outputs"]
                self.assertEqual(len(outputs), 7)
                floating = [0, 1, 2, 4, 5]
                for index, got in zip(floating, _outputs(
                        {"outputs": [outputs[index] for index in floating]})):
                    torch.testing.assert_close(got, expected[index], rtol=1e-5, atol=1e-5)
                self.assertEqual([outputs[3]["bits"], outputs[6]["bits"]],
                                 [[int(expected[3]) % 2 ** 64], [int(expected[6]) % 2 ** 64]])

        def bits(value):
            width = value.element_size() * 8
            integer = {16: torch.int16, 32: torch.int32, 64: torch.int64}[width]
            return [word % 2 ** width for word in value.reshape(-1).view(integer).tolist()]

        # The comparison reference holds the updated buffers after the outputs, read from the
        # capture's copy of the model after the call. The caller's buffers are kept.
        norm = nn.BatchNorm1d(4, momentum=0.2).train()
        x = torch.randn(5, 4)
        reference = copy.deepcopy(norm)
        with torch.no_grad():
            expected = [reference(x), reference.running_mean, reference.running_var,
                        reference.num_batches_tracked]
        artifact = torchlean._capture(norm, (x,), "pytorch", COMPARE)
        self.assertEqual([output["bits"] for output in artifact["reference_outputs"]],
                         [bits(value) for value in expected])
        self.assertEqual((int(norm.num_batches_tracked), norm.running_mean.tolist(),
                          norm.running_var.tolist()), (0, [0.0] * 4, [1.0] * 4))

        class Counter(nn.Module):
            # An in-place increment whose result an integer operator reads: constant folding
            # never runs add_ on the captured buffer, so the buffer's update is 8.
            def __init__(self, step=1):
                super().__init__()
                self.step = step
                self.register_buffer("count", torch.tensor(7))

            def forward(self, x):
                return x * 2, self.count.add_(self.step) * 2

        counter = Counter()
        for core in (False, True):
            with self.subTest(case="increment read by an integer consumer", core=core):
                report = self.verify(counter, torch.ones(2), checks=EXECUTE, core=core)
                self.assert_passes(report)
                self.assertEqual([output["bits"] for output in report["outputs"][1:]],
                                 [[16], [8]])
                self.assertEqual(int(counter.count), 7)

        class Increment(nn.Module):
            # An unused in-place increment other than num_batches_tracked's.
            def __init__(self, step, dtype):
                super().__init__()
                self.step = step
                self.register_buffer("count", torch.zeros((), dtype=dtype))

            def forward(self, x):
                self.count.add_(self.step)
                return x * 2

        class Shared(nn.Module):
            # One buffer as both running statistics.
            def __init__(self):
                super().__init__()
                self.register_buffer("statistic", torch.ones(3))

            def forward(self, x):
                return functional.batch_norm(x, self.statistic, self.statistic, training=True)

        class Views(nn.Module):
            # Two updated buffers that are disjoint views of one storage.
            def __init__(self):
                super().__init__()
                base = torch.zeros(6)
                self.register_buffer("mean", base[:3])
                self.register_buffer("var", base[3:])

            def forward(self, x):
                return functional.batch_norm(x, self.mean, self.var, training=True)

        class Reader(nn.Module):
            # A buffer read after the update that shares the updated running_mean's storage.
            def __init__(self):
                super().__init__()
                mean = torch.zeros(3)
                self.register_buffer("mean", mean)
                self.register_buffer("var", torch.ones(3))
                self.register_buffer("saved", mean.view(3))

            def forward(self, x):
                return functional.batch_norm(x, self.mean, self.var, training=True) + self.saved

        class Alias(nn.Module):
            # An input that is the module's running_mean: the exported batch_norm reads the input.
            def __init__(self):
                super().__init__()
                self.norm = nn.BatchNorm1d(3)

            def forward(self, x, mean):
                return self.norm(x) + mean

        alias = Alias().train()
        x = torch.randn(4, 3)
        # The default graph passes the one buffer placeholder as both statistics.
        graph = torch.export.export(Shared(), (x,)).graph
        (node,) = [node for node in graph.nodes if node.op == "call_function"]
        self.assertIs(node.args[3], node.args[4])
        refused = {"increment by 2": (Increment(2, torch.int64), (x,), ("aten.add_.Tensor",)),
                   "binary32 increment": (Increment(1, torch.float32), (x,),
                                          ("aten.add_.Tensor",)),
                   "shared statistics": (Shared(), (x,),
                                         ("must be two buffers", "two distinct buffers")),
                   "statistics sharing storage": (Views(), (x,), ("shares storage",) * 2),
                   "a reader sharing storage": (Reader(), (x,), ("shares storage",)),
                   "statistic passed as input": (alias, (x, alias.norm.running_mean),
                                                 ("must be two buffers",))}
        for name, (model, arguments, texts) in refused.items():
            for core, text in zip((False, True), texts):
                with self.subTest(case=name, core=core):
                    report = self.verify(model, *arguments, checks=("structure",), core=core)
                    self.assert_refused(report, text)

    def test_distances_interpolation_and_grid_sampling_within_tolerance(self):
        # cdist (direct, and the matmul expansion beyond 25 rows), pdist, linear and bilinear
        # upsampling with PyTorch's per-axis indices and weights, and grid_sampler_2d through a
        # runtime gather at the floored corners: within a tolerance of the CPU kernels.
        import math
        from fractions import Fraction

        torch = self.torch
        functional = torch.nn.functional
        generator = torch.Generator().manual_seed(9)
        a, b = torch.randn(2, 5, 3, generator=generator), torch.randn(2, 4, 3, generator=generator)
        many = torch.randn(30, 3, generator=generator)
        image = torch.randn(2, 3, 5, 6, generator=generator)
        grid = torch.rand(2, 4, 7, 2, generator=generator) * 2.4 - 1.2

        def distances(a, b, many):
            return (torch.cdist(a, b), torch.cdist(a, b, p=1.0), torch.cdist(a, b, p=0.0),
                    torch.cdist(a, b, p=float("inf")), torch.cdist(a, b, p=3.0),
                    torch.cdist(many[:25], many[:7]), functional.pdist(a[0]),
                    functional.pdist(a[1], p=1.5))

        def expanded(many, image):
            # Beyond 25 rows cdist takes the matmul expansion. Its core form takes a raw sqrt of a
            # clamp with the lower bound 0, which is max_elem, never -0, so the sqrt imports. The
            # core form of upsample_linear1d clamps with one bound, max_elem or min_elem.
            return (torch.cdist(many, many[:7]), torch.cdist(many[:26], many[:3]),
                    functional.interpolate(image[0], size=9, mode="linear"),
                    torch.ops.aten.upsample_linear1d.default(image[0], [4], False, 2.0))

        def resized(image):
            return (functional.interpolate(image, scale_factor=2, mode="bilinear"),
                    functional.interpolate(image, size=(7, 4), mode="bilinear",
                                           align_corners=True))

        def sampled(image, grid):
            return (functional.grid_sample(image, grid, align_corners=False),
                    functional.grid_sample(image, grid, align_corners=True,
                                           padding_mode="border"),
                    functional.grid_sample(image, grid, mode="nearest", align_corners=False))

        cases = {"distances": (distances, (a, b, many)), "resized": (resized, (image,)),
                 "sampled": (sampled, (image, grid)), "expanded": (expanded, (many, image))}
        # A NaN neighbour reaches the output even at weight zero, and a corner far outside the
        # image contributes zero even where its weight overflows.
        for name, function, inputs in (
                ("nan neighbour", lambda v: functional.interpolate(v, size=4, mode="linear"),
                 (torch.tensor([[[1.0, float("nan")]]]),)),
                ("far grid", lambda x, g: functional.grid_sample(x, g, align_corners=False),
                 (torch.zeros(1, 1, 2, 2), torch.full((1, 1, 1, 2), 3e38)))):
            with self.subTest(name=name):
                report = self.verify(function, *inputs, checks=EXECUTE)
                self.assert_passes(report)
                torch.testing.assert_close(_outputs(report)[0], function(*inputs), equal_nan=True)
        for name, (function, inputs) in cases.items():
            for dtype in (torch.float32, torch.float64):
                for core in (False, True):
                    with self.subTest(name=name, dtype=dtype, core=core):
                        arguments = tuple(value.to(dtype) for value in inputs)
                        report = self.verify(function, *arguments, checks=EXECUTE, core=core)
                        self.assert_passes(report)
                        self.assertEqual(len(_outputs(report)), len(function(*arguments)))
                        for index, (got, want) in enumerate(zip(_outputs(report), function(*arguments))):
                            if name == "expanded" and index < 2:
                                # Self-distances cancel in the squared-distance expansion. Two
                                # d-term norms and the (d+2)-term dot have total absolute terms
                                # 4||x||²; γ_(2d+4) bounds their rounding paths. The clamped root
                                # is at most (1+u) sqrt(4γ||x||²). Compute that bound outward from
                                # exact input rationals; keep the off-diagonal comparison strict.
                                width = arguments[0].shape[-1]
                                u = Fraction(torch.finfo(dtype).eps) / 2
                                gamma = (2 * width + 4) * u / (1 - (2 * width + 4) * u)
                                bounds = []
                                for row in arguments[0][:min(got.shape)].tolist():
                                    energy = sum(Fraction(v) ** 2 for v in row)
                                    squared = 4 * gamma * energy * (1 + u) ** 2
                                    bounds.append(math.nextafter(math.sqrt(math.nextafter(
                                        float(squared), math.inf)), math.inf) if squared else 0.0)
                                bound = torch.tensor(bounds, dtype=torch.float64)
                                for diagonal in (got.diagonal(), want.diagonal()):
                                    self.assertTrue(bool(((diagonal >= 0) & (diagonal <= bound)).all()),
                                                    (diagonal, bound))
                                other = ~torch.eye(*got.shape, dtype=torch.bool)
                                torch.testing.assert_close(got[other], want[other],
                                                           rtol=1e-5, atol=1e-5)
                            else:
                                torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)

    def test_random_operators_use_keyed_streams(self):
        # rand, randperm and training-mode dropout draw from TorchLean's generator under keys
        # recorded in the operator records, and PyTorch's stream plays no part. Entry i of a node
        # with key k is the 32-bit draw splitmix64(k + i) mod 2^32 over 2^32, rounded to the
        # dtype, and the keys of distinct nodes differ by multiples of 2^32. The checks recompute
        # those draws, the permutation of randperm and dropout's mask and output
        # (x * mask) * (1 / (1 - p)).
        torch = self.torch
        functional = torch.nn.functional
        x = torch.randn(4, 5)
        mask64 = (1 << 64) - 1

        def unit(key, count, dtype):
            draws = [_splitmix64((key + i) & mask64) % 2 ** 32 for i in range(count)]
            return torch.tensor(draws, dtype=torch.float64).div(2 ** 32).to(dtype)

        def draws(x):
            kept, mask = torch.ops.aten.native_dropout(x, 0.25, True)
            return (torch.rand(3, 4), torch.randperm(7), kept, mask,
                    functional.dropout(x, 0.5, training=False),
                    torch.ops.aten.native_dropout(x, 1.0, True)[0],
                    torch.rand(2, dtype=torch.float64))

        for core in (False, True):
            with self.subTest(core=core):
                report = self.verify(draws, x, checks=EXECUTE, core=core)
                self.assert_passes(report)
                records = _records(self.artifact(draws, x, core=core))
                keys = [record["fields"][1]["nat"] for record in records
                        if record["tag"] in ("randUniform", "bernoulliMask")]
                self.assertEqual(len({key % 2 ** 32 for key in keys}), 1, keys)
                self.assertEqual(len(set(keys)), len(keys), keys)
                outputs = report["outputs"]
                uniform = _outputs({"outputs": [outputs[0]]})[0]
                first = [record for record in records if record["tag"] == "randUniform"][0]
                self.assertTrue(torch.equal(uniform.reshape(-1),
                                            unit(first["fields"][1]["nat"], 12, torch.float32)))
                order = [record for record in records if record["tag"] == "randUniform"][1]
                draw = unit(order["fields"][1]["nat"], 7, torch.float64).tolist()
                self.assertEqual(outputs[1]["bits"], sorted(range(7), key=lambda i: (draw[i], i)))
                kept = _outputs({"outputs": [outputs[2]]})[0]
                mask = torch.tensor(outputs[3]["bits"]).reshape(4, 5).bool()
                (dropout,) = [record for record in records if record["tag"] == "bernoulliMask"][:1]
                expected = unit(dropout["fields"][1]["nat"], 20, torch.float32) < 0.75
                self.assertTrue(torch.equal(mask.reshape(-1), expected))
                self.assertTrue(torch.equal(kept, (x * mask) * (1 / 0.75)))
                self.assertTrue(torch.equal(_outputs({"outputs": [outputs[4]]})[0], x))
                self.assertTrue(bool((_outputs({"outputs": [outputs[5]]})[0] == 0).all()))
                wide = _outputs({"outputs": [outputs[6]]})[0]
                self.assertTrue(bool(((wide >= 0) & (wide < 1)).all()))
        generator = torch.Generator().manual_seed(0)
        for name, function in (("rand", lambda x: x + torch.rand(4, 5, generator=generator)),
                               ("randn", lambda x: x + torch.randn(4, 5, generator=generator))):
            with self.subTest(generator=name):
                # Some PyTorch versions refuse Generator while capturing, before an ATen node
                # reaches the importer. Both stages must report it as unsupported.
                report = self.verify(function, x, checks=("structure",))
                self.assertFalse(report["ok"])
                status, message = _statuses(report)["structure"]
                self.assertEqual(status, "unsupported")
                self.assertRegex(message, rf"aten\.{name}\.generator|torch\._C\.Generator")

    def test_randn_uses_a_keyed_normal_sampler(self):
        # randn is Op.randn (Random.normal) under a key k = splitmix64(0) + j 2^32 modulo
        # 2^64, j distinct for each sampler: entry i reads m = splitmix64(k + 2 i) and
        # n = splitmix64(k + 2 i + 1) mod 2^32 and is sqrt(-2 log max(u1, e)) cos(2 pi u2) of
        # u1 = m / 2^32 and u2 = n / 2^32, each rounded once to the dtype (to nearest, ties to
        # even), and e = 1e-6 rounded to it, in the dtype's arithmetic. The reference rounds
        # those inputs exactly and evaluates Box-Muller in binary64: a loose check within a
        # tolerance of the dtype, which counter, offset, swapped-input and sine mutations of the
        # reference fail on these draws, and an unclamped reference fails at u1 = 0. Keys chosen
        # through splitmix64's inverse give entries with u1 zero, below and around e, at rounding
        # midpoints and rounding to 1. The draws are not PyTorch's stream, and the test makes no
        # claim of a normal law.
        import math
        from fractions import Fraction

        torch = self.torch
        mask64 = (1 << 64) - 1
        names = {torch.float16: "float16", torch.bfloat16: "bfloat16",
                 torch.float32: "float32", torch.float64: "float64"}

        def unmix(value):
            # The inverse of splitmix64 on 64-bit words.
            def unshift(z, shift):
                result = z
                for _ in range(64 // shift + 1):
                    result = z ^ (result >> shift)
                return result

            z = unshift(value, 31)
            z = unshift((z * pow(0x94d049bb133111eb, -1, 1 << 64)) & mask64, 27)
            z = unshift((z * pow(0xbf58476d1ce4e5b9, -1, 1 << 64)) & mask64, 30)
            return (z - 0x9e3779b97f4a7c15) & mask64

        def reference(key, count, dtype, mutation=None):
            # Pairs of the reference value and its clamped radius, under an optional mutation.
            floor = _nearest(Fraction(1, 10 ** 6), names[dtype])
            step = 1 if mutation == "index" else 2
            shift = 1 if mutation == "offset" else 0
            order = (1, 0) if mutation == "swap" else (0, 1)
            pairs = []
            for i in range(count):
                words = (_splitmix64((key + step * i + offset + shift) & mask64) % 2 ** 32
                         for offset in order)
                u1, u2 = (_nearest(Fraction(word, 2 ** 32), names[dtype]) for word in words)
                radius = math.sqrt(-2 * math.log(max(u1, floor)))
                if mutation == "clamp":
                    radius = math.sqrt(-2 * math.log(u1)) if u1 > 0 else math.inf
                angle = 2 * math.pi * u2
                pairs.append((radius * (math.sin(angle) if mutation == "sine" else
                                        math.cos(angle)),
                              math.sqrt(-2 * math.log(max(u1, floor)))))
            return pairs

        tolerance = {torch.float32: 2 ** -18, torch.float64: 2 ** -40, torch.float16: 2 ** -6,
                     torch.bfloat16: 2 ** -3}

        def close(got, key, dtype, mutation=None):
            # Every entry within the dtype's tolerance scaled by 1 + the radius.
            pairs = reference(key, got.numel(), dtype, mutation)
            return all(abs(value - want) <= tolerance[dtype] * (1 + radius)
                       for value, (want, radius) in zip(got.double().reshape(-1).tolist(), pairs))

        x = torch.zeros(3, 4)

        def draws(x):
            n = torch.randn(3, 4)
            return (x + n, n * 2, torch.rand(6), torch.randn(5, dtype=torch.float64),
                    torch.randn(64, dtype=torch.float16), torch.randn(64, dtype=torch.bfloat16),
                    torch.randn(()), torch.randn(0, 2), torch.randn(256))

        dtypes = [torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.float32,
                  torch.float32, torch.float32]
        base = _splitmix64(0)
        for core in (False, True):
            with self.subTest(core=core):
                report = self.verify(draws, x, checks=EXECUTE, core=core)
                self.assert_passes(report)
                artifact = self.artifact(draws, x, core=core)
                # Each sampler's key is splitmix64(0) + j 2^32 modulo 2^64 for its own j, so the
                # samplers read disjoint blocks of 2^32 counters (two per entry for randn).
                samplers = [(record, scalar) for record, scalar in _ops(artifact)
                            if record["tag"] in ("randn", "randUniform")]
                for record, _ in samplers:
                    self.assertEqual((record["fields"][1]["nat"] - base) % 2 ** 32, 0, record)
                self.assertEqual(len({record["fields"][1]["nat"] for record, _ in samplers}),
                                 len(samplers), samplers)
                normals = [(record, scalar) for record, scalar in samplers
                           if record["tag"] == "randn"]
                self.assertEqual((len(normals), len(samplers)), (7, 8), _steps(artifact))
                self.assertEqual([record["fields"][0]["shape"] for record, _ in normals],
                                 [[3, 4], [5], [64], [64], [], [0, 2], [256]])
                self.assertEqual([scalar for _, scalar in normals],
                                 [names[dtype] for dtype in dtypes])
                outputs = _outputs(report)
                chosen = [outputs[index] for index in (0, 3, 4, 5, 6, 7, 8)]
                self.assertEqual([got.dtype for got in chosen], dtypes)
                keys = [record["fields"][1]["nat"] for record, _ in normals]
                for got, key, dtype in zip(chosen, keys, dtypes):
                    self.assertTrue(close(got, key, dtype), (dtype, key))
                for mutation in ("index", "offset", "swap", "sine"):
                    self.assertFalse(all(close(got, key, dtype, mutation)
                                         for got, key, dtype in zip(chosen, keys, dtypes)),
                                     mutation)
                self.assertTrue(torch.equal(outputs[1], outputs[0] * 2))
                self.assertEqual(report["outputs"][6]["shape"], [])
                self.assertEqual(report["outputs"][7]["bits"], [])
                # A loose check of the moments of one key's 256 draws, with no claim of a law.
                wide = outputs[8].double()
                self.assertLess(abs(float(wide.mean())), 0.25)
                self.assertLess(abs(float(wide.var()) - 1), 0.35)

        def one(x):
            return torch.randn(8, dtype=x.dtype)

        for dtype in dtypes[:4]:
            floor = int(_nearest(Fraction(1, 10 ** 6), names[dtype]) * 2 ** 32)
            artifact = torchlean._capture(one, (torch.zeros(1, dtype=dtype),), "pytorch", EXECUTE)
            (record,) = [record for record in _records(artifact) if record["tag"] == "randn"]
            for draw in (0, 2, floor - 300, floor + 300, 2 ** 31 + 2 ** 23 + 1,
                         2 ** 31 + 2 ** 20 + 1, 2 ** 32 - 1):
                with self.subTest(dtype=dtype, draw=draw):
                    key = unmix(draw)
                    self.assertEqual(_splitmix64(key) % 2 ** 32, draw)
                    record["fields"][1]["nat"] = key
                    report = torchlean.verify(artifact, checks=EXECUTE, checker=CHECKER)
                    self.assert_passes(report)
                    (got,) = _outputs(report)
                    self.assertTrue(close(got, key, dtype), (got, reference(key, 8, dtype)))
                    if draw == 0:
                        self.assertFalse(close(got, key, dtype, "clamp"))
            if dtype == torch.float32:
                # A fixed regression vector: the binary32 words at key 0, near the reference.
                record["fields"][1]["nat"] = 0
                report = torchlean.verify(artifact, checks=EXECUTE, checker=CHECKER)
                self.assert_passes(report)
                self.assertEqual(report["outputs"][0]["bits"],
                                 [3214351319, 1067772856, 3198406722, 3204610764,
                                  3160702220, 1060559600, 1065232126, 3210050667])
                self.assertTrue(close(_outputs(report)[0], 0, dtype))

    def test_random_keys_bound_the_counters(self):
        # The key allocator accepts a node reading 2^32 counters at position 2^32 - 1 and refuses
        # more counters or a later position. randn reads two counters per entry, so a draw of
        # 2^31 entries is accepted and one more refused, without allocating either.
        from torchlean._artifact import _splitmix64
        from torchlean._emit import _emit_structure
        from torchlean._structure import _random_key

        base = _splitmix64(0)
        self.assertEqual(_random_key(2 ** 32 - 1, 2 ** 32),
                         (base + (2 ** 32 - 1) * 2 ** 32) % 2 ** 64)
        self.assertEqual(_random_key(2 ** 32 - 2, 2 ** 32 - 1),
                         (base + (2 ** 32 - 2) * 2 ** 32) % 2 ** 64)
        for position, counters in ((2 ** 32, 1), (0, 2 ** 32 + 1)):
            with self.assertRaises(NotImplementedError):
                _random_key(position, counters)
        for position, counters in ((-1, 1), (0, -1)):
            with self.assertRaises(ValueError):
                _random_key(position, counters)
        plan = {"op": "random", "operands": [], "sampler": "normal"}
        nodes = []
        _emit_structure(plan, nodes, {}, [2 ** 31], "float32")
        self.assertEqual(nodes[-1]["op"]["tag"], "randn")
        with self.assertRaises(NotImplementedError):
            _emit_structure(plan, [], {}, [2 ** 31 + 1], "float32")

    def test_random_generator_overloads_are_refused_at_the_importer(self):
        # The importer itself refuses the generator overloads of rand, randn and randperm, raw
        # randn_like and a randn of an integer dtype, on constructed graphs: torch.export can
        # refuse such calls before the importer sees them.
        torch = self.torch
        aten = torch.ops.aten
        unsupported = "unsupported PyTorch op"
        cases = {"randn.generator": (aten.randn.generator, ([3],), {"generator": None},
                                     torch.float32, unsupported),
                 "rand.generator": (aten.rand.generator, ([3],), {"generator": None},
                                    torch.float32, unsupported),
                 "randperm.generator": (aten.randperm.generator, (3,), {"generator": None},
                                        torch.int64, unsupported),
                 "randn_like": (aten.randn_like.default, None, {}, torch.float32, unsupported),
                 "randn of int64": (aten.randn.default, ([3],), {"dtype": torch.int64},
                                    torch.int64, "randn of")}
        for name, (target, arguments, keywords, dtype, text) in cases.items():
            with self.subTest(case=name):
                graph = torch.fx.Graph()
                x = graph.placeholder("x")
                x.meta["val"] = torch.zeros(3)
                node = graph.call_function(target, (x,) if arguments is None else arguments,
                                           keywords)
                node.meta["val"] = torch.zeros(3, dtype=dtype)
                graph.output(node)
                with self.assertRaises(NotImplementedError) as caught:
                    torchlean._lower_pytorch_graph(torch.nn.Module(), graph, ["x"])
                self.assertIn(text, str(caught.exception))
                if text == unsupported:
                    self.assertIn(str(target), str(caught.exception))

    def test_masked_scatter_nonzero_static_and_embedding_bag(self):
        # masked_scatter and nonzero_static read prefix counts of the mask (exact in binary64).
        # embedding_bag with constant offsets reduces each bag's rows, an empty bag zero.
        torch = self.torch
        nn = torch.nn
        torch.manual_seed(10)
        x = torch.randn(3, 4)
        x[0, 1] = 0.0

        class Bags(nn.Module):
            def __init__(self, mode, last=False):
                super().__init__()
                self.bag = nn.EmbeddingBag(10, 3, mode=mode, include_last_offset=last)
                with torch.no_grad():
                    self.bag.weight[4] = -torch.rand(3) - 1
                self.register_buffer("offsets", torch.tensor([0, 2, 2, 4] + ([6] if last else [])))

            def forward(self, indices, weights=None):
                return self.bag(indices, self.offsets, per_sample_weights=weights)

        def addressed(x, source):
            return (x.masked_scatter(x > 0, source), torch.nonzero_static(x, size=7),
                    torch.nonzero_static(x > 0.5, size=2, fill_value=0),
                    x.masked_scatter(torch.tensor([True, False, True, False]), source),
                    x[0].masked_scatter(x > 0, source), torch.nonzero_static(x, size=0),
                    torch.nonzero_static(x > 9, size=3, fill_value=2 ** 53 + 1))

        def empty_source(x, source):
            return x.masked_scatter(x > 100, source[:0])

        indices = torch.tensor([1, 2, 4, 5, 4, 9])
        for core in (False, True):
            with self.subTest(core=core):
                self.assert_passes(self.verify(addressed, x, torch.randn(12), core=core))
                self.assert_passes(self.verify(empty_source, x, torch.randn(3), core=core))
                weights = torch.rand(6)
                for mode, last, extra in (("sum", False, ()), ("mean", False, ()),
                                          ("max", False, ()), ("sum", True, (weights,)),
                                          ("max", True, ())):
                    model = Bags(mode, last).eval()
                    report = self.verify(model, indices, *extra, checks=EXECUTE, core=core)
                    self.assert_passes(report)
                    with torch.no_grad():
                        torch.testing.assert_close(_outputs(report)[0], model(indices, *extra),
                                                   rtol=1e-6, atol=1e-6)
        # Captured with two true entries and five source values, run where every entry is true:
        # the twelfth entry reads source index 11, and execution fails.
        small = torch.zeros(3, 4)
        small[0, 0] = small[1, 1] = 1.0
        artifact = torchlean._capture(lambda x, source: x.masked_scatter(x > 0, source),
                                      (small, torch.randn(5)), "pytorch", EXECUTE)
        values = dict(artifact["arguments"][0])
        values["bits"] = [1065353216] * 12
        artifact["arguments"][0] = values
        artifact.pop("reference_outputs", None)
        report = torchlean.verify(artifact, checks=EXECUTE, checker=CHECKER)
        self.assert_unsupported_execution(report, "outside")

    def test_orders_indices_and_selections_name_their_reason(self):
        torch = self.torch
        functional = torch.nn.functional
        x = self.x.reshape(2, 12)

        class Repeated(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("index", torch.tensor([[0, 0, 1]]))

            def forward(self, x):
                return x.scatter(1, self.index, x[:1, :3])

        cases = {
            "unsorted topk": (lambda x: torch.topk(x, 2, sorted=False).values, "sorted=False"),
            "sort without stable": (lambda x: torch.sort(x)[0], "the IR sort is stable"),
            "leaky_relu with slope 0": (lambda x: functional.leaky_relu(x, 0.0),
                                        "slope that is not positive"),
            "Boolean mask index": (lambda x: x[x > 0], "Boolean mask index"),
            "integer scatter_reduce": (lambda x: x.long().scatter_reduce(
                1, (x > 0).long(), x.long(), "amax"), "scatter_reduce of a"),
            "repeated scatter target": (Repeated(), "scatter with a repeated target"),
        }
        for name, (function, text) in cases.items():
            for core in (False, True):
                with self.subTest(name=name, core=core):
                    self.assert_refused(self.verify(function, x, checks=("structure",),
                                                    core=core), text)
        integers = torch.arange(24).reshape(2, 12)
        for core in (False, True):
            with self.subTest(name="integer cumsum", core=core):
                report = self.verify(lambda x: torch.cumsum(x, 1), integers, checks=("structure",),
                                     core=core)
                self.assert_refused(report, "a cumulative sum or product of a")

    def test_real_models_import_without_unsupported_nodes(self):
        torch = self.torch
        nn = torch.nn
        functional = nn.functional
        torch.manual_seed(0)

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1, self.fc2 = nn.Linear(8, 16), nn.Linear(16, 4)

            def forward(self, x):
                return functional.log_softmax(self.fc2(functional.relu(self.fc1(x))), dim=-1)

        class CNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1, self.bn = nn.Conv2d(1, 4, 3, padding=1), nn.BatchNorm2d(4)
                self.conv2, self.pool = nn.Conv2d(4, 8, 3, stride=2), nn.MaxPool2d(2)
                self.fc = nn.Linear(8, 3)

            def forward(self, x):
                x = functional.relu(self.bn(self.conv1(x)))
                return self.fc(torch.flatten(self.pool(functional.relu(self.conv2(x))), 1))

        class Block(nn.Module):
            def __init__(self, width=8, heads=2):
                super().__init__()
                self.heads = heads
                self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
                self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
                self.fc, self.out = nn.Linear(width, 2 * width), nn.Linear(2 * width, width)

            def forward(self, x):
                b, t, c = x.shape
                q, k, v = self.qkv(self.ln1(x)).split(c, dim=-1)
                q, k, v = (y.view(b, t, self.heads, c // self.heads).transpose(1, 2)
                           for y in (q, k, v))
                scores = (q @ k.transpose(-2, -1)) * (1.0 / (c // self.heads) ** 0.5)
                y = (functional.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(b, t, c)
                x = x + self.proj(y)
                return x + self.out(functional.gelu(self.fc(self.ln2(x)), approximate="tanh"))

        # The batch norm statistics are set so that the payload matters. The models execute, and
        # their values agree with PyTorch's within 1e-5 relative and absolute: the IR sums,
        # normalizes and computes elementary functions in other orders and formulas.
        cnn = CNN()
        with torch.no_grad():
            cnn.bn.running_mean.copy_(torch.randn(4))
            cnn.bn.running_var.copy_(torch.rand(4) + 0.5)
        models = {"mlp": (MLP(), torch.randn(2, 8)), "cnn": (cnn, torch.randn(2, 1, 6, 6)),
                  "block": (Block(), torch.randn(1, 4, 8))}
        for name, (model, x) in models.items():
            model.eval()
            for core in (False, True):
                with self.subTest(model=name, core=core):
                    report = self.verify(model, x, checks=EXECUTE, core=core)
                    self.assert_passes(report)
                    with torch.no_grad():
                        torch.testing.assert_close(_outputs(report)[0], model(x), rtol=1e-5,
                                                   atol=1e-5)

    def test_operator_models_import_or_name_their_lane(self):
        # A CNN with dilation, groups and the values of a max pool with indices, a model of
        # sorts, prefix sums and extrema, a classifier's argmax and an attention with a causal
        # mask, whose comparisons of integer positions fold to a Boolean constant at capture,
        # a topk and the indices of a max pool import.
        torch = self.torch
        nn = torch.nn
        functional = nn.functional
        torch.manual_seed(0)

        class CNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(2, 4, 3, padding=2, dilation=2)
                self.conv2 = nn.Conv2d(4, 8, 3, groups=2, dilation=(1, 2), padding=(1, 2))
                self.pool = nn.MaxPool2d(2, return_indices=True)
                self.fc = nn.Linear(8 * 3 * 3, 3)

            def forward(self, x):
                x = functional.leaky_relu(self.conv1(x), 0.1)
                values, indices = self.pool(functional.relu(self.conv2(x)))
                return self.fc(torch.flatten(values, 1))

        class PoolIndices(CNN):
            def forward(self, x):
                return self.pool(self.conv1(x))[1]

        class Ranks(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(6, 6)

            def forward(self, x):
                h = functional.leaky_relu(self.fc(x), 0.2)
                s = torch.sort(h, dim=-1, stable=True, descending=True).values
                best = torch.max(h, dim=-1, keepdim=True).values
                return (torch.cummax(torch.cumsum(s, dim=-1) - best, dim=0).values,
                        torch.min(h, dim=0).values)

        class Argmax(Ranks):
            def forward(self, x):
                return torch.argmax(self.fc(x), dim=-1)

        class Topk(Ranks):
            def forward(self, x):
                return torch.topk(self.fc(x), 2, dim=-1).values

        class Causal(nn.Module):
            def __init__(self, width=8, heads=2):
                super().__init__()
                self.heads = heads
                self.ln = nn.LayerNorm(width)
                self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)

            def forward(self, x):
                b, t, c = x.shape
                q, k, v = self.qkv(self.ln(x)).split(c, dim=-1)
                q, k, v = (y.view(b, t, self.heads, c // self.heads).transpose(1, 2)
                           for y in (q, k, v))
                i = torch.arange(t)
                mask = torch.logical_and(i[None, :] <= i[:, None], i[:, None] >= 0)
                scores = (q @ k.transpose(-2, -1)) * (1.0 / (c // self.heads) ** 0.5)
                scores = torch.where(mask, scores, float("-inf"))
                y = (functional.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(b, t, c)
                return x + self.proj(y)

        image, rows = torch.randn(2, 2, 6, 6), torch.randn(4, 6)
        for core in (False, True):
            with self.subTest(model="cnn", core=core):
                self.assert_passes(self.verify(CNN().eval(), image, checks=("structure", "types"),
                                               core=core))
            with self.subTest(model="ranks", core=core):
                self.assert_passes(self.verify(Ranks().eval(), rows, checks=EXECUTE, core=core))
            with self.subTest(model="argmax", core=core):
                self.assert_passes(self.verify(Argmax().eval(), rows, core=core))
            with self.subTest(model="causal", core=core):
                causal, sequence = Causal().eval(), torch.randn(1, 4, 8)
                report = self.verify(causal, sequence, checks=EXECUTE, core=core)
                self.assert_passes(report)
                with torch.no_grad():
                    torch.testing.assert_close(_outputs(report)[0], causal(sequence), rtol=1e-5,
                                               atol=1e-5)
            with self.subTest(model="pool indices", core=core):
                self.assert_passes(self.verify(PoolIndices().eval(), image, core=core))
            with self.subTest(model="topk", core=core):
                self.assert_passes(self.verify(Topk().eval(), rows, checks=EXECUTE, core=core))
        kinds = _tags(self.artifact(Ranks().eval(), rows))
        for kind in ("sort", "cumsum", "cummax", "topk", "max"):
            self.assertIn(kind, kinds)

    def test_scaled_dot_product_attention_follows_its_docstring(self):
        # The default graph's scaled_dot_product_attention is the docstring's equivalent code:
        # scores q @ k^T (left folds from +0) times the scale rounded once, plus the bias (zeros,
        # -inf where a Boolean or the causal mask is false, or the floating mask plus zeros),
        # softmax over the keys (the IR's two-pass softmax: the maximum, exp of the differences
        # correctly rounded, their left-fold sum, and the quotients, taken over the exact sum
        # where the rounded one swallows a term), times v. Binary16 and bfloat16 compute in
        # binary32 and round once. A row whose keys are all blocked is NaN, as that code
        # computes it (eager PyTorch gives the row zero weights). The reference evaluates those
        # steps with exact rationals rounded each time.
        import math
        from fractions import Fraction

        torch = self.torch
        F = torch.nn.functional

        class Attention(torch.nn.Module):
            def __init__(self, **options):
                super().__init__()
                self.options = options

            def forward(self, q, k, v, *mask):
                return F.scaled_dot_product_attention(q, k, v, *mask, **self.options)

        def softmax(row, name):
            peak = row[0]
            for value in row[1:]:
                peak = _ieee_maximum(peak, value)
            shifted = [_nearest(Fraction(v) - Fraction(peak), name)
                       if math.isfinite(v) and math.isfinite(peak) else v - peak for v in row]
            exps = [_exp(v, name) for v in shifted]
            total, stalled = 0.0, False
            for e in exps:
                following = _nearest(Fraction(total) + Fraction(e), name) if \
                    math.isfinite(total) and math.isfinite(e) else total + e
                stalled = stalled or (e != 0 and following == total)
                total = following
            normal = math.isfinite(total) and abs(total) >= 2.0 ** FORMATS[name][1]
            if not stalled and normal and total != 0:
                return [_nearest(Fraction(e) / Fraction(total), name) if math.isfinite(e)
                        else e / total for e in exps]
            if not all(math.isfinite(e) for e in exps) or sum(map(Fraction, exps)) == 0:
                return [e / total if total else math.nan for e in exps]
            exact = sum(map(Fraction, exps))
            return [_nearest(Fraction(e) / exact, name) for e in exps]

        def rounded(operation, a, b, name):
            # One IEEE operation: exact and rounded once on finite operands (an exact zero sum is
            # -0 only for -0 + -0, a zero product has the sign of the product), Python's binary64
            # arithmetic, exact for infinities and NaN, otherwise.
            if not (math.isfinite(a) and math.isfinite(b)):
                return a * b if operation == "mul" else a + b
            exact = Fraction(a) * Fraction(b) if operation == "mul" else Fraction(a) + Fraction(b)
            if exact:
                return _nearest(exact, name)
            negative = (math.copysign(1, a) * math.copysign(1, b) < 0 if operation == "mul"
                        else math.copysign(1, a) < 0 and math.copysign(1, b) < 0)
            return -0.0 if negative else 0.0

        def dot(left, right, name):
            total = 0.0
            for a, b in zip(left, right):
                total = rounded("add", total, rounded("mul", a, b, name), name)
            return total

        def reference(q, k, v, bias, scale, name):
            # q, k, v, bias as nested lists of Python floats over (batch, rows, ...). `name` is
            # the computing format.
            out = []
            for qb, kb, vb, bb in zip(q, k, v, bias):
                rows = []
                for i, qi in enumerate(qb):
                    scores = []
                    for j, kj in enumerate(kb):
                        s = rounded("mul", dot(qi, kj, name), scale, name)
                        scores.append(rounded("add", s, bb[i][j], name))
                    weights = softmax(scores, name)
                    rows.append([dot(weights, [vj[c] for vj in vb], name)
                                 for c in range(len(vb[0]))])
                out.append(rows)
            return out

        generator = torch.Generator().manual_seed(15)
        q = torch.randn(3, 4, 5, generator=generator)
        k = torch.randn(3, 6, 5, generator=generator)
        v = torch.randn(3, 6, 2, generator=generator)
        mask = torch.rand(4, 6, generator=generator) > 0.4
        mask[1] = False
        additive = torch.randn(4, 6, generator=generator)
        causal = torch.ones(4, 6, dtype=torch.bool).tril()
        for dtype in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
            name = str(dtype).removeprefix("torch.")
            compute = "float64" if dtype == torch.float64 else "float32"
            qd, kd, vd = (t.to(dtype) for t in (q, k, v))
            wide = [t.to(torch.float64).tolist() for t in (qd, kd, vd)]
            cases = {"boolean": ({}, (mask,), torch.zeros(4, 6).masked_fill(~mask, -math.inf),
                                 1 / math.sqrt(5)),
                     "floating": ({"scale": 0.3}, (additive.to(dtype),),
                                  additive.to(dtype).to(torch.float64), 0.3),
                     "causal": ({"is_causal": True}, (), torch.zeros(4, 6).masked_fill(
                         ~causal, -math.inf), 1 / math.sqrt(5))}
            for case, (options, extra, bias, scale) in cases.items():
                with self.subTest(dtype=name, case=case):
                    report = self.verify(Attention(**options).eval(), qd, kd, vd, *extra,
                                         checks=EXECUTE)
                    self.assert_passes(report)
                    (got,) = _outputs(report)
                    terms = [[[_nearest(b, compute) + 0.0 for b in row]
                              for row in bias.tolist()]] * 3
                    want = reference(*wide, terms, _nearest(scale, compute), compute)
                    expected = [[[_nearest(value, name) if math.isfinite(value) else value
                                  for value in row] for row in batch] for batch in want]
                    _same(self, got.to(torch.float64).flatten().tolist(),
                          [value for batch in expected for row in batch for value in row])
                    if case == "boolean":
                        self.assertTrue(all(math.isnan(value) for batch in expected
                                            for value in batch[1]))
        # Scores 0 and -17 (scale 1): exp(-17) is swallowed by the binary32 sum, and the IR's
        # softmax, which divides by the exact sum, gives the first weight 1 - 2^-24 where the
        # rounded sum would give 1.
        repair = Attention(scale=1.0).eval()
        q1, k1 = torch.ones(1, 1, 1), torch.tensor([[[0.0], [-17.0]]])
        v1 = torch.tensor([[[1.0], [0.0]]])
        report = self.verify(repair, q1, k1, v1, checks=EXECUTE)
        self.assert_passes(report)
        want = reference(q1.tolist(), k1.tolist(), v1.tolist(), [[[0.0, 0.0]]], 1.0, "float32")
        _same(self, _outputs(report)[0].flatten().tolist(), want[0][0])
        self.assertEqual(want[0][0], [1 - 2.0 ** -24])
        artifact = self.artifact(Attention().eval(), q, k, v, mask)
        kinds = _tags(artifact)
        self.assertEqual([kind for kind in kinds if kind in ("matmul", "softmax")],
                         ["matmul", "softmax", "matmul"])
        self.assertFalse(any("cast" in step for step in _steps(artifact)))
        self.assertEqual(_registered(artifact), [])
        # Grouped heads repeat the key and value heads, and dropout_p > 0 is refused.
        group = Attention(enable_gqa=True).eval()
        qg, kg, vg = torch.randn(2, 4, 3, 5), torch.randn(2, 2, 6, 5), torch.randn(2, 2, 6, 2)
        report = self.verify(group, qg, kg, vg, checks=EXECUTE)
        self.assert_passes(report)
        repeated = Attention().eval()(qg, kg.repeat_interleave(2, dim=1),
                                      vg.repeat_interleave(2, dim=1))
        plain = self.verify(Attention().eval(), qg, kg.repeat_interleave(2, dim=1),
                            vg.repeat_interleave(2, dim=1), checks=EXECUTE)
        self.assertEqual(_words(report), _words(plain))
        self.assertEqual(tuple(_outputs(report)[0].shape), tuple(repeated.shape))
        report = self.verify(Attention(dropout_p=0.1).eval(), q, k, v, checks=("structure",))
        self.assert_refused(report, "dropout_p > 0")

    def test_integer_bitwise_operators_and_shifts(self):
        # bitwise_and, _or, _xor and _not of integer tensors act on their words. << and >> (sign
        # fill for a signed type, zeros for an unsigned one) shift every bit out at an amount of
        # the width or more, a negative one included, which PyTorch documents as undefined. A
        # Python int operand converts to the tensor's dtype. nextafter is C's. The reference is
        # Python integer arithmetic on the words.
        import math

        import numpy as np

        torch = self.torch

        class Bits(torch.nn.Module):
            def forward(self, x, k):
                return (x & k, x | k, x ^ k, ~x, x << k, x >> k, x & 6, torch.bitwise_xor(x, -3))

        rng = np.random.default_rng(16)
        for dtype in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
            info = torch.iinfo(dtype)
            width = info.bits
            x = torch.tensor(rng.integers(info.min, info.max, 24, endpoint=True), dtype=dtype)
            k = torch.tensor(rng.integers(0, width + 3, 24), dtype=dtype)
            k[:3] = torch.tensor([width, width + 1, -1 if info.min < 0 else 200])
            mask = 1 << width
            signed = info.min < 0
            words = lambda t: [int(v) % mask for v in t.tolist()]
            expected = [[] for _ in range(8)]
            for a, b in zip(words(x), words(k)):
                top = (a >> (width - 1)) & 1
                right = (((mask - 1 if top else 0) if b >= width else (a - mask if top else a) >> b)
                         if signed else (0 if b >= width else a >> b))
                for out, value in zip(expected, (a & b, a | b, a ^ b, ~a,
                                                 0 if b >= width else a << b, right, a & 6,
                                                 a ^ (-3 % mask))):
                    out.append(value % mask)
            with self.subTest(dtype=str(dtype)):
                for core in (False, True):
                    report = self.verify(Bits().eval(), x, k, checks=EXECUTE, core=core)
                    self.assert_passes(report)
                    self.assertEqual(_words(report), expected)
        kinds = _tags(self.artifact(Bits().eval(), x, k))
        for kind in ("bitwiseAnd", "bitwiseNot", "bitwiseLeftShift", "shiftRightArithmetic"):
            self.assertIn(kind, kinds)
        unsigned = _tags(self.artifact(Bits().eval(), x.to(torch.uint8), k.to(torch.uint8)))
        self.assertIn("shiftRightLogical", unsigned)
        a = torch.tensor([0.0, -0.0, 1.0, 3.0, float("inf"), float("nan"), 2.0 ** -149])
        b = torch.tensor([1.0, 0.0, 1.0, -float("inf"), 0.0, 1.0, 0.0])
        report = self.verify(lambda a, b: torch.nextafter(a, b), a, b, checks=EXECUTE)
        self.assert_passes(report)
        (got,) = _words(report)
        self.assertEqual(got[:5] + got[6:], [1, 0, 0x3f800000, 0x403fffff, 0x7f7fffff, 0])
        self.assertTrue(math.isnan(_float_of_word(got[5], np.float32)))


@unittest.skipUnless(CHECKER and HAS_JAX, "needs the built verify executable and JAX")
class JaxGeneralOperatorTests(CheckerCase):
    def setUp(self):
        import jax
        import jax.numpy as jnp

        self.jax, self.jnp = jax, jnp
        self.x = jnp.arange(24.0, dtype=jnp.float32).reshape(2, 3, 4) - 10

    def verify(self, function, *inputs, checks=COMPARE):
        return torchlean.verify(function, *inputs, frontend="jax", checks=checks,
                                checker=CHECKER)

    def artifact(self, function, *inputs):
        from torchlean.jaxpr import export

        return export(function, *inputs)

    def test_reductions_and_softmax(self):
        jax, jnp = self.jax, self.jnp

        def reductions(x):
            return (jnp.max(x, axis=(0, 2)), jnp.min(x, axis=1), jnp.prod(x[:, :, :2], axis=2),
                    jnp.sum(x, axis=(0, 1)))

        from torchlean._table import op_table

        self.assert_passes(self.verify(reductions, self.x))
        document = self.artifact(reductions, self.x)
        self.assertEqual(document["artifact"]["format"], op_table()["format"])
        codes = op_table()["codes"]["reduce"]
        self.assertEqual({record["fields"][1]["nat"] for record in _records(document)
                          if record["tag"] == "reduce"},
                         {codes[name] for name in ("max", "min", "prod", "sum")})
        softmax = lambda x: (jax.nn.softmax(x, axis=-1), jax.nn.log_softmax(x, axis=1))
        self.assert_passes(self.verify(softmax, self.x / 7, checks=EXECUTE))

    def test_data_movement_and_contraction(self):
        jax, jnp = self.jax, self.jnp
        lax = jax.lax

        def unit(x):
            return (lax.pad(x, 0.5, ((0, 0, 0), (-1, 1, 1), (1, 0, 0))),
                    jnp.flip(x, axis=(0, 2)), jnp.concatenate([x, x], axis=1),
                    jnp.split(x, [1], axis=2)[1], jnp.tile(x, (1, 2, 1)),
                    jnp.broadcast_to(x[:, None], (2, 5, 3, 4)),
                    x + lax.broadcasted_iota(jnp.float32, x.shape, 1),
                    jnp.einsum("bij,bkj->bik", x, x))

        movement = lambda x: (x[:, 1:3, ::2], *unit(x))
        document = self.artifact(movement, self.x)
        tags = _tags(document)
        for tag in ("slice", "pad", "flip", "cat", "broadcast", "contract"):
            self.assertIn(tag, tags)
        self.assertNotIn("matmul", tags)
        # A slice record's fields are its operand shape, dimensions, starts, stops and steps.
        steps = [record["fields"][4]["nats"] for record in _records(document)
                 if record["tag"] == "slice"]
        self.assertEqual([step for step in steps if step != [1] * len(step)], [[1, 2]])
        self.assert_passes(self.verify(unit, self.x))

    def test_elementary_exact_and_sorting_primitives(self):
        # The elementary functions execute, and floor, ceil, round to nearest even, rem and the
        # stable one-operand sort, which keeps -0 and +0 in order and puts NaN last, compare.
        jax, jnp = self.jax, self.jnp
        lax = jax.lax
        x = jnp.linspace(-2.0, 2.0, 24, dtype=jnp.float32).reshape(3, 8)
        x = x.at[0, :3].set(jnp.array([0.0, -0.0, jnp.nan]))

        def elementary(x):
            return (lax.expm1(x), lax.log1p(jnp.abs(x)), lax.tan(x / 4), jnp.sinh(x),
                    jnp.cosh(x), jnp.arcsinh(x), jnp.arccosh(jnp.abs(x) + 1),
                    jnp.arctanh(x / 4), jnp.exp2(x))

        def exact(x):
            return (jnp.floor(x * 3), jnp.ceil(x * 3), jnp.round(x * 2.5),
                    lax.rem(x * 5, jnp.asarray(1.5, x.dtype)), jnp.sort(x, axis=-1),
                    jnp.sort(x, axis=0))

        for dtype in (jnp.float32, jnp.bfloat16):
            with self.subTest(dtype=dtype):
                self.assert_passes(self.verify(elementary, x.astype(dtype), checks=EXECUTE))
        for dtype in (jnp.float32, jnp.float16):
            with self.subTest(dtype=dtype):
                self.assert_passes(self.verify(exact, x.astype(dtype)))
        tags = _tags(self.artifact(elementary, x)) + _tags(self.artifact(exact, x))
        for tag in ("expm1", "log1p", "tan", "sinh", "cosh", "asinh", "acosh", "atanh", "exp2",
                    "floor", "ceil", "round", "fmod", "sort"):
            self.assertIn(tag, tags)

    def test_argmax_and_argmin_match_eager(self):
        # lax.argmax and lax.argmin are the registered Op.argmax and Op.argmin, their index
        # results encoded in the primitive's index dtype. `compare` requires the eager words, at
        # repeated values, signed zeros and NaNs.
        import numpy as np

        jnp = self.jnp
        nan = np.nan
        x = np.array([[1.0, 3.0, 3.0, -0.0, 0.0], [nan, 2.0, nan, -1.0, 5.0],
                      [0.0, -0.0, 0.0, -0.0, -0.0], [7.0, -2.0, 7.0, nan, 1.0]],
                     dtype=np.float32)

        def extrema(x):
            return (jnp.argmax(x, axis=1), jnp.argmin(x, axis=0), jnp.argmax(x),
                    jnp.argmin(x, axis=1, keepdims=True))

        for dtype in (np.float32, jnp.bfloat16, np.float16):
            with self.subTest(dtype=dtype):
                self.assert_passes(self.verify(extrema, jnp.asarray(x, dtype=dtype)))
        records = _registered(self.artifact(extrema, jnp.asarray(x)))
        self.assertEqual({record["tag"] for record in records}, {"argmax", "argmin"})

    def test_sign_scans_and_top_k_match_eager(self):
        # lax.sign keeps -0 and NaN, and of integers is -1, 0 or 1 (the type's limits included).
        # cumsum and cumprod are the IR's left folds, compared where every partial result of any
        # grouping is exact and elsewhere equal word for word to NumPy's sequential binary32
        # cumsum. cummax and cummin are running IEEE extrema, the words exact at mixed signed
        # zeros and infinities, a NaN propagating. top_k is the stable topk, equal entries in
        # input order, for k = 0, k = n and is_stable=False too, and its execution fails on a
        # fiber holding a NaN or both zeros. `compare` checks the positions of NaNs and ignores
        # their words. Eager and one jax.jit computation.
        import numpy as np
        import torch

        jax, jnp = self.jax, self.jnp
        lax = jax.lax
        nan, inf = np.nan, np.inf
        signs = np.array([[-2.5, -0.0, 0.0, nan, 3.0, -inf], [inf, 1e-30, -1e-30, 7.0, -0.0, 0.0]])
        exact = np.array([[1.0, -2.0, 0.5, 4.0, -0.0, 3.0], [0.0, 2.0, -0.25, -0.0, 8.0, 1.0]])
        zeros = np.array([[0.0, -0.0, 0.0, -1.0, nan, 2.0], [-0.0, 0.0, -0.0, -3.0, -0.0, 0.0],
                          [-inf, nan, 1.0, nan, inf, -inf], [inf, -inf, -0.0, -inf, 0.0, nan]])
        # No fiber holds a NaN or both zeros, where the stable order and XLA's totalOrder agree.
        ranked = np.array([[3.0, 0.0, 3.0, -inf, 0.0, 1.0], [-1.0, -1.0, 2.0, -0.0, 2.0, -0.0]])

        def unit(s, e, z, r):
            return (jnp.sign(s), jnp.cumsum(e, axis=1), lax.cumprod(e, 1, reverse=True),
                    lax.cumsum(e, 0, reverse=True), lax.cummax(z, 1), lax.cummin(z, 1),
                    lax.cummax(z, 0, reverse=True), lax.cummin(z, 1, reverse=True),
                    *lax.top_k(r, 4), *lax.top_k(r, 1, axis=0), *lax.top_k(r, 0),
                    *lax.top_k(r, 6), *lax.top_k(r, 3, is_stable=False))

        samples = (signs, exact, zeros, ranked)
        for dtype in (jnp.float32, jnp.bfloat16, jnp.float16):
            with self.subTest(dtype=str(dtype)):
                arguments = [jnp.asarray(value, dtype) for value in samples]
                self.assert_passes(self.verify(unit, *arguments))
                self.assert_passes(self.verify(jax.jit(unit), *arguments))
        with jax.enable_x64(True):
            arguments = [jnp.asarray(value, jnp.float64) for value in samples]
            self.assert_passes(self.verify(unit, *arguments))
            self.assert_passes(self.verify(jax.jit(unit), *arguments))
        # A prefix of -0 entries sums to +0 from the fold's seed, as XLA's CPU code computes it.
        self.assert_passes(self.verify(lambda e: jnp.cumsum(e, axis=1),
                                       jnp.array([[-0.0, -0.0, 1.0, -0.0]])))
        for dtype in (jnp.int8, jnp.int16, jnp.int32, jnp.uint8, jnp.uint32):
            with self.subTest(dtype=str(dtype)):
                info = np.iinfo(dtype)
                values = np.array([[info.min, 0, info.max, 1], [info.max - 1, 0, info.min + 1, 7]])
                self.assert_passes(self.verify(jnp.sign, jnp.asarray(values, dtype)))
                self.assert_passes(self.verify(jax.jit(jnp.sign), jnp.asarray(values, dtype)))
        # Where a NaN or both zeros could decide the order, execution fails, but k = 0 decides none.
        for fiber in ([-0.0, 0.0, 1.0], [1.0, nan, 2.0], [-nan, 1.0, 0.0]):
            with self.subTest(fiber=fiber):
                self.assert_unsupported_execution(
                    self.verify(lambda r: lax.top_k(r, 1), jnp.array([fiber, [1.0, 2.0, 3.0]])),
                    "outside")
        self.assert_passes(self.verify(lambda r: lax.top_k(r, 0), jnp.array([-0.0, 0.0, nan])))
        # On general samples the left fold is NumPy's sequential cumsum, word for word.
        x = np.random.default_rng(5).standard_normal((3, 50)).astype(np.float32)
        report = self.verify(lambda x: jnp.cumsum(x, axis=1), x, checks=EXECUTE)
        self.assert_passes(report)
        (actual,) = _outputs(report)
        self.assertTrue(torch.equal(actual.view(torch.int32),
                                    torch.from_numpy(np.cumsum(x, axis=1)).view(torch.int32)))
        tags = _tags(self.artifact(unit, *[jnp.asarray(value, jnp.float32) for value in samples]))
        for tag in ("copysign", "select", "cumsum", "cumprod", "cummax", "cummin", "flip", "topk"):
            self.assertIn(tag, tags)

    def test_runtime_indices_match_eager(self):
        # gather, dynamic_slice, scatter, scatter-add, scatter-sub and dynamic_update_slice at
        # runtime indices are one registered gather or scatter: starts clamped as XLA clamps
        # them, fill_value at a FILL_OR_DROP gather start outside, dropped scatter windows,
        # repeated scatter-add targets summed in index order (1e8 - 1e8 + 1 is 1, where
        # 1e8 + 1 - 1e8 is 0) and the last repeated scatter update kept, overlapping windows and
        # batching dimensions included.
        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        i = jnp.array([2, -1, 0, 5, -4], dtype=jnp.int32)
        j = jnp.array([[0, 3], [1, 1], [2, 0]], dtype=jnp.int32)
        rng = np.random.default_rng(0)
        u = (rng.standard_normal((2, 5, 4)) * 10.0 ** rng.integers(-2, 9, (2, 5, 4)))
        windows = lax.ScatterDimensionNumbers(update_window_dims=(0, 1), inserted_window_dims=(),
                                              scatter_dims_to_operand_dims=(0,))

        def reads(x, i, j, s):
            return (x[:, i], x[1, i[:3]], jnp.take(x, i, axis=2),
                    jnp.take_along_axis(x[0], j, axis=1), x.at[:, i].get(mode="clip"),
                    x.at[:, i].get(mode="fill", fill_value=-1.0),
                    lax.dynamic_slice(x, (s, s, 1), (1, 2, 3)),
                    lax.dynamic_index_in_dim(x, s, axis=2),
                    jax.vmap(lambda row, k: row[k])(x[0], j[:, 0]))

        def writes(x, i, s, u):
            sums = jnp.zeros(3, x.dtype).at[jnp.array([1, 1, 1, 0])].add(
                jnp.array([1e8, -1e8, 1.0, 5.0], x.dtype))
            overlap = lax.scatter_add(u[1, 1:, 2:], jnp.array([[0], [1], [2]]), u[:, :2, :3],
                                      windows)
            return (x.at[:, i].set(7.0), x.at[:, i].add(u), x.at[0, i].add(1e8),
                    x.at[:, i].set(7.0, mode="clip"), x.at[:, i].add(1.0, mode="drop"),
                    x.at[:, i].set(u), x.at[:, i].subtract(u), sums, overlap,
                    lax.dynamic_update_slice(x, u[:1, :2, :3], (s, s, 1)),
                    jax.vmap(lambda row, k: row.at[k].add(3.0))(x[0], j[:, 0]))

        for dtype in (jnp.float32, jnp.bfloat16):
            x = self.x.astype(dtype)
            for s in (1, 7, -9):
                start = jnp.int32(s)
                with self.subTest(dtype=str(dtype), start=s):
                    self.assert_passes(self.verify(reads, x, i, j, start))
                    self.assert_passes(self.verify(writes, x, i, start, jnp.asarray(u, dtype)))
        # A FILL_OR_DROP gather converts int64 indices to int32 first, so 2 ** 32 + 1 reads entry
        # 1 and 2 ** 40 entry 0, as in JAX.
        with jax.enable_x64(True):
            wide = jnp.array([2 ** 32 + 1, -1, 2 ** 40, 4], dtype=jnp.int64)
            self.assert_passes(self.verify(lambda x, k: (jnp.take(x, k, axis=2), x[:, k[1:2]]),
                                           self.x, wide))
        document = self.artifact(writes, self.x, i, jnp.int32(1), jnp.asarray(u, jnp.float32))
        self.assertEqual({record["tag"] for record in _registered(document)}, {"scatter"})
        refused = {"integer gather": (lambda x, i: x.astype(jnp.int32)[:, i], "only floating")}
        for name, (function, text) in refused.items():
            with self.subTest(name=name):
                self.assert_refused(self.verify(function, self.x, i, checks=("structure",)), text)

    def test_runtime_scatter_products_match_eager(self):
        # scatter-mul at runtime indices is one registered Op.scatter with the rule prod. It
        # multiplies repeated targets in row-major order of the index vectors, in the clip and
        # drop modes too. Binary16 and bfloat16 products round at every step, as XLA's CPU code
        # does: the largest finite value times 2 times 0.5 overflows to inf. A NaN update
        # propagates. A select step replaces a dropped update by one before the scatter.
        # Binary64 too. Eager and one jax.jit computation.
        import numpy as np
        from torchlean._table import op_table

        jax, jnp = self.jax, self.jnp
        i = jnp.array([2, -1, 0, 5, -4, 2], dtype=jnp.int32)
        rng = np.random.default_rng(3)
        u = rng.uniform(-2.0, 2.0, (2, 6, 4))
        u[1, 0, 0] = np.nan
        u[0, 3, :] = np.inf

        def writes(x, i, u):
            return (x.at[:, i].multiply(u), x.at[0, i].multiply(2.0, mode="drop"),
                    x.at[:, i].multiply(u, mode="drop"), x.at[:, i].multiply(u, mode="clip"),
                    x.at[1, i[:3]].multiply(-u[0, :3]))

        def overflow(x, u):
            return x.at[jnp.array([0, 0, 0])].multiply(u)

        for dtype in (jnp.float32, jnp.bfloat16, jnp.float16):
            x = self.x.astype(dtype)
            with self.subTest(dtype=str(dtype)):
                self.assert_passes(self.verify(writes, x, i, jnp.asarray(u, dtype)))
                big = float(jnp.finfo(dtype).max)
                ones, factors = jnp.ones(2, dtype), jnp.array([big, 2.0, 0.5], dtype)
                self.assert_passes(self.verify(overflow, ones, factors))
                self.assert_passes(self.verify(jax.jit(overflow), ones, factors))
        with jax.enable_x64(True):
            self.assert_passes(self.verify(writes, self.x.astype(jnp.float64), i,
                                           jnp.asarray(u, jnp.float64)))
        steps = _steps(self.artifact(lambda x, i, u: x.at[:, i].multiply(u, mode="drop"), self.x,
                                     i, jnp.asarray(u, jnp.float32)))
        scatters = [step["op"] for step in steps
                    if "op" in step and step["op"]["record"]["tag"] == "scatter"]
        self.assertEqual([op["record"]["fields"][2] for op in scatters],
                         [{"nat": op_table()["codes"]["scatter"]["prod"]}])
        # The scatter's operands are the operand, the decoded indices and the updates.
        updates = steps[scatters[0]["inputs"][2][0]]
        self.assertEqual(updates["op"]["record"]["tag"], "select")

    def test_runtime_scatter_extrema_match_eager(self):
        # scatter-max and scatter-min at runtime indices are one registered Op.scatter with the
        # rules max and min, IEEE maximum and minimum as XLA's CPU code computes them.
        # max(-0, +0) and max(+0, -0) are +0 and min of both is -0. The compared words are
        # exact at zeros, so the sign of a zero counts. A NaN propagates through later updates.
        # The cases cover repeated targets, the clip and drop modes, binary16, bfloat16 and
        # binary64. Eager and one jax.jit computation.
        import numpy as np
        from torchlean._table import op_table

        jax, jnp = self.jax, self.jnp
        i = jnp.array([2, -1, 0, 5, -4, 2], dtype=jnp.int32)
        rng = np.random.default_rng(4)
        u = rng.uniform(-12.0, 12.0, (2, 6, 4))
        u[1, 0, 0] = np.nan

        def writes(x, i, u):
            return (x.at[:, i].max(u), x.at[:, i].min(u), x.at[0, i].max(2.0, mode="drop"),
                    x.at[:, i].min(u, mode="drop"), x.at[:, i].max(u, mode="clip"))

        def zeros(x, u):
            k = jnp.array([0, 1, 2, 2, 3])
            return x.at[k].max(u), x.at[k].min(u)

        signed = jnp.array([-0.0, 0.0, 1.0, np.nan])
        updates = jnp.array([0.0, -0.0, np.nan, 5.0, -0.0])
        for dtype in (jnp.float32, jnp.bfloat16, jnp.float16):
            with self.subTest(dtype=str(dtype)):
                self.assert_passes(self.verify(writes, self.x.astype(dtype), i,
                                               jnp.asarray(u, dtype)))
                self.assert_passes(self.verify(zeros, signed.astype(dtype), updates.astype(dtype)))
                self.assert_passes(self.verify(jax.jit(zeros), signed.astype(dtype),
                                               updates.astype(dtype)))
        with jax.enable_x64(True):
            self.assert_passes(self.verify(writes, self.x.astype(jnp.float64), i,
                                           jnp.asarray(u, jnp.float64)))
            self.assert_passes(self.verify(zeros, signed.astype(jnp.float64),
                                           updates.astype(jnp.float64)))
        rules = op_table()["codes"]["scatter"]
        document = self.artifact(writes, self.x, i, jnp.asarray(u, jnp.float32))
        self.assertEqual({record["fields"][2]["nat"] for record in _records(document)
                          if record["tag"] == "scatter"}, {rules["max"], rules["min"]})

    def test_runtime_index_windows_and_refusals(self):
        # Windows partly outside a length-four operand are dropped whole or clipped. Window rows
        # apply index vector by index vector ([0, 1e8, 0] gets 1 in the middle). Binary16 and
        # bfloat16 updates round at each step (2048 + 1 + 1 stays 2048 on XLA's CPU code).
        # int8 and uint32 indices and int8 starts at -1, -extent and -extent - 1 read as JAX
        # reads them, a uint32 2 ** 31 clipped to the last entry and filled in FILL_OR_DROP.
        # Raw gathers and scatters of an empty index batch or an empty window are registered
        # operators that read and write nothing. The windows and the rounding also run as one
        # jax.jit computation, compared with its compiled execution. .at[].apply, an operand
        # without entries, uint64 indices and a constant its declared type does not hold
        # (beyond an integer type's bounds, or overflowing to an infinity) are refused.
        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        modes = lax.GatherScatterMode
        rows = lax.ScatterDimensionNumbers(update_window_dims=(1,), inserted_window_dims=(),
                                           scatter_dims_to_operand_dims=(0,))
        columns = lax.ScatterDimensionNumbers(update_window_dims=(0,), inserted_window_dims=(),
                                              scatter_dims_to_operand_dims=(0,))
        last = lax.GatherDimensionNumbers(offset_dims=(0, 1), collapsed_slice_dims=(2,),
                                          start_index_map=(2,))
        line = jnp.array([1.0, 2.0, 3.0, 4.0])
        starts = jnp.array([[-1], [3], [1], [2]], dtype=jnp.int32)
        pieces = jnp.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0], [70.0, 80.0]])

        def edges(line, starts, pieces):
            return tuple(scatter(line, starts, pieces, rows, mode=mode)
                         for scatter in (lax.scatter, lax.scatter_add)
                         for mode in (modes.FILL_OR_DROP, modes.CLIP)) + (
                lax.scatter_add(jnp.array([0.0, 1e8, 0.0]), jnp.array([[0], [1]]),
                                jnp.array([[0.0, 1.0], [-1e8, 0.0]]), columns),)

        def twice(base):
            return base.at[jnp.array([0, 0])].add(jnp.ones(2, base.dtype))

        def narrow(x, small, unsigned, s):
            return (jnp.take(x, small, axis=2), x[:, small[:2]],
                    lax.gather(x, unsigned[:, None], last, (2, 3, 1), mode=modes.CLIP),
                    lax.gather(x, unsigned[:, None], last, (2, 3, 1), mode=modes.FILL_OR_DROP,
                               fill_value=-1.0),
                    lax.dynamic_slice(x, (s, s, s), (1, 2, 3)))

        planes = lax.ScatterDimensionNumbers(update_window_dims=(0, 1), inserted_window_dims=(2,),
                                             scatter_dims_to_operand_dims=(2,))

        def empties(x, unsigned):
            none = unsigned[:0, None]
            return (lax.gather(x, none, last, (2, 3, 1), mode=modes.CLIP),
                    lax.gather(x, unsigned[:, None], last, (2, 0, 1), mode=modes.CLIP),
                    lax.scatter(x, none, jnp.ones((2, 3, 0)), planes),
                    lax.scatter_add(x, unsigned[:, None], jnp.ones((2, 0, 3)), planes))

        self.assert_passes(self.verify(edges, line, starts, pieces))
        self.assert_passes(self.verify(jax.jit(edges), line, starts, pieces))
        self.assertEqual(self.artifact(jax.jit(edges), line, starts, pieces)[
            "reference_execution"], "jit")
        for dtype, value in ((jnp.float16, 2048.0), (jnp.bfloat16, 256.0)):
            with self.subTest(dtype=str(dtype)):
                self.assert_passes(self.verify(twice, jnp.array([value, 0.0], dtype)))
                self.assert_passes(self.verify(jax.jit(twice), jnp.array([value, 0.0], dtype)))
        small = jnp.array([-1, 2, -4, 3], dtype=jnp.int8)
        unsigned = jnp.array([2 ** 31, 1, 3], dtype=jnp.uint32)
        for s in (-1, -4, -5):
            with self.subTest(start=s):
                self.assert_passes(self.verify(narrow, self.x, small, unsigned, jnp.int8(s)))
        self.assert_passes(self.verify(empties, self.x, unsigned))
        records = _registered(self.artifact(empties, self.x, unsigned))
        self.assertEqual(sorted(record["tag"] for record in records),
                         ["gather", "gather", "scatter", "scatter"])
        refused = {"apply": (lambda x, i: x.at[:, i].apply(lambda v: v + 1.0), self.x,
                             "a replacement, a sum, a difference, a product, a maximum"),
                   "empty operand": (lambda x, i: lax.scatter(
                       x[:, :0], i[:, None], jnp.ones((2, 0, 4)),
                       lax.ScatterDimensionNumbers((1, 2), (0,), (0,))), self.x,
                       "without entries")}
        for name, (function, operand, text) in refused.items():
            with self.subTest(name=name):
                self.assert_refused(self.verify(function, operand, jnp.array([1, 0]),
                                                checks=("structure",)), text)
        with jax.enable_x64(True):
            huge = jnp.array([1, 0], dtype=jnp.uint64)
            self.assert_refused(self.verify(
                lambda x, k: lax.gather(x, k[:, None], last, (2, 3, 1), mode=modes.CLIP),
                self.x, huge, checks=("structure",)), "other than uint64")
        from types import SimpleNamespace

        import numpy as np
        from torchlean.jaxpr import Unsupported, _Builder

        def declared(dtype):
            return SimpleNamespace(dtype=np.dtype(dtype), shape=())

        for value, dtype in ((np.int64(-1), np.uint64), (np.uint64(2 ** 63), np.int64),
                             (np.int32(-2 ** 31), np.float16), (np.float64(0.1), np.float32)):
            with self.subTest(constant=f"{value!r} as {np.dtype(dtype).name}"):
                with self.assertRaisesRegex(Unsupported, "declared element type"):
                    _Builder("float32").constant(value, declared(dtype))
        self.assertIsInstance(_Builder("float32").constant(np.int64(-2 ** 31), declared(np.int32)),
                              int)

    def test_sqrt_is_ieee_square_root(self):
        # lax.sqrt imports as Op.sqrt at `.raw`, IEEE squareRoot, with no condition on its operand:
        # -0 at -0, NaN below zero and at a NaN, inf at inf. XLA computes binary16 and bfloat16
        # sqrt in binary32 and rounds once, which for a square root is the correctly rounded
        # result, so every format agrees with XLA.
        import numpy as np

        jax, jnp = self.jax, self.jnp
        x = np.array([-0.0, 0.0, -1.0, np.inf, np.nan, 4.0, 2.0, 3.0])
        for dtype in (np.float32, jnp.bfloat16, np.float16):
            with self.subTest(dtype=dtype):
                self.assert_passes(self.verify(jax.lax.sqrt, jnp.asarray(x, dtype=dtype)))

    def test_log2_within_one_xla_ulp(self):
        # The IR's log2 is the binary dictionary's RN(RN(ln x) / RN(ln 2)), which can differ from
        # the correctly rounded log2, and binary16 and bfloat16 compute it in binary32 and round
        # once. On these samples XLA's CPU code (jax 0.12 on x86-64) is within one ulp of it in
        # each format (its bfloat16 log2 is not its binary32 log2 rounded once), so the words are
        # not compared.
        import numpy as np
        import torch

        jnp = self.jnp
        rng = np.random.default_rng(6)
        x = np.concatenate([np.abs(rng.standard_normal(60)) * 8, 2.0 ** np.arange(-6, 7),
                            [0.0, -0.0, -1.0, np.inf, np.nan, 1e-30, 3.0, 0.75]])
        for dtype in (np.float32, jnp.bfloat16, np.float16):
            with self.subTest(dtype=dtype):
                operand = jnp.asarray(x, dtype=dtype)
                report = self.verify(jnp.log2, operand, checks=EXECUTE)
                self.assert_passes(report)
                (actual,) = _outputs(report)
                wanted = torch.from_numpy(np.asarray(jnp.log2(operand)).astype(np.float32))
                _assert_within_ulps(self, actual, wanted.to(actual.dtype), 1)
        self.assertIn("log2", _tags(self.artifact(jnp.log2, x.astype(np.float32))))

    def test_inverse_trigonometric_functions_within_xla_ulps(self):
        # The IR rounds asin, acos, atan and atan2 correctly. On these inputs, XLA's CPU code
        # (jax 0.12 on x86-64) is within 2, 2, 3 and 1 binary32 ulps of the IR and, computing
        # binary16 and bfloat16 in binary32, within one ulp in those formats.
        import numpy as np
        import torch

        jnp = self.jnp
        rng = np.random.default_rng(3)
        unit = np.concatenate([np.linspace(-1, 1, 41), [0.0, -0.0, 1.0, -1.0, 2.0, np.nan]])
        x = np.concatenate([rng.standard_normal(41) * 4, [0.0, -0.0, np.inf, -np.inf, 1e-30,
                                                          np.nan]])
        y = np.concatenate([rng.standard_normal(41) * 4, [-0.0, 1.0, np.inf, 2.0, -np.inf, 0.5]])

        def inverse(u, x, y):
            return jnp.arcsin(u), jnp.arccos(u), jnp.arctan(x), jnp.arctan2(y, x)

        for dtype, ulps in ((np.float32, (2, 2, 3, 1)), (jnp.bfloat16, (1, 1, 1, 1)),
                            (np.float16, (1, 1, 1, 1))):
            with self.subTest(dtype=dtype):
                arguments = [jnp.asarray(value, dtype=dtype) for value in (unit, x, y)]
                report = self.verify(inverse, *arguments, checks=EXECUTE)
                self.assert_passes(report)
                expected = [torch.from_numpy(np.asarray(value).astype(np.float32))
                            for value in inverse(*arguments)]
                for actual, wanted, bound in zip(_outputs(report), expected, ulps):
                    _assert_within_ulps(self, actual, wanted.to(actual.dtype), bound)
        tags = _tags(self.artifact(inverse, unit.astype(np.float32), x.astype(np.float32),
                                   y.astype(np.float32)))
        for tag in ("asin", "acos", "atan", "atan2"):
            self.assertIn(tag, tags)

    def test_real_models_import_without_unsupported_nodes(self):
        jax, jnp = self.jax, self.jnp
        lax = jax.lax
        keys = jax.random.split(jax.random.PRNGKey(0), 8)
        normal = lambda key, shape: 0.3 * jax.random.normal(key, shape, dtype=jnp.float32)
        w1, w2 = normal(keys[0], (8, 16)), normal(keys[1], (16, 4))
        k1, k2 = normal(keys[2], (4, 1, 3, 3)), normal(keys[3], (8, 4, 3, 3))
        w = normal(keys[4], (8, 3))
        qkv, proj = normal(keys[5], (8, 24)), normal(keys[6], (8, 8))

        def mlp(x):
            return jax.nn.log_softmax(jax.nn.relu(x @ w1) @ w2, axis=-1)

        def cnn(x):
            numbers = ("NCHW", "OIHW", "NCHW")
            x = jax.nn.relu(lax.conv_general_dilated(x, k1, (1, 1), ((1, 1), (1, 1)),
                                                     dimension_numbers=numbers))
            x = lax.conv_general_dilated(x, k2, (2, 2), "VALID", dimension_numbers=numbers)
            x = lax.reduce_window(jax.nn.relu(x), -jnp.inf, lax.max, (1, 1, 2, 2), (1, 1, 2, 2),
                                  "VALID")
            return jax.nn.softmax(x.reshape(x.shape[0], -1) @ w, axis=-1)

        def block(x):
            mean = jnp.mean(x, axis=-1, keepdims=True)
            normed = (x - mean) * lax.rsqrt(jnp.mean(jnp.square(x - mean), axis=-1,
                                                      keepdims=True) + 1e-5)
            q, k, v = jnp.split(normed @ qkv, 3, axis=-1)
            q, k, v = (y.reshape(4, 2, 4).transpose(1, 0, 2) for y in (q, k, v))
            scores = jnp.einsum("htd,hsd->hts", q, k) / 2.0
            y = jnp.einsum("hts,hsd->htd", jax.nn.softmax(scores, axis=-1), v)
            return x + y.transpose(1, 0, 2).reshape(4, 8) @ proj

        cases = {"mlp": (mlp, jnp.ones((2, 8))), "cnn": (cnn, jnp.ones((2, 1, 6, 6))),
                 "block": (block, jnp.ones((4, 8)))}
        for name, (function, x) in cases.items():
            with self.subTest(model=name):
                self.assert_passes(self.verify(function, x, checks=("structure", "types")))

    def test_operator_models_import_or_name_their_lane(self):
        # A CNN with kernel dilation and feature groups, a model of sorts and extrema, a
        # classifier's argmax, top_k of scores, prefix sums and a causal mask import.
        jax, jnp = self.jax, self.jnp
        lax = jax.lax
        keys = jax.random.split(jax.random.PRNGKey(1), 4)
        normal = lambda key, shape: 0.3 * jax.random.normal(key, shape, dtype=jnp.float32)
        k1, k2 = normal(keys[0], (4, 2, 3, 3)), normal(keys[1], (8, 2, 3, 3))
        w, w6 = normal(keys[2], (8 * 9, 3)), normal(keys[3], (6, 6))
        numbers = ("NCHW", "OIHW", "NCHW")

        def cnn(x):
            x = lax.conv_general_dilated(x, k1, (1, 1), ((2, 2), (2, 2)), rhs_dilation=(2, 2),
                                         dimension_numbers=numbers)
            x = lax.conv_general_dilated(jax.nn.relu(x), k2, (1, 1), ((1, 1), (2, 2)),
                                         rhs_dilation=(1, 2), feature_group_count=2,
                                         dimension_numbers=numbers)
            x = lax.reduce_window(jax.nn.relu(x), -jnp.inf, lax.max, (1, 1, 2, 2), (1, 1, 2, 2),
                                  "VALID")
            return x.reshape(x.shape[0], -1) @ w

        def ranks(x):
            h = jnp.tanh(x @ w6)
            s = jnp.sort(h, axis=-1)
            return (s - jnp.max(h, axis=-1, keepdims=True), jnp.floor(s * 4),
                    lax.expm1(jnp.min(h, axis=0)))

        def causal(x):
            i = jnp.arange(x.shape[0])
            mask = jnp.logical_and(i[None, :] <= i[:, None], i[:, None] >= 0)
            return jax.nn.softmax(jnp.where(mask, x @ x.T, -jnp.inf), axis=-1) @ x

        rows = jnp.linspace(-1.0, 1.0, 24, dtype=jnp.float32).reshape(4, 6)
        self.assert_passes(self.verify(cnn, jnp.ones((2, 2, 6, 6)), checks=("structure", "types")))
        self.assert_passes(self.verify(ranks, rows, checks=EXECUTE))
        self.assert_passes(self.verify(lambda x: jnp.argmax(x @ w6, axis=-1), rows,
                                       checks=EXECUTE))
        self.assert_passes(self.verify(causal, jnp.ones((4, 8)), checks=EXECUTE))
        self.assert_passes(self.verify(lambda x: lax.top_k(x @ w6, 2), rows, checks=EXECUTE))
        self.assert_passes(self.verify(lambda x: jnp.cumsum(x @ w6, axis=-1), rows,
                                       checks=EXECUTE))

    def test_mean_over_an_empty_axis_is_nan(self):
        # jnp.mean is a sum and a division by the count, so over an empty axis it is 0 / 0, NaN,
        # as in JAX. The IR mean, which the PyTorch capture refuses there, is not used.
        jnp = self.jnp
        empty = jnp.ones((3, 0), dtype=jnp.float32)
        self.assert_passes(self.verify(lambda x: jnp.mean(x, axis=1), empty))
        self.assertNotIn("mean", _tags(self.artifact(lambda x: jnp.mean(x, axis=1), empty)))

    def test_comparisons_and_selections(self):
        # lax comparisons and is_finite give Boolean steps, the logical operators of Booleans
        # stay logical, select_n is select with its cases in the other order, and
        # lax.clamp(lo, x, hi) is the IR clamp(x, lo, hi), a maximum and then a minimum. Integer
        # comparisons read the values, integer add, sub and mul wrap modulo 2 ** width, and
        # integer max, min and clamp follow the type's order.
        jnp, lax = self.jnp, self.jax.lax
        x = jnp.array([[jnp.nan, -0.0, 0.0, 1.5], [-jnp.inf, jnp.inf, -2.0, 30.0]],
                      dtype=jnp.float32)
        y = jnp.array([[1.0, 0.0, -0.0, 1.5], [-jnp.inf, 2.0, jnp.nan, 30.0]], dtype=jnp.float32)

        def boolean(x, y):
            p, q = x > 0, y < 1
            return (x == y, x != y, x < y, x <= y, x > y, x >= 0.25, jnp.isfinite(x),
                    jnp.logical_and(p, q), p | q, p ^ q, ~p, jnp.where(x > y, x, y),
                    lax.clamp(-1.0, x, 2.0))

        self.assert_passes(self.verify(boolean, x, y))
        # reduce_or, reduce_and and reduce_xor of Booleans: a binary64 count compared with zero,
        # its parity for xor. An empty fiber gives false, true and false.
        flags = jnp.array([[True, False, True], [False, False, False]])
        empty = jnp.zeros((2, 0), dtype=bool)
        self.assert_passes(self.verify(
            lambda b, e: (jnp.any(b, axis=1), jnp.all(b, axis=0), jnp.all(b),
                          lax.reduce_xor(b, (1,)), jnp.any(e, axis=1), jnp.all(e, axis=1),
                          lax.reduce_xor(e, (1,))), flags, empty))
        tags = set(_tags(self.artifact(boolean, x, y)))
        for tag in ("eq", "isFinite", "logicalAnd", "logicalNot", "select", "max", "min"):
            self.assertIn(tag, tags)
        def integers(i, j):
            one = jnp.ones((), dtype=i.dtype)
            return (i == j, i != j, i < j, i <= 2, i > j, i >= j, i + j, i - j, i * j,
                    jnp.maximum(i, j), jnp.minimum(i, j), lax.clamp(one, i, j),
                    jnp.where(i < j, i, j))

        for dtype in (jnp.int8, jnp.uint8, jnp.int32):
            info = jnp.iinfo(dtype)
            i = jnp.array([info.max, info.min, 5, 0, 2, 7], dtype=dtype)
            j = jnp.array([1, info.max, 5, 3, 2, 1], dtype=dtype)
            with self.subTest(dtype=str(dtype)):
                self.assert_passes(self.verify(integers, i, j))

    def test_unsupported_primitives_are_refused(self):
        jnp, lax = self.jnp, self.jax.lax
        cases = {"integer argmax": (lambda x: jnp.argmax(x.astype(jnp.int32), axis=1),
                                    "only floating operands"),
                 "argsort": (lambda x: jnp.argsort(x, axis=1), "sorting axis's index"),
                 "cumlogsumexp": (lambda x: lax.cumlogsumexp(x, 1),
                                  "unsupported primitive cumlogsumexp"),
                 "integer cumsum": (lambda x: jnp.cumsum(x.astype(jnp.int32), axis=1),
                                    "only floating operands"),
                 "integer top_k": (lambda x: lax.top_k(x.astype(jnp.int32), 2)[1],
                                   "only floating operands"),
                 "accuracy request": (lambda x: lax.exp(x, accuracy=lax.Tolerance(
                     atol=1e-3, rtol=0.0, ulps=0)), "accuracy request"),
                 "round away from zero": (lax.round, "TO_NEAREST_EVEN")}
        for name, (function, text) in cases.items():
            with self.subTest(name=name):
                self.assert_refused(self.verify(function, self.x, checks=("structure",)), text)

    def test_windowed_reductions_fold_the_init_value(self):
        # reduce_window_max, _min, _sum and reduce_window with a mul or a max body: the operand
        # padded with the init value (and base dilation's init values between entries), each
        # window folded from the init value in row-major cell order, max and min the IEEE maximum
        # and minimum (+0 above -0 whatever their order, a NaN propagated, -inf for a window of
        # padding only), integers wrapping. The reference is that fold, entry by entry, in
        # NumPy's binary32 arithmetic and in Python integers modulo 2^32.
        import itertools
        import math

        import numpy as np

        jnp, lax = self.jnp, self.jax.lax

        def reference(x, init, combine, window, strides, padding, base, dilation):
            x = np.asarray(x)
            dilated = [(n - 1) * b + 1 if n else 0 for n, b in zip(x.shape, base)]
            grid = np.full(dilated, init, dtype=object)
            grid[tuple(slice(None, None, b) for b in base)] = x.astype(object)
            padded = np.full([n + lo + hi for n, (lo, hi) in zip(dilated, padding)], init,
                             dtype=object)
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

        f32 = np.float32
        add = lambda a, b: float(f32(a) + f32(b))
        mul = lambda a, b: float(f32(a) * f32(b))
        x = np.arange(60, dtype=np.float32).reshape(2, 5, 6) / 7 - 4
        x[0, 0, :4] = [-0.0, 0.0, 0.0, -0.0]
        x[1, 2, 3], x[1, 4, 0], x[0, 3, 5] = np.nan, -np.inf, np.inf
        window, strides = (1, 2, 3), (1, 2, 1)
        padding, base, dilation = ((0, 0), (1, 2), (2, 0)), (1, 1, 2), (1, 2, 1)

        def windows(x):
            fold = lambda init, body, **k: lax.reduce_window(
                x, jnp.float32(init), body, window, strides, padding, base_dilation=base,
                window_dilation=dilation, **k)
            return (fold(-np.inf, lax.max), fold(np.inf, lax.min), fold(0, lax.add),
                    fold(1, lax.mul), fold(0.5, lax.max),
                    lax.reduce_window(x, -jnp.inf, lax.max, (1, 1, 3), (1, 1, 3),
                                      ((0, 0), (0, 0), (4, 4))))

        document = self.artifact(windows, x)
        self.assertNotIn("pool", _tags(document))
        # jax keeps a base- or window-dilated monoid as the general reduce_window. Without
        # dilation the monoids are reduce_window_max, _min and _sum.
        primitives = str(self.jax.make_jaxpr(windows)(x))
        self.assertIn("reduce_window[", primitives)
        self.assertIn("reduce_window_max", primitives)
        # (an init value that is a Python number, which jax compares with the identity).
        plain = lambda x: (lax.reduce_window(x, np.inf, lax.min, window, strides, padding),
                           lax.reduce_window(x, 0.0, lax.add, window, strides, padding))
        names = str(self.jax.make_jaxpr(plain)(x))
        self.assertIn("reduce_window_min", names)
        self.assertIn("reduce_window_sum", names)
        report = self.verify(plain, x, checks=EXECUTE)
        self.assert_passes(report)
        flat = np.asarray([float(v) for v in x.reshape(-1)], dtype=object).reshape(x.shape)
        for output, (init, combine) in zip(_outputs(report), ((math.inf, _ieee_minimum),
                                                               (0.0, add))):
            _same(self, output.flatten().tolist(),
                  reference(flat, init, combine, window, strides, padding, (1, 1, 1), (1, 1, 1)))
        report = self.verify(windows, x, checks=EXECUTE)
        self.assert_passes(report)
        flat = np.asarray([float(v) for v in x.reshape(-1)], dtype=object).reshape(x.shape)
        general = (window, strides, padding, base, dilation)
        cases = [(-math.inf, _ieee_maximum, general), (math.inf, _ieee_minimum, general),
                 (0.0, add, general), (1.0, mul, general), (0.5, _ieee_maximum, general),
                 (-math.inf, _ieee_maximum, ((1, 1, 3), (1, 1, 3), ((0, 0), (0, 0), (4, 4)),
                                             (1, 1, 1), (1, 1, 1)))]
        for output, (init, combine, shape) in zip(_outputs(report), cases):
            _same(self, output.flatten().tolist(), reference(flat, init, combine, *shape))
        # The first window of each row of the last case holds padding only.
        self.assertEqual(_outputs(report)[5].flatten().tolist()[::4], [-math.inf] * 10)
        zeros = np.array([-0.0, 0.0, 0.0, -0.0], dtype=np.float32)
        pairs = lambda z: (lax.reduce_window(z, -jnp.inf, lax.max, (2,), (2,), "VALID"),
                           lax.reduce_window(z, jnp.inf, lax.min, (2,), (2,), "VALID"))
        report = self.verify(pairs, zeros, checks=EXECUTE)
        self.assert_passes(report)
        _same(self, [v for t in _outputs(report) for v in t.tolist()], [0.0, 0.0, -0.0, -0.0])
        # A negative padding crops the operand, and cropping past it leaves no window.
        crops = lambda z: (lax.reduce_window(z, -jnp.inf, lax.max, (1, 2), (1, 1),
                                             ((0, 0), (-2, 1))),
                           lax.reduce_window(z, 0.0, lax.add, (1, 1), (1, 1), ((0, 0), (-6, 0))))
        report = self.verify(crops, x[0], checks=EXECUTE)
        self.assert_passes(report)
        _same(self, _outputs(report)[0].flatten().tolist(),
              reference(np.asarray([float(v) for v in x[0].reshape(-1)], dtype=object)
                        .reshape(5, 6)[:, 2:], -math.inf, _ieee_maximum, (1, 2), (1, 1),
                        ((0, 0), (0, 1)), (1, 1), (1, 1)))
        self.assertEqual(tuple(_outputs(report)[1].shape), (5, 0))
        # lax.reduce with a lambda body and the identity init is the IR reduce.
        report = self.verify(lambda z: lax.reduce(z, 0.0, lambda a, b: lax.add(a, b), (1,)),
                             x[0], checks=EXECUTE)
        self.assert_passes(report)
        sums = []
        for row in x[0]:
            total = f32(0)
            for value in row:
                total = f32(total + value)
            sums.append(float(total))
        _same(self, _outputs(report)[0].flatten().tolist(), sums)
        # Integers: the type's order and wrapping sums and products.
        i = (np.arange(24, dtype=np.int64).reshape(4, 6) * 0x2345679 - 2 ** 31).astype(np.int32)
        wrap = lambda value: (value + 2 ** 31) % 2 ** 32 - 2 ** 31

        def integers(i):
            return (lax.reduce_window(i, jnp.int32(-2 ** 31), lax.max, (2, 3), (1, 2),
                                      ((1, 0), (0, 1))),
                    lax.reduce_window(i, jnp.int32(0), lax.add, (2, 2), (2, 1), "VALID"),
                    lax.reduce_window(i, jnp.int32(1), lax.mul, (1, 3), (1, 1), "VALID"))

        report = self.verify(integers, i, checks=EXECUTE)
        self.assert_passes(report)
        ints = np.asarray([int(v) for v in i.reshape(-1)], dtype=object).reshape(i.shape)
        expected = [reference(ints, -2 ** 31, max, (2, 3), (1, 2), ((1, 0), (0, 1)), (1, 1),
                              (1, 1)),
                    reference(ints, 0, lambda a, b: wrap(a + b), (2, 2), (2, 1), ((0, 0), (0, 0)),
                              (1, 1), (1, 1)),
                    reference(ints, 1, lambda a, b: wrap(a * b), (1, 3), (1, 1), ((0, 0), (0, 0)),
                              (1, 1), (1, 1))]
        for words, wanted in zip(_words(report), expected):
            self.assertEqual(words, [v % 2 ** 32 for v in wanted])

    def test_reduce_precision_rounds_the_mantissa_and_clamps_the_exponent(self):
        # StableHLO's reduce_precision(x, e, m): the mantissa rounded to m bits, to nearest even,
        # at the operand's own exponent (the operand's subnormal spacing for its subnormals),
        # then an infinity of the sign where the exponent exceeds the range of e bits and a zero of
        # the sign below its normal range, and a NaN word unchanged. The reference rounds exact
        # rationals.
        import math
        from fractions import Fraction

        import numpy as np

        jnp, lax = self.jnp, self.jax.lax
        fields = {"float32": (8, 23, 127), "bfloat16": (8, 7, 127), "float16": (5, 10, 15)}

        def reference(value, name, e, m):
            width, fraction, bias = fields[name]
            if value != value or value == 0 or math.isinf(value):
                return value
            magnitude = Fraction(abs(value))
            exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
            if Fraction(2) ** exponent > magnitude:
                exponent -= 1
            if m < fraction:
                quantum = Fraction(2) ** (max(exponent, 1 - bias) - m)
                q, r = divmod(magnitude, quantum)
                q = int(q) + (1 if r > quantum / 2 or r == quantum / 2 and int(q) % 2 else 0)
                magnitude = q * quantum
                if magnitude >= 2 ** (bias + 1):
                    return math.copysign(math.inf, value)
            if e < width:
                largest = 2 ** (e - 1) - 1
                if magnitude >= Fraction(2) ** (largest + 1):
                    return math.copysign(math.inf, value)
                if magnitude < Fraction(2) ** (1 - largest):
                    return math.copysign(0.0, value)
            return math.copysign(float(magnitude), value)

        rng = np.random.default_rng(11)
        for dtype, settings in ((np.float32, ((5, 10), (8, 7), (4, 3), (1, 2), (11, 30),
                                              (6, 23), (8, 1))),
                                (jnp.bfloat16, ((5, 3), (3, 7), (8, 2))),
                                (np.float16, ((3, 2), (5, 4), (2, 10)))):
            name = np.dtype(dtype).name
            width, fraction, bias = fields[name]
            values = list(rng.standard_normal(40) * 2.0 ** rng.integers(-bias - fraction,
                                                                         bias, 40))
            for e, m in settings:
                largest = 2 ** (min(e, width) - 1) - 1
                for power in (largest + 1, 1 - largest, -bias + 1, -bias - fraction + 2):
                    for step in (-2.0 ** (-m - 2), -2.0 ** (-m - 1), 0.0, 2.0 ** (-m - 1)):
                        values.append(2.0 ** power * (1 + step))
            values += [0.0, -0.0, math.inf, -math.inf, 1.0, -3.5]
            operand = np.asarray(values, dtype=dtype)
            words = _float_words(operand, dtype)
            unsigned = {2: np.uint16, 4: np.uint32}[np.dtype(dtype).itemsize]
            nans = np.asarray({"float32": [0x7fc00001, 0x7f800001, 0xffc12345],
                               "bfloat16": [0x7fc1, 0x7f81, 0xffc3],
                               "float16": [0x7e01, 0x7c01, 0xfe23]}[name],
                              dtype=unsigned).view(dtype)
            operand = jnp.asarray(np.concatenate([operand, nans]))
            function = lambda x: tuple(lax.reduce_precision(x, e, m) for e, m in settings)
            with self.subTest(dtype=name):
                report = self.verify(function, operand, checks=EXECUTE)
                self.assert_passes(report)
                for (e, m), got in zip(settings, _words(report)):
                    expected = _float_words([reference(_float_of_word(w, dtype), name, e, m)
                                             for w in words], dtype)
                    self.assertEqual(got[:len(words)], expected, (e, m))
                    self.assertEqual(got[len(words):], _float_words(nans, dtype), (e, m))
        self.assert_refused(self.verify(lambda x: lax.reduce_precision(x, 8, 0),
                                        self.x, checks=("structure",)), "mantissa_bits = 0")

    def test_cube_root_and_nextafter(self):
        # cbrt is IEEE rootn(x, 3), rounded once in the operand's own format (the reference
        # rounds the exact cube root, certified by cubes of the midpoints), -0 at -0 and the
        # infinities at the infinities. nextafter is C's: a NaN where either operand is one, y
        # where x == y, else the adjacent value toward y (subnormals included), which the
        # reference computes on the words.
        import math

        import numpy as np

        jnp, lax = self.jnp, self.jax.lax
        rng = np.random.default_rng(12)
        for dtype in (np.float32, jnp.bfloat16, np.float16):
            name = np.dtype(dtype).name
            tiny = _float_of_word(1, dtype)
            values = np.concatenate([rng.standard_normal(30) * 4.0 ** rng.integers(-8, 8, 30),
                                     [tiny, -tiny, 3 * tiny, 8.0, -27.0, 0.0, -0.0, math.inf,
                                      -math.inf, math.nan, 1e-3, 1000.0]])
            x = np.asarray(values, dtype=dtype)
            y = np.asarray(np.concatenate([rng.standard_normal(30), [0.0, 0.0, -1.0, 8.0, -27.0,
                                                                     -0.0, 0.0, 0.0, math.inf,
                                                                     1.0, math.nan, 1000.0]]),
                           dtype=dtype)
            with self.subTest(dtype=name):
                report = self.verify(lambda x, y: (lax.cbrt(x), lax.nextafter(x, y)),
                                     jnp.asarray(x), jnp.asarray(y), checks=EXECUTE)
                self.assert_passes(report)
                roots, adjacent = _words(report)
                xs = [_float_of_word(w, dtype) for w in _float_words(x, dtype)]
                ys = [_float_of_word(w, dtype) for w in _float_words(y, dtype)]
                expected = _float_words([_cube_root(v, dtype) for v in xs], dtype)
                for got, want, v in zip(roots, expected, xs):
                    if v != v:
                        self.assertTrue(math.isnan(_float_of_word(got, dtype)))
                    else:
                        self.assertEqual(got, want, v)
                width = np.dtype(dtype).itemsize * 8
                for got, a, b, word in zip(adjacent, xs, ys, _float_words(x, dtype)):
                    value = _float_of_word(got, dtype)
                    if a != a or b != b:
                        self.assertTrue(math.isnan(value))
                    elif a == b:
                        self.assertEqual(got, _float_words([b], dtype)[0])
                    elif a == 0:
                        self.assertEqual(got, 1 | (1 << (width - 1) if b < 0 else 0))
                    else:
                        self.assertEqual(got, word + (1 if (a < b) == (a > 0) else -1), (a, b))
        self.assertIn("cbrt", _tags(self.artifact(lax.cbrt, self.x)))

    def test_integer_bit_operators_act_on_words(self):
        # and, or, xor and not of integers, the shifts (an amount of the width or more, a negative
        # one read as its unsigned word among them, shifts every bit out: zero, or the top bit in
        # every position for the arithmetic shift, even of an unsigned type), population_count,
        # clz and mulhi (the floor of the exact product over 2^width). The reference is Python
        # integer arithmetic on the words.
        import numpy as np

        jnp, lax = self.jnp, self.jax.lax
        rng = np.random.default_rng(13)
        for dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32):
            info, width = np.iinfo(dtype), np.dtype(dtype).itemsize * 8
            x = rng.integers(info.min, int(info.max) + 1, 40, dtype=np.int64)
            x[:4] = [info.min, info.max, 0, -1 if info.min < 0 else 1]
            k = rng.integers(0, width + 3, 40)
            k[:4] = [width, width + 1, -1 if info.min < 0 else 255, 0]
            x, k = x.astype(dtype), k.astype(dtype)
            mask = 1 << width
            word = lambda v: int(v) % mask
            signed = lambda w: w - mask if info.min < 0 and w >= mask // 2 else w

            def bits(x, k):
                return (x & k, x | k, x ^ k, ~x, lax.shift_left(x, k),
                        lax.shift_right_arithmetic(x, k), lax.shift_right_logical(x, k),
                        lax.population_count(x), lax.clz(x), lax.mulhi(x, k))

            expected = [[] for _ in range(10)]
            for a, b in zip(map(word, x), map(word, k)):
                top = (a >> (width - 1)) & 1
                for out, value in zip(expected, (
                        a & b, a | b, a ^ b, ~a, 0 if b >= width else a << b,
                        (mask - 1 if top else 0) if b >= width else
                        ((a - mask if top else a) >> b),
                        0 if b >= width else a >> b, bin(a).count("1"), width - a.bit_length(),
                        (signed(a) * signed(b)) >> width)):
                    out.append(value % mask)
            with self.subTest(dtype=np.dtype(dtype).name):
                report = self.verify(bits, jnp.asarray(x), jnp.asarray(k), checks=EXECUTE)
                self.assert_passes(report)
                for name, got, want in zip(("and", "or", "xor", "not", "shl", "sra", "srl",
                                            "popcnt", "clz", "mulhi"), _words(report), expected):
                    self.assertEqual(got, want, name)
        tags = _tags(self.artifact(lambda x: (x & 3, lax.clz(x)), jnp.arange(4, dtype=np.int32)))
        self.assertIn("bitwiseAnd", tags)
        self.assertIn("countLeadingZeros", tags)

    def test_bitcast_between_equal_widths_keeps_every_bit(self):
        # bitcast_convert_type between types of one width reads each word at the new type, NaN
        # payloads and signaling NaNs included. Other widths are refused.
        import numpy as np

        jnp, lax = self.jnp, self.jax.lax
        words = np.asarray([0x3f800000, 0x80000000, 0x7fc00001, 0x7f800001, 0xffffffff, 1],
                           dtype=np.uint32)
        floats = words.view(np.float32)
        halves = np.asarray([0x3c00, 0x8000, 0x7e01, 0x7c01, 0xffff, 1], dtype=np.uint16)

        def casts(f, h):
            as_int = lax.bitcast_convert_type(f, jnp.int32)
            return (as_int, lax.bitcast_convert_type(f, jnp.uint32),
                    lax.bitcast_convert_type(as_int, jnp.float32),
                    lax.bitcast_convert_type(h, jnp.float16),
                    lax.bitcast_convert_type(lax.bitcast_convert_type(h, jnp.bfloat16),
                                             jnp.int16))

        report = self.verify(casts, jnp.asarray(floats), jnp.asarray(halves), checks=EXECUTE)
        self.assert_passes(report)
        got = _words(report)
        for output in got[:3]:
            self.assertEqual(output, [int(w) for w in words])
        for output in got[3:]:
            self.assertEqual(output, [int(w) for w in halves])
        self.assert_refused(self.verify(lambda f: lax.bitcast_convert_type(f, jnp.int16),
                                        jnp.asarray(floats), checks=("structure",)),
                            "one word width")

    def test_triangular_solve_cholesky_and_symmetric_product(self):
        # triangular_solve is substitution on op(a) x = b or x op(a) = b, reading only the used
        # triangle (and no diagonal for unit_diagonal): a lower system forward, an upper one
        # backward, row i (b_i - sum_j op(a)_ij x_j) / op(a)_ii with the sum a left fold from +0
        # over the rows solved before it, in substitution order. A right-side equation is solved
        # transposed. cholesky of the symmetrized operand is the left-looking column algorithm,
        # NaN where a pivot is not positive, under lax.linalg.cholesky's zeros above the diagonal.
        # symmetric_product is alpha (a a^T) + beta c. The reference is that arithmetic in NumPy's
        # binary32, operation by operation.
        import itertools
        import math

        import numpy as np

        jnp, lax = self.jnp, self.jax.lax
        f32 = np.float32
        rng = np.random.default_rng(14)
        n = 4
        a = rng.standard_normal((2, n, n)).astype(f32)
        a[:, range(n), range(n)] += 3
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
        for got, (left, lower, transpose, unit) in zip(_outputs(report), cases):
            want = np.stack([reference(a[k], (b_left if left else b_right)[k], left, lower,
                                       transpose, unit) for k in range(2)])
            _same(self, got.flatten().tolist(), [float(v) for v in want.reshape(-1)],
                  (left, lower, transpose, unit))
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
        d = np.asarray([2.0, 4.0, 0.5], dtype=f32)
        solve = lambda b: lax.custom_linear_solve(lambda x: d * x, b, lambda mv, r: r / d)
        rhs = np.asarray([1.0, 3.0, -2.0], dtype=f32)
        report = self.verify(solve, jnp.asarray(rhs), checks=EXECUTE)
        self.assert_passes(report)
        _same(self, _outputs(report)[0].tolist(), [float(f32(r / s)) for r, s in zip(rhs, d)])

    def test_rows_without_an_executable_meaning_are_refused(self):
        # Special functions without a FloatLib function, complex values, factorizations whose
        # outputs the API leaves to the implementation, lax.reduce with a non-identity init (its
        # documentation calls the result undefined) and a reduce_window body that is not one
        # monoid operation are refused, each by the general unsupported-primitive rule or the
        # rule's own reason.
        import numpy as np

        jnp, lax = self.jnp, self.jax.lax
        x = jnp.asarray(np.linspace(0.1, 0.9, 6, dtype=np.float32).reshape(2, 3))
        m = jnp.asarray(np.eye(3, dtype=np.float32) * 2)
        cases = {"lgamma": (lax.lgamma, x), "digamma": (lax.digamma, x),
                 "erf_inv": (lax.erf_inv, x), "erfc": (lax.erfc, x),
                 "igamma": (lambda x: lax.igamma(x, x), x),
                 "zeta": (lambda x: lax.zeta(x + 1, x), x), "bessel_i0e": (lax.bessel_i0e, x),
                 "complex": (lambda x: lax.complex(x, x).real, x),
                 "lu": (lambda m: lax.linalg.lu(m)[0], m), "qr": (lambda m: lax.linalg.qr(m)[0], m),
                 "eigh": (lambda m: lax.linalg.eigh(m)[1], m),
                 "reduce": (lambda x: lax.reduce(x, jnp.float32(0.5), lax.max, (1,)), x),
                 "complex input": (lambda z: z * 2, jnp.asarray(np.ones(3, dtype=np.complex64))),
                 "reduce_window": (lambda x: lax.reduce_window(x, jnp.float32(0), lax.sub, (1, 2),
                                                               (1, 1), "VALID"), x)}
        for name, (function, operand) in cases.items():
            with self.subTest(name=name):
                report = self.verify(function, operand, checks=("structure",))
                self.assert_refused(report, {"reduce_window": "reduce_window",
                                             "reduce": "body's identity",
                                             "complex input": "complex values",
                                             "complex": "complex values",
                                             "lu": "multiple primitive outputs",
                                             "qr": "multiple primitive outputs",
                                             "eigh": "multiple primitive outputs"}.get(
                                                 name, "unsupported primitive"))


@unittest.skipUnless(CHECKER and HAS_TORCH, "needs the built verify executable and PyTorch")
class PyTorchCompilationTests(CheckerCase):
    # Lean's graph proofs do not cover Dynamo's guards, Python descriptors or live
    # Parameter bindings. These checks pin that compiler boundary and leave the algebra to the
    # proofs.
    def test_model_decorator_preserves_training_and_specializations(self):
        import torch

        @torchlean.compile(checker=CHECKER)
        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 1)
                self.scale = 2.0

            def forward(self, x):
                y = self.linear(x) * self.scale
                return y if self.training else y + 1

        model = Network()
        parameters = tuple(model.parameters())
        keys = tuple(model.state_dict())
        optimizer = torch.optim.SGD(parameters, lr=0.1)
        x = torch.ones(1, 2)
        initial = model(x).detach()
        model(x).sum().backward()
        optimizer.step()
        self.assertFalse(torch.equal(model(x).detach(), initial))
        self.assertEqual(len(model.torchlean.graphs), 1)
        self.assertEqual(tuple(model.state_dict()), keys)
        self.assertTrue(all(a is b for a, b in zip(parameters, model.parameters())))
        steps = _steps(model.torchlean.artifact)
        self.assertEqual(len([step for step in steps if "input" in step]), 3)
        model.scale = 3.0
        model(x)
        model.eval()
        model(x)
        model.train()
        model(x)
        self.assertEqual(len(model.torchlean.graphs), 3)
        self.assertIsNone(Network().torchlean.report)
        # Backend identity must not consume a Dynamo cache slot per instance.
        for _ in range(12):
            other = Network()
            other(x)
            self.assert_passes(other.torchlean.report)
            self.assertEqual(len(other.torchlean.graphs), 1)

    def test_existing_module_copy_keeps_its_own_parameters(self):
        import inspect
        import torch

        model = torch.nn.Linear(2, 1)
        signature = inspect.signature(model.forward)
        self.assertIs(torchlean.compile(model, checker=CHECKER), model)
        self.assertEqual(inspect.signature(model.forward), signature)
        x = torch.ones(1, 2)
        original = model(x).detach()
        copied = copy.deepcopy(model)
        self.assertIsNone(copied.torchlean.report)
        copied.bias.data.add_(2)
        torch.testing.assert_close(copied(x), original + 2)
        self.assert_passes(copied.torchlean.report)
        torch.testing.assert_close(model(x), original)

    def test_hooks_are_captured_and_later_changes_invalidate_the_graph(self):
        import torch

        @torchlean.compile(checker=CHECKER)
        class Square(torch.nn.Module):
            def forward(self, inputs):
                return inputs * inputs

        for model in (Square(), torchlean.compile(torch.nn.Linear(2, 2), checker=CHECKER)):
            with self.subTest(model=type(model).__name__):
                inputs = torch.ones(3, 2)
                initial = model(inputs).detach()
                hook = model.register_forward_hook(lambda _, args, output: output + 1)
                torch.testing.assert_close(model(inputs), initial + 1)
                self.assertEqual(len(model.torchlean.graphs), 2)
                hook.remove()
                torch.testing.assert_close(model(inputs), initial)
                model.register_forward_hook(lambda _, args, output: torch.sort(output).values)
                with self.assertRaises(Exception):
                    model(inputs)
                self.assertEqual(len(model.torchlean.graphs), 2)

    def test_call_interface_aliases_and_metadata(self):
        import torch

        @torchlean.compile(checker=CHECKER)
        def function(values, *, bias, scale=2):
            y = values["x"] * scale + bias
            return {"value": y, "square": (y * y,)}

        x = torch.ones(2, requires_grad=True)
        result = function({"x": x}, bias=x)
        result["value"].sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 3))
        self.assert_passes(function.torchlean.report)
        copied = function.torchlean.report
        copied["ok"] = False
        self.assertTrue(function.torchlean.report["ok"])
        self.assert_passes(torchlean.verify(function, {"x": x}, example_kwargs={"bias": x},
                                           checks=COMPARE, checker=CHECKER))

        @torchlean.compile(checker=CHECKER)
        def alias(a, b):
            return a + b if a is b else a - b

        torch.testing.assert_close(alias(x, x), x + x)
        torch.testing.assert_close(alias(x, x.clone()), x - x)
        self.assertEqual(len(alias.torchlean.graphs), 2)

    def test_no_unchecked_compiler_fallback(self):
        import torch

        @torchlean.compile(checker=CHECKER)
        def unsupported(x):
            return torch.sort(x).values

        with self.assertRaises(Exception):
            unsupported(torch.ones(2))
        self.assertEqual(unsupported.torchlean.graphs, ())

        @torchlean.compile(checker=CHECKER)
        def no_tensor_graph(x):
            return x.shape[0]

        with self.assertRaises(NotImplementedError):
            no_tensor_graph(torch.ones(2))
        self.assertIsNone(no_tensor_graph.torchlean.report)

        @torchlean.compile(checker=CHECKER)
        def square(x):
            return x * x

        external = torch.compile(square, backend="eager", fullgraph=True)
        with self.assertRaises(Exception):
            external(torch.ones(2))
        self.assertIsNone(square.torchlean.report)

    def test_registered_pytree_selects_pytorch(self):
        import torch

        class Box:
            def __init__(self, value):
                self.value = value

        torch.utils._pytree.register_pytree_node(
            Box, lambda box: ([box.value], None), lambda values, _: Box(values[0]))

        @torchlean.compile(checker=CHECKER)
        def square(box):
            return box.value * box.value

        torch.testing.assert_close(square(Box(torch.full((3,), 2.))), torch.full((3,), 4.))
        self.assert_passes(square.torchlean.report)


@unittest.skipUnless(CHECKER and HAS_JAX, "needs the built verify executable and JAX")
class JAXCompilationTests(CheckerCase):
    def test_registered_pytree_selects_jax(self):
        import jax
        import jax.numpy as jnp

        @jax.tree_util.register_pytree_node_class
        class Box:
            def __init__(self, value):
                self.value = value

            def tree_flatten(self):
                return (self.value,), None

            @classmethod
            def tree_unflatten(cls, _, values):
                return cls(values[0])

        @torchlean.compile(checker=CHECKER)
        def square(box):
            return box.value * box.value

        self.assertTrue(bool(jnp.all(square(Box(jnp.full(3, 2.))) == 4)))
        self.assert_passes(square.torchlean.report)

    def test_pytree_parameters_kwargs_and_native_gradients(self):
        import jax
        import jax.numpy as jnp

        @torchlean.compile(checker=CHECKER)
        def function(params, x, *, bias):
            y = params["scale"] * x + bias
            return {"value": y, "square": (y * y,)}

        x = jnp.ones(2)
        params = {"scale": x}
        self.assertTrue(bool(jnp.all(function(params, x, bias=x)["value"] == 2)))
        function({"scale": x * 2}, x, bias=x)
        self.assertEqual(len(function.torchlean.graphs), 1)
        gradient = jax.grad(lambda p: function(p, x, bias=x)["value"].sum())(params)
        self.assertTrue(bool(jnp.all(gradient["scale"] == 1)))
        function({"scale": jnp.ones(3)}, jnp.ones(3), bias=jnp.ones(3))
        self.assertEqual(len(function.torchlean.graphs), 2)
        self.assert_passes(function.torchlean.report)
        self.assert_passes(torchlean.verify(function, params, x, example_kwargs={"bias": x},
                                           checks=COMPARE, checker=CHECKER))

    def test_static_specialization_and_unsupported_graph(self):
        import jax
        import jax.numpy as jnp

        @torchlean.compile(checker=CHECKER)
        def evolve(x, steps=2):
            for _ in range(steps):
                x = x * x + 1
            return x

        self.assertTrue(bool(jnp.all(evolve(jnp.ones(2)) == 5)))
        self.assertTrue(bool(jnp.all(evolve(jnp.ones(2), steps=1) == 2)))
        self.assertEqual(len(evolve.torchlean.graphs), 2)

        @torchlean.compile(checker=CHECKER)
        def unsupported(x):
            return jax.lax.cumlogsumexp(x, 0)

        with self.assertRaises(Exception):
            unsupported(jnp.ones(2))
        self.assertEqual(unsupported.torchlean.graphs, ())


if __name__ == "__main__":
    unittest.main()
