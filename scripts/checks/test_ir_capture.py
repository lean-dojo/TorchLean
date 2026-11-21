"""Capture framework computations and check them with ``verify ir``.

Each case captures a tiny computation, then requests structure, type and execution checks
from the built ``verify`` executable. Operations whose IR arithmetic is PyTorch's or XLA's also
compare interchange words with the frontend, in binary32, bfloat16, binary16 and integers. Where
XLA's compiled CPU code departs from JAX's meaning (reciprocals of broadcast and constant
divisors, jax.jit's rewrites, subnormal flushing, its own rsqrt and power), the cases pin the IR's
IEEE result and check that ``compare`` fails and lists the behaviour in ``consistent_with`` only
when its emulation of XLA's code reproduces the reference, and lists nothing for importer bugs
placed next to those operations, including those that coincide with jax.jit's rewrites.
Cases skip when the checker has not been built or a frontend is not installed, and the compilation
of a Lean export outside Lake's environment (``lake test``, ``lake env``).

The cases are this script's ``OperatorTableTests``, ``StateExportTests``,
``PyTorchCaptureTests`` and ``PyTorchCompilationTests``, ``PyTorchGeneralOperatorTests`` in
``ir_capture_pytorch_operators.py``, ``PyTorchFlowTests`` in ``ir_capture_pytorch_flow.py``,
``JaxCaptureTests`` and ``JAXCompilationTests`` in ``ir_capture_jax.py``, ``JaxComplexTests`` in
``ir_capture_jax_complex.py``, ``JaxGeneralOperatorTests`` in ``ir_capture_jax_operators.py``,
``JaxContractionTests`` in ``ir_capture_jax_contraction.py`` and ``JaxLinalgTests`` in
``ir_capture_jax_linalg.py``, with their helpers in ``ir_capture_support.py``. This script runs
them all, or the named ``Class`` or ``Class.test`` cases, as ``unittest`` does.
"""

import copy
import json
import operator
import os
from pathlib import Path
import sys
import tempfile
import unittest

# The modules beside this script, which -I leaves off the path. They are not compiled to
# bytecode when run without -B: scripts/ must hold none (scripts/checks/repo_lint.py).
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ir_capture_support import (  # noqa: E402
    CHECKER, COMPARE, EXECUTE, HAS_TORCH, CheckerCase, _ieee_maximum, _ieee_minimum, _nearest,
    _ops, _outputs, _power, _same, _scalar_name, _statuses, _steps, _tags, _words,
)
import torchlean  # noqa: E402
from ir_capture_jax import JAXCompilationTests, JaxCaptureTests  # noqa: E402,F401
from ir_capture_jax_complex import JaxComplexTests  # noqa: E402,F401
from ir_capture_jax_operators import JaxGeneralOperatorTests  # noqa: E402,F401
from ir_capture_jax_contraction import JaxContractionTests  # noqa: E402,F401
from ir_capture_jax_linalg import JaxLinalgTests  # noqa: E402,F401
from ir_capture_pytorch_operators import PyTorchGeneralOperatorTests  # noqa: E402,F401
from ir_capture_pytorch_flow import PyTorchFlowTests  # noqa: E402,F401


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


@unittest.skipUnless(HAS_TORCH, "needs PyTorch")
class StateExportTests(unittest.TestCase):
    def test_lazily_negated_views_give_their_words(self):
        import torch
        from torchlean import state

        for imag, word in ((2.0, 0xC0000000), (0.0, 0x80000000)):
            value = torch.tensor(complex(1, imag), dtype=torch.complex64).conj().imag
            self.assertTrue(value.is_neg())
            self.assertEqual(state.document({"x": value})["entries"]["x"],
                             {"dtype": "float32", "shape": [], "words": [word]})


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

    def test_comparison_preserves_grad_mode(self):
        torch = self.torch

        def branch(x):
            return x + 10 if torch.is_grad_enabled() else x - 10

        x = torch.tensor([2.0], requires_grad=True)
        for enabled, expected in ((True, 12.0), (False, -8.0)):
            with self.subTest(grad_enabled=enabled), torch.set_grad_enabled(enabled):
                report = self.verify(branch, x, checks=COMPARE)
                self.assert_passes(report)
                self.assertEqual(_outputs(report)[0].item(), expected)
                self.assertEqual(torch.is_grad_enabled(), enabled)

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

        # Even detached, the imaginary part of a conjugate is a lazily negated strided tensor.
        imag = torch.tensor([1 + 2j, 3 + 4j]).conj().imag.detach()

        def sign_view(x):
            return x + 10 if x.is_neg() and x.stride() == (2,) else x - 10

        report = self.verify(sign_view, imag, checks=COMPARE)
        self.assert_passes(report)
        torch.testing.assert_close(_outputs(report)[0], sign_view(imag))
        self.assertTrue(imag.is_neg())

    def test_reference_preserves_input_autograd_metadata(self):
        torch = self.torch
        leaf = torch.tensor([1.0], requires_grad=True)
        computed = leaf * 3

        def branch(x, base):
            if x.is_leaf:
                return x + 20
            return x - 30 if x._base is base else x - 20

        for value in (leaf, computed, computed.view(1)):
            with self.subTest(value=value):
                report = self.verify(branch, value, computed, checks=COMPARE)
                self.assert_passes(report)
                torch.testing.assert_close(_outputs(report)[0], branch(value, computed))

        class WithView(torch.nn.Module):
            def __init__(self, view):
                super().__init__()
                self.register_buffer("row", view)

            def forward(self, x):
                return x + 7 if x._base is not None and x is self.row else x - 7

        view = leaf.detach()[:]
        model = WithView(view)
        report = self.verify(model, view, checks=COMPARE)
        self.assert_passes(report)
        torch.testing.assert_close(_outputs(report)[0], model(view))

        # Export can specialize a version-counter branch differently from eager execution.
        # The sample comparison must still use the caller's counter.
        fresh = torch.ones(1)
        fresh.add_(1).add_(1)
        def versioned(x):
            return x + 3 if x._version == 2 else x - 3

        report = self.verify(versioned, fresh, checks=COMPARE)
        actual = _outputs(report)[0]
        self.assertEqual(report["ok"], torch.equal(actual, versioned(fresh)))
        self.assertEqual(fresh._version, 2)

        def history(x):
            return x + 1 if "MulBackward" in x.grad_fn.name() else x - 1

        report = self.verify(history, computed, checks=COMPARE)
        self.assertEqual(_statuses(report)["compare"][0], "unsupported")
        self.assertIn("backward graph", _statuses(report)["compare"][1])

    def test_capture_releases_lifted_tensors(self):
        from torchlean import _fx

        torch = self.torch

        class WithBuffer(torch.nn.Module):
            def __init__(self, unsupported):
                super().__init__()
                self.register_buffer("offset", torch.ones(3))
                self.unsupported = unsupported

            def forward(self, x):
                y = self.offset + x
                return torch.linalg.eigvalsh(torch.diag(y)) if self.unsupported else y

        for unsupported in (False, True):
            report = self.verify(WithBuffer(unsupported), torch.ones(3))
            self.assertEqual(report["ok"], not unsupported)
            self.assertFalse(_fx._TENSOR_VALUES)

    def test_error_function_needs_an_implementation(self):
        torch = self.torch
        for function in (torch.erf, torch.nn.functional.gelu):
            with self.subTest(function=function.__name__):
                # The typed runtime has no float32 rule for erf or for gelu without an
                # approximation: the types check passes and execution refuses the step.
                report = self.verify(lambda x, f=function: f(x), self.x)
                self.assert_unsupported_execution(report, "no rule at float32")

    def special_functions(self):
        # Each unary function of torch and torch.special that SpecialFn names, with the member's
        # torch.special name, as PyTorch 2.13's torch.special page names the functions.
        torch, special = self.torch, self.torch.special
        return {torch.erfc: "erfc", special.erfc: "erfc", special.erfcx: "erfcx",
                torch.erfinv: "erfinv", special.erfinv: "erfinv", special.ndtr: "ndtr",
                special.ndtri: "ndtri", torch.lgamma: "gammaln", special.gammaln: "gammaln",
                torch.digamma: "digamma", special.digamma: "digamma", special.psi: "digamma",
                special.bessel_j0: "bessel_j0", special.bessel_j1: "bessel_j1", torch.i0: "i0",
                special.i0: "i0", special.i1: "i1", special.i0e: "i0e", special.i1e: "i1e",
                special.spherical_bessel_j0: "spherical_bessel_j0"}

    def test_special_functions_need_an_implementation(self):
        # torch.export records each unary torch.special function as an ATen overload, which
        # captures as one Op.special step naming the function. No binary format has a rule for
        # Op.special: as for erf, the types check passes and execution refuses the step, in
        # binary32 and binary64. A binary16 operand is refused at capture.
        torch, special = self.torch, self.torch.special
        for function, name in self.special_functions().items():
            for x in (self.x, self.x.double()):
                dtype = str(x.dtype).removeprefix("torch.")
                with self.subTest(function=function, dtype=dtype):
                    document = torchlean._capture(lambda x, f=function: f(x), (x,), "pytorch",
                                                  EXECUTE)
                    self.assert_special(document, dtype, [3, 4], [(name, 0, 0)], [1])
                    report = torchlean.verify(document, checks=EXECUTE, checker=CHECKER)
                    self.assert_unsupported_execution(report, f"no rule at {dtype}")
        document = self.artifact(lambda x: special.i1e(torch.erfc(x)), self.x)
        self.assert_special(document, "float32", [3, 4], [("erfc", 0, 0), ("i1e", 0, 1)], [2])
        report = self.verify(torch.lgamma, self.x.half())
        self.assert_refused(report, "only float32 and float64 operands")

    def test_polygamma_is_a_derivative_of_digamma(self):
        # torch.polygamma(n, x), torch.special.polygamma and the Tensor method are the derivative
        # of order n of digamma. Through torch.export and through FX each captures as one
        # Op.special step of digamma at order n, and execution refuses the step.
        torch = self.torch
        functions = {"torch.polygamma": lambda x: torch.polygamma(2, x),
                     "torch.special.polygamma": lambda x: torch.special.polygamma(2, x),
                     "Tensor.polygamma": lambda x: x.polygamma(2)}
        for function, f in functions.items():
            for x in (self.x, self.x.double()):
                dtype = str(x.dtype).removeprefix("torch.")
                with self.subTest(function=function, dtype=dtype):
                    document = torchlean._capture(f, (x,), "pytorch", EXECUTE)
                    self.assert_special(document, dtype, [3, 4], [("digamma", 2, 0)], [1])
                    report = torchlean.verify(document, checks=EXECUTE, checker=CHECKER)
                    self.assert_unsupported_execution(report, f"no rule at {dtype}")

        class Polygamma(torch.nn.Module):
            def forward(self, x):
                return torch.polygamma(1, x), torch.special.polygamma(3, x), x.polygamma(4)

        with tempfile.TemporaryDirectory() as directory:
            document, report = self.execute_fx(directory, Polygamma(), self.x)
            self.assert_special(document, "float32", [3, 4],
                                [("digamma", 1, 0), ("digamma", 3, 0), ("digamma", 4, 0)],
                                [1, 2, 3])
            self.assert_unsupported_execution(report, "no rule at float32")

    def binary_special_functions(self):
        # Each binary function of torch and torch.special that SpecialFn names, with the
        # member's torch.special name.
        torch, special = self.torch, self.torch.special
        return {torch.igamma: "gammainc", special.gammainc: "gammainc",
                torch.igammac: "gammaincc", special.gammaincc: "gammaincc",
                special.zeta: "zeta"}

    def test_binary_special_functions_need_an_implementation(self):
        # torch.export records each binary torch.special function as an ATen overload, which
        # captures as one Op.special step of the two inputs at orders [0, 0], each one whose
        # shape is not the result's broadcast to it: the second, the first, or both ([3, 1] and
        # [4] give [3, 4]). As for one input, execution refuses the step in binary32 and
        # binary64.
        # Operands of two dtypes, which PyTorch promotes, and a Python number are refused at
        # capture.
        torch, special = self.torch, self.torch.special
        for function, name in self.binary_special_functions().items():
            for x, y in ((self.x, self.y[0]), (self.y[0], self.x),
                         (self.x.double(), self.y[0].double()), (self.x[:, :1], self.y[0])):
                dtype = str(x.dtype).removeprefix("torch.")
                shapes = [list(x.shape), list(y.shape)]
                with self.subTest(function=function, dtype=dtype, shapes=shapes):
                    document = torchlean._capture(lambda x, y, f=function: f(x, y), (x, y),
                                                  "pytorch", EXECUTE)
                    self.assert_broadcast_step(document, dtype, "special",
                                               [{"name": name}, {"nats": [0, 0]}],
                                               shapes, [3, 4])
                    report = torchlean.verify(document, checks=EXECUTE, checker=CHECKER)
                    self.assert_unsupported_execution(report, f"no rule at {dtype}")
        report = self.verify(torch.igamma, self.x, self.y.double())
        self.assert_refused(report, "only two float32 or two float64 operands")
        report = self.verify(lambda x: special.zeta(x, 2.0), self.x)
        self.assert_refused(report, "both operands must be tensors")

    def test_binary_special_functions_capture_through_fx(self):
        # FX records the binary torch.special functions as the Python callables and the Tensor
        # methods igamma and igammac by name; each captures as one Op.special step.
        torch = self.torch
        functions = self.binary_special_functions()
        methods = {"igamma": "gammainc", "igammac": "gammaincc"}

        class Special(torch.nn.Module):
            def forward(self, x, y):
                return (tuple(f(x, y) for f in functions) +
                        tuple(getattr(x, method)(y) for method in methods))

        names = list(functions.values()) + list(methods.values())
        with tempfile.TemporaryDirectory() as directory:
            for x, y in ((self.x, self.y), (self.x.double(), self.y.double())):
                dtype = str(x.dtype).removeprefix("torch.")
                with self.subTest(dtype=dtype):
                    document, report = self.execute_fx(directory, Special(), x, y)
                    nodes = document["artifact"]["nodes"]
                    self.assertEqual([(node["op"]["record"]["tag"],
                                       node["op"]["record"]["fields"], node["op"]["inputs"])
                                      for node in nodes[2:]],
                                     [("special", [{"name": name}, {"nats": [0, 0]},
                                                   {"shape": [3, 4]}],
                                       [[0, 0], [1, 0]]) for name in names])
                    self.assert_unsupported_execution(report, f"no rule at {dtype}")

    def test_special_functions_capture_through_fx(self):
        # FX records the Tensor methods by name and torch.special's functions as the Python
        # callables, not as ATen overloads; each still captures as one Op.special step naming
        # the function, and execution refuses the step.
        torch = self.torch
        functions = self.special_functions()
        methods = {"erfc": "erfc", "erfinv": "erfinv", "lgamma": "gammaln",
                   "digamma": "digamma", "i0": "i0"}

        class Special(torch.nn.Module):
            def forward(self, x):
                return (tuple(f(x) for f in functions) +
                        tuple(getattr(x, method)() for method in methods))

        names = list(functions.values()) + list(methods.values())
        steps = [(name, 0, 0) for name in names]
        with tempfile.TemporaryDirectory() as directory:
            for x in (self.x, self.x.double()):
                dtype = str(x.dtype).removeprefix("torch.")
                with self.subTest(dtype=dtype):
                    document, report = self.execute_fx(directory, Special(), x)
                    self.assert_special(document, dtype, [3, 4], steps,
                                        range(1, len(names) + 1))
                    self.assert_unsupported_execution(report, f"no rule at {dtype}")

    def test_frexp_captures_through_fx(self):
        # FX records torch.frexp as the Python callable and Tensor.frexp as a method, and reads
        # the components by position (negative ones too), by field name or whole at the output:
        # each is the one Op.frexp step of torch.export's capture, and `compare` holds. A
        # signaling NaN gives its quiet NaN, as ATen's frexp does (glibc returns x + x there) and
        # as IEEE 754's scaleB.
        torch = self.torch

        class Whole(torch.nn.Module):
            def forward(self, x):
                return torch.frexp(x)

        class Fields(torch.nn.Module):
            def forward(self, x):
                result = x.frexp()
                return result.exponent, result.mantissa * 2

        class Items(torch.nn.Module):
            def forward(self, x):
                return torch.frexp(x)[1] * 2

        class Negative(torch.nn.Module):
            def forward(self, x):
                return torch.frexp(x)[-1] * 2, x.frexp()[-2]

        x = torch.tensor([8.0, -3.0, 0.1, 0.0, -0.0, float("inf"), float("nan"), 1e-40])
        with tempfile.TemporaryDirectory() as directory:
            for module in (Whole(), Fields(), Items(), Negative()):
                for dtype in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
                    with self.subTest(module=type(module).__name__, dtype=dtype):
                        artifact, report = self.compare_fx(directory, module, x.to(dtype))
                        self.assert_passes(report)
                        self.assertIn("frexp", _tags(artifact))
            for dtype, signaling, quiet in ((torch.float32, 0x7fa00000, 0x7fe00000),
                                            (torch.float64, 0x7ff4000000000000,
                                             0x7ffc000000000000)):
                integer = torch.int32 if dtype == torch.float32 else torch.int64
                operand = torch.tensor([signaling], dtype=integer).view(dtype)
                with self.subTest(signaling=dtype):
                    report = self.compare_fx(directory, Whole(), operand)[1]
                    self.assert_passes(report)
                    self.assertEqual(_words(report), [[quiet], [0]])
        # A position outside the result is refused by name. FX's shape propagation already fails
        # on one, so this graph is written by hand.
        graph = torch.fx.Graph()
        source = graph.placeholder("x")
        source.meta["val"] = x
        frexp = graph.call_function(torch.frexp, (source,))
        graph.output(graph.call_function(operator.getitem, (frexp, -3)))
        with self.assertRaisesRegex(NotImplementedError, "component -3 is out of range"):
            torchlean._lower_pytorch_graph(torch.nn.Module(), graph, ["x"])

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

    def test_convolutions_take_a_weight_not_known_at_capture(self):
        # F.conv1d, conv2d and conv3d whose weight or bias is a forward argument, or computed from
        # one, import it as an operand after the input through torch.export, its core
        # decomposition (aten.convolution) and FX, in binary32 and binary64. A parameter beside it
        # stays in the payload. Integer weights and inputs make every summation order exact.
        torch = self.torch
        functional = torch.nn.functional
        generator = torch.Generator().manual_seed(1)

        # FX traces a forward of fixed arity, so each case's module has one.
        class Two(torch.nn.Module):
            def __init__(self, function, **parameters):
                super().__init__()
                self.function = function
                for name, value in parameters.items():
                    self.register_parameter(name, torch.nn.Parameter(value))

            def forward(self, x, y):
                return self.function(self, x, y)

        class Three(Two):
            def forward(self, x, y, z):
                return self.function(self, x, y, z)

        def integers(*shape):
            return torch.randint(-3, 4, shape, generator=generator).float()

        cases = {
            "weight and bias": (lambda m, x, w, b: functional.conv2d(x, w, b),
                                {}, [(2, 2, 5, 5), (3, 2, 3, 3), (3,)]),
            "keyword weight and bias": (lambda m, x, w, b: functional.conv2d(
                bias=b, weight=w, input=x), {}, [(2, 2, 5, 5), (3, 2, 3, 3), (3,)]),
            "stride and padding, no bias": (
                lambda m, x, w: functional.conv1d(x, w, stride=2, padding=1),
                {}, [(2, 3, 9), (4, 3, 2)]),
            "groups, dilation and padding": (
                lambda m, x, w, b: functional.conv2d(x, w, b, padding=(1, 2), dilation=(1, 2),
                                                     groups=2),
                {}, [(1, 4, 5, 7), (6, 2, 3, 3), (6,)]),
            "three spatial axes": (lambda m, x, w, b: functional.conv3d(x, w, b, stride=(1, 2, 1)),
                                   {}, [(1, 2, 4, 5, 4), (2, 2, 2, 3, 2), (2,)]),
            "unbatched": (lambda m, x, w, b: functional.conv2d(x, w, b),
                          {}, [(2, 5, 5), (3, 2, 3, 3), (3,)]),
            "computed depthwise weight": (
                lambda m, x, w: functional.conv2d(x, w * 2 - 1, groups=3, padding=1),
                {}, [(1, 3, 4, 4), (3, 1, 3, 3)]),
            "parameter weight": (lambda m, x, b: functional.conv2d(x, m.weight, b),
                                 {"weight": integers(3, 2, 3, 3)}, [(2, 2, 5, 5), (3,)]),
            "parameter bias": (lambda m, x, w: functional.conv2d(x, w, m.bias),
                               {"bias": integers(3)}, [(2, 2, 5, 5), (3, 2, 3, 3)]),
        }
        original = torch.export.export
        with tempfile.TemporaryDirectory() as directory:
            for name, (function, parameters, shapes) in cases.items():
                for dtype in (torch.float32, torch.float64):
                    model = (Two if len(shapes) == 2 else Three)(function, **parameters)
                    model = model.to(dtype).eval()
                    inputs = [integers(*shape).to(dtype) for shape in shapes]
                    with self.subTest(case=name, dtype=str(dtype), capture="export"):
                        self.assert_passes(self.verify(model, *inputs, checks=COMPARE))
                    with self.subTest(case=name, dtype=str(dtype), capture="core"):
                        torch.export.export = \
                            lambda *a, **k: original(*a, **k).run_decompositions()
                        try:
                            self.assert_passes(self.verify(model, *inputs, checks=COMPARE))
                        finally:
                            torch.export.export = original
                    with self.subTest(case=name, dtype=str(dtype), capture="fx"):
                        self.assert_passes(self.compare_fx(directory, model, *inputs)[1])

    def test_linear_takes_a_weight_not_known_at_capture(self):
        # F.linear whose weight or bias is a forward argument, or computed from one, imports it as
        # an operand after the input through torch.export, its core decomposition (a transpose and
        # mm or addmm) and FX, in binary32 and binary64. A parameter beside it stays in the
        # payload. Integer weights and inputs make every summation order exact.
        torch = self.torch
        functional = torch.nn.functional
        generator = torch.Generator().manual_seed(1)

        # FX traces a forward of fixed arity, so each case's module has one.
        class Two(torch.nn.Module):
            def __init__(self, function, **parameters):
                super().__init__()
                self.function = function
                for name, value in parameters.items():
                    self.register_parameter(name, torch.nn.Parameter(value))

            def forward(self, x, y):
                return self.function(self, x, y)

        class Three(Two):
            def forward(self, x, y, z):
                return self.function(self, x, y, z)

        def integers(*shape):
            return torch.randint(-3, 4, shape, generator=generator).float()

        cases = {
            "weight and bias": (lambda m, x, w, b: functional.linear(x, w, b),
                                {}, [(4, 5), (3, 5), (3,)]),
            "keyword weight and bias": (lambda m, x, w, b: functional.linear(
                bias=b, weight=w, input=x), {}, [(4, 5), (3, 5), (3,)]),
            "no bias, two leading axes": (lambda m, x, w: functional.linear(x, w),
                                          {}, [(2, 3, 5), (4, 5)]),
            "vector input": (lambda m, x, w, b: functional.linear(x, w, b),
                             {}, [(5,), (3, 5), (3,)]),
            "computed weight": (lambda m, x, w: functional.linear(x, w.t() * 2 - 1),
                                {}, [(4, 5), (5, 3)]),
            "parameter weight": (lambda m, x, b: functional.linear(x, m.weight, b),
                                 {"weight": integers(3, 5)}, [(4, 5), (3,)]),
            "parameter bias": (lambda m, x, w: functional.linear(x, w, m.bias),
                               {"bias": integers(3)}, [(4, 5), (3, 5)]),
        }
        original = torch.export.export
        with tempfile.TemporaryDirectory() as directory:
            for name, (function, parameters, shapes) in cases.items():
                for dtype in (torch.float32, torch.float64):
                    model = (Two if len(shapes) == 2 else Three)(function, **parameters)
                    model = model.to(dtype).eval()
                    inputs = [integers(*shape).to(dtype) for shape in shapes]
                    with self.subTest(case=name, dtype=str(dtype), capture="export"):
                        self.assert_passes(self.verify(model, *inputs, checks=COMPARE))
                    with self.subTest(case=name, dtype=str(dtype), capture="core"):
                        torch.export.export = \
                            lambda *a, **k: original(*a, **k).run_decompositions()
                        try:
                            self.assert_passes(self.verify(model, *inputs, checks=COMPARE))
                        finally:
                            torch.export.export = original
                    with self.subTest(case=name, dtype=str(dtype), capture="fx"):
                        self.assert_passes(self.compare_fx(directory, model, *inputs)[1])

    def test_normalization_takes_a_weight_not_known_at_capture(self):
        # F.layer_norm and eval F.batch_norm whose weight, bias or running statistic is a
        # forward argument, or computed from one, import it as an operand after the input through
        # torch.export, its core decomposition and FX, in binary64, binary32, binary16 and
        # bfloat16, where the operand is cast to binary32 with the input. A parameter or buffer
        # beside it stays in the payload. _native_batch_norm_legit outside training is batch norm
        # with the running statistics too, through torch.export, and through the core
        # decomposition with buffer statistics, which it returns unchanged after its output.
        # There, statistics that are inputs are written back as a mutation of the inputs, which
        # the IR refuses, and FX keeps the ATen call unresolved. The rows, statistics and eps are
        # test_normalizations_execute_with_their_payload's, on which every operation is exact.
        torch = self.torch
        functional = torch.nn.functional
        generator = torch.Generator().manual_seed(2)

        class Two(torch.nn.Module):
            def __init__(self, function, **parameters):
                super().__init__()
                self.function = function
                for name, value in parameters.items():
                    self.register_parameter(name, torch.nn.Parameter(value))
                self.register_buffer("mean", torch.tensor([1.0, -2.0, 0.5]))
                self.register_buffer("var", torch.tensor([3.0, 15.0, 0.0]))

            def forward(self, x, y):
                return self.function(self, x, y)

        class Three(Two):
            def forward(self, x, y, z):
                return self.function(self, x, y, z)

        class Four(Two):
            def forward(self, x, y, z, u):
                return self.function(self, x, y, z, u)

        class Five(Two):
            def forward(self, x, y, z, u, v):
                return self.function(self, x, y, z, u, v)

        rows = torch.tensor([[1.0, 1.0, 1.0, 5.0], [-3.0, -3.0, -3.0, 1.0],
                             [5.0, 5.0, 5.0, 5.0]])
        scale, shift = torch.tensor([2.0, 3.0, 4.0, 5.0]), torch.tensor([1.0, -1.0, 0.5, 0.25])
        channels = torch.randint(-4, 5, (2, 3, 5), generator=generator).float()
        gamma, beta = torch.tensor([2.0, -1.0, 4.0]), torch.tensor([0.5, 3.0, -1.0])
        mean, var = torch.tensor([1.0, -2.0, 0.5]), torch.tensor([3.0, 15.0, 0.0])

        def batch(m, x, w, b):
            return functional.batch_norm(x, m.mean, m.var, w, b, eps=1.0)

        def legit(x, w, b, mean, var):
            return torch.ops.aten._native_batch_norm_legit(x, w, b, mean, var, False, 0.1, 1.0)[0]

        cases = {
            "layer norm": (lambda m, x, w, b: functional.layer_norm(x, (4,), w, b, eps=1.0),
                           {}, [rows, scale, shift]),
            "layer norm without bias": (
                lambda m, x, w: functional.layer_norm(x, (4,), w, eps=1.0), {},
                [rows, scale]),
            "layer norm, computed weight": (
                lambda m, x, w: functional.layer_norm(x, (4,), w * 2 - 1, m.bias, eps=1.0),
                {"bias": shift}, [rows, scale]),
            "batch norm": (batch, {}, [channels, gamma, beta]),
            "batch norm, parameter weight": (lambda m, x, b: batch(m, x, m.weight, b),
                                             {"weight": gamma}, [channels, beta]),
            "batch norm, input statistics": (
                lambda m, x, w, b, mean, var: functional.batch_norm(x, mean, var, w, b, eps=1.0),
                {}, [channels, gamma, beta, mean, var]),
            "batch norm, input variance": (
                lambda m, x, var: functional.batch_norm(x, m.mean, var, eps=1.0), {},
                [channels, var]),
            "batch norm, computed mean": (
                lambda m, x, mean: functional.batch_norm(x, mean * 2 - 1, m.var, m.weight,
                                                         eps=1.0),
                {"weight": gamma}, [channels, (mean + 1) / 2]),
            "_native_batch_norm_legit": (
                lambda m, x, w, b, mean, var: legit(x, w, b, mean, var), {},
                [channels, gamma, beta, mean, var]),
            "_native_batch_norm_legit, buffer statistics": (
                lambda m, x, w, b: legit(x, w, b, m.mean, m.var), {}, [channels, gamma, beta]),
        }
        tuple_valued = "unsupported tuple-valued PyTorch op"
        refused = {("_native_batch_norm_legit", "core"): "other than buffer updates",
                   ("_native_batch_norm_legit", "fx"): tuple_valued,
                   ("_native_batch_norm_legit, buffer statistics", "fx"): tuple_valued}
        arity = {2: Two, 3: Three, 4: Four, 5: Five}
        original = torch.export.export
        with tempfile.TemporaryDirectory() as directory:
            for name, (function, parameters, values) in cases.items():
                for dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16):
                    model = arity[len(values)](function, **parameters).to(dtype).eval()
                    inputs = [value.to(dtype) for value in values]
                    with self.subTest(case=name, dtype=str(dtype), capture="export"):
                        self.assert_passes(self.verify(model, *inputs, checks=COMPARE))
                    with self.subTest(case=name, dtype=str(dtype), capture="core"):
                        torch.export.export = \
                            lambda *a, **k: original(*a, **k).run_decompositions()
                        try:
                            report = self.verify(model, *inputs, checks=COMPARE)
                        finally:
                            torch.export.export = original
                        if (name, "core") in refused:
                            self.assert_refused(report, refused[name, "core"])
                        else:
                            self.assert_passes(report)
                    with self.subTest(case=name, dtype=str(dtype), capture="fx"):
                        if (name, "fx") in refused:
                            with self.assertRaisesRegex(NotImplementedError, refused[name, "fx"]):
                                self.compare_fx(directory, model, *inputs)
                        else:
                            self.assert_passes(self.compare_fx(directory, model, *inputs)[1])
            # The CPU kernel takes a binary32 weight and bias of a binary16 or bfloat16 input
            # ("mixed dtype") and applies them in binary32 before the one rounding. A weight and
            # bias off the 16-bit grid tell that apart from rounding them first: the first entry
            # is -0.5 * (2 + 2^-20) + (1 + 2^-20) = 2^-21 here, and 0 with them rounded first.
            mixed = Two(lambda m, x, w: functional.layer_norm(x, (4,), w, m.bias, eps=1.0),
                        bias=shift + 2 ** -20).eval()
            for dtype in (torch.float16, torch.bfloat16):
                inputs = [rows.to(dtype), scale + 2 ** -20]
                with self.subTest(case="mixed", dtype=str(dtype), capture="export"):
                    self.assert_passes(self.verify(mixed, *inputs, checks=COMPARE))
                with self.subTest(case="mixed", dtype=str(dtype), capture="core"):
                    torch.export.export = lambda *a, **k: original(*a, **k).run_decompositions()
                    try:
                        self.assert_passes(self.verify(mixed, *inputs, checks=COMPARE))
                    finally:
                        torch.export.export = original
                with self.subTest(case="mixed", dtype=str(dtype), capture="fx"):
                    self.assert_passes(self.compare_fx(directory, mixed, *inputs)[1])

    def test_in_place_activations_are_refused(self):
        # An in-place nn.ELU, nn.Hardtanh or nn.SiLU whose result is unused mutates its input. FX
        # records the call, and the capture refuses it, since dropping it would lose the mutation.
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
            for activation in (nn.ELU(inplace=True), nn.Hardtanh(inplace=True),
                               nn.SiLU(inplace=True)):
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
        # running variance below -eps gives NaN in its channel in both.
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
        # Without running statistics the batch norm normalizes by the batch's moments, within a
        # tolerance.
        batch = nn.BatchNorm1d(3, track_running_stats=False).eval()
        inputs = torch.randn(4, 3, generator=generator)
        report = self.verify(batch, inputs, checks=EXECUTE)
        self.assert_passes(report)
        with torch.no_grad():
            torch.testing.assert_close(_outputs(report)[0], batch(inputs), rtol=1e-5, atol=1e-5)
        # PyTorch and the IR take the root of var + eps as it stands, here 1, -0.25 and 4, so
        # every other step is exact.
        negative = nn.BatchNorm1d(3, eps=0.25).eval()
        with torch.no_grad():
            negative.running_var.copy_(torch.tensor([0.75, -0.5, 3.75]))
        x = torch.randn(2, 3, generator=generator)
        for dtype in (torch.float64, torch.float32):
            with self.subTest(case="negative running variance", dtype=dtype):
                report = self.verify(copy.deepcopy(negative).to(dtype), x.to(dtype),
                                     checks=COMPARE)
                self.assert_passes(report)
                self.assertTrue(_outputs(report)[0][:, 1].isnan().all())

    def test_argument_aliases_bind_once_on_every_path(self):
        # PyTorch's parser reads NumPy's `axis` and `keepdims` as `dim` and `keepdim`, so a sum, a
        # mean and a cumulative sum keep their axis through torch.export, FX and Dynamo alike.
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
            "Tensor.mean dim -1 keepdims": lambda v: v.mean(dim=-1, keepdims=True),
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

    def test_records_without_an_integer_form_are_refused(self):
        # A record Lean's decoder has only at a binary format is refused at an integer type with
        # that reason, not exported to fail at the import.
        torch = self.torch
        m = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
        self.assert_refused(self.verify(lambda v: torch.sum(v, axis=1, keepdims=True), m),
                            "reduce runs at a binary format only")

    def test_floating_trunc_division_truncates_the_rounded_quotient(self):
        # PyTorch truncates the rounded quotient (`10` at `(1, 0.1)`), where the IR's `trunc` mode
        # truncates the exact one (`9`). Binary32, and bfloat16 with a scalar divisor, which ATen
        # divides in binary32.
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
        from torchlean._bind import bind_arguments

        torch = self.torch
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
        report = torchlean.verify(unbound, checks=EXECUTE, checker=CHECKER)
        statuses = _statuses(report)
        self.assertFalse(report["ok"])
        self.assertEqual(statuses["types"][0], "failed", statuses)
        self.assertTrue(statuses["types"][1].startswith("IR types: "), statuses)
        self.assertIn("the IR computes float32 [], the capture declares float32 [2, 1]",
                      statuses["types"][1])
        self.assertEqual(statuses["execute"], statuses["types"])

    def test_a_document_without_utf8_json_text_fails_every_check(self):
        # A lone surrogate, a NaN or a value JSON lacks fails every request and writes no export.
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

    def test_square_and_comparison_aliases_compile(self):
        import torch
        from torchlean._fx import _tensor_value

        @torchlean.compile(checker=CHECKER)
        def compare(x, y):
            return (x.square(), torch.square(y), torch.not_equal(x, y), x.greater(y),
                    torch.less(x, y), x.less_equal(y), torch.greater_equal(x, y))

        x, y = torch.tensor([-2., 1., 4.]), torch.tensor([3., 1., -2.])
        eager = (x * x, y * y, x != y, x > y, x < y, x <= y, x >= y)
        torch.testing.assert_close(compare(x, y), eager)
        self.assert_passes(compare.torchlean.report)
        document = compare.torchlean.artifact
        document.update(arguments=[_tensor_value(v) for v in (x, y)],
                        reference_outputs=[_tensor_value(v) for v in eager])
        self.assert_passes(torchlean.verify(document, checks=("compare_bitwise",), checker=CHECKER))

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


if __name__ == "__main__":
    unittest.main()
