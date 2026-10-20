"""Capture pointwise operators from PyTorch and JAX and check them with ``verify ir``.

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
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torchlean  # noqa: E402


def _checker():
    path = os.environ.get("TORCHLEAN_VERIFY") or str(ROOT / ".lake/build/bin/verify")
    return path if Path(path).is_file() else None


CHECKER = _checker()
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_JAX = importlib.util.find_spec("jax") is not None
EXECUTE = ("structure", "types", "execute")
COMPARE = ("structure", "types", "execute", "compare")


def _statuses(report):
    return {check["name"]: (check["status"], check["message"]) for check in report["checks"]}


class CheckerCase(unittest.TestCase):
    def assert_passes(self, report):
        self.assertTrue(report["ok"], _statuses(report))

    def assert_unsupported_execution(self, report, text):
        statuses = _statuses(report)
        self.assertEqual(statuses["structure"][0], "checked", statuses)
        self.assertEqual(statuses["types"][0], "checked", statuses)
        self.assertEqual(statuses["execute"][0], "failed", statuses)
        self.assertIn(text, statuses["execute"][1])


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

    def fx_adapter(self, directory):
        # The emitted capture script, with the torch.export graph skipped so that the capture
        # falls back to FX symbolic tracing.
        path = Path(directory) / "capture.py"
        subprocess.run([CHECKER, "ir", "--export-pytorch", str(path)], check=True)
        spec = importlib.util.spec_from_file_location("torchlean_capture_fx", path)
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        adapter._graph_is_lowerable = lambda graph, model: False
        return adapter

    def compare_fx(self, adapter, directory, module, *inputs):
        artifact = adapter.export_torchlean_graph_json(module, inputs,
                                                       str(Path(directory) / "graph.json"))
        artifact["arguments"] = [adapter._constant(x) for x in inputs]
        outputs = module(*inputs)
        outputs = outputs if isinstance(outputs, tuple) else (outputs,)
        artifact["reference_outputs"] = [adapter._constant(y) for y in outputs]
        return artifact, torchlean.verify(artifact, checks=COMPARE, checker=CHECKER)

    def test_exact_operations_match_eager_words(self):
        torch = self.torch

        def exact(x, y):
            return (torch.div(x, y), torch.div(x, y, rounding_mode="trunc"),
                    torch.div(x, y, rounding_mode="floor"), x / 2.5, x // 0.75,
                    torch.neg(x), torch.pow(x, 2), torch.pow(x, 3), torch.pow(x, -2),
                    torch.pow(x, -1), torch.pow(x, 0), torch.pow(x, 1))

        self.assert_passes(self.verify(exact, self.x, self.y, checks=COMPARE))

    def test_integer_division_negation_and_power(self):
        torch = self.torch
        a = torch.tensor([7, -7, 9, -9], dtype=torch.int32)
        b = torch.tensor([2, 2, -4, -4], dtype=torch.int32)

        def integers(a, b):
            return (torch.div(a, b, rounding_mode="trunc"), torch.div(a, b, rounding_mode="floor"),
                    -a, torch.pow(a, 3), torch.div(a, b))

        self.assert_passes(self.verify(integers, a, b, checks=COMPARE))

    def test_reduced_division_and_product_match_eager_words(self):
        # Reduced-precision division and multiplication run in binary32 when the right operand is
        # one element, with that element's own value: a Python scalar, a binary32 rank-zero
        # tensor, a one-element tensor. A scalar dividend is rounded to the tensor dtype and
        # divided in that dtype.
        torch = self.torch
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                def division(x, y, scalar, single):
                    return (torch.div(x, y, rounding_mode="floor"), x / 0.3, x // 0.3,
                            torch.div(0.1, y), 1000 // y,
                            torch.div(x, scalar, rounding_mode="trunc"),
                            torch.div(x, single, rounding_mode="floor"), x * scalar, y * single)

                self.assert_passes(self.verify(
                    division, (self.x * 100).to(dtype), self.y.to(dtype), torch.tensor(0.3),
                    torch.tensor([0.3], dtype=dtype), checks=COMPARE))

    def test_reduced_power_matches_eager_words(self):
        # PyTorch rounds a scalar exponent or base to the tensor dtype, through binary32, before
        # its dispatch compares it: bfloat16 2.001 and 3.005 are computed as x * x and x * x * x.
        # A binary64 rank-zero exponent is rounded the same way.
        torch = self.torch
        x = torch.tensor([10.0, 3.0, 7.0, 100.0, 0.5, 1.5, 2.25, 6.0])
        k = torch.tensor([0.0, 1.0, 2.0, 3.0, -1.0, -2.0, 4.0, 5.0])
        tie = {torch.bfloat16: 1 + 2 ** -8 + 2 ** -30, torch.float16: 1 + 2 ** -11 + 2 ** -40}
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                def power(x, k, e):
                    return (x ** 2.1, x ** 2.001, x ** 3.005, x ** -2.001, torch.pow(1.3, k),
                            torch.pow(0.1, k), x ** e)

                self.assert_passes(self.verify(
                    power, x.to(dtype), k.to(dtype), torch.tensor(tie[dtype], dtype=torch.float64),
                    checks=COMPARE))

    def test_reduced_conversions_round_through_binary32(self):
        # Tensor.to and type promotion make bfloat16 and binary16 from binary32, as c10::convert
        # does, so 16842753 becomes bfloat16 16777216; rounded once it would be 16908288.
        # Integers convert to binary16 the same either way: binary32 holds every integer below
        # binary16's overflow threshold.
        torch = self.torch
        a = torch.tensor([16842753, -16842753, 3, 65519, 65520], dtype=torch.int32)
        b = torch.tensor([16842753, 2 ** 40 + 2 ** 32 + 1, -(2 ** 40 + 2 ** 32 + 1),
                          2 ** 62 + 2 ** 54 + 1], dtype=torch.int64)
        c = torch.tensor([1 + 2 ** -8 + 2 ** -30, -(1 + 2 ** -8 + 2 ** -30),
                          1 + 2 ** -11 + 2 ** -40, 0.1], dtype=torch.float64)
        self.assertEqual(int(a[:1].to(torch.bfloat16).view(torch.int16)), 0x4B80)

        def conversions(a, b, c, zero, one):
            return (a.to(torch.bfloat16), a.to(torch.float16), b.to(torch.bfloat16),
                    b.to(torch.float16), c.to(torch.bfloat16), c.to(torch.float16),
                    a + zero, b * one)

        self.assert_passes(self.verify(conversions, a, b, c, torch.zeros(5, dtype=torch.bfloat16),
                                       torch.ones(4, dtype=torch.bfloat16), checks=COMPARE))

    def test_python_scalar_operands(self):
        # torch.export records Python's 0.1 / x as reciprocal(x) * 0.1. A scalar is a constant:
        # add and sub round it to the tensor dtype, and reduced-precision mul multiplies by its
        # binary32 value. An int beyond 2 ** 53 rounds once to binary32, from int64.
        torch = self.torch
        large = 2 ** 60 + 2 ** 36 + 1
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                def scalars(x):
                    return 0.1 / x, x * 0.3, 1.7 * x, x + 0.1, x - 0.3, x * 3

                self.assert_passes(self.verify(scalars, self.x.to(dtype), checks=COMPARE))
        self.assert_passes(self.verify(lambda x: (x + large, x * large, x / large), self.x,
                                       checks=COMPARE))

    def test_reflected_division_fx_path(self):
        # FX records Python's 0.1 / x as operator.truediv(0.1, x); the capture writes
        # Tensor.__rtruediv__'s reciprocal(x) * 0.1, with the product in binary32 for bfloat16.
        torch = self.torch

        class Reflected(torch.nn.Module):
            def forward(self, x):
                return 0.1 / x

        with tempfile.TemporaryDirectory() as directory:
            adapter = self.fx_adapter(directory)
            for dtype in (torch.bfloat16, torch.float32):
                with self.subTest(dtype=dtype):
                    artifact, report = self.compare_fx(adapter, directory, Reflected(),
                                                       self.x.to(dtype))
                    kinds = [node["kind"] for node in artifact["nodes"]]
                    self.assertIn("inv", kinds)
                    self.assertNotIn("div", kinds)
                    self.assert_passes(report)

    def test_scalar_first_products_follow_eager(self):
        # Python's s * x calls x.__rmul__(s), which is x * s: a reduced-precision product with
        # s's binary32 value. An explicit torch.mul(s, x) keeps s on the left, where mul_kernel
        # rounds it to the tensor's dtype, so bfloat16 torch.mul(0.1, x) is bfloat16(0.1) * x.
        # torch.export records the two forms as mul.Tensor(x, s) and mul.Tensor(s, x), FX as
        # operator.mul(s, x) and torch.mul(s, x).
        torch = self.torch
        x = torch.randn(256, generator=torch.Generator().manual_seed(1)) * 4

        def products(x):
            return (torch.mul(0.1, x), torch.mul(0.3, x), 0.1 * x, x * 0.1, torch.mul(3, x),
                    torch.mul(257, x))

        class Products(torch.nn.Module):
            def forward(self, x):
                return products(x)

        with tempfile.TemporaryDirectory() as directory:
            adapter = self.fx_adapter(directory)
            for dtype in (torch.bfloat16, torch.float16):
                with self.subTest(dtype=dtype):
                    reduced = x.to(dtype)
                    self.assertNotEqual(torch.mul(0.1, reduced).view(torch.int16).tolist(),
                                        (0.1 * reduced).view(torch.int16).tolist())
                    self.assert_passes(self.verify(products, reduced, checks=COMPARE))
                    self.assert_passes(self.compare_fx(adapter, directory, Products(), reduced)[1])

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
        self.assert_passes(self.verify(Clamped(), zero, checks=COMPARE))

    def test_clamp_min_bound_converts_to_the_result_type(self):
        # clamp_min of an integer tensor with a float bound computes in the promoted binary32:
        # an int32 tensor clamped at 0.5 gives binary32 0.5 where the tensor is below it. A
        # buffer is captured by torch.export; FX folds a computation on buffers alone into a
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
            adapter = self.fx_adapter(directory)
            for dtype in (torch.int32, torch.int64):
                integers = torch.arange(-32, 32, dtype=dtype)
                for bound in (0.5, 0.1, 0.3):
                    with self.subTest(dtype=dtype, bound=bound):
                        model = Clamped(integers, bound)
                        with torch.no_grad():
                            self.assertEqual(model(zero)[0].item(), torch.tensor(bound).item())
                        self.assert_passes(self.verify(model, zero, checks=COMPARE))
                        model = ClampedInput(bound)
                        self.assertEqual(model(integers)[0].dtype, torch.float32)
                        self.assert_passes(self.verify(model, integers, checks=COMPARE))
                        self.assert_passes(
                            self.compare_fx(adapter, directory, model, integers)[1])

    def test_clamp_min_of_any_tensor(self):
        # The bound's dtype comes from the result, so clamp_min applies to inputs and
        # intermediate values as well as to parameters and buffers.
        torch = self.torch
        x = torch.randn(256, generator=torch.Generator().manual_seed(3))

        class Clamped(torch.nn.Module):
            def forward(self, x):
                return torch.clamp_min(x, 0.1), x.clamp_min(-0.3), (x * 2).clamp_min(0.3)

        with tempfile.TemporaryDirectory() as directory:
            adapter = self.fx_adapter(directory)
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                with self.subTest(dtype=dtype):
                    self.assert_passes(self.verify(Clamped(), x.to(dtype), checks=COMPARE))
                    self.assert_passes(
                        self.compare_fx(adapter, directory, Clamped(), x.to(dtype))[1])

    def test_clamp_min_keeps_eager_signed_zeros_and_nans(self):
        # Eager clamp_min with a scalar bound returns the bound where the tensor is below it and
        # the tensor's value elsewhere, so clamp_min(-0.0, 0) is -0.0; a NaN stays a NaN and a
        # NaN bound gives NaNs. Four elements run in PyTorch's scalar loop, 256 in its vectorized
        # loop, under torch.export and FX.
        torch = self.torch
        values = [-0.0, 0.0, float("nan"), -float("inf"), float("inf"), -1.0, 1.0, 0.5]
        self.assertEqual(torch.clamp_min(torch.tensor([-0.0]), 0).view(torch.int32).item(),
                         -2 ** 31)

        class Clamped(torch.nn.Module):
            def __init__(self, bound):
                super().__init__()
                self.bound = bound

            def forward(self, x):
                return torch.clamp_min(x, self.bound), x.clamp_min(self.bound)

        with tempfile.TemporaryDirectory() as directory:
            adapter = self.fx_adapter(directory)
            for dtype in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
                for length in (4, 256):
                    x = torch.tensor((values * 32)[:length], dtype=dtype)
                    for bound in (0, 0.0, -0.0, float("nan")):
                        with self.subTest(dtype=dtype, length=length, bound=bound):
                            self.assert_passes(self.verify(Clamped(bound), x, checks=COMPARE))
                            self.assert_passes(
                                self.compare_fx(adapter, directory, Clamped(bound), x)[1])

    def test_maximum_and_minimum_promote_through_binary32(self):
        # torch.maximum and torch.minimum convert an int32 operand to bfloat16 through binary32,
        # as add and mul do: 16842753 becomes 16777216, not 16908288.
        torch = self.torch
        a = torch.tensor([16842753, -16842753, 3, 65519, 65520], dtype=torch.int32)

        def extremes(a, low, high, low16, high16):
            return (torch.maximum(a, low), torch.minimum(a, high), torch.maximum(a, low16),
                    torch.minimum(a, high16))

        low, high = torch.full((5,), -1e38), torch.full((5,), 1e38)
        self.assertEqual(torch.maximum(a, low.to(torch.bfloat16))[0].item(), 16777216)
        self.assert_passes(self.verify(extremes, a, low.to(torch.bfloat16),
                                       high.to(torch.bfloat16), torch.full((5,), -6e4).half(),
                                       torch.full((5,), 6e4).half(), checks=COMPARE))

    def test_nan_results_agree_whatever_their_sign_and_payload(self):
        # On x86, PyTorch's 0 / 0 is the negative default NaN and FloatLib's a positive quiet NaN;
        # the IR leaves both unspecified. compare_bitwise compares the words.
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

    def test_error_function_needs_an_implementation(self):
        torch = self.torch
        for function in (torch.erf, torch.nn.functional.gelu):
            with self.subTest(function=function.__name__):
                report = self.verify(lambda x, f=function: f(x), self.x)
                self.assert_unsupported_execution(report, "has no implementation for float32")


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

    def advice(self, report):
        # Every mismatch fails; the behaviours whose emulation reproduces it are advice.
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
        # The artifact torchlean.verify writes for a JAX function, before the checker runs; the
        # reference may come from another function, as an importer bug would make it.
        from NN.Runtime.JAX.jaxpr import encode, export, references

        artifact = export(function, *inputs)
        artifact["arguments"] = [encode(value) for value in inputs]
        artifact.update(references(reference or function, *inputs))
        return artifact

    def swapped(self, artifact, kind, which=0):
        # An importer bug: the operands of one node of `kind` in the wrong order.
        artifact = copy.deepcopy(artifact)
        node = [node for node in artifact["nodes"] if node["kind"] == kind][which]
        node["parents"] = node["parents"][::-1]
        return artifact

    def with_constant(self, artifact, old, new):
        # An importer bug: every constant node holding `old` written as `new`.
        from NN.Runtime.JAX.jaxpr import encode

        artifact = copy.deepcopy(artifact)
        for node in artifact["nodes"]:
            if node["kind"] == "const" and node.get("values") and set(node["values"]) == {old}:
                value = self.np.full(len(node["values"]), new, dtype=self.jnp.dtype(node["dtype"]))
                node["bits"] = encode(self.jnp.asarray(value))["bits"]
                node["values"] = [float(item) for item in value]
                return artifact
        raise AssertionError(f"no constant {old}")

    def assert_fails(self, artifact):
        # A failure with no documented XLA behaviour listed.
        report = torchlean.verify(artifact, checks=COMPARE, checker=CHECKER)
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

    def test_division_by_a_constant_keeps_the_ieee_quotient(self):
        # The importer gives lax.div its IEEE meaning. XLA's CPU code multiplies by binary32
        # 1/3 instead, so its x / 3.0 differs from the quotient: compare fails and lists the
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
        # jax.jit folds the array and multiplies by its reciprocal; primitive by primitive, XLA
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
        # in an eager function the computation of each node: x / 3.0 broadcasts its divisor
        # inside the division, jnp.broadcast_to in a computation of its own. Each node also
        # records its primitive, inside a nested jax.jit call too. The reference records its
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
        nodes = self.artifact(lambda v: v / jnp.broadcast_to(three, v.shape), self.x)["nodes"]
        (division,) = [node for node in nodes if node["kind"] == "div"]
        (broadcast,) = [node for node in nodes if node["kind"] == "broadcast_to"]
        self.assertNotEqual(division["computation"], broadcast["computation"])
        nodes = eager["nodes"]
        (division,) = [node for node in nodes if node["kind"] == "div"]
        (broadcast,) = [node for node in nodes if node["kind"] == "broadcast_to"]
        self.assertEqual(division["computation"], broadcast["computation"])
        self.assertEqual(division["primitive_index"], broadcast["primitive_index"])
        nested = self.artifact(jax.jit(lambda u, v, w: (u / v) / w), self.x, self.y, self.x)
        first, second = [node for node in nested["nodes"] if node["kind"] == "div"]
        self.assertEqual(first["computation"], second["computation"])
        self.assertNotEqual(first["primitive_index"], second["primitive_index"])

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
        for name, artifact in cases.items():
            with self.subTest(case=name):
                self.assert_fails(artifact)

    def test_xla_functions_are_reproduced_within_their_windows(self):
        # XLA's rsqrt and power are its own approximations: a result they compute counts as
        # reproduced within 2 and 1 binary32 ulps, the largest distances measured between them
        # and the IR's. A power qualifies only when both operands are arguments, so a constant
        # exponent gets no window; FloatLib's power is farther from XLA's for large |y log x|.
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
        # the window applies to the binary32 value; a binary64 rsqrt converted to binary32 has a
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
            for name, artifact in cases.items():
                with self.subTest(case=name):
                    self.assert_fails(artifact)
            self.assert_passes(torchlean.verify(correct, checks=COMPARE, checker=CHECKER))
        factor = jnp.float32(1.0000003)
        self.assertEqual(self.advice(self.verify(
            lambda v: lax.rsqrt(v * factor), jnp.asarray(positive), checks=COMPARE)), ["rsqrt"])

    def test_power_windows_need_argument_operands(self):
        # A changed exponent, or a factor dropped before a small exponent, moves the power by
        # about one ulp, inside a window; with a constant operand the window does not apply.
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
        for name, artifact in cases.items():
            with self.subTest(case=name):
                self.assert_fails(artifact)

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
        inner, outer = [node for node in swapped["nodes"] if node["kind"] == "div"]
        inner["parents"][1], outer["parents"][1] = outer["parents"][1], inner["parents"][1]
        nested = copy.deepcopy(self.artifact(jax.jit(lambda a, b, c: a / (b / c)), x, y, z))
        first, second = [node for node in nested["nodes"] if node["kind"] == "div"]
        inner, outer = (first, second) if first["parents"][0] != 0 else (second, first)
        outer["parents"][0], inner["parents"][1] = inner["parents"][1], outer["parents"][0]
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
        for name, artifact in cases.items():
            with self.subTest(case=name):
                entry = self.assert_fails(artifact)
                self.assertEqual(entry["eager_reference"]["status"], "not_reproduced")
        report = torchlean.verify(quotients, checks=COMPARE, checker=CHECKER)
        self.assertEqual(self.advice(report), ["rewritten_division"])
        self.assertEqual(report["checks"][-1]["eager_reference"]["status"], "agrees")

    def test_emulation_needs_a_cpu_reference(self):
        # XLA's CPU behaviour is emulated only for a reference that XLA's CPU backend computed.
        x = self.jnp.asarray((self.np.random.default_rng(1).standard_normal(4096) * 3)
                             .astype(self.np.float32))
        artifact = self.artifact(lambda v: v / 3.0, x)
        self.assertEqual(self.advice(torchlean.verify(artifact, checks=COMPARE, checker=CHECKER)),
                         ["divisor_reciprocal"])
        for platform in ("gpu", None):
            with self.subTest(platform=platform):
                changed = copy.deepcopy(artifact)
                if platform is None:
                    del changed["reference_platform"]
                else:
                    changed["reference_platform"] = platform
                self.assert_fails(changed)

    def test_rewritten_operations_are_consistent_with_jit(self):
        # Compiled with jax.jit, XLA computes (x / y) / z as x / (y * z) and x ** 3.0 as
        # x * (x * x); the IR keeps the two quotients and the correctly rounded cube.
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
        # has no window, so nothing is listed; on the first 64 it returns the IR's words.
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
        # to binary16; the IR's power is binary32's, rounded once.
        jnp, np = self.jnp, self.np
        x = jnp.asarray((np.random.default_rng(11).standard_normal(4096) * 3).astype(np.float16))
        cube = self.jax.jit(lambda v: v ** 3.0)
        f32 = np.asarray(x).astype(np.float32)
        steps = (f32 * (f32 * f32).astype(np.float16).astype(np.float32)).astype(np.float16)
        self.assertTrue((np.asarray(cube(x)).view(np.uint16) == steps.view(np.uint16)).all())
        self.assertEqual(self.advice(self.verify(cube, x, checks=COMPARE)), ["pow_products"])

    def test_flushed_subnormals_are_consistent_with_flushing(self):
        # XLA's CPU code flushes subnormal operands and results to zero; the IR underflows
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

    def test_reduced_conversions_round_through_binary32(self):
        # XLA's CPU backend makes bfloat16 and binary16 from binary32, so int32 16842753 becomes
        # bfloat16 16777216; rounded once it would be 16908288.
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

            self.assert_passes(self.verify(conversions, a, b, c, zero, checks=COMPARE))

    def test_rounded_operations_execute(self):
        lax, jnp = self.jax.lax, self.jnp

        def rounded(x, y):
            mean = jnp.mean(x, axis=-1, keepdims=True)
            variance = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
            normalized = (x - mean) * lax.rsqrt(variance + 1e-5)
            return lax.pow(jnp.abs(x) + 1, y), self.jax.nn.gelu(normalized)

        self.assert_passes(self.verify(rounded, self.x, self.y))

    def test_error_function_needs_an_implementation(self):
        report = self.verify(self.jax.lax.erf, self.x)
        self.assert_unsupported_execution(report, "has no implementation for float32")


if __name__ == "__main__":
    unittest.main()
