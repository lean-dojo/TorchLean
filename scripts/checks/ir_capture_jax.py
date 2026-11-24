"""JAX captures checked with ``verify ir``: the importer's structure (``JaxCaptureTests``) and
``jax.jit``-compiled functions (``JAXCompilationTests``). Run them through
``scripts/checks/test_ir_capture.py``, which describes the checks.
"""

import copy
import json
import re
import unittest

from ir_capture_support import (CHECKER, COMPARE, EXECUTE, HAS_JAX, CheckerCase, _blocks,
                                _flows, _names, _nearest, _outputs, _records, _same,
                                _scalar_name, _statuses, _steps)
import torchlean


def _words(array):
    # The words of a float8 or Boolean array, each NaN as -1, or the integers.
    import numpy as np

    array = np.asarray(array)
    if array.dtype.kind == "i":
        return array.astype(np.int64).ravel()
    words = array.view(np.uint8).astype(np.int64)
    words[np.isnan(array.astype(np.float32))] = -1
    return words.ravel()


def _float8_outputs(report, expected, dtype):
    # The words (``_words``) of the IR's outputs, each read at the type of the matching output
    # of ``expected``: integers, Booleans or the float8 ``dtype``.
    import numpy as np

    return [_words(np.asarray(output["bits"], np.int64) if np.asarray(wanted).dtype.kind == "i"
                   else np.asarray(output["bits"], np.uint8).view(
                       np.bool_ if output["element_type"] == "bool" else dtype))
            for output, wanted in zip(report["outputs"], expected, strict=True)]



def _without_jax_object_ids(document):
    # Captured variable names include the Python object's address.
    return re.sub(r"\(id=\d+\)", "", json.dumps(document, sort_keys=True))

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

    def same(self, got, wanted):
        # Each pair of word arrays (``_words``) equal: the first positions where they differ,
        # which keeps a failure's message short.
        import numpy as np

        for k, (actual, expected) in enumerate(zip(got, wanted, strict=True)):
            differ = np.flatnonzero(actual != expected)[:4].tolist()
            self.assertEqual(differ, [], f"output {k}: IR {actual[differ].tolist()}, "
                                         f"expected {expected[differ].tolist()}")

    def test_arithmetic_without_a_boolean_form_is_refused(self):
        # Boolean arithmetic other than the maximum and minimum (logical or and and), which Lean's
        # decoder does not have and neither frontend emits, is refused at the lowering with the
        # reason.
        from torchlean import _artifact
        from torchlean._table import op_table

        nodes = [dict(id=0, kind="input", parents=[], shape=[2], dtype="bool"),
                 dict(id=1, kind="sub", parents=[0, 0], shape=[2], dtype="bool")]
        with self.assertRaisesRegex(NotImplementedError, "sub has no Boolean form"):
            _artifact.lower(nodes, [0], [1], op_table(CHECKER))

    def test_an_uncopyable_model_is_refused(self):
        # The capture copies the model, and a model holding a lock is refused.
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
        # The importer gives lax.div its IEEE meaning. XLA's CPU code for jax.jit multiplies by the
        # binary32 value of 1/3, so its x / 3.0 differs from the quotient: compare fails and lists
        # the reciprocal as the documented behaviour the mismatch is consistent with.
        jnp, np = self.jnp, self.np
        values = (np.random.default_rng(1).standard_normal(4096) * 3).astype(np.float32)
        x = jnp.asarray(values)
        divide = self.jax.jit(lambda v: v / 3.0)
        report = self.verify(divide, x, checks=COMPARE)
        words = np.array(report["outputs"][0]["bits"], dtype=np.uint32)
        self.assertTrue((words == (values / np.float32(3)).view(np.uint32)).all())
        xla = np.asarray(divide(x))
        self.assertTrue((xla == values * (np.float32(1) / np.float32(3))).all())
        self.assertGreater(int((xla.view(np.uint32) != words).sum()), 0)
        self.assertEqual(self.advice(report), ["divisor_reciprocal"])

        def reciprocal_report(function):
            # The report where XLA multiplies by the reciprocal, and None where it divides as the
            # IR does, which compare then passes.
            report = self.verify(function, x, checks=COMPARE)
            if (np.asarray(function(x)).view(np.uint32) == words).all():
                self.assert_passes(report)
                return None
            self.assertEqual(self.advice(report), ["divisor_reciprocal"])
            return report

        # Eager x / 3.0 is the reciprocal product in JAX 0.11.1 and the quotient in 0.12.1.
        reciprocal_report(lambda v: v / 3.0)
        # Eager JAX divides by a closed-over array of threes as written, so the IR agrees.
        threes = jnp.full(4096, 3.0, dtype=jnp.float32)
        self.assert_passes(self.verify(lambda v: v / threes, x, checks=COMPARE))
        # jax.jit in JAX 0.11.1 folds the array and multiplies by its reciprocal, where primitive
        # by primitive XLA divides as the IR does. In 0.12.1 jax.jit divides too.
        report = reciprocal_report(self.jax.jit(lambda v: v / threes))
        if report is not None:
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
        # Under jax.jit the division is the reciprocal product in JAX 0.11.1 and 0.12.1.
        x = self.jnp.asarray((self.np.random.default_rng(1).standard_normal(4096) * 3)
                             .astype(self.np.float32))
        document = self.artifact(self.jax.jit(lambda v: v / 3.0), x)
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

    def test_integer_remainders_and_magnitudes(self):
        # lax.rem of integers is C's %, the remainder of the quotient truncated toward zero, and
        # lax.abs of a signed integer is -x below 0, both modulo 2^width: the remainder of the
        # least signed value and -1 is 0, which StableHLO leaves implementation-defined, and
        # that value is its own magnitude, as XLA's CPU code gives them. jnp's %, //, fmod and
        # sign build on them. Every width of both signs, against Python's integers and XLA's
        # words, eager and under jax.jit. A zero divisor fails at execution.
        from fractions import Fraction

        jax, lax, jnp, np = self.jax, self.jax.lax, self.jnp, self.np
        rng = np.random.default_rng(23)

        def functions(a, b):
            return lax.rem(a, b), jnp.abs(a), a % b, a // b, jnp.fmod(a, b), jnp.sign(a)

        def check(dtype):
            info, width = np.iinfo(dtype), 8 * np.dtype(dtype).itemsize
            a, b = (rng.integers(info.min, info.max, 64, dtype=dtype, endpoint=True)
                    for _ in range(2))
            b[:16] = rng.integers(1, 9, 16) * (rng.choice([-1, 1], 16) if info.min else 1)
            edges = ([(info.min, -1), (info.min, 3), (info.max, -1), (info.min, info.min)]
                     if info.min else [(info.max, 1), (info.max, info.max), (0, 5), (1, 2)])
            for k, (x, y) in enumerate(edges):
                a[16 + k], b[16 + k] = x, y
            b[b == 0] = 1
            with self.subTest(dtype=np.dtype(dtype).name):
                report = self.verify(functions, a, b, checks=COMPARE)
                self.assert_passes(report)
                pairs = [(int(x), int(y)) for x, y in zip(a, b)]
                expected = [[(x - y * int(Fraction(x, y))) % 2 ** width for x, y in pairs],
                            [abs(x) % 2 ** width for x, _ in pairs]]
                self.assertEqual([np.asarray(report["outputs"][k]["bits"], np.uint64).tolist()
                                  for k in (0, 1)], expected)
                if dtype in (np.int8, np.uint32):
                    self.assert_passes(self.verify(jax.jit(functions), a, b, checks=COMPARE))

        for dtype in (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32):
            check(dtype)
        with jax.enable_x64(True):
            check(np.int64)
            check(np.uint64)
        report = self.verify(lax.rem, np.array([5, -3], np.int32), np.array([0, 2], np.int32),
                             checks=COMPARE)
        self.assertFalse(report["ok"])
        self.assertIn("division by zero", {check["name"]: check["message"]
                                           for check in report["checks"]}["execute"])

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

    def test_special_functions_need_an_implementation(self):
        # Each unary special primitive imports as one Op.special step naming the SpecialFn
        # member by its torch.special name. No binary format has a rule for Op.special: as for
        # erf, the types check passes and execution refuses the step, in binary32 and, with
        # 64-bit mode on, binary64. Binary16 operands are refused at import.
        jax, lax, np = self.jax, self.jax.lax, self.np
        names = {lax.erfc: "erfc", lax.erf_inv: "erfinv", lax.lgamma: "gammaln",
                 lax.digamma: "digamma", lax.bessel_i0e: "i0e", lax.bessel_i1e: "i1e"}
        for function, name in names.items():
            for dtype in ("float32", "float64"):
                with self.subTest(function=function.__name__, dtype=dtype), \
                        jax.enable_x64(dtype == "float64"):
                    x = np.asarray(self.x, dtype=dtype)
                    document = torchlean._capture(function, (x,), "jax", EXECUTE)
                    self.assert_special(document, dtype, [3, 4], [(name, 0, 0)], [1])
                    report = torchlean.verify(document, checks=EXECUTE, checker=CHECKER)
                    self.assert_unsupported_execution(report, f"no rule at {dtype}")
        document = torchlean._capture(lambda x: lax.bessel_i1e(lax.erfc(x)), (self.x,), "jax",
                                      ("structure",))
        self.assert_special(document, "float32", [3, 4], [("erfc", 0, 0), ("i1e", 0, 1)], [2])
        report = self.verify(lax.lgamma, self.x.astype(self.jnp.float16), checks=("structure",))
        self.assert_refused(report, "only binary32 and binary64 operands")

    def test_polygamma_is_a_derivative_of_digamma(self):
        # lax.polygamma(m, x), where m holds one constant natural number, imports as one
        # Op.special step of digamma at order m, and execution refuses the step. When m has
        # more entries than x, x is broadcast to the result's shape first. Any other order is
        # refused at import.
        jax, lax, jnp, np = self.jax, self.jax.lax, self.jnp, self.np
        for dtype in ("float32", "float64"):
            for extents in ([3, 4], [1, 4]):
                with self.subTest(dtype=dtype, shape=extents), \
                        jax.enable_x64(dtype == "float64"):
                    x = np.asarray(self.x[:extents[0]], dtype=dtype)
                    document = torchlean._capture(
                        lambda x: lax.polygamma(jnp.full((3, 4), 2, x.dtype), x), (x,), "jax",
                        EXECUTE)
                    nodes = document["artifact"]["nodes"]
                    steps = [(node["op"]["record"]["tag"], node["op"]["record"]["fields"],
                              node["op"]["inputs"]) for node in nodes if "op" in node]
                    special = [step for step in steps if step[0] == "special"]
                    self.assertEqual(len(special), 1, steps)
                    self.assertEqual(special[0][1], [{"name": "digamma"}, {"nats": [2]},
                                                     {"shape": [3, 4]}])
                    [[operand, port]] = special[0][2]
                    if extents == [3, 4]:
                        self.assertEqual((operand, port), (0, 0))
                    else:
                        self.assertIn("op", nodes[operand], steps)
                        self.assertEqual((nodes[operand]["op"]["record"]["tag"],
                                          nodes[operand]["op"]["inputs"], port),
                                         ("broadcast", [[0, 0]], 0))
                    report = torchlean.verify(document, checks=EXECUTE, checker=CHECKER)
                    self.assert_unsupported_execution(report, f"no rule at {dtype}")
        for order in (lambda x: x, lambda x: jnp.full_like(x, 1.5),
                      lambda x: jnp.full_like(x, -1)):
            with self.subTest(order=order):
                report = self.verify(lambda x: lax.polygamma(order(x), x), self.x,
                                     checks=("structure",))
                self.assert_refused(report, "the order must be one constant natural number")

    def test_binary_special_functions_need_an_implementation(self):
        # Each binary special primitive imports as one Op.special step of its two operands,
        # naming the SpecialFn member by its torch.special name, at orders [0, 0], and
        # igamma_grad_a, the derivative of igamma in its first argument, at orders [1, 0] of
        # igamma. An operand with an axis of extent 1 where the result's is larger is broadcast
        # to the result's shape: the second, or both ([3, 1] and [1, 4] give [3, 4]). As for one
        # operand, execution refuses the step in binary32 and binary64, and binary16 operands
        # are refused at import.
        jax, lax, np = self.jax, self.jax.lax, self.np
        names = {lax.igamma: ("gammainc", [0, 0]), lax.igammac: ("gammaincc", [0, 0]),
                 lax.zeta: ("zeta", [0, 0]), lax.igamma_grad_a: ("gammainc", [1, 0])}
        for function, (name, orders) in names.items():
            for dtype in ("float32", "float64"):
                x, y = (np.asarray(value, dtype=dtype) for value in (self.x, self.y))
                for a, b in ((x, y), (x, y[:1]), (x[:, :1], y[:1])):
                    shapes = [list(a.shape), list(b.shape)]
                    with self.subTest(function=function.__name__, dtype=dtype, shapes=shapes), \
                            jax.enable_x64(dtype == "float64"):
                        document = torchlean._capture(function, (a, b), "jax", EXECUTE)
                        self.assert_broadcast_step(
                            document, dtype, "special", [{"name": name}, {"nats": orders}],
                            shapes, [3, 4])
                        report = torchlean.verify(document, checks=EXECUTE, checker=CHECKER)
                        self.assert_unsupported_execution(report, f"no rule at {dtype}")
        half = self.jnp.float16
        for function in (lax.zeta, lax.igamma_grad_a):
            with self.subTest(function=function.__name__, dtype="float16"):
                report = self.verify(function, self.x.astype(half), self.y.astype(half),
                                     checks=("structure",))
                self.assert_refused(report, "only binary32 and binary64 operands")

    def test_regularized_incomplete_beta_needs_an_implementation(self):
        # lax.betainc(a, b, x), the primitive regularized_incomplete_beta, imports as one
        # Op.special step of betainc of its three operands at orders [0, 0, 0]. An operand with
        # an axis of extent 1 where the result's is larger is broadcast to the result's shape:
        # none, all three ([3, 1], [1, 4] and [1, 1] give [3, 4]), or the second. As for one
        # operand, execution refuses the step in binary32 and binary64, and binary16 operands
        # are refused at import.
        jax, lax, np = self.jax, self.jax.lax, self.np
        for dtype in ("float32", "float64"):
            a, b, x = (np.asarray(np.abs(value), dtype=dtype)
                       for value in (self.x, self.y, self.x * self.y))
            for operands in ((a, b, x), (a[:, :1], b[:1], x[:1, :1]), (a, b[:1], x)):
                shapes = [list(operand.shape) for operand in operands]
                with self.subTest(dtype=dtype, shapes=shapes), jax.enable_x64(dtype == "float64"):
                    document = torchlean._capture(lax.betainc, operands, "jax", EXECUTE)
                    self.assert_broadcast_step(document, dtype, "special",
                                               [{"name": "betainc"}, {"nats": [0, 0, 0]}],
                                               shapes, [3, 4])
                    report = torchlean.verify(document, checks=EXECUTE, checker=CHECKER)
                    self.assert_unsupported_execution(report, f"no rule at {dtype}")
        report = self.verify(lax.betainc, *(np.abs(value).astype(np.float16)
                                            for value in (self.x, self.y, self.x * self.y)),
                             checks=("structure",))
        self.assert_refused(report, "only binary32 and binary64 operands")

    def test_convolutions_execute_with_their_payload(self):
        # The kernel operand of conv_general_dilated becomes the convolution payload, plain and
        # with dilation and groups, and so does a constant kernel that jnp.convolve and
        # jax.scipy.signal.convolve2d flip and broadcast, and jnp.correlate broadcasts, before
        # they convolve, or that a jitted function slices, negates, copies, concatenates,
        # transposes, reshapes, expands, reverses and squeezes. A kernel computed by arithmetic
        # is the convolution's operand instead
        # (JaxContractionTests.test_convolutions_take_a_kernel_not_known_at_capture). Integer
        # kernels and inputs make every summation order exact.
        jax, jnp, lax, np = self.jax, self.jnp, self.jax.lax, self.np
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
        taps = np.asarray(rng.integers(-3, 4, (4,)), np.float32)
        stencil = np.asarray(rng.integers(-3, 4, (2, 3)), np.float32)
        modes = ("full", "same", "valid")

        @jax.jit
        def arranged(w):
            w = jnp.concatenate([-w[:2], jnp.copy(w[2:])])
            w = lax.expand_dims(w.transpose(1, 0, 2, 3).reshape(2, 4, 9), (0,))
            return lax.squeeze(lax.rev(w, (2,)), (0,)).reshape(2, 4, 3, 3).transpose(1, 0, 2, 3)

        cases.update({
            "convolve": (lambda v: [jnp.convolve(v, taps, mode) for mode in modes],
                         integers((9,))),
            "correlate": (lambda v: [jnp.correlate(v, taps, mode) for mode in modes],
                          integers((9,))),
            "convolve2d": (lambda x: jax.scipy.signal.convolve2d(x, stencil), integers((5, 6))),
            "moved": (lambda x: lax.conv_general_dilated(x, arranged(plain), (1, 1), "VALID",
                                                         dimension_numbers=numbers),
                      integers((2, 2, 6, 6)))})
        for name, (function, x) in cases.items():
            with self.subTest(case=name):
                self.assert_passes(self.verify(function, x, checks=COMPARE))
        self.assertIn("rev", _names(jax.make_jaxpr(cases["convolve2d"][0])(
            integers((5, 6))).jaxpr))
        moved = _names(jax.make_jaxpr(cases["moved"][0])(integers((2, 2, 6, 6))).jaxpr)
        self.assertLessEqual({"slice", "neg", "copy", "concatenate", "transpose", "reshape",
                              "broadcast_in_dim", "rev", "squeeze"}, moved)
        self.assert_passes(self.verify(lambda v: jnp.convolve(v, jnp.asarray(taps) * 2.0),
                                       integers((9,)), checks=COMPARE))

    def test_cond_selects_the_branch_of_its_index(self):
        # lax.cond and lax.switch bind cond, which is Lean's Flow.cond: only the branch that its
        # index selects runs, as StableHLO's case selects it (lax.switch clamps its index first).
        # The branches scale, shift, negate and swap their operands, with several outputs, a
        # constant of their own, a nested switch and a complex value's conjugate, all in exact or
        # once-rounded arithmetic, so XLA's words are the IR's. No public function binds an index
        # outside [0, n), so cond is also bound directly, where XLA, as StableHLO, selects the
        # last branch. An index known at capture, here the result of a chain of more than 1,500
        # integer equations on constants, longer than Python's recursion limit, selects its
        # branch there, and only that branch is imported, without a flow.
        from jax._src.lax.control_flow.conditionals import cond_p
        from torchlean.jaxpr import export

        jax, jnp, lax, np = self.jax, self.jnp, self.jax.lax, self.np
        x, y = self.x, self.y
        scale = jnp.asarray([4.0, 0.5, -1.0, 2.0], dtype=jnp.float32)

        def two(p, a, b):
            return lax.cond(p > 0, lambda u, v: (u * 2.0, v - u), lambda u, v: (v, -u), a, b)

        def three(i, a):
            return lax.switch(i, [lambda u: u + 1.0, lambda u: u * scale, lambda u: -u], a)

        def nested(p, i, a):
            return lax.cond(p > 0, lambda u: lax.switch(i, [lambda v: v * 0.5,
                                                            lambda v: v + 3.0], u),
                            lambda u: u - 1.0, a)

        def conjugated(p, a, b):
            z = lax.cond(p > 0, lambda c: c, lambda c: jnp.conj(c), lax.complex(a, b))
            return jnp.real(z), jnp.imag(z)

        branches = tuple(jax.make_jaxpr(f)(x) for f in (lambda u: u + 1.0, lambda u: u * 2.0,
                                                        lambda u: -u))

        def bound(i, a):
            return cond_p.bind(i, a, branches=branches)[0]

        signs = (np.float32(1), np.float32(-1))
        cases = [("two", two, [(s, x, y) for s in signs]),
                 ("three", three, [(np.int32(i), x) for i in (-1, 0, 1, 2, 3)]),
                 ("nested", nested, [(s, np.int32(i), x) for s in signs for i in (0, 1)]),
                 ("conjugated", conjugated, [(s, x, y) for s in signs]),
                 ("bound", bound, [(np.int32(i), x) for i in (-1, 0, 1, 3, 7)])]
        for name, function, argument_sets in cases:
            for arguments in argument_sets:
                scalars = [arguments[0].item(), *[a.item() for a in arguments[1:2]
                                                  if np.ndim(a) == 0]]
                with self.subTest(case=name, scalars=scalars):
                    self.assertIn("cond", {equation.primitive.name for equation in
                                           jax.make_jaxpr(function)(*arguments).eqns})
                    self.assertIn("cond", _flows(export(function, *arguments)))
                    self.assert_passes(self.verify(function, *arguments, checks=COMPARE))

        def chained(a):
            k = jnp.int32(3)
            for _ in range(1500):
                k = k + 0
            return lax.cond(k > 2, lambda u: u * 2.0, lambda u: u + 1.0, a)

        self.assertIn("cond", _names(jax.make_jaxpr(chained)(x)))
        self.assertEqual(_flows(export(chained, x)), [])
        report = self.verify(chained, x, checks=COMPARE)
        self.assert_passes(report)
        self.assertTrue(np.array_equal(_outputs(report)[0].numpy(), np.asarray(x) * 2.0))

    def test_cond_runs_only_the_selected_branch(self):
        # Flow.cond runs only the branch that its index selects, as JAX does, so an unselected
        # branch never fails at execution: neither its integer division by zero, nor its
        # conversion of an infinity or a NaN to an integer type, nor its failing read of
        # rng_uniform's empty bounds or of top_k's NaN, nor its loop that never stops. Where the
        # branch is selected, it fails as it does outside a cond.
        jnp, lax, np = self.jnp, self.jax.lax, self.np
        integers = np.arange(-6, 6, dtype=np.int32).reshape(3, 4)

        def divided(a, d):
            return lax.cond(d != 0, lambda u, e: lax.div(u, lax.broadcast(e, u.shape)),
                            lambda u, e: u, a, d)

        def converted(s):
            return lax.cond(jnp.isfinite(s) & (jnp.abs(s) < 2.0 ** 31),
                            lambda v: v.astype(jnp.int32), lambda v: jnp.int32(7), s)

        def drawn(low, high):
            return lax.cond(low < high, lambda u, v: lax.rng_uniform(u, v, (4,)) * 0.0,
                            lambda u, v: jnp.ones(4, jnp.float32), low, high)

        def ranked(a):
            return lax.cond(jnp.any(jnp.isnan(a)), lambda u: jnp.zeros((3, 2), u.dtype),
                            lambda u: lax.top_k(u, 2)[0], a)

        def nested(p, a, d):
            # The inner cond divides by d where d < 1, so by zero for d == 0, which runs only
            # where p > 0.
            return lax.cond(p > 0, lambda u, e: lax.cond(e < 1, lambda v, f: lax.div(
                v, lax.broadcast(f, v.shape)), lambda v, f: v, u, e), lambda u, e: u, a, d)

        def endless(a):
            # Selected where an entry is a NaN: a loop whose counter stays 0, below its bound 1.
            return lax.cond(jnp.isnan(a[0, 0]), lambda u: lax.while_loop(
                lambda c: c[0] < 1, lambda c: (c[0] * 2, c[1] * 2.0), (0, u))[1],
                lambda u: u + 1.0, a)

        cases = [("divided", divided, [(integers, np.int32(d)) for d in (0, 3, -2)]),
                 ("converted", converted, [(np.float32(s),) for s in
                                           (np.inf, -np.inf, np.nan, 1e10, -3.75, 2.5)]),
                 ("ranked", ranked, [(np.asarray(self.x),),
                                     (np.where(np.arange(12).reshape(3, 4) == 6, np.nan,
                                               np.asarray(self.x)).astype(np.float32),)]),
                 ("nested", nested, [(np.float32(p), integers, np.int32(d))
                                     for p, d in ((-1, 0), (1, -2), (-1, -2), (1, 3))]),
                 ("endless", endless, [(np.asarray(self.x),)])]
        for name, function, argument_sets in cases:
            for arguments in argument_sets:
                with self.subTest(case=name, arguments=[np.ravel(a)[:1].tolist()
                                                        for a in arguments]):
                    self.assert_passes(self.verify(function, *arguments, checks=COMPARE))
        for low, high in ((2.0, 1.0), (0.0, 0.0), (0.0, 1.0)):
            with self.subTest(case="drawn", low=low, high=high):
                report = self.verify(drawn, np.float32(low), np.float32(high), checks=EXECUTE)
                self.assert_passes(report)
                self.assertEqual(_outputs(report)[0].tolist(), [1.0 if low >= high else 0.0] * 4)
        selected = (lambda a, d: lax.cond(d == 0, lambda u, e: lax.div(u, lax.broadcast(
            e, u.shape)), lambda u, e: u, a, d))
        self.assert_unsupported_execution(self.verify(selected, integers, np.int32(0)),
                                          "integer division by zero")

    def test_what_gives_no_returned_value_is_dropped_before_the_flow(self):
        # A node or a flow that gives no value the function returns is dropped before the flow is
        # written (torchlean._flow.prune), as JAX's dead-code elimination drops it, so that no
        # block is without outputs. A cond whose branches return nothing, one of them computing
        # an unused x + 1, then 7, has no flow left, and its artifact is that of the other nodes,
        # as is that of a while loop whose results are unused. A branch's unused value is not
        # computed, so its operand does not pass into the cond.
        from torchlean.jaxpr import export
        jnp, lax, np = self.jnp, self.jax.lax, self.np
        a = np.asarray(self.x)

        def unused(p, x):
            lax.cond(p, lambda v: (v + 1, ())[1], lambda v: (), x)
            return jnp.int32(7)

        def looped(u):
            lax.while_loop(lambda c: c[0] < c[1].sum(), lambda c: (c[0] + 1.0, c[1]), (0.0, u))
            return u + 1.0

        def partly(p, u, d):
            return lax.cond(p > 0, lambda v, e: (v * 2.0, e + 1)[0], lambda v, e: v, u, d)

        self.assertIn("cond", _names(self.jax.make_jaxpr(unused)(np.int32(1), np.int32(3))))
        self.assertEqual(_flows(export(unused, np.int32(1), np.int32(3))), [])
        for p in (0, 1):
            with self.subTest(case="unused", p=p):
                report = self.verify(unused, np.int32(p), np.int32(3), checks=COMPARE)
                self.assert_passes(report)
                self.assertEqual(_outputs(report)[0].tolist(), 7)
        self.assertEqual(_flows(export(looped, a)), [])
        report = self.verify(looped, a, checks=COMPARE)
        self.assert_passes(report)
        self.assertTrue(np.array_equal(_outputs(report)[0].numpy(), a + np.float32(1.0)))
        document = export(partly, np.float32(1), a, np.int32(2))
        self.assertEqual(_flows(document), ["cond"])
        # The index and u before the cond, u * 2.0 in its second branch; the first passes u.
        self.assertEqual([len(block["outputs"]) for block in _blocks(document["artifact"]["flow"])],
                         [2, 1])
        for p in (-1.0, 1.0):
            with self.subTest(case="partly", p=p):
                self.assert_passes(self.verify(partly, np.float32(p), a, np.int32(2),
                                               checks=COMPARE))

    def test_an_eager_cond_evaluates_its_selected_branch(self):
        # The eager reference of a cond evaluates the branch that its index selects one
        # primitive at a time (torchlean.jaxpr.evaluate), the last branch for an index outside
        # [0, n), which only a direct binding gives. The document of a flow has no record of
        # XLA's computations, since the checker emulates XLA's CPU code only for an artifact
        # without control flow, and the branches' steps are in blocks of their own.
        from jax._src.lax.control_flow.conditionals import cond_p
        from torchlean.jaxpr import evaluate

        jax, lax, np = self.jax, self.jax.lax, self.np

        def function(p, a):
            return lax.cond(p > 0, lambda u: u / 3.0, lambda u: u / 5.0, a)

        for p in (np.float32(1), np.float32(-1)):
            with self.subTest(p=p.item()):
                document = self.artifact(function, p, self.x)
                self.assertEqual(len(self.tagged(document, "div")), 2)
                self.assertEqual([len(self.tagged(dict(artifact=block), "div"))
                                  for block in _blocks(document["artifact"]["flow"])],
                                 [0, 1, 1])
                self.assertFalse({"computation", "jit_computations"} & set(document))
                self.assertEqual(len(document["eager_reference_outputs"]), 1)
                (value,) = evaluate(jax.make_jaxpr(function)(p, self.x), p, self.x)
                self.assertTrue(np.array_equal(value, self.x / (3.0 if p > 0 else 5.0)))
        branches = tuple(jax.make_jaxpr(f)(self.x) for f in (lambda u: u / 3.0, lambda u: u / 5.0))

        def bound(i, a):
            return cond_p.bind(i, a, branches=branches)[0]

        for i in (np.int32(-1), np.int32(2)):
            (value,) = evaluate(jax.make_jaxpr(bound)(i, self.x), i, self.x)
            self.assertTrue(np.array_equal(value, self.x / 5.0))

    def test_captured_types_survive_a_local_x64_context(self):
        # The Jaxpr keeps int64 constants after the function's enable_x64 context exits.
        # Folding them under the caller's 32-bit default used to select the wrong branch
        # or reject a zero-iteration loop with an int32/int64 mismatch.
        from torchlean.jaxpr import evaluate

        jax, jnp, lax, np = self.jax, self.jnp, self.jax.lax, self.np

        def branched(x):
            with jax.enable_x64():
                large = jnp.int64(2 ** 40) + jnp.int64(1)
                return lax.cond(large > 10, lambda a: a + 1, lambda a: a - 1, x)

        def looped(x):
            with jax.enable_x64():
                return lax.while_loop(lambda c: c[0] < 10,
                                      lambda c: (c[0] + 1, c[1] + 1),
                                      (jnp.int64(2 ** 40), x))[1]

        def returned64(x):
            with jax.enable_x64():
                return x.astype(jnp.float64) + jnp.float64(2 ** 40)

        with jax.enable_x64(False):
            cases = ((branched, 3.0, np.float32, np.uint32),
                     (looped, 2.0, np.float32, np.uint32),
                     (returned64, 2 ** 40 + 2, np.float64, np.uint64))
            for function, expected, dtype, words in cases:
                with self.subTest(function=function.__name__):
                    x = np.float32(2.0)
                    (value,) = evaluate(jax.make_jaxpr(function)(x), x)
                    self.assertEqual(float(value), expected)
                    report = self.verify(function, x, checks=COMPARE)
                    self.assert_passes(report)
                    self.assertEqual(report["outputs"][0]["bits"],
                                     [dtype(expected).view(words).item()])
                    self.assertEqual(_flows(self.artifact(function, x)), [])
                    self.assertFalse(jax.config.x64_enabled)
            (value,) = evaluate(jax.make_jaxpr(lambda n: n + 1)(2), 2)
            self.assertEqual((value.dtype, int(value)), (np.dtype("int32"), 3))
            with jax.enable_x64():
                integer = np.int64(2 ** 40)
                closed = jax.make_jaxpr(lambda n: n + 1)(integer)
            (value,) = evaluate(closed, integer)
            self.assertEqual((value.dtype, int(value)), (np.dtype("int64"), 2 ** 40 + 1))
            self.assertFalse(jax.config.x64_enabled)

    def test_platform_dependent_is_the_branch_of_cpu_and_cuda(self):
        # lax.platform_dependent binds platform_index, the position of the branch for the
        # platform that JAX lowers for, and a cond on it. The IR stands for JAX on the CPU and on
        # CUDA, so the import is the one branch that both select, and only that branch. For a
        # square matrix, jnp.diagonal and jnp.diag are the default branch's gather, whose words
        # are the diagonal's, where the other branch's sum of products (for Pallas's "mosaic")
        # gives NaN for an infinity or a NaN off the diagonal and +0 for a -0 on it. A branch
        # for another platform is not imported, so a refused one (qr's several outputs) does
        # not refuse the function. Platforms that select different branches or none are refused.
        jax, jnp, lax, np = self.jax, self.jnp, self.jax.lax, self.np

        square = np.arange(16, dtype=np.float32).reshape(4, 4)
        square[0, 1], square[2, 0], square[3, 1], square[2, 2] = np.inf, -np.inf, np.nan, -0.0
        for name, function in (("diagonal", jnp.diagonal), ("diag", jnp.diag)):
            with self.subTest(function=name):
                self.assertIn("platform_index", _names(jax.make_jaxpr(function)(square)))
                self.assertEqual(self.tagged(self.artifact(function, square), "mul"), [])
                report = self.verify(function, square, checks=COMPARE)
                self.assert_passes(report)
                self.assertTrue(np.array_equal(_outputs(report)[0].numpy().view(np.uint32),
                                               np.diagonal(square).view(np.uint32)))
        x = np.asarray(self.x)

        def double(u):
            return u * 2.0

        def negated(u):
            return -u

        imported = {"cpu and cuda": dict(cpu=double, cuda=double, default=negated),
                    "default": dict(tpu=lambda u: jnp.linalg.qr(u.T)[0].T, default=double)}
        for name, branches in imported.items():
            with self.subTest(case=name):
                function = (lambda a, branches=branches: lax.platform_dependent(a, **branches))
                report = self.verify(function, x, checks=COMPARE)
                self.assert_passes(report)
                self.assertTrue(np.array_equal(_outputs(report)[0].numpy(), x * 2.0))
        refused = {"cpu and the default for cuda": dict(cpu=double, default=negated),
                   "neither": dict(tpu=double, rocm=negated)}
        for name, branches in refused.items():
            with self.subTest(case=name):
                function = (lambda a, branches=branches: lax.platform_dependent(a, **branches))
                self.assert_refused(self.verify(function, x),
                                    "platform_index: the platforms cpu, cuda, on which the IR runs")

    def test_scan_is_its_body_once_per_iteration(self):
        # scan is its body imported once per iteration, the carry passed on and the other outputs
        # stacked along a new leading axis, as lax.scan's reference implementation loops, so each
        # iteration's counter is a constant. The scans carry several values, run forwards and in
        # reverse, close over a constant, read no operand for a fixed length, run once, divide
        # under a cond in each iteration where the divisor is not zero, and are lax.fori_loop's,
        # all in exact or once-rounded arithmetic, so XLA's words are the IR's, and scan rows
        # without entries. Only the conds on the scanned data are flows, where the scan is one
        # Flow.scan of its body (test_a_scan_of_conds_on_its_data_is_one_flow). A cond on the
        # counter selects its branch at capture, so a branch that the counter never selects is not
        # imported: one that would compute a QR factorization, which is refused, runs too. A
        # scan of length zero, with or without scanned operands, of floats or integers, forwards
        # or in reverse, never runs its body: its outputs are the carry and empty stacked outputs.
        from torchlean.jaxpr import export

        jax, jnp, lax, np = self.jax, self.jnp, self.jax.lax, self.np
        x, y = self.x, self.y
        scale = jnp.asarray([4.0, 0.5, -1.0, 2.0], dtype=jnp.float32)
        integers = np.arange(-6, 6, dtype=np.int32).reshape(3, 4)

        def running(a, b):
            def step(carry, rows):
                (product, total), (u, v) = carry, rows
                return (product * u, total + v), (product - v) * scale

            return lax.scan(step, (jnp.ones(4, jnp.float32), jnp.zeros(4, jnp.float32)), (a, b))

        def backwards(a):
            return lax.scan(lambda c, u: (c * 0.5 + u, c), jnp.zeros(4, jnp.float32), a,
                            reverse=True)

        def counted(a):
            return lax.scan(lambda c, _: ((c[0] + 1, c[1] * 2.0 + c[0].astype(jnp.float32)),
                                          c[0]), (jnp.int32(0), a), None, length=3)

        def once(a):
            return lax.scan(lambda c, u: (c + u, c * u), jnp.zeros(4, jnp.float32), a[:1])

        def divided(a, d):
            return lax.scan(lambda c, e: (c, lax.cond(e != 0, lambda u, f: lax.div(
                u, lax.broadcast(f, u.shape)), lambda u, f: u, c, e)), a, d)

        def looped(a):
            return lax.fori_loop(0, 4, lambda i, c: c * 2.0 - i.astype(jnp.float32), a)

        def factored(length):
            # The counter is 0 and 1 in the first two iterations, which add 1; a third one
            # replaces the carried matrix by its Q factor.
            return lambda a: lax.scan(lambda c, _: ((c[0] + 1, lax.cond(
                c[0] < 2, lambda u: u + 1.0, lambda u: jnp.linalg.qr(u)[0], c[1])), None),
                (jnp.int32(0), a), None, length=length)[0][1]

        def empty(a):
            return lax.scan(lambda c, _: (c, c), a, None, length=0)

        def unread(c, a):
            return lax.scan(lambda c, u: (c + u, c * u), c, a[:0])

        def unread_backwards(c, a):
            return lax.scan(lambda c, u: (c * 0.5 + u, c - u), c, a[:0], reverse=True)

        def hollow(a):
            return lax.scan(lambda c, u: (c + u.sum(), u * 2.0), jnp.float32(0), a)

        square = np.eye(4, dtype=np.float32)
        cases = [("running", running, (x, y), []), ("backwards", backwards, (x,), []),
                 ("counted", counted, (x,), []), ("once", once, (x,), []),
                 ("divided", divided, (integers[0], np.asarray([3, 0, -2], np.int32)),
                  ["cond", "scan"]),
                 ("looped", looped, (x,), []), ("factored", factored(2), (square,), []),
                 ("empty", empty, (x,), []),
                 ("hollow", hollow, (np.zeros((3, 0, 2), np.float32),), []),
                 ("unread integers", unread, (integers[0], integers), []),
                 ("unread backwards", unread_backwards, (x[0], x), []),
                 ("unread", unread, (x[0], x), [])]
        for name, function, arguments, flows in cases:
            with self.subTest(case=name):
                self.assertIn("scan", {equation.primitive.name for equation in
                                       jax.make_jaxpr(function)(*arguments).eqns})
                self.assertEqual(_flows(export(function, *arguments)), flows)
                report = self.verify(function, *arguments, checks=COMPARE)
                self.assert_passes(report)
        self.assertEqual(list(_outputs(report)[1].shape), [0, 4])
        self.assert_refused(self.verify(factored(3), square), "qr: multiple primitive outputs")

    def test_a_scan_of_conds_on_its_data_is_one_flow(self):
        # A scan with a cond on its carry or its rows in each iteration, none of whose iterations
        # would decide a flow at capture, is one Flow.scan of its body, which reads row i of each
        # scanned operand, row length - 1 - i in reverse, at a counter i and writes its stacked
        # outputs there: its document is the size of one iteration at 1,000 and at 100,000 of
        # them. So is a scan whose carry starts at a constant, and lax.fori_loop's, whose counter
        # is known at capture but decides no cond, even where it is added to the data a cond
        # reads. Where a cond after the scan reads the last counter, the scan is unrolled, its
        # iterations in a seq of halves (torchlean._flow._sequence), so that 1,000 of them nest a
        # few levels. Python's json and deepcopy, which recurse, take either within their
        # recursion limit.
        import copy
        import json

        from torchlean.jaxpr import export

        jnp, lax, np = self.jnp, self.jax.lax, self.np

        def flip(c):
            return lax.cond(c > 0, lambda y: y + 1, lambda y: y - 1, c)

        def stepped(length):
            return lambda c: lax.scan(lambda c, _: (flip(c), None), c, None, length=length)[0]

        def mapped(reverse):
            def step(c, x):
                c = lax.cond(c.sum() > 0, lambda y: y * x, lambda y: y + x, c)
                return c, c.sum() * 2.0
            return lambda c, xs: lax.scan(step, c, xs, reverse=reverse)

        def counted(c):
            return lax.fori_loop(0, 1000, lambda i, c: flip(c), c)

        def added(c):
            return lax.fori_loop(0, 1000, lambda i, c: flip(c + (i & 1)), c)

        def started(xs):
            def step(c, x):
                c = lax.cond(c.sum() > 0, lambda y: y * x, lambda y: y + x, c)
                return c, c.sum()
            return lax.scan(step, jnp.zeros(3, jnp.float32), xs)

        def returned(c):
            (i, c), _ = lax.scan(lambda s, _: ((s[0] + 1, flip(s[1])), None),
                                 (jnp.int32(0), c), None, length=1000)
            return lax.cond(i > 5, lambda y: y * 2, lambda y: y, c)

        def depth(flow):
            deepest, pending = 0, [(flow, 1)]
            while pending:
                (tag, *parts), level = pending.pop()
                deepest = max(deepest, level)
                pending += [(part, level + 1) for part in parts if isinstance(part, list)]
            return deepest

        signs = np.where(np.arange(1000) % 3 == 0, -1.0, 1.0).astype(np.float32)
        rows = np.stack([signs, -signs, np.ones(1000, np.float32)], axis=1)
        cases = [("1,000", stepped(1000), (np.int32(3),), ["cond", "scan"], 3 + 1000),
                 ("100,000", stepped(100000), (np.int32(-3),), ["cond", "scan"], -3 - 100000),
                 ("rows", mapped(False), (np.ones(3, np.float32), rows), ["cond", "scan"], None),
                 ("reversed", mapped(True), (np.ones(3, np.float32), rows), ["cond", "scan"],
                  None),
                 ("counted", counted, (np.int32(-1),), ["cond", "scan"], -1 - 1000),
                 ("added", added, (np.int32(-1),), ["cond", "scan"], None),
                 ("started", started, (rows,), ["cond", "scan"], None),
                 ("returned", returned, (np.int32(-1),), ["cond"] * 1000, 2 * (-1 - 1000))]
        for name, function, arguments, flows, last in cases:
            with self.subTest(case=name):
                document = export(function, *arguments)
                self.assertEqual(_flows(document), flows)
                self.assertLessEqual(depth(document["artifact"]["flow"]), 24)
                self.assertLess(len(json.dumps(document)), 3_000_000)
                self.assertEqual(copy.deepcopy(document), document)
                report = self.verify(function, *arguments, checks=COMPARE)
                self.assert_passes(report)
                if last is not None:
                    self.assertEqual(_outputs(report)[0].tolist(), last)

    def test_a_scan_that_its_flow_refuses_is_unrolled(self):
        # A Flow.scan of a scan is a trial (torchlean.jaxpr._Builder.looped): a scan of rows with
        # a zero-sized axis, and one where the trial is refused, here by a refusal put in the
        # gather and scatter of its rows (torchlean._flow.entry), import exactly as the unrolled
        # scan does.
        from unittest import mock

        from torchlean import _flow
        from torchlean.jaxpr import Unsupported, _Builder, export

        lax, np = self.jax.lax, self.np

        def scanned(c, xs):
            def step(c, row):
                return lax.cond(c > 0, lambda z: z + 1, lambda z: z - 1, c), row * 2
            return lax.scan(step, c, xs)

        def refused(*_):
            raise Unsupported("scan: refused in the trial")

        rows = np.ones((4, 2), np.float32)
        self.assertEqual(_flows(export(scanned, np.int32(3), rows)), ["cond", "scan"])
        cases = [("zero-sized rows", np.zeros((3, 0, 2), np.float32), None),
                 ("refused trial", rows, refused)]
        for name, xs, refusal in cases:
            with self.subTest(case=name):
                with mock.patch.object(_Builder, "looped", lambda *_: None):
                    unrolled = export(scanned, np.int32(3), xs)
                with mock.patch.object(_flow, "entry", refusal or _flow.entry):
                    document = export(scanned, np.int32(3), xs)
                self.assertEqual(_flows(document), ["cond"] * len(xs))
                self.assertEqual(_without_jax_object_ids(document),
                                 _without_jax_object_ids(unrolled))
                self.assert_passes(self.verify(scanned, np.int32(3), xs, checks=COMPARE))

    def test_a_scan_whose_known_carry_decides_a_flow_is_unrolled(self):
        # A scan whose carry known at capture would decide a flow in an unrolled iteration
        # imports as the unrolled scan does (torchlean.jaxpr._Builder.trial),
        # as one whose last counter a cond after it reads does
        # (test_a_scan_of_conds_on_its_data_is_one_flow): one with carry b that each iteration
        # computes as 2 * a from the counter a, so that the cond on b is decided from the second
        # iteration on; a fori_loop nested in another whose cond reads both counters, which
        # decide it; and one with a convolution's kernel, negated in each iteration (JAX makes a
        # kernel carried unchanged a constant of the scan), which a known kernel makes a
        # constant. So does a scan that draws from rng_uniform, whose unrolled iterations draw
        # under keys of their own, which one region's draws would repeat.
        from unittest import mock

        from torchlean.jaxpr import _Builder, export

        jnp, lax, np = self.jnp, self.jax.lax, self.np

        def doubled(b, c):
            def step(s, _):
                a, b, c = s
                return (a + 1, a * 2, lax.cond(b > 2, lambda y: y + 1, lambda y: y - 1, c)), None
            return lax.scan(step, (jnp.int32(0), b, c), None, length=4)[0][2]

        def nested(c):
            return lax.fori_loop(0, 4, lambda i, c: lax.fori_loop(0, 3, lambda j, c: lax.cond(
                i + j < 3, lambda y: y + 1, lambda y: y * 2, c), c), c)

        def convolved(x):
            def step(s, _):
                w, y = s
                y = lax.conv_general_dilated(y[None, None], w, (1,), [(1, 1)])[0, 0]
                return (-w, lax.cond(y.sum() > 0, lambda z: z * 0.5, lambda z: z + 1.0, y)), None
            return lax.scan(step, (jnp.ones((1, 1, 3), jnp.float32), x), None, length=3)[0][1]

        def drawn(c):
            def step(c, _):
                u = lax.rng_uniform(jnp.float32(0), jnp.float32(1), (2,))
                return lax.cond(c.sum() > 0, lambda y: y + u, lambda y: y - u, c), None
            return lax.scan(step, c, None, length=4)[0]

        cases = [("doubled", doubled, (np.int32(5), np.int32(0)), ["cond"], COMPARE),
                 ("nested", nested, (np.int32(1),), [], COMPARE),
                 ("convolved", convolved, (np.arange(5, dtype=np.float32),), ["cond"] * 3,
                  COMPARE),
                 ("drawn", drawn, (np.ones(2, np.float32),), ["cond"] * 4, EXECUTE)]
        for name, function, arguments, flows, checks in cases:
            with self.subTest(case=name):
                with mock.patch.object(_Builder, "looped", lambda *_: None):
                    unrolled = export(function, *arguments)
                document = export(function, *arguments)
                self.assertEqual(_flows(document), flows)
                self.assertEqual(_without_jax_object_ids(document),
                                 _without_jax_object_ids(unrolled))
                self.assert_passes(self.verify(function, *arguments, checks=checks))
        keys = {record["fields"][1]["nat"]
                for record in _records(self.artifact(drawn, np.ones(2, np.float32)))
                if record["tag"] == "randUniform"}
        self.assertEqual(len(keys), 4)

    def test_an_eager_scan_is_one_computation(self):
        # XLA compiles an eager scan whole, as it compiles a jax.jit call, so the steps of every
        # iteration are in one computation among the jit_computations, and the eager reference
        # is the body evaluated one primitive at a time in each iteration.
        from torchlean.jaxpr import evaluate

        jax, jnp, lax, np = self.jax, self.jnp, self.jax.lax, self.np

        for reverse in (False, True):
            def function(a):
                return lax.scan(lambda c, u: (c / 3.0 + u, c), jnp.zeros(4, jnp.float32), a,
                                reverse=reverse)

            with self.subTest(reverse=reverse):
                document = self.artifact(function, self.x)
                steps = self.tagged(document, "div")
                self.assertEqual(len(steps), 3)
                self.assertEqual({document["computation"][k] for k in steps},
                                 set(document["jit_computations"]))
                self.assertEqual(len(document["jit_computations"]), 1)
                self.assertEqual(len(document["eager_reference_outputs"]), 2)
                carry, rows = jnp.zeros(4, jnp.float32), []
                for row in self.x[::-1] if reverse else self.x:
                    rows.append(carry)
                    carry = carry / 3.0 + row
                last, stacked = evaluate(jax.make_jaxpr(function)(self.x), self.x)
                self.assertTrue(np.array_equal(last, carry))
                self.assertTrue(np.array_equal(stacked, jnp.stack(rows[::-1] if reverse
                                                                  else rows)))

    def test_while_is_its_body_per_iteration_decided_at_capture_or_a_flow(self):
        # A while loop whose predicate reads only values known at capture is its body imported
        # once per iteration it runs, counted at capture by running the predicate and the parts
        # of the body that compute what it reads, and has no flow. The loops count from a literal
        # by one and by two, read a closed-over bound, close over an input in the body, step by
        # a carried stride that a closed-over constant widens, run no iteration, and divide under
        # a cond where the divisor, 1 - i, is not zero, a cond that the counter decides. Bounds
        # that integer operations compute from constants alone are folded at capture, also
        # through a chain of 1,501 results. A loop of no iteration imports no body, so one that
        # would compute a QR factorization, which is refused, runs too, also with its predicate
        # and body under jax.jit. Another loop is Lean's Flow.whileLoop, which runs its body while
        # its test holds, at execution, as the Python loop that gives lax.while_loop's semantics
        # runs it, its body's steps in the flow once: a bound that is an input or a conversion of
        # a floating constant, and loops that stop on their data, halving until an entry is at
        # most 1, Newton's iteration for a square root in binary32 and binary64 until an iterate
        # repeats, and jnp.gcd and jnp.lcm, which loop while a remainder is not zero, at scalars,
        # at 0 and at vectors whose lanes stop together. All are in exact or once-rounded
        # arithmetic, so XLA's words are the IR's.
        from torchlean.jaxpr import export

        jax, jnp, lax, np = self.jax, self.jnp, self.jax.lax, self.np
        x = self.x
        bound = jnp.asarray([3, 7], dtype=jnp.int32)
        widening = jnp.asarray([1], dtype=jnp.int32)
        pair = jnp.asarray([2, 5], dtype=jnp.int32)
        scale = jnp.asarray([3.0], dtype=jnp.float32)
        integers = np.arange(-6, 6, dtype=np.int32).reshape(3, 4)

        def counted(a):
            return lax.while_loop(lambda c: c[0] < 4, lambda c: (
                c[0] + 1, c[1] * 2.0 + c[0].astype(jnp.float32)), (0, a))

        def stepped(a):
            return lax.while_loop(lambda c: c[0] < 5, lambda c: (c[0] + 2, c[1] * 0.5),
                                  (jnp.int32(0), a))

        def bounded(a):
            return lax.while_loop(lambda c: c[0] < bound[0], lambda c: (c[0] + 1, c[1] - a),
                                  (0, a))

        def strided(a):
            # The counter goes 0, 1, 3, 6 and 10: the predicate reads it, its step reads the
            # stride, and the stride's step reads a constant of the body.
            return lax.while_loop(lambda c: c[0] < 9, lambda c: (
                c[0] + c[1], c[1] + widening[0], c[2] * 2.0), (0, 1, a))

        def skipped(a):
            return lax.while_loop(lambda c: c[0] < 0, lambda c: (c[0] + 1, c[1] * 2.0), (0, a))

        def factored(bound, wrap=lambda f: f):
            # Replace the carried matrix by its Q factor while the counter is below `bound`; the
            # comparison may be a call of its own, on the counter.
            return lambda a: lax.while_loop(lambda c: wrap(lambda i: i < bound)(c[0]), lambda c: (
                c[0] + 1, jnp.linalg.qr(c[1])[0]), (0, a))

        def divided(a):
            # The divisors are 1, 0 and -1; the cond skips the division by zero.
            def step(c):
                i, u = c
                return i + 1, lax.cond(1 - i != 0, lambda v, e: lax.div(
                    v, lax.broadcast(e, v.shape)), lambda v, e: v, u, 1 - i)

            return lax.while_loop(lambda c: c[0] < 3, step, (0, a))

        def folded(a):
            # The bound is max(5 - 2, 1) of slices of a closed-over array: 3 iterations.
            return lax.fori_loop(0, jnp.maximum(pair[1] - pair[0], 1), lambda i, c: c * 2.0, a)

        def multiplied(a):
            return lax.fori_loop(0, jnp.int32(2) * 2, lambda i, c: c - 1.0, a)

        def chained(a):
            # The bound is the counter of a loop of 1,500 iterations, less 1,497: 3 iterations,
            # folded through a chain of 1,501 integer results, longer than Python's recursion.
            n = lax.while_loop(lambda i: i < 1500, lambda i: i + 1, jnp.int32(0))
            return lax.fori_loop(0, n - 1497, lambda i, c: c * 2.0, a)

        def given(m, a):
            return lax.fori_loop(0, m, lambda i, c: c * 2.0, a)

        def converted(a):
            return lax.fori_loop(0, scale[0].astype(jnp.int32), lambda i, c: c * 2.0, a)

        def halved(a):
            return lax.while_loop(lambda c: c[0, 0] > 1.0, lambda c: c * 0.5, a)

        def newton(a):
            # x -> (x + a / x) / 2 from a, until an iterate equals the one before it.
            return lax.while_loop(lambda c: c[0] != c[1], lambda c: (
                0.5 * (c[0] + a / c[0]), c[0]), (a, a + 1.0))[0]

        def scalars(*values):
            return tuple(np.int32(v) for v in values)

        square = np.eye(4, dtype=np.float32)
        decided = [("counted", counted, (x,)), ("stepped", stepped, (x,)),
                   ("bounded", bounded, (x,)), ("strided", strided, (x,)),
                   ("skipped", skipped, (x,)), ("divided", divided, (integers[0],)),
                   ("folded", folded, (x,)), ("multiplied", multiplied, (x,)),
                   ("chained", chained, (x,)), ("factored", factored(0), (square,)),
                   ("factored under jit", factored(0, jax.jit), (square,))]
        flows = [("given", given, (np.int32(3), x)), ("converted", converted, (x,)),
                 ("halved", halved, (np.arange(1, 13, dtype=np.float32).reshape(3, 4) * 8,)),
                 ("newton", newton, (np.float32(2),)),
                 ("gcd", jnp.gcd, scalars(84, 36)), ("gcd of zeros", jnp.gcd, scalars(0, 0)),
                 ("gcd of vectors", jnp.gcd, (np.asarray([84, 21, 10], np.int32),
                                              np.asarray([36, 6, 4], np.int32))),
                 ("lcm", jnp.lcm, scalars(21, 6)), ("lcm of zero", jnp.lcm, scalars(0, 5))]
        for cases, flow in ((decided, []), (flows, ["while"])):
            for name, function, arguments in cases:
                with self.subTest(case=name):
                    self.assertIn("while", _names(jax.make_jaxpr(function)(*arguments)))
                    self.assertEqual(_flows(export(function, *arguments)), flow)
                    self.assert_passes(self.verify(function, *arguments, checks=COMPARE))
        for wrap in (lambda f: f, jax.jit):
            self.assert_refused(self.verify(factored(1, wrap), square),
                                "qr: multiple primitive outputs")
        with jax.enable_x64(True):
            for name, function, arguments in (
                    ("newton", newton, (np.float64(2),)),
                    ("gcd", jnp.gcd, (np.int64(2 ** 40 * 3), np.int64(2 ** 35 * 9)))):
                with self.subTest(case=name, x64=True):
                    self.assert_passes(self.verify(function, *arguments, checks=COMPARE))
        steps = {name: len(self.tagged(export(function, x), "mul"))
                 for name, function in (("counted", counted), ("stepped", stepped),
                                        ("strided", strided), ("folded", folded),
                                        ("chained", chained), ("converted", converted))}
        self.assertEqual(steps, {"counted": 4, "stepped": 3, "strided": 4, "folded": 3,
                                 "chained": 3, "converted": 1})

    def test_while_runs_at_most_its_fuel_and_in_ieee_arithmetic(self):
        # Each flow loop runs at most the artifact's fuel of iterations, 65,536 unless the capture
        # names it, each time it is entered: a loop whose test still holds after them fails at
        # execution. A fuel of another integer type, such as numpy's, is held as an int, and one
        # that is negative, a Boolean (Python's or numpy's) or not an integer, such as an array of
        # one integer, is refused before the capture, also for a function without a flow. The loops
        # of one import share 65,536 iterations counted at capture: a loop that never stops, also
        # inside another loop, spends them and is a flow, and so are the import's later loops, as
        # the second of two iterations around a loop of 40,000, compared at values whose 80,000
        # products by 1.0001 stay finite, so a missing iteration would show; two iterations around a
        # loop of 3 are imported. jnp.gcd of vectors whose lanes stop at different iterations, and
        # jnp.lcm(0, 0), divide by zero in lanes that a select then discards: JAX documents
        # lax.rem's result there as implementation-defined, and the IR's integer division fails at a
        # zero divisor. The trip count at capture is JAX's, so every primitive that decides it must
        # compute as the IR does, and a loop whose predicate reads floating values is a flow. The IR
        # rounds as IEEE 754 does in a loop too: a binary32 value halved from 2^-126 while it is
        # positive stops after 24 iterations, where XLA's CPU code, which flushes 2^-127 to zero,
        # stops after 1, and the checker's emulation of XLA's CPU code covers only artifacts without
        # control flow, so the comparison fails without advice.
        from torchlean.jaxpr import encode, export

        jax, jnp, lax, np = self.jax, self.jnp, self.jax.lax, self.np
        x = self.x

        def given(m, a):
            return lax.while_loop(lambda c: c[0] < m, lambda c: (c[0] + 1, c[1] * 2.0), (0, a))

        def endless(a):
            # The counter stays 0, below its bound 1.
            return lax.while_loop(lambda c: c[0] < 1, lambda c: (c[0] * 2, c[1] * 2.0), (0, a))

        def around(trips, factor=2.0):
            def step(c):
                inner = lax.while_loop(lambda d: d[0] < trips,
                                       lambda d: (d[0] + 1, d[1] * factor), (0, c[1]))
                return c[0] + 1, inner[1]

            return lambda a: lax.while_loop(lambda c: c[0] < 2, step, (0, a))[1]

        def outer(a):
            return lax.while_loop(lambda c: c[0] < 2, lambda c: (c[0] + 1, endless(c[1])[1]),
                                  (0, a))

        for fuel, ok in ((3, False), (np.int64(4), True)):
            with self.subTest(fuel=fuel):
                document = export(given, np.int32(4), x, fuel=fuel)
                self.assertEqual(document["artifact"]["fuel"], fuel)
                self.assertIs(type(document["artifact"]["fuel"]), int)
                document["arguments"] = [encode(np.int32(4)), encode(x)]
                report = torchlean.verify(document, checks=EXECUTE, checker=CHECKER)
                if ok:
                    self.assert_passes(report)
                else:
                    self.assert_unsupported_execution(report, "while: out of fuel")
        self.assertEqual(self.artifact(given, np.int32(4), x)["artifact"]["fuel"], 65536)
        traced = []

        def doubled(a):
            traced.append(a)
            return a * 2.0

        for fuel, error in ((-1, ValueError), (True, TypeError), (np.True_, TypeError),
                            (np.array(4), TypeError), (1.5, TypeError), ("4", TypeError)):
            for function, inputs in ((given, (np.int32(4), x)), (doubled, (x,))):
                with self.subTest(fuel=fuel, flow=function is given):
                    with self.assertRaisesRegex(error, "fuel must be a nonnegative integer"):
                        export(function, *inputs, fuel=fuel)
        self.assertEqual(traced, [])
        for name, function, flows in (("endless", endless, ["while"]),
                                      ("nested", outer, ["while", "while"])):
            with self.subTest(case=name):
                self.assertEqual(_flows(export(function, x)), flows)
                self.assert_unsupported_execution(self.verify(function, x), "while: out of fuel")
        self.assertEqual(_flows(export(around(3), x)), [])
        self.assertEqual(len(self.tagged(self.artifact(around(3), x), "mul")), 6)
        self.assert_passes(self.verify(around(3), x, checks=COMPARE))
        ones = np.ones((1, 1), np.float32)
        self.assertEqual(_flows(export(around(40000), ones)), ["while"])
        self.assert_passes(self.verify(around(40000, 1.0001),
                                       np.asarray([[1.0, -0.5, 0.0, 3.0]], np.float32),
                                       checks=COMPARE))
        for name, arguments in (("gcd", (np.asarray([4, 6, 21], np.int32),
                                         np.asarray([6, 10, 6], np.int32))),
                                ("lcm", (np.int32(0), np.int32(0)))):
            with self.subTest(case=name):
                self.assert_unsupported_execution(
                    self.verify(getattr(jnp, name), *arguments), "integer division by zero")

        positive = lambda c: c[0] > 0
        half = lambda c: (c[0] * jnp.float32(0.5), c[1] + 1)

        def halved(test, step):
            return lambda a: lax.while_loop(test, step, (jnp.float32(2.0 ** -126), jnp.int32(0)))

        self.assertEqual([int(v) for v in halved(positive, half)(x)], [0, 1])
        for name, function in (("halved", halved(positive, half)),
                               ("halved under jit", halved(jax.jit(positive), jax.jit(half)))):
            with self.subTest(case=name):
                report = self.verify(function, x, checks=COMPARE)
                statuses = _statuses(report)
                self.assertEqual(statuses["execute"][0], "executed", statuses)
                self.assertEqual([np.ravel(output["bits"]).tolist()
                                  for output in report["outputs"]], [[0], [24]])
                self.assertEqual(statuses["compare"][0], "failed", statuses)
                self.assertIn("The artifact has control flow", statuses["compare"][1])
                self.assertNotIn("consistent_with", report["checks"][-1])

    def test_an_eager_while_is_one_computation(self):
        # XLA compiles an eager while loop whole, as it compiles a jax.jit call, so the steps of
        # every iteration are in one computation among the jit_computations, and the eager
        # reference runs the predicate and the body in turn, one primitive at a time.
        from torchlean.jaxpr import evaluate

        jax, lax, np = self.jax, self.jax.lax, self.np

        def function(a):
            return lax.while_loop(lambda c: c[0] < 3,
                                  lambda c: (c[0] + 1, c[1] / 3.0 + a[0]), (0, a))

        document = self.artifact(function, self.x)
        steps = self.tagged(document, "div")
        self.assertEqual(len(steps), 3)
        self.assertEqual({document["computation"][k] for k in steps},
                         set(document["jit_computations"]))
        self.assertEqual(len(document["jit_computations"]), 1)
        self.assertEqual(len(document["eager_reference_outputs"]), 2)
        carry = self.x
        for _ in range(3):
            carry = carry / 3.0 + self.x[0]
        count, last = evaluate(jax.make_jaxpr(function)(self.x), self.x)
        self.assertEqual(int(count), 3)
        self.assertTrue(np.array_equal(last, carry))

    def test_complex_loops_carry_and_stack_pairs(self):
        # A complex carry, scanned operand or stacked output of a scan, of length zero too, and a
        # complex carry or constant of a while loop, are carried, cut and stacked in paired-real
        # form, and a predicate may read a complex value. The parts are small integers, so every
        # product is exact and XLA's words are the IR's.
        jax, jnp, lax, np = self.jax, self.jnp, self.jax.lax, self.np
        ramp = np.arange(12, dtype=np.float32).reshape(3, 4)
        z = (ramp % 3 - 1 + 1j * (ramp % 2)).astype(np.complex64)
        rotation = jnp.asarray(1 + 1j, jnp.complex64)

        def stacked(a):
            return lax.scan(lambda c, v: (c * v + 1j, (c - v, c.real)), a[0], a)

        def backwards(a):
            return lax.scan(lambda c, v: (c + v.imag, jnp.conj(v) * c), jnp.ones(4, jnp.float32),
                            a, reverse=True)

        def once(a):
            return lax.scan(lambda c, v: (c + v, c * 2), jnp.zeros(4, jnp.complex64), a[:1])

        def unread(a):
            return lax.scan(lambda c, v: (c * v, (c - v, c.real)), a[0], a[:0])

        def looped(a):
            return lax.while_loop(lambda s: s[0] < 3,
                                  lambda s: (s[0] + 1, s[1] * (1 + 1j) + a[0]), (0, a))

        def rotated(a):
            # The predicate reads the real part of (1 + i)^k, which is first 4 or more at k = 7.
            return lax.while_loop(lambda s: s[1].real < 4,
                                  lambda s: (s[0] + a, s[1] * rotation), (a, rotation))

        cases = [("stacked", stacked, "scan"), ("backwards", backwards, "scan"),
                 ("once", once, "scan"), ("unread", unread, "scan"), ("looped", looped, "while"),
                 ("rotated", rotated, "while")]
        for name, function, primitive in cases:
            with self.subTest(case=name):
                self.assertIn(primitive, {equation.primitive.name for equation in
                                          jax.make_jaxpr(function)(z).eqns})
                self.assert_passes(self.verify(function, z, checks=COMPARE))
        with jax.enable_x64(True):
            self.assert_passes(self.verify(stacked, z.astype(np.complex128), checks=COMPARE))

    def test_float8_arithmetic_rounds_once_in_the_format(self):
        # Movement, abs, max, min, add, sub, mul, square, reduce_sum, reduce_prod, cumsum, cumprod,
        # integer_pow and the scatters that add, subtract or multiply import float8 values as the
        # IR's operations in the format, and neg, the comparisons, select_n, div, sign, sqrt, floor,
        # ceil, round, the gathers, the dynamic slices, the scatters that replace or take a maximum
        # or minimum, and reduce_max, reduce_min, argmax, argmin, cummax and cummin in binary32
        # (jaxpr._FLOAT8_PRIMITIVES). At every word and every pair of words of each float8 format,
        # in one row of every word but the NaNs (rest), in 64 rows of 32 words drawn from every
        # word, NaNs among them (mixed), in the sums of 64 rows of 32 words of magnitude at most 2
        # (rows) and the products of 64 rows of 32 of magnitude 0.75 to 1.5
        # (factors), in the forward cumsum and cumprod of their first 16 columns (scanned), and in
        # 512 updates of such words scattered into 16 entries and 64 rows into 4 (scattered), some
        # indices past the end, eager and under jax.jit, the IR's words are XLA's, which computes in
        # binary16 and converts each result back, also inside a fusion (x c + y, with a constant c,
        # is two roundings), and the sums, differences and products are the binary64 results rounded
        # once to the format. Those are exact, and the reference rounds them to binary32 and then
        # with ml_dtypes to the format. Binary32 holds a product exactly, and a sum too unless it
        # needs more than 24 bits, which takes addends whose last places are 24 - p or more apart,
        # for the format's precision p <= 4: the smaller is then below a relative 2^-19 of the
        # larger, x, which is normal, so the sum and its binary32 rounding lie within a relative
        # 2^-18 of x, nearer than any midpoint of the format (a relative 2^-(p + 1) or more away),
        # and both round to x. The negations, comparisons and selections are binary64's too, and
        # each quotient is the exact one rounded once, which ``quotient`` computes with fractions,
        # without binary32, and so is each square root (``root``). sign, floor, ceil and round are
        # binary64's, exact in the format but for a -0 of a format without one, which ml_dtypes
        # converts to 0. The gathers and the replacing scatter move words, which NumPy's indexing
        # gives. The reductions of a pair are IEEE's maximum and minimum, a NaN where an operand is
        # one and +0 above -0, and argmax and argmin the first position of the maximum or minimum, a
        # NaN counting as both; of a row without NaN, NumPy's. cummax and cummin give each prefix's
        # maximum or minimum so, and reversed, each suffix's. The sum and the product of a pair or
        # of a row are the left fold from 0 or 1, each step rounded once as above, and an integer
        # power is lax._integer_pow's square-and-multiply, each product rounded once, and a negative
        # power's reciprocal rounded once. A scan gives each prefix's fold, and reversed, the IR's
        # scan between two flips, each suffix's fold last to first; XLA's CPU code folds each suffix
        # first to last, which the test checks of XLA's words apart. A scatter applies its updates
        # in turn, in the order of the indices, each step rounded once, and drops those past the
        # end; JAX leaves the order of colliding updates open, and XLA's CPU code follows this one
        # here. A NaN matches any NaN, as the IR leaves a NaN's sign and payload open.
        import math
        from fractions import Fraction

        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(31)

        # Each format's fraction bits and least normal exponent.
        formats = {"float8_e5m2": (2, -14), "float8_e4m3fn": (3, -6), "float8_e4m3fnuz": (3, -7),
                   "float8_e5m2fnuz": (2, -15)}
        for name, (fraction, least) in formats.items():
            dtype = jnp.dtype(name)
            every = np.arange(256, dtype=np.uint8).view(dtype)
            values = np.asarray(every, np.float64)
            largest = float(np.max(np.abs(values[np.isfinite(values)])))
            a, b = np.repeat(every, 256), np.tile(every, 256)
            c = jnp.asarray(every[0x3a])
            permutation = np.arange(256, dtype=np.int32) * 77 % 256

            def arithmetic(x, y):
                return (x + y, x - y, x * y, lax.max(x, y), lax.min(x, y), x * c + y,
                        lax.select(x < y, x, -y), x == y, x != y, x <= y, x > y, x >= y, x / y)

            def moved(v):
                w = v.reshape(16, 16).T
                return (lax.abs(v), lax.rev(w, (1,)), lax.concatenate([v[3:9], v[:2]], 0),
                        lax.pad(v, c, [(2, 1, 1)]), jnp.broadcast_to(w[:, None], (16, 3, 16)),
                        lax.squeeze(v.reshape(1, 256), (0,)), -v, lax.is_finite(v),
                        *jnp.split(w, [3, 7], axis=1), *lax.unstack(w[:2]),
                        jnp.stack([w[0], w[5], v[:16]], axis=1))

            def unary(v):
                return (lax.sign(v), lax.sqrt(v), lax.floor(v), lax.ceil(v),
                        lax.round(v, lax.RoundingMethod.TO_NEAREST_EVEN), lax.square(v))

            def indexed(v, i):
                # i is a permutation, so the scatters write each entry once.
                w = v.reshape(16, 16)
                return (v[i], v.at[i].set(v[::-1]),
                        jnp.take(v, i * 3, mode="fill", fill_value=-1.5),
                        lax.dynamic_slice(w, (i[0], i[1]), (5, 7)),
                        lax.dynamic_update_slice(w, w[:3, :4], (i[2], i[3])),
                        v.at[i].max(v[::-1]), v.at[i].min(v[::-1]))

            def reduced(p, w, m):
                # p holds each pair of words in both orders, w every word but the NaNs, and m rows
                # of words drawn from every word.
                return (lax.reduce_max(p, (1,)), lax.reduce_min(p, (1,)),
                        lax.argmax(p, 1, jnp.int32), lax.argmin(p, 1, jnp.int32), jnp.max(w),
                        jnp.min(w), jnp.argmax(w), jnp.argmin(w), lax.cummax(p, 1),
                        lax.cummin(p, 1, reverse=True), lax.cummax(m, 1), lax.cummin(m, 1),
                        lax.cummax(m, 1, reverse=True), lax.cummin(m, 1, reverse=True))

            def folded(p, r, f, v):
                # r holds rows of 32 words of magnitude at most 2, and f rows of 32 of magnitude
                # 0.75 to 1.5, whose products stay in range. XLA's CPU code sums and multiplies
                # longer rows in another order (TorchLean/NN/IR/README.md).
                return (jnp.sum(p, 1), jnp.prod(p, 1), jnp.sum(r, 1), jnp.prod(f, 1),
                        *(lax.integer_pow(v, n) for n in powers))

            def scanned(r, f):
                # The first 16 columns of rows and factors: XLA's CPU code scans longer rows in
                # another order.
                return (lax.cumsum(r, 1), lax.cumprod(f, 1), lax.cumsum(r, 1, reverse=True),
                        lax.cumprod(f, 1, reverse=True))

            def scattered(t, w, i, u, f, k, r):
                # i holds 512 indices into the 16 entries of t and k 64 into the 4 rows of w, some
                # past the end, whose updates are dropped.
                return (t.at[i].add(u), t.at[i].subtract(u), t.at[i].multiply(f), w.at[k].add(r))

            def rounded(exact):
                return np.asarray(exact, np.float32).astype(dtype)

            def root(x):
                # sqrt x rounded once from the exact root: n steps of 2^(e - fraction), for e the
                # root's exponent or the least normal one if that is larger, n the integer
                # nearest the root of f = x / step^2, ties to even, found from isqrt(floor(f))
                # and the midpoint's square.
                result = np.sqrt(x)
                for k in np.flatnonzero(np.isfinite(x) & (x > 0)):
                    v = Fraction(x[k])
                    e = (v.numerator.bit_length() - v.denominator.bit_length()) // 2
                    while Fraction(2) ** (2 * e) > v:
                        e -= 1
                    while Fraction(2) ** (2 * e + 2) <= v:
                        e += 1
                    step = Fraction(2) ** (max(e, least) - fraction)
                    f = v / step ** 2
                    n = math.isqrt(f.numerator // f.denominator)
                    middle = Fraction(2 * n + 1, 2) ** 2
                    if f > middle or (f == middle and n % 2):
                        n += 1
                    result[k] = float(n * step)
                return result.astype(dtype)

            def quotient(x, y):
                # x / y rounded once from the exact quotient: to nearest, ties to even, in steps of
                # 2^(e - fraction) for e the quotient's exponent or the least normal one if that
                # is larger, and past the largest finite value an overflow, which ml_dtypes takes
                # to the format's infinity or NaN. binary64's quotient gives the sign and the
                # special values.
                with np.errstate(divide="ignore", invalid="ignore"):
                    signed = x / y
                result = signed.copy()
                for k in np.flatnonzero(np.isfinite(signed) & (signed != 0)):
                    q = abs(Fraction(x[k]) / Fraction(y[k]))
                    e = q.numerator.bit_length() - q.denominator.bit_length()
                    if q < Fraction(2) ** e:
                        e -= 1
                    step = Fraction(2) ** (max(e, least) - fraction)
                    v = round(q / step) * step
                    result[k] = math.copysign(float(v) if v <= largest else math.inf, signed[k])
                return result.astype(dtype)

            x, y = np.asarray(a, np.float64), np.asarray(b, np.float64)
            with np.errstate(invalid="ignore", over="ignore"):
                exact = [rounded(x + y), rounded(x - y), rounded(x * y)]
                exact.append(rounded(np.asarray(rounded(x * float(c)), np.float64) + y))
                exact += [np.where(x < y, a, rounded(-y)), x == y, x != y, x <= y, x > y, x >= y,
                          quotient(x, y)]
                signs = np.where(np.isnan(values) | (values == 0), values,
                                 np.copysign(1.0, values))
                functions = [rounded(signs), root(values), rounded(np.floor(values)),
                             rounded(np.ceil(values)), rounded(np.round(values)),
                             rounded(values * values)]
            pairs = np.stack([a, b], 1)
            rest = every[permutation][~np.isnan(values[permutation])]
            numbers = np.asarray(rest, np.float64)
            nan_x, nan_y = np.isnan(x), np.isnan(y)
            selections = [
                np.where(nan_x | ~nan_y & ((x > y) | (x == y) & ~np.signbit(x)), a, b),
                np.where(nan_x | ~nan_y & ((x < y) | (x == y) & np.signbit(x)), a, b),
                np.where(nan_x | ~nan_y & (x >= y), 0, 1),
                np.where(nan_x | ~nan_y & (x <= y), 0, 1),
                rest[np.argmax(numbers)], rest[np.argmin(numbers)], np.argmax(numbers),
                np.argmin(numbers)]
            magnitudes = np.abs(values)
            small, near = every[magnitudes <= 2], every[(magnitudes >= 0.75) & (magnitudes <= 1.5)]
            rows, factors = rng.choice(small, (64, 32)), rng.choice(near, (64, 32))
            target, table = rng.choice(small, 16), rng.choice(small, (4, 8))
            spots = rng.integers(0, 20, 512).astype(np.int32)
            lines = rng.integers(0, 5, 64).astype(np.int32)
            addends, scales, terms = rng.choice(small, 512), rng.choice(near, 512), \
                rng.choice(small, (64, 8))
            powers = (0, 1, 2, 3, 5, -1, -3)
            mixed = rng.choice(every, (64, 32))

            def running(words, largest, reverse=False):
                # Each prefix's maximum or minimum (reversed, each suffix's): a NaN from the first
                # NaN on, and +0 above -0.
                words = np.asarray(words)[:, ::-1 if reverse else 1]
                values, best = np.asarray(words, np.float64), words[:, 0]
                out = [best]
                for k in range(1, words.shape[1]):
                    b, x = np.asarray(best, np.float64), values[:, k]
                    beyond = (x > b if largest else x < b) | (x == b) & (
                        np.signbit(b) & ~np.signbit(x) if largest else
                        ~np.signbit(b) & np.signbit(x))
                    best = np.where(~np.isnan(b) & (np.isnan(x) | beyond), words[:, k], best)
                    out.append(best)
                return np.stack(out, 1)[:, ::-1 if reverse else 1]

            selections += [running(pairs, True), running(pairs, False, True), running(mixed, True),
                           running(mixed, False), running(mixed, True, True),
                           running(mixed, False, True)]

            def fold(entries, multiply):
                # The left fold from 0 or 1, each step rounded once (``rounded``).
                total = np.full(len(entries), 1.0 if multiply else 0.0)
                for column in np.asarray(entries, np.float64).T:
                    total = np.asarray(rounded(total * column if multiply else total + column),
                                       np.float64)
                return rounded(total)

            def scan(entries, multiply, reverse=False):
                # Each prefix's left fold from 0 or 1, each step rounded once; reversed, each
                # suffix's, last to first.
                entries = np.asarray(entries, np.float64)[:, ::-1 if reverse else 1]
                total, out = np.full(len(entries), 1.0 if multiply else 0.0), []
                for column in entries.T:
                    total = np.asarray(rounded(total * column if multiply else total + column),
                                       np.float64)
                    out.append(rounded(total))
                return np.stack(out, 1)[:, ::-1 if reverse else 1]

            def power(n):
                # lax._integer_pow's square-and-multiply, each product rounded once, and then the
                # reciprocal rounded once (``quotient``).
                if n == 0:
                    return rounded(np.ones(256))
                base, result, k = values, None, abs(n)
                while k:
                    if k & 1:
                        result = base if result is None else np.asarray(rounded(result * base),
                                                                        np.float64)
                    k >>= 1
                    if k:
                        base = np.asarray(rounded(base * base), np.float64)
                return rounded(result) if n > 0 else quotient(np.ones(256), result)

            def applied(start, index, updates, sign):
                # The updates applied in turn, each sum (sign 1 or -1) or product (sign 0) rounded
                # once, and those past the end dropped.
                result = np.asarray(start, np.float64).copy()
                for k, u in zip(index, np.asarray(updates, np.float64)):
                    if k < len(result):
                        result[k] = np.asarray(rounded(result[k] + sign * u if sign else
                                                       result[k] * u), np.float64)
                return rounded(result)

            with np.errstate(invalid="ignore", over="ignore"):
                folds = [fold(pairs, False), fold(pairs, True), fold(rows, False),
                         fold(factors, True), *(power(n) for n in powers)]
                short, scales16 = rows[:, :16], factors[:, :16]
                scans = [scan(short, False), scan(scales16, True), scan(short, False, True),
                         scan(scales16, True, True)]
                # XLA's reverse scans: each suffix folded first to last.
                suffixes = [np.stack([fold(v[:, k:], multiply) for k in range(16)], 1)
                            for v, multiply in ((short, False), (scales16, True))]
                scatters = [applied(target, spots, addends, 1), applied(target, spots, addends, -1),
                            applied(target, spots, scales, 0), applied(table, lines, terms, 1)]
            for mode, function, operands in (
                    ("eager", arithmetic, (a, b)), ("jit", jax.jit(arithmetic), (a, b)),
                    ("eager", moved, (every,)), ("jit", jax.jit(moved), (every,)),
                    ("eager", unary, (every,)), ("jit", jax.jit(unary), (every,)),
                    ("eager", indexed, (every, permutation)),
                    ("jit", jax.jit(indexed), (every, permutation)),
                    ("eager", reduced, (pairs, rest, mixed)),
                    ("jit", jax.jit(reduced), (pairs, rest, mixed)),
                    ("eager", folded, (pairs, rows, factors, every)),
                    ("jit", jax.jit(folded), (pairs, rows, factors, every)),
                    ("eager", scanned, (short, scales16)),
                    ("jit", jax.jit(scanned), (short, scales16)),
                    ("eager", scattered, (target, table, spots, addends, scales, lines, terms)),
                    ("jit", jax.jit(scattered),
                     (target, table, spots, addends, scales, lines, terms))):
                with self.subTest(dtype=name, mode=mode, function=function.__name__):
                    operands = [jnp.asarray(v) for v in operands]
                    report = self.verify(function, *operands)
                    self.assert_passes(report)
                    expected = function(*operands)
                    got = _float8_outputs(report, expected, dtype)
                    if function.__name__ == "scanned":
                        self.same(got[:2], [_words(v) for v in expected[:2]])
                        self.same([_words(v) for v in expected[2:]],
                                  [_words(v) for v in suffixes])
                    else:
                        self.same(got, [_words(v) for v in expected])
                    if function.__name__ == "arithmetic":
                        self.same([got[k] for k in (0, 1, 2, *range(5, 13))],
                                  [_words(v) for v in exact])
                    elif function.__name__ == "unary":
                        self.same(got, [_words(v) for v in functions])
                    elif function.__name__ == "reduced":
                        self.same(got, [_words(v) for v in selections])
                    elif function.__name__ == "folded":
                        self.same(got, [_words(v) for v in folds])
                    elif function.__name__ == "scanned":
                        self.same(got, [_words(v) for v in scans])
                    elif function.__name__ == "scattered":
                        self.same(got, [_words(v) for v in scatters])
                    elif function.__name__ == "indexed":
                        written = every.copy()
                        written[permutation] = every[::-1]
                        self.same(got[:2], [_words(every[permutation]), _words(written)])
                    else:
                        self.same(got[6:8], [_words(rounded(-np.asarray(every, np.float64))),
                                             _words(np.isfinite(np.asarray(every, np.float64)))])
            self.assert_refused(self.verify(lambda v: lax.erf(v), jnp.asarray(every),
                                            checks=("structure",)),
                                "float8 values are imported only through")
            self.assert_refused(self.verify(lambda v: jnp.take(v, permutation, mode="fill",
                                                               fill_value=0.3),
                                            jnp.asarray(every), checks=("structure",)),
                                "a float8 fill value is imported only where the format holds it")

    def test_float8_elementary_functions_round_once_through_binary32(self):
        # exp, exp2, expm1, log, log1p, sin, cos, tan, tanh, logistic, sinh, cosh, asinh, acosh,
        # atanh, asin, acos, atan, rsqrt and cbrt of a float8 operand are the IR's binary32 function
        # of the converted operand, rounded once to the format (jaxpr._FLOAT8_ELEMENTARY). At every
        # word of each format, eager and under jax.jit, the IR's words are the exact value rounded
        # once, ties to even. Where binary64's function (NumPy's) is finite and nonzero, the exact
        # value is mpmath's at 200 bits, or at an infinity binary64's limit, and elsewhere it is
        # binary64's 0, infinity or NaN. The reference rounds it to binary64, to binary32 and with
        # ml_dtypes to the format, which is the exact value rounded once: below twice the format's
        # largest finite value it is, as checked here, a value of the format to 150 bits, more than
        # a relative 2^-17 from every midpoint of the format (the overflow threshold among them),
        # or exp2(-10) in e4m3fn or exp2(-11) in e4m3fnuz, midpoints next to 0 that binary32
        # holds, and above it every rounding overflows. The least distance is 2^-16.54, logistic's
        # at -0.0625 in the e4m3 formats. XLA's CPU code computes in binary16 and converts each
        # result back. Its words are the exact value rounded to binary16 and then to the format
        # for the functions marked "binary16" below, and within one place of the IR's for those
        # marked "place", which differ at 18 (acos) to 172 (asinh) of the 1024 words. Its exp2
        # and logistic are not compared: they differ at 112 and 262 words, exp2 by up to seven
        # places (exp2(3) is 7 in e5m2), and at 2 and 144 one of the two is a NaN. A NaN matches
        # any NaN, as the IR leaves a NaN's sign and payload open.
        import math

        import mpmath
        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        # Each function's lax form, its real form in mpmath, its binary64 form in NumPy, and how
        # XLA's words compare (above).
        functions = {
            "exp": (lax.exp, mpmath.exp, np.exp, "binary16"),
            "exp2": (lax.exp2, lambda x: mpmath.mpf(2) ** x, np.exp2, None),
            "expm1": (lax.expm1, mpmath.expm1, np.expm1, "binary16"),
            "log": (lax.log, mpmath.log, np.log, "binary16"),
            "log1p": (lax.log1p, mpmath.log1p, np.log1p, "binary16"),
            "sin": (lax.sin, mpmath.sin, np.sin, "binary16"),
            "cos": (lax.cos, mpmath.cos, np.cos, "binary16"),
            "tan": (lax.tan, mpmath.tan, np.tan, "binary16"),
            "tanh": (lax.tanh, mpmath.tanh, np.tanh, "binary16"),
            "logistic": (lax.logistic, lambda x: 1 / (1 + mpmath.exp(-x)),
                         lambda x: 1 / (1 + np.exp(-x)), None),
            "sinh": (lax.sinh, mpmath.sinh, np.sinh, "binary16"),
            "cosh": (lax.cosh, mpmath.cosh, np.cosh, "binary16"),
            "asinh": (lax.asinh, mpmath.asinh, np.arcsinh, "place"),
            "acosh": (lax.acosh, mpmath.acosh, np.arccosh, "place"),
            "atanh": (lax.atanh, mpmath.atanh, np.arctanh, "place"),
            "asin": (lax.asin, mpmath.asin, np.arcsin, "place"),
            "acos": (lax.acos, mpmath.acos, np.arccos, "place"),
            "atan": (lax.atan, mpmath.atan, np.arctan, "binary16"),
            "rsqrt": (lax.rsqrt, lambda x: 1 / mpmath.sqrt(x), lambda x: 1 / np.sqrt(x),
                      "binary16"),
            "cbrt": (lax.cbrt, lambda x: mpmath.sign(x) * mpmath.cbrt(abs(x)), np.cbrt,
                     "binary16"),
        }
        # Each format's fraction bits and least normal exponent.
        formats = {"float8_e5m2": (2, -14), "float8_e4m3fn": (3, -6), "float8_e4m3fnuz": (3, -7),
                   "float8_e5m2fnuz": (2, -15)}
        found = set()
        for name, (fraction, least) in formats.items():
            dtype = jnp.dtype(name)
            every = np.arange(256, dtype=np.uint8).view(dtype)
            values = np.asarray(every, np.float64)
            largest = float(np.max(np.abs(values[np.isfinite(values)])))

            def nearness(v):
                # "value" where the exact nonzero v is a value of the format to 150 bits,
                # "midpoint" where it is a midpoint, and otherwise its relative distance from the
                # nearest midpoint, in the binade of v or the subnormal range.
                magnitude = abs(v)
                step = mpmath.ldexp(1, max(mpmath.frexp(magnitude)[1] - 1, least) - fraction)
                part = magnitude / step - mpmath.floor(magnitude / step)
                if min(part, 1 - part) < mpmath.ldexp(1, -150):
                    return "value"
                if abs(part - 0.5) < mpmath.ldexp(1, -150):
                    return "midpoint"
                return abs(part - 0.5) * step / magnitude

            exact = []
            for function, (_, real, binary64, _) in functions.items():
                with np.errstate(all="ignore"):
                    result = np.asarray(binary64(values), np.float64)
                for k in np.flatnonzero(np.isfinite(result) & (result != 0)):
                    with mpmath.workprec(200):
                        v = real(mpmath.mpf(values[k])) if math.isfinite(values[k]) else \
                            mpmath.mpf(result[k])
                        near = nearness(v) if abs(v) < 2 * largest else "value"
                    if near == "midpoint":
                        found.add((name, function, float(values[k])))
                    elif near != "value":
                        self.assertGreater(near, mpmath.ldexp(1, -17), (name, function, values[k]))
                    result[k] = float(v)
                exact.append(result)
            with np.errstate(all="ignore"):
                reference = [np.asarray(v, np.float32).astype(dtype) for v in exact]
                binary16 = [np.asarray(np.asarray(v, np.float16), np.float32).astype(dtype)
                            for v in exact]

            def elementary(v):
                return tuple(function(v) for function, *_ in functions.values())

            def place(array):
                # Each word's position in the order of the format's values.
                words = np.asarray(array).view(np.uint8).astype(np.int64)
                return np.where(words >> 7, -(words & 0x7F), words & 0x7F)

            operand = jnp.asarray(every)
            xla = [np.asarray(v) for v in elementary(operand)]
            for mode, function in (("eager", elementary), ("jit", jax.jit(elementary))):
                with self.subTest(dtype=name, mode=mode):
                    report = self.verify(function, operand)
                    self.assert_passes(report)
                    self.same(_float8_outputs(report, xla, dtype), [_words(v) for v in reference])
            with self.subTest(dtype=name, mode="xla"):
                marks = [mark for *_, mark in functions.values()]
                through = [k for k, mark in enumerate(marks) if mark == "binary16"]
                self.same([_words(xla[k]) for k in through], [_words(binary16[k]) for k in through])
                for k in (k for k, mark in enumerate(marks) if mark == "place"):
                    nan = np.isnan(np.asarray(xla[k], np.float32))
                    self.assertEqual(nan.tolist(),
                                     np.isnan(np.asarray(reference[k], np.float32)).tolist())
                    apart = np.abs(place(xla[k]) - place(reference[k]))[~nan]
                    self.assertLessEqual(int(apart.max(initial=0)), 1, list(functions)[k])
        self.assertEqual(found, {("float8_e4m3fn", "exp2", -10.0),
                                 ("float8_e4m3fnuz", "exp2", -11.0)})

    def test_float8_atan2_and_pow_round_once_through_binary64(self):
        # atan2 and pow of float8 operands are the IR's binary64 function of the converted
        # operands, rounded once to the format (jaxpr._FLOAT8_PAIRS). At every pair of words of
        # each format, eager and under jax.jit, the IR's words are the exact value rounded once,
        # ties to even. Where binary64's function (NumPy's) is finite and nonzero, the exact value
        # is mpmath's at 200 bits, or at an infinite operand binary64's, and elsewhere binary64's
        # 0, infinity or NaN. Below twice the format's largest finite value it is, as checked
        # here, a value of the format to 150 bits, more than a relative 2^-38 from every midpoint
        # of the format (the overflow threshold among them), or a midpoint, a power counted
        # below; above it every rounding overflows. The least distances are 2^-37.58, atan2's at
        # (-0.21875, 57344) in e5m2fnuz, and 2^-23.82, pow's at (104, -0.0068359375) in
        # e4m3fnuz. The reference rounds the exact value to the format here, as ml_dtypes rounds
        # binary64 through binary32 (TorchLean/NN/IR/README.md). Through binary32 instead, atan2
        # would round twice: binary32 holds the e5m2 midpoint 3 * 2^-17, within a relative 2^-32
        # of atan2(3 * 2^-16, 2), which the format's tie then rounds away from the exact value.
        # XLA's CPU code computes in binary16: its words are the exact value rounded to binary16
        # and then to the format at every pair, one place from the IR's at 110, 436, 428 and 112
        # of the 65536 pairs for atan2 and at 12, 57, 64 and 12 for pow, in e5m2, e4m3fn,
        # e4m3fnuz and e5m2fnuz. A NaN matches any NaN, as the IR leaves a NaN's sign and payload
        # open.
        import math

        import mpmath
        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax

        # Each function's lax form, its real form of two binary64 values in mpmath, whose zero
        # has no sign, and its binary64 form in NumPy.
        def real_atan2(y, x):
            # atan2(±0, -0) is ±π (IEEE 754-2019 9.2.1); the sign is binary64's (below).
            if y == 0 and x == 0:
                return mpmath.pi if math.copysign(1, x) < 0 else mpmath.mpf(0)
            return mpmath.atan2(y, x)

        def real_pow(x, y):
            # x^y of a negative x only at an integral y, where binary64's is finite.
            if x < 0:
                return (-1) ** int(y) * mpmath.mpf(-x) ** y
            return mpmath.mpf(x) ** y

        functions = {"atan2": (lax.atan2, real_atan2, np.arctan2),
                     "pow": (lax.pow, real_pow, np.power)}
        formats = {"float8_e5m2": (2, -14), "float8_e4m3fn": (3, -6), "float8_e4m3fnuz": (3, -7),
                   "float8_e5m2fnuz": (2, -15)}
        midpoints = {}
        for name, (fraction, least) in formats.items():
            dtype = jnp.dtype(name)
            every = np.arange(256, dtype=np.uint8).view(dtype)
            values = np.asarray(every, np.float64)
            largest = float(np.max(np.abs(values[np.isfinite(values)])))
            words = np.repeat(every, 256), np.tile(every, 256)
            left, right = [np.asarray(v, np.float64) for v in words]

            def spacing(magnitude):
                return mpmath.ldexp(1, max(mpmath.frexp(magnitude)[1] - 1, least) - fraction)

            def nearness(v):
                # As in test_float8_elementary_functions_round_once_through_binary32.
                magnitude = abs(v)
                part = magnitude / spacing(magnitude) - mpmath.floor(magnitude / spacing(magnitude))
                if min(part, 1 - part) < mpmath.ldexp(1, -150):
                    return "value"
                if abs(part - 0.5) < mpmath.ldexp(1, -150):
                    return "midpoint"
                return abs(part - 0.5) * spacing(magnitude) / magnitude

            def rounded(v):
                # The multiple of the spacing at v nearest v, ties to even: the format's value,
                # or beyond its largest finite value one that the conversion overflows.
                step = spacing(abs(v))
                n = mpmath.floor(abs(v) / step)
                part = abs(v) / step - n
                if part > 0.5 or part == 0.5 and int(n) % 2 == 1:
                    n += 1
                return math.copysign(float(n * step), v)

            reference, binary16 = [], []
            for function, (_, real, binary64) in functions.items():
                with np.errstate(all="ignore"):
                    result = np.asarray(binary64(left, right), np.float64)
                exact = result.copy()
                for k in np.flatnonzero(np.isfinite(result) & (result != 0)):
                    with mpmath.workprec(200):
                        finite = math.isfinite(left[k]) and math.isfinite(right[k])
                        v = real(left[k], right[k]) if finite else mpmath.mpf(result[k])
                        v = abs(v) if result[k] > 0 else -abs(v)
                        near = nearness(v) if abs(v) < 2 * largest else "value"
                        if near == "midpoint":
                            midpoints[name, function] = midpoints.get((name, function), 0) + 1
                        elif near != "value":
                            self.assertGreater(near, mpmath.ldexp(1, -38),
                                               (name, function, left[k], right[k]))
                        exact[k] = float(v)
                        result[k] = rounded(v)
                with np.errstate(all="ignore"):
                    reference.append(result.astype(dtype))
                    binary16.append(np.asarray(np.asarray(exact, np.float16), np.float32)
                                    .astype(dtype))

            def pairs(x, y):
                return tuple(function(x, y) for function, *_ in functions.values())

            # Every pair eagerly, and under jax.jit the pairs of every eighth word.
            sample = (np.arange(0, 256, 8)[:, None] * 256 + np.arange(0, 256, 8)).ravel()
            for mode, function, chosen in (("eager", pairs, slice(None)),
                                           ("jit", jax.jit(pairs), sample)):
                with self.subTest(dtype=name, mode=mode):
                    operands = [jnp.asarray(v[chosen]) for v in words]
                    report = self.verify(function, *operands)
                    self.assert_passes(report)
                    self.same(_float8_outputs(report, reference, dtype),
                              [_words(v[chosen]) for v in reference])
            with self.subTest(dtype=name, mode="xla"):
                xla = pairs(*[jnp.asarray(v) for v in words])
                self.same([_words(np.asarray(v)) for v in xla], [_words(v) for v in binary16])
        self.assertEqual(midpoints, {("float8_e5m2", "pow"): 30, ("float8_e4m3fn", "pow"): 49,
                                     ("float8_e4m3fnuz", "pow"): 40,
                                     ("float8_e5m2fnuz", "pow"): 46})

    def test_float8_orders_windows_and_remainders(self):
        # clamp, rem, sort, top_k, approx_top_k and the window reductions import float8 values:
        # clamp as the IR's max_elem then min_elem in the format, rem and sort in binary32, top_k
        # and approx_top_k ranked there, and the window reductions as folds in the format
        # (jaxpr._FLOAT8_EXACT, _FLOAT8_WINDOWS). At every word between bounds drawn from NaN, the
        # zeros, 1, 0.5, -0.5, -1, -2, the largest finite magnitudes and the infinities (clamped),
        # at every pair of words (rem), in 64 rows of 32 words drawn from every word, sorted along
        # each axis, and in every word, sorted and reversed then sorted (ordered), in 64 rows of 32
        # words with no NaN or zero ranked to k = 1, 5 and 32, and every finite nonzero word ranked
        # whole, with the approximate top and bottom 5 of those rows and the approximate top of
        # every finite nonzero word (ranked), and in the 2 x 3 windows at strides 1 and 2 of those
        # 64 rows of 32, unpadded and padded, and products of rows of magnitude 0.75 to 1.5
        # (windowed), eager and under jax.jit, the IR's words are the references'. clamp is IEEE
        # minimum(maximum(x, lo), hi), a NaN where an operand is one and +0 above -0, StableHLO's
        # formula, not the cases of lax.clamp's docstring, which differ at a NaN bound, at -0 and
        # where lo > hi; rem is binary64's fmod, exact, converted to the format; sort is NumPy's
        # stable sort, NaN last and -0 equal to +0, as lax.sort documents; top_k and approx_max_k
        # are the stable descending order and approx_min_k the ascending one; and a window is the
        # row-major left fold of its cells, padding among them, from JAX's identity (an infinity, or
        # the largest finite magnitude of a format without one), each sum or product rounded once
        # (in binary32 and then the format, as in test_float8_arithmetic_rounds_once_in_the_format).
        # XLA's words are the IR's, but for its sort of a fnuz format, which puts a NaN first, and
        # its sort and top_k of float8_e5m2fnuz, which take the five words 4 to 8 (2^-15 to 2^-14)
        # as equal, and their negatives, where its approx_top_k orders them; the test checks those
        # of XLA's words apart. A NaN matches any NaN, as the IR leaves a NaN's sign and payload
        # open.
        import math

        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(37)

        def extreme(u, v, largest):
            # IEEE maximum or minimum: a NaN where an operand is one, and +0 above -0.
            beyond = (u > v if largest else u < v) | (u == v) & (
                np.signbit(v) & ~np.signbit(u) if largest else ~np.signbit(v) & np.signbit(u))
            return np.where(np.isnan(u) | np.isnan(v), np.nan, np.where(beyond, u, v))

        for name in ("float8_e5m2", "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2fnuz"):
            dtype = jnp.dtype(name)
            infinite, fnuz = name == "float8_e5m2", name.endswith("fnuz")
            every = np.arange(256, dtype=np.uint8).view(dtype)
            values = np.asarray(every, np.float64)
            largest = float(jnp.finfo(dtype).max)
            bounds = [math.nan, -largest, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, largest]
            bounds += ([] if fnuz else [-0.0]) + ([-math.inf, math.inf] if infinite else [])
            bounds = np.asarray(bounds).astype(dtype)
            low = np.repeat(np.repeat(bounds, len(bounds))[:, None], 256, 1)
            high = np.repeat(np.tile(bounds, len(bounds))[:, None], 256, 1)
            middle = np.broadcast_to(every, low.shape).copy()
            a, b = np.repeat(every, 256), np.tile(every, 256)
            mixed = rng.choice(every, (64, 32))
            numbers = np.asarray(mixed, np.float64)
            ranks = mixed.copy()
            ranks[np.isnan(numbers) | (numbers == 0)] = np.asarray(1.0, dtype)
            whole = every[np.isfinite(values) & (values != 0)]
            magnitudes = np.abs(values)
            factors = rng.choice(every[(magnitudes >= 0.75) & (magnitudes <= 1.5)], (64, 32))
            identity = math.inf if infinite else largest
            paddings = ("VALID", ((1, 0), (2, 1)))

            def clamped(lo, x, hi):
                return (lax.clamp(lo, x, hi),)

            def remainders(x, y):
                return (lax.rem(x, y),)

            def ordered(m, v):
                return (lax.sort(m, dimension=1), lax.sort(m, dimension=0), lax.sort(v),
                        lax.sort(v[::-1]))

            def ranked(r, w):
                return (*lax.top_k(r, 1), *lax.top_k(r, 5), *lax.top_k(r, 32),
                        *lax.top_k(w, len(whole)), *lax.approx_max_k(r, 5),
                        *lax.approx_min_k(r, 5), *lax.approx_max_k(w, len(whole)))

            def windowed(m, f):
                out = []
                for padding in paddings:
                    out += [lax.reduce_window(m, np.asarray(0, dtype), lax.add, (2, 3), (1, 2),
                                              padding),
                            lax.reduce_window(m, np.asarray(-identity, dtype), lax.max, (2, 3),
                                              (1, 2), padding),
                            lax.reduce_window(m, np.asarray(identity, dtype), lax.min, (2, 3),
                                              (1, 2), padding),
                            lax.reduce_window(f, np.asarray(1, dtype), lax.mul, (2, 3), (1, 2),
                                              padding)]
                return tuple(out)

            def rounded(exact):
                return np.asarray(exact, np.float32).astype(dtype)

            def sort(words, axis):
                # NumPy's stable sort of the values, -0 as +0 and NaN last.
                keys = np.asarray(words, np.float64) + 0.0
                return np.take_along_axis(words, np.argsort(keys, axis, kind="stable"), axis)

            def top(words, k, largest=True):
                # The first k in the stable descending (or ascending) order, and their positions.
                keys = np.asarray(words, np.float64)
                order = np.argsort(-keys if largest else keys, -1, kind="stable")[..., :k]
                return [np.take_along_axis(words, order, -1), order.astype(np.int32)]

            def fold(words, init, combine, padding):
                # Each 2 x 3 window at strides 1 and 2 of the words padded with init, folded from
                # init in the row-major order of its cells.
                pad = [(0, 0), (0, 0)] if padding == "VALID" else list(padding)
                cells = np.pad(np.asarray(words, np.float64), pad, constant_values=init)
                rows, columns = cells.shape[0] - 1, (cells.shape[1] - 3) // 2 + 1
                total = np.full((rows, columns), init)
                for i in range(2):
                    for j in range(3):
                        total = combine(total, cells[i:i + rows, j:j + 2 * columns - 1:2])
                return total.astype(dtype)

            with np.errstate(invalid="ignore", over="ignore"):
                x, y = np.asarray(a, np.float64), np.asarray(b, np.float64)
                references = {
                    "clamped": [extreme(extreme(np.asarray(middle, np.float64),
                                                np.asarray(low, np.float64), True),
                                        np.asarray(high, np.float64), False).astype(dtype)],
                    "remainders": [np.fmod(x, y).astype(dtype)],
                    "ordered": [sort(mixed, 1), sort(mixed, 0), sort(every, 0),
                                sort(every[::-1], 0)],
                    "ranked": [*top(ranks, 1), *top(ranks, 5), *top(ranks, 32),
                               *top(whole, len(whole)), *top(ranks, 5), *top(ranks, 5, False),
                               *top(whole, len(whole))],
                    "windowed": []}
                for padding in paddings:
                    references["windowed"] += [
                        fold(mixed, 0.0, lambda t, c: np.asarray(rounded(t + c), np.float64),
                             padding),
                        fold(mixed, -identity, lambda t, c: extreme(t, c, True), padding),
                        fold(mixed, identity, lambda t, c: extreme(t, c, False), padding),
                        fold(factors, 1.0, lambda t, c: np.asarray(rounded(t * c), np.float64),
                             padding)]
            primitives = {equation.primitive.name
                          for equation in jax.make_jaxpr(windowed)(mixed, factors).jaxpr.eqns}
            self.assertLessEqual({"reduce_window_sum", "reduce_window_max", "reduce_window_min",
                                  "reduce_window"}, primitives)
            for function, operands in ((clamped, (low, middle, high)), (remainders, (a, b)),
                                       (ordered, (mixed, every)), (ranked, (ranks, whole)),
                                       (windowed, (mixed, factors))):
                for mode, run in (("eager", function), ("jit", jax.jit(function))):
                    with self.subTest(dtype=name, mode=mode, function=function.__name__):
                        operands = [jnp.asarray(v) for v in operands]
                        report = self.verify(run, *operands)
                        self.assert_passes(report)
                        expected = run(*operands)
                        got = _float8_outputs(report, expected, dtype)
                        self.same(got, [_words(v) for v in references[function.__name__]])
                        xla = [_words(v) for v in expected]
                        if function is ordered and fnuz:
                            # XLA's sort puts the NaN, the word 0x80, first.
                            self.assertEqual([v[0] for v in xla[2:]], [-1, -1])
                            if name == "float8_e5m2fnuz":
                                # The words 4 to 8 keep their input order, here descending.
                                self.assertEqual([w for w in xla[3] if 4 <= w <= 8],
                                                 [8, 7, 6, 5, 4])
                        elif function is ranked and name == "float8_e5m2fnuz":
                            # Of the whole row, ascending by word, the words 4 to 8 keep their
                            # input order, where the descending order reverses them.
                            self.assertEqual([w for w in xla[6] if 4 <= w <= 8], [4, 5, 6, 7, 8])
                            self.assertEqual([w for w in got[6] if 4 <= w <= 8], [8, 7, 6, 5, 4])
                            # Its approx_top_k orders them.
                            self.same(got[8:], xla[8:])
                        else:
                            self.same(got, xla)

    def test_float8_steps_bitcasts_and_indices(self):
        # nextafter, bitcast_convert_type and iota import float8 values (jaxpr._FLOAT8_WORDS):
        # nextafter as the IR's in the format, C's: a NaN where an operand is one, y where x == y,
        # and otherwise the next value of the format from x toward y, subnormals and zeros among
        # them (-0 above the least negative value but in a fnuz format, and +0 below the least
        # positive one); a bitcast as each word read at the other type, an 8-bit integer type or
        # another float8 format; and iota as each index rounded once to nearest, ties to even
        # (StableHLO's iota, the index a constant of the type), a NaN or an infinity past the
        # largest value. At every pair of words (stepped), at every word and every 8-bit integer
        # (bitcasts), and in iotas of 600 entries and along each axis of a 3 x 600 and a 600 x 2
        # tensor, one of them in a maximum with a float8 operand (indices), eager and under jax.jit,
        # the IR's words are the references' and XLA's, a NaN matching any NaN but in a bitcast,
        # which keeps every word, and the integer results are uint8 and int8. XLA's nextafter alone
        # differs: in a fnuz format, from the least negative value toward any larger y, it gives the
        # word 0x80, which is -0 in the other formats and the NaN in these, where C's next value up
        # is +0.
        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        formats = ("float8_e5m2", "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2fnuz")
        for name in formats:
            dtype = jnp.dtype(name)
            fnuz = name.endswith("fnuz")
            every = np.arange(256, dtype=np.uint8).view(dtype)
            values = np.asarray(every, np.float64)
            nan = np.isnan(values)
            ordered = np.unique(values[~nan]) + 0.0
            least = ordered[ordered > 0][0]

            def word(v):
                return int(np.flatnonzero(~nan & (values == v) &
                                          (np.signbit(values) == np.signbit(v)))[0])

            def step(v, up):
                # The word of the next value up or down from v, or -2 past the ends.
                if v == (-least if up else least):
                    return 0 if fnuz or not up else word(-0.0)
                if v == 0:
                    return word(least if up else -least)
                k = int(np.searchsorted(ordered, v)) + (1 if up else -1)
                return word(ordered[k]) if 0 <= k < len(ordered) else -2

            ups = np.asarray([-1 if n else step(v, True) for v, n in zip(values, nan)])
            downs = np.asarray([-1 if n else step(v, False) for v, n in zip(values, nan)])
            first, second = np.repeat(np.arange(256), 256), np.tile(np.arange(256), 256)
            x, y = values[first], values[second]
            integers = np.arange(256, dtype=np.uint8)
            others = [other for other in formats if other != name]
            largest = ordered[np.isfinite(ordered)][-1]
            spacing = largest - ordered[np.isfinite(ordered)][-2]
            operand = np.asarray(np.linspace(-400, 400, 1200).reshape(2, 600), np.float64)

            def stepped(a, b):
                return (lax.nextafter(a, b),)

            def bitcasts(v, u):
                return (lax.bitcast_convert_type(v, jnp.uint8),
                        lax.bitcast_convert_type(v, jnp.int8),
                        lax.bitcast_convert_type(u, dtype),
                        lax.bitcast_convert_type(lax.bitcast_convert_type(u, jnp.int8), dtype),
                        *[lax.bitcast_convert_type(v, jnp.dtype(other)) for other in others])

            def indices(m):
                return (lax.iota(dtype, 600), lax.broadcasted_iota(dtype, (3, 600), 1),
                        lax.broadcasted_iota(dtype, (600, 2), 0),
                        lax.max(m, lax.broadcasted_iota(dtype, (2, 600), 1)))

            def rounded(k):
                # k rounded once to nearest, ties to even: the nearer of the format's values
                # around k, or at a tie the one with the even word; past the largest value and
                # half its spacing, or at that tie where the largest value's word is odd, an
                # infinity or a NaN.
                k = np.asarray(k, np.float64)
                finite = ordered[np.isfinite(ordered)]
                above = finite[np.minimum(np.searchsorted(finite, k), len(finite) - 1)]
                below = finite[np.maximum(np.searchsorted(finite, k, "right") - 1, 0)]
                below_words = np.asarray([word(v) for v in below.ravel()]).reshape(k.shape)
                above_words = np.asarray([word(v) for v in above.ravel()]).reshape(k.shape)
                nearer = np.where(k - below < above - k, below_words, np.where(
                    k - below > above - k, above_words,
                    np.where(below_words % 2 == 0, below_words, above_words)))
                over = (k > largest + spacing / 2) | (k == largest + spacing / 2) & (
                    word(largest) % 2 == 1)
                return np.where(over, word(np.inf) if name == "float8_e5m2" else -1, nearer)

            ramp = rounded(np.arange(600))
            ramp_values = np.where(ramp < 0, np.nan, np.asarray(
                np.maximum(ramp, 0).astype(np.uint8).view(dtype), np.float64))
            m = np.asarray(operand.astype(dtype), np.float64)
            references = {
                "stepped": [np.where(np.isnan(x) | np.isnan(y), -1, np.where(
                    x == y, second, np.where(x < y, ups[first], downs[first])))],
                "bitcasts": [np.arange(256)] * (4 + len(others)),
                "indices": [ramp, np.tile(ramp, 3), np.repeat(ramp, 2), _words(
                    np.maximum(m, np.tile(ramp_values, 2).reshape(2, 600)).astype(dtype))]}
            for function, operands in ((stepped, (every[first], every[second])),
                                       (bitcasts, (every, integers)),
                                       (indices, (operand.astype(dtype),))):
                for mode, run in (("eager", function), ("jit", jax.jit(function))):
                    with self.subTest(dtype=name, mode=mode, function=function.__name__):
                        operands = [jnp.asarray(v) for v in operands]
                        report = self.verify(run, *operands)
                        self.assert_passes(report)
                        expected = run(*operands)
                        if function is bitcasts:
                            self.assertEqual([output["element_type"]
                                              for output in report["outputs"][:2]],
                                             ["uint8", "int8"])
                            got = [np.asarray(output["bits"], np.int64).ravel()
                                   for output in report["outputs"]]
                            xla = [np.asarray(v).view(np.uint8).astype(np.int64).ravel()
                                   for v in expected]
                        else:
                            got = _float8_outputs(report, expected, dtype)
                            xla = [_words(v) for v in expected]
                        if function is stepped and fnuz:
                            apart = (first == 0x81) & (y > x)
                            self.assertEqual(np.flatnonzero(xla[0] != got[0]).tolist(),
                                             np.flatnonzero(apart).tolist())
                            self.assertTrue(np.all(xla[0][apart] == -1))
                            xla[0][apart] = got[0][apart]
                        self.same(got, references[function.__name__])
                        self.same(got, xla)

    def test_float8_reduce_precision_rounds_at_the_format_exponents(self):
        # reduce_precision(x, e, m) of a float8 operand is StableHLO's at the format's exponents
        # (jaxpr._Builder.reduce_precision): the mantissa rounded to m bits, to nearest even, at
        # the operand's exponent, or at the least normal one below it; then, where e is narrower
        # than the format's exponent field, an infinity of the sign at or above 2^(2^(e - 1)) and
        # a zero of the sign below 2^(2 - 2^(e - 1)); then the result converted to the format, an
        # infinity to the NaN of a format without one and -0 to 0 in the fnuz formats. The
        # reference rounds exact rationals and converts with ml_dtypes. At every word of each
        # format, e from 1 to 6 and m from 1 to 4, eager and under jax.jit, the IR's words are the
        # reference's. XLA's CPU code converts the operand to binary16, reduces it there and
        # converts the result back, and its words are that reference's, which differs at 20 of the
        # 96 settings: at e = 4 its range step flushes the e4m3 formats' values below 2^-6 and
        # takes e4m3fn's from 2^8 to its NaN, and at m below the format's fraction width it rounds
        # some subnormals at binary16's spacing. A NaN matches any NaN there, and the IR's NaN words
        # are compared exactly: the operand's, quieted where the setting reduces it (the IR has no
        # float8 selection, and FloatLib's conversions quiet a NaN), so that e5m2's signaling 0x7d
        # and 0xfd give 0x7f and 0xff.
        import math
        from fractions import Fraction

        import numpy as np

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax

        def reduced(value, fraction, least, width, e, m):
            # The reduction in a format of `fraction` fraction bits, least normal exponent
            # `least` and an exponent field of `width` bits.
            if value != value or value == 0 or math.isinf(value):
                return value
            magnitude = Fraction(abs(value))
            exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
            if Fraction(2) ** exponent > magnitude:
                exponent -= 1
            if m < fraction:
                quantum = Fraction(2) ** (max(exponent, least) - m)
                q, r = divmod(magnitude, quantum)
                q = int(q) + (1 if r > quantum / 2 or r == quantum / 2 and int(q) % 2 else 0)
                magnitude = q * quantum
            if e < width:
                largest = 2 ** (e - 1) - 1
                if magnitude >= Fraction(2) ** (largest + 1):
                    return math.copysign(math.inf, value)
                if magnitude < Fraction(2) ** (1 - largest):
                    return math.copysign(0.0, value)
            return math.copysign(float(magnitude), value)

        # Each format's fraction bits, least normal exponent and exponent field width, and the bit
        # that a quiet NaN of e5m2, the one format with signaling NaNs, has set.
        formats = {"float8_e5m2": (2, -14, 5), "float8_e4m3fn": (3, -6, 4),
                   "float8_e4m3fnuz": (3, -7, 4), "float8_e5m2fnuz": (2, -15, 5)}
        quiet = {"float8_e5m2": 0x02}
        settings = [(e, m) for e in range(1, 7) for m in range(1, 5)]
        differ = []
        for name, (fraction, least, width) in formats.items():
            dtype = jnp.dtype(name)
            every = np.arange(256, dtype=np.uint8).view(dtype)
            values = np.asarray(every, np.float64)
            nan = np.isnan(values)
            words = np.arange(256)[nan]
            quieted = [words if m >= fraction and e >= width else words | quiet.get(name, 0)
                       for e, m in settings]

            def function(v):
                return tuple(lax.reduce_precision(v, e, m) for e, m in settings)

            with np.errstate(over="ignore"):
                own = [np.asarray([reduced(v, fraction, least, width, e, m) for v in values])
                       .astype(dtype) for e, m in settings]
                half = [np.asarray([reduced(v, 10, -14, 5, e, m) for v in values], np.float16)
                        .astype(dtype) for e, m in settings]
            differ += [(name, e, m) for (e, m), a, b in zip(settings, own, half)
                       if (_words(a) != _words(b)).any()]
            for mode, run in (("eager", function), ("jit", jax.jit(function))):
                with self.subTest(dtype=name, mode=mode):
                    report = self.verify(run, jnp.asarray(every))
                    self.assert_passes(report)
                    expected = run(jnp.asarray(every))
                    self.same(_float8_outputs(report, expected, dtype), [_words(v) for v in own])
                    self.same([np.asarray(output["bits"], np.int64)[nan]
                               for output in report["outputs"]], quieted)
                    self.same([_words(v) for v in expected], [_words(v) for v in half])
        self.assertEqual(len(differ), 20, differ)

    def test_float8_variance_and_median_read_a_nan_literal(self):
        # jnp.var, and medians whose trace uses supported stable sorting, import float8 values.
        # An unstable median sort is explicitly refused, rather than assigned stable tie bits.
        # Each holds a NaN that JAX
        # stores as a Python float declared binary32 and converts to the format, which the import
        # reads at NumPy's conversion, as JAX's lowering does (jaxpr._Builder.constant). var
        # computes in the format (_var and _mean in jax/_src/numpy/reductions.py): the mean, the
        # left fold of the fiber from 0 over its length, then the differences from it, their squares
        # and their left fold, and the quotient by the length less ddof, each step rounded once to
        # the format, and a NaN where that divisor is not positive. median is the quantile at 0.5 by
        # midpoints: a NaN where the fiber holds one, and otherwise the sum of the sorted fiber's
        # two middle entries, or of its middle one twice, times 0.5, each rounded once; nanmedian
        # takes the fiber's entries but its NaNs so, and gives a NaN where it has none. In 64 fibers
        # of 4 words of magnitude 1/8 to 2 or zero, of either sign, NaNs in four of them, one fiber
        # all NaN, at each float8 format, eager and under jax.jit, the IR's words are the
        # references', for ddof 0, 1 and 4, a NaN matching any NaN, and XLA's but for nanmedian in a
        # fnuz format, whose sort in XLA's CPU code puts the NaN first
        # (test_float8_orders_windows_and_remainders); the test checks those of XLA's words apart.
        # Binary32 holds each step's binary64 result exactly, which the reference checks, but a
        # quotient by 3, which is exact or repeats 01 in binary and so is no midpoint of the format
        # after its rounding to binary32 either.
        import numpy as np

        jax, jnp = self.jax, self.jnp
        rng = np.random.default_rng(37)
        for name in ("float8_e5m2", "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2fnuz"):
            dtype = jnp.dtype(name)
            every = np.arange(256, dtype=np.uint8).view(dtype)
            values = np.asarray(every, np.float64)
            with np.errstate(invalid="ignore"):
                usable = every[(values == 0) | (np.abs(values) >= 0.125) & (np.abs(values) <= 2)]
            x = rng.choice(usable, (64, 4))
            x[0, 1] = x[1, :] = x[2, 3] = x[3, :2] = np.nan
            fibers = np.asarray(x, np.float64)

            def once(v, exact=True):
                # v rounded once to the format, through binary32.
                v = np.asarray(v, np.float64)
                if exact:
                    self.assertTrue(np.array_equal(v.astype(np.float32), v, equal_nan=True))
                return np.asarray(v.astype(np.float32).astype(dtype), np.float64)

            def fold(v):
                total = np.zeros(len(v))
                for k in range(v.shape[1]):
                    total = once(total + v[:, k])
                return total

            def variance(ddof):
                with np.errstate(invalid="ignore"):
                    mean = once(fold(fibers) / 4)
                    total = fold(once(once(fibers - mean[:, None]) ** 2))
                    return (once(total / (4 - ddof), exact=ddof != 1) if ddof < 4
                            else np.full(len(fibers), np.nan))

            def middle(fiber):
                kept = np.sort(fiber[~np.isnan(fiber)], kind="stable")
                if not len(kept):
                    return np.nan
                return once(once(kept[(len(kept) - 1) // 2] + kept[len(kept) // 2]) * 0.5)

            references = [_words(np.asarray(v, np.float32).astype(dtype)) for v in (
                *[variance(ddof) for ddof in (0, 1, 4)],
                [np.nan if np.isnan(fiber).any() else middle(fiber) for fiber in fibers],
                [middle(fiber) for fiber in fibers])]
            cases = (
                ("variance", lambda v: tuple(jnp.var(v, axis=1, ddof=d) for d in (0, 1, 4)),
                 references[:3]),
                ("median", lambda v: (jnp.median(v, axis=1), jnp.nanmedian(v, axis=1)),
                 references[3:]),
            )
            for group, function, reference in cases:
                for mode, run in (("eager", function), ("jit", jax.jit(function))):
                    with self.subTest(dtype=name, group=group, mode=mode):
                        operand = jnp.asarray(x)
                        pending, unstable = [jax.make_jaxpr(run)(operand).jaxpr], False
                        while pending:
                            for equation in pending.pop().eqns:
                                unstable |= (equation.primitive.name == "sort"
                                             and not equation.params["is_stable"])
                                for value in equation.params.values():
                                    for inner in value if isinstance(value, (tuple, list)) else (value,):
                                        body = getattr(inner, "jaxpr", inner)
                                        if hasattr(body, "eqns"):
                                            pending.append(body)
                        report = self.verify(run, operand)
                        if unstable:
                            self.assertEqual(group, "median")
                            self.assert_refused(report, "sort: only a stable sort")
                            continue
                        self.assert_passes(report)
                        expected = run(operand)
                        got = _float8_outputs(report, expected, dtype)
                        self.same(got, reference)
                        xla = [_words(v) for v in expected]
                        if group == "median" and name.endswith("fnuz"):
                            # XLA's sort puts the NaN first, so its nanmedian of the fibers with
                            # three entries but the NaN is the least of them, and of two a NaN.
                            least = [np.nanmin(fibers[0]), np.nanmin(fibers[2]), np.nan]
                            self.same([xla[1][[0, 2, 3]]],
                                      [_words(np.asarray(least, np.float32).astype(dtype))])
                            xla[1][[0, 2, 3]] = got[1][[0, 2, 3]]
                        self.same(got, xla)

    def test_python_floats_are_their_numpy_conversion(self):
        # A Python float or complex number that a jaxpr holds as a literal imports as NumPy's
        # conversion to its declared type, as JAX 0.11.1 lowers it (jaxpr._Builder.constant).
        # jnp.clip's bounds, jnp.where's value, jnp.pad's and nan_to_num's are such literals,
        # binary32 arguments of the jitted function that converts each to the operand's type. In
        # 64 entries, a NaN, -0 and both bounds among them, of binary32, binary16, bfloat16 and
        # each float8 format, eager and under jax.jit, the IR's words are XLA's and the
        # references': each Python float the binary32 value nearest it and then the operand
        # type's value nearest that, by exact rationals, a tie to the even word, and the
        # selections and the padding of NumPy's binary64 arithmetic, a NaN matching any NaN. The
        # bounds +-(1 + 2^-8 + 2^-25) tell two roundings from one: in binary32 they are
        # +-(1 + 2^-8), a tie in bfloat16 that goes to +-1, where bfloat16's values nearest them
        # are +-(1 + 2^-7). A complex number that jnp.where and jnp.pad put in a complex64 array,
        # 0.1 + 0.7j, is declared complex64, and in 16 entries, NaN and signed-zero parts among
        # them, the IR's words of the parts are XLA's and those of 0.1 and 0.7 each rounded once
        # to binary32. A bound past binary32's largest finite value is refused.
        from fractions import Fraction

        import numpy as np

        jax, jnp = self.jax, self.jnp
        rng = np.random.default_rng(41)

        def nearest(q, dtype):
            # The finite value of dtype nearest the binary64 q, of two at the same distance the
            # one whose word is even, which in each format here ends a significand in 0.
            width = 8 * dtype.itemsize
            if dtype == np.float32:
                seed = np.float32(q)
                values = np.asarray([np.nextafter(seed, np.float32(-np.inf)), seed,
                                     np.nextafter(seed, np.float32(np.inf))], np.float32)
            else:
                values = np.arange(2 ** width, dtype=f"uint{width}").view(dtype)
                with np.errstate(invalid="ignore"):
                    values = values[np.isfinite(values.astype(np.float64))]
            close = values[np.argsort(np.abs(values.astype(np.float64) - q))[:3]]
            distances = sorted((abs(Fraction(float(v)) - Fraction(q)),
                                int(np.asarray(v).view(f"uint{width}")) % 2, float(v))
                               for v in close)
            self.assertNotEqual(distances[0][:2], distances[1][:2])
            return distances[0][2]

        tie = 1 + 2 ** -8 + 2 ** -25

        def functions(v):
            return (jnp.clip(v, 0.1, 0.9), jnp.where(v > 0, v, 0.7), jnp.nan_to_num(v, nan=0.1),
                    jnp.pad(v, 1, constant_values=0.3), jnp.clip(v, -tie, tie))

        def words(array):
            # The words of an array of floating values, each NaN as -1.
            array = np.asarray(array)
            unsigned = array.view(f"uint{8 * array.dtype.itemsize}").astype(np.int64)
            unsigned[np.isnan(array.astype(np.float64))] = -1
            return unsigned.ravel()

        draws = np.concatenate([[np.nan, -0.0, 0.1, 0.9], rng.uniform(-2, 2, 60)])
        for name in ("float32", "float16", "bfloat16", "float8_e5m2", "float8_e4m3fn",
                     "float8_e4m3fnuz", "float8_e5m2fnuz"):
            dtype = jnp.dtype(name)
            x = draws.astype(np.float32).astype(dtype)
            v = np.asarray(x, np.float64)
            low, high, other, replaced, padding, below, above = [
                nearest(nearest(q, np.dtype(np.float32)), dtype)
                for q in (0.1, 0.9, 0.7, 0.1, 0.3, -tie, tie)]
            with np.errstate(invalid="ignore"):
                references = [np.where(np.isnan(v), np.nan, np.minimum(np.maximum(v, low), high)),
                              np.where(v > 0, v, other), np.where(np.isnan(v), replaced, v),
                              np.concatenate([[padding], v, [padding]]),
                              np.where(np.isnan(v), np.nan,
                                       np.minimum(np.maximum(v, below), above))]
            references = [words(np.asarray(r, np.float64).astype(np.float32).astype(dtype))
                          for r in references]
            for mode, run in (("eager", functions), ("jit", jax.jit(functions))):
                with self.subTest(dtype=name, mode=mode):
                    report = self.verify(run, jnp.asarray(x))
                    self.assert_passes(report)
                    got = [words(np.asarray(output["bits"], np.uint64).astype(
                        f"uint{8 * dtype.itemsize}").view(dtype)) for output in report["outputs"]]
                    self.same(got, references)
                    self.same(got, [words(v) for v in run(jnp.asarray(x))])
            if name == "bfloat16":
                self.assertEqual((below, above), (-1.0, 1.0))
                self.assertEqual(nearest(tie, dtype), 1 + 2 ** -7)
        parts = np.concatenate([[np.nan, -0.0, 0.0, 1.0], rng.uniform(-2, 2, 28)])
        z = (parts[:16] + 1j * parts[16:]).astype(np.complex64)
        z.imag[1] = -0.0
        constant = np.complex64(complex(nearest(0.1, np.dtype(np.float32)),
                                        nearest(0.7, np.dtype(np.float32))))
        references = [np.where(z.real > 0, z, constant),
                      np.concatenate([[constant], z, [constant]])]
        references = [words(np.stack([r.real, r.imag], axis=-1)) for r in references]

        def selections(z):
            return jnp.where(z.real > 0, z, 0.1 + 0.7j), jnp.pad(z, 1, constant_values=0.1 + 0.7j)

        for mode, run in (("eager", selections), ("jit", jax.jit(selections))):
            with self.subTest(dtype="complex64", mode=mode):
                report = self.verify(run, jnp.asarray(z))
                self.assert_passes(report)
                got = [words(np.asarray(output["bits"], np.uint64).astype(np.uint32).view(
                    np.float32)) for output in report["outputs"]]
                self.same(got, references)
                self.same(got, [words(np.stack([np.asarray(r).real, np.asarray(r).imag], axis=-1))
                                for r in run(jnp.asarray(z))])
        self.assert_refused(self.verify(lambda v: jnp.clip(v, -1e39, 1e39),
                                        np.zeros(4, np.float32)),
                            "a Python number overflows or loses its infinity")

    def test_float8_pooling_transposes_select_in_binary32(self):
        # select_and_scatter_add imports float8 values (jaxpr._FLOAT8_SCATTERS): in each window the
        # operand cell that a left fold in row-major order selects, compared in binary32, which
        # holds each value and keeps their order, keeping the selected cell where
        # select(selected, cell) holds, and each source value added to its window's cell from +0
        # in row-major window order, each sum rounded once in the format; a padding cell is never
        # selected (StableHLO's select_and_scatter). In 5 x 7 operands drawn from each format's
        # finite words (finite), from 4 of them (ties), with NaNs (nans), from its least and
        # greatest finite values and 0 (extremes), at its least finite value alone (least), and in
        # float8_e5m2fnuz from the ten values 2^-15 to 2^-14 of each sign (tens), through ge and le
        # windows that overlap, tile, and lie partly in padding before and after the operand, eager
        # and under jax.jit, the IR's words are that fold's. JAX's CPU lowering pads the operand
        # with -inf or +inf, or in a format without infinities its least or greatest finite value,
        # and gives XLA no padding, so XLA folds the padding cells too, and its words are that
        # fold's (expand): in such a format they differ where a window's operand cells all hold
        # that value. Each sum is the binary64 one, exact, rounded to binary32 and then with
        # ml_dtypes to the format, one rounding (test_float8_arithmetic_rounds_once_in_the_format).
        # A NaN matches any NaN.
        import numpy as np
        from jax._src.lax import windowed_reductions

        jax, jnp, lax = self.jax, self.jnp, self.jax.lax
        rng = np.random.default_rng(47)
        shapes = [((2, 2), (1, 1), ((0, 0), (0, 0))), ((2, 2), (2, 2), ((0, 0), (0, 0))),
                  ((3, 3), (2, 2), ((1, 1), (1, 1))), ((3, 2), (1, 2), ((1, 0), (0, 1)))]
        windows = [tuple((lo + n + hi - w) // s + 1 for (lo, hi), n, w, s in zip(
            padding, (5, 7), window, strides)) for window, strides, padding in shapes]
        selections = [(shape, ge) for shape in shapes for ge in (True, False)]

        def pooled(x, *sources):
            return tuple(windowed_reductions._select_and_scatter_add(
                g, x, lax.ge_p if ge else lax.le_p, window, strides, padding)
                for g, ((window, strides, padding), ge) in zip(sources, selections))

        def scattered(x, g, window, strides, padding, ge, expand):
            # The fold over each window's operand cells, or with expand over its cells of the
            # operand padded with the identity too, and the sums in turn, cropped to the operand.
            finfo = jnp.finfo(x.dtype)
            identity = (-1 if ge else 1) * (np.inf if np.isinf(np.asarray(
                np.inf, x.dtype).astype(np.float64)) else float(finfo.max))
            values = np.pad(np.asarray(x, np.float64), padding, constant_values=identity)
            inside = np.pad(np.ones(x.shape, bool), padding, constant_values=False)
            out = np.zeros(values.shape, x.dtype)
            for w in np.ndindex(*g.shape):
                chosen = None
                for c in np.ndindex(*window):
                    p = tuple(i * s + j for i, s, j in zip(w, strides, c))
                    if not (expand or inside[p]):
                        continue
                    if chosen is None or not (values[chosen] >= values[p] if ge else
                                              values[chosen] <= values[p]):
                        chosen = p
                if chosen is not None:
                    total = np.float64(out[chosen]) + np.float64(g[w])
                    out[chosen] = np.float32(total).astype(x.dtype)
            return _words(out[tuple(slice(lo, lo + n) for (lo, _), n in zip(padding, x.shape))])

        for name in ("float8_e5m2", "float8_e4m3fn", "float8_e4m3fnuz", "float8_e5m2fnuz"):
            dtype = jnp.dtype(name)
            every = np.arange(256, dtype=np.uint8).view(dtype)
            values = np.asarray(every, np.float64)
            finite = every[np.isfinite(values)]
            small = every[np.isfinite(values) & (np.abs(values) <= 2)]
            least = finite[np.argmin(finite.astype(np.float64))]
            operands = {"finite": rng.choice(finite, (5, 7)),
                        "ties": rng.choice(rng.choice(finite, 4), (5, 7)),
                        "nans": np.where(rng.random((5, 7)) < 0.2, every[np.isnan(values)][0],
                                         rng.choice(finite, (5, 7))),
                        "extremes": rng.choice(np.asarray(
                            [least, -least, 0], dtype), (5, 7)),
                        "least": np.full((5, 7), least)}
            if name == "float8_e5m2fnuz":
                tens = np.r_[4:9, 0x84:0x89].astype(np.uint8).view(dtype)
                operands["tens"] = rng.choice(tens, (5, 7))
            for label, x in operands.items():
                x = np.asarray(x, dtype)
                sources = [rng.choice(small, extent) for extent in windows for _ in (0, 1)]
                plain = [scattered(x, g, *shape, ge, False)
                         for g, (shape, ge) in zip(sources, selections)]
                padded = [scattered(x, g, *shape, ge, True)
                          for g, (shape, ge) in zip(sources, selections)]
                if label == "least" and name != "float8_e5m2":
                    self.assertTrue(any(np.any(a != b) for a, b in zip(plain, padded)))
                arguments = [jnp.asarray(v) for v in (x, *sources)]
                for mode, run in (("eager", pooled), ("jit", jax.jit(pooled))):
                    with self.subTest(dtype=name, operand=label, mode=mode):
                        report = self.verify(run, *arguments)
                        self.assert_passes(report)
                        expected = run(*arguments)
                        self.same(_float8_outputs(report, expected, dtype), plain)
                        self.same([_words(v) for v in expected], padded)


@unittest.skipUnless(CHECKER and HAS_JAX, "needs the built verify executable and JAX")
class JAXCompilationTests(CheckerCase):
    def test_index_ramp_can_be_lowered_inside_jit(self):
        import jax
        import jax.numpy as jnp
        import numpy as np

        @torchlean.compile(checker=CHECKER)
        def add_indices(x):
            return x + jnp.arange(x.shape[0], dtype=x.dtype)

        for dtype in (jnp.int32, jnp.float32):
            with self.subTest(dtype=dtype):
                x = jnp.full(4, 2, dtype=dtype)
                np.testing.assert_array_equal(add_indices(x), [2, 3, 4, 5])
                self.assert_passes(add_indices.torchlean.report)
        self.assertEqual(len(add_indices.torchlean.graphs), 2)

        def wide_indices(x):
            with jax.enable_x64():
                return x + jnp.arange(x.shape[0], dtype=jnp.float64)

        with jax.enable_x64(False):
            report = torchlean.verify(wide_indices, jnp.ones(3), checks=COMPARE, checker=CHECKER)
            self.assert_passes(report)
            self.assertEqual(report["outputs"][0]["element_type"], "float64")
            np.testing.assert_array_equal(_outputs(report)[0].numpy(), [1, 2, 3])
            self.assertFalse(jax.config.x64_enabled)

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

    def test_static_float_specialization_preserves_nan_words(self):
        import struct
        import jax.numpy as jnp

        @torchlean.compile(checker=CHECKER)
        def adjust(x, n):
            word, = struct.unpack(">Q", struct.pack(">d", n))
            return x + (word & 0xff) if word >> 63 == 0 else x - (word & 0xff)

        for word, expected in ((0x7ff8000000000001, 11), (0xfff8000000000001, 9),
                               (0x7ff8000000000002, 12), (0x7ff8000000000001, 11)):
            n, = struct.unpack(">d", word.to_bytes(8, "big"))
            self.assertEqual(int(adjust(jnp.int32(10), n)), expected)
        self.assertEqual(len(adjust.torchlean.graphs), 3)

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
            bits = jax.lax.rng_bit_generator(jnp.zeros(4, jnp.uint32), x.shape)[1]
            return x + bits.astype(x.dtype)

        with self.assertRaises(Exception):
            unsupported(jnp.ones(2))
        self.assertEqual(unsupported.torchlean.graphs, ())
