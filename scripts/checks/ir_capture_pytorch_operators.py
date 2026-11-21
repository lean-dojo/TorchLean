"""PyTorch operator captures checked with ``verify ir`` (``PyTorchGeneralOperatorTests``).
Run them through ``scripts/checks/test_ir_capture.py``, which describes the checks.
"""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from ir_capture_support import (CHECKER, COMPARE, EXECUTE, FORMATS, HAS_TORCH, CheckerCase,
                                _assert_within_ulps, _exp, _float_of_word, _ieee_maximum,
                                _ieee_minimum, _nearest, _ops, _outputs, _records, _reflect32,
                                _registered, _same, _splitmix64, _statuses, _steps, _tags,
                                _uniform, _words)
import torchlean


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
        # PyTorch fixes no order for a sum, so the record holds the left fold.
        self.assertIn([{"nat": 0}, {"nats": [0, 2]}, {"bool": False}, {"order": "leftFold"}],
                      reduce)
        # PyTorch reads an empty dim list as every axis.
        self.assertIn([{"nat": 0}, {"nats": [0, 1, 2]}, {"bool": False}, {"order": "leftFold"}],
                      reduce)
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

    def test_unbind_components_are_selects(self):
        # Every component of unbind is the slice along dim without that axis, at any dim, also
        # where the program reads only some of them.
        torch = self.torch
        x, n = self.x, self.x.to(torch.int32)
        functions = (lambda x: x.unbind(), lambda x: torch.unbind(x, dim=2),
                     lambda x: tuple(part * 2 for part in x.unbind(-2)),
                     lambda x: x.unbind(1)[2] - x.unbind(1)[0], lambda x: x.unbind(-1)[-1])
        for index, function in enumerate(functions):
            for operand in (x, n):
                for core in (False, True):
                    with self.subTest(index=index, dtype=operand.dtype, core=core):
                        self.assert_passes(self.verify(function, operand, core=core))
        self.assertEqual(_tags(self.artifact(lambda x: x.unbind(2), x)).count("slice"), 4)

    def test_created_tensors_of_one_value_are_one_broadcast_word(self):
        # zeros, ones, full, scalar_tensor, their _like and new_ forms and fill hold one value:
        # one constant word broadcast to the shape, whatever the shape.
        torch = self.torch
        x = self.x
        functions = (lambda x: x + torch.zeros(3, 4), lambda x: torch.full((5, 2, 3, 4), -2.5) - x,
                     lambda x: torch.ones(2, 3, 4, dtype=torch.float16) * x.half(),
                     lambda x: torch.full_like(x, 0.1, dtype=torch.bfloat16) + x.bfloat16(),
                     lambda x: x.new_zeros(4) - x, lambda x: x.new_full((3, 1), 2 ** -3) * x,
                     lambda x: torch.ones_like(x).cumsum(0) * x, lambda x: torch.fill(x, 3.0) + x,
                     lambda x: torch.zeros_like(x[0]) - x,
                     lambda x: torch.zeros(3, 4, dtype=torch.bool) | (x > 0),
                     lambda x: torch.full((3, 4), 7) * (x > 0),
                     lambda x: x.new_ones((3, 3)).index_add(1, torch.tensor([2, 0, 2, 1]), x[0]))
        for index, function in enumerate(functions):
            for core in (False, True):
                with self.subTest(index=index, core=core):
                    self.assert_passes(self.verify(function, x, core=core))
        document = self.artifact(lambda x: torch.zeros(4096, 4096) + x.sum(), x)
        self.assertIn("broadcast", _tags(document))
        self.assertLessEqual(sum(len(words) for words in document["artifact"]["constants"]
                                 if words), 4)

    def test_linspace_and_logspace_are_constants(self):
        # linspace and logspace of constant arguments are the tensors PyTorch creates with them on
        # the CPU, as arange is. The core graph writes them as an arange and its arithmetic.
        torch = self.torch
        x = self.x
        functions = (lambda x: x + torch.linspace(-1, 2, 4),
                     lambda x: x.double() * torch.linspace(0, 1, 4, dtype=torch.float64),
                     lambda x: x.half() - torch.linspace(-3, 3, 4, dtype=torch.float16),
                     lambda x: x * torch.linspace(5, 5, 1),
                     lambda x: x * torch.linspace(0, 10, 4, dtype=torch.int64),
                     lambda x: x / torch.logspace(0, 3, 4, base=2),
                     lambda x: x + torch.logspace(-1, 1, 4, dtype=torch.bfloat16).float())
        for index, function in enumerate(functions):
            for core in (False, True):
                with self.subTest(index=index, core=core):
                    self.assert_passes(self.verify(function, x, core=core))

    def test_unfold_is_strided_slices(self):
        # Tensor.unfold(dimension, size, step): window i holds the `size` entries along dimension
        # from i * step, on a new last axis.
        x = self.x
        functions = (lambda x: x.unfold(2, 2, 1), lambda x: x.unfold(-1, 3, 2) * 2,
                     lambda x: x.unfold(0, 2, 5) - 1, lambda x: x.unfold(1, 3, 1).unfold(2, 2, 2),
                     lambda x: (x > 0).unfold(2, 4, 4), lambda x: x.int().unfold(1, 1, 1))
        for index, function in enumerate(functions):
            for core in (False, True):
                with self.subTest(index=index, core=core):
                    self.assert_passes(self.verify(function, x, core=core))

    def test_composites_without_a_lowering_are_their_aten_operators(self):
        # outer, type_as, expand_as and conv_transpose2d have no CPU kernel of their own: eager
        # PyTorch runs their CompositeImplicitAutograd composites, which the capture reads in their
        # place, here from torch.export's graph (the core graph holds the composite already).
        # conv_transpose2d's is a transposed convolution.
        torch = self.torch
        functional = torch.nn.functional
        x, v, n = self.x, torch.arange(5.) - 2, torch.arange(3, dtype=torch.int32)
        image, kernel = x.reshape(1, 2, 3, 4), (torch.arange(24.) % 5 - 2).reshape(2, 3, 2, 2)
        cases = {"outer": (lambda v, n: torch.outer(v, n), v, n),
                 "outer method": (lambda v: v.outer(v) + 1, v),
                 "type_as": (lambda x, n: x.type_as(n) * 3, x, n),
                 "type_as half": (lambda x: x.type_as(x.half()) + 1, x),
                 "expand_as": (lambda x, v: v[:4].expand_as(x) - x, x, v),
                 "conv_transpose2d": (lambda image, kernel: functional.conv_transpose2d(
                     image, kernel, stride=2, padding=1), image, kernel)}
        for name, (function, *inputs) in cases.items():
            for core in (False, True):
                with self.subTest(case=name, core=core):
                    self.assert_passes(self.verify(function, *inputs, core=core))
        # item has only a CompositeImplicitAutograd kernel too, of a number, and stays refused.
        report = self.verify(lambda x: x * x.sum().item(), x)
        self.assertFalse(report["ok"])
        self.assertIn("aten.item.default", str(_statuses(report)))

    def test_transposed_convolutions_are_convolutions_of_the_spread_input(self):
        # conv_transpose1d, 2d and 3d and ConvTranspose2d, with strides, dilations, groups,
        # output padding, a padding beyond the kernel's reach (a crop), no bias, and a weight and
        # bias not known at capture, as the convolution of the input spread by its stride. Integer
        # weights and inputs make every summation order exact, so the words compare.
        torch = self.torch
        functional = torch.nn.functional
        generator = torch.Generator().manual_seed(2)

        def integers(*shape):
            return torch.randint(-3, 4, shape, generator=generator).float()

        x1, x2, x3 = integers(2, 4, 5), integers(1, 4, 3, 4), integers(1, 2, 2, 3, 2)
        w1, w2, w3 = integers(4, 3, 3), integers(4, 2, 2, 3), integers(2, 3, 2, 2, 2)
        b = integers(6)
        cases = {
            "1d": (lambda x: functional.conv_transpose1d(x, w1, b[:3], stride=2), x1),
            "1d groups": (lambda x: functional.conv_transpose1d(
                x, w1, b, stride=3, padding=1, output_padding=2, groups=2), x1),
            "1d crop": (lambda x: functional.conv_transpose1d(
                x, w1, None, stride=2, padding=3, dilation=1), x1),
            "2d": (lambda x: functional.conv_transpose2d(
                x, w2, b[:2], stride=(2, 1), padding=(0, 1), dilation=(1, 2)), x2),
            "2d groups": (lambda x: functional.conv_transpose2d(
                x, w2, None, stride=2, output_padding=1, groups=4), x2),
            "3d": (lambda x: functional.conv_transpose3d(
                x, w3, b[:3], stride=(1, 2, 2), padding=(1, 0, 1)), x3),
            "weight and bias operands": (
                lambda x, w, b: functional.conv_transpose2d(x, w * 2, b, stride=2), x2, w2,
                b[:2]),
        }
        module = torch.nn.ConvTranspose2d(4, 6, (3, 2), stride=2, padding=1, groups=2)
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.copy_(integers(*parameter.shape))
        cases["ConvTranspose2d"] = (module, x2)
        for name, (function, *inputs) in cases.items():
            for core in (False, True):
                with self.subTest(case=name, core=core):
                    self.assert_passes(self.verify(function, *inputs, core=core))
        # A zero of the spread input multiplies the kernel, so an infinite kernel entry makes NaN
        # entries that eager PyTorch, which adds only the input's products, leaves finite.
        w = w2.clone()
        w[0, 0, 0, 0] = float("inf")
        x = x2.abs() + 1
        eager = functional.conv_transpose2d(x, w, stride=2)
        self.assertTrue(bool(eager.isfinite().any()))
        report = self.verify(lambda x: functional.conv_transpose2d(x, w, stride=2), x,
                             checks=EXECUTE)
        self.assert_passes(report)
        report = self.verify(lambda x: functional.conv_transpose2d(x, w, stride=2), x)
        self.assertFalse(report["ok"])

    def test_einsum_is_its_aten_composite(self):
        # torch.einsum is ATen's composite traced at the operands' shapes: diagonals, sums and bmm
        # contractions in the order of the recorded path, opt_einsum's from three operands on, from
        # torch.export's graph, the core graph and FX's.
        from torchlean._fx import _tensor_value
        from torchlean.export import export_torchlean_graph_json

        torch = self.torch
        generator = torch.Generator().manual_seed(6)
        cases = (("bnd,bnk->dk", (2, 5, 3), (2, 5, 4)),
                 ("bcihs,bcjhs->bcijh", (2, 2, 3, 2, 4), (2, 2, 3, 2, 4)),
                 ("bnijc,ikc->bnijk", (2, 2, 3, 4, 5), (3, 6, 5)),
                 ("...ij,...jk->...ik", (2, 3, 4), (2, 4, 5)), ("i,i->", (7,), (7,)),
                 ("ab,cd->abcd", (2, 3), (4, 5)), ("ii->i", (4, 4)), ("iij->ji", (3, 3, 4)),
                 ("ij->", (3, 4)), ("ij,jk,kl->il", (3, 40), (40, 5), (5, 60)))
        contractions = {1: lambda equation: lambda x: torch.einsum(equation, x),
                        2: lambda equation: lambda x, y: torch.einsum(equation, x, y),
                        3: lambda equation: lambda x, y, z: torch.einsum(equation, [x, y, z])}
        for equation, *shapes in cases:
            operands = [torch.randint(-8, 8, shape, generator=generator).float()
                        for shape in shapes]
            function = contractions[len(shapes)](equation)
            for core in (False, True):
                with self.subTest(equation=equation, core=core):
                    self.assert_passes(self.verify(function, *operands, core=core))
        self.assertIn("matmul", _tags(self.artifact(function, *operands)))

        class Contraction(torch.nn.Module):
            def forward(self, x, y):
                return torch.einsum("bij,bjk->bik", x, y)

        x, y = (torch.randint(-8, 8, shape, generator=generator).float()
                for shape in ((2, 3, 4), (2, 4, 5)))
        with tempfile.TemporaryDirectory() as directory:
            artifact = export_torchlean_graph_json(Contraction(), (x, y),
                                                   str(Path(directory) / "graph.json"),
                                                   prefer_torch_export=False)
        artifact["arguments"] = [_tensor_value(x), _tensor_value(y)]
        report = torchlean.verify(artifact, checks=EXECUTE, checker=CHECKER)
        self.assert_passes(report)
        self.assertEqual([value.tolist() for value in _outputs(report)],
                         [Contraction()(x, y).tolist()])

    def test_reduced_float_sums_and_means_accumulate_in_binary32(self):
        # ATen's CPU sum and mean of binary16 and bfloat16 accumulate in binary32 and round once,
        # where a fold in the format rounds each step. Sixteenths below 2^8 round
        # in both formats, and their binary32 sums are exact in every order.
        torch = self.torch
        generator = torch.Generator().manual_seed(5)
        x = torch.randint(-2 ** 12, 2 ** 12, (2, 3, 32), generator=generator) / 16

        def reductions(x):
            return (x.sum(-1), x.sum((0, 1)), torch.sum(x), x.mean(1),
                    x.mean((0, 2), keepdim=True), torch.mean(x))

        for dtype in (torch.float16, torch.bfloat16):
            for core in (False, True):
                with self.subTest(dtype=dtype, core=core):
                    self.assert_passes(self.verify(reductions, x.to(dtype), core=core))
            artifact = self.artifact(lambda x: (x.sum(-1), x.mean(0)), x.to(dtype))
            self.assertEqual([scalar for record, scalar in _ops(artifact)
                              if record["tag"] in ("reduce", "mean")], ["float32"] * 2)
            # The fold in the format differs from eager PyTorch here.
            folded = x.to(dtype)[..., 0]
            for k in range(1, 32):
                folded = folded + x.to(dtype)[..., k]
            self.assertFalse(torch.equal(folded, x.to(dtype).sum(-1)))

    def test_dtype_arguments_cast_the_input_first(self):
        # A dtype argument of sum, prod, mean, cumsum, cumprod, softmax and log_softmax casts the
        # input to dtype before the operation (their PyTorch 2.13 docstrings), an integer input
        # included. Eager mean to binary16 or bfloat16 from another dtype sums the uncast input in
        # binary32 and rounds the quotient once instead, and the comparison fails.
        # A cast to binary16 or bfloat16 rounds each entry to an odd integer, whose sums are exact.
        torch = self.torch
        functional = torch.nn.functional
        x = self.x * 2 + 1 + 2 ** -12

        def reductions(dtype, mean=True):
            return lambda x: (x.sum(1, dtype=dtype), torch.sum(x, dtype=dtype),
                              x.prod(0, dtype=dtype), torch.cumsum(x, -1, dtype=dtype),
                              x.cumprod(1, dtype=dtype),
                              *([torch.mean(x, (0, 2), dtype=dtype)] if mean else []))

        def normalized(dtype):
            return lambda x: (functional.softmax(x, 1, dtype=dtype),
                              torch.softmax(x, -1, dtype=dtype),
                              functional.log_softmax(x, 0, dtype=dtype),
                              x.log_softmax(2, dtype=dtype))

        pairs = ((torch.float32, torch.float64), (torch.float64, torch.float32),
                 (torch.bfloat16, torch.float32), (torch.float16, torch.float32),
                 (torch.float32, torch.float16), (torch.float32, torch.bfloat16),
                 (torch.int64, torch.float32))
        for source, target in pairs:
            for core in (False, True):
                with self.subTest(source=source, target=target, core=core):
                    mean = target not in (torch.float16, torch.bfloat16)
                    self.assert_passes(self.verify(reductions(target, mean), x.to(source),
                                                   core=core))
                    self.assert_passes(self.verify(normalized(target), x.to(source),
                                                   checks=EXECUTE, core=core))
        artifact = self.artifact(normalized(torch.float32), x.to(torch.bfloat16))
        self.assertEqual([scalar for record, scalar in _ops(artifact)
                          if record["tag"] in ("softmax", "logSoftmax")], ["float32"] * 4)
        quarter = torch.tensor([257.0, 257.0, 257.0, 259.0])
        mean = lambda x: torch.mean(x, dtype=torch.bfloat16)
        for core in (False, True):
            with self.subTest(core=core):
                report = self.verify(mean, quarter, core=core)
                self.assertEqual({name: status for name, (status, _) in
                                  _statuses(report).items()},
                                 {"structure": "checked", "types": "checked",
                                  "execute": "executed", "compare": "failed"})
                self.assertEqual([value.tolist() for value in _outputs(report)], [256.0])
        self.assertEqual(mean(quarter).item(), 258.0)

    def test_comparisons_logical_operators_and_selections_match_eager_words(self):
        # Comparisons, class predicates and logical and bitwise operators give Boolean nodes, as do
        # the maximum and minimum of Booleans (logical or and and), where and masked_fill select
        # with them, clamp, clip and hardtanh are the IR clamp, elu, selu and celu are Op.elu with
        # their alpha, scale and input_scale, and softplus is Op.softplus of x * beta divided by
        # beta, or x where x * beta passes the threshold. run_decompositions writes selu as elu,
        # celu with expm1 and softplus with exp and log1p. In binary32 and binary16 the words agree
        # with PyTorch's on entries with NaNs, zeros of both signs and infinities, the bounds away
        # from zero, elu's expm1 and softplus's logarithm exact. Bitwise operators of integers and
        # operands of two dtypes are refused. Below a raised threshold PyTorch's softplus_kernel
        # overflows, where Op.softplus keeps x.
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
                    torch.logical_not(x), p & q, p | q, p ^ q, ~p, torch.maximum(p, q),
                    torch.minimum(p, q))

        def selections(x, y):
            return (torch.where(x > y, x, y), torch.where(x > 0, x, 0.5),
                    x.masked_fill(x < 0, -1.0), torch.clamp(x, -1.0, 2.0),
                    torch.clamp(x, y - 4, y + 4), torch.clip(x, -2.0, 1.0),
                    functional.hardtanh(x))

        def elus(e):
            return (functional.elu(e), functional.elu(e, alpha=0.5),
                    torch.ops.aten.elu(e, 0.5, 2.0, 0.5), functional.selu(e))

        def celu(e):
            return functional.celu(e, alpha=2.0)

        def softplus(e):
            return (functional.softplus(e), functional.softplus(e, beta=2.0, threshold=10.0),
                    torch.ops.aten.softplus(e, 0.5, 5.0))

        cases = {"comparisons": (comparisons, x, y), "logical": (logical, x, y),
                 "selections": (selections, x, y), "elu": (elus, exact), "celu": (celu, exact),
                 "softplus": (softplus, large)}
        for name, (function, *inputs) in cases.items():
            for dtype in (torch.float32, torch.float16):
                for core in (False, True):
                    with self.subTest(case=name, dtype=dtype, core=core):
                        typed = [value.to(dtype) for value in inputs]
                        self.assert_passes(self.verify(function, *typed, core=core))
        for function, core in ((elus, False), (elus, True), (celu, False)):
            tags = _tags(self.artifact(function, exact, core=core))
            self.assertIn("elu", tags)
            self.assertNotIn("expm1", tags)
        tags = _tags(self.artifact(functional.softplus, large))
        self.assertEqual([tag for tag in tags if tag != "broadcast"], ["softplus", "gt", "select"])
        tags = _tags(self.artifact(softplus, large))
        self.assertEqual([tags.count(tag) for tag in ("softplus", "mul", "div")], [3, 2, 2])
        raised = torch.tensor([95.0])
        report = self.verify(lambda e: functional.softplus(e, threshold=100.0), raised)
        self.assertEqual({name: status for name, (status, _) in _statuses(report).items()},
                         {"structure": "checked", "types": "checked", "execute": "executed",
                          "compare": "failed"})
        self.assertEqual([value.tolist() for value in _outputs(report)], [[95.0]])
        self.assertEqual(functional.softplus(raised, threshold=100.0).tolist(), [inf])
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

    def test_transposes_and_moved_axes_are_permutes(self):
        # torch.export keeps Tensor.T, mT, mH, H and adjoint, swapdims, swapaxes and movedim as
        # their own ATen calls, and each is the IR permute of its axes. Below rank 2, T is its
        # operand. Core ATen writes them all as permute.
        torch = self.torch
        m, v = self.x[0], torch.arange(4.)

        def transposes(x, m, v):
            return (x.T, x.mT, x.mH, x.adjoint(), torch.adjoint(x), m.H, m.T, v.T + 1,
                    x.swapdims(0, 2), torch.swapaxes(x, -1, 1), torch.movedim(x, 0, 2),
                    torch.movedim(x, (0, -1), (-1, 0)), x.moveaxis(1, 0))

        for core in (False, True):
            with self.subTest(core=core):
                self.assert_passes(self.verify(transposes, self.x, m, v, core=core))
        self.assertEqual(_tags(self.artifact(transposes, self.x, m, v)).count("permute"), 12)

    def test_stacks_rolls_and_triangles_move_words(self):
        # torch.stack is an axis of extent one in each tensor and their concatenation, and
        # repeat_interleave with an int count an axis after `dim` broadcast and merged into it.
        # torch.roll is two slices concatenated along each axis, and torch.tril and torch.triu a
        # selection between the tensor and zero by a constant mask. Without an axis,
        # repeat_interleave and roll take the flattened tensor's. They move words, so a NaN and -0
        # keep theirs, in every dtype. A roll by a multiple of the extent is its operand. Core
        # ATen writes them its own way.
        torch = self.torch
        x = self.x.clone()
        x[0, 1, 2], x[1, 0, 0] = float('nan'), -0.0
        y = torch.linspace(-3, 3, 24).reshape(2, 3, 4)

        def moved(x, y):
            return (torch.stack([x, y]), torch.stack((x, y, x), dim=-1), torch.stack([x], 1),
                    torch.roll(x, 1, 0), torch.roll(x, (-1, 5), (1, -1)), x.roll(7),
                    torch.roll(x, 3, 0), torch.roll(y, 4, 2), torch.tril(x), torch.triu(x, 1),
                    x.tril(-1), torch.triu(y, -2), x.triu(9), torch.tril(x[0], 2),
                    torch.repeat_interleave(x, 2, 1), x.repeat_interleave(3),
                    y.repeat_interleave(1, dim=-1))

        for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16, torch.int32,
                      torch.bool):
            for core in (False, True):
                with self.subTest(dtype=str(dtype), core=core):
                    self.assert_passes(self.verify(moved, x.to(dtype), y.to(dtype), core=core))
        self.assertEqual(_tags(self.artifact(moved, x, y)).count("cat"), 7)

    def test_in_place_detach_stops_the_gradient(self):
        # torch.export keeps Tensor.detach_ of an input or of a value computed from one, and the
        # program's later reads of the tensor use its result. It writes autograd state, not data:
        # Op.detach, the operand's value with a zero tangent, so the IR's derivative of
        # y = 2 x; y.detach_(); y * x is PyTorch's 2 x, not the 4 x of (2 x) x. A tensor literal's
        # detach_ is its operand. PyTorch 2.13's core decomposition drops detach_ with no trace,
        # and the document of a decomposed program records that its derivatives assume no
        # detach_. A symbolic trace keeps the later reads on the operand, and is refused.
        # x.detach() is Op.detach too, written as alias with detach as its original_aten in a
        # decomposed program. A no_grad block is refused, with no fallback to a symbolic trace,
        # which would not see the grad mode, and so is its decomposed form, whose nodes record the
        # block in their provenance (from_node).
        from torchlean.export import export_torchlean_graph_json

        torch = self.torch

        def detached(x):
            y = x * 2
            y.detach_()
            return x.detach_() + 1, y

        def product(x):
            y = x * 2
            y.detach_()
            return y * x * torch.tensor([1.0, 2.0, 3.0, 4.0])

        for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            with self.subTest(dtype=str(dtype)):
                self.assert_passes(self.verify(detached, self.x.to(dtype)))
                self.assert_passes(self.verify(product, self.x.to(dtype)))
        self.assertEqual(_tags(self.artifact(detached, self.x)).count("detach"), 2)
        steps = _steps(self.artifact(product, self.x))
        detach = [index for index, step in enumerate(steps)
                  if "op" in step and step["op"]["record"]["tag"] == "detach"]
        self.assertEqual(len(detach), 1)
        readers = [step["op"]["record"]["tag"] for step in steps
                   if "op" in step and detach[0] in [port[0] for port in step["op"]["inputs"]]]
        self.assertEqual(readers, ["mul"])
        x = self.x.clone().requires_grad_(True)
        product(x).sum().backward()
        torch.testing.assert_close(x.grad, 2 * self.x * torch.tensor([1.0, 2.0, 3.0, 4.0]))
        core = self.artifact(product, self.x, core=True)
        self.assertNotIn("detach", _tags(core))
        self.assertEqual([constraint.get("kind") for constraint in core["capture_constraints"]],
                         ["decomposed_program"])
        self.assertNotIn("decomposed_program",
                         [constraint.get("kind") for constraint in
                          self.artifact(product, self.x)["capture_constraints"]])

        def copied(x):
            return (x * 2).detach() * x

        for dtype in (torch.float32, torch.bfloat16):
            for core in (False, True):
                with self.subTest(dtype=str(dtype), copied=True, core=core):
                    self.assert_passes(self.verify(copied, self.x.to(dtype), core=core))
        for core in (False, True):
            self.assertEqual(_tags(self.artifact(copied, self.x, core=core)).count("detach"), 1)

        def quiet(x):
            with torch.no_grad():
                y = x * 2
            return y * x

        for core in (False, True):
            self.assert_refused(self.verify(quiet, self.x, checks=("structure",), core=core),
                                "grad-mode region")

        class Traced(torch.nn.Module):
            def forward(self, x):
                return product(x)

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(NotImplementedError, "reads after it"):
                export_torchlean_graph_json(Traced(), (self.x,), str(Path(directory) / "g.json"),
                                            prefer_torch_export=False)

        class Quiet(torch.nn.Module):
            def forward(self, x):
                return quiet(x)

        original = torch.export.export
        for core in (False, True):
            if core:
                torch.export.export = lambda *a, **k: original(*a, **k).run_decompositions()
            try:
                with tempfile.TemporaryDirectory() as directory, self.subTest(core=core):
                    with self.assertRaisesRegex(NotImplementedError, "grad-mode region"):
                        export_torchlean_graph_json(Quiet(), (self.x,),
                                                    str(Path(directory) / "q.json"))
            finally:
                torch.export.export = original

    def test_empty_and_as_strided(self):
        # A full overwrite of a fresh allocation imports. Reading an uninitialized value is
        # refused. A strided view imports where every addressed word belongs to its operand.
        torch = self.torch
        aten = torch.ops.aten
        x = self.x[0].clone()

        def created(x):
            return (torch.empty(3, 4).zero_() + x, torch.empty_strided((3, 4), (1, 3)).fill_(2) * x,
                    torch.empty_like(x).fill_(-1) + x, x.new_empty((3, 4)).zero_() - x,
                    x.new_empty_strided((3, 4), (4, 1)).fill_(5) + x,
                    torch.empty(3, 4, dtype=torch.int32).fill_(3) + x)

        def uninitialized(x):
            return (torch.empty(3, 4) + x, torch.empty_like(x, dtype=torch.int32) - 1,
                    torch.empty(3, 4, dtype=torch.bool) & (x > 0))

        class Empty(torch.nn.Module):
            def forward(self, x):
                return torch.empty(x.shape, dtype=x.dtype) + x

        from torchlean.export import export_torchlean_graph_json

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "empty.json"
            with self.assertRaisesRegex(NotImplementedError, "read of an uninitialized"):
                export_torchlean_graph_json(Empty(), x, str(target))
            self.assertFalse(target.exists())

        def strided(x):
            return (aten.as_strided(x, [4, 3], [1, 4]), aten.as_strided(x * 2, [2, 2], [1, 4], 5),
                    aten.as_strided(x.t(), [3, 2], [4, 1]), aten.as_strided(x[1:], [5], [1]),
                    aten.as_strided(x[:, ::2], [3], [4]),
                    aten.as_strided(x[0].expand(2, 4), [4], [1]),
                    aten.as_strided(x.sum(), [3], [0]), aten.as_strided(x, [], [], 7) * x,
                    aten.as_strided(x, [0, 2], [1, 1]))

        for core in (False, True):
            with self.subTest(core=core):
                self.assert_passes(self.verify(created, x, core=core))
                self.assert_passes(self.verify(strided, x, core=core))
                # Each names the value it constrains by its reference, as top_k_mask's does.
                reads = [constraint for constraint in
                         self.artifact(strided, x, core=core)["capture_constraints"]
                         if constraint["kind"] == "strided_read"]
                self.assertEqual(len(reads), 9)
                self.assertTrue(all("ref" in read and "node" not in read for read in reads))
                self.assert_refused(
                    self.verify(uninitialized, x, checks=("structure",), core=core),
                    "read of an uninitialized PyTorch allocation")
                for outside in (lambda x: aten.as_strided(x, [13], [1]),
                                lambda x: aten.as_strided(x[:, ::2], [2], [1])):
                    self.assert_refused(self.verify(outside, x, checks=("structure",), core=core),
                                        "as_strided reading memory outside its operand")

        def view(x):
            y = x * 1
            y[0].zero_()
            return y

        self.assert_refused(self.verify(view, x, checks=("structure",)), "zero_")

        def autocast(x):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                return x @ x.transpose(-1, -2)

        # A symbolic trace would erase autocast and import a binary32 matrix product instead.
        self.assertEqual(autocast(self.x).dtype, torch.bfloat16)
        self.assert_refused(self.verify(autocast, self.x, checks=("structure",)), "autocast region")

    def test_device_and_memory_format_conversions_convert_the_dtype(self):
        # torch.export writes Tensor.cpu(), to(device=...) and to(memory_format=...) as
        # to.dtype_layout, and to(device, dtype) and to(other) as to.device. Each converts to its
        # dtype, here the same one or binary16 and int32, and core ATen writes _to_copy.
        torch = self.torch
        x = self.x / 3

        def moved(x):
            return (x.cpu() + 1, x.to(device="cpu") * 2,
                    x.to(memory_format=torch.contiguous_format), x.to("cpu", torch.float16),
                    x.to(torch.ones(1, dtype=torch.int32)))

        for core in (False, True):
            with self.subTest(core=core):
                self.assert_passes(self.verify(moved, x, core=core))

    def test_float8_conversions_round_once_with_device_overflow(self):
        # A value of float8_e5m2, float8_e4m3fn, float8_e4m3fnuz or float8_e5m2fnuz is an input, a
        # constant or a conversion's result, and only conversions and the outputs read it. Every
        # word of each converts exactly to binary64, binary32, bfloat16 and binary16, and a
        # binary32, bfloat16, binary16, integer or Boolean value converts to it rounded once to
        # nearest, ties to even, as PyTorch does word for word: an e5m2 value from 61440 up is
        # infinite, a fnuz format's overflow is its NaN and its -0 is +0, and float8_e4m3fn
        # follows the captured device's conversion policy: finite saturation or overflow to NaN.
        # A NaN's payload is left open. PyTorch rounds a binary64 value through binary32 first, and the
        # capture rounds it once: 1.0625 + 2^-40 is above the midpoint of 1 and 1.125 of the
        # e4m3 formats, which it is not in binary32. A float8 value is moved too, by reshape, view,
        # flatten, permute, transpose, t, unsqueeze, squeeze, expand, slice, contiguous and clone.
        # float8_e8m0fnu and every other operator of a float8 value are refused.
        torch = self.torch
        # The precision and exponent range of each format, for `_nearest`.
        formats = {"float8_e5m2": (3, -14, 15), "float8_e4m3fn": (4, -6, 8),
                   "float8_e4m3fnuz": (4, -7, 7), "float8_e5m2fnuz": (3, -15, 15)}
        generator = torch.Generator().manual_seed(3)
        values = torch.cat([
            torch.randn(256, generator=generator)
            * 2.0 ** torch.randint(-20, 18, (256,), generator=generator).float(),
            torch.tensor([float("inf"), -float("inf"), float("nan"), -0.0, 448, 449, 464, 464.5,
                          479.9, 480, 57344, 61439, 61440, 1e30, -1e30, 240, 248, 256,
                          2.0 ** -10, 2.0 ** -17, 1.5 * 2.0 ** -17])])
        doubles = [1.0625 + 2.0 ** -40, -1.0625 - 2.0 ** -40, 1.125 + 2.0 ** -40,
                   61440 - 2.0 ** -20]
        integers = torch.tensor([0, 1, -3, 100, 255, 447, 449, 465, 480, 1000, -1000, 70000])
        # Compile with an empty policy cache too: a backend's tracing context must not make the
        # real device probe into a fake tensor or require a prior verify call.
        from torchlean._pointwise import _e4m3_saturates

        _e4m3_saturates.cache_clear()
        converted = torchlean.compile(lambda value: value.to(torch.float8_e4m3fn), checker=CHECKER)
        converted(values)
        self.assertTrue(converted.torchlean.report["ok"], converted.torchlean.report)
        movements = (lambda v: v.view(8, 32).t(), lambda v: v.permute(2, 0, 1).reshape(-1, 4),
                     lambda v: v.transpose(0, 1)[:, 2:5].float(),
                     lambda v: v[1:2].unsqueeze(1).expand(1, 3, 8, 8).squeeze(0),
                     lambda v: torch.flatten(v, 1).contiguous().clone(),
                     lambda v: (v.float() / 4).to(v.dtype).view(-1, 16).double())
        for name, layout in formats.items():
            dtype = getattr(torch, name)
            every = torch.arange(256, dtype=torch.uint8).view(dtype)
            for source in (torch.float32, torch.bfloat16, torch.float16):
                for core in (False, True):
                    with self.subTest(name=name, source=source, core=core):
                        self.assert_passes(self.verify(lambda v: (v.to(dtype), v.to(dtype).float()),
                                                       values.to(source), core=core))
            for target in (torch.float64, torch.float32, torch.bfloat16, torch.float16,
                           *(getattr(torch, other) for other in formats if other != name)):
                with self.subTest(name=name, target=target):
                    self.assert_passes(self.verify(lambda v: v.to(target), every))
            for index, function in enumerate(movements):
                for core in (False, True):
                    with self.subTest(name=name, movement=index, core=core):
                        self.assert_passes(self.verify(function, every.reshape(4, 8, 8),
                                                       core=core))
            for source in (torch.int64, torch.int32, torch.int16, torch.uint8, torch.bool):
                with self.subTest(name=name, source=source):
                    self.assert_passes(self.verify(lambda v: v.to(dtype).float(),
                                                   integers.to(source)))
            with self.subTest(name=name, source=torch.float64):
                x = torch.cat([values.double(), torch.tensor(doubles, dtype=torch.float64)])
                report = self.verify(lambda v: v.to(dtype), x, checks=EXECUTE)
                self.assert_passes(report)
                once = [torch.tensor(_nearest(value, layout)).to(dtype) for value in doubles]
                wanted = torch.cat([values.to(dtype), torch.stack(once)])
                got = torch.tensor(report["outputs"][0]["bits"], dtype=torch.uint8).view(dtype)
                _same(self, got.double().tolist(), wanted.double().tolist())
                self.assertNotEqual(x[-4:].to(dtype).view(torch.uint8).tolist(),
                                    torch.stack(once).view(torch.uint8).tolist())
        # FX's ShapeProp keeps the real device too, even though TensorMetadata omits it.
        from torchlean._fx import _tensor_value
        from torchlean.export import export_torchlean_graph_json

        class Convert(torch.nn.Module):
            def forward(self, value):
                return value.to(torch.float8_e4m3fn)

        with tempfile.TemporaryDirectory() as directory:
            artifact = export_torchlean_graph_json(Convert(), (values,),
                str(Path(directory) / "graph.json"), prefer_torch_export=False)
        artifact["arguments"] = [_tensor_value(values)]
        artifact["reference_outputs"] = [_tensor_value(Convert()(values))]
        self.assert_passes(torchlean.verify(artifact, checks=COMPARE, checker=CHECKER))
        to_e4m3 = lambda v: v.to(torch.float8_e4m3fn)
        for function in (lambda v: to_e4m3(v) * 2, lambda v: -to_e4m3(v),
                         lambda v: torch.cat([to_e4m3(v), to_e4m3(v)])):
            self.assert_refused(self.verify(function, values, checks=("structure",)),
                                "a float8 value is read only by conversions and movements")
        self.assert_refused(self.verify(lambda v: v.to(torch.float8_e8m0fnu).float(), values,
                                        checks=("structure",)), "torch.float8_e8m0fnu")

    def test_reflected_subtraction_matches_eager_words(self):
        # torch.export writes s - x as rsub.Scalar and torch.rsub(x, y) as rsub.Tensor, which are
        # sub of their operands swapped. The scalar converts to the result dtype, as PyTorch
        # converts it, and an integer tensor below a float promotes to binary32. rsub with an
        # alpha is refused, as sub with one is.
        torch = self.torch
        y = torch.linspace(-3, 3, 24).reshape(2, 3, 4)

        def reflected(x, y):
            i = x.to(torch.int32)
            return (1 - x, 0.1 - y, torch.rsub(x, y), torch.rsub(y, 2.5), 3 - i, 1.5 - i)

        for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            for core in (False, True):
                with self.subTest(dtype=dtype, core=core):
                    self.assert_passes(self.verify(reflected, self.x.to(dtype), y.to(dtype),
                                                   core=core))
        self.assert_refused(self.verify(lambda x: torch.rsub(x, 3, alpha=2), self.x,
                                        checks=("structure",)), "alpha != 1")

    def test_silu_is_the_registered_operator(self):
        # F.silu and nn.SiLU are Op.silu, binary16 and bfloat16 in binary32 rounded once. Where exp
        # is exact or overflows the words agree with eager PyTorch 2.13, and on 256 Gaussian
        # samples the results are within four ulps of eager in binary32 and binary64 and one in
        # binary16 and bfloat16. Eager's x / (1 + exp (-x)) is a NaN at -inf, where Op.silu returns
        # the limit -0, and -0 once exp (-x) overflows, where binary32 Op.silu(-100) is within a
        # subnormal step of -100 * exp(-100), about -3.72e-42.
        import math

        torch = self.torch
        functional = torch.nn.functional
        inf, nan = float("inf"), float("nan")
        exact = torch.tensor([nan, inf, 0.0, -0.0, 50.0, 800.0, -800.0, 1e4, -1e4])
        samples = torch.randn(256, generator=torch.Generator().manual_seed(2),
                              dtype=torch.float64) * 8
        module = torch.nn.SiLU()
        for dtype, ulps in ((torch.float32, 4), (torch.float64, 4), (torch.float16, 1),
                            (torch.bfloat16, 1)):
            with self.subTest(dtype=dtype):
                self.assert_passes(self.verify(functional.silu, exact.to(dtype)))
                self.assert_passes(self.verify(module, exact.to(dtype)))
                x = samples.to(dtype)
                report = self.verify(functional.silu, x, checks=EXECUTE)
                self.assert_passes(report)
                (actual,) = _outputs(report)
                _assert_within_ulps(self, actual, functional.silu(x), ulps)
                self.assert_passes(self.verify(functional.silu, x, checks=EXECUTE, core=True))
                limit = self.verify(functional.silu, torch.tensor([-inf], dtype=dtype))
                self.assertEqual({name: status for name, (status, _) in
                                  _statuses(limit).items()},
                                 {"structure": "checked", "types": "checked",
                                  "execute": "executed", "compare": "failed"})
                (value,) = _outputs(limit)
                self.assertEqual((value.item(), value.signbit().item()), (0.0, True))
                self.assertTrue(functional.silu(torch.tensor([-inf], dtype=dtype)).isnan().all())
        (tail,) = _outputs(self.verify(functional.silu, torch.tensor([-100.0]), checks=EXECUTE))
        self.assertLessEqual(abs(tail.item() + 100 * math.exp(-100)), 2.0 ** -149)
        self.assertEqual(functional.silu(torch.tensor([-100.0])).item(), 0.0)
        kinds = _tags(self.artifact(functional.silu, samples))
        self.assertIn("silu", kinds)
        self.assertNotIn("div", kinds)
        # A binary16 operand is computed in binary32 and rounded once.
        half = self.artifact(functional.silu, samples.to(torch.float16))
        (scalar,) = [scalar for record, scalar in _ops(half) if record["tag"] == "silu"]
        self.assertEqual(scalar, "float32")

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

    def test_integer_and_boolean_sorts_match_eager(self):
        # The stable sort of an integer type orders its values, signed or unsigned, and of
        # Booleans puts false first: Op.sort at the input's element. Equal entries keep their
        # input order in both directions, so the values and int64 indices are the eager words.
        torch = self.torch
        x = torch.tensor([[3, -1, 3, 0, -1, 7], [0, 0, -128, 127, 0, -128], [5, 4, 5, 4, 5, 4]])

        def sorts(x):
            return (*torch.sort(x, dim=1, stable=True),
                    *torch.sort(x, dim=0, descending=True, stable=True),
                    torch.sort(x, descending=True, stable=True).indices)

        for dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8):
            with self.subTest(dtype=dtype):
                self.assert_passes(self.verify(sorts, x.to(dtype)))
        with self.subTest(dtype=torch.bool):
            self.assert_passes(self.verify(sorts, x > 0))
        elements = [record["fields"][3] for record in _records(self.artifact(sorts, x))
                    if record["tag"] == "sort"]
        integer = {"element": {"integer": {"signed": True, "width": 64}}}
        self.assertGreaterEqual(len(elements), 3)
        self.assertEqual(elements, [integer] * len(elements))

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
        # leaky_relu is Op.leakyReLU at every slope, zero and negative ones included, whose pullback
        # sends the slope at x = 0 as torch's does (TorchLeanTest.OpTable), where the maximum of x
        # and x * slope the capture wrote before split the seed at the tie. diagonal is a
        # permutation, a reshape and a strided slice.
        torch = self.torch
        functional = torch.nn.functional
        generator = torch.Generator().manual_seed(1)
        x = torch.randn(3, 20, generator=generator) * 10
        x[0, :5] = torch.tensor([0.0, -0.0, float("nan"), float("inf"), -float("inf")])

        def leaky(x):
            return (functional.leaky_relu(x, 0.2), functional.leaky_relu(x, 3.0),
                    torch.nn.LeakyReLU()(x), functional.leaky_relu(x, 0.0),
                    functional.leaky_relu(x, -0.5))

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
        for core in (False, True):
            kinds = _tags(self.artifact(leaky, x, core=core))
            self.assertIn("leakyRelu", kinds)
            self.assertNotIn("max", kinds)
            self.assertNotIn("min", kinds)
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
        # and scatter and index_put fold
        # repeated targets in row-major order of the index tensor, which these small cases share
        # with ATen's CPU kernels (PyTorch documents those orders as unspecified). index_add folds
        # them in the order of its index, as documented, and index_copy keeps the last copy, as
        # its CPU kernel does. index_put_, index_add_, index_copy_, scatter_, scatter_add_ and
        # copy_ are their functional operators on a fresh destination.
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
                    torch.take_along_dim(x, i[:, :2].repeat(2, 1)[:3], 1), x[n], x[:, n],
                    x[n, n % 4], cube[:, i[0, :2], i[1, :2]], cube[i[:, :2] % 2, :, i[:, 1:3]],
                    x[torch.tensor(1)], x.gather(1, x.argmax(1, keepdim=True)),
                    torch.index_select(x, 1, (x[0] > 0).long()))

        def writes(x, i, r, u, h):
            return (x.scatter(0, i[:, :3], u[:2, :3]), torch.scatter_add(x, 0, i, u[:2]),
                    x.scatter(1, i[:, :2], 2.5), torch.scatter_add(x, 1, r, h), x.scatter(1, r, u),
                    torch.index_put(x, (i[0, :3],), u[:3]),
                    torch.index_put(x, (i[0], i[1]), h[0], accumulate=True),
                    torch.zeros(4).scatter_add(0, r[2], h[0]), x.index_add(0, r[0, 1:], u),
                    torch.index_add(x, 1, r[1], h), torch.index_add(x[0], -1, r[2], h[0]),
                    x.index_copy(0, r[0, 1:], u), torch.index_copy(x, 1, r[1], h),
                    x[0].index_copy(-1, r[2], h[0]))

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

        def scattered(x, i, r, u, h):
            # The default graph keeps scatter_, scatter_add_, index_add_, index_copy_ and copy_,
            # read as scatter, scatter_add, index_add, index_copy and copy of their fresh
            # destinations. y and z are written more than once.
            y = x.clone()
            y.index_add_(1, r[1], h)
            y.scatter_(0, i[:, :3], u[:2, :3])
            z = torch.zeros(3, 4)
            z.scatter_add_(1, r, h)
            z.scatter_(1, i[:, :1], 1.5)
            z.index_add_(0, r[0, 1:], u)
            w = torch.zeros(2, 4, dtype=torch.float64)
            w.copy_(x[0])
            v = x * 1.0
            v.index_copy_(1, r[1], h)
            return y, z * 2, w + 1, v

        u = torch.randn(3, 4, generator=generator)
        for core in (False, True):
            with self.subTest(core=core):
                self.assert_passes(self.verify(reads, x, rows, signed, torch.tensor([[1, 4, 0],
                                                                                     [3, 3, 2]]),
                                               core=core))
                self.assert_passes(self.verify(writes, x, rows, repeated, u, heavy, core=core))
                self.assert_passes(self.verify(scattered, x, rows, repeated, u, heavy, core=core))
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
        self.assert_refused(self.verify(lambda x, i, u: x.scatter_(0, i[:, :3], u[:2, :3]),
                                        x.clone(), rows, u), "scatter_ into an input")
        self.assert_refused(self.verify(lambda x, i, u: x.index_add_(0, i[0, :3], u),
                                        x.clone(), rows, u), "index_add_ into an input")
        self.assert_refused(self.verify(lambda x, i, u: x.index_copy_(0, i[0, :3], u),
                                        x.clone(), rows, u), "index_copy_ into an input")
        # ATen fuses alpha into the sum, as add does, and add with alpha is refused.
        self.assert_refused(self.verify(lambda x, i, u: x.index_add(0, i[0, :3], u, alpha=2),
                                        x, rows, u, checks=("structure",)), "alpha other than 1")
        # index_add and index_copy reject a negative index, and a constant one is refused.
        for function in (lambda x: x.index_add(0, torch.tensor([-1]), x[:1]),
                         lambda x: x.index_copy(0, torch.tensor([-1]), x[:1])):
            self.assert_refused(self.verify(function, x, checks=("structure",)), "outside an axis")
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

    def test_take_along_dim_keeps_the_runtime_index_rule(self):
        torch = self.torch
        x = torch.arange(4, dtype=torch.float32).reshape(1, 4)
        indices = torch.tensor([[-1, 4]])

        def take(x, index):
            return torch.take_along_dim(x, index, dim=1)

        try:
            expected = take(x, indices)
        except RuntimeError as error:
            self.assertIn("out of bounds", str(error))
            expected = None
        for core in (False, True):
            with self.subTest(core=core):
                report = self.verify(take, x, indices, checks=EXECUTE, core=core)
                if expected is None:
                    self.assert_unsupported_execution(report, "outside")
                else:
                    self.assert_passes(report)
                    torch.testing.assert_close(_outputs(report)[0], expected, atol=0, rtol=0)

    def test_runtime_gathers_and_replacing_writes_move_integer_and_boolean_words(self):
        # index.Tensor, index_select, gather and take_along_dim at a runtime index, and index_put
        # without accumulate, index_copy and scatter of a tensor, move entries without reading
        # them: Op.gather and Op.scatter at the element of an integer or Boolean input, the eager
        # words (repeated targets as in test_runtime_indices_match_eager). The writes that sum,
        # and scatter of a Python value, are imported at floating types only.
        torch = self.torch
        generator = torch.Generator().manual_seed(5)
        x = torch.randint(-128, 128, (3, 4), generator=generator)
        cube = torch.randint(-128, 128, (2, 3, 4), generator=generator)
        u = torch.randint(-128, 128, (3, 4), generator=generator)
        rows = torch.tensor([[0, 2, 1, 0], [1, 1, 0, 2]])
        signed = torch.tensor([[-1, 2, -3], [0, -2, 1]])
        repeated = torch.tensor([[0, 0, 0, 1], [2, 2, 1, 1], [3, 3, 3, 3]])

        def moves(x, c, u, i, n, r):
            y = x.clone()
            y[:, i[1, :2]] = u[:, :2]
            return (torch.gather(x, 0, i), torch.gather(x, 1, i[:, :2]),
                    torch.index_select(x, 1, i[0]), torch.take_along_dim(x, i),
                    torch.take_along_dim(x, i[:, :2].repeat(2, 1)[:3], 1), x[n], x[:, n],
                    c[:, i[0, :2], i[1, :2]], x.scatter(0, i[:, :3], u[:2, :3]),
                    x.scatter(1, r, u), torch.index_put(x, (i[0, :3],), u[:3]),
                    torch.index_put(x, (i[0], i[1]), u[0]), x.index_copy(0, r[0, 1:], u),
                    torch.index_copy(x, 1, r[1], u), y)

        for dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool):
            def cast(t):
                return t > 0 if dtype == torch.bool else t.to(dtype)
            for core in (False, True):
                with self.subTest(dtype=dtype, core=core):
                    self.assert_passes(self.verify(moves, cast(x), cast(cube), cast(u), rows,
                                                   signed, repeated, core=core))
        elements = {json.dumps(record["fields"][-1]) for record in _records(self.artifact(
            moves, x.int(), cube.int(), u.int(), rows, signed, repeated))
                    if record["tag"] in ("gather", "scatter")}
        self.assertEqual(elements, {json.dumps({"element": {"integer": {"signed": True,
                                                                         "width": 32}}})})
        for function in (lambda x, u, i: torch.index_put(x, (i[0],), u[0], accumulate=True),
                         lambda x, u, i: torch.scatter_add(x, 0, i, u[:2]),
                         lambda x, u, i: x.index_add(0, i[0, :3], u),
                         lambda x, u, i: x.scatter(1, i[:, :2], 2)):
            for values in (x, x > 0):
                self.assert_refused(self.verify(function, values, u.to(values.dtype), rows,
                                                checks=("structure",)), "tensor is not imported")

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
                 "embedding": (lambda x, n: functional.embedding(n, x), [[-1, 2]], "outside"),
                 # PyTorch's index_add and index_copy reject a negative index, which index_put
                 # wraps.
                 "index_add": (lambda x, n: x.index_add(0, n[0], x[:2]), [[-1, 0]], "outside"),
                 "index_copy": (lambda x, n: x.index_copy(0, n[0], x[:2]), [[-1, 0]], "outside")}
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
        # index_put with accumulate rounds each step, as its kernel does, and so does index_add,
        # as its CPU kernel of a vector and PyTorch's decomposition do. Its CPU kernel of a matrix
        # sums in binary32: there the comparison fails, the IR's binary16 2048 + 1 + 1 is 2048
        # and eager's 2050 (bfloat16 rounds both to 2048). An earlier reader of an
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

        def index_sums(zero, at, small):
            return torch.index_add(zero, 0, at, small), zero.clone().index_add_(0, at, small)

        def column_sums(zero, at, small):
            return torch.index_add(zero[:, None], 0, at, small[:, None])

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
                    zero = torch.zeros(2, dtype=dtype)
                    self.assert_passes(self.verify(index_sums, zero, at, small.to(dtype),
                                                   core=core))
                    report = self.verify(column_sums, zero, at, small.to(dtype), core=core)
                    if dtype == torch.float16:
                        self.assertEqual({name: status for name, (status, _) in
                                          _statuses(report).items()},
                                         {"structure": "checked", "types": "checked",
                                          "execute": "executed", "compare": "failed"})
                        self.assertEqual([value.tolist() for value in _outputs(report)],
                                         [[[2048.0], [65504.0]]])
                        self.assertEqual(column_sums(zero, at, small.to(dtype)).tolist(),
                                         [[2050.0], [65504.0]])
                    else:
                        self.assert_passes(report)
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
        # not pocketfft's schedule, so the values agree within a tolerance. The imaginary parts
        # of frequency 0 and, for an even n, n / 2 are ignored, as the irfft page says, so
        # infinities and NaNs there leave the signal finite.
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
        z = torch.randn(2, 4, 2, generator=generator)
        z[:, 0, 1] = torch.tensor([float("inf"), float("nan")])
        z[:, 3, 1] = torch.tensor([float("nan"), -float("inf")])
        w = z.transpose(0, 1).contiguous()

        def inverses(z, w):
            return (torch.fft.irfft(torch.view_as_complex(z), n=6),
                    torch.fft.irfft(torch.view_as_complex(z), n=4),
                    torch.fft.irfft(torch.view_as_complex(z), n=5, norm="ortho"),
                    torch.fft.irfft(torch.view_as_complex(w), dim=0))

        for dtype in (torch.float32, torch.float64):
            for core in (False, True):
                with self.subTest(dtype=dtype, core=core, ignored=True):
                    report = self.verify(inverses, z.to(dtype), w.to(dtype), checks=EXECUTE,
                                         core=core)
                    self.assert_passes(report)
                    for got, want in zip(_outputs(report), inverses(z.to(dtype), w.to(dtype)),
                                         strict=True):
                        self.assertTrue(bool(torch.isfinite(want).all()))
                        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
        report = self.verify(lambda x: torch.fft.rfft(x).abs(), x, checks=("structure",))
        self.assert_refused(report, "paired-real form")

    def test_complex_values_in_paired_real_form(self):
        # Between the transforms a complex value is held in paired-real form: its slices and
        # reshapes, real and imag, and its quotient by a number, c10::complex's division at
        # (d, 0), give eager's words. The product of two complex tensors, broadcast, agrees within
        # a tolerance, as in SOL-ExecBench's Hyena convolutions (L1/007, L2/051,
        # L2/052).
        torch = self.torch
        generator = torch.Generator().manual_seed(11)
        special = [0.0, -0.0, 1.5, -2.5, float("inf"), -float("inf"), float("nan")]
        y = torch.tensor([[[p, q] for q in special] for p in special])

        def exact(y):
            z = torch.view_as_complex(y)
            return (*[torch.view_as_real(z / d) for d in (3, -7, 0.1, 2 ** 40 + 1)],
                    torch.view_as_real(z[1:5:2, -3:].unsqueeze(1).reshape(2, 3)), z.real, z.imag,
                    torch.view_as_real(z.view(49)[::3]))

        for dtype in (torch.float32, torch.float64):
            for core in (False, True):
                with self.subTest(dtype=dtype, core=core, exact=True):
                    self.assert_passes(self.verify(exact, y.to(dtype), checks=COMPARE, core=core))
        x = torch.randn(2, 3, 5, generator=generator)
        k = torch.randn(3, 5, generator=generator)

        def hyena(x, k):
            n = 2 * x.shape[-1]
            x_f = torch.fft.rfft(x, n=n)
            k_f = torch.fft.rfft(k, n=n) / n
            scaled = x_f / n
            return (scaled.real.contiguous(), scaled.imag.contiguous(),
                    torch.fft.irfft(x_f * k_f, n=n, norm="forward")[..., :x.shape[-1]],
                    torch.fft.irfft(x_f * k_f.unsqueeze(0), n=n, norm="forward"),
                    torch.view_as_real(k_f * x_f[1]))

        for dtype in (torch.float32, torch.float64):
            for core in (False, True):
                with self.subTest(dtype=dtype, core=core):
                    report = self.verify(hyena, x.to(dtype), k.to(dtype), checks=EXECUTE,
                                         core=core)
                    self.assert_passes(report)
                    for got, want in zip(_outputs(report), hyena(x.to(dtype), k.to(dtype)),
                                         strict=True):
                        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
        for function, reason in [
                (lambda y: torch.view_as_complex(y), "an output of a complex value"),
                (lambda y: torch.view_as_complex(y).abs(), "paired-real form"),
                (lambda y: torch.view_as_real(torch.view_as_complex(y) / y[..., 0]),
                 "only by a real number"),
                (lambda y: torch.view_as_real(torch.view_as_complex(y) / 0), "by zero"),
                (lambda y: torch.view_as_real(torch.view_as_complex(y) * 2), "a real value")]:
            with self.subTest(reason=reason):
                self.assert_refused(self.verify(function, y, checks=("structure",)), reason)
        self.assert_refused(self.verify(torch.view_as_real, torch.view_as_complex(y),
                                        checks=("structure",)), "complex64 of an input")

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
        # A negative or NaN eps: the root of var + eps as it stands. Channels 0 and 1, the first
        # group, are constant, so there the sum of each row, group and channel is negative or NaN
        # and both give NaN. functional.batch_norm refuses an eps <= 0 in training, the ATen
        # operator does not.
        y = x.clone()
        y[:, :2] = 1
        for eps in (-0.25, float("nan")):
            cases = {"group norm": nn.GroupNorm(3, 6, eps=eps).eval(),
                     "layer norm": nn.LayerNorm(4, eps=eps).eval(),
                     "batch norm": lambda x: torch.ops.aten._native_batch_norm_legit.no_stats(
                         x, None, None, True, 0.1, eps)[0]}
            for name, model in cases.items():
                for dtype in (torch.float32, torch.float64):
                    for core in (False, True):
                        with self.subTest(name=name, eps=eps, dtype=dtype, core=core):
                            if isinstance(model, nn.Module): model.to(dtype)
                            report = self.verify(model, y.to(dtype), checks=EXECUTE, core=core)
                            self.assert_passes(report)
                            with torch.no_grad():
                                expected = model(y.to(dtype))
                            self.assertTrue(bool(expected[:, :2].isnan().all()))
                            torch.testing.assert_close(_outputs(report)[0], expected, rtol=1e-5,
                                                       atol=1e-5, equal_nan=True)

    def test_training_batch_norm_updates_running_statistics(self):
        # Training-mode batch norm with running statistics: the program returns the normalized
        # values, then the updated running_mean, running_var and num_batches_tracked in buffer
        # order, with no mutation in the IR. The moments agree within a tolerance and the counter
        # exactly. The default and core graphs import one fixture to the same words. So does a
        # direct aten._native_batch_norm_legit call in training, whose saved mean and reciprocal
        # standard deviation come before the updates.
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
            "_native_batch_norm_legit, statistics as inputs": (
                lambda x, m, v: torch.ops.aten._native_batch_norm_legit(
                    x, None, None, m, v, True, 0.1, 1e-5)[0], inputs,
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

        from torchlean.export import export_torchlean_graph_json

        # A distinct view of the buffer reaches the shared storage check in both entry points.
        arguments = (x, alias.norm.running_mean.view(3))
        self.assert_refused(self.verify(alias, *arguments, checks=("structure",)),
                            "shares storage with an input")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "alias.json"
            with self.assertRaisesRegex(NotImplementedError, "shares storage with an input"):
                export_torchlean_graph_json(alias, arguments, str(path), require_torch_export=True)
            self.assertFalse(path.exists())

        class Legit(nn.Module):
            # aten._native_batch_norm_legit in training with buffer statistics.
            def __init__(self, affine):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(3)) if affine else None
                self.bias = nn.Parameter(torch.randn(3)) if affine else None
                self.register_buffer("mean", torch.randn(3))
                self.register_buffer("var", torch.rand(3) + 0.5)

            def forward(self, x):
                return torch.ops.aten._native_batch_norm_legit(
                    x, self.weight, self.bias, self.mean, self.var, True, 0.3, 1e-5)

        for affine in (True, False):
            for dtype in (torch.float32, torch.float64):
                model = Legit(affine).to(dtype)
                x = (torch.randn(5, 3, 2) * 2 + 1).to(dtype)
                kept = {key: value.clone() for key, value in model.state_dict().items()}
                reference = copy.deepcopy(model)
                with torch.no_grad():
                    expected = (*reference(x), reference.mean, reference.var)
                words = {}
                for core in (False, True):
                    with self.subTest(case="_native_batch_norm_legit", affine=affine,
                                      dtype=dtype, core=core):
                        report = self.verify(model, x, checks=EXECUTE, core=core)
                        self.assert_passes(report)
                        outputs = _outputs(report)
                        self.assertEqual(len(outputs), 5)
                        for got, want in zip(outputs, expected):
                            self.assertEqual(got.dtype, dtype)
                            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
                        for key, value in model.state_dict().items():
                            self.assertTrue(torch.equal(value, kept[key]), key)
                        words[core] = [output["bits"] for output in report["outputs"]]
                with self.subTest(case="_native_batch_norm_legit", affine=affine, dtype=dtype,
                                  words="default = core"):
                    self.assertEqual(sorted(words), [False, True])
                    self.assertEqual(words[False], words[True])

    def test_grid_sampling_nonfinite_coordinates_follow_the_docstring(self):
        # grid_sample's docstring reads a NaN coordinate as -1, and the capture reads an infinite
        # one as the limit of the finite ones beyond it: 0 outside the image with zeros padding,
        # the edge with border padding, the only pixel of a singleton axis that align_corners or
        # border padding keeps there. So the reference is eager CPU at the grid with NaN
        # replaced by -1 and an infinity by 1e6 of its sign, equal values with equal zero signs.
        # Eager CPU at the grid itself gives NaN in bilinear mode at both and 0 in nearest mode
        # at NaN with zeros padding. A 1x2 and a 2x1 image have one singleton axis, and so one
        # coordinate that zeros padding can put outside the image.
        torch = self.torch
        grid = torch.tensor([[[[float("nan"), 0.0], [float("inf"), 0.0],
                               [-float("inf"), 0.0], [0.0, float("nan")],
                               [0.0, float("inf")], [0.0, -float("inf")],
                               [0.0, 0.0], [-1.0, -1.0], [1.0, 1.0]]]])
        image = torch.arange(1.0, 5.0).reshape(1, 1, 2, 2)
        singleton = torch.ones(1, 1, 1, 1)
        row, column = torch.tensor([[[[1.0, 2.0]]]]), torch.tensor([[[[1.0], [2.0]]]])

        def sampled(image, singleton, row, column, grid):
            return tuple(torch.nn.functional.grid_sample(
                value, grid, mode=mode, padding_mode=padding, align_corners=align)
                for value in (image, singleton, row, column) for mode in ("bilinear", "nearest")
                for padding in ("zeros", "border") for align in (False, True))

        for dtype in (torch.float32, torch.float64):
            for core in (False, True):
                with self.subTest(dtype=dtype, core=core):
                    arguments = tuple(value.to(dtype)
                                      for value in (image, singleton, row, column, grid))
                    report = self.verify(sampled, *arguments, checks=EXECUTE, core=core)
                    self.assert_passes(report)
                    documented = arguments[4].nan_to_num(nan=-1.0, posinf=1e6, neginf=-1e6)
                    outputs = _outputs(report)
                    expected = sampled(*arguments[:4], documented)
                    self.assertEqual(len(outputs), len(expected))
                    for actual, want in zip(outputs, expected):
                        _same(self, actual.reshape(-1).tolist(), want.reshape(-1).tolist())

    def test_vector_norms_within_tolerance(self):
        # linalg.vector_norm, and torch.norm of a number p, at ord 2, 1, 0, inf, -inf, 3, -1.5 and
        # 0.5, over one axis, two and all, kept or not, and with a dtype argument. Binary16 and
        # bfloat16 norms are binary32 ones rounded once. The CPU kernel vectorizes its sums, so the
        # values agree within a tolerance.
        torch = self.torch
        linalg = torch.linalg
        generator = torch.Generator().manual_seed(11)
        x = torch.randn(3, 4, 5, generator=generator)
        x[0, 0, 0] = 0
        cases = {"two": lambda x: linalg.vector_norm(x, dim=-1),
                 "one kept": lambda x: linalg.vector_norm(x, 1, (0, 2), keepdim=True),
                 "zero": lambda x: linalg.vector_norm(x, 0, 1),
                 "inf": lambda x: linalg.vector_norm(x, float("inf")),
                 "-inf": lambda x: linalg.vector_norm(x, float("-inf"), dim=(1, 2)),
                 "three": lambda x: linalg.vector_norm(x, 3, 0),
                 "negative": lambda x: linalg.vector_norm(x, -1.5, 2),
                 "half power": lambda x: linalg.vector_norm(x, 0.5, 2),
                 "torch.norm": lambda x: torch.norm(x, p=2, dim=(1, 2), keepdim=True) * x,
                 "dtype": lambda x: linalg.vector_norm(x, dim=1, dtype=torch.float64)}
        for name, function in cases.items():
            for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
                for core in (False, True):
                    with self.subTest(name=name, dtype=dtype, core=core):
                        value = x.to(dtype)
                        report = self.verify(function, value, checks=EXECUTE, core=core)
                        self.assert_passes(report)
                        torch.testing.assert_close(_outputs(report)[0], function(value))
        # PyTorch raises for an infinite ord over an empty fiber and for an integer input before
        # the capture sees the call. The importer refuses both on constructed graphs too.
        for order, value, result in ((float("inf"), torch.zeros(0, 3), torch.zeros(3)),
                                     (2, torch.zeros(3, dtype=torch.int32), torch.zeros(()))):
            with self.subTest(order=order, dtype=value.dtype):
                graph = torch.fx.Graph()
                operand = graph.placeholder("x")
                operand.meta["val"] = value
                node = graph.call_function(torch.ops.aten.linalg_vector_norm.default,
                                           (operand, order, [0]))
                node.meta["val"] = result
                graph.output(node)
                with self.assertRaises(NotImplementedError):
                    torchlean._lower_pytorch_graph(torch.nn.Module(), graph, ["x"])

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
                        results = zip(_outputs(report), function(*arguments))
                        for index, (got, want) in enumerate(results):
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
                                    inside = (diagonal >= 0) & (diagonal <= bound)
                                    self.assertTrue(bool(inside.all()), (diagonal, bound))
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
                                            _uniform(first["fields"][1]["nat"], 12,
                                                     torch.float32)))
                order = [record for record in records if record["tag"] == "randUniform"][1]
                draw = _uniform(order["fields"][1]["nat"], 7, torch.float64).tolist()
                self.assertEqual(outputs[1]["bits"], sorted(range(7), key=lambda i: (draw[i], i)))
                kept = _outputs({"outputs": [outputs[2]]})[0]
                mask = torch.tensor(outputs[3]["bits"]).reshape(4, 5).bool()
                (dropout,) = [record for record in records if record["tag"] == "bernoulliMask"][:1]
                expected = _uniform(dropout["fields"][1]["nat"], 20, torch.float32) < 0.75
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
        from torchlean._artifact import _random_key, _splitmix64
        from torchlean._emit import _emit_structure

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
        for kind in ("sort", "cumsum", "cummax", "topk", "leakyRelu"):
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

    def test_householder_products_follow_their_docstrings(self):
        # householder_product(input, tau), its alias orgqr and ormqr(input, tau, other, left,
        # transpose) apply the reflectors H_i = 1 - tau_i v_i v_i^T of input's columns one at a
        # time in the order of Q = H_0 ... H_{k-1} (LinalgImport.reflected), householder_product
        # to the identity's first n columns, also with k < n reflectors (an input of their own,
        # as FX imports no tensor indexing). The reference is that arithmetic in NumPy's binary32
        # (_reflect32); PyTorch's LAPACK results are within 1e-5 of it. The capture through
        # torch.export, its core decomposition and FX agree.
        import itertools

        import numpy as np

        from torchlean._fx import _tensor_value
        from torchlean.export import export_torchlean_graph_json

        torch = self.torch
        generator = torch.Generator().manual_seed(15)
        a, taus = torch.geqrf(torch.randn(2, 5, 3, generator=generator))
        c_left = torch.randn(2, 5, 4, generator=generator)
        c_right = torch.randn(2, 4, 5, generator=generator)
        cases = list(itertools.product((True, False), repeat=2))

        def products(a, taus, few, c_left, c_right):
            return (torch.linalg.householder_product(a, taus), torch.orgqr(a, taus),
                    a.orgqr(taus), torch.linalg.householder_product(a, few),
                    *(torch.ormqr(a, taus, c_left if left else c_right, left=left,
                                  transpose=transpose) for left, transpose in cases),
                    a.ormqr(taus, c_left, transpose=True))

        class Products(torch.nn.Module):
            def forward(self, a, taus, few, c_left, c_right):
                return products(a, taus, few, c_left, c_right)

        inputs = (a, taus, taus[..., :2].clone(), c_left, c_right)
        a32, taus32 = a.numpy(), taus.numpy()
        eye = np.eye(5, 3, dtype=np.float32)
        arguments = [(taus32, eye, True, False)] * 3 + [(taus32[:, :2], eye, True, False)]
        arguments += [(taus32, (c_left if left else c_right).numpy(), left, transpose)
                      for left, transpose in cases]
        arguments += [(taus32, c_left.numpy(), True, True)]
        wants = [np.stack([_reflect32(a32[b], t[b], c if c.ndim == 2 else c[b], left, transpose)
                           for b in range(2)]) for t, c, left, transpose in arguments]
        eager = products(*inputs)
        reports = [self.verify(products, *inputs, checks=EXECUTE, core=core)
                   for core in (False, True)]
        with tempfile.TemporaryDirectory() as directory:
            artifact = export_torchlean_graph_json(Products(), inputs,
                                                   str(Path(directory) / "graph.json"),
                                                   prefer_torch_export=False)
        artifact["arguments"] = [_tensor_value(x) for x in inputs]
        reports.append(torchlean.verify(artifact, checks=EXECUTE, checker=CHECKER))
        for capture, report in zip(("export", "core", "fx"), reports):
            self.assert_passes(report)
            for i, (got, want, lapack) in enumerate(zip(_outputs(report), wants, eager)):
                _same(self, got.flatten().tolist(), [float(v) for v in want.reshape(-1)],
                      (capture, i))
                self.assertTrue(np.allclose(want, lapack.numpy(), atol=1e-5), (capture, i))
        # Complex values are refused as an element type, and binary16 as LinalgImport's.
        z, z_taus = torch.geqrf(torch.randn(5, 3, dtype=torch.complex64, generator=generator))
        self.assert_refused(self.verify(torch.linalg.householder_product, z, z_taus,
                                        checks=EXECUTE), "element type: torch.complex64")
        self.assert_refused(self.verify(torch.linalg.householder_product, a.half(), taus.half(),
                                        checks=EXECUTE), "binary32 and binary64")

    def test_frexp_is_one_registered_operator(self):
        # torch.frexp is C's frexp, Op.frexp: a mantissa of magnitude in [1/2, 1) and an int32
        # exponent, the normalized mantissa of a subnormal, and (x, 0) at both zeros, the
        # infinities and a NaN. `compare` requires the eager words of both outputs.
        torch = self.torch
        x = torch.tensor([8.0, -3.0, 1.0, 0.5, 0.1, 0.0, -0.0, float("inf"), -float("inf"),
                          float("nan"), 1e-40, 2.0 ** -149, 3.4028234663852886e38])
        for dtype in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                self.assert_passes(self.verify(lambda x: torch.frexp(x), x.to(dtype)))
                self.assert_passes(self.verify(lambda x: torch.frexp(x).exponent * 2,
                                               x.to(dtype)))
        report = self.verify(lambda x: torch.frexp(x), x)
        mantissa, exponent = _words(report)
        self.assertEqual(mantissa[:4] + mantissa[10:],
                         [0x3f000000, 0xbf400000, 0x3f000000, 0x3f000000, 0x3f0b6100, 0x3f000000,
                          0x3f7fffff])
        self.assertEqual(exponent,
                         [4, 2, 1, 0, 0xfffffffd, 0, 0, 0, 0, 0, 0xffffff7c, 0xffffff6c, 128])
        self.assertIn("frexp", _tags(self.artifact(lambda x: torch.frexp(x), x)))
