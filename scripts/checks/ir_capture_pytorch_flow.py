"""PyTorch captures of torch.export's control-flow operators checked with ``verify ir``
(``PyTorchFlowTests``). Run them through ``scripts/checks/test_ir_capture.py``, which describes
the checks.
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from ir_capture_support import (CHECKER, COMPARE, EXECUTE, HAS_TORCH, ROOT, CheckerCase,
                                _blocks, _flows, _outputs, _records, _tags, _uniform)
import torchlean


@unittest.skipUnless(CHECKER and HAS_TORCH, "needs the built verify executable and PyTorch")
class PyTorchFlowTests(CheckerCase):
    # torch.cond, while_loop and map are Lean's flows (torchlean._hop) and invoke_subgraph is
    # inlined. Every case is in exact or once-rounded arithmetic, so PyTorch's words are the IR's.
    def setUp(self):
        import torch
        from torch._higher_order_ops.map import map as mapped
        from torch._higher_order_ops.while_loop import while_loop

        self.torch, self.while_loop, self.map = torch, while_loop, mapped
        self.x = torch.arange(12.0).reshape(3, 4) - 5

    def verify(self, function, *inputs, checks=COMPARE):
        return torchlean.verify(function, *inputs, checks=checks, checker=CHECKER)

    def artifact(self, function, *inputs):
        # The document the capture writes, before the checker reads it.
        return torchlean._capture(function, inputs, "pytorch", ("structure",))

    def module(self, forward, **attributes):
        # A module whose forward is `forward(self, *inputs)`, with the given tensor attributes:
        # parameters for floating tensors, buffers otherwise and under a name ending in `_buffer`.
        torch = self.torch

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                for name, value in attributes.items():
                    if name.endswith("_buffer") or not value.is_floating_point():
                        self.register_buffer(name, value)
                    else:
                        setattr(self, name, torch.nn.Parameter(value))

        Module.forward = forward
        return Module()

    def test_cond_selects_the_branch_of_its_predicate(self):
        # torch.cond is Lean's Flow.cond on its predicate's truth, a Boolean as it is and another
        # value compared with zero, of shape [] or [1]: only the selected branch runs. The
        # branches read a parameter, a buffer and a constant of the module, give two outputs and
        # nest a cond in a cond. A predicate known at capture, here of an integer buffer, selects
        # its branch there, and only that branch is checked and imported, without a flow, so the
        # other may hold what the capture refuses (`rsub` with `alpha`).
        torch = self.torch
        x, y = self.x, self.x.flip(0)
        scale = torch.tensor([4.0, 0.5, -1.0, 2.0])

        def read(self, p, a):
            return torch.cond(p > 0, lambda u: u @ self.weight + scale,
                              lambda u: u * self.shift_buffer, (a,))

        def truth(self, p, a):
            return torch.cond(p, lambda u: u * 2.0, lambda u: u - 1.0, (a,))

        def two(self, p, a, b):
            return torch.cond(p.sum() > 0, lambda u, v: (u * 2.0, v - u),
                              lambda u, v: (v * 1.0, -u), (a, b))

        def nested(self, p, q, a):
            return torch.cond(p > 0, lambda u, r: torch.cond(
                r > 0, lambda v: v * 0.5, lambda v: v + 3.0, (u,)), lambda u, r: u - 1.0, (a, q))

        def known(self, a):
            return torch.cond(self.count > 2, lambda u: u * 2.0,
                              lambda u: torch.rsub(u, 3, alpha=2), (a,))

        weighted = self.module(read, weight=torch.arange(16.0).reshape(4, 4) - 8,
                               shift_buffer=torch.full((4,), 3.0))
        signs = (torch.tensor(1.0), torch.tensor(-1.0))
        cases = [("read", weighted, [(s, x) for s in signs]),
                 ("truth", self.module(truth),
                  [(p, x) for p in (torch.tensor(3), torch.tensor(0), torch.tensor(-2),
                                    torch.tensor([True]), torch.tensor([False]))]),
                 ("two", self.module(two), [(torch.tensor([1.0, -3.0]), x, y),
                                            (torch.tensor([1.0, 3.0]), x, y)]),
                 ("nested", self.module(nested), [(p, q, x) for p in signs for q in signs])]
        for name, model, argument_sets in cases:
            for arguments in argument_sets:
                with self.subTest(case=name, scalars=[a.tolist() for a in arguments[:-1]]):
                    self.assertIn("cond", _flows(self.artifact(model, *arguments)))
                    self.assert_passes(self.verify(model, *arguments))
        model = self.module(known, count=torch.tensor(3))
        self.assertEqual(_flows(self.artifact(model, x)), [])
        report = self.verify(model, x)
        self.assert_passes(report)
        self.assertTrue(torch.equal(_outputs(report)[0], x * 2.0))
        self.assert_refused(self.verify(self.module(known, count=torch.tensor(1)), x),
                            "alpha != 1")

    def test_cond_runs_only_the_selected_branch(self):
        # An unselected branch never fails at execution: neither its integer division by zero
        # nor its loop that never stops. Where the division's branch is selected, it fails as it
        # does outside a cond.
        torch, while_loop = self.torch, self.while_loop
        integers = torch.arange(-6, 6).reshape(3, 4)

        def divided(self, a, d):
            return torch.cond(d != 0, lambda u, e: torch.div(u, e, rounding_mode="trunc"),
                              lambda u, e: u + 0, (a, d))

        def selected(self, a, d):
            return torch.cond(d == 0, lambda u, e: torch.div(u, e, rounding_mode="trunc"),
                              lambda u, e: u + 0, (a, d))

        def endless(self, a):
            # The loop's counter stays 0, below its bound 1; it is selected for a NaN entry.
            return torch.cond(a[0, 0].isnan(), lambda u: while_loop(
                lambda i, v: i < 1, lambda i, v: (i * 2, v * 2.0), (torch.tensor(0), u))[1],
                lambda u: u + 1.0, (a,))

        for d in (0, 3, -2):
            with self.subTest(case="divided", d=d):
                self.assert_passes(self.verify(self.module(divided), integers, torch.tensor(d)))
        self.assert_passes(self.verify(self.module(endless), self.x))
        self.assert_unsupported_execution(
            self.verify(self.module(selected), integers, torch.tensor(0), checks=EXECUTE),
            "integer division by zero")

    def test_while_loop_is_a_flow_while_loop(self):
        # while_loop is Lean's Flow.whileLoop, which runs its body while its test holds, at
        # execution, as while_loop's documented Python loop does. The loops count to a literal,
        # to a bound read from an input, so their trip count is known only at run time, run no
        # iteration, read an operand that they do not carry, nest a cond, and halve until an
        # entry is at most 1, a stop on their data. The body's steps are in the flow once,
        # whatever the trip count. A loop that carries a Python number is refused.
        torch, while_loop = self.torch, self.while_loop
        x = self.x

        def counted(self, i, a):
            return while_loop(lambda j, u: j < 4, lambda j, u: (j + 1, u * 2.0 + 1.0), (i, a))

        def bounded(self, n, a):
            return while_loop(lambda j, u: j < n, lambda j, u: (j + 1, u - 1.5),
                              (torch.zeros_like(n), a))

        def closed(self, i, a, b):
            return while_loop(lambda j, u: j < 3, lambda j, u: (j + 1, u + b), (i, a))

        def nested(self, i, a):
            return while_loop(lambda j, u: j < 4, lambda j, u: (j + 1, torch.cond(
                j > 1, lambda w: w * 2.0, lambda w: w + 1.0, (u,))), (i, a))

        def halved(self, a):
            return while_loop(lambda u: u[0, 0] > 1.0, lambda u: (u * 0.5,), (a,))

        cases = [("counted", counted, [(torch.tensor(i), x) for i in (0, 3, 4, 9)]),
                 ("bounded", bounded, [(torch.tensor(n), x) for n in (0, 1, 7)]),
                 ("closed", closed, [(torch.tensor(0), x, x.flip(1))]),
                 ("nested", nested, [(torch.tensor(0), x)]),
                 ("halved", halved, [(torch.arange(1.0, 13.0).reshape(3, 4) * 8,)])]
        for name, forward, argument_sets in cases:
            for arguments in argument_sets:
                with self.subTest(case=name, first=arguments[0].ravel()[:1].tolist()):
                    model = self.module(forward)
                    self.assertIn("while", _flows(self.artifact(model, *arguments)))
                    self.assert_passes(self.verify(model, *arguments))
        document = self.artifact(self.module(counted), torch.tensor(0), x)
        self.assertEqual(_tags(document).count("mul"), 1)

        def literal(self, a):
            return while_loop(lambda j, u: j < 4, lambda j, u: (j + 1, u * 2.0), (0, a))

        self.assert_refused(self.verify(self.module(literal), x),
                            "while_loop: an operand that is not a tensor of the graph")

    def test_while_loop_runs_at_most_its_fuel(self):
        # Each loop runs at most the artifact's fuel of iterations, 65,536 unless the capture
        # names it, each time it is entered: a loop whose test still holds after them fails at
        # execution, such as one that never stops, which the capture does not run. A fuel of
        # another integer type, such as numpy's, is held as an int, and one that is negative, a
        # Boolean or not an integer, such as a tensor of one integer or Boolean, is refused before
        # the capture, also for a module without a flow.
        import numpy as np
        from torchlean._fx import _tensor_value
        from torchlean.export import export_torchlean_graph_json

        torch, while_loop = self.torch, self.while_loop
        x = self.x

        def counted(self, i, a):
            return while_loop(lambda j, u: j < 4, lambda j, u: (j + 1, u * 2.0), (i, a))

        def endless(self, i, a):
            return while_loop(lambda j, u: j < 1, lambda j, u: (j * 2, u * 2.0), (i, a))

        traced = []

        def flat(self, i, a):
            traced.append(a)
            return a * 2.0

        inputs = (torch.tensor(0), x)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "graph.json")
            for fuel, ok in ((3, False), (np.int64(4), True)):
                with self.subTest(fuel=fuel):
                    document = export_torchlean_graph_json(self.module(counted), inputs, path,
                                                           fuel=fuel)
                    self.assertEqual(document["artifact"]["fuel"], fuel)
                    self.assertIs(type(document["artifact"]["fuel"]), int)
                    document["arguments"] = [_tensor_value(value) for value in inputs]
                    report = torchlean.verify(document, checks=EXECUTE, checker=CHECKER)
                    if ok:
                        self.assert_passes(report)
                    else:
                        self.assert_unsupported_execution(report, "while: out of fuel")
            for fuel, error in ((-1, ValueError), (True, TypeError), (1.5, TypeError),
                                (torch.tensor(True), TypeError), (torch.tensor(4), TypeError)):
                for forward in (counted, flat):
                    with self.subTest(fuel=fuel, flow=forward is counted):
                        with self.assertRaisesRegex(error, "fuel must be a nonnegative integer"):
                            export_torchlean_graph_json(self.module(forward), inputs, path,
                                                        fuel=fuel)
        self.assertEqual(traced, [])
        self.assertEqual(self.artifact(self.module(counted), *inputs)["artifact"]["fuel"], 65536)
        self.assert_unsupported_execution(
            self.verify(self.module(endless), *inputs, checks=EXECUTE), "while: out of fuel")

    def test_the_command_line_writes_a_flow(self):
        # `python3 -m torchlean.export` writes the capture document of a module with a torch.cond
        # on its input, whose artifact is a flow, and reports it.
        import subprocess

        source = ("import torch\n\n\nclass Net(torch.nn.Module):\n"
                  "    def forward(self, x):\n"
                  "        return torch.cond(x.sum() > 0, lambda u: u * 2.0, lambda u: u - 1.0,"
                  " (x,))\n")
        with tempfile.TemporaryDirectory() as directory:
            model, path = Path(directory) / "net.py", Path(directory) / "graph.json"
            model.write_text(source, encoding="utf-8")
            run = subprocess.run([sys.executable, "-m", "torchlean.export", str(model), "Net",
                                  str(path), "--example-shape", "3,4", "--require-torch-export"],
                                 capture_output=True, text=True, timeout=600,
                                 env=dict(os.environ, PYTHONPATH=str(ROOT / "python")))
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("flow", run.stdout)
            self.assertEqual(_flows(json.loads(path.read_text(encoding="utf-8"))), ["cond"])

    def test_a_lean_export_compiles(self):
        # `export="Model.lean"` writes a module whose proofs Lean checks when it compiles it:
        # `artifact_decoded` and, for a torch.cond on its input, `programs_checked` of each
        # block's program, or `program_checked` of a capture without a flow. Lake's environment
        # (`lake test`, `lake env`) names Lean and the search path of the built library.
        import subprocess

        lean, search = os.environ.get("LEAN"), os.environ.get("LEAN_PATH")
        if not (lean and search):
            self.skipTest("needs Lake's LEAN and LEAN_PATH (run under lake test or lake env)")
        torch = self.torch
        cases = (("Flow", "programs_checked",
                  lambda x: torch.cond(x.sum() > 0, lambda u: u * 2.0, lambda u: u - 1.0, (x,))),
                 ("Flat", "program_checked", lambda x: x * 3.0 + 1.0))
        with tempfile.TemporaryDirectory() as directory:
            for name, proof, function in cases:
                with self.subTest(name=name):
                    path = Path(directory) / f"{name}.lean"
                    self.assert_passes(torchlean.verify(function, self.x, checks=EXECUTE,
                                                        checker=CHECKER, export=path))
                    self.assertIn(f"\ntheorem {proof} ", path.read_text(encoding="utf-8"))
                    run = subprocess.run([lean, str(path)], capture_output=True, text=True,
                                         timeout=1200, cwd=directory)
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_map_is_a_flow_scan(self):
        # map is Lean's Flow.scan over the leading axis of its mapped operands: the state
        # carries a counter and the stacked outputs, which start at zeros, and iteration i reads
        # entry i of each mapped operand by Op.gather and writes its outputs there by
        # Op.scatter. The maps give two outputs, one of them a reduction, read an operand that
        # they do not map, map integers, run once, and hold a flow in their body: a torch.cond on
        # the sum of the row, which selects each branch, and a while_loop that doubles the row
        # until its sum is 10 or more, 2, 1 and 0 times. PyTorch refuses a map of length zero.
        torch, mapped, while_loop = self.torch, self.map, self.while_loop
        xs = torch.arange(6.0).reshape(3, 2) - 2

        def two(self, a, b, c):
            return mapped(lambda u, v, w: (u * w + v, u.sum()), a, b, c)

        def integers(self, a):
            return mapped(lambda u: u * 3 - 1, a)

        def selected(self, a):
            return mapped(lambda u: torch.cond(u.sum() > 0, lambda v: v * 2.0,
                                               lambda v: v - 1.0, (u,)), a)

        def doubled(self, a):
            return mapped(lambda u: while_loop(lambda i, v: v.sum() < 10,
                                               lambda i, v: (i + 1, v * 2.0),
                                               (torch.tensor(0), u))[1], a)

        # The operands of a map are tensors of their own: torch.export refuses views of one.
        cases = [("two", two, (xs, xs.flip(0), torch.tensor([2.0, -0.5])), ["scan"]),
                 ("integers", integers, (torch.arange(-4, 8).reshape(4, 3),), ["scan"]),
                 ("once", two, (xs[:1].clone(), xs[1:2].clone(), torch.tensor([1.0, 3.0])),
                  ["scan"]),
                 ("selected", selected, (xs,), ["cond", "scan"]),
                 ("doubled", doubled, (xs + 3,), ["scan", "while"])]
        for name, forward, arguments, flows in cases:
            with self.subTest(case=name):
                model = self.module(forward)
                self.assertEqual(_flows(self.artifact(model, *arguments)), flows)
                self.assert_passes(self.verify(model, *arguments))

    def test_a_block_holds_the_constants_it_reads(self):
        # A flow's states leave constants out, and each block that reads one holds it: a weight
        # that both branches of a torch.cond read is in each branch and one that a map reads in
        # its body, and neither is in the block before the flow, which does not read it.
        torch, mapped = self.torch, self.map
        weight = torch.arange(16.0).reshape(4, 4) - 8

        def both(self, a):
            return torch.cond(a.sum() > 0, lambda u: u @ self.weight,
                              lambda u: u @ self.weight - 1.0, (a,))

        def rows(self, a):
            return mapped(lambda u: u @ self.weight, a)

        for name, forward, held in (("cond", both, [False, True, True]),
                                    ("map", rows, [False, True, False])):
            with self.subTest(case=name):
                model = self.module(forward, weight=weight)
                blocks = _blocks(self.artifact(model, self.x)["artifact"]["flow"])
                self.assertEqual([any(step.get("const", {}).get("value") == [4, 4]
                                      for step in block["nodes"]) for block in blocks], held)
                self.assert_passes(self.verify(model, self.x))

    def test_a_map_that_draws_is_unrolled(self):
        # A flow's region is one set of nodes, each drawing from TorchLean's generator under a key
        # of its own, so a map whose body draws, here rand and native_dropout, is unrolled, and
        # row i draws under keys of its own: entry j of a node of key k is the draw of k + j.
        # Dropout in evaluation draws nothing, and its map stays a flow. The body of a while loop
        # draws the same entries in every iteration.
        torch, mapped, while_loop, x = self.torch, self.map, self.while_loop, self.x

        def noised(self, a):
            return mapped(lambda u: u + torch.rand(4), a)

        def dropped(self, a):
            return mapped(lambda u: torch.ops.aten.native_dropout(u, 0.5, True), a)

        def evaluated(self, a):
            return mapped(lambda u: torch.nn.functional.dropout(u, 0.5, training=False), a)

        def looped(self, i, a):
            return while_loop(lambda j, u: j < 3, lambda j, u: (j + 1, u + torch.rand(3, 4)),
                              (i, a))

        def keys(document, tag):
            return [record["fields"][1]["nat"] for record in _records(document)
                    if record["tag"] == tag]

        model = self.module(noised)
        document = self.artifact(model, x)
        self.assertEqual(_flows(document), [])
        drawn = keys(document, "randUniform")
        self.assertEqual(len(set(drawn)), 3, drawn)
        report = self.verify(model, x, checks=EXECUTE)
        self.assert_passes(report)
        noise = torch.stack([_uniform(key, 4, torch.float32) for key in drawn])
        self.assertTrue(torch.equal(_outputs(report)[0], x + noise))

        model = self.module(dropped)
        document = self.artifact(model, x)
        self.assertEqual(_flows(document), [])
        drawn = keys(document, "bernoulliMask")
        self.assertEqual(len(set(drawn)), 3, drawn)
        report = self.verify(model, x, checks=EXECUTE)
        self.assert_passes(report)
        kept, mask = _outputs(report)
        expected = torch.stack([_uniform(key, 4, torch.float32) < 0.5 for key in drawn])
        self.assertTrue(torch.equal(mask, expected))
        self.assertTrue(torch.equal(kept, (x * expected) * 2.0))

        model = self.module(evaluated)
        self.assertEqual(_flows(self.artifact(model, x)), ["scan"])
        report = self.verify(model, x, checks=EXECUTE)
        self.assert_passes(report)
        self.assertTrue(torch.equal(_outputs(report)[0], x))

        model = self.module(looped)
        document = self.artifact(model, torch.tensor(0), x)
        self.assertEqual(_flows(document), ["while"])
        (key,) = keys(document, "randUniform")
        report = self.verify(model, torch.tensor(0), x, checks=EXECUTE)
        self.assert_passes(report)
        noise = _uniform(key, 12, torch.float32).reshape(3, 4)
        self.assertTrue(torch.equal(_outputs(report)[1], x + noise + noise + noise))

    def test_invoke_subgraph_is_inlined(self):
        # PyTorch 2.11's export records a call of a function marked by mark_compile_region as
        # invoke_subgraph, whose graph is inlined at each call, here twice, each with its own
        # operands and the module's parameters; 2.13's export inlines the calls itself. Either
        # way the artifact has no flow, and each call's dropout draws a mask of its own.
        from torch._higher_order_ops.invoke_subgraph import mark_compile_region

        torch = self.torch

        @mark_compile_region
        def scaled(a, weight):
            return (a @ weight) * 3.0

        def forward(self, a):
            return scaled(a, self.weight) + scaled(a * 2.0, self.weight)

        model = self.module(forward, weight=torch.arange(16.0).reshape(4, 4) - 8)
        self.assertEqual(_flows(self.artifact(model, self.x)), [])
        self.assert_passes(self.verify(model, self.x))

        @mark_compile_region
        def dropped(a):
            return torch.ops.aten.native_dropout(a, 0.5, True)

        def twice(self, a):
            return (*dropped(a), *dropped(a))

        model = self.module(twice)
        drawn = [record["fields"][1]["nat"] for record in _records(self.artifact(model, self.x))
                 if record["tag"] == "bernoulliMask"]
        self.assertEqual(len(set(drawn)), 2, drawn)
        report = self.verify(model, self.x, checks=EXECUTE)
        self.assert_passes(report)
        for key, mask in zip(drawn, _outputs(report)[1::2], strict=True):
            self.assertTrue(torch.equal(mask.reshape(-1), _uniform(key, 12, torch.float32) < 0.5))

        @mark_compile_region
        def detached(a):
            with torch.no_grad():
                return a * a

        def with_grad_mode(self, a):
            return detached(a) + a

        # Falling back to FX would drop the nested no_grad and change the input gradient.
        self.assert_refused(self.verify(self.module(with_grad_mode), self.x), "grad-mode")
