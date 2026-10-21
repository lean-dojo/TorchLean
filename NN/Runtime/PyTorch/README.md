# PyTorch Interop (Runtime)

This folder contains TorchLean's bridge layer to and from PyTorch.

We use PyTorch for two practical reasons:

1. Execution and training: PyTorch is a practical runtime for trying models, training on real data,
   and checking that shapes and numerics behave as expected.
2. Moving artifacts: many workflows already produce PyTorch checkpoints or `state_dict`s. This
   bridge gives those parameters a path back into Lean for verification and proofs.

This is an interop layer, not a second semantics for TorchLean. PyTorch may produce code, weights,
graphs, or JSON artifacts, but TorchLean still parses the artifact, checks the shapes and supported
operators, and connects the result to TorchLean IR/spec objects. PyTorch autograd is not part of the
trusted proof boundary.

These files move artifacts between PyTorch and Lean; they are separate from the optional LibTorch
execution provider described in the
[backend chapter](https://lean-dojo.github.io/TorchLean/blueprint/Runtime___-Autograd___-and-Interop/Inside-The-Backend-Planner/).

## Export

`Export/` contains reusable exporters and adapters.

- `Export/Core.lean` holds shared string utilities, import/header rendering, and Python
  `state_dict` save/load helpers. Callers supply any checkpoint-to-JSON conversion.
- `Export/IRPyTorch.lean` is the general model code path: it exports an `NN.IR.Graph` plus a
  parameter store into a standalone PyTorch module.
- `Export/ONNX.lean` emits a conservative Python adapter that reads an ONNX graph and writes the
  same `torchlean.ir.v2` JSON artifact used by the graph importer. It includes static-shape
  lowerings for common tensor ops plus Conv/Gemm/BatchNorm graph structure where the current IR can
  represent it. Use `Export.PyTorch.ONNX.script` with `ONNX.Options` to generate the adapter.
- `Export/TorchExport.lean` emits a Python adapter that captures a PyTorch `nn.Module` with
  `torch.export`/FX and writes TorchLean IR JSON for the supported op subset. The script is
  emitted by `Export.PyTorch.TorchExport.script`, configured with `TorchExport.Options`.
  Python-generation sections live under `TorchExport.Internal`.

## Semantic operators and wire strings

`NN/IR/Operator.lean` defines `OpKind`, its static attributes, and the constructor identities in
`NN.IR.OpTag`. For example, `(.softmax 1).opTag` is `.softmax`; its axis belongs to `OpKind`.
`OpTag.metadata` gives the parent count and diagnostic name without constructing a dummy operator.
`OpTag.toKind?` reconstructs operations whose tags supply all their static information.

`Wire.lean` assigns each tag its fixed spelling in `torchlean.ir.v2`. Both Python-emitting adapters
write through `Wire.tag`, and `Import/TorchExport.lean` reads through `Wire.parseTag?`. Pointwise
operators are written by operator instead (`Wire.pointwiseTag`, with an attribute field such as
`rounding_mode`) and read with `Wire.parsePointwise?`; no pointwise spelling collides with a
constructor's. The wire strings have their own table, so changing a diagnostic name does not change
existing files. The round trips are proved in Lean:

```lean
theorem Wire.parse_op_tag (tag : OpTag) :
    Wire.parseTag? (Wire.tag tag) = some tag
theorem Wire.parse_op_kind (kind : OpKind) (h : kind.opTag.hasAttributes = false) :
    Wire.parse? (Wire.tag kind.opTag) = some kind
```

Operators with axes, shapes, or convolution geometry parse their tag first and then read those
attributes. Tensor parameters, including the weights of `.linear`, come from the payload store.
Value-graph markers (`tuple_getitem`, `multihead_attention`) and the format
marker also live in `Wire`.

The PyTorch adapter matches complete ATen overload names and explicit FX callables or methods.
For example, `aten.log.default` lowers to `log`; `log1p`, `log2`, and `log_softmax` need their own
lowerings and are rejected. `aten.slice.Tensor` with step one lowers to `slice`. The
mixture-of-experts routing idiom `torch.zeros_like(p).scatter(-1, torch.topk(p, k).indices, 1)`
lowers to one `top_k_mask` node; the `topk` call and the zeros it scatters into must have no other
users. For partial selection (`0 < k < width`), the artifact's `capture_constraints` records the
assumption that each score row is finite and its `k`-th largest score is strictly above its
`(k+1)`-th. Without that gap, PyTorch may select different tied experts from the IR's lower-index
rule. Selecting zero or all entries needs no gap. The check report preserves these constraints
with `capture_constraints_status="not_checked"`; a successful structure, type or sample check
does not prove them for other inputs.

Mutating overloads, `view(dtype)` reinterpretation, `dtype` arguments of reductions and
softmax, and unsupported tuple producers are rejected before the adapter writes an artifact.

The core ATen pointwise operators `neg`, `rsqrt`, `erf` and `gelu`, every core overload of `div` and
`pow`, and `clamp_min` with a scalar bound become IR pointwise nodes in the arithmetic of the
vectorized loops of PyTorch's CPU kernels. Binary16 and bfloat16 inputs are widened to binary32
where those loops compute in binary32. Reduced-precision division and multiplication do so only for
a one-element right operand, and `pow` with a scalar exponent follows PyTorch's dispatch on the
exponent rounded to the tensor's dtype. Conversions follow `c10::convert`, which makes binary16 and
bfloat16 values from binary32, so an integer, Boolean or binary64 source becomes two IR casts. A
Python scalar operand, of these operators or of `add`, `sub` and `mul`, and the bound of `clamp_min`
become constants converted from the int64, uint64, binary64 or Boolean value PyTorch wraps them in.
An integer beyond `2^53` therefore rounds once to binary32 or binary64, and twice, through binary32,
to bfloat16 or binary16. The bound of `clamp_min` is converted to the operation's result type, so an
int32 tensor clamped at `0.5` gives binary32. Like PyTorch's kernel, `clamp_min` returns the bound
where the tensor is below it and keeps the tensor's value elsewhere, so `clamp_min(-0.0, 0)` is
`-0.0`, and a NaN bound gives NaNs. The IR's `relu` returns `+0.0` at `-0.0`, where eager PyTorch
returns `-0.0`. A scalar on the left of an explicit `torch.mul(s, x)` is converted to the tensor's
dtype, while Python's `s * x` is `x * s`. The tanh GELU follows PyTorch's CPU kernels at their AVX2
and AVX-512 capability levels, which fuse `x + κ x³` on x86-64. At the DEFAULT level, which
`ATEN_CPU_CAPABILITY=default` selects, PyTorch rounds that sum twice, and other processors and CUDA
were not measured. The [IR guide](../../IR/README.md#frontend-lowering) lists the rules and the
elements that PyTorch's scalar loop computes differently.

## Import

`Import/` parses JSON encoded weights and graphs into TorchLean artifacts.

The bridge uses JSON because Python can write it directly and Lean can parse it without depending on
Python pickle formats.

- `Import/Core.lean` defines `parseTensor`, which turns nested JSON arrays into a `Tensor Float s` when the JSON shape matches `s`.
  Lookups return `Option`, so a missing key or a shape that does not match is a `none` the caller has to
  handle rather than a panic.
- `Import/CrownParamstore.lean` bridges loaded tensors into the graph backend's `ParamStore` when a workflow needs node id keyed parameters.
- `Import/TorchExport.lean` parses TorchLean IR JSON from the generated graph capture adapter and
  accepts only graphs that pass the shared IR validators. All array access is bounds checked and
  reports an `Except` error naming the node; nothing panics on a malformed artifact.

## Model-family adapters and examples

`Import/{MLP,CNN,Transformer}.lean` loads the supported state-dict conventions into typed parameter
records. The corresponding `Export/` modules generate Python classes and embedded parameters.
These reusable adapters are available as `Import.PyTorch.MLP`, `Export.PyTorch.MLP`, and the
matching CNN/Transformer namespaces.

Runnable examples and small reference artifacts live under `NN/Examples/Interop/PyTorch`.
The [graph-import guide](https://lean-dojo.github.io/TorchLean/examples/verify/) describes
capture, validation, and reference comparisons through the shared IR.

## What Users Can Do Today

- Supply PyTorch weights as nested-list JSON, optionally wrapped in a `"params"` object.
- Parse those JSON tensors in Lean with exact shape checks.
- Capture supported PyTorch `nn.Module` graphs as TorchLean IR JSON and validate them in Lean.
- Lower a conservative ONNX static graph fragment into the same TorchLean IR JSON path. Graph
  validation and payload loading stay separate: imported Conv/Gemm/BatchNorm structure can be
  checked as IR, while execution needs the corresponding payload store.
- Reuse the Lean IR semantics after import. Elementwise ops, reshape/flatten/broadcast/sum, direct
  leading-axis concat plus generic concat through the shared evaluator, axis reductions, axis
  permutation, supported transpose forms, matrix and batched matrix multiplication, arbitrary-axis
  softmax, and eval-mode channel-first BatchNorm have theorem-level IR evaluator bridge facts.
  Payload-backed `linear`, arbitrary-rank no-dilation convolution, payload-backed constants, and
  eval-mode channel-first BatchNorm are also covered at the actual one-step `Graph.evalAt` path as
  well as at their helper evaluators. Arbitrary-rank pooling has the same local bridge to its spec
  operations. LayerNorm is covered through the IR evaluator's rank-independent matrix view, and graph-structural nodes
  such as input, detach, and scalar MSE are covered as well. The
  `NN.Verification.Builtin.Proved.Correctness.Eval` import collects the corresponding concrete
  theorem modules; those imported theorems are the current record of evaluator support.
- Load verification-owned PINN/FNO checkpoints through their `NN/Verification` adapters.
- Emit readable PyTorch code from a TorchLean `NN.IR.Graph` and `ParamStore`.

## Reference (PyTorch)

- PyTorch "Saving and Loading Models" / `state_dict`: https://pytorch.org/tutorials/beginner/saving_loading_models.html
- PyTorch `torch.export`: https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html
- PyTorch FX: https://docs.pytorch.org/docs/stable/fx.html
