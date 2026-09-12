# Deep Dives

This folder contains examples for readers who already know the quickstart path and want to inspect
TorchLean's internal boundaries more carefully. The files cover advanced autograd, float semantics,
GraphSpec lowering, IR shape behavior, widgets, PyTorch export, and graph-based verification.

Build the whole slice with:

```bash
lake build NN.Examples.DeepDives
```

## Files

| File | Command or action | What it shows |
| --- | --- | --- |
| `AutogradTransforms.lean` | `lake exe torchlean autograd_transforms` | Jacobians, Hessians, directional derivatives, and explicit gradient stopping. |
| `TensorOperations.lean` | build as part of `NN.Examples.DeepDives` | Indexing, reshaping, `einsum` (the tensor-pattern syntax needs `open TorchLean.Tensor`), elementwise operations, and tensor widgets. |
| `Floats/Float32Semantics.lean` | `lake exe torchlean float32_semantics` | Native binary32 execution compared with the bit-level IEEE reference. |
| `Floats/ArbIEEEExecCompare.lean` | `lake exe torchlean floats_arb_ieee_compare` | Arb/python-flint interval evidence compared with TorchLean's executable IEEE32 path. |
| `Floats/EffectiveRounding.lean` | `lake build NN.Examples.DeepDives.Floats.EffectiveRounding` | Kernel-checked links between finite bit-level results and rounded-real mantissa/exponent calculations; no CLI. |
| `Floats/GraphNumericalCertificate.lean` | `lake exe torchlean numerical_certificate` | Checks range certificates, tamper rejection, and IEEE replay, including a two-layer MLP; separate exact-real enclosure proofs remain required. |
| `GraphSpec/Tutorial.lean` | `lake exe torchlean graphspec` | Authoring a small graph-style architecture and lowering it into the public trainer path. |
| `IRAxisOps.lean` | `lake exe torchlean ir_axis_ops` | Middle-axis softmax, LayerNorm, and concatenation through the shared checked IR evaluator. |
| `TorchIRPyTorch.lean` | `lake exe torchlean torch_ir_pytorch --arch mlp > exported_model.py` | Emitting readable PyTorch code from a TorchLean IR graph and payload shape. |
| `OneSemanticUniverse.lean` | `lake exe torchlean one_semantic_universe --samples 50` | One IR graph interpreted through execution, interval bounds, and checker-facing semantics. |
| `Widgets.lean` | open in an editor with the Lean infoview | Tensor, graph, Float32, CROWN, autograd, training, and runtime-context widgets. |

## Topics

Each file isolates one boundary:

- `AutogradTransforms.lean` keeps higher-order and directional differentiation out of the first
  autograd tutorial.
- `TensorOperations.lean` continues after the tensor quickstart without creating a second
  “tensor basics” path.
- `Floats/` distinguishes Lean-defined executable IEEE behavior from rounded-real proof models and
  external Arb oracle evidence.
- `GraphSpec/` shows architecture-level graph authoring before runtime execution.
- `IRAxisOps.lean` and `TorchIRPyTorch.lean` show the lower-level IR artifact that exporters,
  widgets, and verifiers inspect.
- `OneSemanticUniverse.lean` ties a single graph to multiple semantic views, making clear which
  objects are executable checks and which objects are Lean statements.
- `Widgets.lean` renders Lean objects for human inspection without making the widget output a proof.

Reusable algorithms should not live here. If a deep dive grows into an API, move the reusable part
under `NN/Spec`, `NN/Runtime`, `NN/IR`, `NN/Verification`, or `NN/MLTheory`, then keep the example
as a small tour of that API.
