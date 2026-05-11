# `NN/Spec`: specification layer (reference definitions)

This folder is TorchLean's specification layer. It holds the reference definitions of tensors, ops,
layers, and models that we reuse for:

- execution (instantiating scalars as `Float`, IEEE-754 models, etc.),
- verification/bounds (instantiating scalars as `Interval`, etc.),
- proofs (instantiating scalars as `ℝ`).

The practical goal is to avoid a gap between the network we run and the network we reason about:
define it once, then reuse the same definition for execution and verification.

This layout mirrors the TorchLean paper (`arXiv:2602.22631`): we separate the focused
`NN.Spec.*` modules, runtime entrypoints, and verification entrypoints, but we keep them aligned by
building everything on top of the same core semantics. The public spec doorway is
`NN.Entrypoint.Spec` (or `import NN` / `import NN.Library` for downstream users).

## How to navigate

- `Core/`
  - `shape.lean`: type-level tensor shapes + broadcast/axis utilities.
  - `tensor/Core.lean`: the `Spec.Tensor` datatype (a pure tensor representation).
  - `tensor_ops.lean`, `tensor_reduction_shape.lean`: elementwise ops + reductions.
  - `context.lean`: `Context α`, the numeric backend interface (arithmetic, order, exp/tanh/etc.).
- `Layers/`: forward + backward specs for common NN layers (linear/conv/attention/etc.).
- `Autograd/`: spec level reverse mode building blocks (`OpSpec`) used by runtime AD wrappers.
- `Module/`: thin wrappers that package layer specs as modules with export metadata.
- `Models/`: model compositions (MLP/CNN/Transformer/ResNet/Seq2Seq/etc.).
- `NN/Examples/`: executable examples that exercise the specs.

## Terminology

- spec = pure reference definitions (this folder).
- runtime = tape/graph execution, compilation, and training loops (see `NN/Runtime/*` and
  `NN.Entrypoint.Runtime`).
- verification = sound bound propagation / certificate checking (see `NN/Verification/*` and
  `NN.Entrypoint.Verification`).
