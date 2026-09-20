# `NN.Spec.Core`

This directory is the spec-level core of TorchLean: shapes, scalar contexts, and shape-indexed
tensors. It is the vocabulary used by layers, losses, graph semantics, runtime approximation
theorems, and many verification statements.

The key idea is that a tensor carries more than a blob of numbers. Its shape, scalar
interpretation, and allowed operations are part of the object being specified.

## Files

- `Context.lean`: the `Context` typeclass for scalar backends: algebraic operations, comparisons,
  casts (including `NatCast`), and the small amount of structure needed by specs; the
  `LawfulContext` class with its instance for `ℝ`; and the `Spec.RationalAlgebraic` helpers.
- `Scalar.lean`: scalar helpers and instances used across the spec layer.
- `Shape.lean`: the `Shape` datatype, runtime axis validation (`validAxis?`), axis-permutation
  planning (`swapAdjacentAxes`), well-formedness, and broadcasting: the decidable proposition
  `CanBroadcastTo` with its typeclass wrapper `BroadcastTo`.
- `Tensor/`: the core tensor datatype plus constructors, rank-one operations, linear algebra, and
  factorizations.
- `Tensor/SomeTensor.lean`: internal runtime shape erasure for evaluators that must store tensors of
  different shapes together. The stored shape is checked before recovering a typed tensor.
- `Tensor.lean`: umbrella import for the core tensor API.
- `TensorOps.lean`: elementwise maps and pointwise tensor operations.
- `TensorReductionShape.lean` and `TensorReductionShape/`: reductions, reshape/flatten/unflatten,
  concat/slice, broadcasting, and shape-changing helpers.
- `Sequence.lean`: helpers for common time and sequence-axis patterns.
- `Tensor/Constructors.lean`: total builders, including checked flat-list and flat-array
  boundaries. `Tensor.full`, `Tensor.zeros`, and `Tensor.ones` are the canonical constant
  constructors.
- `TensorGrad.lean`: gradient-related specs, including clipping helpers.
- `Complex.lean`: small complex-number support used by FFT/FNO-style specifications.
- `Random.lean`: the deterministic `Spec.Random` helpers (`splitmix64`, `keyOf`, `nextSeed`,
  `sampleNat`, uniform sampling, dropout keep bits) shared by the spec layer and the runtime.
- `FloatInstances.lean`: one-way adapters that let the `NN.Floats` numeric types instantiate
  `Context`.

Model and layer APIs live under `NN/Spec/Layers` and `NN/Spec/Models`. They should be built from
these primitives rather than inventing a second tensor language.

## Boundary

This folder defines meanings, not fast kernels. Runtime code may execute the same operations over
`Float`, FloatLib’s `ExecFloat.Binary 8 23`, CUDA buffers, or external providers, but a theorem
should say which spec object that runtime path is meant to approximate or preserve.
