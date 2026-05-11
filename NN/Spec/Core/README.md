# `NN.Spec.Core`

This directory defines the spec level core of TorchLean: shapes, scalar contexts, and a
shape-indexed tensor type with the minimal set of operations we want to reason about (and, for
executable scalar backends, also run).

The code is organized by theme:

- `context.lean`: the `Context` typeclass for scalar backends (algebraic ops, comparisons, casts).
- `scalar.lean`: small scalar helpers and instances used across the spec layer.
- `shape.lean`: the `Shape` datatype, axis validity, well formedness, and broadcasting evidence (`CanBroadcastTo`).
- `tensor/`: the core tensor datatype and the most basic constructors/accessors and linear algebra primitives.
- `tensor.lean`: umbrella import for the core tensor API (`tensor/Core`, `tensor/Constructors`, `tensor/Linalg`).
- `tensor_ops.lean`: elementwise maps and pointwise tensor operations (no reductions).
- `tensor_reduction_shape.lean`: reductions, reshape/flatten/unflatten, and other shape changing helpers.
- `sequence.lean`: sequence helpers for common time and sequence axis patterns.
- `tensor_array.lean`: array backed representations and helpers used by runtime/backends.
- `tensor_grad.lean`: gradient helpers, such as gradient clipping specs.
- `utils.lean`: small glue utilities (casting maps, `*_like` constructors, list↔tensor helpers, pretty printing).

If you are looking for model and layer APIs, those live under `NN/Spec/Layers` and `NN/Spec/Models`,
and are typically built using the primitives in this directory.
