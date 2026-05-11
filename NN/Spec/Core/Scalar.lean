/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Context
public import NN.Spec.Core.Tensor.Core

/-!
# Scalar

Spec-only scalar conventions.

The spec layer fixes its scalar to `ℝ` for mathematical reasoning. Runtime scalars
remain `Float`/`NeuralFloat` and are handled separately.

References / context:
- TorchLean paper (overall scalar-polymorphic architecture and trust boundary discussion):
  arXiv:2602.22631.
- IEEE 754-2019 is the reference point for the executable Float32 model (`IEEE32Exec`) used in the
  runtime/numerics layers (not this file).
-/

@[expose] public section


namespace Spec

/-- Spec scalars live in `ℝ`. -/
abbrev SpecScalar := ℝ

/-- Spec tensors are Real-typed tensors. -/
abbrev SpecTensor (s : Shape) := Tensor SpecScalar s

/-- Use the existing `Context` for `SpecScalar` when needed. -/
abbrev SpecContext := Context SpecScalar

end Spec
