/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor.Core
public import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Scalar

Spec-only scalar conventions.

These aliases fix their scalar to `ℝ` for mathematical reasoning. Typed tensors and model execution
also support native `Float`/`Float32` and FloatLib configured binary scalars through `Context`.

Because `SpecScalar` is `ℝ`, this is the module that brings in the real dictionary
(`NN.Spec.Core.Context.Real`) and with it the real-analysis hierarchy. Anything that only needs
`Float` or a general `[Context α]` should import `NN.Spec.Core.Context` instead and stay light.

References / context:
- TorchLean paper (overall scalar-polymorphic architecture and trust boundary discussion):
  arXiv:2602.22631.
- IEEE 754-2019 is the reference point for the executable Float32 model (`ExecFloat.Binary 8 23`)
used in the
  runtime/numerics layers (not this file).
-/

@[expose] public section


open TorchLean

namespace Spec

/-- Spec scalars live in `ℝ`. -/
abbrev SpecScalar := ℝ

/-- Spec tensors are Real-typed tensors. -/
abbrev SpecTensor (s : Shape) := Tensor SpecScalar s

end Spec
