/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Floats.IEEEExec.Exec32
public import NN.Floats.NeuralFloat.Core
public import NN.Spec.Core.Tensor.Core

/-!
# Scalar

Runtime scalar conventions.

Runtime execution stays in `IEEE32Exec`/`Float`/`NeuralFloat`; each backend has a different proof
status relative to the spec layer.

Note on trust boundaries:
- Lean's `Float` is an *implementation* type; this repo does not prove that `Float` execution
  matches an IEEE-754 spec. Claims connecting `Float` execution to spec-level `ℝ` therefore cross a
  trusted runtime interface.
- `IEEE32Exec` is an executable bit-level IEEE-754 binary32 model; connecting it to Lean/runtime
  hardware float32 is out of scope (treat that bridge as trusted).
- For proof-relevant numeric execution, use the rounding model backends (`NeuralFloat` / `NF`),
  where per-op error bounds can be stated and composed.
- `NeuralFloat`/`NF` are formal *models* implemented in Lean. Relating them to real hardware
  floating-point (or Lean's `Float`) is a separate backend-correlation assumption, not a theorem in
  this module.
-/

@[expose] public section


namespace Runtime

/-- Default runtime scalar for execution. -/
abbrev RuntimeScalar := TorchLean.Floats.IEEE754.IEEE32Exec

/-- Runtime tensors are Float-typed tensors. -/
abbrev RuntimeTensor (s : Spec.Shape) := Spec.Tensor RuntimeScalar s

/-- NeuralFloat runtime scalar (precision-aware). -/
abbrev RuntimeNeuralScalar (β : TorchLean.Floats.NeuralRadix) := TorchLean.Floats.NeuralFloat β

/-- Runtime tensors backed by NeuralFloat. -/
abbrev RuntimeNeuralTensor (β : TorchLean.Floats.NeuralRadix) (s : Spec.Shape) :=
  Spec.Tensor (RuntimeNeuralScalar β) s

end Runtime
