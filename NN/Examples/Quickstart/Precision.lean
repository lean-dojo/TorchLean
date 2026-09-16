/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Precision

/-!
# A model with precision beyond binary64

The scalar is FloatLib's configured binary128 value, with 112 stored fraction bits. Both the input
constant and model state are constructed in that type, so no binary64 initializer discards the
additional precision. The same typed graph provides the forward value and input derivative.

Call `NN.Examples.Quickstart.Precision.run` from an application. Its output uses exact rational
decoding rather than a native floating-point display conversion. This is CPU software execution;
changing the scalar format does not create arbitrary-precision CUDA kernels.
-/

@[expose] public section

namespace NN.Examples.Quickstart.Precision

open TorchLean
open FloatLib.Floats

/-- A user-selected format; the upstream type checks the width and bias requirements. -/
abbrev Scalar := ExecFloat.Binary (exponentBits := 15) (fractionBits := 112)

/-- The model is `x ↦ weight*x + bias`, with explicit typed state supplied when it runs. -/
def model : nn.Sequential [1] [1] :=
  nn.build 0 (nn.linear 1 1)

/-- Execute the model and its input derivative without a native floating-point conversion. -/
def run : IO Unit := do
  let exact : Rat := 1 + 1 / (2 ^ 100 : Nat)
  let coefficient : Scalar := Rat.cast exact
  let state : nn.State Scalar (nn.stateShapes model) := nn.State.full coefficient
  let input : Tensor Scalar [1] := Tensor.full [1] 2
  let graph ← nn.lowerToTypedGraph model (α := Scalar)
  let output := nn.TypedGraphModel.forward graph state input
  let derivative := nn.TypedGraphModel.jvp graph state nn.State.zeros input (Tensor.full [1] 1)
  IO.println s!"f(2) = {ExecFloat.Binary.toRat? (output.getScalar ⟨0, by decide⟩)}"
  IO.println s!"f'(2) = {ExecFloat.Binary.toRat? (derivative.getScalar ⟨0, by decide⟩)}"

end NN.Examples.Quickstart.Precision
