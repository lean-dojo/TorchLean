/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Autograd.Differential
public import NN.API.Seeded

/-!
# Neural-field derivative regressions

For `u(x,y) = (a*x+b*y+c)^3`, mixed coordinate derivatives and their parameter gradients have
independent polynomial formulas. Testing the parameter gradients distinguishes trainable
residuals from a coordinate-only derivative demonstration.
-/

@[expose] public section

namespace NN.Tests.API.Differential

open TorchLean

/-- A parameter-free cubic layer, composed with the ordinary affine builder below. -/
def cube : nn.Layer [1] [1] :=
  { kind := "Cube"
    stateShapes := []
    initState := .nil
    requiresGrad := #[]
    forward := fun _ {α} _ _ {m} _ _ => fun input =>
      ((do
        let squared ← Runtime.Autograd.Torch.mul (m := m) (α := α) input input
        Runtime.Autograd.Torch.mul squared input) :
        m (Runtime.Autograd.Model.RefTy (m := m) (α := α) [1])) }

/-- A scalar polynomial field on two coordinates, with three trainable coefficients. -/
def field : nn.Sequential [2] [1] :=
  nn.compose (nn.build 0 (nn.linear 2 1)) cube

/-- Check repeated, mixed, zero-order, and fourth-order derivatives and residual pullbacks. -/
def run : IO Unit := do
  let state : nn.State Float (nn.stateShapes field) :=
    (nn.State.full 1).set ⟨0, by decide⟩ ([[2, 3]] : Tensor Float [1, 2])
  let input : Tensor Float [2] := [1, 2]
  let dx : Tensor Float [2] := [1, 0]
  let dy : Tensor Float [2] := [0, 1]
  for (directions, expected) in
      [([], 729.0), ([dx], 486.0), ([dx, dy], 324.0),
        ([dx, dx, dy], 72.0), ([dx, dy, dx, dy], 0.0),
        ([([2, -1] : Tensor Float [2]), [2, -1]], 54.0)] do
    let actual ← autograd.model.derivative field state input directions
    unless (actual.to (Array Float)) == #[expected] do
      throw <| IO.userError s!"field derivative: expected {expected}, got {actual}"
  let (gradient, inputGradient) ← autograd.model.derivativeVjp field state input
    [dx, dy] (Tensor.ones [1])
  unless ((gradient.get ⟨0, by decide⟩).to (Array Float)) == #[198, 180] &&
      ((gradient.get ⟨1, by decide⟩).to (Array Float)) == #[36] &&
      inputGradient.to (Array Float) == #[72, 108] do
    throw <| IO.userError "mixed derivative pullback: incorrect parameter or input gradient"
  let (thirdGradient, thirdInputGradient) ← autograd.model.derivativeVjp field state input
    [dx, dx, dy] (Tensor.ones [1])
  unless ((thirdGradient.get ⟨0, by decide⟩).to (Array Float)) == #[72, 24] &&
      ((thirdGradient.get ⟨1, by decide⟩).to (Array Float)) == #[0] &&
      thirdInputGradient.to (Array Float) == #[0, 0] do
    throw <| IO.userError "third derivative pullback: incorrect parameter or input gradient"
  let wideState : nn.State (FloatLib.Floats.ExecFloat.Binary 15 112)
      (nn.stateShapes field) := nn.State.full 2
  let wideDerivative ← autograd.model.derivative field wideState [1, 2] [[1, 0], [0, 1]]
  unless (wideDerivative.to (Array (FloatLib.Floats.ExecFloat.Binary 15 112))).map
      FloatLib.Floats.ExecFloat.Binary.toRat? == #[some 192] do
    throw <| IO.userError "binary128 mixed coordinate derivative"
  IO.println "  neural field derivatives and residual parameter gradients: passed"

end NN.Tests.API.Differential
