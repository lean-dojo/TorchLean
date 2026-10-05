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
  let check : {order : Nat} → Tensor Float [order, 2] → Float → IO Unit :=
      fun directions expected => do
    let actual ← autograd.model.derivative field state input directions
    unless actual[0] == expected do
      throw <| IO.userError s!"field derivative: expected {expected}, got {actual}"
  check (Tensor.zeros [0, 2]) 729
  check [dx] 486
  check [dx, dy] 324
  check [dx, dx, dy] 72
  check [dx, dy, dx, dy] 0
  check [[2, -1], [2, -1]] 54
  let (gradient, inputGradient) ← autograd.model.derivativeVjp field state input
    [dx, dy] (Tensor.ones [1])
  let weights : Tensor Float [1, 2] := gradient.get ⟨0, by decide⟩
  let bias : Tensor Float [1] := gradient.get ⟨1, by decide⟩
  unless weights[0][0] == 198 && weights[0][1] == 180 && bias[0] == 36 &&
      inputGradient[0] == 72 && inputGradient[1] == 108 do
    throw <| IO.userError "mixed derivative pullback: incorrect parameter or input gradient"
  let (thirdGradient, thirdInputGradient) ← autograd.model.derivativeVjp field state input
    [dx, dx, dy] (Tensor.ones [1])
  let thirdWeights : Tensor Float [1, 2] := thirdGradient.get ⟨0, by decide⟩
  let thirdBias : Tensor Float [1] := thirdGradient.get ⟨1, by decide⟩
  unless thirdWeights[0][0] == 72 && thirdWeights[0][1] == 24 && thirdBias[0] == 0 &&
      thirdInputGradient[0] == 0 && thirdInputGradient[1] == 0 do
    throw <| IO.userError "third derivative pullback: incorrect parameter or input gradient"
  let wideState : nn.State (FloatLib.Floats.ExecFloat.Binary 15 112)
      (nn.stateShapes field) := nn.State.full 2
  let wideDerivative ← autograd.model.derivative field wideState [1, 2] [[1, 0], [0, 1]]
  unless FloatLib.Floats.ExecFloat.Binary.toRat? wideDerivative[0] == some 192 do
    throw <| IO.userError "binary128 mixed coordinate derivative"
  IO.println "  neural field derivatives and residual parameter gradients: passed"

end NN.Tests.API.Differential
