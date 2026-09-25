/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tests.MLTheory.Utils
public import NN.Runtime.Autograd.IRExec

/-!
# LayerNorm Derivative Regressions

Constant rows give exact first and mixed derivatives that can be checked without differentiating
the interval algorithm. A second case compares against finite differences of the IR evaluator.
-/

public section

namespace NN.Tests.MLTheory.CROWNLayerNormDerivatives

open Spec TorchLean
open NN.IR NN.MLTheory.CROWN NN.MLTheory.CROWN.Graph

private def require (condition : Bool) (message : String) : IO Unit :=
  unless condition do throw <| IO.userError s!"LayerNorm derivative: {message}"

private def parameters : LayerNormParams Float :=
  { normalizedShape := [2, 2], gamma := [[1, -2], [0, 3]], beta := [[4, 5], [6, 7]], eps := 1 }

private def point (values : Tensor Float [8]) : FlatBox Float := FlatBox.ofTensor values

private def contains (box : FlatBox Float) (i : Nat) (value : Float) : Bool :=
  getAtOrZero box.lo [i] ≤ value && value ≤ getAtOrZero box.hi [i]

/-- The same directed transfer encloses exact constant-row derivatives in each host format. -/
def checkConstantRows (α : Type) [Storage α] [Context α] [BoundOps α]
    [NonlinearBoundOps α] (label : String) : IO Unit := do
  let affine : LayerNormParams α :=
    { normalizedShape := [2], gamma := [1, 1], beta := [0, 0], eps := 1 }
  let input : Tensor α [4] := [0, 0, 1, 1]
  let direction : Tensor α [4] := [1, -1, 0, 0]
  let zero := FlatBox.ofTensor (Tensor.full (α := α) [4] 0)
  let some (first, second) := layerNormDerivativeBoxes? [2, 2] 1 (some affine)
      (FlatBox.ofTensor input) (FlatBox.ofTensor direction) (FlatBox.ofTensor direction) zero
    | throw <| IO.userError s!"LayerNorm derivative: {label} constant rows rejected"
  for i in [0:4] do
    require (!(decide (getAtOrZero first.lo [i] > getAtOrZero direction [i])) &&
      !(decide (getAtOrZero direction [i] > getAtOrZero first.hi [i])))
      s!"{label} first derivative"
    require (!(decide (getAtOrZero second.lo [i] > 0)) &&
      !(decide (0 > getAtOrZero second.hi [i])))
      s!"{label} second derivative"

def run : IO Unit := do
  checkConstantRows Float "binary64"
  checkConstantRows Float32 "binary32"
  let left : Tensor Float [8] := [1, 0, -2, 3, 0, 0, 0, 0]
  let right : Tensor Float [8] := [0, 2, -1, -2, 0, 0, 0, 0]
  let mixed : Tensor Float [8] := [2, -1, 0, 3, 0, 0, 0, 0]
  let constant : Tensor Float [8] := [0, 0, 0, 0, 5, 5, 5, 5]
  let some (first, second) := layerNormDerivativeBoxes? [2, 2, 2] 1 (some parameters)
      (point constant) (point left) (point right) (point mixed)
    | throw <| IO.userError "LayerNorm derivative: valid constant rows rejected"
  -- At a constant row and epsilon one, J[u] = gamma * center(u), and H[u,v] = 0.
  let expectedFirst : Tensor Float [8] := [0.5, 1, 0, 7.5, 0, 0, 0, 0]
  let expectedSecond : Tensor Float [8] := [1, 4, 0, 6, 0, 0, 0, 0]
  for i in [0:8] do
    require (contains first i (getAtOrZero expectedFirst [i])) "constant-row first differential"
    require (contains second i (getAtOrZero expectedSecond [i])) "constant-row mixed differential"
  for i in [4:8] do
    require (Float.abs (getAtOrZero first.lo [i]) < 0.00000001 &&
      Float.abs (getAtOrZero first.hi [i]) < 0.00000001)
      "a direction in the first row coupled into the second row"
  let zero := point (Tensor.full [8] 0)
  for epsilon in [0, -1] do
    require ((layerNormDerivativeBoxes? [2, 2, 2] 1
      (some { parameters with eps := epsilon }) (point constant) (point left)
      (point right) zero).isNone) "nonpositive epsilon accepted"
  require ((layerNormDerivativeBoxes? [2, 2, 2] 3 (some parameters)
    (point constant) (point left) (point right) zero).isNone) "invalid axis accepted"
  require ((layerNormDerivativeBoxes? (α := Float) [2, 2, 2] 1
    (some { normalizedShape := [4], gamma := [1, 1, 1, 1], beta := [0, 0, 0, 0], eps := 1 })
    (point constant) (point left) (point right) zero).isNone) "wrong affine suffix accepted"
  let empty := FlatBox.ofTensor (Tensor.full (α := Float) [0] 0)
  require ((layerNormDerivativeBoxes? [0, 2, 2] 1 (some parameters)
    empty empty empty empty).isSome) "valid empty batch rejected"
  require ((ibpLayerNormPayloadBox? [0, 2, 2] 1 parameters empty).isSome)
    "valid empty value batch rejected"
  let emptyGraph : NN.IR.Graph := { nodes := #[
    { id := 0, kind := .input, parents := #[], outShape := [0, 2, 2] },
    { id := 1, kind := .layernorm 1, parents := #[0], outShape := [0, 2, 2] }] }
  let emptyInput : Tensor Float [0, 2, 2] := Tensor.full [0, 2, 2] 0
  let emptyPayload : Payload Float := { layerNorm? := fun _ => some parameters }
  require emptyGraph.checkShapes.isOk "empty batch failed shape inference"
  let .ok emptyResult := NN.IR.Graph.denote emptyGraph emptyPayload
      (SomeTensor.ofTensor emptyInput) 1
    | throw <| IO.userError "LayerNorm derivative: empty IR batch rejected"
  require (emptyResult.shape == [0, 2, 2] &&
    (Tensor.to emptyResult.tensor (Array Float)).isEmpty) "empty IR output changed"
  let .ok emptyLowered := Runtime.Autograd.IRExec.lowerToForwardGraph emptyGraph emptyPayload
    | throw <| IO.userError "LayerNorm derivative: empty batch lowering rejected"
  if h : [0, 2, 2] = emptyLowered.inShape then
    let some result := (emptyLowered.denoteAll (Tensor.castShape emptyInput h))[1]?
      | throw <| IO.userError "LayerNorm derivative: empty lowered output missing"
    require (result.shape == [0, 2, 2] && (Tensor.to result.tensor (Array Float)).isEmpty)
      "empty lowered output changed"
  else
    throw <| IO.userError "LayerNorm derivative: empty lowered input shape changed"
  require (!(OpContracts.layerNormMatrixDims 1 [2, 0, 2]).isOk)
    "empty normalized suffix accepted"
  for epsilon in [0, -1] do
    let invalid := { parameters with eps := epsilon }
    require ((layerNormDerivativeBoxes? [0, 2, 2] 1 (some invalid)
      empty empty empty empty).isNone) "empty derivative batch bypassed epsilon validation"
    require ((ibpLayerNormPayloadBox? [0, 2, 2] 1 invalid empty).isNone)
      "empty value batch bypassed epsilon validation"
  let input : Tensor Float [8] := [1, 2, 3, 4, 7, -2, 0, 1]
  let some (first, second) := layerNormDerivativeBoxes? [2, 2, 2] 1 (some parameters)
      (point input) (point left) (point right) zero
    | throw <| IO.userError "LayerNorm derivative: finite rows rejected"
  let graph : NN.IR.Graph := { nodes := #[
    { id := 0, kind := .input, parents := #[], outShape := [2, 2, 2] },
    { id := 1, kind := .layernorm 1, parents := #[0], outShape := [2, 2, 2] }] }
  let payload : Payload Float := { layerNorm? := fun _ => some parameters }
  let store : ParamStore Float :=
    { inputBoxes := ({} : Std.HashMap Nat (FlatBox Float)).insert 0 (point input)
      layerNorm := ({} : Std.HashMap Nat (LayerNormParams Float)).insert 1 parameters }
  let values := runIBP graph store
  let firstPass := runDirectionalDerivative graph store values (point left)
  let rightPass := runDirectionalDerivative graph store values (point right)
  let secondPass := runMixedSecondDerivative graph store values firstPass rightPass
  let some firstGraph := (firstPass[1]?).join
    | throw <| IO.userError "LayerNorm derivative: graph first transfer missing"
  let some secondGraph := (secondPass[1]?).join
    | throw <| IO.userError "LayerNorm derivative: graph mixed transfer missing"
  require (((runMixedSecondDerivative graph store values #[] rightPass)[1]?).join.isNone)
    "missing left derivative table accepted"
  require (((runDirectionalDerivative graph store #[] (point left))[1]?).join.isNone)
    "missing value table accepted"
  let evaluate (u v : Float) : IO (Tensor Float [8]) := do
    let perturbed : Tensor Float [8] := Tensor.ofFn fun i =>
      input.getScalar i + u * left.getScalar i + v * right.getScalar i
    let tensor : Tensor Float [2, 2, 2] := Tensor.reshapeSpec perturbed (by decide)
    let .ok result := NN.IR.Graph.denote graph payload (SomeTensor.ofTensor tensor) 1
      | throw <| IO.userError "LayerNorm derivative: IR evaluation failed"
    if h : result.shape.size = 8 then
      pure (Tensor.reshapeSpec result.tensor (by simpa [Shape.size] using h))
    else throw <| IO.userError "LayerNorm derivative: IR output shape changed"
  let step : Float := 0.001
  let plus ← evaluate step 0
  let minus ← evaluate (-step) 0
  let pp ← evaluate step step
  let pm ← evaluate step (-step)
  let mp ← evaluate (-step) step
  let mm ← evaluate (-step) (-step)
  for i in [0:4] do
    let firstDifference := (getAtOrZero plus [i] - getAtOrZero minus [i]) / (2 * step)
    let mixedDifference := (getAtOrZero pp [i] - getAtOrZero pm [i] -
      getAtOrZero mp [i] + getAtOrZero mm [i]) / (4 * step * step)
    require (contains first i firstDifference) "IR first finite difference escaped"
    require (contains second i mixedDifference) "IR mixed finite difference escaped"
    require (contains firstGraph i firstDifference) "graph first finite difference escaped"
    require (contains secondGraph i mixedDifference) "graph mixed finite difference escaped"
  IO.println "LayerNorm derivative: exact constant rows, row separation and IR differences passed"

end NN.Tests.MLTheory.CROWNLayerNormDerivatives
