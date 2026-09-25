/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tests.MLTheory.Utils

/-!
# Eval-mode BatchNorm interval regressions

Perfect-square running variances give independently computable rational expectations. Both signs
of scale, several channel axes, and empty leading dimensions exercise the directed transfer.
-/

public section

namespace NN.Tests.MLTheory.BatchNormBounds

open Spec TorchLean
open NN.IR NN.MLTheory.CROWN NN.MLTheory.CROWN.Graph
open FloatLib.Floats

private def require (condition : Bool) (message : String) : IO Unit :=
  unless condition do throw <| IO.userError s!"BatchNorm bounds: {message}"

private def checkBackend {α : Type} [Storage α] [Context α] [BoundOps α]
    [NonlinearBoundOps α] (label : String) (decode : α → Option Rat) : IO Unit := do
  let config : BatchNormEvalParams α :=
    { c := 2, gamma := [-2, 3], beta := [Rat.cast (1 / 2 : Rat), -1],
      mean := [1, -2], var := [4, 9], eps := 0 }
  let cases : List (Shape × Nat × Array Rat) :=
    [([2, 2, 2], 1, #[3/2, 1/2, 3, 4, -5/2, -7/2, 7, 8]),
     ([2, 2, 2], 2, #[3/2, 2, -1/2, 4, -5/2, 6, -9/2, 8]),
     ([2, 4], 0, #[3/2, 1/2, -1/2, -3/2, 5, 6, 7, 8]),
     ([0, 2, 2], 1, #[])]
  for (shape, axis, expected) in cases do
    let radius : α := Rat.cast (1 / 4 : Rat)
    let input : FlatBox α :=
      { dim := shape.size
        lo := Tensor.ofFn fun i => (i.val : α) - radius
        hi := Tensor.ofFn fun i => (i.val : α) + radius }
    let graph : NN.IR.Graph := { nodes := #[
      { id := 0, kind := .input, parents := #[], outShape := shape },
      { id := 1, kind := .batchNormEval axis 2, parents := #[0], outShape := shape }] }
    let parameters : ParamStore α :=
      { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0 input
        batchNormEval := Std.HashMap.emptyWithCapacity.insert 1 config }
    let some box := (runIBP graph parameters)[1]?.join
      | throw <| IO.userError s!"{label}: missing bounds for {shape}/{axis}"
    require (box.dim == expected.size) s!"{label}: output dimension"
    for h : i in [0:expected.size] do
      let some lo := decode (getAtOrZero box.lo [i])
        | throw <| IO.userError s!"{label}: nonfinite lower endpoint"
      let some hi := decode (getAtOrZero box.hi [i])
        | throw <| IO.userError s!"{label}: nonfinite upper endpoint"
      require (lo ≤ expected[i] - 1/4 && expected[i] + 1/4 ≤ hi)
        s!"{label}/{shape}/{axis}/{i}: endpoints exclude rational range"
    let wrongNode := { (graph.nodes[1]!) with kind := .batchNormEval axis 3 }
    let wrongChannels := { graph with nodes := graph.nodes.set! 1 wrongNode }
    require ((runIBP wrongChannels parameters)[1]?.join).isNone
      s!"{label}: inconsistent declared channels accepted"
  let point : FlatBox α := FlatBox.ofTensor (Tensor.zeros (α := α) [2])
  require (ibpBatchNormEval? [2] 1 config point).isNone s!"{label}: invalid channel axis"
  require (ibpBatchNormEval? [3] 0 config point).isNone s!"{label}: mismatched shape"
  require (ibpBatchNormEval? [2] 0 { config with var := Tensor.zeros [2] } point).isNone
    s!"{label}: zero denominator accepted"
  require (ibpBatchNormEval? [2] 0 { config with eps := -10 } point).isNone
    s!"{label}: negative stabilized variance accepted"

def run : IO Unit := do
  checkBackend (α := Float) "Float"
    (fun value => ExecFloat.Binary.toRat? (ExecFloat.Binary.ofFloat value))
  checkBackend (α := Float32) "Float32"
    (fun value => ExecFloat.Binary.toRat? (ExecFloat.Binary.ofFloat32 value))
  checkBackend (α := ExecFloat.Binary 8 23) "Binary 8 23" ExecFloat.Binary.toRat?
  IO.println "  directed BatchNorm bounds: passed"

end NN.Tests.MLTheory.BatchNormBounds
