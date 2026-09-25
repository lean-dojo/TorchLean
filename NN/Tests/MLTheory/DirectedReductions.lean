/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Tests.MLTheory.Utils

/-!
# Directed reduction, average-pooling, and input-region regressions

Exact rational expectations expose endpoint rounding that a comparison in the tested scalar
would lose. The graph checks cover positive and negative sums, means, different axes, and empty
outer dimensions. Pooling checks include padded cells in the divisor and preserve spatial windows
and leading dimensions. Input boxes include the exact center plus and minus the stored radius.
-/

public section

namespace NN.Tests.MLTheory.DirectedReductions

open Spec TorchLean
open NN.IR NN.MLTheory.CROWN NN.MLTheory.CROWN.Graph
open FloatLib.Floats

private def require (condition : Bool) (message : String) : IO Unit :=
  unless condition do throw <| IO.userError s!"directed reductions: {message}"

private def checkBox {α : Type} [Storage α] [Context α]
    (decode : α → Option Rat) (label : String) (box : FlatBox α)
    (expected : Array Rat) : IO Unit := do
  require (box.dim == expected.size) s!"{label}: output dimension"
  for h : i in [0:expected.size] do
    let some lo := decode (getAtOrZero box.lo [i])
      | throw <| IO.userError s!"{label}: nonfinite lower endpoint"
    let some hi := decode (getAtOrZero box.hi [i])
      | throw <| IO.userError s!"{label}: nonfinite upper endpoint"
    require (lo ≤ expected[i] && expected[i] ≤ hi)
      s!"{label}[{i}]: [{lo}, {hi}] excludes {expected[i]}"

private def poolBox? {α : Type} [Storage α] [Context α] [BoundOps α]
    [NonlinearBoundOps α] (config : WindowConfig) (shape outShape : Shape)
    (input : FlatBox α) : Option (FlatBox α) :=
  let graph : NN.IR.Graph := { nodes := #[
    { id := 0, kind := .input, parents := #[], outShape := shape },
    { id := 1, kind := .avgPool config, parents := #[0], outShape := outShape }] }
  let parameters : ParamStore α :=
    { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0 input }
  (runIBP graph parameters)[1]?.join

private def checkPooling {α : Type} [Storage α] [Context α] [BoundOps α]
    [NonlinearBoundOps α] (label : String) (decode : α → Option Rat)
    (gap : Rat) : IO Unit := do
  let small : α := Rat.cast gap
  let window : WindowConfig :=
    { spatialRank := 1, kernel := [2], stride := [1], padding := [0] }
  let plane : WindowConfig :=
    { spatialRank := 2, kernel := [2, 2], stride := [1, 1], padding := [0, 0] }
  let volume : WindowConfig :=
    { spatialRank := 3, kernel := [2, 1, 2], stride := [1, 1, 1], padding := [0, 0, 0] }
  let mean := (1 + gap) / 2
  let cases : List (String × WindowConfig × Shape × Shape × Array α × Array Rat) :=
    [("point", window, [2], [1], #[1, small], #[mean]),
     ("negative", window, [2], [1], #[-1, -small], #[-mean]),
     ("padding", { window with padding := [1] }, [2], [3],
       #[-1, -small], #[-1 / 2, -mean, -gap / 2]),
     ("thirds", { window with kernel := [3], padding := [1] }, [2], [2],
       #[1, small], #[(1 + gap) / 3, (1 + gap) / 3]),
     ("spatial2", plane, [2, 2], [1, 1], #[1, small, 1, small], #[mean]),
     ("strided2", { plane with stride := [1, 2], padding := [0, 1] }, [2, 3], [1, 2],
       #[1, small, 3, -1, -small, 5], #[0, 2]),
     ("leading2", plane, [2, 1, 2, 2], [2, 1, 1, 1],
       #[1, small, 1, small, -1, -small, -1, -small], #[mean, -mean]),
     ("spatial3", volume, [2, 1, 2], [1, 1, 1], #[1, small, 1, small], #[mean]),
     ("empty-leading", window, [0, 2], [0, 1], #[], #[])]
  for (name, config, shape, outShape, values, expected) in cases do
    let point := FlatBox.ofTensor (Tensor.from values)
    let some box := poolBox? config shape outShape point
      | throw <| IO.userError s!"{label}/pool/{name}: missing bounds"
    checkBox decode s!"{label}/pool/{name}" box expected
  let interval : FlatBox α :=
    { dim := 2, lo := [1, small], hi := [2, small + small] }
  let some enclosure := poolBox? window [2] [1] interval
    | throw <| IO.userError s!"{label}/pool/interval: missing bounds"
  checkBox decode s!"{label}/pool/interval lower" enclosure #[mean]
  checkBox decode s!"{label}/pool/interval upper" enclosure #[1 + gap]
  let point := FlatBox.ofTensor (Tensor.from (#[1, small] : Array α))
  let rankZero : WindowConfig :=
    { spatialRank := 0
      kernel := Tensor.from (#[] : Array Nat)
      stride := Tensor.from (#[] : Array Nat)
      padding := Tensor.from (#[] : Array Nat) }
  let invalid : List (WindowConfig × Shape × Shape) :=
    [({ window with kernel := [0] }, [2], [1]),
     ({ window with stride := [0] }, [2], [1]),
     (rankZero, [2], [2]), (plane, [2], [1]),
     (window, [3], [2]), (window, [2], [1, 1])]
  for (config, shape, outShape) in invalid do
    require (poolBox? config shape outShape point).isNone
      s!"{label}/pool: invalid geometry or input dimension accepted"

private def checkBackend {α : Type} [Storage α] [Context α] [BoundOps α]
    [NonlinearBoundOps α] (label : String) (decode : α → Option Rat)
    (gap : Rat) : IO Unit := do
  let small : α := Rat.cast gap
  require (decode small == some gap) s!"{label}: the test gap must be exactly representable"
  let values : Tensor α [4] := [1, small, -1, -small]
  let point := FlatBox.ofTensor values
  let cases : List (Shape × Nat × Array Rat) :=
    [([2, 2], 1, #[1 + gap, -1 - gap]),
     ([2, 2], 0, #[0, 0]),
     ([2, 1, 2], 2, #[1 + gap, -1 - gap]),
     ([2, 1, 2], 0, #[0, 0])]
  for (shape, axis, sums) in cases do
    let graph : NN.IR.Graph := { nodes := #[
      { id := 0, kind := .input, parents := #[], outShape := shape },
      { id := 1, kind := .reduceSum axis, parents := #[0],
        outShape := Tensor.shapeAfterSum shape axis },
      { id := 2, kind := .reduceMean axis, parents := #[0],
        outShape := Tensor.shapeAfterSum shape axis }] }
    let parameters : ParamStore α :=
      { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0 point }
    let boxes := runIBP graph parameters
    let some sum := boxes[1]?.join
      | throw <| IO.userError s!"{label}: missing sum bounds"
    let some mean := boxes[2]?.join
      | throw <| IO.userError s!"{label}: missing mean bounds"
    checkBox decode s!"{label}/sum/{shape}/{axis}" sum sums
    checkBox decode s!"{label}/mean/{shape}/{axis}" mean (sums.map (· / 2))
  let center : Tensor α [1] := [1]
  let region := FlatBox.lInfBall center small
  checkBox decode s!"{label}/input lower edge" region #[1 - gap]
  checkBox decode s!"{label}/input upper edge" region #[1 + gap]
  let allPositive : FlatBox α := FlatBox.ofTensor (Tensor.from (#[1, small] : Array α))
  let some mean := boxMean? allPositive
    | throw <| IO.userError s!"{label}: missing flat mean bounds"
  checkBox decode s!"{label}/flat mean" mean #[(1 + gap) / 2]
  let empty := FlatBox.ofTensor (Tensor.zeros (α := α) [0])
  let some emptySum := ibpReduceSumAxis 1 empty [0, 2]
    | throw <| IO.userError s!"{label}: empty outer sum rejected"
  let some emptyMean := ibpReduceMeanAxis 1 empty [0, 2]
    | throw <| IO.userError s!"{label}: empty outer mean rejected"
  require (emptySum.dim == 0 && emptyMean.dim == 0) s!"{label}: empty outer dimensions"
  require ((ibpReduceSumAxis 2 point [2, 2]).isNone &&
      (ibpReduceMeanAxis 2 point [2, 2]).isNone) s!"{label}: invalid axis accepted"
  require ((ibpReduceSumAxis 0 point [3]).isNone &&
      (ibpReduceMeanAxis 0 point [3]).isNone) s!"{label}: mismatched input shape accepted"
  require ((ibpReduceMeanAxis 0 empty [0]).isNone && (boxMean? empty).isNone)
    s!"{label}: empty mean accepted"
  checkPooling label decode gap

def run : IO Unit := do
  checkBackend (α := Float) "Float"
    (fun value => ExecFloat.Binary.toRat? (ExecFloat.Binary.ofFloat value))
    (1 / (2 ^ 55 : Nat))
  checkBackend (α := Float32) "Float32"
    (fun value => ExecFloat.Binary.toRat? (ExecFloat.Binary.ofFloat32 value))
    (1 / (2 ^ 25 : Nat))
  checkBackend (α := ExecFloat.Binary 8 23) "FloatLib binary32"
    ExecFloat.Binary.toRat? (1 / (2 ^ 25 : Nat))
  IO.println "  directed average pooling: passed"
  IO.println "  directed reductions and input regions: passed"

end NN.Tests.MLTheory.DirectedReductions
