/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.Derivatives
public import NN.Spec.Core.Context.Rational
public import NN.Tensor

/-!
# Softmax Derivative Row Regressions

Exact rational checks exercise the polynomial derivative rules with supplied probability boxes.
They do not evaluate rational exponentials or claim that a finite backend differentiates the ideal
softmax. Separate Float and Float32 checks retain the backend capability boundary.
-/

public section

namespace NN.Tests.MLTheory.SoftmaxDerivatives

open Spec TorchLean
open NN.IR NN.MLTheory.CROWN NN.MLTheory.CROWN.Graph
open scoped Spec.RationalAlgebraic

private instance : BoundOps Rat where
  addDown := (· + ·)
  addUp := (· + ·)
  subDown := (· - ·)
  subUp := (· - ·)
  mulDown := (· * ·)
  mulUp := (· * ·)
  supportsExactAffineReassociation := true

-- Only the algebraic transfer is under test; no transcendental operation is enabled.
private instance : NonlinearBoundOps Rat :=
  { instNonlinearBoundOpsConservative with supportsIdealCoupledDerivatives := true }

private def point (values : Array Rat) : FlatBox Rat :=
  FlatBox.ofTensor (Tensor.from values)

private def graph (shape : Shape) (axis : Nat) : NN.IR.Graph :=
  { nodes := #[
    { id := 0, kind := .input, parents := #[], outShape := shape },
    { id := 1, kind := .softmax axis, parents := #[0], outShape := shape }] }

private def require (condition : Bool) (label : String) : IO Unit :=
  unless condition do throw <| IO.userError s!"softmax derivatives: {label}"

private def expectPoint (label : String) (boxes : Array (Option (FlatBox Rat)))
    (id : Nat) (expected : Array Rat) : IO Unit := do
  let some box := boxes[id]?.join
    | throw <| IO.userError s!"{label}: missing derivative box"
  require (box.dim == expected.size) s!"{label}: dimension"
  require (Tensor.to box.lo (Array Rat) == expected) s!"{label}: lower endpoints"
  require (Tensor.to box.hi (Array Rat) == expected) s!"{label}: upper endpoints"

private def checkRows (label : String) (shape : Shape) (axis : Nat)
    (probabilities left right expectedFirst expectedMixed : Array Rat) : IO Unit := do
  let g := graph shape axis
  let input := point (Array.replicate shape.size 0)
  let ps : ParamStore Rat := { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0 input }
  let ibp := #[some input, some (point probabilities)]
  let dLeft := runDirectionalDerivative g ps ibp (point left)
  let dRight := runDirectionalDerivative g ps ibp (point right)
  expectPoint s!"{label} input seed" dLeft 0 left
  expectPoint s!"{label} first" dLeft 1 expectedFirst
  expectPoint s!"{label} mixed"
    (runMixedSecondDerivative g ps ibp dLeft dRight) 1 expectedMixed
  expectPoint s!"{label} mixed symmetry"
    (runMixedSecondDerivative g ps ibp dRight dLeft) 1 expectedMixed

private def checkIndependentRows : IO Unit := do
  -- For two entries, Dy₀ = p(1-p)(u₀-u₁) and
  -- D²y₀[u,v] = p(1-p)(1-2p)(u₀-u₁)(v₀-v₁); y₁ has the opposite derivative.
  checkRows "last axis" [2, 2] 1
    #[1/4, 3/4, 1/3, 2/3] #[2, -1, 4, 2] #[3, 5, -2, 1]
    #[9/16, -9/16, 4/9, -4/9] #[-9/16, 9/16, -4/9, 4/9]
  checkRows "outer axis" [2, 2] 0
    #[1/4, 1/3, 3/4, 2/3] #[2, 4, -1, 2] #[3, -2, 5, 1]
    #[9/16, 4/9, -9/16, -4/9] #[-9/16, -4/9, 9/16, 4/9]
  checkRows "interior axis" [2, 2, 2] 1
    #[1/4, 1/3, 3/4, 2/3, 1/2, 1/4, 1/2, 3/4]
    #[2, 4, -1, 2, 0, 1, 0, 0] #[3, -2, 5, 1, 2, 0, -3, 1]
    #[9/16, 4/9, -9/16, -4/9, 0, 3/16, 0, -3/16]
    #[-9/16, -4/9, 9/16, 4/9, 0, -3/32, 0, 3/32]
  checkRows "disjoint rows" [2, 2] 1
    #[1/4, 3/4, 1/3, 2/3] #[2, -1, 0, 0] #[0, 0, 4, -2]
    #[9/16, -9/16, 0, 0] #[0, 0, 0, 0]
  checkRows "three-entry rows and constant shift" [2, 3] 1
    #[1/3, 1/3, 1/3, 1/4, 1/4, 1/2] #[1, 0, -1, 2, 2, 2] #[0, 2, -2, 3, -1, 1]
    #[1/3, 0, -1/3, 0, 0, 0] #[-2/9, -2/9, 4/9, 0, 0, 0]
  checkRows "vector" [2] 0 #[1/4, 3/4] #[2, -1] #[3, 5]
    #[9/16, -9/16] #[-9/16, 9/16]
  checkRows "singleton rows" [2, 1, 2] 1
    #[1, 1, 1, 1] #[2, -1, 4, 2] #[3, 5, -2, 1]
    #[0, 0, 0, 0] #[0, 0, 0, 0]
  checkRows "empty normalization axis" [2, 0, 3] 1 #[] #[] #[] #[] #[]
  checkRows "no rows" [0, 2] 1 #[] #[] #[] #[] #[]

private def checkHessianVectorProduct : IO Unit := do
  let base := graph [2, 2] 1
  let g : NN.IR.Graph := { nodes := base.nodes ++ #[
    { id := 2, kind := .flatten [2, 2], parents := #[1], outShape := [4] },
    { id := 3, kind := .linear, parents := #[2], outShape := [1] }] }
  let input := point #[0, 0, 0, 0]
  let ps : ParamStore Rat :=
    { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0 input
      linearWB := Std.HashMap.emptyWithCapacity.insert 3
        { m := 1, n := 4, w := [[2, -1, 3, 4]], b := [0] } }
  let ibp := #[some input, some (point #[1/4, 3/4, 1/3, 2/3]), none, none]
  let directional := runDirectionalDerivative g ps ibp (point #[3, 5, -2, 1])
  let coordinates := fun i : Fin 4 =>
    runDirectionalDerivative g ps ibp
      (point (Array.ofFn fun j : Fin 4 => if i = j then 1 else 0))
  let hvp := runHessianVectorProduct g ps ibp coordinates directional
  let expected : Array Rat := #[-9/16, 9/16, 2/9, -2/9]
  for i in List.finRange 4 do
    expectPoint s!"scalar HVP coordinate {i.val}" (hvp i) 3 #[expected[i.val]!]

/-- A nonlinear parent exercises the `J D²z` term as well as the softmax Hessian. -/
private def checkNonlinearParent : IO Unit := do
  let g : NN.IR.Graph :=
    { nodes := #[
      { id := 0, kind := .input, parents := #[], outShape := [2, 2] },
      { id := 1, kind := .mulElem, parents := #[0, 0], outShape := [2, 2] },
      { id := 2, kind := .softmax 1, parents := #[1], outShape := [2, 2] }] }
  let input := point #[1, 2, 3, 4]
  let ps : ParamStore Rat := { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0 input }
  let ibp := #[some input, some (point #[1, 4, 9, 16]), some (point #[1/4, 3/4, 1/2, 1/2])]
  let left := runDirectionalDerivative g ps ibp (point #[2, -1, 1, 2])
  let right := runDirectionalDerivative g ps ibp (point #[3, 5, -2, 4])
  expectPoint "nonlinear parent first" left 2 #[3/2, -3/2, -5/2, 5/2]
  expectPoint "nonlinear parent mixed"
    (runMixedSecondDerivative g ps ibp left right) 2 #[-51/8, 51/8, -5, 5]

private def checkIntervalDirections : IO Unit := do
  let g := graph [2] 0
  let input := point #[0, 0]
  let ps : ParamStore Rat := { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0 input }
  let probabilities : FlatBox Rat := { dim := 2, lo := [1/4, 1/2], hi := [1/2, 3/4] }
  let seed : FlatBox Rat := { dim := 2, lo := [-1, 0], hi := [2, 3] }
  let ibp := #[some input, some probabilities]
  let left := runDirectionalDerivative g ps ibp seed
  let right := runDirectionalDerivative g ps ibp (point #[3, -2])
  let mixed := runMixedSecondDerivative g ps ibp left right
  let some firstBox := left[1]?.join
    | throw <| IO.userError "interval first derivative missing"
  let some mixedBox := mixed[1]?.join
    | throw <| IO.userError "interval mixed derivative missing"
  for p in (#[1/4, 1/3, 1/2] : Array Rat) do
    for u in (#[-1, 0, 2] : Array Rat) do
      for v in (#[0, 1, 3] : Array Rat) do
        let first := p * (1-p) * (u-v)
        let second := p * (1-p) * (1-2*p) * (u-v) * 5
        for (box, value) in [(firstBox, first), (mixedBox, second)] do
          for i in [0:2] do
            let value := if i = 0 then value else -value
            require (decide (getAtOrZero box.lo [i] ≤ value) &&
              decide (value ≤ getAtOrZero box.hi [i])) "interval sample escaped bounds"

private def checkRejectedInputs : IO Unit := do
  let input := point #[0, 0, 0, 0]
  let ps : ParamStore Rat := { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0 input }
  let g := graph [2, 2] 1
  let ibp := #[some input, some (point #[1/4, 3/4, 1/3, 2/3])]
  let wrongSeed := runDirectionalDerivative g ps ibp (point #[1, 2])
  require (wrongSeed[1]?.join.isNone) "wrong input direction dimension accepted"
  let wrongOutput := #[some input, some (point #[1/2, 1/2])]
  let derivative := runDirectionalDerivative g ps wrongOutput input
  require (derivative[1]?.join.isNone) "wrong probability box dimension accepted"
  let correct := runDirectionalDerivative g ps ibp input
  let wrongDirection := correct.set! 0 (some (point #[1, 2]))
  require ((runMixedSecondDerivative g ps ibp wrongDirection correct)[1]?.join.isNone)
    "wrong left derivative dimension accepted"
  require ((runMixedSecondDerivative g ps ibp correct wrongDirection)[1]?.join.isNone)
    "wrong right derivative dimension accepted"
  for (shape, axis) in [([2, 2], 2), ([], 0)] do
    let invalid := graph shape axis
    require ((runDirectionalDerivative invalid ps ibp input)[1]?.join.isNone)
      "invalid axis accepted"

private def checkFiniteBackend {α : Type} [Storage α] [Context α]
    [BoundOps α] [NonlinearBoundOps α] : IO Unit := do
  require (!NonlinearBoundOps.supportsIdealCoupledDerivatives (α := α))
    "finite backend enabled ideal coupled derivatives"
  let g := graph [2, 2] 0
  let input : FlatBox α := FlatBox.ofTensor (Tensor.full [4] 0)
  let ps : ParamStore α := { inputBoxes := Std.HashMap.emptyWithCapacity.insert 0 input }
  let ibp := runIBP g ps
  require (ibp[1]?.join.isSome) "finite backend lost value bounds"
  let direction := runDirectionalDerivative g ps ibp input
  require (direction[1]?.join.isNone) "finite backend accepted ideal first derivative"
  require ((runMixedSecondDerivative g ps ibp direction direction)[1]?.join.isNone)
    "finite backend accepted ideal mixed second derivative"

def run : IO Unit := do
  checkIndependentRows
  checkHessianVectorProduct
  checkNonlinearParent
  checkIntervalDirections
  checkRejectedInputs
  checkFiniteBackend (α := Float)
  checkFiniteBackend (α := Float32)
  IO.println "softmax derivatives: independent rows, axes, directions, and HVP passed"

end NN.Tests.MLTheory.SoftmaxDerivatives
