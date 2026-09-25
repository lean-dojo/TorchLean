/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Graph.Engine.Base

/-!
# Rowwise LayerNorm Derivative Bounds

Positive epsilon bounds the inverse standard deviation by `1 / sqrt epsilon`. Centered input
and direction bounds then control the first differential and the mixed second differential.
Each row uses its own input box and the stored scale; the fixed bias has zero derivative.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN NN.IR BoundOps

variable {α : Type} [TorchLean.Storage α] [Context α]
variable [BoundOps α] [NonlinearBoundOps α]

/-- Upper bound on absolute value from a pair of interval endpoints. -/
def derivativeMagnitude (bounds : α × α) : α :=
  max2 (subUp 0 bounds.1) bounds.2

/-- Bound each coordinate after subtracting the same row's mean. -/
def centeredDerivativeRadii? {n : Nat} (bounds : Fin n → α × α) :
    Option (Fin n → α) := do
  let _ ← Internal.traverseFin fun i => checkedFiniteBounds? (bounds i)
  let mean ← directedRowMean? bounds
  Internal.traverseFin fun i => do
    let centered ← checkedFiniteBounds?
      (subDown (bounds i).1 mean.2, subUp (bounds i).2 mean.1)
    pure (derivativeMagnitude centered)

/-- Upper bound on the mean of nonnegative coordinate radii. -/
def meanDerivativeRadius? {n : Nat} (radius : Fin n → α) : Option α := do
  let result ← directedRowMean? fun i => (0, radius i)
  pure result.2

/--
Enclose one row's first and mixed-second differentials with fixed affine parameters.

For centered input `z` and centered directions `a`, `b`, `c`, the variance differentials are
`qL = 2 mean(z*a)`, `qR = 2 mean(z*b)`, and `qLR = 2 mean(a*b + z*c)`.
The inverse-square-root differential contributes both `r³ qLR / 2` and
`3 r⁵ qL qR / 4`. Keeping the latter term is essential for mixed directions.
-/
def layerNormDerivativeRow? {n : Nat}
    (input left right mixed : Fin n → α × α) (gamma : Tensor α [n]) (epsilon : α) :
    Option ((Tensor α [n] × Tensor α [n]) × (Tensor α [n] × Tensor α [n])) := do
  let _ ← checkedFiniteBounds? (epsilon, epsilon)
  if !(epsilon > 0) then none else do
    let root ← NonlinearBoundOps.sqrtBounds epsilon epsilon
    let _ ← checkedFiniteBounds? root
    if !(root.1 > 0) then none else do
      let reciprocal ← NonlinearBoundOps.divBounds 1 1 root.1 root.2
      let _ ← checkedFiniteBounds? reciprocal
      let t := reciprocal.2
      let t3 := mulUp (mulUp t t) t
      let t5 := mulUp (mulUp t3 t) t
      let u ← centeredDerivativeRadii? input
      let a ← centeredDerivativeRadii? left
      let b ← centeredDerivativeRadii? right
      let c ← centeredDerivativeRadii? mixed
      let qLeft ← meanDerivativeRadius? fun i => mulUp 2 (mulUp (u i) (a i))
      let qRight ← meanDerivativeRadius? fun i => mulUp 2 (mulUp (u i) (b i))
      let qMixed ← meanDerivativeRadius? fun i =>
        mulUp 2 (addUp (mulUp (a i) (b i)) (mulUp (u i) (c i)))
      let half ← NonlinearBoundOps.divBounds 1 1 2 2
      let threeQuarters ← NonlinearBoundOps.divBounds 3 3 4 4
      let rLeft := mulUp (mulUp half.2 t3) qLeft
      let rRight := mulUp (mulUp half.2 t3) qRight
      let rMixed := addUp (mulUp (mulUp (mulUp threeQuarters.2 t5) qLeft) qRight)
        (mulUp (mulUp half.2 t3) qMixed)
      let radii ← Internal.traverseFin fun i : Fin n => do
        let scale ← checkedFiniteBounds? (gamma.getScalar i, gamma.getScalar i)
        let scaleRadius := derivativeMagnitude scale
        let first := mulUp scaleRadius (addUp (mulUp (a i) t) (mulUp (u i) rLeft))
        let second := mulUp scaleRadius
          (addUp (addUp (mulUp (c i) t) (mulUp (a i) rRight))
            (addUp (mulUp (b i) rLeft) (mulUp (u i) rMixed)))
        let firstBounds ← checkedFiniteBounds? (subDown 0 first, first)
        let secondBounds ← checkedFiniteBounds? (subDown 0 second, second)
        pure (firstBounds, secondBounds)
      pure
        ((Tensor.ofFn fun i => (radii i).1.1, Tensor.ofFn fun i => (radii i).1.2),
         (Tensor.ofFn fun i => (radii i).2.1, Tensor.ofFn fun i => (radii i).2.2))

/-- Restore the matrix view used by the LayerNorm semantics after validating a flat box. -/
def layerNormDerivativeMatrix? (s : Shape) (rows width : Nat) (box : FlatBox α) :
    Option (Tensor α [rows, width] × Tensor α [rows, width]) :=
  if hInput : box.dim = s.size then
    if hMatrix : s.size = Shape.size [rows, width] then
      some (ibpUnflatten box.dim box.lo (hInput.trans hMatrix),
        ibpUnflatten box.dim box.hi (hInput.trans hMatrix))
    else none
  else none

/-- Enclose first and mixed-second LayerNorm differentials over any valid trailing shape. -/
def layerNormDerivativeBoxes? (s : Shape) (axis : Nat)
    (parameters : Option (NN.IR.LayerNormParams α))
    (input left right mixed : FlatBox α) : Option (FlatBox α × FlatBox α) := do
  let (rows, width) ← (OpContracts.layerNormMatrixDims axis s).toOption
  let affine ← (NN.IR.Graph.resolveLayerNormAffine
    { layerNorm? := fun _ => parameters } 0 axis s width).toOption
  let _ ← checkedFiniteBounds? (affine.epsilon, affine.epsilon)
  let _ ← if affine.epsilon > 0 then some () else none
  let _ ← Internal.traverseFin fun i : Fin width => do
    let _ ← checkedFiniteBounds? (affine.gamma.getScalar i, affine.gamma.getScalar i)
    checkedFiniteBounds? (affine.beta.getScalar i, affine.beta.getScalar i)
  let x ← layerNormDerivativeMatrix? s rows width input
  let u ← layerNormDerivativeMatrix? s rows width left
  let v ← layerNormDerivativeMatrix? s rows width right
  let w ← layerNormDerivativeMatrix? s rows width mixed
  let endpoints (box : Tensor α [rows, width] × Tensor α [rows, width]) (i : Fin rows) :=
    fun j : Fin width => ((box.1.unstack i).getScalar j, (box.2.unstack i).getScalar j)
  let bounds ← Internal.traverseFin fun i : Fin rows =>
    layerNormDerivativeRow? (endpoints x i) (endpoints u i) (endpoints v i)
      (endpoints w i) affine.gamma affine.epsilon
  let collect (select : ((Tensor α [width] × Tensor α [width]) ×
      (Tensor α [width] × Tensor α [width])) → Tensor α [width] × Tensor α [width]) :
      FlatBox α :=
    { dim := Shape.size [rows, width]
      lo := Tensor.flattenSpec (Tensor.dim fun i => (select (bounds i)).1)
      hi := Tensor.flattenSpec (Tensor.dim fun i => (select (bounds i)).2) }
  pure (collect Prod.fst, collect Prod.snd)

end NN.MLTheory.CROWN.Graph
