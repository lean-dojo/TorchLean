/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.CROWN.FlatBox.Transfer
public import TorchLean.Verify.CROWN.Op.Exec

/-!
# Rowwise LayerNorm derivative bounds

Positive epsilon bounds the inverse standard deviation by `1 / sqrt epsilon`. Centered input
and direction bounds then control the first differential and the mixed second differential.
Each row uses its own input box and the stored scale, and the fixed bias has zero derivative.
-/

@[expose] public section

namespace TorchLean.CROWN.Graph

open TorchLean.Tensor
open BoundOps

variable {α : Type} [Storage α] [Context α]
variable [BoundOps α] [NonlinearBoundOps α]

/-- Bound each coordinate after subtracting the same row's mean. -/
def centeredRadii? {n : Nat} (bounds : Fin n → α × α) :
    Option (Fin n → α) := do
  let _ ← Tensor.sequenceFinM fun i => finiteBounds? (bounds i)
  let mean ← rowMean? bounds
  Tensor.sequenceFinM fun i => do
    let centered ← finiteBounds?
      (subDown (bounds i).1 mean.2, subUp (bounds i).2 mean.1)
    pure (absBound centered.1 centered.2)

/-- Upper bound on the mean of nonnegative coordinate radii. -/
def meanRadius? {n : Nat} (radius : Fin n → α) : Option α := do
  let result ← rowMean? fun i => (0, radius i)
  pure result.2

/--
Enclose one row's first and mixed-second differentials with fixed affine parameters.

For centered input `z` and centered directions `a`, `b`, `c`, the variance differentials are
`qL = 2 mean(z*a)`, `qR = 2 mean(z*b)`, and `qLR = 2 mean(a*b + z*c)`.
The inverse-square-root differential contributes both `r³ qLR / 2` and
`3 r⁵ qL qR / 4`. Keeping the latter term is essential for mixed directions.

The chain-rule algebra behind these formulas is ours. Each result is an interval `[0 - r, r]`, its
lower end rounded down, so it gives up the derivative's sign. `layerNormDerivRow?_sound` proves that
a successful result on a nonempty row contains both derivatives at a lawful scalar whose literals
`2`, `3` and `4` have those real values, and `layerNormDerivRow?_sound_real` drops the premise on
the literals over `ℝ`.
-/
def layerNormDerivRow? {n : Nat}
    (input left right mixed : Fin n → α × α) (gamma : Tensor α [n]) (epsilon : α) :
    Option ((Tensor α [n] × Tensor α [n]) × (Tensor α [n] × Tensor α [n])) := do
  let _ ← finiteBounds? (epsilon, epsilon)
  if !(epsilon > 0) then none else do
    let root ← NonlinearBoundOps.sqrtBounds epsilon epsilon
    let _ ← finiteBounds? root
    if !(root.1 > 0) then none else do
      let reciprocal ← NonlinearBoundOps.divBounds 1 1 root.1 root.2
      let _ ← finiteBounds? reciprocal
      let t := reciprocal.2
      let t3 := mulUp (mulUp t t) t
      let t5 := mulUp (mulUp t3 t) t
      let u ← centeredRadii? input
      let a ← centeredRadii? left
      let b ← centeredRadii? right
      let c ← centeredRadii? mixed
      let qLeft ← meanRadius? fun i => mulUp 2 (mulUp (u i) (a i))
      let qRight ← meanRadius? fun i => mulUp 2 (mulUp (u i) (b i))
      let qMixed ← meanRadius? fun i =>
        mulUp 2 (addUp (mulUp (a i) (b i)) (mulUp (u i) (c i)))
      let half ← NonlinearBoundOps.divBounds 1 1 2 2
      let threeQuarters ← NonlinearBoundOps.divBounds 3 3 4 4
      let rLeft := mulUp (mulUp half.2 t3) qLeft
      let rRight := mulUp (mulUp half.2 t3) qRight
      let rMixed := addUp (mulUp (mulUp (mulUp threeQuarters.2 t5) qLeft) qRight)
        (mulUp (mulUp half.2 t3) qMixed)
      let radii ← Tensor.sequenceFinM fun i : Fin n => do
        let scale ← finiteBounds? (gamma.getScalar i, gamma.getScalar i)
        let scaleRadius := absBound scale.1 scale.2
        let first := mulUp scaleRadius (addUp (mulUp (a i) t) (mulUp (u i) rLeft))
        let second := mulUp scaleRadius
          (addUp (addUp (mulUp (c i) t) (mulUp (a i) rRight))
            (addUp (mulUp (b i) rLeft) (mulUp (u i) rMixed)))
        let firstBounds ← finiteBounds? (subDown 0 first, first)
        let secondBounds ← finiteBounds? (subDown 0 second, second)
        pure (firstBounds, secondBounds)
      pure
        ((Tensor.ofFn fun i => (radii i).1.1, Tensor.ofFn fun i => (radii i).1.2),
         (Tensor.ofFn fun i => (radii i).2.1, Tensor.ofFn fun i => (radii i).2.2))

end TorchLean.CROWN.Graph
