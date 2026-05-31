/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common
meta import NN.Examples.Factorization.Common

/-!
# Example: the cyclic Jacobi sweep makes progress (per-rotation off-diagonal decrease)

These checks corroborate the **Tier 2** development in
`NN.Proofs.Tensor.Basic.FactorizationsJacobiDecrease`: the exact per-rotation identity behind Jacobi
convergence. For a symmetric `A`, conjugating by the Givens rotation that annihilates the pivot
`(p, q)` removes exactly `2 · A[p,q]²` of squared off-diagonal mass:

`‖offDiag(Jᵀ A J)‖² = ‖offDiag A‖² − 2 · A[p,q]²`     (`jacobi_off_decrease`)

while preserving the total Frobenius mass `‖A‖²` (`frobSq_orthogonal_conj`).

The checks exhibit both halves of the theorem, *and* its hypotheses biting (negative controls):

* **Positive — exact decrease.** One annihilating rotation drops the off-diagonal mass by precisely
  `2 · A[p,q]²` (independent computations of the two sides agree).
* **Positive — pivot annihilated.** The rotated `A'[p,q]` is `≈ 0` (the defining property of the
  angle; this is the `hannih` hypothesis holding on the concrete matrix).
* **Positive — Frobenius mass preserved.** `‖A'‖² = ‖A‖²`: the orthogonal similarity moves mass from
  the off-diagonal *onto the diagonal* without creating or destroying any.
* **Negative — the angle matters.** A *wrong-angle* (but still orthogonal) Givens rotation fails to
  achieve the `2 · A[p,q]²` decrease: the annihilation hypothesis `hannih` is genuinely needed.
* **Negative — orthogonality matters.** A *non-orthogonal* conjugation (`c² + s² ≠ 1`) does **not**
  preserve `‖A‖²`, so it is not a similarity and the whole argument collapses without
  `givens_orthogonal`.
-/

@[expose] public section


namespace NN.Examples.Factorization.JacobiDecrease

/-- A symmetric `3×3` test matrix; the `(0,1)` pivot is `A[0,1] = 1`, so the predicted off-diagonal
drop from annihilating it is `2 · 1² = 2`. -/
def A : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) :=
  mkMat [[2, 1, 1],
         [1, 3, 1],
         [1, 1, 4]]

/-- The pivot we annihilate. -/
def p : Nat := 0
def q : Nat := 1

/-- `A' = Jᵀ A J` after the annihilating rotation at `(0,1)`. -/
def A' : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) := jacobiRotateAt A p q

/-- Off-diagonal mass before and after the rotation. -/
def offBefore : Float := offDiagFrobSq A
def offAfter : Float := offDiagFrobSq A'

/-- The squared pivot `A[0,1]²` and the predicted post-rotation off-diagonal mass. -/
def pivotSq : Float := let x := Spec.get2 A ⟨p, by decide⟩ ⟨q, by decide⟩; x * x
def offPredicted : Float := offBefore - 2 * pivotSq

#eval IO.println s!"off-diagonal mass: before = {offBefore}, after = {offAfter}, predicted = {offPredicted}"
#eval IO.println s!"pivot A[0,1] = {Spec.get2 A ⟨p, by decide⟩ ⟨q, by decide⟩}, rotated A'[0,1] = {Spec.get2 A' ⟨p, by decide⟩ ⟨q, by decide⟩}"

-- Positive — the exact per-rotation decrease `‖offDiag A'‖² = ‖offDiag A‖² − 2·A[p,q]²`
-- (`jacobi_off_decrease`). The two sides are computed independently and shown to agree.
#eval assertApproxEq "Jacobi(1 rot) off-diagonal decrease = 2·A[p,q]²" offAfter offPredicted

-- Positive — the pivot really is annihilated (the `hannih` hypothesis holds here).
#eval assertLt "Jacobi rotation annihilates the pivot A'[p,q] ≈ 0"
  (Float.abs (Spec.get2 A' ⟨p, by decide⟩ ⟨q, by decide⟩))

-- Positive — total Frobenius mass is preserved (`frobSq_orthogonal_conj`): the orthogonal similarity
-- shifts mass from the off-diagonal onto the diagonal but conserves the total.
#eval assertApproxEq "Jacobi rotation preserves total Frobenius mass ‖A'‖² = ‖A‖²"
  (totalFrobSq A') (totalFrobSq A)

/-! ## Negative control 1: the rotation angle matters

A wrong-angle (but orthogonal, `c² + s² = 1`) Givens rotation does not annihilate the pivot, so the
exact `2 · A[p,q]²` decrease fails. This is the numerical teeth of the `hannih` hypothesis. -/

/-- A fixed orthogonal rotation with the *wrong* angle (`c = 0.6, s = 0.8`, so `c² + s² = 1`). -/
def Awrong : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) := givensConjAt A p q 0.6 0.8
def offWrong : Float := offDiagFrobSq Awrong

#eval IO.println s!"wrong-angle off-diagonal mass = {offWrong} (predicted-if-annihilating = {offPredicted})"

-- The wrong angle misses the predicted decrease by a wide margin.
#eval assertGe "wrong-angle rotation fails the 2·A[p,q]² decrease (annihilation hypothesis needed)"
  (Float.abs (offWrong - offPredicted)) 0.5

/-! ## Negative control 2: orthogonality matters

A non-orthogonal conjugation (`c² + s² ≠ 1`) is not a similarity, so it does **not** preserve the
total Frobenius mass — `frobSq_orthogonal_conj` genuinely needs `givens_orthogonal`. -/

/-- A non-orthogonal "rotation" (`c = 0.6, s = 0.6`, so `c² + s² = 0.72 ≠ 1`). -/
def Askew : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) := givensConjAt A p q 0.6 0.6

#eval IO.println s!"non-orthogonal conj total mass = {totalFrobSq Askew} (original = {totalFrobSq A})"

-- A non-orthogonal conjugation changes the total Frobenius mass.
#eval assertGe "non-orthogonal conjugation breaks Frobenius-mass invariance (orthogonality needed)"
  (Float.abs (totalFrobSq Askew - totalFrobSq A)) 0.5

end NN.Examples.Factorization.JacobiDecrease
