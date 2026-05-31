/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Examples.Factorization.Common
meta import NN.Examples.Factorization.Common

/-!
# Example: the classical Jacobi sweep contracts at a fixed linear rate

These checks corroborate the **Tier 3** development in
`NN.Proofs.Tensor.Basic.FactorizationsJacobiRate`: the *aggregate* convergence rate of the classical
(largest-pivot) Jacobi strategy. Annihilating the **largest** off-diagonal pivot multiplies the
off-diagonal mass by at most `1 − 2/(n² − n) < 1`:

`‖offDiag(Jᵀ A J)‖² ≤ (1 − 2/(n² − n)) · ‖offDiag A‖²`     (`jacobi_off_decrease_classical`)

because the largest pivot carries at least the average share `‖offDiag A‖²/(n² − n)` of the mass
(`offSq_le_count_mul_max`). The test matrix has one dominant off-diagonal entry, so the contrast
between annihilating it and annihilating a tiny one is stark.

The checks exhibit the theorem *and* its largest-pivot hypothesis biting (negative control):

* **Positive — pivot carries ≥ average share.** `‖offDiag A‖² ≤ (n² − n) · A[p,q]²` for the largest
  pivot (`offSq_le_count_mul_max` on the concrete matrix).
* **Positive — largest pivot meets the rate.** Annihilating the dominant entry `A[0,1]` contracts the
  off-diagonal mass below `(1 − 2/(n² − n)) · ‖offDiag A‖²` (in fact far below — it nearly diagonalises).
* **Negative — a non-largest pivot misses the rate.** Annihilating a *tiny* off-diagonal entry
  `A[0,2]` still removes `2·A[0,2]²` of mass (the per-rotation identity always holds), but that is far
  too little to meet the guaranteed factor: the off-diagonal mass stays *above* `(1 − 2/(n²−n))·‖offDiag A‖²`.
  This is exactly why the rate is for the *largest-pivot* strategy and the cyclic sweep needs the
  research-grade Forsythe–Henrici bound instead.
-/

@[expose] public section


namespace NN.Examples.Factorization.JacobiRate

/-- A symmetric `3×3` matrix with one dominant off-diagonal entry `A[0,1] = 5` and two tiny ones
`A[0,2] = A[1,2] = 0.1`. -/
def A : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) :=
  mkMat [[1, 5, 0.1],
         [5, 2, 0.1],
         [0.1, 0.1, 3]]

/-- Off-diagonal count `n² − n` for `n = 3`, and the guaranteed contraction factor `1 − 2/(n²−n)`. -/
def offCount : Float := 3 * 3 - 3          -- = 6
def factor : Float := 1 - 2 / offCount     -- = 2/3

def offBefore : Float := offDiagFrobSq A

/-- The largest off-diagonal entry is `A[0,1] = 5`; its square is the per-rotation drop budget. -/
def bigSq : Float := let x := Spec.get2 A ⟨0, by decide⟩ ⟨1, by decide⟩; x * x   -- = 25
/-- A tiny off-diagonal entry `A[0,2] = 0.1`. -/
def smallSq : Float := let x := Spec.get2 A ⟨0, by decide⟩ ⟨2, by decide⟩; x * x -- = 0.01

/-- Annihilate the **largest** pivot `(0,1)` — the classical choice. -/
def Abig : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) := jacobiRotateAt A 0 1
def offBig : Float := offDiagFrobSq Abig

/-- Annihilate a **tiny** pivot `(0,2)` — a non-largest (e.g. cyclic-order) choice. -/
def Asmall : Spec.Tensor Float (.dim 3 (.dim 3 .scalar)) := jacobiRotateAt A 0 2
def offSmall : Float := offDiagFrobSq Asmall

#eval IO.println s!"off-diagonal mass = {offBefore}; average share = {offBefore / offCount}; \
  guaranteed post-rotation bound (1-2/(n²-n))·mass = {factor * offBefore}"
#eval IO.println s!"largest pivot²  = {bigSq}  (≥ average) → off-mass after = {offBig}"
#eval IO.println s!"tiny    pivot²  = {smallSq} (< average) → off-mass after = {offSmall}"

-- Positive — the largest pivot carries at least the average share: `‖offDiag A‖² ≤ (n²−n)·A[p,q]²`
-- (`offSq_le_count_mul_max`). The "violation amount" is `0` when the bound holds.
#eval assertLt "largest pivot carries ≥ average share: ‖offDiag A‖² ≤ (n²−n)·A[0,1]²"
  (max (0.0 : Float) (offBefore - offCount * bigSq))

-- Positive — annihilating the largest pivot meets the linear rate (`jacobi_off_decrease_classical`).
#eval assertLt "classical contraction: ‖offDiag A'‖² ≤ (1−2/(n²−n))·‖offDiag A‖²"
  (max (0.0 : Float) (offBig - factor * offBefore))

-- Positive — the largest pivot really is annihilated.
#eval assertLt "largest-pivot rotation annihilates A'[0,1] ≈ 0"
  (Float.abs (Spec.get2 Abig ⟨0, by decide⟩ ⟨1, by decide⟩))

/-! ## Negative control: the largest-pivot hypothesis is necessary

Annihilating a *tiny* off-diagonal entry obeys the per-rotation identity (mass drops by `2·A[0,2]²`)
but removes far too little to meet the guaranteed factor — the off-diagonal mass stays above
`(1 − 2/(n²−n))·‖offDiag A‖²`. -/

-- The tiny pivot is below the average share, so the count bound does *not* certify the rate for it.
#eval IO.println s!"tiny pivot still annihilated: A'[0,2] = {Spec.get2 Asmall ⟨0, by decide⟩ ⟨2, by decide⟩}, \
  and mass did drop ({offBefore} → {offSmall}) — just not by enough"

-- Negative — a non-largest pivot misses the guaranteed contraction by a wide margin.
#eval assertGe "non-largest pivot fails the (1−2/(n²−n)) rate (largest-pivot hypothesis needed)"
  (offSmall - factor * offBefore) 0.5

end NN.Examples.Factorization.JacobiRate
