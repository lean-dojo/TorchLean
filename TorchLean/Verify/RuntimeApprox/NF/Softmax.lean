/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

import Mathlib.Algebra.BigOperators.Field
public import TorchLean.NN.Softmax
public import TorchLean.NN.Softmax.FDeriv
public import TorchLean.Verify.RuntimeApprox.NF.ShapeOps
public import TorchLean.Tensor.Layers.Attention

/-!
# Axis softmax

Forward and backward error bounds for the coupled, vector-valued softmax used in attention, in the
rounded-real `NF` model, with analytic facts about the exact softmax and its JVP. It is separate
from the elementwise NF sigmoid: axis softmax has a shared denominator and a dense Jacobian, while
the sigmoid acts on each tensor entry by itself.

The stable spec implementation is `TorchLean.Softmax.vec`, and
`TorchLean.NN.Softmax` proves that its entries are positive, sum to one, and lie in
`[0,1]`. The analytic derivative is `TorchLean.Vec.softmaxJVP`. Its Jacobian is self-adjoint, so
the same formula implements the VJP. The rounded-real bounds follow max subtraction, exponential,
sequential summation, and division, and they take their denominator-error margins as proof premises.
The module is noncomputable and contains no executable certificate checker, and a native softmax
kernel needs its own proof. The conservation law `sum_softmaxJVP` concerns the exact real JVP, whose
coordinates sum to zero, and a rounded VJP need not satisfy it exactly.

References:

* A. Griewank and A. Walther, *Evaluating Derivatives*, 2nd ed., 2008, for forward/reverse
  differentiation of coupled maps.
* A. A. Baydin et al., "Automatic Differentiation in Machine Learning: a Survey," JMLR 2018.
* PyTorch `torch.nn.functional.softmax` documentation for the runtime axis convention.
-/

@[expose] public section

namespace TorchLean

open scoped BigOperators

open Vec
open Tensor Softmax
open Robustness
open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq

noncomputable section

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

/-! ## Maximum and max shift -/

omit [ValidRndToNearest rnd] in
/-- Forgetting the `NF` format commutes with the nonempty-vector maximum exactly.

The `NF` maximum selects one operand, so it performs no arithmetic and introduces no rounding
error. The proof is a single fold homomorphism, which spares each stable normalization operator a
coordinate induction of its own.
-/
theorem NF.Softmax.toReal_maxVec {n : Nat} (xR : Tensor R [Nat.succ n]) :
    NF.toReal
        (Tensor.item (Softmax.maxVec xR)) =
      Tensor.item
        (Softmax.maxVec
          (Tensor.map (NF.toReal) xR)) := by
  let first : Fin (Nat.succ n) := ⟨0, Nat.succ_pos n⟩
  let runtimeValue : Fin (Nat.succ n) → R := fun i => xR.getScalar i
  let realValue : Fin (Nat.succ n) → ℝ := fun i =>
    NF.toReal (runtimeValue i)
  have hfold := List.foldl_hom
    (f := NF.toReal)
    (g₁ := fun acc i => max acc (runtimeValue i))
    (g₂ := fun acc i => max acc (realValue i))
    (l := List.finRange (Nat.succ n))
    (init := runtimeValue first)
    (fun acc i => by
      simpa [realValue] using
        (Monotone.map_max (a := acc) (b := runtimeValue i)
          (show Monotone (NF.toReal) from
            fun _ _ h => h)).symm)
  rw [Softmax.item_maxVec]
  simpa [Softmax.maxVec, NF.maximum_eq_max, runtimeValue, realValue, first] using hfold.symm

omit [ValidRndToNearest rnd] in
/-- The maximum of a rounded vector approximates the real maximum with the same infinity-norm
budget as the vector itself. No additional ULP term appears because maximum is a selection.
-/
theorem ApproxTensor.maxVec {n : Nat}
    {xS : Tensor ℝ [Nat.succ n]}
    {xR : Tensor R [Nat.succ n]} {eps : ℝ}
    (hx : ApproxTensor (α := R)
      (toSpec := NF.toReal) xS xR eps) :
    ApproxTensor (α := R)
      (toSpec := NF.toReal)
      (Softmax.maxVec xS) (Softmax.maxVec xR) eps := by
  classical
  let xHat : Tensor ℝ [Nat.succ n] :=
    Tensor.map (NF.toReal) xR
  let mS : ℝ := Tensor.item (Softmax.maxVec xS)
  let mHat : ℝ := Tensor.item (Softmax.maxVec xHat)
  have hpoint :
      ∀ i, |Tensor.getScalar xHat i - Tensor.getScalar xS i| ≤ eps := by
    intro i
    have hi := unstack (α := R)
      (toSpec := NF.toReal) hx i
    have hi' := (approxTensor_scalar_item_iff (α := R)
      (toSpec := NF.toReal)).mp hi
    simpa [xHat, Tensor.getScalar] using hi'
  rcases Softmax.exists_getScalar_eq_maxVec xS with ⟨iS, hiS⟩
  rcases Softmax.exists_getScalar_eq_maxVec xHat with ⟨iHat, hiHat⟩
  have hmS_le : mS ≤ mHat + eps := by
    have hcoord := (abs_sub_le_iff.mp (hpoint iS)).2
    have hmax := Softmax.getScalar_le_maxVec xHat iS
    dsimp [mS, mHat] at *
    linarith
  have hmHat_le : mHat ≤ mS + eps := by
    have hcoord := (abs_sub_le_iff.mp (hpoint iHat)).1
    have hmax := Softmax.getScalar_le_maxVec xS iHat
    dsimp [mS, mHat] at *
    linarith
  have hmaxError : |mHat - mS| ≤ eps := by
    rw [abs_le]
    constructor <;> linarith
  have hbridge := NF.Softmax.toReal_maxVec (β := β) (fexp := fexp) (rnd := rnd) xR
  rw [← Tensor.scalar_item (Softmax.maxVec xS),
    ← Tensor.scalar_item (Softmax.maxVec xR)]
  apply (approxTensor_scalar_iff (α := R)
    (toSpec := NF.toReal)).mpr
  rw [hbridge]
  exact hmaxError

/-! ## Rounded stable softmax -/

/-- Error after subtracting the rounded maximum from every logit. -/
def NF.Softmax.shiftErrorBound {n : Nat} (eps : ℝ)
    (xR : Tensor R [Nat.succ n]) : ℝ :=
  let maxR := Softmax.maxVec xR
  let maxRepR : Tensor R [Nat.succ n] := Tensor.full _ maxR.item
  linfNorm (subBoundTensor (β := β) (fexp := fexp) eps eps xR maxRepR)

/-- Error after exponentiating the max-shifted logits. -/
def NF.Softmax.exponentErrorBound {n : Nat} (eps : ℝ)
    (xR : Tensor R [Nat.succ n]) : ℝ :=
  let maxRepR : Tensor R [Nat.succ n] :=
    Tensor.full _ (Softmax.maxVec xR).item
  let shiftedR := xR - maxRepR
  linfNorm
    (expBoundTensor (β := β) (fexp := fexp)
      (shiftErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR) shiftedR)

/-- Error in the sequentially rounded denominator reduction. -/
def NF.Softmax.denominatorErrorBound {n : Nat} (eps : ℝ)
    (xR : Tensor R [Nat.succ n]) : ℝ :=
  sumBound (β := β) (fexp := fexp) (rnd := rnd)
    (exponentErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR)
    (Softmax.maxShiftedExpVec xR)

/-- Per-coordinate output budget for stable softmax.

The exact denominator is at least one, and the forward theorem also takes a proof of
`denominatorErrorBound eps xR < 1`, which keeps the rounded denominator positive. That premise is a
proposition about a real number, proved by the caller.
-/
def NF.Softmax.boundTensor {n : Nat} (eps : ℝ)
    (xR : Tensor R [Nat.succ n]) : Tensor ℝ [Nat.succ n] :=
  let exR := Softmax.maxShiftedExpVec xR
  let denomR : R := Tensor.sum exR
  let epsNum := exponentErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR
  let epsDenom := denominatorErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR
  Tensor.map
    (fun numR => divPosErrorBound (β := β) (fexp := fexp)
      1 epsNum epsDenom
      (NF.toReal numR)
      (NF.toReal denomR))
    exR

/-- Infinity-norm forward-error budget for stable vector softmax. -/
def NF.Softmax.errorBound {n : Nat} (eps : ℝ)
    (xR : Tensor R [Nat.succ n]) : ℝ :=
  linfNorm (boundTensor (β := β) (fexp := fexp) (rnd := rnd) eps xR)

-- Select the ordinary branch before instantiating the full rounded-real scalar dictionary.
private theorem NF.Softmax.normalizeWeights_eq_div {α : Type} [Storage α] [Context α]
    {n : Nat} (values : Tensor α [n])
    (h : Numeric.QuotientArithmetic.supported (α := α) = false) :
    Softmax.normalizeWeights values =
      values / Tensor.full _ (Tensor.sum values) := by
  simp only [Softmax.normalizeWeights, h, Bool.not_false, ↓reduceIte]

/-- The max-shifted `NF` implementation approximates real vector softmax.

The theorem follows the rounded-real stages: maximum, subtraction, exponential, sequential sum,
and division. Besides the input approximation, it requires a proof that the denominator error is
below `1`, the lower bound of the exact denominator.
-/
theorem ApproxTensor.softmaxVec {n : Nat}
    {xS : Tensor ℝ [Nat.succ n]}
    {xR : Tensor R [Nat.succ n]} {eps : ℝ}
    (hx : ApproxTensor (α := R)
      (toSpec := NF.toReal) xS xR eps)
    (hdenom : NF.Softmax.denominatorErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR < 1) :
    ApproxTensor (α := R)
      (toSpec := NF.toReal)
      (Softmax.vec xS) (Softmax.vec xR)
      (NF.Softmax.errorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR) := by
  classical
  let maxS := Softmax.maxVec xS
  let maxR := Softmax.maxVec xR
  let maxRepS : Tensor ℝ [Nat.succ n] := Tensor.full _ maxS.item
  let maxRepR : Tensor R [Nat.succ n] := Tensor.full _ maxR.item
  let shiftedS := xS - maxRepS
  let shiftedR := xR - maxRepR
  let exS := Softmax.maxShiftedExpVec xS
  let exR := Softmax.maxShiftedExpVec xR
  let denomS : ℝ := Tensor.sum exS
  let denomR : R := Tensor.sum exR
  let epsShift := NF.Softmax.shiftErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR
  let epsNum := NF.Softmax.exponentErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR
  let epsDenom := NF.Softmax.denominatorErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR
  let outBound := NF.Softmax.errorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR

  have hmax := maxVec (β := β) (fexp := fexp) (rnd := rnd) hx
  have hmaxRep : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      maxRepS maxRepR eps := by
    simpa [maxRepS, maxRepR, maxS, maxR] using
      (full_item (β := β) (fexp := fexp) (rnd := rnd)
        (s := .dim (Nat.succ n) .scalar) hmax)
  have hshift : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      shiftedS shiftedR epsShift := by
    have h := sub (β := β) (fexp := fexp) (rnd := rnd) hx hmaxRep
    simpa [shiftedS, shiftedR, epsShift, NF.Softmax.shiftErrorBound, maxRepR, maxR] using h
  have hexp : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      exS exR epsNum := by
    have h := exp (β := β) (fexp := fexp) (rnd := rnd) hshift
    simpa [exS, exR, shiftedS, shiftedR, epsNum, NF.Softmax.exponentErrorBound,
      Softmax.maxShiftedExpVec, maxRepS, maxRepR, maxS, maxR] using h
  have hsum : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      (Tensor.scalar denomS) (Tensor.scalar denomR) epsDenom := by
    have h := sum (β := β) (fexp := fexp) (rnd := rnd) hexp
    simpa [denomS, denomR, epsDenom, NF.Softmax.denominatorErrorBound, epsNum, exR] using h
  have hdenomLower : (1 : ℝ) ≤ denomS := by
    simpa [denomS, exS] using (Softmax.sum_maxShiftedExpVec_bounds xS).1
  have hbudget : epsDenom < (1 : ℝ) := by
    simpa [epsDenom] using hdenom
  have hOutNonneg : 0 ≤ outBound := by
    simpa [outBound, NF.Softmax.errorBound] using
      (linfNorm_nonneg
        (t := NF.Softmax.boundTensor (β := β) (fexp := fexp) (rnd := rnd) eps xR))

  refine approxTensor_dim_of_forall
    (xS := Softmax.vec xS)
    (xR := Softmax.vec xR)
    (eps := outBound) hOutNonneg ?_
  intro i
  let numS := (exS.unstack i).item
  let numR := (exR.unstack i).item
  have hnumI := unstack (α := R)
    (toSpec := NF.toReal) hexp i
  have hnumScalar := (approxTensor_scalar_item_iff (α := R)
    (toSpec := NF.toReal)).mp hnumI
  have hdenomScalar := (approxTensor_scalar_iff (α := R)
    (toSpec := NF.toReal)).mp hsum
  have hdiv := NF.approx_div_of_pos_lb
    (β := β) (fexp := fexp) (rnd := rnd) (η := (1 : ℝ))
    hdenomLower hbudget hnumScalar hdenomScalar
  have hcoord := linfNorm_unstack_le
    (t := NF.Softmax.boundTensor (β := β) (fexp := fexp) (rnd := rnd) eps xR) i
  have hdivBound :
      NF.divPosErrorBound (β := β) (fexp := fexp) 1 epsNum epsDenom
          (NF.toReal numR)
          (NF.toReal denomR) ≤
        outBound := by
    refine le_trans (le_abs_self _) ?_
    simpa [outBound, NF.Softmax.errorBound, NF.Softmax.boundTensor, exR, denomR,
      epsNum, epsDenom, numR,
      linfNorm, Numerics.MathFunctions.abs] using hcoord
  have hscalarOut : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      (Tensor.scalar (numS / denomS)) (Tensor.scalar (numR / denomR)) outBound :=
    (approxTensor_scalar_iff (α := R)
      (toSpec := NF.toReal)).mpr
        (le_trans hdiv hdivBound)
  have hsoftS :
      (Softmax.vec xS).unstack i =
        Tensor.scalar (numS / denomS) := by
    change (Softmax.normalizeWeights exS).unstack i = _
    rw [NF.Softmax.normalizeWeights_eq_div _ (by rfl)]
    apply Tensor.Rep.ext
    intro coordinate
    cases coordinate
    simp [exS, denomS, numS,
      Tensor.div_def,
      Tensor.item, Tensor.scalar,
      Tensor.unstack_apply]
  have hsoftR :
      (Softmax.vec xR).unstack i =
        Tensor.scalar (numR / denomR) := by
    change (Softmax.normalizeWeights exR).unstack i = _
    rw [NF.Softmax.normalizeWeights_eq_div _ (by rfl)]
    apply Tensor.Rep.ext
    intro coordinate
    cases coordinate
    simp [exR, denomR, numR,
      Tensor.div_def,
      Tensor.item, Tensor.scalar,
      Tensor.unstack_apply]
  rw [hsoftS, hsoftR]
  exact hscalarOut

/-- Row-wise error tensor for axis-`1` softmax on a matrix. -/
def NF.Softmax.rowsBoundTensor {m n : Nat} (eps : ℝ)
    (xR : Tensor R [m, Nat.succ n]) :
    Tensor ℝ [m, Nat.succ n] :=
  Tensor.dim (fun i =>
    boundTensor (β := β) (fexp := fexp) (rnd := rnd) eps (xR.unstack i))

/-- Global infinity-norm budget for row-wise axis-`1` softmax. -/
def NF.Softmax.rowsErrorBound {m n : Nat} (eps : ℝ)
    (xR : Tensor R [m, Nat.succ n]) : ℝ :=
  linfNorm (rowsBoundTensor (β := β) (fexp := fexp) (rnd := rnd) eps xR)

/-- Matrix-level stable softmax theorem, obtained by applying the vector theorem independently to
each row.

The denominator premise is proved separately for each row. The output uses one global
infinity-norm budget, as required by the matrix multiplication and graph composition contracts.
-/
theorem ApproxTensor.softmax_rows {m n : Nat}
    {xS : Tensor ℝ [m, Nat.succ n]}
    {xR : Tensor R [m, Nat.succ n]} {eps : ℝ}
    (hx : ApproxTensor (α := R)
      (toSpec := NF.toReal) xS xR eps)
    (hdenom : ∀ i : Fin m,
      NF.Softmax.denominatorErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps
        (Tensor.unstack xR i) < 1) :
    ApproxTensor (α := R)
      (toSpec := NF.toReal)
      (Tensor.softmax xS ⟨1, Nat.one_lt_two⟩) (Tensor.softmax xR ⟨1, Nat.one_lt_two⟩)
      (NF.Softmax.rowsErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR) := by
  classical
  change ApproxTensor
    (toSpec := NF.toReal)
    (Softmax.innermost xS)
    (Softmax.innermost xR)
    (NF.Softmax.rowsErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR)
  let bound := NF.Softmax.rowsErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR
  have hbound : 0 ≤ bound := by
    simpa [bound, NF.Softmax.rowsErrorBound] using
      (linfNorm_nonneg
        (t := NF.Softmax.rowsBoundTensor (β := β) (fexp := fexp) (rnd := rnd) eps xR))
  refine approxTensor_dim_of_forall
    (xS := Softmax.innermost xS)
    (xR := Softmax.innermost xR)
    (eps := bound) hbound ?_
  intro i
  have hrow := unstack (α := R)
    (toSpec := NF.toReal) hx i
  have hsoft := softmaxVec (β := β) (fexp := fexp) (rnd := rnd) hrow
    (by simpa using hdenom i)
  have hrowLe :
      NF.Softmax.errorBound (β := β) (fexp := fexp) (rnd := rnd) eps (xR.unstack i) ≤ bound := by
    have h := linfNorm_unstack_le
      (t := NF.Softmax.rowsBoundTensor (β := β) (fexp := fexp) (rnd := rnd) eps xR) i
    simpa [bound, NF.Softmax.rowsErrorBound, NF.Softmax.rowsBoundTensor,
      NF.Softmax.errorBound] using h
  simpa using mono hsoft hrowLe

/-! ## Exact hard-masked softmax -/

/-- Stable softmax numerators for a row whose allowed maximum is already known.

Blocked coordinates are set to literal zero after exponentiation. The result equals the numerators
of the `some rowMax` branch of `Softmax.hardMaskedVec`, and no finite masking sentinel
appears.
-/
def NF.Softmax.hardMaskedNumerators {α : Type} [Storage α] [Context α] {n : Nat}
    (scores : Tensor α [n])
    (mask : Tensor Bool [n]) (rowMax : α) :
    Tensor α [n] :=
  let maxRep : Tensor α [n] := Tensor.full _ rowMax
  let exponentials := Tensor.exp (scores - maxRep)
  zipWith
    (fun value allowed => if allowed then value else 0) exponentials mask

/-- The staged numerator computation equals the fused entrywise expression in the public spec, as
an equality of tensor expressions.
-/
theorem NF.Softmax.hardMaskedNumerators_eq_fused {α : Type} [Storage α] [Context α] {n : Nat}
    (scores : Tensor α [n])
    (mask : Tensor Bool [n]) (rowMax : α) :
    hardMaskedNumerators scores mask rowMax =
      zipWith
        (fun score allowed => if allowed then Numerics.MathFunctions.exp (score - rowMax) else 0)
        scores mask := by
  apply Tensor.Rep.ext
  intro coordinate
  rcases coordinate with ⟨i, coordinate⟩
  cases coordinate
  simp only [hardMaskedNumerators, Tensor.exp, sub_def,
    Tensor.zipWith_apply]
  by_cases hallowed : mask (i, PUnit.unit) = true
  · simp only [hallowed, ite_true]
    change
      Tensor.map Numerics.MathFunctions.exp
          (Tensor.zipWith (· - ·) scores
            (Tensor.full _ rowMax))
          (i, PUnit.unit) =
        Numerics.MathFunctions.exp (scores (i, PUnit.unit) - rowMax)
    simp
  · simp [hallowed]

/-- Error after subtracting an approximate allowed-row maximum from every score. -/
def NF.Softmax.hardMaskedShiftError {n : Nat} (epsScores epsMax : ℝ)
    (scoresR : Tensor R [n]) (rowMaxR : R) : ℝ :=
  let maxRepR : Tensor R [n] := Tensor.full _ rowMaxR
  linfNorm
    (subBoundTensor (β := β) (fexp := fexp)
      epsScores epsMax scoresR maxRepR)

/-- Error in the hard-masked numerator vector, independent of the mask because applying it adds
no rounding error. -/
def NF.Softmax.hardMaskedNumeratorError {n : Nat} (epsScores epsMax : ℝ)
    (scoresR : Tensor R [n])
    (rowMaxR : R) : ℝ :=
  let maxRepR : Tensor R [n] := Tensor.full _ rowMaxR
  let shiftedR := scoresR - maxRepR
  let epsShift := hardMaskedShiftError (β := β) (fexp := fexp) (rnd := rnd)
    epsScores epsMax scoresR rowMaxR
  linfNorm
    (expBoundTensor (β := β) (fexp := fexp) epsShift shiftedR)

/-- Error in the sequentially rounded sum of the allowed numerators. -/
def NF.Softmax.hardMaskedDenominatorError {n : Nat} (epsScores epsMax : ℝ)
    (scoresR : Tensor R [n])
    (mask : Tensor Bool [n]) (rowMaxR : R) : ℝ :=
  sumBound (β := β) (fexp := fexp) (rnd := rnd)
    (hardMaskedNumeratorError (β := β) (fexp := fexp) (rnd := rnd)
      epsScores epsMax scoresR rowMaxR)
    (hardMaskedNumerators scoresR mask rowMaxR)

/-- Final per-coordinate budget for a nonempty hard-masked softmax row. -/
def NF.Softmax.hardMaskedBoundTensor {n : Nat} (η epsScores epsMax : ℝ)
    (scoresR : Tensor R [n])
    (mask : Tensor Bool [n]) (rowMaxR : R) :
    Tensor ℝ [n] :=
  let numeratorsR := hardMaskedNumerators scoresR mask rowMaxR
  let denominatorR : R := Tensor.sum numeratorsR
  let denominatorRepR : Tensor R [n] :=
    Tensor.full _ denominatorR
  divPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
    η
    (hardMaskedNumeratorError (β := β) (fexp := fexp) (rnd := rnd)
      epsScores epsMax scoresR rowMaxR)
    (hardMaskedDenominatorError (β := β) (fexp := fexp) (rnd := rnd)
      epsScores epsMax scoresR mask rowMaxR)
    numeratorsR denominatorRepR

/-- Rounded-real error theorem for the nonempty branch of hard-masked vector softmax.

`hmaxS` and `hmaxR` identify the selected allowed-row maxima, `hmax` relates them numerically, and
`hdenomLower` supplies the exact positive lower bound used by division. For the canonical selected
maximum this lower bound is `1`, because one allowed shifted score is zero and contributes
`exp 0 = 1`.
-/
theorem ApproxTensor.hardMaskedSoftmaxVec_of_max {n : Nat}
    {scoresS : Tensor ℝ [n]}
    {scoresR : Tensor R [n]}
    (mask : Tensor Bool [n])
    {rowMaxS : ℝ} {rowMaxR : R} {epsScores epsMax η : ℝ}
    (hscores : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      scoresS scoresR epsScores)
    (hmax : _root_.abs
      (NF.toReal rowMaxR - rowMaxS) ≤ epsMax)
    (hmaxS : Softmax.hardMaskedMax? scoresS mask = some rowMaxS)
    (hmaxR : Softmax.hardMaskedMax? scoresR mask = some rowMaxR)
    (hdenomLower : η ≤ Tensor.sum (NF.Softmax.hardMaskedNumerators scoresS mask rowMaxS))
    (hdenomMargin :
      NF.Softmax.hardMaskedDenominatorError (β := β) (fexp := fexp) (rnd := rnd)
        epsScores epsMax scoresR mask rowMaxR < η) :
    ApproxTensor (α := R)
      (toSpec := NF.toReal)
      (Softmax.hardMaskedVec scoresS mask)
      (Softmax.hardMaskedVec scoresR mask)
      (linfNorm
        (NF.Softmax.hardMaskedBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          η epsScores epsMax scoresR mask rowMaxR)) := by
  let maxS : Tensor ℝ .scalar := Tensor.scalar rowMaxS
  let maxR : Tensor R .scalar := Tensor.scalar rowMaxR
  let maxRepS : Tensor ℝ [n] := Tensor.full _ maxS.item
  let maxRepR : Tensor R [n] := Tensor.full _ maxR.item
  let shiftedS := scoresS - maxRepS
  let shiftedR := scoresR - maxRepR
  let expS := Tensor.exp shiftedS
  let expR := Tensor.exp shiftedR
  let numeratorsS := NF.Softmax.hardMaskedNumerators scoresS mask rowMaxS
  let numeratorsR := NF.Softmax.hardMaskedNumerators scoresR mask rowMaxR
  let denominatorS : ℝ := Tensor.sum numeratorsS
  let denominatorR : R := Tensor.sum numeratorsR
  let denominatorRepS : Tensor ℝ [n] :=
    Tensor.full _ denominatorS
  let denominatorRepR : Tensor R [n] :=
    Tensor.full _ denominatorR
  let epsShift := NF.Softmax.hardMaskedShiftError (β := β) (fexp := fexp) (rnd := rnd)
    epsScores epsMax scoresR rowMaxR
  let epsNumerator := NF.Softmax.hardMaskedNumeratorError (β := β) (fexp := fexp) (rnd := rnd)
    epsScores epsMax scoresR rowMaxR
  let epsDenominator := NF.Softmax.hardMaskedDenominatorError (β := β) (fexp := fexp) (rnd := rnd)
    epsScores epsMax scoresR mask rowMaxR

  have hmaxTensor : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      maxS maxR epsMax :=
    (approxTensor_scalar_iff (α := R)
      (toSpec := NF.toReal)).2 hmax
  have hmaxRep := full_item
    (β := β) (fexp := fexp) (rnd := rnd) (s := .dim n .scalar) hmaxTensor
  have hshift := sub
    (β := β) (fexp := fexp) (rnd := rnd) hscores hmaxRep
  have hexp := exp
    (β := β) (fexp := fexp) (rnd := rnd) hshift
  have hnumerators := applyBoolMask
    (β := β) (fexp := fexp) (rnd := rnd) mask hexp
  have hsum := sum
    (β := β) (fexp := fexp) (rnd := rnd) hnumerators
  have hdenominatorRep := full_item
    (β := β) (fexp := fexp) (rnd := rnd)
    (s := .dim n .scalar) hsum
  have hdenominatorDomain :
      Tensor.Forall (fun z : ℝ => η ≤ z) denominatorRepS := by
    exact Tensor.forall_full (by simpa [denominatorS, numeratorsS] using hdenomLower)
  have hout := div_of_pos_lb
    (β := β) (fexp := fexp) (rnd := rnd) η
    hnumerators hdenominatorRep hdenominatorDomain hdenomMargin
  simp only [Softmax.hardMaskedVec, hmaxS, hmaxR]
  rw [← NF.Softmax.hardMaskedNumerators_eq_fused scoresS mask rowMaxS,
    ← NF.Softmax.hardMaskedNumerators_eq_fused scoresR mask rowMaxR]
  rw [NF.Softmax.normalizeWeights_eq_div _ (by rfl),
    NF.Softmax.normalizeWeights_eq_div _ (by rfl)]
  simpa [
    NF.Softmax.hardMaskedBoundTensor, NF.Softmax.hardMaskedNumerators,
    NF.Softmax.hardMaskedShiftError, NF.Softmax.hardMaskedNumeratorError,
    NF.Softmax.hardMaskedDenominatorError,
    maxS, maxR, maxRepS, maxRepR, shiftedS, shiftedR, expS, expR,
    numeratorsS, numeratorsR, denominatorS, denominatorR,
    denominatorRepS, denominatorRepR, epsShift, epsNumerator, epsDenominator] using hout

/-- If both semantics find no allowed key, hard-masked softmax agrees exactly on the zero row. -/
theorem ApproxTensor.hardMaskedSoftmaxVec_allBlocked {n : Nat}
    (scoresS : Tensor ℝ [n])
    (scoresR : Tensor R [n])
    (mask : Tensor Bool [n])
    (hmaxS : Softmax.hardMaskedMax? scoresS mask = none)
    (hmaxR : Softmax.hardMaskedMax? scoresR mask = none) :
    ApproxTensor (α := R)
      (toSpec := NF.toReal)
      (Softmax.hardMaskedVec scoresS mask)
      (Softmax.hardMaskedVec scoresR mask) 0 := by
  simp only [Softmax.hardMaskedVec, hmaxS, hmaxR]
  change ApproxTensor (α := R)
    (toSpec := NF.toReal)
    (Tensor.full (.dim n .scalar) (0 : ℝ)) (Tensor.full (.dim n .scalar) (0 : R)) 0
  exact full_zero (β := β) (fexp := fexp) (rnd := rnd)
    (s := .dim n .scalar)

/-- Proof-bearing evidence for nonempty hard-masked softmax rows.

The data fields give each row's exact and rounded maxima, their error budget, and a lower bound on
the exact denominator, and the proposition fields relate them to the exact and rounded rows. The
caller supplies the proofs, and because the structure is indexed by both score matrices, the mask,
and the score error budget, they are tied to those inputs.
-/
structure NF.Softmax.HardMaskedRowsEvidence {m n : Nat}
    (scoresS : Tensor ℝ [m, n])
    (scoresR : Tensor R [m, n])
    (mask : Tensor Bool [m, n])
    (epsScores : ℝ) where
  /-- Exact allowed-row maximum of each row. -/
  rowMaxS : Fin m → ℝ
  /-- Rounded allowed-row maximum of each row. -/
  rowMaxR : Fin m → R
  /-- Error budget relating the two maxima of each row. -/
  epsMax : Fin m → ℝ
  /-- Positive lower bound on the exact denominator of each row. -/
  eta : Fin m → ℝ
  /-- The rounded maximum approximates the exact one within `epsMax`. -/
  approx_rowMax : ∀ i,
    abs (NF.toReal (rowMaxR i) -
      rowMaxS i) ≤ epsMax i
  /-- `rowMaxS` is the maximum selected by the exact hard-masked softmax. -/
  rowMaxS_spec : ∀ i,
    Softmax.hardMaskedMax? (Tensor.unstack scoresS i) (Tensor.unstack mask i) = some (rowMaxS i)
  /-- `rowMaxR` is the maximum selected by the rounded hard-masked softmax. -/
  rowMaxR_spec : ∀ i,
    Softmax.hardMaskedMax? (Tensor.unstack scoresR i) (Tensor.unstack mask i) = some (rowMaxR i)
  /-- The exact denominator of each row is at least `eta`. -/
  eta_le_sum : ∀ i,
    eta i ≤ Tensor.sum
      (hardMaskedNumerators (Tensor.unstack scoresS i) (Tensor.unstack mask i) (rowMaxS i))
  /-- The rounded denominator error of each row stays strictly below `eta`. -/
  error_lt_eta : ∀ i,
    hardMaskedDenominatorError (β := β) (fexp := fexp) (rnd := rnd)
      epsScores (epsMax i) (Tensor.unstack scoresR i) (Tensor.unstack mask i) (rowMaxR i) < eta i

/-- Rowwise hard-masked softmax bounds with one independently certified maximum per row. -/
def NF.Softmax.hardMaskedRowsBoundTensor {m n : Nat}
    (η : Fin m → ℝ) (epsScores : ℝ)
    (scoresR : Tensor R [m, n])
    (mask : Tensor Bool [m, n])
    (rowMaxR : Fin m → R) (epsMax : Fin m → ℝ) :
    Tensor ℝ [m, n] :=
  Tensor.dim (fun i =>
    hardMaskedBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      (η i) epsScores (epsMax i) (scoresR.unstack i) (mask.unstack i) (rowMaxR i))

/-- Matrix-level hard-masked softmax when every row has at least one allowed coordinate.

The selected maxima and denominator premises are stated per row, which fits causal attention,
where row `i` always admits key `i`. Each row keeps its own intermediate scale until the final
infinity norm combines them.
-/
theorem ApproxTensor.hardMaskedSoftmax_of_max {m n : Nat}
    {scoresS : Tensor ℝ [m, n]}
    {scoresR : Tensor R [m, n]}
    (mask : Tensor Bool [m, n])
    {epsScores : ℝ}
    (hscores : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      scoresS scoresR epsScores)
    (evidence : NF.Softmax.HardMaskedRowsEvidence (β := β) (fexp := fexp) (rnd := rnd)
      scoresS scoresR mask epsScores) :
    ApproxTensor (α := R)
      (toSpec := NF.toReal)
      (Softmax.hardMasked scoresS mask)
      (Softmax.hardMasked scoresR mask)
      (linfNorm
        (NF.Softmax.hardMaskedRowsBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          evidence.eta epsScores scoresR mask evidence.rowMaxR evidence.epsMax)) := by
  let bound := linfNorm
    (NF.Softmax.hardMaskedRowsBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      evidence.eta epsScores scoresR mask evidence.rowMaxR evidence.epsMax)
  have hbound : 0 ≤ bound := by
    simpa [bound] using
      (linfNorm_nonneg
        (t := NF.Softmax.hardMaskedRowsBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          evidence.eta epsScores scoresR mask evidence.rowMaxR evidence.epsMax))
  refine approxTensor_dim_of_forall
    (toSpec := NF.toReal)
    (xS := Softmax.hardMasked scoresS mask)
    (xR := Softmax.hardMasked scoresR mask)
    (eps := bound) hbound ?_
  intro i
  have hscoresI := unstack (α := R)
    (toSpec := NF.toReal) hscores i
  have hrow := hardMaskedSoftmaxVec_of_max
    (β := β) (fexp := fexp) (rnd := rnd)
    (mask.unstack i) hscoresI (evidence.approx_rowMax i)
    (by simpa using evidence.rowMaxS_spec i)
    (by simpa using evidence.rowMaxR_spec i)
    (by simpa using evidence.eta_le_sum i)
    (by simpa using evidence.error_lt_eta i)
  have hrowLe :
      linfNorm
          (NF.Softmax.hardMaskedBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (evidence.eta i) epsScores (evidence.epsMax i)
            (scoresR.unstack i) (mask.unstack i) (evidence.rowMaxR i)) ≤
        bound := by
    have h := linfNorm_unstack_le
      (t := NF.Softmax.hardMaskedRowsBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
        evidence.eta epsScores scoresR mask evidence.rowMaxR evidence.epsMax) i
    simpa [bound, NF.Softmax.hardMaskedRowsBoundTensor] using h
  simpa [Softmax.hardMasked] using mono hrow hrowLe

/-! ## Rounded softmax backward -/

/-- Error in the `dY * softmax(x)` product used by the softmax VJP. -/
def NF.Softmax.vjpProductError {n : Nat} (epsDY epsY : ℝ)
    (dYR yR : Tensor R [Nat.succ n]) : ℝ :=
  linfNorm (mulBoundTensor (β := β) (fexp := fexp) epsDY epsY dYR yR)

/-- Error in the rounded dot product `sum (dY * softmax(x))`. -/
def NF.Softmax.vjpDotError {n : Nat} (epsDY epsY : ℝ)
    (dYR yR : Tensor R [Nat.succ n]) : ℝ :=
  let productR := dYR * yR
  sumBound (β := β) (fexp := fexp) (rnd := rnd)
    (vjpProductError (β := β) (fexp := fexp) epsDY epsY dYR yR) productR

/-- Error after subtracting the replicated rounded softmax dot product from `dY`. -/
def NF.Softmax.vjpCenteredError {n : Nat} (epsDY epsY : ℝ)
    (dYR yR : Tensor R [Nat.succ n]) : ℝ :=
  let dotR : R := Tensor.sum (dYR * yR)
  let dotRepR : Tensor R [Nat.succ n] := Tensor.full _ dotR
  linfNorm (subBoundTensor (β := β) (fexp := fexp)
    epsDY (vjpDotError (β := β) (fexp := fexp) (rnd := rnd) epsDY epsY dYR yR)
    dYR dotRepR)

/-- End-to-end infinity-norm budget for the rounded softmax VJP.

With `center = some i`, the budget also charges the rounded subtraction of upstream coordinate `i`
from every coordinate before the weighted reduction. Selecting the coordinate adds no rounding, and
both operands of the subtraction carry the upstream budget `epsDY`. The default `none` gives the
uncentered formula for arbitrary supplied weights, and attention uses the centered one.
-/
def NF.Softmax.vjpErrorBound {n : Nat} (epsDY epsY : ℝ)
    (dYR yR : Tensor R [Nat.succ n]) (center : Option (Fin (Nat.succ n)) := none) : ℝ :=
  let uncentered (eps : ℝ) (upstream : Tensor R [Nat.succ n]) :=
    let dotR : R := Tensor.sum (upstream * yR)
    let centeredR := upstream -
      (Tensor.full _ dotR : Tensor R [Nat.succ n])
    linfNorm (mulBoundTensor (β := β) (fexp := fexp)
      epsY (vjpCenteredError (β := β) (fexp := fexp) (rnd := rnd) eps epsY upstream yR)
      yR centeredR)
  match center with
  | none => uncentered epsDY dYR
  | some i =>
      let anchor : Tensor R [Nat.succ n] := Tensor.full _ (dYR.getScalar i)
      let centered := dYR - anchor
      let eps := linfNorm (subBoundTensor (β := β) (fexp := fexp)
        epsDY epsDY dYR anchor)
      uncentered eps centered

/-- Rounded VJP theorem for any already-certified softmax weight vector.

Ordinary and hard-masked softmax share the formula. Blocked weights are exactly zero, so the
gradient at blocked logits is exactly zero and the masked case needs no finite-sentinel derivative
rule.
-/
theorem ApproxTensor.softmaxBackwardFromWeights {n : Nat}
    {yS dYS : Tensor ℝ [Nat.succ n]}
    {yR dYR : Tensor R [Nat.succ n]} {epsY epsDY : ℝ}
    (hy : ApproxTensor (α := R)
      (toSpec := NF.toReal) yS yR epsY)
    (hdY : ApproxTensor (α := R)
      (toSpec := NF.toReal) dYS dYR epsDY) :
    ApproxTensor (α := R)
      (toSpec := NF.toReal)
      (Softmax.backwardFromWeights yS dYS)
      (Softmax.backwardFromWeights yR dYR)
      (NF.Softmax.vjpErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        epsDY epsY dYR yR) := by
  let productS := dYS * yS
  let productR := dYR * yR
  let epsProduct := NF.Softmax.vjpProductError (β := β) (fexp := fexp) epsDY epsY dYR yR
  let dotS : ℝ := Tensor.sum productS
  let dotR : R := Tensor.sum productR
  let epsDot := NF.Softmax.vjpDotError (β := β) (fexp := fexp) (rnd := rnd) epsDY epsY dYR yR
  let dotRepS : Tensor ℝ [Nat.succ n] := Tensor.full _ dotS
  let dotRepR : Tensor R [Nat.succ n] := Tensor.full _ dotR
  let centeredS := dYS - dotRepS
  let centeredR := dYR - dotRepR
  let epsCentered :=
    NF.Softmax.vjpCenteredError (β := β) (fexp := fexp) (rnd := rnd) epsDY epsY dYR yR

  have hproduct : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      productS productR epsProduct := by
    have h := mul (β := β) (fexp := fexp) (rnd := rnd) hdY hy
    simpa [productS, productR, epsProduct, NF.Softmax.vjpProductError] using h
  have hdot : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      (Tensor.scalar dotS) (Tensor.scalar dotR) epsDot := by
    have h := sum (β := β) (fexp := fexp) (rnd := rnd) hproduct
    simpa [dotS, dotR, epsDot, NF.Softmax.vjpDotError, productR, epsProduct] using h
  have hdotRep : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      dotRepS dotRepR epsDot := by
    simpa [dotRepS, dotRepR] using
      (full_item (β := β) (fexp := fexp) (rnd := rnd)
        (s := .dim (Nat.succ n) .scalar) hdot)
  have hcentered : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      centeredS centeredR epsCentered := by
    have h := sub (β := β) (fexp := fexp) (rnd := rnd) hdY hdotRep
    simpa [centeredS, centeredR, epsCentered, NF.Softmax.vjpCenteredError, dotRepR, dotR, productR,
      epsDot] using h
  have hout := mul (β := β) (fexp := fexp) (rnd := rnd) hy hcentered
  simpa [Softmax.backwardFromWeights, productS, productR, dotS, dotR,
    dotRepS, dotRepR, centeredS, centeredR, epsCentered, NF.Softmax.vjpErrorBound] using hout

-- Rewrite the unsupported branch abstractly, before instantiating its scalar dictionary with NF.
-- Reducing the concrete dictionary through the complete replay body makes kernel checking costly.
private theorem NF.Softmax.pullback_eq_of_unsupported {α : Type} [Storage α] [Context α]
    {n : Nat} (weights gradient : Tensor α [n]) (anchor : α)
    (h : Numeric.QuotientArithmetic.supported (α := α) = false) :
    Softmax.pullback weights gradient anchor =
      let centered := gradient - Tensor.full _ anchor
      let mean := Tensor.sum (centered * weights)
      weights * (centered - Tensor.full _ mean) := by
  simp only [Softmax.pullback, h, Bool.not_false, ↓reduceIte]

/-- Numerical bound for the centered helper used by attention's JVP and VJP.

The exact row must be normalized or identically zero. The pivot of the rounded weights selects the
anchor, and the exact weights may have their maximum elsewhere. The budget counts the upstream
error of both subtraction operands and the rounding of the subtraction itself, which precedes the
weighted reduction.
-/
theorem ApproxTensor.softmaxBackwardFromWeights_centered {n : Nat}
    {yS dYS : Tensor ℝ [Nat.succ n]}
    {yR dYR : Tensor R [Nat.succ n]} {epsY epsDY : ℝ}
    (hy : ApproxTensor (α := R)
      (toSpec := NF.toReal) yS yR epsY)
    (hdY : ApproxTensor (α := R)
      (toSpec := NF.toReal) dYS dYR epsDY)
    (hmass : (∑ j, yS.getScalar j) = 1 ∨ ∀ j, yS.getScalar j = 0) :
    let center : Fin (Nat.succ n) := Softmax.pivot yR
    ApproxTensor (α := R)
      (toSpec := NF.toReal)
      (Softmax.backwardFromWeights yS dYS (center := true))
      (Softmax.backwardFromWeights yR dYR (center := true))
      (NF.Softmax.vjpErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        epsDY epsY dYR yR (center := some center)) := by
  dsimp only
  let center : Fin (Nat.succ n) := Softmax.pivot yR
  have hanchor : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      (Tensor.scalar (dYS.getScalar center)) (Tensor.scalar (dYR.getScalar center)) epsDY := by
    apply approxTensor_scalar_iff.mpr
    have hi := unstack hdY center
    have hscalar := approxTensor_scalar_item_iff.mp hi
    simpa [Tensor.getScalar] using hscalar
  let anchorS : Tensor ℝ [Nat.succ n] := Tensor.full _ (dYS.getScalar center)
  let anchorR : Tensor R [Nat.succ n] := Tensor.full _ (dYR.getScalar center)
  let shiftedS := dYS - anchorS
  let shiftedR := dYR - anchorR
  let epsShifted := linfNorm
    (NF.subBoundTensor (β := β) (fexp := fexp) epsDY epsDY dYR anchorR)
  have hshifted : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      shiftedS shiftedR epsShifted :=
    sub hdY (full_item hanchor)
  have hout := softmaxBackwardFromWeights
    (β := β) (fexp := fexp) (rnd := rnd) hy hshifted
  have hruntime : Softmax.backwardFromWeights yR dYR (center := true) =
      Softmax.backwardFromWeights yR shiftedR := by
    change Softmax.pullback yR dYR (dYR.getScalar center) = _
    rw [NF.Softmax.pullback_eq_of_unsupported _ _ _ (by rfl)]
    rfl
  rw [Softmax.backwardFromWeights_shift yS dYS (dYS.getScalar center) hmass
    (center := true), hruntime]
  exact hout

-- In real arithmetic the upstream shift cancels because the softmax weights sum to one.
-- Its value need not come from the real argmax: the numerical proof uses the rounded pivot.
private theorem NF.Softmax.backward_centered {n : Nat} (x dY : Tensor ℝ [Nat.succ n])
    (anchor : ℝ) :
    Tensor.softmaxBackward x dY ⟨0, Shape.AxisInBounds.proof⟩ =
      backwardFromWeights (Softmax.vec x)
        (dY - Tensor.full _ anchor) := by
  let y := Softmax.vec x
  let shifted := dY - Tensor.full _ anchor
  have hcoord (j : Fin (Nat.succ n)) : shifted.getScalar j = dY.getScalar j - anchor := by
    simp [shifted, sub_def]
  have hmass : (∑ j, y.getScalar j) = 1 := by
    simpa [Tensor.sum_vec, y] using Softmax.sum_vec x
  have hdot : Tensor.sum (shifted * y) =
      (∑ j, y.getScalar j * dY.getScalar j) - anchor := by
    rw [Tensor.sum_vec]
    simp_rw [getScalar_mul, hcoord, mul_comm (_ - anchor), mul_sub]
    rw [Finset.sum_sub_distrib, ← Finset.sum_mul, hmass, one_mul]
  apply Tensor.ext_vector
  intro i
  rw [Tensor.softmaxBackward_zero_vec, Softmax.getScalar_innermostBackward]
  change _ = (y * (shifted -
    Tensor.full _ (Tensor.sum (shifted * y)))).getScalar i
  rw [getScalar_mul, sub_def, Tensor.getScalar_zipWith, hcoord, hdot]
  simp only [Tensor.getScalar_full]
  change y.getScalar i * _ = y.getScalar i * _
  ring

/-- Error theorem for the rounded-real softmax VJP, including its upstream centering.

It is the training-time companion of `softmaxVec` and takes the same denominator
premise. The anchor is the upstream coordinate at the argmax of the rounded logits, which need not
be the exact argmax. Selecting it adds no rounding, and the budget charges the first subtraction
before `ApproxTensor.softmaxBackwardFromWeights` bounds the weighted reduction.
-/
theorem ApproxTensor.softmaxBackward {n : Nat}
    {xS dYS : Tensor ℝ [Nat.succ n]}
    {xR dYR : Tensor R [Nat.succ n]} {epsX epsDY : ℝ}
    (hx : ApproxTensor (α := R)
      (toSpec := NF.toReal) xS xR epsX)
    (hdY : ApproxTensor (α := R)
      (toSpec := NF.toReal) dYS dYR epsDY)
    (hdenom : NF.Softmax.denominatorErrorBound (β := β) (fexp := fexp) (rnd := rnd) epsX xR < 1) :
    let center : Fin (Nat.succ n) := Softmax.pivot xR
    let yR := Softmax.vec xR
    ApproxTensor (α := R)
      (toSpec := NF.toReal)
      (Tensor.softmaxBackward xS dYS ⟨0, Nat.one_pos⟩)
      (Tensor.softmaxBackward xR dYR ⟨0, Nat.one_pos⟩)
      (NF.Softmax.vjpErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        epsDY (NF.Softmax.errorBound (β := β) (fexp := fexp) (rnd := rnd) epsX xR)
        dYR yR (center := some center)) := by
  dsimp only
  let center : Fin (Nat.succ n) := Softmax.pivot xR
  let yS := Softmax.vec xS
  let yR := Softmax.vec xR
  let epsY := NF.Softmax.errorBound (β := β) (fexp := fexp) (rnd := rnd) epsX xR
  have hy : ApproxTensor (α := R)
      (toSpec := NF.toReal) yS yR epsY := by
    simpa [yS, yR, epsY] using
      (softmaxVec (β := β) (fexp := fexp) (rnd := rnd) hx hdenom)
  have hanchor : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      (Tensor.scalar (dYS.getScalar center)) (Tensor.scalar (dYR.getScalar center)) epsDY := by
    apply approxTensor_scalar_iff.mpr
    have hi := unstack hdY center
    have hscalar := approxTensor_scalar_item_iff.mp hi
    simpa [Tensor.getScalar] using hscalar
  let anchorS : Tensor ℝ [Nat.succ n] := Tensor.full _ (dYS.getScalar center)
  let anchorR : Tensor R [Nat.succ n] := Tensor.full _ (dYR.getScalar center)
  let shiftedS := dYS - anchorS
  let shiftedR := dYR - anchorR
  let epsShifted := linfNorm
    (NF.subBoundTensor (β := β) (fexp := fexp) epsDY epsDY dYR anchorR)
  have hshifted : ApproxTensor (α := R)
      (toSpec := NF.toReal)
      shiftedS shiftedR epsShifted :=
    sub hdY (full_item hanchor)
  have hout := softmaxBackwardFromWeights
    (β := β) (fexp := fexp) (rnd := rnd) hy hshifted
  have hruntime : Tensor.softmaxBackward xR dYR ⟨0, Shape.AxisInBounds.proof⟩ =
      Softmax.backwardFromWeights yR shiftedR := by
    change Softmax.pullback yR dYR (dYR.getScalar center) = _
    rw [NF.Softmax.pullback_eq_of_unsupported _ _ _ (by rfl)]
    rfl
  rw [NF.Softmax.backward_centered xS dYS (dYS.getScalar center), hruntime]
  exact hout

/-- The analytic softmax on a nonempty vector sums to one. -/
theorem Vec.sum_softmax {n : Nat} (x : Vec (Nat.succ n)) :
    (∑ i, softmax x i) = 1 := by
  simp only [softmax, PiLp.toLp_apply]
  rw [← Finset.sum_div]
  exact div_self (sumExp_ne_zero x)

/-- A softmax JVP is tangent to the probability simplex: its coordinates sum to zero. -/
theorem Vec.sum_softmaxJVP {n : Nat} (x dx : Vec (Nat.succ n)) :
    (∑ i, softmaxJVP x dx i) = 0 := by
  classical
  let y : Vec (Nat.succ n) := softmax x
  let s : Real := inner ℝ y dx
  have hy : (∑ i, y i) = 1 := sum_softmax x
  have hs : s = ∑ i, y i * dx i := by
    simp [s, inner_eq_sum_mul]
  calc
    (∑ i, softmaxJVP x dx i) = ∑ i, y i * (dx i - s) := by
      simp [softmaxJVP, y, s]
    _ = (∑ i, y i * dx i) - (∑ i, y i) * s := by
      rw [Finset.sum_mul, ← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun i _ => mul_sub _ _ _
    _ = 0 := by rw [hy, hs]; ring

/-- Coordinatewise VJP/JVP bound in the infinity norm.

If every upstream coordinate has magnitude at most `G`, then every softmax input gradient has
magnitude at most `2G`. The estimate is dimension-free because softmax weights are nonnegative and
sum to one. It is conservative, and a tighter certificate can keep the factor
`2 * y_i * (1 - y_i)` for each coordinate.
-/
theorem Vec.abs_softmaxJVP_le_two_mul {n : Nat} (x dx : Vec (Nat.succ n)) (G : Real)
    (hdx : ∀ i, abs (dx i) ≤ G) (i : Fin (Nat.succ n)) :
    abs (softmaxJVP x dx i) ≤ 2 * G := by
  classical
  let y : Vec (Nat.succ n) := softmax x
  let s : Real := inner ℝ y dx
  have hyPos : ∀ j, 0 < y j := by
    intro j
    simp only [y, softmax, PiLp.toLp_apply]
    exact div_pos (Real.exp_pos (x j)) (sumExp_pos x)
  have hySum : (∑ j, y j) = 1 := sum_softmax x
  have hs : s = ∑ j, y j * dx j := by
    simp [s, inner_eq_sum_mul]
  have hsAbs : abs s ≤ G := by
    rw [hs]
    calc
      abs (∑ j, y j * dx j) ≤ ∑ j, abs (y j * dx j) :=
        Finset.abs_sum_le_sum_abs _ _
      _ = ∑ j, y j * abs (dx j) := by
        refine Finset.sum_congr rfl ?_
        intro j _
        rw [abs_mul, abs_of_pos (hyPos j)]
      _ ≤ ∑ j, y j * G := by
        refine Finset.sum_le_sum ?_
        intro j _
        exact mul_le_mul_of_nonneg_left (hdx j) (le_of_lt (hyPos j))
      _ = G := by rw [← Finset.sum_mul, hySum, one_mul]
  have hyLeOne : y i ≤ 1 := by
    calc
      y i ≤ ∑ j, y j :=
        Finset.single_le_sum (fun j _ => le_of_lt (hyPos j)) (Finset.mem_univ i)
      _ = 1 := hySum
  have hdiff : abs (dx i - s) ≤ 2 * G := by
    calc
      abs (dx i - s) ≤ abs (dx i) + abs s := abs_sub _ _
      _ ≤ G + G := add_le_add (hdx i) hsAbs
      _ = 2 * G := by ring
  simp only [softmaxJVP, PiLp.toLp_apply]
  change abs (y i * (dx i - s)) ≤ 2 * G
  rw [abs_mul, abs_of_pos (hyPos i)]
  calc
    y i * abs (dx i - s) ≤ 1 * abs (dx i - s) :=
      mul_le_mul_of_nonneg_right hyLeOne (abs_nonneg _)
    _ ≤ 1 * (2 * G) := mul_le_mul_of_nonneg_left hdiff zero_le_one
    _ = 2 * G := one_mul _

end

end TorchLean
