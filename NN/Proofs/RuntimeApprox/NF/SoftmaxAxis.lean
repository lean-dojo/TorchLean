/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Analysis.Softmax
public import NN.Proofs.Autograd.FDeriv.Softmax
public import NN.Proofs.RuntimeApprox.NF.ShapeOps
public import NN.Spec.Layers.Attention
public import NN.Proofs.RuntimeApprox.NF.Ops.Elementwise.Binary
public import NN.Proofs.RuntimeApprox.NF.Ops.Elementwise.SafeDivSigmoid
public import NN.Proofs.RuntimeApprox.NF.Ops.Elementwise.Unary
public import NN.Proofs.RuntimeApprox.NF.Ops.Sum

/-!
# Axis softmax

This file collects the mathematical facts needed by numerical bounds for the coupled, vector-valued
softmax used in attention. It is deliberately separate from the older NF scalar logistic helper:
axis softmax has a shared denominator and a dense Jacobian, while logistic acts independently on
each tensor entry.

The stable spec implementation is `Activation.softmaxVecSpec`. `Proofs.Analysis.Softmax` proves
that its entries are positive, sum to one, and lie in `[0,1]`. The analytic derivative is
`Proofs.Autograd.softmaxJvp`; its Jacobian is self-adjoint, so the same formula implements the VJP.
The theorem below adds the conservation law needed for backward error analysis: every softmax JVP
has coordinate sum zero.

References:

* A. Griewank and A. Walther, *Evaluating Derivatives*, 2nd ed., 2008, for forward/reverse
  differentiation of coupled maps.
* A. A. Baydin et al., "Automatic Differentiation in Machine Learning: a Survey," JMLR 2018.
* PyTorch `torch.nn.functional.softmax` documentation for the runtime axis convention.
-/

@[expose] public section

namespace Proofs
namespace RuntimeApprox
namespace AxisSoftmax

open scoped BigOperators

open Proofs.Autograd
open Spec TorchLean
open TorchLean TorchLean.Tensor
open NN.MLTheory.Robustness.Spec
open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq

noncomputable section

variable {β : Radix} {fexp : ℤ -> ℤ} [ValidExp fexp]
variable {rnd : ℝ -> ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

/-! ## Maximum and max shift -/

omit [ValidRndToNearest rnd] in
/-- Forgetting the `NF` format commutes with the nonempty-vector maximum exactly.

`NF.max` only selects one operand; it performs no arithmetic and therefore introduces no rounding
error. The proof uses one fold homomorphism rather than repeating a coordinate induction in every
stable normalization operator.
-/
theorem toSpec_maxVecSpec {n : Nat} (xR : Tensor R [Nat.succ n]) :
    NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)
        (Tensor.item (Activation.maxVecSpec xR)) =
      Tensor.item
        (Activation.maxVecSpec
          (TorchLean.Tensor.map (NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR)) := by
  let first : Fin (Nat.succ n) := ⟨0, Nat.succ_pos n⟩
  let runtimeValue : Fin (Nat.succ n) → R := fun i => xR.getScalar i
  let realValue : Fin (Nat.succ n) → ℝ := fun i =>
    NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd) (runtimeValue i)
  have hfold := List.foldl_hom
    (f := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
    (g₁ := fun acc i => max acc (runtimeValue i))
    (g₂ := fun acc i => max acc (realValue i))
    (l := List.finRange (Nat.succ n))
    (init := runtimeValue first)
    (fun acc i => by
      simpa [realValue] using
        (NFBackend.toSpec_max (β := β) (fexp := fexp) (rnd := rnd)
          acc (runtimeValue i)).symm)
  simpa [Activation.maxVecSpec, runtimeValue, realValue, first] using hfold.symm

omit [ValidRndToNearest rnd] in
/-- The maximum of a rounded vector approximates the real maximum with the same infinity-norm
budget as the vector itself. No additional ULP term appears because maximum is a selection.
-/
theorem approxTensor_maxVecSpec {n : Nat}
    {xS : SpecTensor [Nat.succ n]}
    {xR : Tensor R [Nat.succ n]} {eps : ℝ}
    (hx : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps) :
    approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Activation.maxVecSpec xS) (Activation.maxVecSpec xR) eps := by
  classical
  let xHat : SpecTensor [Nat.succ n] :=
    TorchLean.Tensor.map (NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) xR
  let mS : ℝ := Tensor.item (Activation.maxVecSpec xS)
  let mHat : ℝ := Tensor.item (Activation.maxVecSpec xHat)
  have hpoint :
      ∀ i, |TorchLean.Tensor.getScalar xHat i - TorchLean.Tensor.getScalar xS i| <= eps := by
    intro i
    have hi := approxTensor_dim_get (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) hx i
    have hi' := (approxTensor_scalar_item_iff (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))).mp hi
    simpa [xHat, Tensor.getScalar, Spec.get] using hi'
  rcases Proofs.exists_getScalar_eq_maxVecSpec xS with ⟨iS, hiS⟩
  rcases Proofs.exists_getScalar_eq_maxVecSpec xHat with ⟨iHat, hiHat⟩
  have hmS_le : mS <= mHat + eps := by
    have hcoord := (abs_sub_le_iff.mp (hpoint iS)).2
    have hmax := Proofs.getScalar_le_maxVecSpec xHat iS
    dsimp [mS, mHat] at *
    linarith
  have hmHat_le : mHat <= mS + eps := by
    have hcoord := (abs_sub_le_iff.mp (hpoint iHat)).1
    have hmax := Proofs.getScalar_le_maxVecSpec xS iHat
    dsimp [mS, mHat] at *
    linarith
  have hmaxError : |mHat - mS| <= eps := by
    rw [abs_le]
    constructor <;> linarith
  have hbridge := toSpec_maxVecSpec (β := β) (fexp := fexp) (rnd := rnd) xR
  rw [← Tensor.scalar_item (Activation.maxVecSpec xS),
    ← Tensor.scalar_item (Activation.maxVecSpec xR)]
  apply (approxTensor_scalar_iff (α := R)
    (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))).mpr
  rw [hbridge]
  exact hmaxError

/-! ## Rounded stable softmax -/

/-- Error after subtracting the rounded maximum from every logit. -/
def shiftErrorBound {n : Nat} (eps : ℝ)
    (xR : Tensor R [Nat.succ n]) : ℝ :=
  let maxR := Activation.maxVecSpec xR
  let maxRepR : Tensor R [Nat.succ n] := Tensor.replicate maxR
  linfNorm (NFBackend.subBoundTensor (β := β) (fexp := fexp) eps eps xR maxRepR)

/-- Error after exponentiating the max-shifted logits. -/
def exponentErrorBound {n : Nat} (eps : ℝ)
    (xR : Tensor R [Nat.succ n]) : ℝ :=
  let maxRepR : Tensor R [Nat.succ n] :=
    Tensor.replicate (Activation.maxVecSpec xR)
  let shiftedR := subSpec xR maxRepR
  linfNorm
    (NFBackend.expBoundTensor (β := β) (fexp := fexp)
      (shiftErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR) shiftedR)

/-- Error in the sequentially rounded denominator reduction. -/
def denominatorErrorBound {n : Nat} (eps : ℝ)
    (xR : Tensor R [Nat.succ n]) : ℝ :=
  NFBackend.sumBound (β := β) (fexp := fexp) (rnd := rnd)
    (exponentErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR)
    (Activation.maxShiftedExpVecSpec xR)

/-- Per-coordinate output budget for stable softmax.

The exact denominator is at least one. The checker must additionally establish
`denominatorErrorBound eps xR < 1`; this prevents the rounded denominator from crossing zero and
turns the division condition into an explicit, checkable certificate obligation.
-/
def softmaxBoundTensor {n : Nat} (eps : ℝ)
    (xR : Tensor R [Nat.succ n]) : SpecTensor [Nat.succ n] :=
  let exR := Activation.maxShiftedExpVecSpec xR
  let denomR : R := sumSpec exR
  let epsNum := exponentErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR
  let epsDenom := denominatorErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR
  TorchLean.Tensor.map
    (fun numR => NFBackend.divPosErrorBound (β := β) (fexp := fexp)
      1 epsNum epsDenom
      (NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd) numR)
      (NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd) denomR))
    exR

/-- Infinity-norm forward-error budget for stable vector softmax. -/
def softmaxErrorBound {n : Nat} (eps : ℝ)
    (xR : Tensor R [Nat.succ n]) : ℝ :=
  linfNorm (softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd) eps xR)

/-- The max-shifted `NF` implementation approximates real vector softmax.

Unlike a blanket continuity statement, the theorem follows the executable stages: maximum,
subtraction, exponential, sequential sum, and division. The sole side condition is the numerical
certificate check that the denominator error remains below its proved real lower bound `1`.
-/
theorem approxTensor_softmaxVecSpec {n : Nat}
    {xS : SpecTensor [Nat.succ n]}
    {xR : Tensor R [Nat.succ n]} {eps : ℝ}
    (hx : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps)
    (hdenom : denominatorErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR < 1) :
    approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Activation.softmaxVecSpec xS) (Activation.softmaxVecSpec xR)
      (softmaxErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR) := by
  classical
  let maxS := Activation.maxVecSpec xS
  let maxR := Activation.maxVecSpec xR
  let maxRepS : SpecTensor [Nat.succ n] := Tensor.replicate maxS
  let maxRepR : Tensor R [Nat.succ n] := Tensor.replicate maxR
  let shiftedS := subSpec xS maxRepS
  let shiftedR := subSpec xR maxRepR
  let exS := Activation.maxShiftedExpVecSpec xS
  let exR := Activation.maxShiftedExpVecSpec xR
  let denomS : ℝ := sumSpec exS
  let denomR : R := sumSpec exR
  let epsShift := shiftErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR
  let epsNum := exponentErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR
  let epsDenom := denominatorErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR
  let outBound := softmaxErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR

  have hmax := approxTensor_maxVecSpec (β := β) (fexp := fexp) (rnd := rnd) hx
  have hmaxRep : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      maxRepS maxRepR eps := by
    simpa [maxRepS, maxRepR, maxS, maxR] using
      (NFBackend.approxTensor_replicate (β := β) (fexp := fexp) (rnd := rnd)
        (s := .dim (Nat.succ n) .scalar) hmax)
  have hshift : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      shiftedS shiftedR epsShift := by
    have h := NFBackend.approxTensor_sub_spec (β := β) (fexp := fexp) (rnd := rnd) hx hmaxRep
    simpa [shiftedS, shiftedR, epsShift, shiftErrorBound, maxRepR, maxR] using h
  have hexp : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      exS exR epsNum := by
    have h := NFBackend.approxTensor_exp_spec (β := β) (fexp := fexp) (rnd := rnd) hshift
    simpa [exS, exR, shiftedS, shiftedR, epsNum, exponentErrorBound,
      Activation.maxShiftedExpVecSpec, maxRepS, maxRepR, maxS, maxR] using h
  have hsum : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Tensor.scalar denomS) (Tensor.scalar denomR) epsDenom := by
    have h := NFBackend.approxTensor_sum_spec (β := β) (fexp := fexp) (rnd := rnd) hexp
    simpa [denomS, denomR, epsDenom, denominatorErrorBound, epsNum, exR] using h
  have hdenomLower : (1 : ℝ) ≤ denomS := by
    simpa [denomS, exS] using (Proofs.softmax_shift_denom_bounds xS).1
  have hbudget : epsDenom < (1 : ℝ) := by
    simpa [epsDenom] using hdenom
  have hOutNonneg : 0 ≤ outBound := by
    simpa [outBound, softmaxErrorBound] using
      (linf_norm_nonneg
        (t := softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd) eps xR))

  refine approxTensor_dim_of_forall
    (xS := Activation.softmaxVecSpec xS)
    (xR := Activation.softmaxVecSpec xR)
    (eps := outBound) hOutNonneg ?_
  intro i
  let numS := (exS.unstack i).item
  let numR := (exR.unstack i).item
  have hnumI := approxTensor_dim_get (α := R)
    (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) hexp i
  have hnumScalar := (approxTensor_scalar_item_iff (α := R)
    (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))).mp hnumI
  have hdenomScalar := (approxTensor_scalar_iff (α := R)
    (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))).mp hsum
  have hdiv := NFBackend.approx_div_nf_of_pos_lb
    (β := β) (fexp := fexp) (rnd := rnd) (η := (1 : ℝ))
    hdenomLower hbudget hnumScalar hdenomScalar
  have hcoord := linf_norm_le_get_dim
    (t := softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd) eps xR) i
  have hdivBound :
      NFBackend.divPosErrorBound (β := β) (fexp := fexp) 1 epsNum epsDenom
          (NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd) numR)
          (NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd) denomR) ≤
        outBound := by
    refine le_trans (le_abs_self _) ?_
    simpa [outBound, softmaxErrorBound, softmaxBoundTensor, exR, denomR,
      epsNum, epsDenom, numR, linfNorm, RuntimeApprox.linfNorm,
      tensorLinfNorm, Numerics.MathFunctions.abs] using hcoord
  have hscalarOut : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Tensor.scalar (numS / denomS)) (Tensor.scalar (numR / denomR)) outBound :=
    (approxTensor_scalar_iff (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))).mpr
        (le_trans hdiv hdivBound)
  have hsoftS :
      (Activation.softmaxVecSpec xS).unstack i =
        Tensor.scalar (numS / denomS) := by
    apply TorchLean.Tensor.Internal.Rep.ext
    intro coordinate
    cases coordinate
    simp [Activation.softmaxVecSpec, exS, denomS, numS,
      TorchLean.Tensor.divSpec, TorchLean.Tensor.replicate,
      TorchLean.Tensor.map2Spec, Tensor.item, Tensor.unstack, Tensor.scalar,
      TorchLean.Tensor.Internal.Rep.unstack_apply]
  have hsoftR :
      (Activation.softmaxVecSpec xR).unstack i =
        Tensor.scalar (numR / denomR) := by
    apply TorchLean.Tensor.Internal.Rep.ext
    intro coordinate
    cases coordinate
    simp [Activation.softmaxVecSpec, exR, denomR, numR,
      TorchLean.Tensor.divSpec, TorchLean.Tensor.replicate,
      TorchLean.Tensor.map2Spec, Tensor.item, Tensor.unstack, Tensor.scalar,
      TorchLean.Tensor.Internal.Rep.unstack_apply]
  rw [hsoftS, hsoftR]
  exact hscalarOut

/-- Row-wise error tensor for axis-`1` softmax on a matrix. -/
def softmaxRowsBoundTensor {m n : Nat} (eps : ℝ)
    (xR : Tensor R [m, Nat.succ n]) :
    SpecTensor [m, Nat.succ n] :=
  Tensor.dim (fun i =>
    softmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd) eps (xR.unstack i))

/-- Global infinity-norm budget for row-wise axis-`1` softmax. -/
def softmaxRowsErrorBound {m n : Nat} (eps : ℝ)
    (xR : Tensor R [m, Nat.succ n]) : ℝ :=
  linfNorm (softmaxRowsBoundTensor (β := β) (fexp := fexp) (rnd := rnd) eps xR)

/-- Matrix-level stable softmax theorem, obtained by applying the vector theorem independently to
each row.

The denominator obligation remains row-specific: a certificate may accept well-conditioned rows
without replacing them by a single pessimistic analytic assumption. The output uses one global
infinity-norm budget because that is the contract consumed by matrix multiplication and graph
composition.
-/
theorem approxTensor_softmaxRowsSpec {m n : Nat}
    {xS : SpecTensor [m, Nat.succ n]}
    {xR : Tensor R [m, Nat.succ n]} {eps : ℝ}
    (hx : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps)
    (hdenom : ∀ i : Fin m,
      denominatorErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps (Spec.get xR i) < 1) :
    approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Activation.softmaxSpec 1 xS) (Activation.softmaxSpec 1 xR)
      (softmaxRowsErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR) := by
  classical
  change approxTensor
    (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
    (Activation.Internal.softmaxInnermostSpec xS)
    (Activation.Internal.softmaxInnermostSpec xR)
    (softmaxRowsErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR)
  let bound := softmaxRowsErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps xR
  have hbound : 0 ≤ bound := by
    simpa [bound, softmaxRowsErrorBound] using
      (linf_norm_nonneg
        (t := softmaxRowsBoundTensor (β := β) (fexp := fexp) (rnd := rnd) eps xR))
  refine approxTensor_dim_of_forall
    (xS := Activation.Internal.softmaxInnermostSpec xS)
    (xR := Activation.Internal.softmaxInnermostSpec xR)
    (eps := bound) hbound ?_
  intro i
  have hrow := approxTensor_dim_get (α := R)
    (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) hx i
  have hsoft := approxTensor_softmaxVecSpec (β := β) (fexp := fexp) (rnd := rnd) hrow
    (by simpa [Spec.get] using hdenom i)
  have hrowLe :
      softmaxErrorBound (β := β) (fexp := fexp) (rnd := rnd) eps (xR.unstack i) ≤ bound := by
    have h := linf_norm_le_get_dim
      (t := softmaxRowsBoundTensor (β := β) (fexp := fexp) (rnd := rnd) eps xR) i
    simpa [bound, softmaxRowsErrorBound, softmaxRowsBoundTensor,
      softmaxErrorBound] using h
  simpa using approxTensor_mono hsoft hrowLe

/-! ## Exact hard-masked softmax -/

/-- Stable softmax numerators for a row whose allowed maximum is already known.

Blocked coordinates are set to literal zero after exponentiation. This expression is equivalent to
the `some rowMax` branch of `Spec.hardMaskedSoftmaxVecSpec` and never introduces a finite masking
sentinel.
-/
def hardMaskedNumerators {α : Type} [TorchLean.Storage α] [Context α] {n : Nat}
    (scores : Tensor α [n])
    (mask : Tensor Bool [n]) (rowMax : α) :
    Tensor α [n] :=
  let maxRep : Tensor α [n] := Tensor.replicate (Tensor.scalar rowMax)
  let exponentials := expSpec (subSpec scores maxRep)
  map2Spec
    (fun value allowed => if allowed then value else 0) exponentials mask

/-- The staged numerator computation equals the fused expression used by the public spec. -/
theorem hardMaskedNumerators_eq_fused {α : Type} [TorchLean.Storage α] [Context α] {n : Nat}
    (scores : Tensor α [n])
    (mask : Tensor Bool [n]) (rowMax : α) :
    hardMaskedNumerators scores mask rowMax =
      map2Spec
        (fun score allowed => if allowed then Numerics.MathFunctions.exp (score - rowMax) else 0)
        scores mask := by
  apply TorchLean.Tensor.Internal.Rep.ext
  intro coordinate
  rcases coordinate with ⟨i, coordinate⟩
  cases coordinate
  simp only [hardMaskedNumerators, Tensor.replicate, map2Spec, expSpec, subSpec, mapSpec,
    Tensor.scalar, TorchLean.Tensor.Internal.Rep.zipWith_apply]
  by_cases hallowed : mask (i, PUnit.unit) = true
  · simp only [hallowed, ite_true]
    change
      TorchLean.Tensor.Internal.Rep.map Numerics.MathFunctions.exp
          (TorchLean.Tensor.Internal.Rep.zipWith (· - ·) scores
            (TorchLean.Tensor.Internal.Rep.stack fun _ =>
              TorchLean.Tensor.Internal.Rep.ofFn fun _ => rowMax))
          (i, PUnit.unit) =
        Numerics.MathFunctions.exp (scores (i, PUnit.unit) - rowMax)
    simp
  · simp [hallowed]

/-- Error after subtracting an approximate allowed-row maximum from every score. -/
def hardMaskedShiftError {n : Nat} (epsScores epsMax : ℝ)
    (scoresR : Tensor R [n]) (rowMaxR : R) : ℝ :=
  let maxRepR : Tensor R [n] := Tensor.replicate (Tensor.scalar rowMaxR)
  linfNorm
    (NFBackend.subBoundTensor (β := β) (fexp := fexp)
      epsScores epsMax scoresR maxRepR)

/-- Error in the hard-masked numerator vector; applying the mask adds no rounding error. -/
def hardMaskedNumeratorError {n : Nat} (epsScores epsMax : ℝ)
    (scoresR : Tensor R [n])
    (_mask : Tensor Bool [n]) (rowMaxR : R) : ℝ :=
  let maxRepR : Tensor R [n] := Tensor.replicate (Tensor.scalar rowMaxR)
  let shiftedR := subSpec scoresR maxRepR
  let epsShift := hardMaskedShiftError (β := β) (fexp := fexp) (rnd := rnd)
    epsScores epsMax scoresR rowMaxR
  linfNorm
    (NFBackend.expBoundTensor (β := β) (fexp := fexp) epsShift shiftedR)

/-- Error in the sequentially rounded sum of the allowed numerators. -/
def hardMaskedDenominatorError {n : Nat} (epsScores epsMax : ℝ)
    (scoresR : Tensor R [n])
    (mask : Tensor Bool [n]) (rowMaxR : R) : ℝ :=
  NFBackend.sumBound (β := β) (fexp := fexp) (rnd := rnd)
    (hardMaskedNumeratorError (β := β) (fexp := fexp) (rnd := rnd)
      epsScores epsMax scoresR mask rowMaxR)
    (hardMaskedNumerators scoresR mask rowMaxR)

/-- Final per-coordinate budget for a nonempty hard-masked softmax row. -/
def hardMaskedSoftmaxBoundTensor {n : Nat} (η epsScores epsMax : ℝ)
    (scoresR : Tensor R [n])
    (mask : Tensor Bool [n]) (rowMaxR : R) :
    SpecTensor [n] :=
  let numeratorsR := hardMaskedNumerators scoresR mask rowMaxR
  let denominatorR : R := sumSpec numeratorsR
  let denominatorRepR : Tensor R [n] :=
    Tensor.replicate (Tensor.scalar denominatorR)
  NFBackend.divPosBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
    η
    (hardMaskedNumeratorError (β := β) (fexp := fexp) (rnd := rnd)
      epsScores epsMax scoresR mask rowMaxR)
    (hardMaskedDenominatorError (β := β) (fexp := fexp) (rnd := rnd)
      epsScores epsMax scoresR mask rowMaxR)
    numeratorsR denominatorRepR

/-- Numerical certificate for the nonempty branch of hard-masked vector softmax.

`hmaxS` and `hmaxR` identify the selected allowed-row maxima. The theorem does not trust those
values blindly: `hmax` must relate them numerically, and `hdenomLower` supplies the exact positive
lower bound used by division. For the canonical selected maximum this lower bound is `1`, because
one allowed shifted score is zero and contributes `exp 0 = 1`.
-/
theorem approxTensor_hardMaskedSoftmaxVecSpec_of_max {n : Nat}
    {scoresS : SpecTensor [n]}
    {scoresR : Tensor R [n]}
    (mask : Tensor Bool [n])
    {rowMaxS : ℝ} {rowMaxR : R} {epsScores epsMax η : ℝ}
    (hscores : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      scoresS scoresR epsScores)
    (hmax : abs
      (NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd) rowMaxR - rowMaxS) ≤ epsMax)
    (hmaxS : Spec.hardMaskedMax? scoresS mask = some rowMaxS)
    (hmaxR : Spec.hardMaskedMax? scoresR mask = some rowMaxR)
    (hdenomLower : η ≤ sumSpec (hardMaskedNumerators scoresS mask rowMaxS))
    (hdenomMargin :
      hardMaskedDenominatorError (β := β) (fexp := fexp) (rnd := rnd)
        epsScores epsMax scoresR mask rowMaxR < η) :
    approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Spec.hardMaskedSoftmaxVecSpec scoresS mask)
      (Spec.hardMaskedSoftmaxVecSpec scoresR mask)
      (linfNorm
        (hardMaskedSoftmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          η epsScores epsMax scoresR mask rowMaxR)) := by
  let maxS : SpecTensor .scalar := Tensor.scalar rowMaxS
  let maxR : Tensor R .scalar := Tensor.scalar rowMaxR
  let maxRepS : SpecTensor [n] := Tensor.replicate maxS
  let maxRepR : Tensor R [n] := Tensor.replicate maxR
  let shiftedS := subSpec scoresS maxRepS
  let shiftedR := subSpec scoresR maxRepR
  let expS := expSpec shiftedS
  let expR := expSpec shiftedR
  let numeratorsS := hardMaskedNumerators scoresS mask rowMaxS
  let numeratorsR := hardMaskedNumerators scoresR mask rowMaxR
  let denominatorS : ℝ := sumSpec numeratorsS
  let denominatorR : R := sumSpec numeratorsR
  let denominatorRepS : SpecTensor [n] :=
    Tensor.replicate (Tensor.scalar denominatorS)
  let denominatorRepR : Tensor R [n] :=
    Tensor.replicate (Tensor.scalar denominatorR)
  let epsShift := hardMaskedShiftError (β := β) (fexp := fexp) (rnd := rnd)
    epsScores epsMax scoresR rowMaxR
  let epsNumerator := hardMaskedNumeratorError (β := β) (fexp := fexp) (rnd := rnd)
    epsScores epsMax scoresR mask rowMaxR
  let epsDenominator := hardMaskedDenominatorError (β := β) (fexp := fexp) (rnd := rnd)
    epsScores epsMax scoresR mask rowMaxR

  have hmaxTensor : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      maxS maxR epsMax :=
    (approxTensor_scalar_iff (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))).2 hmax
  have hmaxRep := NFBackend.approxTensor_replicate
    (β := β) (fexp := fexp) (rnd := rnd) (s := .dim n .scalar) hmaxTensor
  have hshift := NFBackend.approxTensor_sub_spec
    (β := β) (fexp := fexp) (rnd := rnd) hscores hmaxRep
  have hexp := NFBackend.approxTensor_exp_spec
    (β := β) (fexp := fexp) (rnd := rnd) hshift
  have hnumerators := NFBackend.approxTensor_applyBoolMask
    (β := β) (fexp := fexp) (rnd := rnd) mask hexp
  have hsum := NFBackend.approxTensor_sum_spec
    (β := β) (fexp := fexp) (rnd := rnd) hnumerators
  have hdenominatorRep := NFBackend.approxTensor_replicate
    (β := β) (fexp := fexp) (rnd := rnd)
    (s := .dim n .scalar) hsum
  have hdenominatorDomain :
      Tensor.Forall (fun z : ℝ => η ≤ z) denominatorRepS := by
    exact Tensor.forall_replicate (by simpa [denominatorS, numeratorsS] using hdenomLower)
  have hout := NFBackend.approxTensor_div_spec_of_pos_lb
    (β := β) (fexp := fexp) (rnd := rnd) η
    hnumerators hdenominatorRep hdenominatorDomain hdenomMargin
  simp only [Spec.hardMaskedSoftmaxVecSpec, hmaxS, hmaxR]
  rw [← hardMaskedNumerators_eq_fused scoresS mask rowMaxS,
    ← hardMaskedNumerators_eq_fused scoresR mask rowMaxR]
  simpa [
    hardMaskedSoftmaxBoundTensor, hardMaskedNumerators,
    hardMaskedShiftError, hardMaskedNumeratorError, hardMaskedDenominatorError,
    maxS, maxR, maxRepS, maxRepR, shiftedS, shiftedR, expS, expR,
    numeratorsS, numeratorsR, denominatorS, denominatorR,
    denominatorRepS, denominatorRepR, epsShift, epsNumerator, epsDenominator] using hout

/-- If both semantics find no allowed key, hard-masked softmax agrees exactly on the zero row. -/
theorem approxTensor_hardMaskedSoftmaxVecSpec_allBlocked {n : Nat}
    (scoresS : SpecTensor [n])
    (scoresR : Tensor R [n])
    (mask : Tensor Bool [n])
    (hmaxS : Spec.hardMaskedMax? scoresS mask = none)
    (hmaxR : Spec.hardMaskedMax? scoresR mask = none) :
    approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Spec.hardMaskedSoftmaxVecSpec scoresS mask)
      (Spec.hardMaskedSoftmaxVecSpec scoresR mask) 0 := by
  simp only [Spec.hardMaskedSoftmaxVecSpec, hmaxS, hmaxR]
  change approxTensor (α := R)
    (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
    (Tensor.full (.dim n .scalar) (0 : ℝ)) (Tensor.full (.dim n .scalar) (0 : R)) 0
  exact NFBackend.approxTensor_full_zero (β := β) (fexp := fexp) (rnd := rnd)
    (s := .dim n .scalar)

/-- Checkable evidence for nonempty hard-masked softmax rows.

The data fields record the maxima and margins used by execution; the proposition fields establish
that they describe the exact and rounded rows. Keeping this evidence together prevents a caller
from accidentally pairing a denominator check with a different score matrix or mask.
-/
structure HardMaskedRowsEvidence {m n : Nat}
    (scoresS : SpecTensor [m, n])
    (scoresR : Tensor R [m, n])
    (mask : Tensor Bool [m, n])
    (epsScores : ℝ) where
  rowMaxS : Fin m → ℝ
  rowMaxR : Fin m → R
  epsMax : Fin m → ℝ
  eta : Fin m → ℝ
  maxApprox : ∀ i,
    abs (NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd) (rowMaxR i) -
      rowMaxS i) ≤ epsMax i
  specMax : ∀ i,
    Spec.hardMaskedMax? (Spec.get scoresS i) (Spec.get mask i) = some (rowMaxS i)
  runtimeMax : ∀ i,
    Spec.hardMaskedMax? (Spec.get scoresR i) (Spec.get mask i) = some (rowMaxR i)
  denominatorLower : ∀ i,
    eta i ≤ sumSpec
      (hardMaskedNumerators (Spec.get scoresS i) (Spec.get mask i) (rowMaxS i))
  denominatorMargin : ∀ i,
    hardMaskedDenominatorError (β := β) (fexp := fexp) (rnd := rnd)
      epsScores (epsMax i) (Spec.get scoresR i) (Spec.get mask i) (rowMaxR i) < eta i

/-- Rowwise hard-masked softmax bounds with one independently certified maximum per row. -/
def hardMaskedRowsBoundTensor {m n : Nat}
    (η : Fin m → ℝ) (epsScores : ℝ)
    (scoresR : Tensor R [m, n])
    (mask : Tensor Bool [m, n])
    (rowMaxR : Fin m → R) (epsMax : Fin m → ℝ) :
    SpecTensor [m, n] :=
  Tensor.dim (fun i =>
    hardMaskedSoftmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      (η i) epsScores (epsMax i) (scoresR.unstack i) (mask.unstack i) (rowMaxR i))

/-- Matrix-level hard-masked softmax when every row has at least one allowed coordinate.

The selected maxima and denominator checks remain row-local. This matches causal attention, where
row `i` always admits key `i`, and avoids replacing all rows by the worst intermediate scale before
the final infinity norm is taken.
-/
theorem approxTensor_hardMaskedSoftmaxRowsSpec_of_max {m n : Nat}
    {scoresS : SpecTensor [m, n]}
    {scoresR : Tensor R [m, n]}
    (mask : Tensor Bool [m, n])
    {epsScores : ℝ}
    (hscores : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      scoresS scoresR epsScores)
    (evidence : HardMaskedRowsEvidence (β := β) (fexp := fexp) (rnd := rnd)
      scoresS scoresR mask epsScores) :
    approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Spec.hardMaskedSoftmaxSpec scoresS mask)
      (Spec.hardMaskedSoftmaxSpec scoresR mask)
      (linfNorm
        (hardMaskedRowsBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          evidence.eta epsScores scoresR mask evidence.rowMaxR evidence.epsMax)) := by
  let bound := linfNorm
    (hardMaskedRowsBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
      evidence.eta epsScores scoresR mask evidence.rowMaxR evidence.epsMax)
  have hbound : 0 ≤ bound := by
    simpa [bound] using
      (linf_norm_nonneg
        (t := hardMaskedRowsBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
          evidence.eta epsScores scoresR mask evidence.rowMaxR evidence.epsMax))
  refine approxTensor_dim_of_forall
    (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
    (xS := Spec.hardMaskedSoftmaxSpec scoresS mask)
    (xR := Spec.hardMaskedSoftmaxSpec scoresR mask)
    (eps := bound) hbound ?_
  intro i
  have hscoresI := approxTensor_dim_get (α := R)
    (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) hscores i
  have hrow := approxTensor_hardMaskedSoftmaxVecSpec_of_max
    (β := β) (fexp := fexp) (rnd := rnd)
    (mask.unstack i) hscoresI (evidence.maxApprox i)
    (by simpa [Spec.get] using evidence.specMax i)
    (by simpa [Spec.get] using evidence.runtimeMax i)
    (by simpa [Spec.get] using evidence.denominatorLower i)
    (by simpa [Spec.get] using evidence.denominatorMargin i)
  have hrowLe :
      linfNorm
          (hardMaskedSoftmaxBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
            (evidence.eta i) epsScores (evidence.epsMax i)
            (scoresR.unstack i) (mask.unstack i) (evidence.rowMaxR i)) ≤
        bound := by
    have h := linf_norm_le_get_dim
      (t := hardMaskedRowsBoundTensor (β := β) (fexp := fexp) (rnd := rnd)
        evidence.eta epsScores scoresR mask evidence.rowMaxR evidence.epsMax) i
    simpa [bound, hardMaskedRowsBoundTensor] using h
  simpa [Spec.hardMaskedSoftmaxSpec] using approxTensor_mono hrow hrowLe

/-! ## Rounded softmax backward -/

/-- Error in the `dY * softmax(x)` product used by the softmax VJP. -/
def vjpProductError {n : Nat} (epsDY epsY : ℝ)
    (dYR yR : Tensor R [Nat.succ n]) : ℝ :=
  linfNorm (NFBackend.mulBoundTensor (β := β) (fexp := fexp) epsDY epsY dYR yR)

/-- Error in the rounded dot product `sum (dY * softmax(x))`. -/
def vjpDotError {n : Nat} (epsDY epsY : ℝ)
    (dYR yR : Tensor R [Nat.succ n]) : ℝ :=
  let productR := mulSpec dYR yR
  NFBackend.sumBound (β := β) (fexp := fexp) (rnd := rnd)
    (vjpProductError (β := β) (fexp := fexp) epsDY epsY dYR yR) productR

/-- Error after subtracting the replicated rounded softmax dot product from `dY`. -/
def vjpCenteredError {n : Nat} (epsDY epsY : ℝ)
    (dYR yR : Tensor R [Nat.succ n]) : ℝ :=
  let dotR : R := sumSpec (mulSpec dYR yR)
  let dotRepR : Tensor R [Nat.succ n] := Tensor.replicate (Tensor.scalar dotR)
  linfNorm (NFBackend.subBoundTensor (β := β) (fexp := fexp)
    epsDY (vjpDotError (β := β) (fexp := fexp) (rnd := rnd) epsDY epsY dYR yR)
    dYR dotRepR)

/-- End-to-end infinity-norm budget for the rounded softmax VJP. -/
def softmaxVjpErrorBound {n : Nat} (epsDY epsY : ℝ)
    (dYR yR : Tensor R [Nat.succ n]) : ℝ :=
  let dotR : R := sumSpec (mulSpec dYR yR)
  let centeredR := subSpec dYR
    (Tensor.replicate (Tensor.scalar dotR) : Tensor R [Nat.succ n])
  linfNorm (NFBackend.mulBoundTensor (β := β) (fexp := fexp)
    epsY (vjpCenteredError (β := β) (fexp := fexp) (rnd := rnd) epsDY epsY dYR yR)
    yR centeredR)

/-- Rounded VJP theorem for any already-certified softmax weight vector.

This formulation is shared by ordinary and hard-masked softmax. In the masked case blocked weights
are exactly zero, so the common formula also returns exactly zero gradient at blocked logits; no
separate finite-sentinel derivative rule is required.
-/
theorem approxTensor_softmaxBackwardFromWeightsVecSpec {n : Nat}
    {yS dYS : SpecTensor [Nat.succ n]}
    {yR dYR : Tensor R [Nat.succ n]} {epsY epsDY : ℝ}
    (hy : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) yS yR epsY)
    (hdY : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) dYS dYR epsDY) :
    approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Spec.softmaxBackwardFromWeightsSpec yS dYS)
      (Spec.softmaxBackwardFromWeightsSpec yR dYR)
      (softmaxVjpErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        epsDY epsY dYR yR) := by
  let productS := mulSpec dYS yS
  let productR := mulSpec dYR yR
  let epsProduct := vjpProductError (β := β) (fexp := fexp) epsDY epsY dYR yR
  let dotS : ℝ := sumSpec productS
  let dotR : R := sumSpec productR
  let epsDot := vjpDotError (β := β) (fexp := fexp) (rnd := rnd) epsDY epsY dYR yR
  let dotRepS : SpecTensor [Nat.succ n] := Tensor.replicate (Tensor.scalar dotS)
  let dotRepR : Tensor R [Nat.succ n] := Tensor.replicate (Tensor.scalar dotR)
  let centeredS := subSpec dYS dotRepS
  let centeredR := subSpec dYR dotRepR
  let epsCentered :=
    vjpCenteredError (β := β) (fexp := fexp) (rnd := rnd) epsDY epsY dYR yR

  have hproduct : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      productS productR epsProduct := by
    have h := NFBackend.approxTensor_mul_spec (β := β) (fexp := fexp) (rnd := rnd) hdY hy
    simpa [productS, productR, epsProduct, vjpProductError] using h
  have hdot : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Tensor.scalar dotS) (Tensor.scalar dotR) epsDot := by
    have h := NFBackend.approxTensor_sum_spec (β := β) (fexp := fexp) (rnd := rnd) hproduct
    simpa [dotS, dotR, epsDot, vjpDotError, productR, epsProduct] using h
  have hdotRep : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      dotRepS dotRepR epsDot := by
    simpa [dotRepS, dotRepR] using
      (NFBackend.approxTensor_replicate (β := β) (fexp := fexp) (rnd := rnd)
        (s := .dim (Nat.succ n) .scalar) hdot)
  have hcentered : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      centeredS centeredR epsCentered := by
    have h := NFBackend.approxTensor_sub_spec (β := β) (fexp := fexp) (rnd := rnd) hdY hdotRep
    simpa [centeredS, centeredR, epsCentered, vjpCenteredError, dotRepR, dotR, productR,
      epsDot] using h
  have hout := NFBackend.approxTensor_mul_spec (β := β) (fexp := fexp) (rnd := rnd) hy hcentered
  simpa [Spec.softmaxBackwardFromWeightsSpec, productS, productR, dotS, dotR,
    dotRepS, dotRepR, centeredS, centeredR, epsCentered, softmaxVjpErrorBound] using hout

/-- Forward-error theorem for the executable softmax VJP.

This is the training counterpart of `approxTensor_softmaxVecSpec`. It follows the implementation's
factorization `y * (dY - sum (dY * y))`; the proof never materializes a dense Jacobian and reuses
the same rounded multiplication, reduction, replication, and subtraction contracts as ordinary
model execution.
-/
theorem approxTensor_softmaxBackwardVecSpec {n : Nat}
    {xS dYS : SpecTensor [Nat.succ n]}
    {xR dYR : Tensor R [Nat.succ n]} {epsX epsDY : ℝ}
    (hx : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsX)
    (hdY : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) dYS dYR epsDY)
    (hdenom : denominatorErrorBound (β := β) (fexp := fexp) (rnd := rnd) epsX xR < 1) :
    let yR := Activation.softmaxVecSpec xR
    approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (Activation.softmaxBackwardSpec 0 xS dYS) (Activation.softmaxBackwardSpec 0 xR dYR)
      (softmaxVjpErrorBound (β := β) (fexp := fexp) (rnd := rnd)
        epsDY (softmaxErrorBound (β := β) (fexp := fexp) (rnd := rnd) epsX xR) dYR yR) := by
  dsimp only
  change approxTensor
    (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd))
    (Activation.Internal.softmaxInnermostBackwardSpec xS dYS)
    (Activation.Internal.softmaxInnermostBackwardSpec xR dYR)
    (softmaxVjpErrorBound (β := β) (fexp := fexp) (rnd := rnd)
      epsDY (softmaxErrorBound (β := β) (fexp := fexp) (rnd := rnd) epsX xR) dYR
      (Activation.softmaxVecSpec xR))
  let yS := Activation.softmaxVecSpec xS
  let yR := Activation.softmaxVecSpec xR
  let epsY := softmaxErrorBound (β := β) (fexp := fexp) (rnd := rnd) epsX xR
  have hy : approxTensor (α := R)
      (toSpec := NFBackend.toSpec (β := β) (fexp := fexp) (rnd := rnd)) yS yR epsY := by
    simpa [yS, yR, epsY] using
      (approxTensor_softmaxVecSpec (β := β) (fexp := fexp) (rnd := rnd) hx hdenom)
  have hout := approxTensor_softmaxBackwardFromWeightsVecSpec
    (β := β) (fexp := fexp) (rnd := rnd) hy hdY
  simpa [Activation.Internal.softmaxInnermostBackwardSpec,
    Spec.softmaxBackwardFromWeightsSpec, yS, yR, epsY]
    using hout

/-- The analytic softmax on a nonempty vector sums to one. -/
theorem sum_softmaxVec {n : Nat} (x : Vec (Nat.succ n)) :
    (∑ i, softmaxVec x i) = 1 := by
  classical
  have hpos : 0 < sumExp x := by
    simpa [sumExp] using
      Finset.sum_pos (fun i (_ : i ∈ (Finset.univ : Finset (Fin (Nat.succ n)))) =>
        Real.exp_pos (x i)) Finset.univ_nonempty
  have hne : sumExp x ≠ 0 := ne_of_gt hpos
  have hne' : (∑ i, Real.exp (x i)) ≠ 0 := by
    simpa [sumExp] using hne
  simp only [softmaxVec, softmaxVecOfFun_apply]
  calc
    (∑ i, Real.exp (x i) / sumExp x) = (∑ i, Real.exp (x i)) / sumExp x := by
      simpa using
        (Finset.sum_div (s := (Finset.univ : Finset (Fin (Nat.succ n))))
          (f := fun i => Real.exp (x i)) (a := sumExp x)).symm
    _ = 1 := div_self hne'

/-- A softmax JVP is tangent to the probability simplex: its coordinates sum to zero. -/
theorem sum_softmaxJvp {n : Nat} (x dx : Vec (Nat.succ n)) :
    (∑ i, softmaxJvp x dx i) = 0 := by
  classical
  let y : Vec (Nat.succ n) := softmaxVec x
  let s : Real := dotCLM y dx
  have hy : (∑ i, y i) = 1 := sum_softmaxVec x
  have hs : s = ∑ i, y i * dx i := by
    simp [s, dotCLM_apply]
  calc
    (∑ i, softmaxJvp x dx i) = ∑ i, y i * (dx i - s) := by
      simp [softmaxJvp, y, s]
    _ = (∑ i, y i * dx i) - s * (∑ i, y i) := by
      calc
        (∑ i, y i * (dx i - s)) = ∑ i, (y i * dx i - s * y i) := by
          refine Finset.sum_congr rfl ?_
          intro i _
          ring
        _ = (∑ i, y i * dx i) - ∑ i, s * y i := by
          rw [Finset.sum_sub_distrib]
        _ = (∑ i, y i * dx i) - s * (∑ i, y i) := by
          rw [Finset.mul_sum]
    _ = 0 := by rw [hy, hs]; ring

/-- Coordinatewise VJP/JVP bound in the infinity norm.

If every upstream coordinate has magnitude at most `G`, then every softmax input gradient has
magnitude at most `2G`. The estimate is dimension-free because softmax weights are nonnegative and
sum to one. It is intentionally conservative; tighter certificates may retain the factor
`2 * y_i * (1 - y_i)` for each coordinate.
-/
theorem abs_softmaxJvp_le_two_mul {n : Nat} (x dx : Vec (Nat.succ n)) (G : Real)
    (hdx : ∀ i, abs (dx i) <= G) (i : Fin (Nat.succ n)) :
    abs (softmaxJvp x dx i) <= 2 * G := by
  classical
  let y : Vec (Nat.succ n) := softmaxVec x
  let s : Real := dotCLM y dx
  have hyPos : ∀ j, 0 < y j := by
    intro j
    simp only [y, softmaxVec, softmaxVecOfFun_apply]
    exact div_pos (Real.exp_pos (x j)) <| by
      simpa [sumExp] using
        Finset.sum_pos (fun k (_ : k ∈ (Finset.univ : Finset (Fin (Nat.succ n)))) =>
          Real.exp_pos (x k)) Finset.univ_nonempty
  have hySum : (∑ j, y j) = 1 := sum_softmaxVec x
  have hs : s = ∑ j, y j * dx j := by
    simp [s, dotCLM_apply]
  have hsAbs : abs s <= G := by
    rw [hs]
    calc
      abs (∑ j, y j * dx j) <= ∑ j, abs (y j * dx j) :=
        Finset.abs_sum_le_sum_abs _ _
      _ = ∑ j, y j * abs (dx j) := by
        refine Finset.sum_congr rfl ?_
        intro j _
        rw [abs_mul, abs_of_pos (hyPos j)]
      _ <= ∑ j, y j * G := by
        refine Finset.sum_le_sum ?_
        intro j _
        exact mul_le_mul_of_nonneg_left (hdx j) (le_of_lt (hyPos j))
      _ = G := by rw [← Finset.sum_mul, hySum, one_mul]
  have hyLeOne : y i <= 1 := by
    calc
      y i <= ∑ j, y j :=
        Finset.single_le_sum (fun j _ => le_of_lt (hyPos j)) (Finset.mem_univ i)
      _ = 1 := hySum
  have hdiff : abs (dx i - s) <= 2 * G := by
    calc
      abs (dx i - s) <= abs (dx i) + abs s := abs_sub _ _
      _ <= G + G := add_le_add (hdx i) hsAbs
      _ = 2 * G := by ring
  simp only [softmaxJvp, softmaxVecOfFun_apply]
  change abs (y i * (dx i - s)) <= 2 * G
  rw [abs_mul, abs_of_pos (hyPos i)]
  calc
    y i * abs (dx i - s) <= 1 * abs (dx i - s) :=
      mul_le_mul_of_nonneg_right hyLeOne (abs_nonneg _)
    _ <= 1 * (2 * G) := mul_le_mul_of_nonneg_left hdiff zero_le_one
    _ = 2 * G := one_mul _

end

end AxisSoftmax
end RuntimeApprox
end Proofs
