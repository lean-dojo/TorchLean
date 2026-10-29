/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.Ops.Scalar

/-!
# NF Sum Reduction Bounds

Forward-error bounds for rounded sum reductions.  The accumulator carries both the runtime value
and a proof budget, so every addition contributes the incoming element error plus one rounding term.
These bounds follow `sumSpec`'s left-to-right tensor traversal, starting at rounded zero. They
do not cover a different reduction tree or establish a native kernel's accumulation order.
-/

@[expose] public section

namespace Proofs
namespace RuntimeApprox

open Spec TorchLean
open TorchLean TorchLean.Tensor
open NN.MLTheory.Robustness.Spec

noncomputable section

namespace NFBackend

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq
open Proofs.RuntimeRoundingApprox

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

/-- Accumulate a sum and its absolute-error budget. Each addition contributes the incoming
element's error and one rounding allowance, in addition to the accumulator's existing error. -/
def sumStep (epsElem : ℝ) : (R × ℝ) → R → (R × ℝ)
  | (accR, epsAcc), xR =>
      let epsAcc' : ℝ :=
        epsAcc + epsElem +
          ulp β fexp
              (toSpec (β := β) (fexp := fexp) (rnd := rnd) accR +
                toSpec (β := β) (fexp := fexp) (rnd := rnd) xR) / 2
      (accR + xR, epsAcc')

/-- Fold over the original tensor, carrying the rounded sum and its error budget together. -/
def sumFoldState {s : Shape} (epsElem : ℝ) (st : R × ℝ) (tR : Tensor R s) : (R × ℝ) :=
  foldlSpec (sumStep (β := β) (fexp := fexp) (rnd := rnd) epsElem) st tR

/-- Absolute-error budget for `sumSpec`, given an elementwise bound `epsElem`.
It starts with a conservative
half ULP at zero and adds one rounding term per sequential addition; it is not a tight bound
for an empty tensor, whose sum is exactly zero. -/
def sumBound {s : Shape} (epsElem : ℝ) (tR : Tensor R s) : ℝ :=
  (sumFoldState (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsElem
    ((0 : R), ulp β fexp 0 / 2) tR).2

omit [ValidRndToNearest rnd] in
/-- Error bookkeeping preserves the sum's values and addition order. -/
private theorem sumFoldState_fst_eq {s : Shape} (epsElem : ℝ) (st : R × ℝ) (tR : Tensor R s) :
    (sumFoldState (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsElem st tR).1 =
      foldlSpec (· + ·) st.1 tR := by
  simpa only [sumFoldState, foldlSpec, Tensor.Internal.Rep.map_id] using
    Tensor.Internal.Rep.map_foldl tR
      (sumStep (β := β) (fexp := fexp) (rnd := rnd) epsElem) st
      id Prod.fst (· + ·) (fun _ _ => rfl)

/-- The scalar addition bound is a fold invariant, for arbitrary tensor shapes. -/
private theorem approx_sumFoldState {s : Shape} :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {accS : ℝ} {st : R × ℝ} {epsElem : ℝ},
      abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) st.1 - accS) ≤ st.2 →
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR epsElem →
        abs
            (toSpec (β := β) (fexp := fexp) (rnd := rnd)
                (sumFoldState (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsElem st xR).1 -
              foldlSpec (· + ·) accS xS) ≤
          (sumFoldState (β := β) (fexp := fexp) (rnd := rnd) (s := s) epsElem st xR).2 := by
  intro xS xR accS st epsElem hAcc hx
  exact Tensor.Internal.Rep.foldl_rel xR xS
    (sumStep (β := β) (fexp := fexp) (rnd := rnd) epsElem) (· + ·)
    (fun state reference => |toSpec state.1 - reference| ≤ state.2) st accS hAcc
    (fun index state reference hstate => by
      simpa only [sumStep] using
        approx_add_nf (β := β) (fexp := fexp) (rnd := rnd)
          (x := reference) (y := xS index) (xR := state.1) (yR := xR index)
          hstate (approxTensor_apply hx index))

/-- The rounded and real sums differ by at most `sumBound eps xR`, when the inputs are
elementwise within `eps`. -/
theorem approxTensor_sum_spec {s : Shape} :
    ∀ {xS : SpecTensor s} {xR : Tensor R s} {eps : ℝ},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps →
        approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
          (Tensor.scalar (sumSpec (α := ℝ) (s := s) xS))
          (Tensor.scalar (sumSpec (α := R) (s := s) xR))
          (sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR) := by
  intro xS xR eps hx
  -- Start from accumulator 0 with a conservative rounding bound.
  let initEps : ℝ := ulp β fexp 0 / 2
  have hAcc : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (0 : R) - (0 : ℝ)) ≤ initEps := by
    have hnonneg : 0 ≤ initEps := div_nonneg (ulp.nonneg β fexp 0) (by norm_num)
    simpa [initEps] using hnonneg
  have h :=
    approx_sumFoldState (β := β) (fexp := fexp) (rnd := rnd) (s := s)
      (xS := xS) (xR := xR) (accS := (0 : ℝ)) (st := ((0 : R), initEps)) (epsElem := eps) hAcc hx
  have hfst :
      (sumFoldState (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps ((0 : R), initEps) xR).1 =
        sumSpec (α := R) (s := s) xR := by
    simpa [sumSpec] using
      (sumFoldState_fst_eq (β := β) (fexp := fexp) (rnd := rnd) (s := s) (epsElem := eps)
        (st := ((0 : R), initEps)) (tR := xR))
  refine (approxTensor_scalar_iff (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (x := sumSpec (α := ℝ) (s := s) xS) (xR := sumSpec (α := R) (s := s) xR)
      (eps := sumBound (β := β) (fexp := fexp) (rnd := rnd) (s := s) eps xR)).2 ?_
  have h' :
      abs
          (toSpec (β := β) (fexp := fexp) (rnd := rnd) (sumSpec (α := R) (s := s) xR) -
            sumSpec (α := ℝ) (s := s) xS) ≤
        (sumFoldState (β := β) (fexp := fexp) (rnd := rnd)
          (s := s) eps ((0 : R), initEps) xR).2 := by
    simpa [hfst, sumSpec] using h
  simpa [sumBound, sumFoldState, initEps] using h'

end NFBackend

end

end RuntimeApprox
end Proofs
