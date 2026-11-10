/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.NF.Nodes -- shake: keep

/-!
# NF Shape Operators

Approximation lemmas for filling, replication, broadcasting, and Boolean masking in the
rounded-real NF model. Copied entries retain their error budgets, masked entries are exactly
zero, and none of these operations introduces arithmetic rounding.
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor

noncomputable section

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

omit [ValidExp fexp] [ValidRndToNearest rnd] in
/-- If `toSpec cR` is within `eps` of `cS`, then `Tensor.full s cR` approximates
`Tensor.full s cS` within `eps` at every shape `s`. -/
theorem ApproxTensor.full {cS : ℝ} {cR : R} {eps : ℝ}
    (h : _root_.abs (NF.toReal cR - cS) ≤ eps) :
    ∀ {s : Shape},
      ApproxTensor (α := R) (toSpec := NF.toReal)
        (Tensor.full s cS) (Tensor.full s cR) eps := by
  intro s
  induction s with
  | scalar =>
      exact (approxTensor_scalar_iff (α := R)
        (toSpec := NF.toReal)).2 h
  | dim n inner ih =>
      have hε : 0 ≤ eps := le_trans (abs_nonneg _) h
      refine approxTensor_dim_of_forall
        (toSpec := NF.toReal)
        (xS := Tensor.full (.dim n inner) cS) (xR := Tensor.full (.dim n inner) cR)
        (eps := eps) hε ?_
      intro i
      have hfillS :
          (Tensor.full (.dim n inner) cS).unstack i = Tensor.full inner cS := by
        apply Tensor.Rep.ext
        intro coordinate
        simp
      have hfillR :
          (Tensor.full (.dim n inner) cR).unstack i = Tensor.full inner cR := by
        apply Tensor.Rep.ext
        intro coordinate
        simp
      simpa [hfillS, hfillR] using ih

/-- A tensor filled with runtime one differs from exact one by at most one construction rounding.

`(1 : R)` is `NF.ofReal 1`, so the scalar bound is the generic `NF.approx_ofReal`. -/
theorem ApproxTensor.full_one :
    ∀ {s : Shape},
      ApproxTensor (α := R) (toSpec := NF.toReal)
        (Tensor.full s (1 : ℝ)) (Tensor.full s (1 : R)) (ulp β fexp (1 : ℝ) / 2) := by
  intro s
  exact full (β := β) (fexp := fexp) (rnd := rnd)
    (NF.approx_ofReal (β := β) (fexp := fexp) (rnd := rnd) 1)

/-- A tensor filled with runtime zero matches exact zero with error `0`, because zero is exactly
representable in every `NF` format. -/
theorem ApproxTensor.full_zero :
    ∀ {s : Shape},
      ApproxTensor (α := R) (toSpec := NF.toReal)
        (Tensor.full s (0 : ℝ)) (Tensor.full s (0 : R)) 0 := by
  intro s
  apply full (β := β) (fexp := fexp) (rnd := rnd)
  simp

/-- Keep a value where the mask allows it, otherwise zero. -/
private def NF.maskValue {α : Type} [Zero α] (value : α) (allowed : Bool) : α :=
  if allowed then value else 0

omit [ValidExp fexp] [ValidRndToNearest rnd] in
/-- Filling a tensor with one approximate scalar gives an approximation with the same tolerance.

Filling copies the scalar into every entry without arithmetic, so no rounding occurs and `eps`
carries over unchanged. -/
theorem ApproxTensor.full_item {s : Shape}
    {xS : Tensor ℝ .scalar} {xR : Tensor R .scalar} {eps : ℝ}
    (hx : ApproxTensor (α := R) (toSpec := NF.toReal) xS xR eps) :
    ApproxTensor (α := R) (toSpec := NF.toReal)
      (Tensor.full (α := ℝ) s xS.item)
      (Tensor.full (α := R) s xR.item)
      eps := by
  classical
  have hε : 0 ≤ eps := eps_nonneg hx
  induction s with
  | scalar =>
      simpa only [Tensor.full_item] using hx
  | dim n inner ih =>
      refine approxTensor_dim_of_forall
        (n := n) (s := inner)
        (xS := Tensor.full (α := ℝ) (.dim n inner) xS.item)
        (xR := Tensor.full (α := R) (.dim n inner) xR.item)
        (eps := eps) hε ?_
      intro i
      simpa only [Tensor.unstack_full] using ih

omit [ValidExp fexp] [ValidRndToNearest rnd] in
/-- Broadcasting preserves the approximation tolerance: each output entry copies an input. -/
theorem ApproxTensor.broadcast {s t : Shape} (dims : Fin s.rank → Fin t.rank)
    (hdims : Shape.BroadcastDims s t dims)
    {xS : Tensor ℝ s} {xR : Tensor R s} {eps : ℝ}
    (hx : ApproxTensor (α := R) (toSpec := NF.toReal) xS xR eps) :
    ApproxTensor (α := R) (toSpec := NF.toReal)
      (Tensor.broadcast xS t dims hdims) (Tensor.broadcast xR t dims hdims) eps :=
  approxTensor_of_forall (eps_nonneg hx) fun index => by
    simpa only [Tensor.broadcast_apply] using abs_sub_le hx _

/-- Applying the same Boolean mask to exact and rounded tensors preserves the error budget.

Masking selects entries without arithmetic: allowed entries are unchanged and blocked entries are
exactly zero in both semantics, so no ULP term appears. A mask with a finite negative sentinel
needs its own bound.
-/
theorem ApproxTensor.applyBoolMask {s : Shape}
    {xS : Tensor ℝ s} {xR : Tensor R s} (mask : Tensor Bool s) {eps : ℝ}
    (hx : ApproxTensor (α := R)
      (toSpec := NF.toReal) xS xR eps) :
    ApproxTensor (α := R)
      (toSpec := NF.toReal)
      (Tensor.zipWith (fun x allowed => if allowed then x else 0) xS mask)
      (Tensor.zipWith (fun x allowed => if allowed then x else 0) xR mask) eps := by
  change ApproxTensor (α := R)
    (toSpec := NF.toReal)
    (Tensor.zipWith NF.maskValue xS mask) (Tensor.zipWith NF.maskValue xR mask) eps
  induction s with
  | scalar =>
      rw [← Tensor.scalar_item xS, ← Tensor.scalar_item xR] at hx
      rw [← Tensor.scalar_item xS, ← Tensor.scalar_item xR,
        ← Tensor.scalar_item mask]
      cases mask.item with
      | false =>
          apply (approxTensor_scalar_iff (α := R)
            (toSpec := NF.toReal)).2
          change _root_.abs (NF.toReal (0 : R) - (0 : ℝ)) ≤ eps
          simpa using eps_nonneg hx
      | true =>
          simpa only [zipWith_scalar, NF.maskValue, ite_true] using hx
  | dim n inner ih =>
      have hε := eps_nonneg hx
      refine approxTensor_dim_of_forall
        (toSpec := NF.toReal)
        (xS := Tensor.zipWith NF.maskValue xS mask)
        (xR := Tensor.zipWith NF.maskValue xR mask)
        (eps := eps) hε ?_
      intro i
      have hlocal :=
        ih (mask.unstack i)
          (unstack (α := R)
            (toSpec := NF.toReal) hx i)
      rw [show
        (Tensor.zipWith NF.maskValue xS mask).unstack i =
          Tensor.zipWith NF.maskValue
            (xS.unstack i) (mask.unstack i) by
              exact (Tensor.Rep.zipWith_unstack
                NF.maskValue xS mask i).symm]
      rw [show
        (Tensor.zipWith NF.maskValue xR mask).unstack i =
          Tensor.zipWith NF.maskValue
            (xR.unstack i) (mask.unstack i) by
              exact (Tensor.Rep.zipWith_unstack
                NF.maskValue xR mask i).symm]
      exact hlocal

end
end TorchLean
