/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.NF.Ops -- shake: keep

/-!
# NF Shape Operators

Approximation lemmas for filling, replication, broadcasting, and Boolean masking in the
rounded-real NF model. Copied entries retain their error budgets; masked entries are exactly
zero. None of these operations introduces arithmetic rounding.

## PyTorch correspondence / citations
Related PyTorch shape operations:
https://pytorch.org/docs/stable/generated/torch.reshape.html
https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
https://pytorch.org/docs/stable/generated/torch.permute.html
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

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

omit [ValidExp fexp] [ValidRndToNearest rnd] in
/-- Filling a tensor preserves a scalar approximation budget at every shape.

Although this fact is used heavily when constructing reverse-mode zero contexts, it is a shape
fact rather than a backward-mode fact. Keeping it here also makes rounded constants available to
normalization, attention, and quantization without importing the reverse-mode implementation.
-/
theorem approxTensor_full_const {cS : ℝ} {cR : R} {eps : ℝ}
    (h : abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) cR - cS) ≤ eps) :
    ∀ {s : Shape},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Tensor.full s cS) (Tensor.full s cR) eps := by
  intro s
  induction s with
  | scalar =>
      exact (approxTensor_scalar_iff (α := R)
        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))).2 h
  | dim n inner ih =>
      have hε : 0 ≤ eps := le_trans (abs_nonneg _) h
      refine approxTensor_dim_of_forall
        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (xS := Tensor.full (.dim n inner) cS) (xR := Tensor.full (.dim n inner) cR)
        (eps := eps) hε ?_
      intro i
      have hfillS :
          (Tensor.full (.dim n inner) cS).unstack i = Tensor.full inner cS := by
        apply TorchLean.Tensor.Internal.Rep.ext
        intro coordinate
        simp [Tensor.full, Tensor.unstack]
      have hfillR :
          (Tensor.full (.dim n inner) cR).unstack i = Tensor.full inner cR := by
        apply TorchLean.Tensor.Internal.Rep.ext
        intro coordinate
        simp [Tensor.full, Tensor.unstack]
      simpa [hfillS, hfillR] using ih

/-- A tensor filled with runtime one differs from exact one by at most one construction rounding.

`(1 : R)` is `NF.ofReal 1`, so the scalar bound is the generic `approx_ofReal_nf`. -/
theorem approxTensor_full_one :
    ∀ {s : Shape},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Tensor.full s (1 : ℝ)) (Tensor.full s (1 : R)) (ulp β fexp (1 : ℝ) / 2) := by
  intro s
  exact approxTensor_full_const (β := β) (fexp := fexp) (rnd := rnd)
    (approx_ofReal_nf (β := β) (fexp := fexp) (rnd := rnd) 1)

/-- Zero is exactly representable in every valid neural floating-point format. -/
theorem approxTensor_full_zero :
    ∀ {s : Shape},
      approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (Tensor.full s (0 : ℝ)) (Tensor.full s (0 : R)) 0 := by
  intro s
  apply approxTensor_full_const (β := β) (fexp := fexp) (rnd := rnd)
  simp

/-- Keep a value where the mask allows it, otherwise zero. -/
private def maskValue {α : Type} [Zero α] (value : α) (allowed : Bool) : α :=
  if allowed then value else 0

omit [ValidExp fexp] [ValidRndToNearest rnd] in
/-- Filling a tensor with one approximate scalar gives an approximation with the same tolerance.

Broadcasting a constant copies a value rather than computing with it, so no new rounding occurs and
the error stays put. That is why `eps` appears unchanged on both sides. -/
theorem approxTensor_replicate {s : Shape}
    {xS : SpecTensor .scalar} {xR : Tensor R .scalar} {eps : ℝ}
    (hx : approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps) :
    approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (TorchLean.Tensor.replicate (α := SpecScalar) (shape := s) xS)
      (TorchLean.Tensor.replicate (α := R) (shape := s) xR)
      eps := by
  classical
  have hε : 0 ≤ eps := approxTensor_eps_nonneg hx
  induction s with
  | scalar =>
      simpa only [Spec.replicate_scalar_shape] using hx
  | dim n inner ih =>
      refine approxTensor_dim_of_forall
        (n := n) (s := inner)
        (xS := TorchLean.Tensor.replicate (α := SpecScalar) (shape := .dim n inner) xS)
        (xR := TorchLean.Tensor.replicate (α := R) (shape := .dim n inner) xR)
        (eps := eps) hε ?_
      intro i
      simpa only [Spec.unstack_replicate] using ih

omit [ValidExp fexp] [ValidRndToNearest rnd] in
/-- Broadcasting preserves the approximation tolerance: each output entry copies an input. -/
theorem approxTensor_broadcastTo
    {s₁ s₂ : Shape} (cb : Shape.CanBroadcastTo s₁ s₂)
    {xS : SpecTensor s₁} {xR : Tensor R s₁} {eps : ℝ}
    (hx : approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps) :
    approxTensor (α := R) (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (TorchLean.Tensor.broadcastTo (α := SpecScalar) (s₁ := s₁) (s₂ := s₂) cb xS)
      (TorchLean.Tensor.broadcastTo (α := R) (s₁ := s₁) (s₂ := s₂) cb xR)
      eps := by
  classical
  have hε : 0 ≤ eps := approxTensor_eps_nonneg (s := s₁) hx
  induction s₂ generalizing s₁ with
  | scalar =>
      cases s₁ with
      | scalar => simpa using hx
      | dim _ _ => exact absurd cb Shape.not_canBroadcastTo_dim_scalar
  | dim n t ih =>
      cases s₁ with
      | scalar =>
          rw [TorchLean.Tensor.broadcastTo_expand (Nat.zero_le _) cb,
            TorchLean.Tensor.broadcastTo_expand (Nat.zero_le _) cb]
          refine approxTensor_dim_of_forall (n := n) (s := t) (eps := eps) hε ?_
          intro _
          simpa using ih _ hx
      | dim m s =>
          by_cases hRank : s.rank = t.rank
          · obtain ⟨hHead, hTail⟩ := (Shape.canBroadcastTo_dim_dim_of_rank_eq hRank).mp cb
            rcases hHead with hmn | hm1
            · subst hmn
              rw [TorchLean.Tensor.broadcastTo_dim_eq hRank cb,
                TorchLean.Tensor.broadcastTo_dim_eq hRank cb]
              refine approxTensor_dim_of_forall (n := m) (s := t) (eps := eps) hε ?_
              intro i
              simpa using ih _
                (approxTensor_unstack (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hx i)
            · subst hm1
              rw [TorchLean.Tensor.broadcastTo_dim_one hRank cb,
                TorchLean.Tensor.broadcastTo_dim_one hRank cb]
              refine approxTensor_dim_of_forall (n := n) (s := t) (eps := eps) hε ?_
              intro _
              simpa using ih _
                (approxTensor_unstack (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hx
                  (0 : Fin 1))
          · have hTail := (Shape.canBroadcastTo_dim_dim_of_rank_ne hRank).mp cb
            rw [TorchLean.Tensor.broadcastTo_expand hTail.rank_le cb,
              TorchLean.Tensor.broadcastTo_expand hTail.rank_le cb]
            refine approxTensor_dim_of_forall (n := n) (s := t) (eps := eps) hε ?_
            intro _
            simpa using ih _ hx

/-- Applying the same Boolean mask to exact and rounded tensors preserves the error budget.

Masking is a selection operation, not arithmetic: allowed entries are unchanged and blocked
entries are exactly zero in both semantics. In particular, this theorem does not model a finite
negative sentinel and introduces no extra ULP term.
-/
theorem approxTensor_applyBoolMask {s : Shape}
    {xS : SpecTensor s} {xR : Tensor R s} (mask : Tensor Bool s) {eps : ℝ}
    (hx : approxTensor (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) xS xR eps) :
    approxTensor (α := R)
      (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
      (map2Spec (fun x allowed => if allowed then x else 0) xS mask)
      (map2Spec (fun x allowed => if allowed then x else 0) xR mask) eps := by
  change approxTensor (α := R)
    (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
    (map2Spec maskValue xS mask) (map2Spec maskValue xR mask) eps
  induction s with
  | scalar =>
      rw [← Tensor.scalar_item xS, ← Tensor.scalar_item xR] at hx
      rw [← Tensor.scalar_item xS, ← Tensor.scalar_item xR,
        ← Tensor.scalar_item mask]
      cases mask.item with
      | false =>
          apply (approxTensor_scalar_iff (α := R)
            (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))).2
          change abs (toSpec (β := β) (fexp := fexp) (rnd := rnd) (0 : R) - (0 : ℝ)) ≤ eps
          simpa using approxTensor_eps_nonneg hx
      | true =>
          simpa only [map2Spec_scalar, maskValue, ite_true] using hx
  | dim n inner ih =>
      have hε := approxTensor_eps_nonneg hx
      refine approxTensor_dim_of_forall
        (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd))
        (xS := map2Spec maskValue xS mask)
        (xR := map2Spec maskValue xR mask)
        (eps := eps) hε ?_
      intro i
      have hlocal :=
        ih (mask.unstack i)
          (approxTensor_unstack (α := R)
            (toSpec := toSpec (β := β) (fexp := fexp) (rnd := rnd)) hx i)
      rw [show
        (map2Spec maskValue xS mask).unstack i =
          map2Spec maskValue
            (xS.unstack i) (mask.unstack i) by
              exact (TorchLean.Tensor.Internal.Rep.zipWith_unstack
                maskValue xS mask i).symm]
      rw [show
        (map2Spec maskValue xR mask).unstack i =
          map2Spec maskValue
            (xR.unstack i) (mask.unstack i) by
              exact (TorchLean.Tensor.Internal.Rep.zipWith_unstack
                maskValue xR mask i).symm]
      exact hlocal

end NFBackend

end
end RuntimeApprox
end Proofs
