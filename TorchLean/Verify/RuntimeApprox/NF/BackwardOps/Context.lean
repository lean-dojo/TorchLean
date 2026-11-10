/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.NF.BackwardOps.Sparse
public import TorchLean.Verify.RuntimeApprox.NF.ShapeOps

/-!
# NF Backward Approximation Backend

Core approximation lemmas for the rounded NF backend: equal shapes at equal context indices,
zero and sparse context writes, and the context-addition bound for accumulating reverse-mode
contributions. The constant-filling lemmas come from `TorchLean.Verify.RuntimeApprox.NF.ShapeOps`.
-/

@[expose] public section

namespace TorchLean

open TorchLean.Tensor
open Robustness

noncomputable section

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq

variable {β : Radix} {fexp : ℤ → ℤ} [ValidExp fexp]
variable {rnd : ℝ → ℤ} [ValidRndToNearest rnd]

local notation "R" => NF β fexp rnd

/-- Two context indices pointing at the same position must have the same shape.

An `Idx Γ s` bundles a position with a proof that `Γ` holds shape `s` there, so equal positions
force equal shapes. -/
theorem Idx.shape_eq_of_i_eq {Γ : List Shape} {s₁ s₂ : Shape} (a : Idx Γ s₁) (b : Idx Γ s₂)
    (h : a.i = b.i) : s₁ = s₂ := by
  have : Γ.get a.i = Γ.get b.i := by simp [h]
  calc
    s₁ = Γ.get a.i := by simpa using a.h.symm
    _ = Γ.get b.i := this
    _ = s₂ := by simpa using b.h

omit [ValidExp fexp] [ValidRndToNearest rnd] in
/-- Approximation is stable under transporting both tensors along the same shape equality. -/
theorem ApproxTensor.castShape {s t : Shape} (h : s = t)
    {xS : Tensor ℝ s} {xR : Tensor R s} {eps : ℝ}
    (hx : ApproxTensor (α := R) (toSpec := NF.toReal) xS xR eps) :
    ApproxTensor (α := R) (toSpec := NF.toReal)
      (Tensor.castShape xS h) (Tensor.castShape xR h) eps := by
  cases h
  simpa using hx

/-- The all-zero context approximates the all-zero context, at zero error.

Zero is exactly representable in `NF`, so the zero slots of a sparse context write carry no
error. -/
theorem NF.approxCtx_zeros {Γ : List Shape} :
    ApproxCtx (α := R) (toSpec := NF.toReal)
      (TensorPack.zero (α := ℝ) (ss := Γ))
      (TensorPack.zero (α := R) (ss := Γ))
      (Pack.fill (fun _ => (0 : ℝ)) (ss := Γ)) := by
  induction Γ with
  | nil =>
      simp [TensorPack.zero, Pack.fill, ApproxCtx]
  | cons s Γ ih =>
      refine And.intro ?_ ih
      simpa [TensorPack.zero, Pack.fill] using
        (ApproxTensor.full_zero (β := β) (fexp := fexp) (rnd := rnd) (s := s))

/-- Writing one approximate tensor into an otherwise-zero context keeps the context approximate,
with the error recorded at that slot alone. -/
theorem NF.approxCtx_setIdx {Γ : List Shape} {s : Shape} (idx : Idx Γ s)
    {tS : Tensor ℝ s} {tR : Tensor R s} {eps : ℝ}
    (ht : ApproxTensor (α := R) (toSpec := NF.toReal) tS tR eps) :
    ApproxCtx (α := R) (toSpec := NF.toReal)
      (TensorPack.setIdx (α := ℝ) (Γ := Γ) (s := s) idx tS)
      (TensorPack.setIdx (α := R) (Γ := Γ) (s := s) idx tR)
      (EList.setIdx (Γ := Γ) (s := s) idx eps) := by
  cases idx with
  | mk i hshape =>
      induction Γ with
      | nil =>
          cases i with
          | mk val isLt =>
              exact False.elim ((Nat.not_lt_zero val) isLt)
      | cons s0 Γ ih =>
          cases i with
          | mk iVal hiVal =>
              cases iVal with
              | zero =>
                  cases hshape
                  refine And.intro ?_ ?_
                  · simpa [TensorPack.setIdx, EList.setIdx] using ht
                  · simpa [TensorPack.setIdx, EList.setIdx] using
                      (approxCtx_zeros (β := β) (fexp := fexp) (rnd := rnd) (Γ := Γ))
              | succ j =>
                  have hshape' : Γ.get ⟨j, Nat.lt_of_succ_lt_succ hiVal⟩ = s := by
                    simpa using hshape
                  have iht := ih (i := ⟨j, Nat.lt_of_succ_lt_succ hiVal⟩) (hshape := hshape')
                  refine And.intro ?_ ?_
                  · simpa [TensorPack.setIdx, EList.setIdx] using
                      (ApproxTensor.full_zero (β := β) (fexp := fexp) (rnd := rnd) (s := s0))
                  · simpa [TensorPack.setIdx, EList.setIdx] using iht

/-- Two writes at distinct positions do not interfere: each slot carries its own error bound.

Distinctness is essential. At a shared index the contributions must be added, with a combined
bound that includes addition rounding as well as both input errors. -/
theorem NF.approxCtx_set2Idx_ne {Γ : List Shape} {s₁ s₂ : Shape} (a : Idx Γ s₁) (b : Idx Γ s₂)
    {t₁S : Tensor ℝ s₁} {t₁R : Tensor R s₁} {eps₁ : ℝ}
    {t₂S : Tensor ℝ s₂} {t₂R : Tensor R s₂} {eps₂ : ℝ}
    (h₁ : ApproxTensor (α := R) (toSpec := NF.toReal)
      t₁S t₁R eps₁)
    (h₂ : ApproxTensor (α := R) (toSpec := NF.toReal)
      t₂S t₂R eps₂)
    (hne : a.i ≠ b.i) :
    ApproxCtx (α := R) (toSpec := NF.toReal)
      (TensorPack.set2Idx (α := ℝ) (Γ := Γ) (s₁ := s₁) (s₂ := s₂) a t₁S b t₂S)
      (TensorPack.set2Idx (α := R) (Γ := Γ) (s₁ := s₁) (s₂ := s₂) a t₁R b t₂R)
      (EList.set2Idx (Γ := Γ) (s₁ := s₁) (s₂ := s₂) a eps₁ b eps₂ 0) := by
  induction Γ with
  | nil =>
      cases a with
      | mk i _ =>
          cases i with
          | mk val isLt =>
              exact False.elim ((Nat.not_lt_zero val) isLt)
  | cons s0 Γ ih =>
      cases a with
      | mk ia haShape =>
          cases b with
          | mk ib hbShape =>
              cases ia with
              | mk iaVal iaLt =>
                  cases ib with
                  | mk ibVal ibLt =>
                      cases iaVal with
                      | zero =>
                          cases ibVal with
                          | zero =>
                              exact False.elim (hne (by rfl))
                          | succ j =>
                              cases haShape
                              refine And.intro ?_ ?_
                              · simpa [TensorPack.set2Idx, EList.set2Idx] using h₁
                              ·
                                let bTail : Idx Γ s₂ :=
                                  ⟨⟨j, Nat.lt_of_succ_lt_succ ibLt⟩, by simpa using hbShape⟩
                                have := approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
                                  (Γ := Γ) (s := s₂) bTail (tS := t₂S) (tR := t₂R) (eps := eps₂) h₂
                                simpa [TensorPack.set2Idx, EList.set2Idx, bTail] using this
                      | succ i =>
                          cases ibVal with
                          | zero =>
                              cases hbShape
                              refine And.intro ?_ ?_
                              · simpa [TensorPack.set2Idx, EList.set2Idx] using h₂
                              ·
                                let aTail : Idx Γ s₁ :=
                                  ⟨⟨i, Nat.lt_of_succ_lt_succ iaLt⟩, by simpa using haShape⟩
                                have := approxCtx_setIdx (β := β) (fexp := fexp) (rnd := rnd)
                                  (Γ := Γ) (s := s₁) aTail (tS := t₁S) (tR := t₁R) (eps := eps₁) h₁
                                simpa [TensorPack.set2Idx, EList.set2Idx, aTail] using this
                          | succ j =>
                              let aTail : Idx Γ s₁ :=
                                ⟨⟨i, Nat.lt_of_succ_lt_succ iaLt⟩, by simpa using haShape⟩
                              let bTail : Idx Γ s₂ :=
                                ⟨⟨j, Nat.lt_of_succ_lt_succ ibLt⟩, by simpa using hbShape⟩
                              have hneTail : aTail.i ≠ bTail.i :=
                                fun hij => hne (congrArg Fin.succ hij)
                              refine And.intro ?_ ?_
                              · simpa [TensorPack.set2Idx, EList.set2Idx] using
                                  (ApproxTensor.full_zero (β := β) (fexp := fexp) (rnd := rnd)
                                    (s := s0))
                              ·
                                have := ih (a := aTail) (b := bTail) hneTail
                                simpa [TensorPack.set2Idx, EList.set2Idx, aTail, bTail]
                                  using this

/-! ## Context-wise addition bound, used by global backprop accumulation -/

/--
Context-wise addition bound (NF runtime vs spec).

Entry `i` is the `linfNorm` of `addBoundTensor` for slot `i` of the two contexts.
Reverse-mode accumulation charges it each time a node's contribution is added to the running seed,
which is where the contributions of several consumers of one node combine.
-/
def NF.ctxAddBound : {Δ : List Shape} → EList Δ → EList Δ →
    TensorPack R Δ → TensorPack R Δ → EList Δ
  | [], .nil, .nil, .nil, .nil => .nil
  | _ :: ss, .cons ex exs, .cons ey eys, .cons x xs, .cons y ys =>
      .cons (linfNorm
        (addBoundTensor (β := β) (fexp := fexp) (rnd := rnd) (s := _) ex ey x y))
        (ctxAddBound (Δ := ss) exs eys xs ys)

/--
Soundness of context-wise addition under `ApproxCtx`.

If `xS ~ xR ± epsx` and `yS ~ yR ± epsy`, then `(xS + yS) ~ (xR + yR)` with error bounded by
`ctxAddBound epsx epsy xR yR`.
-/
theorem ApproxCtx.add {Δ : List Shape} :
    ∀ (xS yS : TensorPack ℝ Δ) (xR yR : TensorPack R Δ)
      (epsx epsy : EList Δ),
      ApproxCtx (α := R) (toSpec := NF.toReal) xS xR epsx →
      ApproxCtx (α := R) (toSpec := NF.toReal) yS yR epsy →
        ApproxCtx (α := R) (toSpec := NF.toReal)
          (TensorPack.add (α := ℝ) xS yS)
          (TensorPack.add (α := R) xR yR)
          (NF.ctxAddBound (β := β) (fexp := fexp) (rnd := rnd) epsx epsy xR yR) := by
  intro xS yS xR yR epsx epsy hx hy
  induction Δ with
  | nil =>
      cases xS
      cases yS
      cases xR
      cases yR
      cases epsx
      cases epsy
      simp [TensorPack.add, NF.ctxAddBound, ApproxCtx]
  | cons s ss ih =>
      cases xS with
      | cons xSh xSt =>
          cases yS with
          | cons ySh ySt =>
              cases xR with
              | cons xRh xRt =>
                  cases yR with
                  | cons yRh yRt =>
                      cases epsx with
                      | cons ex exs =>
                          cases epsy with
                          | cons ey eys =>
                              refine And.intro ?_ ?_
                              · have hx0 : ApproxTensor (α := R) (toSpec := NF.toReal) xSh xRh ex :=
                                  hx.1
                                have hy0 : ApproxTensor (α := R) (toSpec := NF.toReal) ySh yRh ey :=
                                  hy.1
                                simpa [TensorPack.add, NF.ctxAddBound] using
                                  (ApproxTensor.add (β := β) (fexp := fexp) (rnd := rnd)
                                    (s := s) (xS := xSh) (yS := ySh) (xR := xRh) (yR := yRh)
                                    (epsx := ex) (epsy := ey) hx0 hy0)
                              · have hxT : ApproxCtx (α := R) (toSpec := NF.toReal) xSt xRt exs :=
                                  hx.2
                                have hyT : ApproxCtx (α := R) (toSpec := NF.toReal) ySt yRt eys :=
                                  hy.2
                                simpa [TensorPack.add, NF.ctxAddBound] using
                                  ih (xS := xSt) (yS := ySt) (xR := xRt) (yR := yRt) (epsx := exs)
                                    (epsy := eys) hxT hyT

end

end TorchLean
