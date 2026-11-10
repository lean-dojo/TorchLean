/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import TorchLean.CUDA.Kernel.Gradient.FDeriv

/-!
# The gradient kernel of a real program

We apply `Expr.gradient_eq_adjoint` to two entrywise functions, written as kernel expressions over
the reals with one input of `m` entries. The first is `x / (x² + 1)`, whose denominator is never
zero, so every evaluation is regular, and at every point `θ₀` and every cotangent `cot` the theorem
shows that `θ ↦ θ / (θ² + 1)` is differentiable and that the gradient kernel computes the adjoint
of its derivative at `cot`. The second is the function of the test program `rational` in
`TorchLeanTest/CUDA/KernelGradient.lean`, `x² - x` below zero and `x / (x + 1)` from zero, whose
comparison with zero is regular exactly at the points with no zero entry, and there we get the
same conclusion. At a point with a zero entry that function is not differentiable.
-/

@[expose] public section

open Filter hiding pure_bind
open Topology

namespace TorchLean.Test.GradientKernel

open Kernel

/-- `x / (x² + 1)` at each entry of input zero, over the reals. -/
noncomputable def bump : Expr ℝ [.index] .scalar :=
  .binary .div (.load 0 (.var .zero))
    (.binary .add (.binary .mul (.load 0 (.var .zero)) (.load 0 (.var .zero))) (.scalar 1))

/-- The reads of a program with one input of `m` entries. -/
def oneInput (m : UInt64) : Reader ℝ :=
  fun o i => if o = 0 ∧ i < m then .ok 0 else .error (.input o)

/-- The reader of the gradient kernel of `bump`: the point `θ` as input zero and the cotangent
`cot` as input one. -/
noncomputable def gradientInputs {m : UInt64} (θ cot : EuclideanSpace ℝ (Fin m.toNat)) :
    Reader ℝ :=
  fun o i =>
    if h : i.toNat < m.toNat then
      if o = 0 then .ok (θ ⟨i.toNat, h⟩) else if o = 1 then .ok (cot ⟨i.toNat, h⟩)
      else .error (.input o)
    else .error (.bounds o i m.toNat)

/-- The map `θ ↦ θ / (θ² + 1)` that `bump` computes, entry by entry. -/
noncomputable def bumpMap {m : UInt64} (θ : EuclideanSpace ℝ (Fin m.toNat)) :
    EuclideanSpace ℝ (Fin m.toNat) :=
  WithLp.toLp 2 fun r => θ r / (θ r * θ r + 1)

/-- At every point `θ₀` and cotangent `cot`, `bumpMap` is differentiable at `θ₀` and the gradient
kernel of `bump` computes the adjoint of its derivative at `cot`. -/
theorem bump_gradient (m : UInt64) (θ₀ cot : EuclideanSpace ℝ (Fin m.toNat)) :
    DifferentiableAt ℝ bumpMap θ₀ ∧ ∀ a : Fin m.toNat,
      evaluate Arithmetic.ofOps (bump.gradient 1 0 m) (gradientInputs θ₀ cot) a.val.toUInt64 =
        .ok ((fderiv ℝ bumpMap θ₀).adjoint cot a) := by
  have hr (r : Fin m.toNat) : (r.val.toUInt64).toNat = r.val := by
    rw [Nat.toUInt64_eq]
    exact UInt64.toNat_ofNat_of_lt' (lt_trans r.isLt m.toNat_lt_size)
  have hlt (r : Fin m.toNat) : r.val.toUInt64 < m := by
    rw [UInt64.lt_iff_toNat_lt, hr]
    exact r.isLt
  have hread (θ : EuclideanSpace ℝ (Fin m.toNat)) (r : Fin m.toNat) :
      (oneInput m).withValues (withInput (fun _ _ => 0) 0 θ) 0 r.val.toUInt64 = .ok (θ r) := by
    have hc : 0 = 0 ∧ (r.val.toUInt64).toNat < m.toNat := ⟨rfl, by rw [hr]; exact r.isLt⟩
    have hf : (⟨(r.val.toUInt64).toNat, hc.2⟩ : Fin m.toNat) = r := Fin.ext (hr r)
    simp only [Reader.withValues, oneInput, hlt r, and_self, ↓reduceIte, Except.map, withInput,
      hc.2, ↓reduceDIte, hf]
  refine Expr.gradient_eq_adjoint bump (k := 1) (j := 0) (read := oneInput m)
    (fun o i y h => ?_) (fun _ _ => 0) (Eventually.of_forall fun θ r => ?_) (fun r => ?_)
    (fun o i y h => ?_) (fun r => ?_)
  · unfold oneInput at h
    split_ifs at h with hc
    · rw [hc.1]
      exact Nat.zero_lt_one
  · simp only [evaluate, bump, Expr.eval, Env.get_output, hread θ r, ← Expr.except_pure,
      pure_bind, Arithmetic.ofOps_binary_div, Arithmetic.ofOps_binary_mul,
      Arithmetic.ofOps_binary_add, bumpMap, PiLp.toLp_apply]
  · simp only [Expr.Regular, bump, true_and, reduceCtorEq, false_implies, true_implies,
      implies_true]
    intro _ _ b hb
    simp only [Expr.eval, Env.get_output, hread θ₀ r, ← Expr.except_pure, pure_bind,
      Arithmetic.ofOps_binary_mul, Arithmetic.ofOps_binary_add] at hb
    rw [← Except.ok.inj hb]
    exact (add_pos_of_nonneg_of_pos (mul_self_nonneg _) one_pos).ne'
  · obtain ⟨z, hz, rfl⟩ := Reader.exists_ok_of_withValues h
    unfold oneInput at hz
    split_ifs at hz with hc
    obtain ⟨rfl, hi⟩ := hc
    have hi' : i.toNat < m.toNat := UInt64.lt_iff_toNat_lt.mp hi
    simp only [gradientInputs, hi', ↓reduceDIte, ↓reduceIte, withInput, true_and]
  · simp only [gradientInputs, hr, r.isLt, ↓reduceDIte, one_ne_zero, ↓reduceIte]

/-- `x² - x` below zero and `x / (x + 1)` from zero at each entry of input zero, over the reals:
the function of the test program `rational`. -/
noncomputable def kink : Expr ℝ [.index] .scalar :=
  .cond (.compare .lt (.load 0 (.var .zero)) (.scalar 0))
    (.binary .sub (.binary .mul (.load 0 (.var .zero)) (.load 0 (.var .zero)))
      (.load 0 (.var .zero)))
    (.binary .div (.load 0 (.var .zero)) (.binary .add (.load 0 (.var .zero)) (.scalar 1)))

/-- The map that `kink` computes, entry by entry. -/
noncomputable def kinkMap {m : UInt64} (θ : EuclideanSpace ℝ (Fin m.toNat)) :
    EuclideanSpace ℝ (Fin m.toNat) :=
  WithLp.toLp 2 fun r => if θ r < 0 then θ r * θ r - θ r else θ r / (θ r + 1)

/-- The comparison `<` of the real arithmetic is the decision of `<`. -/
theorem ofOps_compare_lt (x y : ℝ) :
    (Arithmetic.ofOps : Arithmetic ℝ).compare .lt x y = decide (x < y) := rfl

/-- At every point `θ₀` with no zero entry and every cotangent `cot`, `kinkMap` is differentiable
at `θ₀` and the gradient kernel of `kink` computes the adjoint of its derivative at `cot`. -/
theorem kink_gradient (m : UInt64) (θ₀ cot : EuclideanSpace ℝ (Fin m.toNat))
    (h₀ : ∀ r, θ₀ r ≠ 0) :
    DifferentiableAt ℝ kinkMap θ₀ ∧ ∀ a : Fin m.toNat,
      evaluate Arithmetic.ofOps (kink.gradient 1 0 m) (gradientInputs θ₀ cot) a.val.toUInt64 =
        .ok ((fderiv ℝ kinkMap θ₀).adjoint cot a) := by
  have hr (r : Fin m.toNat) : (r.val.toUInt64).toNat = r.val := by
    rw [Nat.toUInt64_eq]
    exact UInt64.toNat_ofNat_of_lt' (lt_trans r.isLt m.toNat_lt_size)
  have hlt (r : Fin m.toNat) : r.val.toUInt64 < m := by
    rw [UInt64.lt_iff_toNat_lt, hr]
    exact r.isLt
  have hread (θ : EuclideanSpace ℝ (Fin m.toNat)) (r : Fin m.toNat) :
      (oneInput m).withValues (withInput (fun _ _ => 0) 0 θ) 0 r.val.toUInt64 = .ok (θ r) := by
    have hc : 0 = 0 ∧ (r.val.toUInt64).toNat < m.toNat := ⟨rfl, by rw [hr]; exact r.isLt⟩
    have hf : (⟨(r.val.toUInt64).toNat, hc.2⟩ : Fin m.toNat) = r := Fin.ext (hr r)
    simp only [Reader.withValues, oneInput, hlt r, and_self, ↓reduceIte, Except.map, withInput,
      hc.2, ↓reduceDIte, hf]
  refine Expr.gradient_eq_adjoint kink (k := 1) (j := 0) (read := oneInput m)
    (fun o i y h => ?_) (fun _ _ => 0) (Eventually.of_forall fun θ r => ?_) (fun r => ?_)
    (fun o i y h => ?_) (fun r => ?_)
  · unfold oneInput at h
    split_ifs at h with hc
    · rw [hc.1]
      exact Nat.zero_lt_one
  · simp only [evaluate, kink, Expr.eval, Env.get_output, hread θ r, ← Expr.except_pure,
      pure_bind, ofOps_compare_lt, Bool.cond_decide, Arithmetic.ofOps_binary_div,
      Arithmetic.ofOps_binary_mul, Arithmetic.ofOps_binary_add, Arithmetic.ofOps_binary_sub,
      kinkMap, PiLp.toLp_apply]
    split_ifs <;> rfl
  · refine ⟨⟨trivial, fun a ha => ⟨trivial, fun b hb => ?_⟩⟩, fun _ => ?_, fun hp => ?_⟩
    · simp only [Expr.eval, Env.get_output, ← Expr.except_pure, pure_bind, hread θ₀ r] at ha hb
      rw [← Except.ok.inj ha, ← Except.ok.inj hb]
      exact .inl (h₀ r)
    · simp only [Expr.Regular, true_and, reduceCtorEq, false_implies, implies_true]
    · simp only [Expr.eval, Env.get_output, hread θ₀ r, ← Expr.except_pure, pure_bind,
        ofOps_compare_lt] at hp
      have hn : 0 ≤ θ₀ r := not_lt.mp (of_decide_eq_false (Except.ok.inj hp))
      simp only [Expr.Regular, true_and, forall_const, reduceCtorEq, false_implies, implies_true]
      intro _ _ b hb
      simp only [Expr.eval, Env.get_output, hread θ₀ r, ← Expr.except_pure, pure_bind,
        Arithmetic.ofOps_binary_add] at hb
      rw [← Except.ok.inj hb]
      exact (add_pos_of_nonneg_of_pos hn one_pos).ne'
  · obtain ⟨z, hz, rfl⟩ := Reader.exists_ok_of_withValues h
    unfold oneInput at hz
    split_ifs at hz with hc
    obtain ⟨rfl, hi⟩ := hc
    have hi' : i.toNat < m.toNat := UInt64.lt_iff_toNat_lt.mp hi
    simp only [gradientInputs, hi', ↓reduceDIte, ↓reduceIte, withInput, true_and]
  · simp only [gradientInputs, hr, r.isLt, ↓reduceDIte, one_ne_zero, ↓reduceIte]

end TorchLean.Test.GradientKernel
