/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Autograd.Model.Dual.Nested
public import NN.Proofs.Autograd.FDeriv.Fold
public import NN.Proofs.Gradients.Activation
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv

/-!
# Higher-order semantics of runtime dual numbers

A nested dual value stores mixed derivatives in independent infinitesimal directions. `jet`
gives those coefficients their mathematical meaning using mathlib's Fréchet derivative. The
operation theorems connect that meaning to the actual runtime arithmetic at every finite order.

The last direction occupies the outermost dual layer. Extracting every tangent gives
`iteratedFDeriv` applied to the complete direction tuple, with no factorial scaling. The
input space is any real normed space; directions need not be distinct or coordinate vectors.

These are exact-real scalar rules. Model lowering, effectful evaluation, and floating-point
rounding require their own links to these semantics.
-/

@[expose] public section

open Runtime.Autograd.Model
open scoped ContDiff Topology

namespace Runtime.Autograd.Model.Dual

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- All mixed derivative coefficients of a function along the supplied directions. -/
noncomputable def jet : {n : Nat} → (Fin n → E) → (E → ℝ) → E → Nested ℝ n
  | 0, _, f, x => f x
  | n + 1, directions, f, x =>
    ⟨jet (Fin.init directions) f x,
      jet (Fin.init directions) (fun y => fderiv ℝ f y (directions (Fin.last n))) x⟩

/-- All coefficients of the zero function vanish. -/
theorem jet_zero {n : Nat} (directions : Fin n → E) (x : E) :
    jet directions (fun _ => 0) x = 0 := by
  induction n with
  | zero => rfl
  | succ n ih =>
    change Dual.mk _ _ = Dual.mk 0 0
    congr 1
    · exact ih _
    · simpa only [fderiv_const_apply, zero_apply] using ih (Fin.init directions)

/-- Embedding a constant is valid at every order, not just for one forward pass. -/
theorem jet_const {n : Nat} (directions : Fin n → E) (value : ℝ) (x : E) :
    jet directions (fun _ => value) x = Nested.ofPrimal n value := by
  induction n with
  | zero => rfl
  | succ n ih =>
    change Dual.mk _ _ = Dual.mk _ 0
    congr 1
    · exact ih _
    · simpa only [fderiv_const_apply, zero_apply] using
        jet_zero (Fin.init directions) x

/-- A linear input coordinate has its value and supplied directions, and no higher coefficients. -/
theorem jet_linear {n : Nat} (directions : Fin n → E) (f : E →L[ℝ] ℝ) (x : E) :
    jet directions f x = Nested.seed (fun i => f (directions i)) (f x) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    change Dual.mk _ _ = Dual.mk _ _
    congr 1
    · exact ih _
    · simpa only [f.fderiv] using jet_const (Fin.init directions)
        (f (directions (Fin.last n))) x

/-- Scalar input seeding agrees with the identity function's full derivative data. -/
theorem jet_id {n : Nat} (directions : Fin n → ℝ) (x : ℝ) :
    jet directions (fun y => y) x = Nested.seed directions x :=
  jet_linear directions (ContinuousLinearMap.id ℝ ℝ) x

/-- The runtime negation rule preserves all derivative coefficients. -/
theorem jet_neg {n : Nat} (directions : Fin n → E) (f : E → ℝ) (x : E) :
    jet directions (fun y => -f y) x = -jet directions f x := by
  induction n generalizing f with
  | zero => rfl
  | succ n ih =>
    change Dual.mk _ _ = Dual.mk _ _
    congr 1
    · exact ih _ _
    · change jet (Fin.init directions)
        (fun y => fderiv ℝ (fun z => -f z) y (directions (Fin.last n))) x =
          -jet (Fin.init directions) (fun y => fderiv ℝ f y (directions (Fin.last n))) x
      simpa only [fderiv_fun_neg, neg_apply] using
        ih (Fin.init directions) (fun y => fderiv ℝ f y (directions (Fin.last n)))

/-- The derivative coefficients depend only on the function near the evaluation point. -/
theorem jet_congr {n : Nat} (directions : Fin n → E) {f g : E → ℝ} {x : E}
    (h : f =ᶠ[𝓝 x] g) : jet directions f x = jet directions g x := by
  induction n generalizing f g with
  | zero => exact h.eq_of_nhds
  | succ n ih =>
    change Dual.mk _ _ = Dual.mk _ _
    congr 1
    · exact ih _ h
    · apply ih
      filter_upwards [h.fderiv (𝕜 := ℝ)] with y hy
      exact congrArg (fun d => d (directions (Fin.last n))) hy

/-- Addition preserves mixed derivatives under pointwise smoothness assumptions. -/
theorem jet_add_at {n : Nat} (directions : Fin n → E) {f g : E → ℝ} {x : E}
    (hf : ContDiffAt ℝ n f x) (hg : ContDiffAt ℝ n g x) :
    jet directions (fun y => f y + g y) x = jet directions f x + jet directions g x := by
  induction n generalizing f g with
  | zero => rfl
  | succ n ih =>
    let v := directions (Fin.last n)
    have hf' : ContDiffAt ℝ n (fun y => fderiv ℝ f y v) x :=
      (hf.fderiv_right (by simp)).clm_apply contDiffAt_const
    have hg' : ContDiffAt ℝ n (fun y => fderiv ℝ g y v) x :=
      (hg.fderiv_right (by simp)).clm_apply contDiffAt_const
    have heq : (fun y => fderiv ℝ (fun z => f z + g z) y v) =ᶠ[𝓝 x]
        fun y => fderiv ℝ f y v + fderiv ℝ g y v := by
      filter_upwards [hf.eventually (by simp), hg.eventually (by simp)] with y hfy hgy
      rw [fderiv_fun_add (hfy.differentiableAt (by simp)) (hgy.differentiableAt (by simp))]
      rfl
    change Dual.mk _ _ = Dual.mk _ _
    congr 1
    · exact ih _ (hf.of_le (by simp)) (hg.of_le (by simp))
    · rw [jet_congr _ heq]
      exact ih _ hf' hg'

/-- Subtraction preserves mixed derivatives under pointwise smoothness assumptions. -/
theorem jet_sub_at {n : Nat} (directions : Fin n → E) {f g : E → ℝ} {x : E}
    (hf : ContDiffAt ℝ n f x) (hg : ContDiffAt ℝ n g x) :
    jet directions (fun y => f y - g y) x = jet directions f x - jet directions g x := by
  induction n generalizing f g with
  | zero => rfl
  | succ n ih =>
    let v := directions (Fin.last n)
    have hf' : ContDiffAt ℝ n (fun y => fderiv ℝ f y v) x :=
      (hf.fderiv_right (by simp)).clm_apply contDiffAt_const
    have hg' : ContDiffAt ℝ n (fun y => fderiv ℝ g y v) x :=
      (hg.fderiv_right (by simp)).clm_apply contDiffAt_const
    have heq : (fun y => fderiv ℝ (fun z => f z - g z) y v) =ᶠ[𝓝 x]
        fun y => fderiv ℝ f y v - fderiv ℝ g y v := by
      filter_upwards [hf.eventually (by simp), hg.eventually (by simp)] with y hfy hgy
      rw [fderiv_fun_sub (hfy.differentiableAt (by simp)) (hgy.differentiableAt (by simp))]
      rfl
    change Dual.mk _ _ = Dual.mk _ _
    congr 1
    · exact ih _ (hf.of_le (by simp)) (hg.of_le (by simp))
    · rw [jet_congr _ heq]
      exact ih _ hf' hg'

/-- The runtime product rule needs smoothness only near the evaluation point. -/
theorem jet_mul_at {n : Nat} (directions : Fin n → E) {f g : E → ℝ} {x : E}
    (hf : ContDiffAt ℝ n f x) (hg : ContDiffAt ℝ n g x) :
    jet directions (fun y => f y * g y) x = jet directions f x * jet directions g x := by
  induction n generalizing f g with
  | zero => rfl
  | succ n ih =>
    let v := directions (Fin.last n)
    have hfn : ContDiffAt ℝ n f x := hf.of_le (by simp)
    have hgn : ContDiffAt ℝ n g x := hg.of_le (by simp)
    have hf' : ContDiffAt ℝ n (fun y => fderiv ℝ f y v) x :=
      (hf.fderiv_right (by simp)).clm_apply contDiffAt_const
    have hg' : ContDiffAt ℝ n (fun y => fderiv ℝ g y v) x :=
      (hg.fderiv_right (by simp)).clm_apply contDiffAt_const
    have heq : (fun y => fderiv ℝ (fun z => f z * g z) y v) =ᶠ[𝓝 x]
        fun y => fderiv ℝ f y v * g y + f y * fderiv ℝ g y v := by
      filter_upwards [hf.eventually (by simp), hg.eventually (by simp)] with y hfy hgy
      rw [fderiv_fun_mul (hfy.differentiableAt (by simp)) (hgy.differentiableAt (by simp))]
      simp [smul_eq_mul, mul_comm, add_comm]
    change Dual.mk _ _ = Dual.mk _ _
    congr 1
    · exact ih _ hfn hgn
    · rw [jet_congr _ heq, jet_add_at _ (hf'.mul hgn) (hfn.mul hg'),
        ih _ hf' hgn, ih _ hfn hg']
      rfl

/-- Runtime addition preserves every derivative coefficient through order `n`. -/
theorem jet_add {n : Nat} (directions : Fin n → E) {f g : E → ℝ}
    (hf : ContDiff ℝ n f) (hg : ContDiff ℝ n g) (x : E) :
    jet directions (fun y => f y + g y) x = jet directions f x + jet directions g x :=
  jet_add_at directions hf.contDiffAt hg.contDiffAt

/-- Runtime subtraction preserves mixed derivatives through the requested finite order. -/
theorem jet_sub {n : Nat} (directions : Fin n → E) {f g : E → ℝ}
    (hf : ContDiff ℝ n f) (hg : ContDiff ℝ n g) (x : E) :
    jet directions (fun y => f y - g y) x = jet directions f x - jet directions g x :=
  jet_sub_at directions hf.contDiffAt hg.contDiffAt

/-- The constant one has primal one and no derivative coefficients at any depth. -/
theorem jet_one {n : Nat} (directions : Fin n → E) (x : E) :
    jet directions (fun _ => 1) x = 1 := by
  rw [jet_const]
  induction n with
  | zero => rfl
  | succ n ih =>
      change Dual.mk (Nested.ofPrimal n 1) 0 = Dual.mk 1 0
      rw [ih (Fin.init directions)]

/-- The runtime product rule remains correct under arbitrary nesting. -/
theorem jet_mul {n : Nat} (directions : Fin n → E) {f g : E → ℝ}
    (hf : ContDiff ℝ n f) (hg : ContDiff ℝ n g) (x : E) :
    jet directions (fun y => f y * g y) x = jet directions f x * jet directions g x :=
  jet_mul_at directions hf.contDiffAt hg.contDiffAt

/-- The runtime exponential rule preserves mixed derivatives through order `n`. -/
theorem jet_exp_comp_at {n : Nat} (directions : Fin n → E) {f : E → ℝ}
    {x : E} (hf : ContDiffAt ℝ n f x) :
    jet directions (fun y => Real.exp (f y)) x = MathFunctions.exp (jet directions f x) := by
  induction n generalizing f with
  | zero => rfl
  | succ n ih =>
    let v := directions (Fin.last n)
    have hf' : ContDiffAt ℝ n (fun y => fderiv ℝ f y v) x :=
      (hf.fderiv_right (by simp)).clm_apply contDiffAt_const
    have heq : (fun y => fderiv ℝ (fun z => Real.exp (f z)) y v) =ᶠ[𝓝 x]
        fun y => Real.exp (f y) * fderiv ℝ f y v := by
      filter_upwards [hf.eventually (by simp)] with y hfy
      rw [fderiv_exp (hfy.differentiableAt (by simp))]
      rfl
    change Dual.mk _ _ = Dual.mk _ _
    congr 1
    · exact ih _ (hf.of_le (by simp))
    · rw [jet_congr _ heq, jet_mul_at _ (hf.exp.of_le (by simp)) hf', ih _ (hf.of_le (by simp))]
      rfl

private theorem jet_sin_cos_at {n : Nat} (directions : Fin n → E) {f : E → ℝ}
    {x : E} (hf : ContDiffAt ℝ n f x) :
    jet directions (fun y => Real.sin (f y)) x = MathFunctions.sin (jet directions f x) ∧
      jet directions (fun y => Real.cos (f y)) x = MathFunctions.cos (jet directions f x) := by
  induction n generalizing f with
  | zero => exact ⟨rfl, rfl⟩
  | succ n ih =>
    let v := directions (Fin.last n)
    have hfn : ContDiffAt ℝ n f x := hf.of_le (by simp)
    have hf' : ContDiffAt ℝ n (fun y => fderiv ℝ f y v) x :=
      (hf.fderiv_right (by simp)).clm_apply contDiffAt_const
    obtain ⟨hsin, hcos⟩ := ih (Fin.init directions) hfn
    constructor
    · have heq : (fun y => fderiv ℝ (fun z => Real.sin (f z)) y v) =ᶠ[𝓝 x]
          fun y => Real.cos (f y) * fderiv ℝ f y v := by
        filter_upwards [hf.eventually (by simp)] with y hfy
        rw [fderiv_sin (hfy.differentiableAt (by simp))]
        rfl
      change Dual.mk _ _ = Dual.mk _ _
      apply congrArg₂ Dual.mk
      · exact hsin
      · rw [jet_congr _ heq, jet_mul_at _ hfn.cos hf', hcos]
        rfl
    · have heq : (fun y => fderiv ℝ (fun z => Real.cos (f z)) y v) =ᶠ[𝓝 x]
          fun y => -Real.sin (f y) * fderiv ℝ f y v := by
        filter_upwards [hf.eventually (by simp)] with y hfy
        rw [fderiv_cos (hfy.differentiableAt (by simp))]
        rfl
      change Dual.mk _ _ = Dual.mk _ _
      apply congrArg₂ Dual.mk
      · exact hcos
      · rw [jet_congr _ heq, jet_mul_at _ hfn.sin.neg hf', jet_neg, hsin]
        rfl

/-- The sine and cosine runtime rules preserve one another's higher derivatives. -/
theorem jet_sin_comp_at {n : Nat} (directions : Fin n → E) {f : E → ℝ}
    {x : E} (hf : ContDiffAt ℝ n f x) :
    jet directions (fun y => Real.sin (f y)) x = MathFunctions.sin (jet directions f x) :=
  (jet_sin_cos_at directions hf).1

/-- The cosine rule, including its alternating derivative signs, is valid at every order. -/
theorem jet_cos_comp_at {n : Nat} (directions : Fin n → E) {f : E → ℝ}
    {x : E} (hf : ContDiffAt ℝ n f x) :
    jet directions (fun y => Real.cos (f y)) x = MathFunctions.cos (jet directions f x) :=
  (jet_sin_cos_at directions hf).2

/-- The runtime tanh rule computes every mixed derivative of a smooth real input function. -/
theorem jet_tanh_comp_at {n : Nat} (directions : Fin n → E) {f : E → ℝ}
    {x : E} (hf : ContDiffAt ℝ n f x) :
    jet directions (fun y => Real.tanh (f y)) x = MathFunctions.tanh (jet directions f x) := by
  induction n generalizing f with
  | zero => rfl
  | succ n ih =>
    let v := directions (Fin.last n)
    have hfn : ContDiffAt ℝ n f x := hf.of_le (by simp)
    have ht : ContDiffAt ℝ n (fun y => Real.tanh (f y)) x :=
      Proofs.contDiff_tanh.contDiffAt.comp x hfn
    have hf' : ContDiffAt ℝ n (fun y => fderiv ℝ f y v) x :=
      (hf.fderiv_right (by simp)).clm_apply contDiffAt_const
    have heq : (fun y => fderiv ℝ (fun z => Real.tanh (f z)) y v) =ᶠ[𝓝 x]
        fun y => (1 - Real.tanh (f y) * Real.tanh (f y)) * fderiv ℝ f y v := by
      filter_upwards [hf.eventually (by simp)] with y hfy
      have h := (Proofs.tanh_deriv_correct (f y)).comp_hasFDerivAt y
        (hfy.differentiableAt (by simp)).hasFDerivAt
      simpa only [Function.comp_def, Activation.Math.tanhSpec, Activation.Math.tanhDerivSpec,
        Proofs.mathfunc_tanh_eq_rtanh, smul_apply, smul_eq_mul] using
        congrArg (fun d => d v) h.fderiv
    change Dual.mk _ _ = Dual.mk _ _
    congr 1
    · exact ih _ hfn
    · rw [jet_congr _ heq, jet_mul_at _ (contDiffAt_const.sub (ht.mul ht)) hf',
        jet_sub_at _ contDiffAt_const (ht.mul ht), jet_one, jet_mul_at _ ht ht, ih _ hfn]
      rfl

private theorem jet_sinh_cosh_at {n : Nat} (directions : Fin n → E) {f : E → ℝ}
    {x : E} (hf : ContDiffAt ℝ n f x) :
    jet directions (fun y => Real.sinh (f y)) x = MathFunctions.sinh (jet directions f x) ∧
      jet directions (fun y => Real.cosh (f y)) x = MathFunctions.cosh (jet directions f x) := by
  induction n generalizing f with
  | zero => exact ⟨rfl, rfl⟩
  | succ n ih =>
    let v := directions (Fin.last n)
    have hfn : ContDiffAt ℝ n f x := hf.of_le (by simp)
    have hf' : ContDiffAt ℝ n (fun y => fderiv ℝ f y v) x :=
      (hf.fderiv_right (by simp)).clm_apply contDiffAt_const
    obtain ⟨hsinh, hcosh⟩ := ih (Fin.init directions) hfn
    constructor
    · have heq : (fun y => fderiv ℝ (fun z => Real.sinh (f z)) y v) =ᶠ[𝓝 x]
          fun y => Real.cosh (f y) * fderiv ℝ f y v := by
        filter_upwards [hf.eventually (by simp)] with y hfy
        rw [fderiv_sinh (hfy.differentiableAt (by simp))]
        rfl
      change Dual.mk _ _ = Dual.mk _ _
      apply congrArg₂ Dual.mk
      · exact hsinh
      · rw [jet_congr _ heq, jet_mul_at _ hfn.cosh hf', hcosh]
        rfl
    · have heq : (fun y => fderiv ℝ (fun z => Real.cosh (f z)) y v) =ᶠ[𝓝 x]
          fun y => Real.sinh (f y) * fderiv ℝ f y v := by
        filter_upwards [hf.eventually (by simp)] with y hfy
        rw [fderiv_cosh (hfy.differentiableAt (by simp))]
        rfl
      change Dual.mk _ _ = Dual.mk _ _
      apply congrArg₂ Dual.mk
      · exact hcosh
      · rw [jet_congr _ heq, jet_mul_at _ hfn.sinh hf', hsinh]
        rfl

/-- Hyperbolic sine preserves higher derivatives through the runtime's mutual sinh/cosh rules. -/
theorem jet_sinh_comp_at {n : Nat} (directions : Fin n → E) {f : E → ℝ}
    {x : E} (hf : ContDiffAt ℝ n f x) :
    jet directions (fun y => Real.sinh (f y)) x = MathFunctions.sinh (jet directions f x) :=
  (jet_sinh_cosh_at directions hf).1

/-- Hyperbolic cosine preserves higher derivatives through the runtime's mutual sinh/cosh rules. -/
theorem jet_cosh_comp_at {n : Nat} (directions : Fin n → E) {f : E → ℝ}
    {x : E} (hf : ContDiffAt ℝ n f x) :
    jet directions (fun y => Real.cosh (f y)) x = MathFunctions.cosh (jet directions f x) :=
  (jet_sinh_cosh_at directions hf).2

@[inherit_doc jet_exp_comp_at]
theorem jet_exp_comp {n : Nat} (directions : Fin n → E) {f : E → ℝ}
    (hf : ContDiff ℝ n f) (x : E) :
    jet directions (fun y => Real.exp (f y)) x = MathFunctions.exp (jet directions f x) :=
  jet_exp_comp_at directions hf.contDiffAt

@[inherit_doc jet_sin_comp_at]
theorem jet_sin_comp {n : Nat} (directions : Fin n → E) {f : E → ℝ}
    (hf : ContDiff ℝ n f) (x : E) :
    jet directions (fun y => Real.sin (f y)) x = MathFunctions.sin (jet directions f x) :=
  jet_sin_comp_at directions hf.contDiffAt

@[inherit_doc jet_cos_comp_at]
theorem jet_cos_comp {n : Nat} (directions : Fin n → E) {f : E → ℝ}
    (hf : ContDiff ℝ n f) (x : E) :
    jet directions (fun y => Real.cos (f y)) x = MathFunctions.cos (jet directions f x) :=
  jet_cos_comp_at directions hf.contDiffAt

@[inherit_doc jet_tanh_comp_at]
theorem jet_tanh_comp {n : Nat} (directions : Fin n → E) {f : E → ℝ}
    (hf : ContDiff ℝ n f) (x : E) :
    jet directions (fun y => Real.tanh (f y)) x = MathFunctions.tanh (jet directions f x) :=
  jet_tanh_comp_at directions hf.contDiffAt

@[inherit_doc jet_sinh_comp_at]
theorem jet_sinh_comp {n : Nat} (directions : Fin n → E) {f : E → ℝ}
    (hf : ContDiff ℝ n f) (x : E) :
    jet directions (fun y => Real.sinh (f y)) x = MathFunctions.sinh (jet directions f x) :=
  jet_sinh_comp_at directions hf.contDiffAt

@[inherit_doc jet_cosh_comp_at]
theorem jet_cosh_comp {n : Nat} (directions : Fin n → E) {f : E → ℝ}
    (hf : ContDiff ℝ n f) (x : E) :
    jet directions (fun y => Real.cosh (f y)) x = MathFunctions.cosh (jet directions f x) :=
  jet_cosh_comp_at directions hf.contDiffAt

/-- Direct scalar exponential evaluation on seeded runtime inputs. -/
theorem jet_exp {n : Nat} (directions : Fin n → ℝ) (x : ℝ) :
    jet directions Real.exp x = MathFunctions.exp (Nested.seed directions x) := by
  simpa only [jet_id] using jet_exp_comp directions (f := fun y => y) contDiff_id x

/-- Direct scalar sine evaluation on seeded runtime inputs. -/
theorem jet_sin {n : Nat} (directions : Fin n → ℝ) (x : ℝ) :
    jet directions Real.sin x = MathFunctions.sin (Nested.seed directions x) := by
  simpa only [jet_id] using jet_sin_comp directions (f := fun y => y) contDiff_id x

/-- Direct scalar cosine evaluation on seeded runtime inputs. -/
theorem jet_cos {n : Nat} (directions : Fin n → ℝ) (x : ℝ) :
    jet directions Real.cos x = MathFunctions.cos (Nested.seed directions x) := by
  simpa only [jet_id] using jet_cos_comp directions (f := fun y => y) contDiff_id x

/-- Direct scalar tanh evaluation on seeded runtime inputs. -/
theorem jet_tanh {n : Nat} (directions : Fin n → ℝ) (x : ℝ) :
    jet directions Real.tanh x = MathFunctions.tanh (Nested.seed directions x) := by
  simpa only [jet_id] using jet_tanh_comp directions (f := fun y => y) contDiff_id x

/-- Direct scalar hyperbolic sine evaluation on seeded runtime inputs. -/
theorem jet_sinh {n : Nat} (directions : Fin n → ℝ) (x : ℝ) :
    jet directions Real.sinh x = MathFunctions.sinh (Nested.seed directions x) := by
  simpa only [jet_id] using jet_sinh_comp directions (f := fun y => y) contDiff_id x

/-- Direct scalar hyperbolic cosine evaluation on seeded runtime inputs. -/
theorem jet_cosh {n : Nat} (directions : Fin n → ℝ) (x : ℝ) :
    jet directions Real.cosh x = MathFunctions.cosh (Nested.seed directions x) := by
  simpa only [jet_id] using jet_cosh_comp directions (f := fun y => y) contDiff_id x

/-- Extracting every tangent gives the iterated derivative under local smoothness. -/
theorem tangent_jet_at {n : Nat} (directions : Fin n → E) {f : E → ℝ} {x : E}
    (hf : ContDiffAt ℝ n f x) :
    Nested.tangent (jet directions f x) = iteratedFDeriv ℝ n f x directions := by
  induction n generalizing f with
  | zero => simp [Nested.tangent, jet, iteratedFDeriv_zero_apply]
  | succ n ih =>
    have hf' : ContDiffAt ℝ n (fderiv ℝ f) x := hf.fderiv_right (by simp)
    change Nested.tangent (jet (Fin.init directions)
      (fun y => fderiv ℝ f y (directions (Fin.last n))) x) = _
    rw [ih _ (hf'.clm_apply contDiffAt_const), iteratedFDeriv_succ_apply_right]
    exact congrArg (fun d => d (Fin.init directions))
      ((ContinuousLinearMap.apply ℝ ℝ (directions (Fin.last n))).iteratedFDeriv_comp_left
        hf' (by simp))

/-- The extracted runtime coefficient is mathlib's iterated Fréchet derivative. -/
theorem tangent_jet {n : Nat} (directions : Fin n → E) {f : E → ℝ}
    (hf : ContDiff ℝ n f) (x : E) :
    Nested.tangent (jet directions f x) = iteratedFDeriv ℝ n f x directions :=
  tangent_jet_at directions hf.contDiffAt

/-- A smooth update with a local jet law preserves jets in the supplied traversal order. -/
theorem jet_foldl_at {n : Nat} {ι : Type*} (indices : List ι) (directions : Fin n → E)
    (step : ℝ → ℝ → ℝ) (lifted : Nested ℝ n → Nested ℝ n → Nested ℝ n)
    (hstep : ContDiff ℝ n (fun p : ℝ × ℝ => step p.1 p.2)) {x : E}
    (hjet : ∀ (f g : E → ℝ), ContDiffAt ℝ n f x → ContDiffAt ℝ n g x →
      jet directions (fun y => step (f y) (g y)) x =
        lifted (jet directions f x) (jet directions g x))
    {initial : E → ℝ} {term : ι → E → ℝ} (hinit : ContDiffAt ℝ n initial x)
    (hterm : ∀ i ∈ indices, ContDiffAt ℝ n (term i) x) :
    jet directions
      (fun y => indices.foldl (fun acc i => step acc (term i y)) (initial y)) x =
      indices.foldl (fun acc i => lifted acc (jet directions (term i) x))
        (jet directions initial x) := by
  induction indices generalizing initial with
  | nil => rfl
  | cons i indices ih =>
      rw [List.foldl_cons]
      rw [← hjet initial (term i) hinit (hterm i (by simp))]
      exact ih (hstep.contDiffAt.comp x (hinit.prodMk (hterm i (by simp))))
        (fun j hj => hterm j (by simp [hj]))

/-- Ordered sums preserve locally smooth derivative coefficients of entries and accumulator. -/
theorem jet_foldl_add_at {n : Nat} {ι : Type*} (indices : List ι) (directions : Fin n → E)
    {initial : E → ℝ} {term : ι → E → ℝ} {x : E} (hinit : ContDiffAt ℝ n initial x)
    (hterm : ∀ i ∈ indices, ContDiffAt ℝ n (term i) x) :
    jet directions (fun y => indices.foldl (fun acc i => acc + term i y) (initial y)) x =
      indices.foldl (fun acc i => acc + jet directions (term i) x)
        (jet directions initial x) :=
  jet_foldl_at indices directions (· + ·) (· + ·) (by fun_prop)
    (fun _ _ hf hg => jet_add_at directions hf hg) hinit hterm

/-- Ordered products preserve local jets without requiring nonzero factors. -/
theorem jet_foldl_mul_at {n : Nat} {ι : Type*} (indices : List ι) (directions : Fin n → E)
    {initial : E → ℝ} {term : ι → E → ℝ} {x : E} (hinit : ContDiffAt ℝ n initial x)
    (hterm : ∀ i ∈ indices, ContDiffAt ℝ n (term i) x) :
    jet directions (fun y => indices.foldl (fun acc i => acc * term i y) (initial y)) x =
      indices.foldl (fun acc i => acc * jet directions (term i) x)
        (jet directions initial x) :=
  jet_foldl_at indices directions (· * ·) (· * ·) (by fun_prop)
    (fun _ _ hf hg => jet_mul_at directions hf hg) hinit hterm

/-- A finite fold preserves jets when its update rule does, in the caller's traversal order. -/
theorem jet_foldl {n : Nat} {ι : Type*} (indices : List ι) (directions : Fin n → E)
    (step : ℝ → ℝ → ℝ) (lifted : Nested ℝ n → Nested ℝ n → Nested ℝ n)
    (hstep : ContDiff ℝ n (fun p : ℝ × ℝ => step p.1 p.2))
    (hjet : ∀ (f g : E → ℝ), ContDiff ℝ n f → ContDiff ℝ n g → ∀ x,
      jet directions (fun y => step (f y) (g y)) x =
        lifted (jet directions f x) (jet directions g x))
    {initial : E → ℝ} {term : ι → E → ℝ} (hinit : ContDiff ℝ n initial)
    (hterm : ∀ i ∈ indices, ContDiff ℝ n (term i)) (x : E) :
    jet directions
      (fun y => indices.foldl (fun acc i => step acc (term i y)) (initial y)) x =
      indices.foldl (fun acc i => lifted acc (jet directions (term i) x))
        (jet directions initial x) := by
  induction indices generalizing initial with
  | nil => rfl
  | cons i indices ih =>
      rw [List.foldl_cons]
      rw [← hjet initial (term i) hinit (hterm i (by simp)) x]
      exact ih (hstep.comp (hinit.prodMk (hterm i (by simp))))
        (fun j hj => hterm j (by simp [hj]))

/-- Ordered summation propagates all derivatives of its entries and initial accumulator. -/
theorem jet_foldl_add {n : Nat} {ι : Type*} (indices : List ι) (directions : Fin n → E)
    {initial : E → ℝ} {term : ι → E → ℝ} (hinit : ContDiff ℝ n initial)
    (hterm : ∀ i ∈ indices, ContDiff ℝ n (term i)) (x : E) :
    jet directions (fun y => indices.foldl (fun acc i => acc + term i y) (initial y)) x =
      indices.foldl (fun acc i => acc + jet directions (term i) x)
        (jet directions initial x) :=
  jet_foldl_add_at indices directions hinit.contDiffAt
    (fun i hi => (hterm i hi).contDiffAt)

/-- Ordered products propagate all derivatives without requiring nonzero factors. -/
theorem jet_foldl_mul {n : Nat} {ι : Type*} (indices : List ι) (directions : Fin n → E)
    {initial : E → ℝ} {term : ι → E → ℝ} (hinit : ContDiff ℝ n initial)
    (hterm : ∀ i ∈ indices, ContDiff ℝ n (term i)) (x : E) :
    jet directions (fun y => indices.foldl (fun acc i => acc * term i y) (initial y)) x =
      indices.foldl (fun acc i => acc * jet directions (term i) x)
        (jet directions initial x) :=
  jet_foldl_mul_at indices directions hinit.contDiffAt
    (fun i hi => (hterm i hi).contDiffAt)

end Runtime.Autograd.Model.Dual
