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

/-- Embedding a constant is valid at every order, not just for one forward pass. -/
theorem jet_const {n : Nat} (directions : Fin n → E) (value : ℝ) (x : E) :
    jet directions (fun _ => value) x = Nested.ofPrimal n value := by
  induction n generalizing value with
  | zero => rfl
  | succ n ih =>
    change Dual.mk _ _ = Dual.mk _ 0
    congr 1
    · exact ih _ _
    · -- The derivative of a constant is the zero function, whose jet is the embedded zero.
      simpa only [fderiv_const_apply, zero_apply, Nested.ofPrimal_zero] using
        ih (directions := Fin.init directions) (value := 0)

/-- All coefficients of the zero function vanish. -/
theorem jet_zero {n : Nat} (directions : Fin n → E) (x : E) :
    jet directions (fun _ => 0) x = 0 := by
  rw [jet_const, Nested.ofPrimal_zero]

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

/-- Runtime comparisons inspect the primal value, not its derivative coefficients. -/
theorem jet_beq_zero {n : Nat} (directions : Fin n → E) (f : E → ℝ) (x : E) :
    (jet directions f x == 0) = (f x == 0) := by
  induction n with
  | zero => rfl
  | succ n ih => exact ih (Fin.init directions)

private theorem nested_unsupported (n : Nat) :
    TorchLean.Numeric.QuotientArithmetic.supported (α := Nested ℝ n) = false := by
  induction n with
  | zero => rfl
  | succ n ih => exact ih

private theorem nested_sub_self_beq_zero {n : Nat} (a : Nested ℝ n) : (a - a == 0) = true := by
  induction n with
  | zero =>
    have h : ∀ r : ℝ, (r - r == 0) = true := by
      intro r
      simp
    exact h a
  | succ n ih => exact ih a.re

private theorem div_mk {n : Nat} (a b da db : Nested ℝ n)
    (hz : (b * b == 0) = false) :
    (Dual.mk a da / Dual.mk b db : Nested ℝ (n + 1)) =
      Dual.mk (a / b) ((da * b - a * db) / (b * b)) := by
  change (if TorchLean.Numeric.QuotientArithmetic.supported
    (α := Dual (Nested ℝ n)) then _ else _) = _
  have hs : TorchLean.Numeric.QuotientArithmetic.supported
      (α := Dual (Nested ℝ n)) = false := nested_unsupported n
  simp only [hs, Bool.false_eq_true, ↓reduceIte, hz, nested_sub_self_beq_zero,
    Bool.not_true, Bool.or_self, Bool.false_and]

private theorem fderiv_div_apply {f g : E → ℝ} {x : E}
    (hf : DifferentiableAt ℝ f x) (hg : DifferentiableAt ℝ g x)
    (hzero : g x ≠ 0) (v : E) :
    fderiv ℝ (fun y => f y / g y) x v =
      (fderiv ℝ f x v * g x - f x * fderiv ℝ g x v) / (g x * g x) := by
  simp only [div_eq_mul_inv]
  rw [fderiv_fun_mul hf (hg.fun_inv hzero)]
  have hinv := ((hasDerivAt_inv hzero).comp_hasFDerivAt x hg.hasFDerivAt).fderiv
  dsimp only [Function.comp_def] at hinv
  rw [hinv]
  simp only [add_apply, smul_apply, smul_eq_mul]
  field_simp
  ring

/-- The runtime quotient rule preserves every derivative coefficient away from a zero denominator.
Both functions need be smooth only near the evaluation point. -/
theorem jet_div_at {n : Nat} (directions : Fin n → E) {f g : E → ℝ} {x : E}
    (hf : ContDiffAt ℝ n f x) (hg : ContDiffAt ℝ n g x) (hzero : g x ≠ 0) :
    jet directions (fun y => f y / g y) x = jet directions f x / jet directions g x := by
  induction n generalizing f g with
  | zero => rfl
  | succ n ih =>
    let ds := Fin.init directions
    let v := directions (Fin.last n)
    let df := fun y => fderiv ℝ f y v
    let dg := fun y => fderiv ℝ g y v
    have hfn : ContDiffAt ℝ n f x := hf.of_le (by simp)
    have hgn : ContDiffAt ℝ n g x := hg.of_le (by simp)
    have hdf : ContDiffAt ℝ n df x :=
      (hf.fderiv_right (by simp)).clm_apply contDiffAt_const
    have hdg : ContDiffAt ℝ n dg x :=
      (hg.fderiv_right (by simp)).clm_apply contDiffAt_const
    have hden := hgn.mul hgn
    have hnum := (hdf.mul hgn).sub (hfn.mul hdg)
    have hden0 := mul_ne_zero hzero hzero
    have hdenjet : jet ds (fun y => g y * g y) x = jet ds g x * jet ds g x :=
      jet_mul_at ds hgn hgn
    have htan : jet ds (fun y => (df y * g y - f y * dg y) / (g y * g y)) x =
        (jet ds df x * jet ds g x - jet ds f x * jet ds dg x) /
          (jet ds g x * jet ds g x) := by
      rw [ih _ hnum hden hden0, jet_sub_at _ (hdf.mul hgn) (hfn.mul hdg),
        jet_mul_at _ hdf hgn, jet_mul_at _ hfn hdg, hdenjet]
    have hz : (jet ds g x * jet ds g x == 0) = false := by
      rw [← hdenjet, jet_beq_zero]
      simpa only [beq_eq_false_iff_ne] using hden0
    have heq : (fun y => fderiv ℝ (fun z => f z / g z) y v) =ᶠ[𝓝 x]
        fun y => (df y * g y - f y * dg y) / (g y * g y) := by
      filter_upwards [hf.eventually (by simp), hg.eventually (by simp),
        hg.continuousAt.eventually_ne hzero] with y hfy hgy hy
      exact fderiv_div_apply (hfy.differentiableAt (by simp))
        (hgy.differentiableAt (by simp)) hy v
    change Dual.mk _ _ = Dual.mk (jet ds f x) (jet ds df x) /
      Dual.mk (jet ds g x) (jet ds dg x)
    rw [div_mk _ _ _ _ hz]
    congr 1
    · exact ih ds hfn hgn hzero
    · exact (jet_congr ds heq).trans htan

/-- Strict positivity of a jet depends only on its primal value. -/
theorem jet_pos {n : Nat} (directions : Fin n → E) (f : E → ℝ) (x : E) :
    0 < jet directions f x ↔ 0 < f x := by
  induction n with
  | zero => rfl
  | succ n ih => exact ih (Fin.init directions)

private theorem jet_sigmoid_deriv_at {n : Nat} (directions : Fin n → E) {f : E → ℝ}
    {x : E} (hf : ContDiffAt ℝ n f x) :
    jet directions (fun y => Activation.Math.sigmoidDerivSpec (f y)) x =
      TorchLean.Numeric.sigmoidDeriv (jet directions f x) := by
  have hformula (a : ℝ) : Activation.Math.sigmoidDerivSpec a =
      Real.exp a / ((1 + Real.exp a) * (1 + Real.exp a)) := by
    rw [Proofs.sigmoid_deriv_spec_eq, Proofs.sigmoid_eq_exp_div]
    have hd : (1 : ℝ) + Real.exp a ≠ 0 := by positivity
    field_simp
    ring
  have hnegative (a : ℝ) : Activation.Math.sigmoidDerivSpec a =
      Real.exp (-a) / ((1 + Real.exp (-a)) * (1 + Real.exp (-a))) := by
    rw [hformula, Real.exp_neg]
    have he := Real.exp_ne_zero a
    have hd : (1 : ℝ) + Real.exp a ≠ 0 := by positivity
    field_simp
    ring
  have hjet (g : E → ℝ) (hg : ContDiffAt ℝ n g x) :
      jet directions (fun y => Real.exp (g y) /
        ((1 + Real.exp (g y)) * (1 + Real.exp (g y)))) x =
      let z := MathFunctions.exp (jet directions g x)
      z / ((1 + z) * (1 + z)) := by
    have hz := hg.exp
    have hd : ContDiffAt ℝ n (fun y => 1 + Real.exp (g y)) x := contDiffAt_const.add hz
    have hzero : (1 + Real.exp (g x)) * (1 + Real.exp (g x)) ≠ 0 := by positivity
    rw [jet_div_at _ hz (hd.mul hd) hzero, jet_mul_at _ hd hd,
      jet_add_at _ contDiffAt_const hz, jet_one, jet_exp_comp_at _ hg]
  unfold TorchLean.Numeric.sigmoidDeriv
  simp only [TorchLean.Numeric.Internal.applySlope, nested_unsupported,
    Bool.not_false, Bool.false_eq_true, ↓reduceIte]
  by_cases hx : 0 < f x
  · rw [ite_eq_left ((jet_pos directions f x).mpr hx), ← jet_neg]
    rw [show (fun y => Activation.Math.sigmoidDerivSpec (f y)) =
      (fun y => Real.exp (-f y) / ((1 + Real.exp (-f y)) * (1 + Real.exp (-f y)))) from
      funext fun y => hnegative (f y)]
    exact hjet _ hf.neg
  · rw [ite_eq_right (fun h => hx ((jet_pos directions f x).mp h))]
    rw [show (fun y => Activation.Math.sigmoidDerivSpec (f y)) =
      (fun y => Real.exp (f y) / ((1 + Real.exp (f y)) * (1 + Real.exp (f y)))) from
      funext fun y => hformula (f y)]
    exact hjet _ hf

/-- The analytic sigmoid rule preserves every mixed derivative of a smooth real input function.

The scalar range repair is inactive over exact reals; the runtime's weighted pullback reduces
to the derivative used by mathlib. No floating-point accuracy claim is made here.
-/
theorem jet_sigmoid_comp_at {n : Nat} (directions : Fin n → E) {f : E → ℝ}
    {x : E} (hf : ContDiffAt ℝ n f x) :
    jet directions (fun y => Real.sigmoid (f y)) x = Context.sigmoid (jet directions f x) := by
  induction n generalizing f with
  | zero =>
    change Real.sigmoid (f x) = Activation.Math.sigmoidSpec (f x)
    rw [Proofs.sigmoid_eq_inv_exp, Real.sigmoid_def]
  | succ n ih =>
    let v := directions (Fin.last n)
    have hfn : ContDiffAt ℝ n f x := hf.of_le (by simp)
    have hf' : ContDiffAt ℝ n (fun y => fderiv ℝ f y v) x :=
      (hf.fderiv_right (by simp)).clm_apply contDiffAt_const
    have hsig : (Activation.Math.sigmoidSpec : ℝ → ℝ) = Real.sigmoid := by
      funext a
      rw [Proofs.sigmoid_eq_inv_exp, Real.sigmoid_def]
    have ht : ContDiffAt ℝ n (fun y => Real.sigmoid (f y)) x :=
      (contDiff_sigmoid.of_le (by simp)).contDiffAt.comp x hfn
    have hs : ContDiffAt ℝ n (fun y => Activation.Math.sigmoidDerivSpec (f y)) x := by
      simp_rw [Proofs.sigmoid_deriv_spec_eq, hsig]
      exact ht.mul (contDiffAt_const.sub ht)
    have heq : (fun y => fderiv ℝ (fun z => Real.sigmoid (f z)) y v) =ᶠ[𝓝 x]
        fun y => Activation.Math.sigmoidDerivSpec (f y) * fderiv ℝ f y v := by
      filter_upwards [hf.eventually (by simp)] with y hfy
      have h := (Proofs.sigmoid_deriv_correct (f y)).comp_hasFDerivAt y
        (hfy.differentiableAt (by simp)).hasFDerivAt
      simpa only [Function.comp_def, Proofs.sigmoid_eq_inv_exp, Real.sigmoid_def,
        smul_apply, smul_eq_mul] using congrArg (fun d => d v) h.fderiv
    change Dual.mk _ _ = Dual.mk _ _
    congr 1
    · exact ih _ hfn
    · rw [jet_congr _ heq, jet_mul_at _ hs hf', jet_sigmoid_deriv_at _ hfn]
      simp only [TorchLean.Numeric.sigmoidBackward, TorchLean.Numeric.Internal.applySlope,
        TorchLean.Numeric.sigmoidDeriv, nested_unsupported, Bool.not_false,
        Bool.false_eq_true, ↓reduceIte, jet, v]
      rfl

private theorem jet_tanh_deriv_at {n : Nat} (directions : Fin n → E) {f : E → ℝ}
    {x : E} (hf : ContDiffAt ℝ n f x) :
    jet directions (fun y => 1 - Real.tanh (f y) * Real.tanh (f y)) x =
      TorchLean.Numeric.tanhDeriv (jet directions f x) := by
  have hformula (a : ℝ) : 1 - Real.tanh a * Real.tanh a =
      let z := Real.exp a
      let c := (z + z) / (1 + z * z)
      c * c := by
    rw [Proofs.tanh_exp_eq, Real.exp_neg]
    have he := Real.exp_ne_zero a
    have hd : (1 : ℝ) + Real.exp a * Real.exp a ≠ 0 := by positivity
    field_simp
    ring
  have hjet (g : E → ℝ) (hg : ContDiffAt ℝ n g x) :
      jet directions (fun y =>
        let z := Real.exp (g y)
        let c := (z + z) / (1 + z * z)
        c * c) x =
      let z := MathFunctions.exp (jet directions g x)
      let c := (z + z) / (1 + z * z)
      c * c := by
    have hz := hg.exp
    have hn := hz.add hz
    have hd : ContDiffAt ℝ n (fun y => 1 + Real.exp (g y) * Real.exp (g y)) x :=
      contDiffAt_const.add (hz.mul hz)
    have hzero : 1 + Real.exp (g x) * Real.exp (g x) ≠ 0 := by positivity
    have hc := hn.fun_div hd hzero
    dsimp only
    rw [jet_mul_at _ hc hc]
    have hquot := jet_div_at directions hn hd hzero
    simp only [jet_add_at _ hz hz, jet_add_at _ contDiffAt_const (hz.mul hz),
      jet_mul_at _ hz hz, jet_one, jet_exp_comp_at _ hg] at hquot
    rw [hquot]
  unfold TorchLean.Numeric.tanhDeriv
  simp only [TorchLean.Numeric.Internal.applySlope, nested_unsupported,
    Bool.not_false, ↓reduceIte]
  by_cases hx : 0 < f x
  · rw [ite_eq_left ((jet_pos directions f x).mpr hx), ← jet_neg]
    have heq : (fun y => 1 - Real.tanh (f y) * Real.tanh (f y)) =
        (fun y =>
          let z := Real.exp (-f y)
          let c := (z + z) / (1 + z * z)
          c * c) := by
      funext y
      simpa only [Real.tanh_neg, neg_mul_neg] using hformula (-f y)
    rw [heq]
    exact hjet _ hf.neg
  · rw [ite_eq_right (fun h => hx ((jet_pos directions f x).mp h))]
    have heq : (fun y => 1 - Real.tanh (f y) * Real.tanh (f y)) =
        (fun y =>
          let z := Real.exp (f y)
          let c := (z + z) / (1 + z * z)
          c * c) := funext fun y => hformula (f y)
    rw [heq]
    exact hjet _ hf

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
      simpa only [Function.comp_def, Activation.Math.tanhSpec, Proofs.tanh_deriv_spec_eq,
        Proofs.mathfunc_tanh_eq_rtanh, smul_apply, smul_eq_mul] using
        congrArg (fun d => d v) h.fderiv
    change Dual.mk _ _ = Dual.mk _ _
    congr 1
    · exact ih _ hfn
    · rw [jet_congr _ heq, jet_mul_at _ (contDiffAt_const.sub (ht.mul ht)) hf',
        jet_tanh_deriv_at _ hfn]
      simp only [TorchLean.Numeric.tanhBackward, TorchLean.Numeric.Internal.applySlope,
        TorchLean.Numeric.tanhDeriv, nested_unsupported, Bool.not_false,
        ↓reduceIte, jet, v]
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
