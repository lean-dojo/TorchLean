/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.Autograd.Dual

/-!
# Higher derivatives on local domains

Division, logarithms, and square roots need not be smooth everywhere. Their jet laws use smoothness
near the evaluation point and a nonzero denominator or argument there. Continuity gives the same
domain condition nearby, where mathlib's derivative rules identify the remaining jet coefficients.

The quotient proof follows the runtime implementation, including its floating-point safeguards.
Over exact reals, the checks for non-finite intermediate values are always false; no floating-point
execution is identified with exact-real arithmetic here.
-/

public section

open scoped ContDiff Topology

namespace Runtime.Autograd.Model.Dual

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Runtime comparisons inspect the primal value, not its derivative coefficients. -/
theorem jet_beq_zero {n : Nat} (ds : Fin n → E) (f : E → ℝ) (x : E) :
    (jet ds f x == 0) = (f x == 0) := by
  induction n with
  | zero => rfl
  | succ n ih => exact ih (Fin.init ds)

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

/-- The runtime logarithm preserves mixed derivatives at nonzero inputs.
This uses mathlib's real logarithm, which is smooth for negative as well as positive inputs. -/
theorem jet_log_comp_at {n : Nat} (directions : Fin n → E) {f : E → ℝ} {x : E}
    (hf : ContDiffAt ℝ n f x) (hzero : f x ≠ 0) :
    jet directions (fun y => Real.log (f y)) x = MathFunctions.log (jet directions f x) := by
  induction n generalizing f with
  | zero => rfl
  | succ n ih =>
    let v := directions (Fin.last n)
    have hfn : ContDiffAt ℝ n f x := hf.of_le (by simp)
    have hf' : ContDiffAt ℝ n (fun y => fderiv ℝ f y v) x :=
      (hf.fderiv_right (by simp)).clm_apply contDiffAt_const
    have heq : (fun y => fderiv ℝ (fun z => Real.log (f z)) y v) =ᶠ[𝓝 x]
        fun y => fderiv ℝ f y v / f y := by
      filter_upwards [hf.eventually (by simp), hf.continuousAt.eventually_ne hzero]
        with y hfy hy
      rw [fderiv.log (hfy.differentiableAt (by simp)) hy]
      simp only [smul_apply, smul_eq_mul, div_eq_mul_inv, mul_comm]
    change Dual.mk _ _ = Dual.mk _ _
    congr 1
    · exact ih _ hfn hzero
    · rw [jet_congr _ heq, jet_div_at _ hf' hfn hzero]
      rfl

/-- Direct scalar logarithm evaluation on seeded inputs away from zero. -/
theorem jet_log {n : Nat} (directions : Fin n → ℝ) {x : ℝ} (hx : x ≠ 0) :
    jet directions Real.log x = MathFunctions.log (Nested.seed directions x) := by
  simpa only [jet_id] using jet_log_comp_at directions (f := fun y => y) contDiffAt_id hx

/-- Direct reciprocal evaluation on seeded inputs away from zero. -/
theorem jet_one_div {n : Nat} (directions : Fin n → ℝ) {x : ℝ} (hx : x ≠ 0) :
    jet directions (fun y => 1 / y) x = 1 / Nested.seed directions x := by
  simpa only [jet_one, jet_id] using jet_div_at directions
    (f := fun _ : ℝ => 1) (g := fun y => y) contDiffAt_const contDiffAt_id hx

/-- The square-root branch depends only on the primal input. -/
theorem jet_pos {n : Nat} (ds : Fin n → E) (f : E → ℝ) (x : E) :
    0 < jet ds f x ↔ 0 < f x := by
  induction n with
  | zero => rfl
  | succ n ih => exact ih (Fin.init ds)

private theorem ofPrimal_natCast (n k : Nat) :
    Nested.ofPrimal n (k : ℝ) = (k : Nested ℝ n) := by
  induction n with
  | zero => rfl
  | succ n ih => exact congrArg (fun a => Dual.mk a 0) ih

/-- The runtime square root preserves mixed derivatives at nonzero inputs.
For negative exact-real inputs, both the runtime and mathlib use the locally constant zero branch.
No derivative claim at zero or about native floating-point square roots is made. -/
theorem jet_sqrt_comp_at {n : Nat} (directions : Fin n → E) {f : E → ℝ} {x : E}
    (hf : ContDiffAt ℝ n f x) (hzero : f x ≠ 0) :
    jet directions (fun y => Real.sqrt (f y)) x = MathFunctions.sqrt (jet directions f x) := by
  induction n generalizing f with
  | zero => rfl
  | succ n ih =>
    let ds := Fin.init directions
    let v := directions (Fin.last n)
    have hfn : ContDiffAt ℝ n f x := hf.of_le (by simp)
    have hf' : ContDiffAt ℝ n (fun y => fderiv ℝ f y v) x :=
      (hf.fderiv_right (by simp)).clm_apply contDiffAt_const
    change Dual.mk _ _ = Dual.mk _ (if 0 < jet ds f x then _ else 0)
    congr 1
    · exact ih _ hfn hzero
    · simp only [jet_pos]
      by_cases hpos : 0 < f x
      · simp only [hpos, ↓reduceIte]
        have heq : (fun y => fderiv ℝ (fun z => Real.sqrt (f z)) y v) =ᶠ[𝓝 x]
            fun y => fderiv ℝ f y v / (2 * Real.sqrt (f y)) := by
          filter_upwards [hf.eventually (by simp), hf.continuousAt.eventually_ne hzero]
            with y hfy hy
          rw [fderiv_sqrt (hfy.differentiableAt (by simp)) hy]
          simp only [smul_apply, smul_eq_mul, div_eq_mul_inv, one_mul, mul_comm]
        rw [jet_congr _ heq, jet_div_at _ hf' (contDiffAt_const.mul (hfn.sqrt hzero))
          (mul_ne_zero two_ne_zero (Real.sqrt_pos.mpr hpos).ne'),
          jet_mul_at _ contDiffAt_const (hfn.sqrt hzero), jet_const, ih _ hfn hzero]
        rw [show (2 : ℝ) = ((2 : Nat) : ℝ) from rfl, ofPrimal_natCast]
        rfl
      · simp only [hpos, ↓reduceIte]
        have hneg : f x < 0 := lt_of_le_of_ne (le_of_not_gt hpos) hzero
        have heq : (fun y => fderiv ℝ (fun z => Real.sqrt (f z)) y v) =ᶠ[𝓝 x]
            fun _ => 0 := by
          filter_upwards [hf.eventually (by simp), hf.continuousAt.eventually_lt_const hneg]
            with y hfy hy
          rw [fderiv_sqrt (hfy.differentiableAt (by simp)) hy.ne,
            Real.sqrt_eq_zero_of_nonpos hy.le]
          simp
        rw [jet_congr _ heq, jet_const, Nested.ofPrimal_zero]

/-- Direct square-root evaluation on seeded inputs away from zero. -/
theorem jet_sqrt {n : Nat} (directions : Fin n → ℝ) {x : ℝ} (hx : x ≠ 0) :
    jet directions Real.sqrt x = MathFunctions.sqrt (Nested.seed directions x) := by
  simpa only [jet_id] using jet_sqrt_comp_at directions (f := fun y => y) contDiffAt_id hx

end Runtime.Autograd.Model.Dual
