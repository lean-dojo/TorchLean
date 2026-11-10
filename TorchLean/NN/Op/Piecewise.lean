/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tactic.OpDeriv
public import TorchLean.Tensor.Op.Table.Elementwise
public import TorchLean.Tensor.Elementwise.Real
public import Mathlib.Analysis.SpecialFunctions.Complex.LogDeriv
import Mathlib.Analysis.Calculus.Deriv.Prod

/-!
# Derivatives of copysign, fmax, fmin, atan2, max and min, and conventions at kinks

We prove that the stored backward of `copysign`, `fmax`, `fmin` and `atan2` is the adjoint of the
Fréchet derivative of the real forward on the exact set where that derivative exists and the
stored backward equals its adjoint, and we record the stored backward at every input in a
`convention` certificate. Over the reals, `fmax` and `fmin` are `max` and `min`, `copysign a b`
is `|a|` with the sign of `b` (with `copysign a 0 = |a|`), and `atan2 y x` is the argument of
`x + y i`.

Each forward agrees near a point of its exact set with a smooth branch: a linear map for `fmax`,
`fmin` and `copysign`, and the imaginary part of the complex logarithm on the slit plane for
`atan2`. We prove the scalar derivatives once, lift them to tensors with
`HasPullbackAt.pointwise₂` and the `@[fun_prop]` rules `HasPullbackAt.fmax`, `fmin`, `copysign`
and `atan2`, state the stored backwards over the reals as `@[op_deriv_simps]` equations, and
close the `derivative` certificates with `op_deriv` and the `convention` certificates with
`simp only`.

For `max` and `min` we prove the joint `derivative` certificate off ties from
`TorchLean.Op.pointwise₂_backward_eq_adjoint`, and at a tie we record that the stored backward
gives each input half of the cotangent. For `relu`, `abs`, `leakyReLU`, `elu` and `clamp` we state
the stored backward at every input, and for `smoothAbs` at a zero entry, in `convention`
certificates whose docstrings say which one-sided derivative, or which mean of the two, the stored
value is `δ` times at each kink, a point where both one-sided derivatives exist and differ.
-/

@[expose] public section

open Filter Topology

namespace TorchLean

namespace Context

/-- Over the reals, `copysign x y` is `-|x|` where `y` is negative and `|x|` elsewhere. -/
theorem copysign_real (x y : ℝ) : Context.copysign x y = if y < 0 then -|x| else |x| := by
  show (if 0 > y then -|x| else |x|) = _
  by_cases h : y < 0 <;> simp [h]

/-- Over the reals, `copysign x y` is `-|x|` for negative `y`. -/
theorem copysign_of_neg {y : ℝ} (x : ℝ) (hy : y < 0) : Context.copysign x y = -|x| := by
  rw [copysign_real, ite_eq_left hy]

/-- Over the reals, `copysign x y` is `|x|` for nonnegative `y`. -/
theorem copysign_of_nonneg {y : ℝ} (x : ℝ) (hy : 0 ≤ y) : Context.copysign x y = |x| := by
  rw [copysign_real, ite_eq_right (not_lt.mpr hy)]

/-- Over the reals, the stored tangent of `copysign x y` along `x` is `sign x` times the sign
factor of `y` (`-1` for negative `y`, `1` elsewhere) times `dx`. -/
theorem copysignTangent_real (x y dx : ℝ) :
    Context.copysignTangent x y dx = Real.sign x * (if y < 0 then -1 else 1) * dx := by
  unfold Context.copysignTangent
  rw [copysign_real, copysign_real, abs_one]
  rcases lt_trichotomy x 0 with hx | rfl | hx <;> rcases lt_or_ge y 0 with hy | hy
  · norm_num [hx, hy, hx.ne, Real.sign_of_neg hx]
  · norm_num [hx, not_lt.mpr hy, hx.ne, Real.sign_of_neg hx]
  · simp
  · simp
  · norm_num [not_lt.mpr hx.le, hy, hx.ne', Real.sign_of_pos hx]
  · norm_num [not_lt.mpr hx.le, not_lt.mpr hy, hx.ne', Real.sign_of_pos hx]

/-- Over the reals, the test `fmaxTakesFirst x y`, by which `fmax x y` takes its first operand,
holds exactly where `y ≤ x`. -/
theorem fmaxTakesFirst_real (x y : ℝ) : Context.fmaxTakesFirst x y = decide (y ≤ x) := by
  rcases lt_trichotomy x y with h | rfl | h
  · simp [Context.fmaxTakesFirst, Context.gtBool, h, not_le.mpr h]
  · simp [Context.fmaxTakesFirst, Context.gtBool]
    exact le_total _ _
  · simp [Context.fmaxTakesFirst, Context.gtBool, h, lt_asymm h, h.le]

/-- Over the reals, the test `fminTakesFirst x y`, by which `fmin x y` takes its first operand,
holds exactly where `x ≤ y`. -/
theorem fminTakesFirst_real (x y : ℝ) : Context.fminTakesFirst x y = decide (x ≤ y) := by
  rcases lt_trichotomy x y with h | rfl | h
  · simp [Context.fminTakesFirst, Context.gtBool, h, lt_asymm h, h.le]
  · simp [Context.fminTakesFirst, Context.gtBool]
    exact le_total _ _
  · simp [Context.fminTakesFirst, Context.gtBool, h, not_le.mpr h]

end Context

namespace Numeric

/-- Over the reals, the stored angle tangent is `(x dy - y dx) / (x² + y²)`, zero at the
origin. -/
theorem angleTangent_real (x y dx dy : ℝ) :
    angleTangent x y dx dy = (x * dy - y * dx) / (x * x + y * y) := by
  by_cases horigin : x = 0 ∧ y = 0
  · obtain ⟨rfl, rfl⟩ := horigin
    simp [angleTangent]
  have hs : max |x| |y| ≠ 0 := by
    intro hs
    have hx : x = 0 := abs_nonpos_iff.mp (by simpa [hs] using le_max_left |x| |y|)
    have hy : y = 0 := abs_nonpos_iff.mp (by simpa [hs] using le_max_right |x| |y|)
    exact horigin ⟨hx, hy⟩
  simp only [angleTangent, QuotientArithmetic.supported_real, Bool.not_false, ↓reduceIte]
  rw [ite_eq_right (by simp [horigin])]
  simp only [div_mul_div_comm, ← sub_div, ← add_div]
  exact div_div_div_cancel_right₀ (mul_ne_zero hs hs) _ _

end Numeric

namespace Op

/-- A function on `ℝ × ℝ` that agrees near `p` with the linear map `q ↦ d₀ q.1 + d₁ q.2` has that
map as derivative at `p`. -/
theorem hasFDerivAt_of_eventuallyEq_linear {f : ℝ × ℝ → ℝ} {p : ℝ × ℝ} {d₀ d₁ : ℝ}
    (h : ∀ᶠ q in 𝓝 p, f q = d₀ * q.1 + d₁ * q.2) :
    HasFDerivAt f (d₀ • ContinuousLinearMap.fst ℝ ℝ ℝ + d₁ • ContinuousLinearMap.snd ℝ ℝ ℝ) p :=
  (d₀ • ContinuousLinearMap.fst ℝ ℝ ℝ +
    d₁ • ContinuousLinearMap.snd ℝ ℝ ℝ).hasFDerivAt.congr_of_eventuallyEq h

/-- Away from the diagonal, `max` on `ℝ × ℝ` has the projection onto the larger coordinate as
derivative. -/
theorem hasFDerivAt_max_of_ne {a b : ℝ} (h : a ≠ b) :
    HasFDerivAt (fun q : ℝ × ℝ => Max.max q.1 q.2)
      ((if b ≤ a then 1 else 0 : ℝ) • ContinuousLinearMap.fst ℝ ℝ ℝ +
        (if b ≤ a then 0 else 1 : ℝ) • ContinuousLinearMap.snd ℝ ℝ ℝ) (a, b) := by
  rcases h.lt_or_gt with h | h
  · rw [ite_eq_right (not_le.mpr h), ite_eq_right (not_le.mpr h)]
    refine hasFDerivAt_of_eventuallyEq_linear ?_
    have hq : ∀ᶠ q : ℝ × ℝ in 𝓝 (a, b), q.1 < q.2 :=
      continuous_fst.continuousAt.eventually_lt continuous_snd.continuousAt h
    filter_upwards [hq] with q hq
    rw [max_eq_right hq.le]
    ring
  · rw [ite_eq_left h.le, ite_eq_left h.le]
    refine hasFDerivAt_of_eventuallyEq_linear ?_
    have hq : ∀ᶠ q : ℝ × ℝ in 𝓝 (a, b), q.2 < q.1 :=
      continuous_snd.continuousAt.eventually_lt continuous_fst.continuousAt h
    filter_upwards [hq] with q hq
    rw [max_eq_left hq.le]
    ring

/-- Away from the diagonal, `min` on `ℝ × ℝ` has the projection onto the smaller coordinate as
derivative. -/
theorem hasFDerivAt_min_of_ne {a b : ℝ} (h : a ≠ b) :
    HasFDerivAt (fun q : ℝ × ℝ => Min.min q.1 q.2)
      ((if a ≤ b then 1 else 0 : ℝ) • ContinuousLinearMap.fst ℝ ℝ ℝ +
        (if a ≤ b then 0 else 1 : ℝ) • ContinuousLinearMap.snd ℝ ℝ ℝ) (a, b) := by
  rcases h.lt_or_gt with h | h
  · rw [ite_eq_left h.le, ite_eq_left h.le]
    refine hasFDerivAt_of_eventuallyEq_linear ?_
    have hq : ∀ᶠ q : ℝ × ℝ in 𝓝 (a, b), q.1 < q.2 :=
      continuous_fst.continuousAt.eventually_lt continuous_snd.continuousAt h
    filter_upwards [hq] with q hq
    rw [min_eq_left hq.le]
    ring
  · rw [ite_eq_right (not_le.mpr h), ite_eq_right (not_le.mpr h)]
    refine hasFDerivAt_of_eventuallyEq_linear ?_
    have hq : ∀ᶠ q : ℝ × ℝ in 𝓝 (a, b), q.2 < q.1 :=
      continuous_snd.continuousAt.eventually_lt continuous_fst.continuousAt h
    filter_upwards [hq] with q hq
    rw [min_eq_right hq.le]
    ring

/-- Where both entries are nonzero, the real `copysign` on `ℝ × ℝ` has derivative
`sign a` times the sign factor of `b` in the first coordinate and zero in the second. -/
theorem hasFDerivAt_copysign {a b : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) :
    HasFDerivAt (fun q : ℝ × ℝ => Context.copysign q.1 q.2)
      ((Real.sign a * (if b < 0 then -1 else 1)) • ContinuousLinearMap.fst ℝ ℝ ℝ +
        (0 : ℝ) • ContinuousLinearMap.snd ℝ ℝ ℝ) (a, b) := by
  refine hasFDerivAt_of_eventuallyEq_linear ?_
  have h₁ : ∀ᶠ q : ℝ × ℝ in 𝓝 (a, b), |q.1| = Real.sign a * q.1 := by
    rcases ha.lt_or_gt with ha | ha
    · have hq : ∀ᶠ q : ℝ × ℝ in 𝓝 (a, b), q.1 < 0 :=
        continuous_fst.continuousAt.eventually_lt continuousAt_const ha
      filter_upwards [hq] with q hq
      rw [abs_of_neg hq, Real.sign_of_neg ha, neg_one_mul]
    · have hq : ∀ᶠ q : ℝ × ℝ in 𝓝 (a, b), 0 < q.1 :=
        continuousAt_const.eventually_lt continuous_fst.continuousAt ha
      filter_upwards [hq] with q hq
      rw [abs_of_pos hq, Real.sign_of_pos ha, one_mul]
  have h₂ : ∀ᶠ q : ℝ × ℝ in 𝓝 (a, b),
      Context.copysign q.1 q.2 = (if b < 0 then -1 else 1) * |q.1| := by
    rcases hb.lt_or_gt with hb | hb
    · have hq : ∀ᶠ q : ℝ × ℝ in 𝓝 (a, b), q.2 < 0 :=
        continuous_snd.continuousAt.eventually_lt continuousAt_const hb
      filter_upwards [hq] with q hq
      rw [Context.copysign_of_neg _ hq, ite_eq_left hb, neg_one_mul]
    · have hq : ∀ᶠ q : ℝ × ℝ in 𝓝 (a, b), 0 < q.2 :=
        continuousAt_const.eventually_lt continuous_snd.continuousAt hb
      filter_upwards [hq] with q hq
      rw [Context.copysign_of_nonneg _ hq.le, ite_eq_right (not_lt.mpr hb.le), one_mul]
  filter_upwards [h₁, h₂] with q hq₁ hq₂
  rw [hq₂, hq₁]
  ring

/-- At a nonzero `a`, the real `copysign t b` has derivative `sign a` times the sign factor of `b`
in `t`. -/
theorem hasDerivAt_copysign_left {a : ℝ} (b : ℝ) (ha : a ≠ 0) :
    HasDerivAt (fun t => Context.copysign t b) (Real.sign a * (if b < 0 then -1 else 1)) a := by
  have habs : HasDerivAt (fun t : ℝ => |t|) (Real.sign a) a := by
    rcases ha.lt_or_gt with ha | ha
    · rw [Real.sign_of_neg ha]
      exact hasDerivAt_abs_neg ha
    · rw [Real.sign_of_pos ha]
      exact hasDerivAt_abs_pos ha
  rcases lt_or_ge b 0 with hb | hb
  · rw [ite_eq_left hb]
    exact (habs.neg.congr_deriv (by ring)).congr_of_eventuallyEq
      (Eventually.of_forall fun t => Context.copysign_of_neg t hb)
  · rw [ite_eq_right (not_lt.mpr hb), mul_one]
    exact habs.congr_of_eventuallyEq
      (Eventually.of_forall fun t => Context.copysign_of_nonneg t hb)

/-- Where `a = 0` or `b ≠ 0`, the real `copysign a t` has derivative zero in `t` at `b`. -/
theorem hasDerivAt_copysign_right {a b : ℝ} (h : a = 0 ∨ b ≠ 0) :
    HasDerivAt (fun t => Context.copysign a t) 0 b := by
  rcases h with rfl | hb
  · refine (hasDerivAt_const b (0 : ℝ)).congr_of_eventuallyEq (Eventually.of_forall fun t => ?_)
    simp only [Context.copysign_real, abs_zero, neg_zero, ite_self]
  rcases hb.lt_or_gt with hb | hb
  · refine (hasDerivAt_const b (-|a|)).congr_of_eventuallyEq ?_
    filter_upwards [Iio_mem_nhds hb] with t ht
    exact Context.copysign_of_neg a ht
  · refine (hasDerivAt_const b |a|).congr_of_eventuallyEq ?_
    filter_upwards [Ioi_mem_nhds hb] with t ht
    exact Context.copysign_of_nonneg a (le_of_lt ht)

/-- Off the closed negative real axis, the real `atan2 y x`, the argument of `x + y i`, has
derivative `(x dy - y dx) / (x² + y²)` on `ℝ × ℝ` at `(y, x)`. -/
theorem hasFDerivAt_atan2 {y x : ℝ} (h : y ≠ 0 ∨ 0 < x) :
    HasFDerivAt (fun q : ℝ × ℝ => InverseTrig.atan2 q.1 q.2)
      ((x / (x * x + y * y)) • ContinuousLinearMap.fst ℝ ℝ ℝ +
        (-(y / (x * x + y * y))) • ContinuousLinearMap.snd ℝ ℝ ℝ) (y, x) := by
  let L : ℝ × ℝ →L[ℝ] ℂ := (Complex.equivRealProdCLM.symm : ℝ × ℝ →L[ℝ] ℂ).comp
    ((ContinuousLinearMap.snd ℝ ℝ ℝ).prod (ContinuousLinearMap.fst ℝ ℝ ℝ))
  have hL : ∀ q : ℝ × ℝ, L q = ⟨q.2, q.1⟩ := fun q => rfl
  have hz : L (y, x) ∈ Complex.slitPlane := by
    rw [hL, Complex.mem_slitPlane_iff]
    exact h.symm
  have him := Complex.imCLM.hasFDerivAt.comp (y, x)
    ((Complex.hasStrictFDerivAt_log_real hz).hasFDerivAt.comp (y, x) L.hasFDerivAt)
  refine (him.congr_fderiv (ContinuousLinearMap.ext fun q => ?_)).congr_of_eventuallyEq
    (Eventually.of_forall fun q => ?_)
  · simp only [ContinuousLinearMap.comp_apply, _root_.smul_apply, one_apply_eq_self,
      smul_eq_mul, Complex.imCLM_apply, hL, Complex.mul_im, Complex.inv_re, Complex.inv_im,
      Complex.normSq_mk, _root_.add_apply, ContinuousLinearMap.coe_fst',
      ContinuousLinearMap.coe_snd']
    ring
  · simp only [Function.comp_apply, Complex.imCLM_apply, Complex.log_im, hL]
    rfl

/-- Where `y ≠ 0` or `x ≠ 0`, the real `atan2 y t` has derivative `-y / (x² + y²)` in `t` at
`x`. -/
theorem hasDerivAt_atan2_right {y x : ℝ} (h : y ≠ 0 ∨ x ≠ 0) :
    HasDerivAt (fun t => InverseTrig.atan2 y t) (-(y / (x * x + y * y))) x := by
  by_cases hs : y ≠ 0 ∨ 0 < x
  · refine ((hasFDerivAt_atan2 hs).comp_hasDerivAt x
      (HasDerivAt.prodMk (hasDerivAt_const x y) (hasDerivAt_id' x))).congr_deriv ?_
    simp
  push Not at hs
  obtain ⟨rfl, hx⟩ := hs
  have hx : x < 0 := lt_of_le_of_ne hx (h.resolve_left fun h₀ => h₀ rfl)
  rw [zero_div, neg_zero]
  refine (hasDerivAt_const x Real.pi).congr_of_eventuallyEq ?_
  filter_upwards [Iio_mem_nhds hx] with t ht
  exact Complex.arg_ofReal_of_neg ht

end Op

namespace Tensor

variable {s : Shape}

/-- Over the reals, the stored pullbacks of `fmax a b` route `g` to `a` where `b ≤ a` and to `b`
elsewhere. -/
@[op_deriv_simps] theorem fmaxBackward_real (a b g : Tensor ℝ s) :
    fmaxBackward a b g = (zipWith3 (fun a b g => if b ≤ a then g else 0) a b g,
      zipWith3 (fun a b g => if b ≤ a then 0 else g) a b g) := by
  simp only [fmaxBackward, Context.fmaxTakesFirst_real, decide_eq_true_eq]

/-- Over the reals, the stored pullbacks of `fmin a b` route `g` to `a` where `a ≤ b` and to `b`
elsewhere. -/
@[op_deriv_simps] theorem fminBackward_real (a b g : Tensor ℝ s) :
    fminBackward a b g = (zipWith3 (fun a b g => if a ≤ b then g else 0) a b g,
      zipWith3 (fun a b g => if a ≤ b then 0 else g) a b g) := by
  simp only [fminBackward, Context.fminTakesFirst_real, decide_eq_true_eq]

/-- Over the reals, the stored pullback of `copysign a b` to `a` is `sign a` times the sign factor
of `b` times `g`, and the pullback to `b` is zero. -/
@[op_deriv_simps] theorem copysignBackward_real (a b g : Tensor ℝ s) :
    copysignBackward a b g =
      (zipWith3 (fun a b g => Real.sign a * (if b < 0 then -1 else 1) * g) a b g, 0) := by
  refine Prod.ext (Rep.ext fun c => ?_) (Rep.ext fun c => ?_)
  · simp only [copysignBackward, zipWith3_apply, Context.copysignTangent_real]
  · simp only [copysignBackward, Tensor.map_apply, Rep.zero_apply]

/-- Over the reals, the stored pullbacks of `atan2 y x` are `g x / (x² + y²)` and
`-(g y / (x² + y²))`, zero at a coordinate where both entries are zero. -/
@[op_deriv_simps] theorem atan2Backward_real (y x g : Tensor ℝ s) :
    atan2Backward y x g = (g * x / (x * x + y * y), -(g * y / (x * x + y * y))) := by
  refine Prod.ext (Rep.ext fun c => ?_) (Rep.ext fun c => ?_)
  · simp only [atan2Backward, zipWith3_apply, Numeric.angleTangent_real, Tensor.div_apply,
      Tensor.mul_apply, Tensor.add_apply]
    ring
  · simp only [atan2Backward, zipWith3_apply, Numeric.angleTangent_real, Rep.neg_apply,
      Tensor.div_apply, Tensor.mul_apply, Tensor.add_apply]
    ring

end Tensor

namespace HasPullbackAt

open Tensor Op
open scoped RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] {s : Shape} {x : E}
  {f g : E → Tensor ℝ s} {Df Dg : E →L[ℝ] Tensor ℝ s} {bf bg : Tensor ℝ s → E}

/-- Where no coordinate has `f x c = g x c`, the entrywise `fmax` of two maps has at each
coordinate the derivative of the larger map, and the stored backward of `fmax` at `(f x, g x)`
followed by the pullbacks of the maps is its pullback. -/
@[fun_prop]
theorem fmax (h : ∀ c, f x c ≠ g x c) (hf : HasPullbackAt f Df bf x)
    (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt (fun y => Tensor.fmax (f y) (g y))
      ((slopeCLM fun c => if g x c ≤ f x c then 1 else 0).comp Df +
        (slopeCLM fun c => if g x c ≤ f x c then 0 else 1).comp Dg)
      (fun δ => bf (fmaxBackward (f x) (g x) δ).1 + bg (fmaxBackward (f x) (g x) δ).2) x := by
  refine (pointwise₂ Tensor.fmax (fun _ q => Max.max q.1 q.2)
    (fun y z c => (fmax_apply y z c).trans (Context.fmax_real _ _))
    (fun c => hasFDerivAt_max_of_ne (h c)) hf hg).congr_deriv rfl fun δ => ?_
  have e : fmaxBackward (f x) (g x) δ =
      (slopeCLM (fun c => if g x c ≤ f x c then 1 else 0) δ,
        slopeCLM (fun c => if g x c ≤ f x c then 0 else 1) δ) := by
    refine Prod.ext (Rep.ext fun c => ?_) (Rep.ext fun c => ?_) <;>
      simp only [fmaxBackward_real, zipWith3_apply, slopeCLM_apply, ite_mul, one_mul, zero_mul]
  simp only [e]

/-- Where no coordinate has `f x c = g x c`, the entrywise `fmin` of two maps has at each
coordinate the derivative of the smaller map, and the stored backward of `fmin` at `(f x, g x)`
followed by the pullbacks of the maps is its pullback. -/
@[fun_prop]
theorem fmin (h : ∀ c, f x c ≠ g x c) (hf : HasPullbackAt f Df bf x)
    (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt (fun y => Tensor.fmin (f y) (g y))
      ((slopeCLM fun c => if f x c ≤ g x c then 1 else 0).comp Df +
        (slopeCLM fun c => if f x c ≤ g x c then 0 else 1).comp Dg)
      (fun δ => bf (fminBackward (f x) (g x) δ).1 + bg (fminBackward (f x) (g x) δ).2) x := by
  refine (pointwise₂ Tensor.fmin (fun _ q => Min.min q.1 q.2)
    (fun y z c => (fmin_apply y z c).trans (Context.fmin_real _ _))
    (fun c => hasFDerivAt_min_of_ne (h c)) hf hg).congr_deriv rfl fun δ => ?_
  have e : fminBackward (f x) (g x) δ =
      (slopeCLM (fun c => if f x c ≤ g x c then 1 else 0) δ,
        slopeCLM (fun c => if f x c ≤ g x c then 0 else 1) δ) := by
    refine Prod.ext (Rep.ext fun c => ?_) (Rep.ext fun c => ?_) <;>
      simp only [fminBackward_real, zipWith3_apply, slopeCLM_apply, ite_mul, one_mul, zero_mul]
  simp only [e]

/-- Where every entry of `f x` and of `g x` is nonzero, the entrywise `copysign` of two maps has
slope `sign (f x c)` times the sign factor of `g x c` in its first argument and zero in its
second, and the stored backward of `copysign` at `(f x, g x)` followed by the pullbacks of the
maps is its pullback. -/
@[fun_prop]
theorem copysign (h : ∀ c, f x c ≠ 0 ∧ g x c ≠ 0) (hf : HasPullbackAt f Df bf x)
    (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt (fun y => Tensor.copysign (f y) (g y))
      ((slopeCLM fun c => Real.sign (f x c) * (if g x c < 0 then -1 else 1)).comp Df +
        (slopeCLM fun _ => 0).comp Dg)
      (fun δ => bf (copysignBackward (f x) (g x) δ).1 +
        bg (copysignBackward (f x) (g x) δ).2) x := by
  refine (pointwise₂ Tensor.copysign (fun _ q => Context.copysign q.1 q.2)
    (fun y z c => copysign_apply y z c) (fun c => hasFDerivAt_copysign (h c).1 (h c).2) hf
    hg).congr_deriv rfl fun δ => ?_
  have e : copysignBackward (f x) (g x) δ =
      (slopeCLM (fun c => Real.sign (f x c) * (if g x c < 0 then -1 else 1)) δ,
        slopeCLM (fun _ => 0) δ) := by
    refine Prod.ext (Rep.ext fun c => ?_) (Rep.ext fun c => ?_)
    · simp only [copysignBackward_real, zipWith3_apply, slopeCLM_apply]
    · simp only [copysignBackward_real, Rep.zero_apply, slopeCLM_apply, zero_mul]
  simp only [e]

/-- Where each coordinate has `f x c ≠ 0` or `0 < g x c`, the entrywise `atan2` of two maps has
slopes `g x c / (g x c² + f x c²)` and `-(f x c / (g x c² + f x c²))`, and pulls back through
them. -/
@[fun_prop]
theorem atan2 (h : ∀ c, f x c ≠ 0 ∨ 0 < g x c) (hf : HasPullbackAt f Df bf x)
    (hg : HasPullbackAt g Dg bg x) :
    HasPullbackAt (fun y => Tensor.atan2 (f y) (g y))
      ((slopeCLM fun c => g x c / (g x c * g x c + f x c * f x c)).comp Df +
        (slopeCLM fun c => -(f x c / (g x c * g x c + f x c * f x c))).comp Dg)
      (fun δ => bf (slopeCLM (fun c => g x c / (g x c * g x c + f x c * f x c)) δ) +
        bg (slopeCLM (fun c => -(f x c / (g x c * g x c + f x c * f x c))) δ)) x :=
  pointwise₂ Tensor.atan2 (fun _ q => InverseTrig.atan2 q.1 q.2)
    (fun y z c => atan2_apply y z c) (fun c => hasFDerivAt_atan2 (h c)) hf hg

end HasPullbackAt

namespace Op

variable {s : Shape}

/-- Where no entry of `a` equals the entry of `b`, the stored backward of `fmax` is the adjoint of
the derivative of its forward, jointly in both inputs. The set is exact, also for each input
alone, since at `a c = b c` the entry `max (a c + t) (b c) = a c + max t 0` and its analogue in
`b` have no derivative at `t = 0`. -/
@[op_cert TorchLean.Op.fmax derivative 0 1]
theorem fmax_backward_eq_adjoint (a b δ : Tensor ℝ s) (h : ∀ c, a c ≠ b c) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
        (((Op.fmax (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
      (WithLp.toLp 2 (a, b)) ∧
    WithLp.toLp 2
        ((((Op.fmax (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).1,
        (((Op.fmax (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).2.1) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
        (((Op.fmax (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
        (WithLp.toLp 2 (a, b))).adjoint δ := by
  op_deriv

/-- Where no entry of `a` equals the entry of `b`, the stored backward of `fmin` is the adjoint of
the derivative of its forward, jointly in both inputs. The set is exact, also for each input
alone, since at `a c = b c` the entry `min (a c + t) (b c) = a c + min t 0` and its analogue in
`b` have no derivative at `t = 0`. -/
@[op_cert TorchLean.Op.fmin derivative 0 1]
theorem fmin_backward_eq_adjoint (a b δ : Tensor ℝ s) (h : ∀ c, a c ≠ b c) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
        (((Op.fmin (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
      (WithLp.toLp 2 (a, b)) ∧
    WithLp.toLp 2
        ((((Op.fmin (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).1,
        (((Op.fmin (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).2.1) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
        (((Op.fmin (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
        (WithLp.toLp 2 (a, b))).adjoint δ := by
  op_deriv

/-- At every real input, the stored backward of `fmax` routes `δ c` to `a` where `b c ≤ a c` and to
`b` elsewhere, at a tie `δ c` times the right partial in `a` and the left one in `b`. The reals
have no `-0`, and a float `fmax` routes `δ c` to the operand it returns, which at `(-0, +0)` is
`b`. -/
@[op_cert TorchLean.Op.fmax convention]
theorem fmax_backward_eq (a b δ : Tensor ℝ s) (c : s.Coord) :
    (((Op.fmax (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).1 c =
        (if b c ≤ a c then δ c else 0) ∧
      (((Op.fmax (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).2.1 c =
        (if b c ≤ a c then 0 else δ c) := by
  constructor <;> simp only [Op.rule?_fmax, Option.get_some, Tensor.fmaxBackward_real,
    Tensor.zipWith3_apply]

/-- At every real input, the stored backward of `fmin` routes `δ c` to `a` where `a c ≤ b c` and to
`b` elsewhere, at a tie `δ c` times the left partial in `a` and the right one in `b`. The reals
have no `-0`, and a float `fmin` routes `δ c` to the operand it returns, which at `(+0, -0)` is
`b`. -/
@[op_cert TorchLean.Op.fmin convention]
theorem fmin_backward_eq (a b δ : Tensor ℝ s) (c : s.Coord) :
    (((Op.fmin (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).1 c =
        (if a c ≤ b c then δ c else 0) ∧
      (((Op.fmin (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).2.1 c =
        (if a c ≤ b c then 0 else δ c) := by
  constructor <;> simp only [Op.rule?_fmin, Option.get_some, Tensor.fminBackward_real,
    Tensor.zipWith3_apply]

/-- Where every entry of `a` and of `b` is nonzero, the stored backward of `copysign` is the
adjoint of the derivative of its forward, jointly in both inputs. The set is exact since the entry
`copysign t (b c)`, which is `|t|` or `-|t|`, has no derivative at `t = 0`, and for `a c ≠ 0` the
entry `copysign (a c) t` jumps from `-|a c|` to `|a c|` at `t = 0`. -/
@[op_cert TorchLean.Op.copysign derivative 0 1]
theorem copysign_backward_eq_adjoint (a b δ : Tensor ℝ s) (h : ∀ c, a c ≠ 0 ∧ b c ≠ 0) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
        (((Op.copysign (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
      (WithLp.toLp 2 (a, b)) ∧
    WithLp.toLp 2
        ((((Op.copysign (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).1,
        (((Op.copysign (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).2.1) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
        (((Op.copysign (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
        (WithLp.toLp 2 (a, b))).adjoint δ := by
  op_deriv

/-- Where every entry of `a` is nonzero, the stored backward of `copysign` is the adjoint of the
derivative of its forward in `a`. The set is exact since at `a c = 0` the entry
`copysign t (b c)` is `|t|` or `-|t|`, which has no derivative at `t = 0`. -/
@[op_cert TorchLean.Op.copysign derivative 0]
theorem copysign_backward_eq_adjoint_left (a b δ : Tensor ℝ s) (h : ∀ c, a c ≠ 0) :
    DifferentiableAt ℝ (fun y => (((Op.copysign (s := s)).rule? (α := ℝ) |>.get rfl).forward
        (y, b, ())).1) a ∧
    (((Op.copysign (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).1 =
      (fderiv ℝ (fun y => (((Op.copysign (s := s)).rule? (α := ℝ) |>.get rfl).forward
        (y, b, ())).1) a).adjoint δ := by
  have hd : ∀ c, HasDerivAt (fun t => Context.copysign t (b c))
      (Real.sign (a c) * (if b c < 0 then -1 else 1)) (a c) :=
    fun c => hasDerivAt_copysign_left (b c) (h c)
  op_deriv [Tensor.copysign]

/-- Where every coordinate has `a c = 0` or `b c ≠ 0`, the stored backward of `copysign` is the
adjoint of the derivative of its forward in `b`. The set is exact since `copysign 0 t` is zero for
every `t`, and for `a c ≠ 0` the entry `copysign (a c) t` jumps from `-|a c|` to `|a c|` at
`t = 0`. -/
@[op_cert TorchLean.Op.copysign derivative 1]
theorem copysign_backward_eq_adjoint_right (a b δ : Tensor ℝ s) (h : ∀ c, a c = 0 ∨ b c ≠ 0) :
    DifferentiableAt ℝ (fun y => (((Op.copysign (s := s)).rule? (α := ℝ) |>.get rfl).forward
        (a, y, ())).1) b ∧
    (((Op.copysign (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).2.1 =
      (fderiv ℝ (fun y => (((Op.copysign (s := s)).rule? (α := ℝ) |>.get rfl).forward
        (a, y, ())).1) b).adjoint δ := by
  have hd : ∀ c, HasDerivAt (fun t => Context.copysign (a c) t) 0 (b c) :=
    fun c => hasDerivAt_copysign_right (h c)
  op_deriv [Tensor.copysign]

/-- At every input, the stored backward of `copysign` is `sign (a c)` times the sign factor of
`b c` times `δ c` for `a`, and zero for `b`: at `a c = 0` these are `δ c` times the mean of the
two one-sided partials in `a` and `δ c` times the partial in `b`, and at `b c = 0 ≠ a c` they are
`δ c` times the partial in `a` and the right partial in `b`. The reals have no `-0`, and at a float
`b c = -0` the operator returns `-|a c|` and the pullback `-sign (a c) * δ c` to `a`, `δ c` times
the partial in `a` on the side `b c < 0`. -/
@[op_cert TorchLean.Op.copysign convention]
theorem copysign_backward_eq (a b δ : Tensor ℝ s) (c : s.Coord) :
    (((Op.copysign (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).1 c =
        Real.sign (a c) * (if b c < 0 then -1 else 1) * δ c ∧
      (((Op.copysign (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).2.1 c =
        0 := by
  constructor <;> simp only [Op.rule?_copysign, Option.get_some, Tensor.copysignBackward_real,
    Tensor.zipWith3_apply, Tensor.Rep.zero_apply]

/-- Where every point `x c + y c i` lies off the closed negative real axis, that is `y c ≠ 0` or
`0 < x c`, the stored backward of `atan2 y x` is the adjoint of the derivative of its forward,
jointly in both inputs. The set is exact, also for `y` alone, since for `x c < 0` the entry
`atan2 t (x c)` jumps from near `-π` to `π` at `t = 0`, and `atan2 t 0` is `-π / 2` for `t < 0`
and `π / 2` for `t > 0`. -/
@[op_cert TorchLean.Op.atan2 derivative 0 1]
theorem atan2_backward_eq_adjoint (y x δ : Tensor ℝ s) (h : ∀ c, y c ≠ 0 ∨ 0 < x c) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
        (((Op.atan2 (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
      (WithLp.toLp 2 (y, x)) ∧
    WithLp.toLp 2
        ((((Op.atan2 (s := s)).rule? (α := ℝ) |>.get rfl).backward (y, x, ()) (δ, ())).1,
        (((Op.atan2 (s := s)).rule? (α := ℝ) |>.get rfl).backward (y, x, ()) (δ, ())).2.1) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
        (((Op.atan2 (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
        (WithLp.toLp 2 (y, x))).adjoint δ := by
  have e : (Op.atan2 (s := s)).rule? (α := ℝ) = _ := Op.rule?_atan2 (α := ℝ) (s := s)
  op_deriv [e]

/-- Where every coordinate has `y c ≠ 0` or `x c ≠ 0`, the stored backward of `atan2 y x` is the
adjoint of the derivative of its forward in `x`. The set is exact since `atan2 0 t` is `π` for
`t < 0` and `0` for `t ≥ 0`, so only the origin is excluded. -/
@[op_cert TorchLean.Op.atan2 derivative 1]
theorem atan2_backward_eq_adjoint_right (y x δ : Tensor ℝ s) (h : ∀ c, y c ≠ 0 ∨ x c ≠ 0) :
    DifferentiableAt ℝ (fun x' => (((Op.atan2 (s := s)).rule? (α := ℝ) |>.get rfl).forward
        (y, x', ())).1) x ∧
    (((Op.atan2 (s := s)).rule? (α := ℝ) |>.get rfl).backward (y, x, ()) (δ, ())).2.1 =
      (fderiv ℝ (fun x' => (((Op.atan2 (s := s)).rule? (α := ℝ) |>.get rfl).forward
        (y, x', ())).1) x).adjoint δ := by
  have e : (Op.atan2 (s := s)).rule? (α := ℝ) = _ := Op.rule?_atan2 (α := ℝ) (s := s)
  have hd : ∀ c, HasDerivAt (fun t => InverseTrig.atan2 (y c) t)
      (-(y c / (x c * x c + y c * y c))) (x c) :=
    fun c => hasDerivAt_atan2_right (h c)
  op_deriv [e, Tensor.atan2]

/-- At every input, the stored backward of `atan2 y x` is `δ c * x c / (x c² + y c²)` for `y` and
`-(δ c * y c / (x c² + y c²))` for `x`, zero at the origin. On the cut `y c = 0 > x c` these are
`δ c` times the right partial in `y` and the partial in `x`, and at the origin the value for `x`
is `δ c` times its right partial while `y` has no one-sided partial. -/
@[op_cert TorchLean.Op.atan2 convention]
theorem atan2_backward_eq (y x δ : Tensor ℝ s) (c : s.Coord) :
    (((Op.atan2 (s := s)).rule? (α := ℝ) |>.get rfl).backward (y, x, ()) (δ, ())).1 c =
        δ c * x c / (x c * x c + y c * y c) ∧
      (((Op.atan2 (s := s)).rule? (α := ℝ) |>.get rfl).backward (y, x, ()) (δ, ())).2.1 c =
        -(δ c * y c / (x c * x c + y c * y c)) := by
  have e : (Op.atan2 (s := s)).rule? (α := ℝ) = _ := Op.rule?_atan2 (α := ℝ) (s := s)
  constructor <;> simp only [e, Option.get_some, Tensor.atan2Backward_real, Tensor.div_apply,
    Tensor.mul_apply, Tensor.add_apply, Tensor.Rep.neg_apply]

/-! ## `max` and `min` in both inputs, and their ties -/

/-- Where no entry of `x` equals the entry of `y`, the stored backward of `max` is the adjoint of
the derivative of its forward, jointly in both inputs. The set is exact, since at `x c = y c` the
entry `max (x c + t) (y c) = x c + max t 0` has no derivative at `t = 0`. -/
@[op_cert TorchLean.Op.max derivative 0 1]
theorem max_backward_eq_adjoint_prod (x y δ : Tensor ℝ s) (hxy : ∀ c, x c ≠ y c) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
        (((Op.max (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
      (WithLp.toLp 2 (x, y)) ∧
    WithLp.toLp 2
        ((((Op.max (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).1,
        (((Op.max (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).2.1) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
        (((Op.max (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
        (WithLp.toLp 2 (x, y))).adjoint δ := by
  have hp := pointwise₂_backward_eq_adjoint
    (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
      (((Op.max (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
    (fun _ q => Max.max q.1 q.2) (fun c => if y c ≤ x c then 1 else 0)
    (fun c => if y c ≤ x c then 0 else 1) x y δ
    (((Op.max (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).1
    (((Op.max (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).2.1
    (fun p c => by
      show (Tensor.maximum p.fst p.snd) c = _
      simp only [Tensor.maximum_apply, Context.maximum_real])
    (fun c => hasFDerivAt_max_of_ne (hxy c))
    (fun c => by
      rw [max_backward_apply]
      rcases (hxy c).lt_or_gt with h | h
      · rw [ite_eq_right (not_lt.mpr h.le), ite_eq_left h, ite_eq_right (not_le.mpr h)]
      · rw [ite_eq_left h, ite_eq_left h.le])
    (fun c => by
      show (((Op.max (s := s)).rule? (α := ℝ) |>.get rfl).backward (y, x, ()) (δ, ())).1 c = _
      rw [max_backward_apply]
      rcases (hxy c).lt_or_gt with h | h
      · rw [ite_eq_left h, ite_eq_right (not_le.mpr h)]
      · rw [ite_eq_right (not_lt.mpr h.le), ite_eq_left h, ite_eq_left h.le])
  exact ⟨hp.1.differentiableAt, hp.2⟩

/-- Where no entry of `x` equals the entry of `y`, the stored backward of `min` is the adjoint of
the derivative of its forward, jointly in both inputs. The set is exact, since at `x c = y c` the
entry `min (x c + t) (y c) = x c + min t 0` has no derivative at `t = 0`. -/
@[op_cert TorchLean.Op.min derivative 0 1]
theorem min_backward_eq_adjoint_prod (x y δ : Tensor ℝ s) (hxy : ∀ c, x c ≠ y c) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
        (((Op.min (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
      (WithLp.toLp 2 (x, y)) ∧
    WithLp.toLp 2
        ((((Op.min (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).1,
        (((Op.min (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).2.1) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
        (((Op.min (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
        (WithLp.toLp 2 (x, y))).adjoint δ := by
  have hp := pointwise₂_backward_eq_adjoint
    (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
      (((Op.min (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
    (fun _ q => Min.min q.1 q.2) (fun c => if x c ≤ y c then 1 else 0)
    (fun c => if x c ≤ y c then 0 else 1) x y δ
    (((Op.min (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).1
    (((Op.min (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).2.1
    (fun p c => by
      show (Tensor.minimum p.fst p.snd) c = _
      simp only [Tensor.minimum_apply, Context.minimum_real])
    (fun c => hasFDerivAt_min_of_ne (hxy c))
    (fun c => by
      rw [min_backward_apply]
      rcases (hxy c).lt_or_gt with h | h
      · rw [ite_eq_left h, ite_eq_left h.le]
      · rw [ite_eq_right (not_lt.mpr h.le), ite_eq_left h, ite_eq_right (not_le.mpr h)])
    (fun c => by
      show (((Op.min (s := s)).rule? (α := ℝ) |>.get rfl).backward (y, x, ()) (δ, ())).1 c = _
      rw [min_backward_apply]
      rcases (hxy c).lt_or_gt with h | h
      · rw [ite_eq_right (not_lt.mpr h.le), ite_eq_left h, ite_eq_left h.le]
      · rw [ite_eq_left h, ite_eq_right (not_le.mpr h)])
  exact ⟨hp.1.differentiableAt, hp.2⟩

/-- At a tie `x c = y c`, the stored backward of `max` gives each input half of `δ c`. This is
`δ c` times the mean of the two one-sided partials, `1` and `0`, of either input. -/
@[op_cert TorchLean.Op.max convention]
theorem max_backward_eq_of_eq (x y δ : Tensor ℝ s) (c : s.Coord) (h : x c = y c) :
    (((Op.max (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).1 c =
        1 / 2 * δ c ∧
      (((Op.max (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).2.1 c =
        1 / 2 * δ c :=
  ⟨max_backward_apply_of_eq x y δ c h, max_backward_apply_of_eq y x δ c h.symm⟩

/-- At a tie `x c = y c`, the stored backward of `min` gives each input half of `δ c`. This is
`δ c` times the mean of the two one-sided partials, `1` and `0`, of either input. -/
@[op_cert TorchLean.Op.min convention]
theorem min_backward_eq_of_eq (x y δ : Tensor ℝ s) (c : s.Coord) (h : x c = y c) :
    (((Op.min (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).1 c =
        1 / 2 * δ c ∧
      (((Op.min (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).2.1 c =
        1 / 2 * δ c :=
  ⟨min_backward_apply_of_eq x y δ c h, min_backward_apply_of_eq y x δ c h.symm⟩

/-! ## Conventions at kinks -/

/-- At every input, the stored backward of `relu` is `δ c` where `0 < x c` and zero elsewhere. At
the kink `x c = 0` this is `δ c` times the left derivative `0`. -/
@[op_cert TorchLean.Op.relu convention]
theorem relu_backward_eq (x δ : Tensor ℝ s) (c : s.Coord) :
    (((Op.relu (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 c =
      if 0 < x c then δ c else 0 := by
  show (Tensor.map Activation.reluDeriv x * δ) c = _
  simp only [Tensor.mul_def, Tensor.zipWith_apply, Tensor.map_apply,
    Activation.reluDeriv, gt_iff_lt, ite_mul, one_mul, zero_mul]

/-- At every input, the stored backward of `abs` is `Real.sign (x c) * δ c`. At the kink
`x c = 0` this is zero, `δ c` times the mean of the one-sided derivatives `-1` and `1`. -/
@[op_cert TorchLean.Op.abs convention]
theorem abs_backward_eq (x δ : Tensor ℝ s) (c : s.Coord) :
    (((Op.abs (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 c =
      Real.sign (x c) * δ c := by
  show (Tensor.sign x * δ) c = _
  simp only [Tensor.mul_def, Tensor.zipWith_apply, Tensor.sign_apply, Context.sign_real]

/-- At every input, the stored backward of `leakyReLU` is `δ c` where `0 < x c` and
`slope.item * δ c` elsewhere. At the kink `x c = 0`, for a slope other than `1`, this is `δ c`
times the left derivative `slope.item`. -/
@[op_cert TorchLean.Op.leakyReLU convention]
theorem leakyReLU_backward_eq (x δ : Tensor ℝ s) (slope : Tensor ℝ []) (c : s.Coord) :
    (((Op.leakyReLU (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, slope, ()) (δ, ())).1 c =
      if 0 < x c then δ c else slope.item * δ c := by
  show (Tensor.map (fun z => Activation.leakyReLUDeriv z slope.item) x * δ) c = _
  simp only [Tensor.mul_def, Tensor.zipWith_apply, Tensor.map_apply,
    Activation.leakyReLUDeriv, gt_iff_lt, ite_mul, one_mul]

/-- At every input, the stored backward of `elu` is `δ c` where `0 < x c` and
`alpha.item * Real.exp (x c) * δ c` elsewhere. At the kink `x c = 0`, for a coefficient other than
`1`, this is `alpha.item * δ c`, which is `δ c` times the left derivative `alpha.item`. -/
@[op_cert TorchLean.Op.elu convention]
theorem elu_backward_eq (x δ : Tensor ℝ s) (alpha : Tensor ℝ []) (c : s.Coord) :
    (((Op.elu (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, alpha, ()) (δ, ())).1 c =
      if 0 < x c then δ c else alpha.item * Real.exp (x c) * δ c := by
  show (Tensor.map (fun z => Activation.eluDeriv z alpha.item) x * δ) c = _
  simp only [Tensor.mul_def, Tensor.zipWith_apply, Tensor.map_apply,
    Activation.eluDeriv, MathFunctions.exp_real, gt_iff_lt, ite_mul, one_mul]

/-- At every input, the stored backward of `clamp` is `δ c` strictly between the bounds and zero
elsewhere. For `lo.item < hi.item` this is `δ c` times the left derivative `0` at the kink
`x c = lo.item` and `δ c` times the right derivative `0` at the kink `x c = hi.item`. -/
@[op_cert TorchLean.Op.clamp convention]
theorem clamp_backward_eq (x δ : Tensor ℝ s) (lo hi : Tensor ℝ []) (c : s.Coord) :
    (((Op.clamp (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, lo, hi, ()) (δ, ())).1 c =
      if lo.item < x c ∧ x c < hi.item then δ c else 0 := by
  show (Tensor.clampDerivative x (Tensor.full s lo.item) (Tensor.full s hi.item) * δ) c = _
  simp only [Tensor.mul_def, Tensor.zipWith_apply, Tensor.clampDerivative_apply,
    Tensor.full_apply, gt_iff_lt, ite_mul, one_mul, zero_mul]

/-- At a zero entry, the stored backward of `smoothAbs` over the reals is zero for every epsilon:
at epsilon zero, where the real `smoothAbs` is `|x|`, `δ c` times the mean of the one-sided
derivatives `-1` and `1`, and at every other epsilon `δ c` times the derivative `0`. In a binary
format it is a NaN at a nonpositive epsilon. -/
@[op_cert TorchLean.Op.smoothAbs convention]
theorem smoothAbs_backward_eq_of_eq_zero (x δ : Tensor ℝ s) (epsilon : Tensor ℝ [])
    (c : s.Coord) (hx : x c = 0) :
    (((Op.smoothAbs (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, epsilon, ())
      (δ, ())).1 c = 0 := by
  show (Tensor.map (fun z => Activation.smoothAbsDeriv z epsilon.item) x * δ) c = _
  simp only [Tensor.mul_def, Tensor.zipWith_apply, Tensor.map_apply,
    Activation.smoothAbsDeriv, hx, zero_div, zero_mul]

end Op

end TorchLean
