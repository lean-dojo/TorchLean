/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tactic.OpDeriv
public import TorchLean.Tensor.Op.Table.Elementwise
public import TorchLean.Tensor.Elementwise.Real
public import TorchLean.Tensor.Op.Pointwise.Real

/-!
# Derivatives of `sign`, the integral roundings, `fmod`, `remainder` and the rounded divisions

We prove that the stored backward of `Op.sign`, `Op.floor`, `Op.ceil`, `Op.round`, `Op.trunc`,
`Op.frac`, `Op.fmod`, `Op.remainder` and `Op.div` at each rounding is the adjoint of the Fréchet
derivative of the real forward, on the exact set where that forward is differentiable. At every
input the stored backward is zero for `sign`, the four roundings and the rounded divisions, `δ` for
`frac`, `(δ, -(δ * trunc (a / b)))` for `fmod`
and `(δ, -(δ * ⌊a / b⌋))` for `remainder`, whose divisor pullback is `δ` at a zero divisor and a
negative dividend. A `convention` certificate per operator records this value, and its docstring
names the one-sided derivative the value equals at a jump where there is one.

Each of `Real.sign`, `⌊·⌋`, `⌈·⌉`, rounding with ties to even and truncation is constant near
every point off its jumps, so it has derivative zero there, and `op_deriv` builds the certificates
from these derivatives. Over the reals `fmod a b` is `a - b * trunc (a / b)`. Near a point with a
nonzero divisor and a quotient that is zero or not an integer it agrees with `a - b * n` for the
integer `n = trunc (a / b)` at that point, and `TorchLean.Op.pointwise₂_backward_eq_adjoint`, the
certificate of a binary function applied at every coordinate, gives the joint certificate. Python's
`remainder a b` is `a - b * ⌊a / b⌋` over the reals (`TorchLean.Context.remainder_real`), and the
same argument with the floor in place of the truncation, off integer quotients, gives its
certificates. The rule `TorchLean.HasPullbackAt.zipWith` lets `op_deriv` compose a binary
function of two tensors that vary with one input. The rounded quotients `⌊a / b⌋`
(`TorchLean.Tensor.floorDivide_real`) and `trunc (a / b)`
(`TorchLean.Tensor.copysign_floorDivide_abs_real`) are constant near a point with a nonzero divisor
off their jumps, so one certificate of `Op.div` holds at every rounding, by
`TorchLean.Op.div_prod_backward_eq_adjoint` for true division. We restate the table arms of the
roundings, of `fmod` and of `remainder` over `ℝ`, where `simp` can use them.
-/

@[expose] public section

namespace TorchLean

open FloatLib.Floats.Formats.Flocq (truncRound nearestEven)

section Scalar

open Filter Set Topology

/-- `Real.sign` is constant near every nonzero real. -/
theorem eventually_sign_eq {x : ℝ} (hx : x ≠ 0) : ∀ᶠ y in 𝓝 x, Real.sign y = Real.sign x := by
  rcases hx.lt_or_gt with h | h
  · filter_upwards [eventually_lt_nhds h] with y hy
    rw [Real.sign_of_neg hy, Real.sign_of_neg h]
  · filter_upwards [eventually_gt_nhds h] with y hy
    rw [Real.sign_of_pos hy, Real.sign_of_pos h]

/-- The floor is constant near every real that is not an integer. -/
theorem eventually_floor_eq {x : ℝ} (hx : Int.fract x ≠ 0) : ∀ᶠ y in 𝓝 x, ⌊y⌋ = ⌊x⌋ := by
  filter_upwards [Ioo_mem_nhds (Int.floor_lt_self_iff.mpr (Int.fract_ne_zero_iff.mp hx))
    (Int.lt_floor_add_one x)] with y hy
  exact Int.floor_eq_on_Ico ⌊x⌋ y (Ioo_subset_Ico_self hy)

/-- The ceiling is constant near every real that is not an integer. -/
theorem eventually_ceil_eq {x : ℝ} (hx : Int.fract x ≠ 0) : ∀ᶠ y in 𝓝 x, ⌈y⌉ = ⌈x⌉ := by
  have key : ∀ y ∈ Ioo (⌊x⌋ : ℝ) (⌊x⌋ + 1), ⌈y⌉ = ⌊x⌋ + 1 := fun y hy =>
    Int.ceil_eq_iff.mpr ⟨by push_cast; linarith [hy.1], by push_cast; exact hy.2.le⟩
  have hx' : x ∈ Ioo (⌊x⌋ : ℝ) (⌊x⌋ + 1) :=
    ⟨Int.floor_lt_self_iff.mpr (Int.fract_ne_zero_iff.mp hx), Int.lt_floor_add_one x⟩
  filter_upwards [Ioo_mem_nhds hx'.1 hx'.2] with y hy
  rw [key y hy, key x hx']

/-- Rounding with ties to even sends every real strictly within `1 / 2` of an integer `n` to
`n`. -/
theorem nearestEven_eq_of_mem_Ioo {n : ℤ} {y : ℝ}
    (hy : y ∈ Ioo ((n : ℝ) - 1 / 2) (n + 1 / 2)) : nearestEven y = n := by
  rcases le_or_gt (n : ℝ) y with h | h
  · have hf : ⌊y⌋ = n := Int.floor_eq_iff.mpr ⟨h, by linarith [hy.2]⟩
    rw [FloatLib.Floats.Formats.Flocq.nearestEven_eq_floor_of_frac_lt_half y
      (by rw [hf]; linarith [hy.2]), hf]
  · have hf : ⌊y⌋ = n - 1 :=
      Int.floor_eq_iff.mpr ⟨by push_cast; linarith [hy.1], by push_cast; linarith⟩
    rw [FloatLib.Floats.Formats.Flocq.nearestEven_eq_ceil_of_frac_gt_half y
      (by rw [hf]; push_cast; linarith [hy.1]), hf, sub_add_cancel]

/-- Rounding with ties to even is constant near every real that is not a half-integer. -/
theorem eventually_nearestEven_eq {x : ℝ} (hx : Int.fract x ≠ 1 / 2) :
    ∀ᶠ y in 𝓝 x, nearestEven y = nearestEven x := by
  obtain ⟨n, hn⟩ : ∃ n : ℤ, x ∈ Ioo ((n : ℝ) - 1 / 2) (n + 1 / 2) := by
    have h0 := Int.fract_nonneg x
    have h1 := Int.fract_lt_one x
    rw [← Int.self_sub_floor] at h0 h1 hx
    rcases hx.lt_or_gt with h | h
    · exact ⟨⌊x⌋, by linarith, by linarith⟩
    · exact ⟨⌊x⌋ + 1, by push_cast; linarith, by push_cast; linarith⟩
  filter_upwards [Ioo_mem_nhds hn.1 hn.2] with y hy
  rw [nearestEven_eq_of_mem_Ioo hy, nearestEven_eq_of_mem_Ioo hn]

/-- Truncation toward zero is constant near zero and near every real that is not an integer. -/
theorem eventually_truncRound_eq {x : ℝ} (hx : x = 0 ∨ Int.fract x ≠ 0) :
    ∀ᶠ y in 𝓝 x, truncRound y = truncRound x := by
  rcases hx with rfl | hx
  · have h0 : ∀ y ∈ Ioo (-1 : ℝ) 1, truncRound y = 0 := fun y hy => by
      by_cases h : y < 0
      · simp only [truncRound, h, ↓reduceIte]
        exact Int.ceil_eq_iff.mpr ⟨by push_cast; linarith [hy.1], by push_cast; linarith⟩
      · simp only [truncRound, h, ↓reduceIte]
        exact Int.floor_eq_iff.mpr ⟨by push_cast; linarith, by push_cast; linarith [hy.2]⟩
    filter_upwards [Ioo_mem_nhds (by norm_num : (-1 : ℝ) < 0) one_pos] with y hy
    rw [h0 y hy, h0 0 ⟨by norm_num, one_pos⟩]
  · have hx0 : x ≠ 0 := fun h => hx (by rw [h, Int.fract_zero])
    rcases hx0.lt_or_gt with h | h
    · filter_upwards [eventually_lt_nhds h, eventually_ceil_eq hx] with y hy hc
      simp only [truncRound, hy, h, ↓reduceIte, hc]
    · filter_upwards [eventually_gt_nhds h, eventually_floor_eq hx] with y hy hc
      simp only [truncRound, not_lt.mpr hy.le, not_lt.mpr h.le, ↓reduceIte, hc]

/-- `Real.sign` has derivative zero at every nonzero real. -/
theorem hasDerivAt_sign {x : ℝ} (hx : x ≠ 0) : HasDerivAt Real.sign 0 x :=
  (hasDerivAt_const x (Real.sign x)).congr_of_eventuallyEq (eventually_sign_eq hx)

/-- An integer-valued function that is constant near `x` has, cast to the reals, derivative zero
at `x`. -/
theorem hasDerivAt_intCast_of_eventually_eq {r : ℝ → ℤ} {x : ℝ}
    (h : ∀ᶠ y in 𝓝 x, r y = r x) : HasDerivAt (fun y => (r y : ℝ)) 0 x :=
  (hasDerivAt_const x (r x : ℝ)).congr_of_eventuallyEq (h.mono fun y hy => by simp only [hy])

/-- An integer-valued function `r` of the quotient that is constant near `a / b`, for a nonzero
`b`, has, cast to the reals, derivative zero at `(a, b)`. -/
theorem hasFDerivAt_intCast_div_of_eventually_eq {r : ℝ → ℤ} {a b : ℝ} (hb : b ≠ 0)
    (h : ∀ᶠ t in 𝓝 (a / b), r t = r (a / b)) :
    HasFDerivAt (fun q : ℝ × ℝ => (r (q.1 / q.2) : ℝ))
      ((0 : ℝ) • ContinuousLinearMap.fst ℝ ℝ ℝ + (0 : ℝ) • ContinuousLinearMap.snd ℝ ℝ ℝ)
      (a, b) := by
  have ht : Tendsto (fun q : ℝ × ℝ => q.1 / q.2) (𝓝 (a, b)) (𝓝 (a / b)) :=
    (continuousAt_fst.div₀ continuousAt_snd hb).tendsto
  refine ((hasFDerivAt_const (r (a / b) : ℝ) (a, b)).congr_fderiv ?_).congr_of_eventuallyEq ?_
  · rw [zero_smul, zero_smul, add_zero]
  · filter_upwards [ht.eventually h] with q hq
    rw [hq]

/-- Where the divisor is nonzero and the quotient is zero or not an integer, `fmodReal` has
derivative `1` in the dividend and `-trunc (a / b)` in the divisor. -/
theorem hasFDerivAt_fmodReal {a b : ℝ} (hb : b ≠ 0) (h : a = 0 ∨ Int.fract (a / b) ≠ 0) :
    HasFDerivAt (fun q : ℝ × ℝ => Context.fmodReal q.1 q.2)
      ((1 : ℝ) • ContinuousLinearMap.fst ℝ ℝ ℝ +
        (-(truncRound (a / b) : ℝ)) • ContinuousLinearMap.snd ℝ ℝ ℝ) (a, b) := by
  have hq : a / b = 0 ∨ Int.fract (a / b) ≠ 0 := h.imp_left fun ha => by rw [ha, zero_div]
  have ht : Tendsto (fun q : ℝ × ℝ => q.1 / q.2) (𝓝 (a, b)) (𝓝 (a / b)) :=
    (continuousAt_fst.div₀ continuousAt_snd hb).tendsto
  have hev : (fun q : ℝ × ℝ => Context.fmodReal q.1 q.2) =ᶠ[𝓝 (a, b)]
      fun q => q.1 - q.2 * (truncRound (a / b) : ℝ) := by
    filter_upwards [ht.eventually (eventually_truncRound_eq hq)] with q hq'
    simp only [Context.fmodReal, hq']
  refine ((hasFDerivAt_fst.sub (hasFDerivAt_snd.mul_const _)).congr_fderiv
    (ContinuousLinearMap.ext fun q => ?_)).congr_of_eventuallyEq hev
  simp only [_root_.sub_apply, _root_.add_apply, _root_.smul_apply,
    ContinuousLinearMap.coe_fst', ContinuousLinearMap.coe_snd', smul_eq_mul]
  ring

/-- Where the divisor is zero, the dividend is zero or the quotient is not an integer,
`fmodReal` has derivative `1` in the dividend. -/
theorem hasDerivAt_fmodReal_left {a b : ℝ} (h : b = 0 ∨ a = 0 ∨ Int.fract (a / b) ≠ 0) :
    HasDerivAt (fun t => Context.fmodReal t b) 1 a := by
  have hq : a / b = 0 ∨ Int.fract (a / b) ≠ 0 := by
    rcases h with hb | ha | hf
    · exact Or.inl (by rw [hb, div_zero])
    · exact Or.inl (by rw [ha, zero_div])
    · exact Or.inr hf
  have ht : Tendsto (fun t : ℝ => t / b) (𝓝 a) (𝓝 (a / b)) :=
    (continuous_id'.div_const b).tendsto a
  have hev : (fun t => Context.fmodReal t b) =ᶠ[𝓝 a]
      fun t => t - b * (truncRound (a / b) : ℝ) := by
    filter_upwards [ht.eventually (eventually_truncRound_eq hq)] with t ht'
    simp only [Context.fmodReal, ht']
  exact ((hasDerivAt_id' a).sub_const _).congr_of_eventuallyEq hev

/-- Where the dividend is zero, or the divisor is nonzero and the quotient is not an integer,
`fmodReal` has derivative `-trunc (a / b)` in the divisor. -/
theorem hasDerivAt_fmodReal_right {a b : ℝ} (h : a = 0 ∨ (b ≠ 0 ∧ Int.fract (a / b) ≠ 0)) :
    HasDerivAt (fun t => Context.fmodReal a t) (-(truncRound (a / b) : ℝ)) b := by
  rcases h with rfl | ⟨hb, hf⟩
  · have h0 : ∀ t : ℝ, Context.fmodReal 0 t = 0 := fun t => by
      rw [Context.fmodReal, zero_div, FloatLib.Floats.Formats.Flocq.ValidRnd.map_zero truncRound,
        Int.cast_zero, mul_zero, sub_zero]
    rw [zero_div, FloatLib.Floats.Formats.Flocq.ValidRnd.map_zero truncRound, Int.cast_zero,
      neg_zero]
    exact (hasDerivAt_const b (0 : ℝ)).congr_of_eventuallyEq (Eventually.of_forall h0)
  · have ht : Tendsto (fun t : ℝ => a / t) (𝓝 b) (𝓝 (a / b)) :=
      (continuousAt_const.div₀ continuousAt_id hb).tendsto
    have hev : (fun t => Context.fmodReal a t) =ᶠ[𝓝 b]
        fun t => a - t * (truncRound (a / b) : ℝ) := by
      filter_upwards [ht.eventually (eventually_truncRound_eq (Or.inr hf))] with t ht'
      simp only [Context.fmodReal, ht']
    exact ((hasDerivAt_mul_const (truncRound (a / b) : ℝ)).const_sub a).congr_of_eventuallyEq hev

end Scalar

namespace Context

/-- Over the reals, `Context.fmodQuotient a b`, the factor in the divisor pullback of `fmod`, is
the truncated quotient `trunc (a / b)`, zero for a zero divisor. -/
theorem fmodQuotient_real (a b : ℝ) : Context.fmodQuotient a b = (truncRound (a / b) : ℝ) := by
  have hq : (a - Rounding.fmod a b) / b = (truncRound (a / b) : ℝ) := by
    rw [Context.fmod_real]
    rcases eq_or_ne b 0 with rfl | hb
    · simp [truncRound]
    · rw [sub_sub_cancel, mul_comm, mul_div_assoc, div_self hb, mul_one]
  have hneg : 0 > a / b → truncRound (a / b) ≤ 0 := by
    intro h
    have h' : a / b < 0 := h
    simp only [truncRound, h', ↓reduceIte]
    exact Int.ceil_le.mpr (by simpa using h'.le)
  have hnonneg : ¬ 0 > a / b → 0 ≤ truncRound (a / b) := by
    intro h
    have h' : ¬ a / b < 0 := h
    simp only [truncRound, h', ↓reduceIte]
    exact Int.floor_nonneg.mpr (not_lt.mp h')
  unfold Context.fmodQuotient
  rw [hq]
  generalize truncRound (a / b) = t at hneg hnonneg ⊢
  have hr : Rounding.round (t : ℝ) = (t : ℝ) := by
    rw [Context.round_real, FloatLib.Floats.Formats.Flocq.ValidRnd.id (rnd := nearestEven) t]
  rw [hr]
  show (if 0 > a / b then -|(t : ℝ)| else |(t : ℝ)|) = (t : ℝ)
  split_ifs with h
  · rw [abs_of_nonpos (by exact_mod_cast hneg h)]
    ring
  · exact abs_of_nonneg (by exact_mod_cast hnonneg h)

end Context

namespace Tensor

variable {s : Shape}

/-- Over the reals, `sign x` applies `Real.sign` at every entry. -/
@[op_deriv_simps] theorem sign_real (x : Tensor ℝ s) : sign x = map Real.sign x :=
  Rep.ext fun c => by simp only [sign_apply, Context.sign_real, Tensor.map_apply]

/-- Over the reals, `floor x` is `⌊x c⌋` at every entry. -/
@[op_deriv_simps] theorem floor_real (x : Tensor ℝ s) :
    floor x = map (fun y => ((⌊y⌋ : ℤ) : ℝ)) x :=
  Rep.ext fun c => by simp only [floor_apply, Context.floor_real, Tensor.map_apply]

/-- Over the reals, `ceil x` is `⌈x c⌉` at every entry. -/
@[op_deriv_simps] theorem ceil_real (x : Tensor ℝ s) :
    ceil x = map (fun y => ((⌈y⌉ : ℤ) : ℝ)) x :=
  Rep.ext fun c => by simp only [ceil_apply, Context.ceil_real, Tensor.map_apply]

/-- Over the reals, `round x` rounds every entry to the nearest integer, a tie to the even one. -/
@[op_deriv_simps] theorem round_real (x : Tensor ℝ s) :
    round x = map (fun y => ((nearestEven y : ℤ) : ℝ)) x :=
  Rep.ext fun c => by simp only [round_apply, Context.round_real, Tensor.map_apply]

/-- Over the reals, `trunc x` truncates every entry toward zero. -/
@[op_deriv_simps] theorem trunc_real (x : Tensor ℝ s) :
    trunc x = map (fun y => ((truncRound y : ℤ) : ℝ)) x :=
  Rep.ext fun c => by simp only [trunc_apply, Context.trunc_real, Tensor.map_apply]

/-- Over the reals, `frac x` is `x - trunc x`. -/
@[op_deriv_simps] theorem frac_real (x : Tensor ℝ s) : frac x = x - trunc x :=
  Rep.ext fun c => by
    simp only [frac_apply, Context.frac_real, Rep.hSub_apply, trunc_apply, Context.trunc_real]

/-- Over the reals, `fmod a b` applies `Context.fmodReal` at every coordinate. -/
@[op_deriv_simps] theorem fmod_real (a b : Tensor ℝ s) :
    fmod a b = zipWith Context.fmodReal a b :=
  rfl

/-- Over the reals, `floorDivide a b` is the floor of `a / b` at every entry, and zero where `b` is
zero, since Mathlib sets `a / 0 = 0`. -/
@[op_deriv_simps] theorem floorDivide_real (a b : Tensor ℝ s) :
    floorDivide a b = floor (a / b) := by
  have hZero : Context.copysign (0 : ℝ) = fun _ => 0 := by
    funext y
    show (if 0 > y then -|(0 : ℝ)| else |(0 : ℝ)|) = 0
    simp
  have hHalf : ((1 / 2 : Rat) : ℝ) = 1 / 2 := by norm_num
  refine Rep.ext fun c => ?_
  simp only [floorDivide, zipWith_apply, floor_apply, Tensor.div_apply, hZero, hHalf]
  exact divFloorSchedule_real (a c) (b c)

end Tensor

namespace Op

variable {s : Shape}

/-! ## The table arms over the reals

The arms `Op.rule?_floor`, ..., `Op.rule?_fmod` hold at the capabilities that synthesis builds
from `[Rounding α]` alone. Over `ℝ` synthesis also finds the error function, the inverse
trigonometric functions and decidable equality, so `simp` cannot rewrite `Op.rule? (α := ℝ)`
with them. We restate each arm at `ℝ`, proved by the table arm.
-/

/-- Over the reals, the rule of `Op.floor` is its table arm. -/
@[op_deriv_simps] theorem rule?_floor_real :
    (Op.floor (s := s)).rule? (α := ℝ) = some
      { forward := fun xs => (Tensor.floor xs.1, ())
        backward := fun _ g => (Tensor.map (fun _ => (0 : ℝ)) g.1, ())
        jvp? := some fun _ dxs => (Tensor.map (fun _ => (0 : ℝ)) dxs.1, ()) } :=
  Op.rule?_floor (α := ℝ) (s := s)

/-- Over the reals, the rule of `Op.ceil` is its table arm. -/
@[op_deriv_simps] theorem rule?_ceil_real :
    (Op.ceil (s := s)).rule? (α := ℝ) = some
      { forward := fun xs => (Tensor.ceil xs.1, ())
        backward := fun _ g => (Tensor.map (fun _ => (0 : ℝ)) g.1, ())
        jvp? := some fun _ dxs => (Tensor.map (fun _ => (0 : ℝ)) dxs.1, ()) } :=
  Op.rule?_ceil (α := ℝ) (s := s)

/-- Over the reals, the rule of `Op.round` is its table arm. -/
@[op_deriv_simps] theorem rule?_round_real :
    (Op.round (s := s)).rule? (α := ℝ) = some
      { forward := fun xs => (Tensor.round xs.1, ())
        backward := fun _ g => (Tensor.map (fun _ => (0 : ℝ)) g.1, ())
        jvp? := some fun _ dxs => (Tensor.map (fun _ => (0 : ℝ)) dxs.1, ()) } :=
  Op.rule?_round (α := ℝ) (s := s)

/-- Over the reals, the rule of `Op.trunc` is its table arm. -/
@[op_deriv_simps] theorem rule?_trunc_real :
    (Op.trunc (s := s)).rule? (α := ℝ) = some
      { forward := fun xs => (Tensor.trunc xs.1, ())
        backward := fun _ g => (Tensor.map (fun _ => (0 : ℝ)) g.1, ())
        jvp? := some fun _ dxs => (Tensor.map (fun _ => (0 : ℝ)) dxs.1, ()) } :=
  Op.rule?_trunc (α := ℝ) (s := s)

/-- Over the reals, the rule of `Op.frac` is its table arm. -/
@[op_deriv_simps] theorem rule?_frac_real :
    (Op.frac (s := s)).rule? (α := ℝ) = some
      { forward := fun xs => (Tensor.frac xs.1, ())
        backward := fun _ g => (g.1, ())
        jvp? := some fun _ dxs => (dxs.1, ()) } :=
  Op.rule?_frac (α := ℝ) (s := s)

/-- Over the reals, the rule of `Op.fmod` is its table arm. -/
@[op_deriv_simps] theorem rule?_fmod_real :
    (Op.fmod (s := s)).rule? (α := ℝ) = some
      { forward := fun xs => (Tensor.fmod xs.1 xs.2.1, ())
        backward := fun xs g =>
          let (da, db) := Tensor.fmodBackward xs.1 xs.2.1 g.1
          (da, db, ())
        jvp? := some fun xs dxs =>
          ((Tensor.fmodBackward xs.1 xs.2.1 dxs.1).1 +
            (Tensor.fmodBackward xs.1 xs.2.1 dxs.2.1).2, ()) } :=
  Op.rule?_fmod (α := ℝ) (s := s)

/-! ## `sign` -/

/-- Where no entry is zero, the real forward of `Op.sign` is differentiable and its stored
backward, zero, is the adjoint of the derivative. This is the exact set: `Real.sign` jumps from
`-1` to `1` at zero, so for `s = [1]` and `x = 0` the forward is not continuous. -/
@[op_cert TorchLean.Op.sign derivative]
theorem sign_backward_eq_adjoint (x δ : Tensor ℝ s) (hx : ∀ c, x c ≠ 0) :
    DifferentiableAt ℝ (fun y => (((Op.sign (s := s)).rule? (α := ℝ) |>.get rfl).forward
        (y, ())).1) x ∧
      (((Op.sign (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.sign (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ := by
  have hd : ∀ c, HasDerivAt Real.sign 0 (x c) := fun c => hasDerivAt_sign (hx c)
  op_deriv

/-- At every input, the stored backward of `Op.sign` is zero. Off zero this is the derivative,
and at a zero entry, where `sign` jumps on both sides, there is no one-sided derivative. -/
@[op_cert TorchLean.Op.sign convention]
theorem sign_backward_eq (x δ : Tensor ℝ s) :
    (((Op.sign (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 = 0 :=
  Tensor.Rep.ext fun c => by
    simp only [Op.rule?_sign, Option.get_some, Tensor.map_apply,
      Tensor.Rep.zero_apply]

/-! ## `floor` and `ceil` -/

/-- Where no entry is an integer, the real forward of `Op.floor` is differentiable and its stored
backward, zero, is the adjoint of the derivative. This is the exact set: `⌊t⌋` jumps from `n - 1`
to `n` at an integer `n`, so for `s = [1]` and `x = 0` the forward is not continuous. -/
@[op_cert TorchLean.Op.floor derivative]
theorem floor_backward_eq_adjoint (x δ : Tensor ℝ s) (hx : ∀ c, Int.fract (x c) ≠ 0) :
    DifferentiableAt ℝ (fun y => (((Op.floor (s := s)).rule? (α := ℝ) |>.get rfl).forward
        (y, ())).1) x ∧
      (((Op.floor (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.floor (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ := by
  have hd : ∀ c, HasDerivAt (fun y : ℝ => ((⌊y⌋ : ℤ) : ℝ)) 0 (x c) := fun c =>
    hasDerivAt_intCast_of_eventually_eq (eventually_floor_eq (hx c))
  op_deriv

/-- At every input, the stored backward of `Op.floor` is zero. At an integer `n` this is the
derivative from the right, where `⌊t⌋` is `n` on `[n, n + 1)`, and from the left the floor jumps.
-/
@[op_cert TorchLean.Op.floor convention]
theorem floor_backward_eq (x δ : Tensor ℝ s) :
    (((Op.floor (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 = 0 :=
  Tensor.Rep.ext fun c => by
    simp only [rule?_floor_real, Option.get_some, Tensor.map_apply,
      Tensor.Rep.zero_apply]

/-- Where no entry is an integer, the real forward of `Op.ceil` is differentiable and its stored
backward, zero, is the adjoint of the derivative. This is the exact set: `⌈t⌉` jumps from `n` to
`n + 1` at an integer `n`, so for `s = [1]` and `x = 0` the forward is not continuous. -/
@[op_cert TorchLean.Op.ceil derivative]
theorem ceil_backward_eq_adjoint (x δ : Tensor ℝ s) (hx : ∀ c, Int.fract (x c) ≠ 0) :
    DifferentiableAt ℝ (fun y => (((Op.ceil (s := s)).rule? (α := ℝ) |>.get rfl).forward
        (y, ())).1) x ∧
      (((Op.ceil (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.ceil (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ := by
  have hd : ∀ c, HasDerivAt (fun y : ℝ => ((⌈y⌉ : ℤ) : ℝ)) 0 (x c) := fun c =>
    hasDerivAt_intCast_of_eventually_eq (eventually_ceil_eq (hx c))
  op_deriv

/-- At every input, the stored backward of `Op.ceil` is zero. At an integer `n` this is the
derivative from the left, where `⌈t⌉` is `n` on `(n - 1, n]`, and from the right the ceiling
jumps. -/
@[op_cert TorchLean.Op.ceil convention]
theorem ceil_backward_eq (x δ : Tensor ℝ s) :
    (((Op.ceil (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 = 0 :=
  Tensor.Rep.ext fun c => by
    simp only [rule?_ceil_real, Option.get_some, Tensor.map_apply,
      Tensor.Rep.zero_apply]

/-! ## `round` -/

/-- Where no entry is a half-integer, the real forward of `Op.round` is differentiable and its
stored backward, zero, is the adjoint of the derivative. This is the exact set: for `s = [1]` the
forward is `0` at `x = 1 / 2` and `1` just above it, and near an integer it is constant. -/
@[op_cert TorchLean.Op.round derivative]
theorem round_backward_eq_adjoint (x δ : Tensor ℝ s) (hx : ∀ c, Int.fract (x c) ≠ 1 / 2) :
    DifferentiableAt ℝ (fun y => (((Op.round (s := s)).rule? (α := ℝ) |>.get rfl).forward
        (y, ())).1) x ∧
      (((Op.round (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.round (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ := by
  have hd : ∀ c, HasDerivAt (fun y : ℝ => ((nearestEven y : ℤ) : ℝ)) 0 (x c) := fun c =>
    hasDerivAt_intCast_of_eventually_eq (eventually_nearestEven_eq (hx c))
  op_deriv

/-- At every input, the stored backward of `Op.round` is zero. At a half-integer `n + 1 / 2`
this is the derivative from the left for an even `n` and from the right for an odd `n`, the side
on which rounding with ties to even keeps its value at the tie. -/
@[op_cert TorchLean.Op.round convention]
theorem round_backward_eq (x δ : Tensor ℝ s) :
    (((Op.round (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 = 0 :=
  Tensor.Rep.ext fun c => by
    simp only [rule?_round_real, Option.get_some, Tensor.map_apply,
      Tensor.Rep.zero_apply]

/-! ## `trunc` and `frac` -/

/-- Where every entry is zero or not an integer, the real forward of `Op.trunc` is differentiable
and its stored backward, zero, is the adjoint of the derivative. This is the exact set: for
`s = [1]` the truncation is `0` just below `x = 1` and `1` at it, and it is zero on `(-1, 1)`. -/
@[op_cert TorchLean.Op.trunc derivative]
theorem trunc_backward_eq_adjoint (x δ : Tensor ℝ s)
    (hx : ∀ c, x c = 0 ∨ Int.fract (x c) ≠ 0) :
    DifferentiableAt ℝ (fun y => (((Op.trunc (s := s)).rule? (α := ℝ) |>.get rfl).forward
        (y, ())).1) x ∧
      (((Op.trunc (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.trunc (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ := by
  have hd : ∀ c, HasDerivAt (fun y : ℝ => ((truncRound y : ℤ) : ℝ)) 0 (x c) := fun c =>
    hasDerivAt_intCast_of_eventually_eq (eventually_truncRound_eq (hx c))
  op_deriv

/-- At every input, the stored backward of `Op.trunc` is zero. At a nonzero integer `n` this is
the derivative on the side away from zero, from the right for `n > 0` and from the left for
`n < 0`, where the truncation stays `n`. -/
@[op_cert TorchLean.Op.trunc convention]
theorem trunc_backward_eq (x δ : Tensor ℝ s) :
    (((Op.trunc (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 = 0 :=
  Tensor.Rep.ext fun c => by
    simp only [rule?_trunc_real, Option.get_some, Tensor.map_apply,
      Tensor.Rep.zero_apply]

/-- Where every entry is zero or not an integer, the real forward of `Op.frac` is differentiable
and its stored backward, `δ`, is the adjoint of the derivative. This is the exact set: for
`s = [1]` the forward is `1 - ε` at `1 - ε` and `0` at `x = 1`, and it is the identity on
`(-1, 1)`. -/
@[op_cert TorchLean.Op.frac derivative]
theorem frac_backward_eq_adjoint (x δ : Tensor ℝ s)
    (hx : ∀ c, x c = 0 ∨ Int.fract (x c) ≠ 0) :
    DifferentiableAt ℝ (fun y => (((Op.frac (s := s)).rule? (α := ℝ) |>.get rfl).forward
        (y, ())).1) x ∧
      (((Op.frac (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.frac (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ := by
  have hd : ∀ c, HasDerivAt (fun y : ℝ => ((truncRound y : ℤ) : ℝ)) 0 (x c) := fun c =>
    hasDerivAt_intCast_of_eventually_eq (eventually_truncRound_eq (hx c))
  op_deriv

/-- At every input, the stored backward of `Op.frac` is `δ`. At a nonzero integer `n` this is
`δ` times the derivative `1` from the side away from zero, where `frac t` is `t - n`. -/
@[op_cert TorchLean.Op.frac convention]
theorem frac_backward_eq (x δ : Tensor ℝ s) :
    (((Op.frac (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 = δ := by
  simp only [rule?_frac_real, Option.get_some]

/-! ## `fmod` -/

/-- Where every divisor entry is nonzero and every quotient `a c / b c` is zero or not an
integer, the real forward of `Op.fmod` is differentiable in both inputs and its stored pair is the
adjoint of the derivative. This is the exact set of the real model, whose value `a` at a zero
divisor rests on Lean's `a / 0 = 0` where the float operator returns a NaN: for `s = [1]` and
small `ε > 0` the forward is `1 - ε` at `(1 - ε, 1)` and `0` at `(1, 1)`, it is `1` at `(1, 0)`
and below `ε` at `(1, ε)`, and at `(0, 0)` it is `t` along `(t, 0)` and `0` along `(0, t)` and
`(t, t)`, which no linear map fits. -/
@[op_cert TorchLean.Op.fmod derivative 0 1]
theorem fmod_backward_eq_adjoint (a b δ : Tensor ℝ s)
    (h : ∀ c, b c ≠ 0 ∧ (a c = 0 ∨ Int.fract (a c / b c) ≠ 0)) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
        (((Op.fmod (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
      (WithLp.toLp 2 (a, b)) ∧
    WithLp.toLp 2
        ((((Op.fmod (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).1,
        (((Op.fmod (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).2.1) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
        (((Op.fmod (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
        (WithLp.toLp 2 (a, b))).adjoint δ := by
  apply And.imp_left HasFDerivAt.differentiableAt
  refine pointwise₂_backward_eq_adjoint _ (fun _ q => Context.fmodReal q.1 q.2) (fun _ => 1)
    (fun c => -(truncRound (a c / b c) : ℝ)) a b δ _ _ (fun p c => ?_)
    (fun c => hasFDerivAt_fmodReal (h c).1 (h c).2) (fun c => ?_) (fun c => ?_)
  · exact Tensor.fmod_apply p.fst p.snd c
  · show δ c = 1 * δ c
    rw [one_mul]
  · show (Tensor.zipWith3 (fun a b g => -(g * Context.fmodQuotient a b)) a b δ) c = _
    rw [Tensor.zipWith3_apply, Context.fmodQuotient_real]
    ring

/-- With the divisor fixed, where every coordinate has a zero divisor entry, a zero dividend entry
or a quotient that is not an integer, the real forward of `Op.fmod` is differentiable in the
dividend and its stored pullback `δ` is the adjoint of the derivative. This is the exact set: for
`s = [1]` and small `ε > 0`, `fmod (1 - ε) 1` is `1 - ε` and `fmod 1 1` is `0`, and the real
`fmod t 0 = t` is an artefact of Lean's `t / 0 = 0`, where the float operator returns a NaN. -/
@[op_cert TorchLean.Op.fmod derivative 0]
theorem fmod_backward_eq_adjoint_dividend (a b δ : Tensor ℝ s)
    (h : ∀ c, b c = 0 ∨ a c = 0 ∨ Int.fract (a c / b c) ≠ 0) :
    DifferentiableAt ℝ (fun z => (((Op.fmod (s := s)).rule? (α := ℝ) |>.get rfl).forward
        (z, b, ())).1) a ∧
      (((Op.fmod (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).1 =
        (fderiv ℝ (fun z => (((Op.fmod (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (z, b, ())).1) a).adjoint δ := by
  have hd : ∀ c, HasDerivAt (fun t => Context.fmodReal t (b c)) 1 (a c) := fun c =>
    hasDerivAt_fmodReal_left (h c)
  op_deriv [Tensor.fmodBackward]

/-- With the dividend fixed, where every coordinate has a zero dividend entry, or a nonzero
divisor entry and a quotient that is not an integer, the real forward of `Op.fmod` is
differentiable in the divisor and its stored pullback `-(δ * trunc (a / b))` is the adjoint of the
derivative. This is the exact set: for `s = [1]` and small `ε > 0`, `fmod 1 (1 + ε)` is `1` and
`fmod 1 1` is `0`, and the real `fmod 1 0 = 1`, an artefact of Lean's `1 / 0 = 0` where the float
operator returns a NaN, differs from `fmod 1 t`, which lies below `|t|`. -/
@[op_cert TorchLean.Op.fmod derivative 1]
theorem fmod_backward_eq_adjoint_divisor (a b δ : Tensor ℝ s)
    (h : ∀ c, a c = 0 ∨ (b c ≠ 0 ∧ Int.fract (a c / b c) ≠ 0)) :
    DifferentiableAt ℝ (fun z => (((Op.fmod (s := s)).rule? (α := ℝ) |>.get rfl).forward
        (a, z, ())).1) b ∧
      (((Op.fmod (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).2.1 =
        (fderiv ℝ (fun z => (((Op.fmod (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (a, z, ())).1) b).adjoint δ := by
  have hd : ∀ c, HasDerivAt (fun t => Context.fmodReal (a c) t)
      (-(truncRound (a c / b c) : ℝ)) (b c) := fun c => hasDerivAt_fmodReal_right (h c)
  op_deriv [Tensor.fmodBackward, Context.fmodQuotient_real]

/-- At every input, the stored pullbacks of `Op.fmod` are `δ` and `-(δ * trunc (a / b))`. Where
the divisor is nonzero and the quotient is a nonzero integer `n` they are `δ` times the partials
from the side where the truncated quotient stays `n`, and at a zero divisor they rest on Lean's
`a / 0 = 0`, where the float operator returns a NaN. -/
@[op_cert TorchLean.Op.fmod convention]
theorem fmod_backward_eq (a b δ : Tensor ℝ s) :
    (((Op.fmod (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).1 = δ ∧
      (((Op.fmod (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).2.1 =
        -(δ * Tensor.trunc (a / b)) := by
  refine ⟨by simp only [rule?_fmod_real, Option.get_some, Tensor.fmodBackward],
    Tensor.Rep.ext fun c => ?_⟩
  simp only [rule?_fmod_real, Option.get_some, Tensor.fmodBackward, Tensor.zipWith3_apply,
    Context.fmodQuotient_real, Tensor.Rep.neg_apply, Tensor.mul_apply, Tensor.trunc_apply,
    Context.trunc_real, Tensor.div_apply]

end Op

/-! ## `remainder` -/

section Remainder

open Filter Topology

/-- Where the divisor is nonzero and the quotient is not an integer, the real `remainder` has
derivative `1` in the dividend and `-⌊a / b⌋` in the divisor. -/
theorem hasFDerivAt_remainder {a b : ℝ} (hb : b ≠ 0) (h : Int.fract (a / b) ≠ 0) :
    HasFDerivAt (fun q : ℝ × ℝ => Context.remainder q.1 q.2)
      ((1 : ℝ) • ContinuousLinearMap.fst ℝ ℝ ℝ +
        (-(⌊a / b⌋ : ℝ)) • ContinuousLinearMap.snd ℝ ℝ ℝ) (a, b) := by
  have ht : Tendsto (fun q : ℝ × ℝ => q.1 / q.2) (𝓝 (a, b)) (𝓝 (a / b)) :=
    (continuousAt_fst.div₀ continuousAt_snd hb).tendsto
  have hev : (fun q : ℝ × ℝ => Context.remainder q.1 q.2) =ᶠ[𝓝 (a, b)]
      fun q => q.1 - q.2 * (⌊a / b⌋ : ℝ) := by
    filter_upwards [ht.eventually (eventually_floor_eq h)] with q hq'
    rw [Context.remainder_real, hq']
  refine ((hasFDerivAt_fst.sub (hasFDerivAt_snd.mul_const _)).congr_fderiv
    (ContinuousLinearMap.ext fun q => ?_)).congr_of_eventuallyEq hev
  simp only [_root_.sub_apply, _root_.add_apply, _root_.smul_apply,
    ContinuousLinearMap.coe_fst', ContinuousLinearMap.coe_snd', smul_eq_mul]
  ring

/-- Where the divisor is zero or the quotient is not an integer, the real `remainder` has
derivative `1` in the dividend. -/
theorem hasDerivAt_remainder_left {a b : ℝ} (h : b = 0 ∨ Int.fract (a / b) ≠ 0) :
    HasDerivAt (fun t => Context.remainder t b) 1 a := by
  rcases h with rfl | hf
  · refine (hasDerivAt_id' a).congr_of_eventuallyEq (Eventually.of_forall fun t => ?_)
    simp only [Context.remainder_real, zero_mul, sub_zero]
  · have ht : Tendsto (fun t : ℝ => t / b) (𝓝 a) (𝓝 (a / b)) :=
      (continuous_id'.div_const b).tendsto a
    have hev : (fun t => Context.remainder t b) =ᶠ[𝓝 a] fun t => t - b * (⌊a / b⌋ : ℝ) := by
      filter_upwards [ht.eventually (eventually_floor_eq hf)] with t ht'
      rw [Context.remainder_real, ht']
    exact ((hasDerivAt_id' a).sub_const _).congr_of_eventuallyEq hev

/-- Where the dividend is zero, or the divisor is nonzero and the quotient is not an integer, the
real `remainder` has derivative `-⌊a / b⌋` in the divisor. -/
theorem hasDerivAt_remainder_right {a b : ℝ} (h : a = 0 ∨ (b ≠ 0 ∧ Int.fract (a / b) ≠ 0)) :
    HasDerivAt (fun t => Context.remainder a t) (-(⌊a / b⌋ : ℝ)) b := by
  rcases h with rfl | ⟨hb, hf⟩
  · rw [zero_div, Int.floor_zero, Int.cast_zero, neg_zero]
    refine (hasDerivAt_const b (0 : ℝ)).congr_of_eventuallyEq (Eventually.of_forall fun t => ?_)
    simp only [Context.remainder_real, zero_div, Int.floor_zero, Int.cast_zero, mul_zero,
      sub_zero]
  · have ht : Tendsto (fun t : ℝ => a / t) (𝓝 b) (𝓝 (a / b)) :=
      (continuousAt_const.div₀ continuousAt_id hb).tendsto
    have hev : (fun t => Context.remainder a t) =ᶠ[𝓝 b] fun t => a - t * (⌊a / b⌋ : ℝ) := by
      filter_upwards [ht.eventually (eventually_floor_eq hf)] with t ht'
      rw [Context.remainder_real, ht']
    exact ((hasDerivAt_mul_const (⌊a / b⌋ : ℝ)).const_sub a).congr_of_eventuallyEq hev

end Remainder

namespace Context

/-- Over the reals, `Context.remainderQuotient a b`, the factor in the divisor pullback of
`remainder`, is the floored quotient `⌊a / b⌋`, except at a zero divisor and a negative dividend,
where it is `-1`. -/
theorem remainderQuotient_real (a b : ℝ) :
    Context.remainderQuotient a b = if b = 0 ∧ a < 0 then -1 else (⌊a / b⌋ : ℝ) := by
  simp only [Context.remainderQuotient, fmodQuotient_real, Context.fmod_real]
  rcases eq_or_ne b 0 with rfl | hb
  · rcases lt_trichotomy a 0 with ha | rfl | ha
    · simp [truncRound, ha, ha.ne]
    · simp [truncRound]
    · simp [truncRound, not_lt.mpr ha.le]
  have hFloor := Context.floor_eq_truncRound (a / b)
  obtain ⟨t, ht⟩ : ∃ t : ℤ, truncRound (a / b) = t := ⟨_, rfl⟩
  rw [ht] at hFloor
  have hRemainder : a - b * (t : ℝ) = b * (a / b - t) := by
    field_simp
  have hc : ¬(b = 0 ∧ a < 0) := fun h => hb h.1
  rw [ite_eq_right hc]
  simp only [ht, hRemainder]
  rcases lt_trichotomy (a / b - t) 0 with hNeg | hZero | hPos
  · rw [hFloor.1 hNeg]
    rcases lt_or_gt_of_ne hb with hbNeg | hbPos
    · have hm : 0 < b * (a / b - t) := mul_pos_of_neg_of_neg hbNeg hNeg
      simp [hbNeg, hm.ne', not_lt.mpr hm.le]
    · have hm : b * (a / b - t) < 0 := mul_neg_of_pos_of_neg hbPos hNeg
      simp [not_lt.mpr hbPos.le, hm.ne, hm]
  · rw [hFloor.2 hZero.ge, hZero]
    simp
  · rw [hFloor.2 hPos.le]
    rcases lt_or_gt_of_ne hb with hbNeg | hbPos
    · have hm : b * (a / b - t) < 0 := mul_neg_of_neg_of_pos hbNeg hPos
      simp [hbNeg, hm]
    · have hm : 0 < b * (a / b - t) := mul_pos hbPos hPos
      simp [not_lt.mpr hbPos.le, not_lt.mpr hm.le]

end Context

namespace Op

variable {s : Shape}

/-- Over the reals, the rule of `Op.remainder` is its table arm. -/
@[op_deriv_simps] theorem rule?_remainder_real :
    (Op.remainder (s := s)).rule? (α := ℝ) = some
      { forward := fun xs => (Tensor.remainder xs.1 xs.2.1, ())
        backward := fun xs g =>
          let (da, db) := Tensor.remainderBackward xs.1 xs.2.1 g.1
          (da, db, ())
        jvp? := some fun xs dxs =>
          ((Tensor.remainderBackward xs.1 xs.2.1 dxs.1).1 +
            (Tensor.remainderBackward xs.1 xs.2.1 dxs.2.1).2, ()) } :=
  Op.rule?_remainder (α := ℝ) (s := s)

/-- Where every divisor entry is nonzero and no quotient `a c / b c` is an integer, the real
forward of `Op.remainder` is differentiable in both inputs and its stored pair is the adjoint of
the derivative. This is the exact set of the real model, whose value `a` at a zero divisor rests
on Lean's `a / 0 = 0` where the float operator returns a NaN: for `s = [1]` and small `ε > 0` the
forward is `1 - ε` at `(-ε, 1)` and `0` at `(0, 1)`, it is `1` at `(1, 0)` and below `ε` at
`(1, ε)`, and at `(0, 0)` it is `t` along `(t, 0)` and `0` along `(0, t)` and `(t, t)`, which no
linear map fits. -/
@[op_cert TorchLean.Op.remainder derivative 0 1]
theorem remainder_backward_eq_adjoint (a b δ : Tensor ℝ s)
    (h : ∀ c, b c ≠ 0 ∧ Int.fract (a c / b c) ≠ 0) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
        (((Op.remainder (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
      (WithLp.toLp 2 (a, b)) ∧
    WithLp.toLp 2
        ((((Op.remainder (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).1,
        (((Op.remainder (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).2.1) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
        (((Op.remainder (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
        (WithLp.toLp 2 (a, b))).adjoint δ := by
  apply And.imp_left HasFDerivAt.differentiableAt
  refine pointwise₂_backward_eq_adjoint _ (fun _ q => Context.remainder q.1 q.2) (fun _ => 1)
    (fun c => -(⌊a c / b c⌋ : ℝ)) a b δ _ _ (fun p c => ?_)
    (fun c => hasFDerivAt_remainder (h c).1 (h c).2) (fun c => ?_) (fun c => ?_)
  · exact Tensor.remainder_apply p.fst p.snd c
  · show δ c = 1 * δ c
    rw [one_mul]
  · show (Tensor.zipWith3 (fun a b g => -(g * Context.remainderQuotient a b)) a b δ) c = _
    rw [Tensor.zipWith3_apply, Context.remainderQuotient_real, ite_eq_right fun h' => (h c).1 h'.1]
    ring

/-- With the divisor fixed, where every coordinate has a zero divisor entry or a quotient that is
not an integer, the real forward of `Op.remainder` is differentiable in the dividend and its stored
pullback `δ` is the adjoint of the derivative. This is the exact set: for `s = [1]` and small
`ε > 0`, `remainder (-ε) 1` is `1 - ε` and `remainder 0 1` is `0`, and the real
`remainder t 0 = t` is an artefact of Lean's `t / 0 = 0`, where the float operator returns a NaN.
-/
@[op_cert TorchLean.Op.remainder derivative 0]
theorem remainder_backward_eq_adjoint_dividend (a b δ : Tensor ℝ s)
    (h : ∀ c, b c = 0 ∨ Int.fract (a c / b c) ≠ 0) :
    DifferentiableAt ℝ (fun z => (((Op.remainder (s := s)).rule? (α := ℝ) |>.get rfl).forward
        (z, b, ())).1) a ∧
      (((Op.remainder (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).1 =
        (fderiv ℝ (fun z => (((Op.remainder (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (z, b, ())).1) a).adjoint δ := by
  have hd : ∀ c, HasDerivAt (fun t => Context.remainder t (b c)) 1 (a c) := fun c =>
    hasDerivAt_remainder_left (h c)
  op_deriv [Tensor.remainder, Tensor.remainderBackward]

/-- With the dividend fixed, where every coordinate has a zero dividend entry, or a nonzero
divisor entry and a quotient that is not an integer, the real forward of `Op.remainder` is
differentiable in the divisor and its stored pullback `-(δ * ⌊a / b⌋)` is the adjoint of the
derivative. This is the exact set: for `s = [1]` and small `ε > 0`, `remainder 1 (1 + ε)` is `1`
and `remainder 1 1` is `0`, and the real `remainder 1 0 = 1`, an artefact of Lean's `1 / 0 = 0`
where the float operator returns a NaN, differs from `remainder 1 t`, which lies between `0` and
`t`. -/
@[op_cert TorchLean.Op.remainder derivative 1]
theorem remainder_backward_eq_adjoint_divisor (a b δ : Tensor ℝ s)
    (h : ∀ c, a c = 0 ∨ (b c ≠ 0 ∧ Int.fract (a c / b c) ≠ 0)) :
    DifferentiableAt ℝ (fun z => (((Op.remainder (s := s)).rule? (α := ℝ) |>.get rfl).forward
        (a, z, ())).1) b ∧
      (((Op.remainder (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).2.1 =
        (fderiv ℝ (fun z => (((Op.remainder (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (a, z, ())).1) b).adjoint δ := by
  have hd : ∀ c, HasDerivAt (fun t => Context.remainder (a c) t)
      (-Context.remainderQuotient (a c) (b c)) (b c) := fun c => by
    rw [Context.remainderQuotient_real, ite_eq_right fun h' => ?_]
    · exact hasDerivAt_remainder_right (h c)
    · rcases h c with h0 | ⟨hb, _⟩
      · exact h'.2.ne h0
      · exact hb h'.1
  op_deriv [Tensor.remainder, Tensor.remainderBackward]

/-- At every input, the stored pullbacks of `Op.remainder` are `δ` and, at each entry,
`-(δ c * ⌊a c / b c⌋)`, or `δ c` at a zero divisor and a negative dividend. Where the divisor is
nonzero and the quotient is an integer they are `δ` times the partials from the side where the
floored quotient keeps its value, and at a zero divisor they rest on Lean's `a / 0 = 0`, where the
float operator returns a NaN. -/
@[op_cert TorchLean.Op.remainder convention]
theorem remainder_backward_eq (a b δ : Tensor ℝ s) :
    (((Op.remainder (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).1 = δ ∧
      ∀ c, (((Op.remainder (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ())
        (δ, ())).2.1 c = if b c = 0 ∧ a c < 0 then δ c else -(δ c * ⌊a c / b c⌋) := by
  refine ⟨by simp only [rule?_remainder_real, Option.get_some, Tensor.remainderBackward],
    fun c => ?_⟩
  simp only [rule?_remainder_real, Option.get_some, Tensor.remainderBackward,
    Tensor.zipWith3_apply, Context.remainderQuotient_real]
  split_ifs <;> ring

/-! ## `div` at every rounding -/

/-- Where every divisor entry is nonzero, and the quotient `x c / y c` is not an integer under
flooring division and is zero or not an integer under truncating division, the real forward of
`Op.div` is differentiable in both inputs and its stored pair of pullbacks is the adjoint of the
derivative, zero for the rounded divisions. This is the exact set: true division is not continuous
at a zero divisor entry, and a rounded quotient jumps at the other points. -/
@[op_cert TorchLean.Op.div derivative]
theorem div_rounding_backward_eq_adjoint (rounding : DivRounding) (x y δ : Tensor ℝ s)
    (hy : ∀ c, y c ≠ 0) (hfloor : rounding = .floor → ∀ c, Int.fract (x c / y c) ≠ 0)
    (htrunc : rounding = .trunc → ∀ c, x c = 0 ∨ Int.fract (x c / y c) ≠ 0) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
        (((Op.div (s := s) rounding).rule? (α := ℝ) |>.get (by cases rounding <;> rfl)).forward
          (p.fst, p.snd, ())).1)
      (WithLp.toLp 2 (x, y)) ∧
    WithLp.toLp 2
        ((((Op.div (s := s) rounding).rule? (α := ℝ) |>.get (by cases rounding <;> rfl)).backward
          (x, y, ()) (δ, ())).1,
        (((Op.div (s := s) rounding).rule? (α := ℝ) |>.get (by cases rounding <;> rfl)).backward
          (x, y, ()) (δ, ())).2.1) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
        (((Op.div (s := s) rounding).rule? (α := ℝ) |>.get (by cases rounding <;> rfl)).forward
          (p.fst, p.snd, ())).1)
        (WithLp.toLp 2 (x, y))).adjoint δ := by
  cases rounding
  · exact (div_prod_backward_eq_adjoint x y δ hy).imp_left HasFDerivAt.differentiableAt
  · apply And.imp_left HasFDerivAt.differentiableAt
    refine pointwise₂_backward_eq_adjoint _ (fun _ q => ((truncRound (q.1 / q.2) : ℤ) : ℝ))
      (fun _ => 0) (fun _ => 0) x y δ _ _ (fun p c => ?_) (fun c => ?_) (fun c => ?_)
      (fun c => ?_)
    · show (Tensor.copysign (Tensor.floorDivide (Tensor.abs p.fst) (Tensor.abs p.snd))
        (p.fst / p.snd)) c = _
      rw [Tensor.copysign_floorDivide_abs_real, Tensor.trunc_real]
      simp only [Tensor.map_apply, Tensor.div_apply]
    · exact hasFDerivAt_intCast_div_of_eventually_eq (r := truncRound) (hy c)
        (eventually_truncRound_eq ((htrunc rfl c).imp_left fun h => by rw [h, zero_div]))
    all_goals
      show (Tensor.map (fun _ => (0 : ℝ)) δ) c = _
      simp only [Tensor.map_apply, zero_mul]
  · apply And.imp_left HasFDerivAt.differentiableAt
    refine pointwise₂_backward_eq_adjoint _ (fun _ q => ((⌊q.1 / q.2⌋ : ℤ) : ℝ))
      (fun _ => 0) (fun _ => 0) x y δ _ _ (fun p c => ?_) (fun c => ?_) (fun c => ?_)
      (fun c => ?_)
    · show (Tensor.floorDivide p.fst p.snd) c = _
      rw [Tensor.floorDivide_real, Tensor.floor_real]
      simp only [Tensor.map_apply, Tensor.div_apply]
    · exact hasFDerivAt_intCast_div_of_eventually_eq (r := Int.floor) (hy c)
        (eventually_floor_eq (hfloor rfl c))
    all_goals
      show (Tensor.map (fun _ => (0 : ℝ)) δ) c = _
      simp only [Tensor.map_apply, zero_mul]

/-- At every input, entry `c` of each stored pullback of `Op.div` is zero where the divisor entry
`y c` is zero, where true division has no derivative, and at every coordinate for the rounded
divisions, whose zero pullbacks are their derivative off the jumps of the rounding. -/
@[op_cert TorchLean.Op.div convention]
theorem div_backward_eq (rounding : DivRounding) (x y δ : Tensor ℝ s) (c : s.Coord)
    (h : rounding ≠ .none ∨ y c = 0) :
    (((Op.div (s := s) rounding).rule? (α := ℝ) |>.get (by cases rounding <;> rfl)).backward
        (x, y, ()) (δ, ())).1 c = 0 ∧
      (((Op.div (s := s) rounding).rule? (α := ℝ) |>.get (by cases rounding <;> rfl)).backward
        (x, y, ()) (δ, ())).2.1 c = 0 := by
  cases rounding
  · have hy := h.resolve_left fun h' => h' rfl
    constructor
    · show (δ / y) c = 0
      rw [Tensor.div_apply, hy, div_zero]
    · show (Tensor.neg (δ * ((x / y) / y))) c = 0
      simp only [Tensor.neg, Tensor.map_apply, Tensor.mul_apply,
        Tensor.div_apply, hy, div_zero, mul_zero, neg_zero]
  all_goals
    constructor <;>
    · show (Tensor.map (fun _ => (0 : ℝ)) δ) c = 0
      simp only [Tensor.map_apply]

end Op

end TorchLean
