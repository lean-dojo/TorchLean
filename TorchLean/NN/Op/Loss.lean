/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Tactic.OpDeriv
public import TorchLean.Tensor.Op.Table.Loss
public import TorchLean.NN.Op.Softmax
public import TorchLean.NN.Loss.Clamp
public import TorchLean.NN.Loss.Cosine
public import TorchLean.Tensor.Elementwise.Real

/-!
# Derivative certificates of the loss operators

We prove the `derivative` certificates of the loss operators over the reals: the stored backward
map of each loss is the adjoint of the Fréchet derivative of its forward map in the value inputs,
at every input for the smooth losses and, for the others, on the exact set where the forward is
differentiable and the stored backward is its adjoint. Each loss is first stated over ℝ with
`Tensor.map`, `Tensor.sum`, entrywise products, constant multiples, softmax and log-softmax (the
`_real` lemmas, registered with `@[op_deriv_simps]`), and `op_deriv` composes the pullback rules
and matches the stored backward with the result. Log-softmax along an axis enters through the
rule `HasPullbackAt.logSoftmax`, which we build from its in-tree certificate. For the
distillation loss we show, fiber by fiber, that both numerical branches of the stored gradient
give the student softmax minus the teacher softmax. For the cosine loss we glue the two branches
of the clipped norm `max ‖x‖ ε` along the sphere `‖x‖ = ε` and register the result as the rule
`HasPullbackAt.cosineLoss`.

For a loss with kinks, the derivative certificate assumes this exact set, which contains the
points where two branches meet with equal slopes, for every epsilon. For a nonpositive epsilon the
probability losses are not clamped away from the singularity of the logarithm and the cosine loss
is not clamped away from a zero norm, so their exact sets also exclude some inputs with a
probability `0` or `1` or a zero prediction. A `convention` certificate states the stored backward
in closed form, which records its value at the kinks.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)
open TorchLean.Shape TorchLean.Vec TorchLean.LogSoftmax TorchLean.Softmax
open scoped RealInnerProductSpace

namespace TorchLean

/-! ## Scalar derivatives and identities -/

/-- The derivative of `fun d => Real.log (Real.cosh d)` at `x` is `Real.tanh x`. -/
theorem hasDerivAt_log_cosh (x : ℝ) :
    HasDerivAt (fun d => Real.log (Real.cosh d)) (Real.tanh x) x := by
  have h := (Real.hasDerivAt_cosh x).log (Real.cosh_pos x).ne'
  rwa [← Real.tanh_eq_sinh_div_cosh] at h

/-- Away from `m = 1`, the hinge `max (1 - m) 0` has derivative `-1` below one and `0` above
one. -/
theorem hasDerivAt_max_one_sub {m : ℝ} (hm : m ≠ 1) :
    HasDerivAt (fun m => max (1 - m) 0) (-(if m < 1 then 1 else 0)) m := by
  rcases lt_or_gt_of_ne hm with h | h
  · simp only [h, ↓reduceIte]
    have hl : HasDerivAt (fun m : ℝ => 1 - m) (-1) m := by
      simpa using (hasDerivAt_id m).const_sub 1
    refine hl.congr_of_eventuallyEq ?_
    filter_upwards [Iio_mem_nhds h] with y hy
    exact max_eq_left (by simp only [Set.mem_Iio] at hy; linarith)
  · simp only [not_lt.mpr h.le, ↓reduceIte, neg_zero]
    refine (hasDerivAt_const m (0 : ℝ)).congr_of_eventuallyEq ?_
    filter_upwards [Ioi_mem_nhds h] with y hy
    exact max_eq_right (by simp only [Set.mem_Ioi] at hy; linarith)

/-- Over the reals, both branches of the log-cosh entry, the one for `|d| < 1` and the one for
larger `|d|`, equal `Real.log (Real.cosh d)`. -/
theorem logCosh_branch_real (d : ℝ) :
    (if |d| < 1 then Context.log1p (2 * Real.sinh (|d| / 2) * Real.sinh (|d| / 2))
      else |d| + Context.log1p (Real.exp (-(2 * |d|))) - Real.log 2) =
      Real.log (Real.cosh d) := by
  rw [← Real.cosh_abs d]
  split_ifs
  · rw [Context.log1p_eq_log_one_add]
    congr 1
    have h := Real.cosh_two_mul (|d| / 2)
    rw [mul_div_cancel₀ _ two_ne_zero, Real.cosh_sq] at h
    rw [h]
    ring
  · rw [Context.log1p_eq_log_one_add, Real.cosh_eq]
    have h1 : Real.exp |d| + Real.exp (-|d|) = Real.exp |d| * (1 + Real.exp (-(2 * |d|))) := by
      rw [mul_add, mul_one, ← Real.exp_add]
      ring_nf
    rw [h1, Real.log_div (by positivity) two_ne_zero, Real.log_mul (by positivity)
      (by positivity), Real.log_exp]

/-- A function that agrees with `g₁` on `Set.Ioc b a` and with `g₂` on `Set.Ico a c`, where
`b < a < c` and both have derivative `f'` at `a`, has derivative `f'` at `a`. -/
theorem hasDerivAt_of_eqOn_Ioc_Ico {f g₁ g₂ : ℝ → ℝ} {a b c f' : ℝ} (hb : b < a) (hc : a < c)
    (h₁ : Set.EqOn f g₁ (Set.Ioc b a)) (h₂ : Set.EqOn f g₂ (Set.Ico a c))
    (hg₁ : HasDerivAt g₁ f' a) (hg₂ : HasDerivAt g₂ f' a) : HasDerivAt f f' a := by
  have h := (hg₁.hasDerivWithinAt.congr_of_mem (fun y hy => h₁ hy) ⟨hb, le_rfl⟩).union
    (hg₂.hasDerivWithinAt.congr_of_mem (fun y hy => h₂ hy) ⟨le_rfl, hc⟩)
  rw [Set.Ioc_union_Ico_eq_Ioo hb hc] at h
  exact h.hasDerivAt (Ioo_mem_nhds hb hc)

/-- The derivative of `fun y => y * y / 2` at `a` is `a`. -/
theorem hasDerivAt_mul_self_div_two (a : ℝ) : HasDerivAt (fun y : ℝ => y * y / 2) a a := by
  convert ((hasDerivAt_id' a).mul (hasDerivAt_id' a)).div_const 2 using 1
  ring

/-- Where `0 ≤ δ` or `e ≠ 0`, the Huber entry with threshold `δ` has derivative `e` at `e` when
`|e| < δ` and `δ * sign e` otherwise. -/
theorem hasDerivAt_huber {δ e : ℝ} (h : 0 ≤ δ ∨ e ≠ 0) :
    HasDerivAt (fun y => if |y| < δ then |y| / 2 * |y| else δ * (|y| - δ / 2))
      (if |e| < δ then e else δ * (SignType.sign e : ℝ)) e := by
  rcases lt_trichotomy |e| δ with he | he | he
  · simp only [he, ↓reduceIte]
    refine (hasDerivAt_mul_self_div_two e).congr_of_eventuallyEq ?_
    filter_upwards [(isOpen_lt continuous_abs continuous_const).mem_nhds he] with y hy
    simp only [hy, ↓reduceIte, div_mul_eq_mul_div, abs_mul_abs_self]
  · rcases eq_or_lt_of_le (abs_nonneg e) with h0 | h0
    · have hδ : δ = 0 := he ▸ h0.symm
      subst hδ
      have hf : (fun y : ℝ => if |y| < 0 then |y| / 2 * |y| else 0 * (|y| - 0 / 2)) =
          fun _ => 0 := by
        funext y
        simp only [not_lt.mpr (abs_nonneg y), ↓reduceIte, zero_mul]
      rw [hf]
      simp only [not_lt.mpr (abs_nonneg e), ↓reduceIte, zero_mul]
      exact hasDerivAt_const e 0
    · have hδ : 0 < δ := he ▸ h0
      rcases (abs_eq hδ.le).mp he with he' | he'
      · rw [he', abs_of_pos hδ]
        simp only [lt_irrefl, ↓reduceIte, sign_pos hδ, SignType.coe_one, mul_one]
        refine hasDerivAt_of_eqOn_Ioc_Ico (b := -δ) (c := δ + 1) (g₁ := fun y => y * y / 2)
          (g₂ := fun y => δ * (y - δ / 2)) (by linarith) (by linarith) ?_ ?_
          (hasDerivAt_mul_self_div_two δ) ?_
        · intro y hy
          rcases eq_or_lt_of_le hy.2 with rfl | hy2
          · simp only [abs_of_pos hδ, lt_irrefl, ↓reduceIte]
            ring
          · simp only [abs_lt.mpr ⟨hy.1, hy2⟩, ↓reduceIte, div_mul_eq_mul_div, abs_mul_abs_self]
        · intro y hy
          simp only [abs_of_pos (hδ.trans_le hy.1), not_lt.mpr hy.1, ↓reduceIte]
        · have h2 := ((hasDerivAt_id' δ).sub_const (δ / 2)).const_mul δ
          rwa [mul_one] at h2
      · rw [he', abs_neg, abs_of_pos hδ]
        simp only [lt_irrefl, ↓reduceIte, sign_neg (neg_neg_of_pos hδ), SignType.coe_neg_one,
          mul_neg_one]
        refine hasDerivAt_of_eqOn_Ioc_Ico (b := -δ - 1) (c := δ)
          (g₁ := fun y => δ * (-y - δ / 2)) (g₂ := fun y => y * y / 2) (by linarith)
          (by linarith) ?_ ?_ ?_ (hasDerivAt_mul_self_div_two (-δ))
        · intro y hy
          have hy0 : y < 0 := hy.2.trans_lt (neg_neg_of_pos hδ)
          simp only [abs_of_neg hy0, not_lt.mpr (by linarith [hy.2] : δ ≤ -y), ↓reduceIte]
        · intro y hy
          rcases eq_or_lt_of_le hy.1 with rfl | hy1
          · simp only [abs_neg, abs_of_pos hδ, lt_irrefl, ↓reduceIte]
            ring
          · simp only [abs_lt.mpr ⟨hy1, hy.2⟩, ↓reduceIte, div_mul_eq_mul_div, abs_mul_abs_self]
        · have h2 := (((hasDerivAt_id' (-δ)).neg).sub_const (δ / 2)).const_mul δ
          rwa [mul_neg_one] at h2
  · have he0 : e ≠ 0 := h.elim (fun hδ => abs_pos.mp (hδ.trans_lt he)) id
    simp only [not_lt.mpr he.le, ↓reduceIte]
    refine (((hasDerivAt_abs he0).sub_const (δ / 2)).const_mul δ).congr_of_eventuallyEq ?_
    filter_upwards [(isOpen_lt continuous_const continuous_abs).mem_nhds he] with y hy
    simp only [not_lt.mpr hy.le, ↓reduceIte]

/-- Over the reals, the stored Huber slope, written with nested comparisons, is `e` when
`|e| < δ` and `δ * sign e` otherwise. -/
theorem huber_slope_real (δ e : ℝ) :
    (if (if e > 0 then e else -e) < δ then e else if e > 0 then δ else if e < 0 then -δ else 0) =
      if |e| < δ then e else δ * (SignType.sign e : ℝ) := by
  rcases lt_trichotomy e 0 with h | rfl | h
  · simp only [gt_iff_lt, not_lt.mpr h.le, ↓reduceIte, h, abs_of_neg h, sign_neg h,
      SignType.coe_neg_one, mul_neg_one]
  · simp only [gt_iff_lt, lt_irrefl, ↓reduceIte, neg_zero, abs_zero, sign_zero,
      SignType.coe_zero, mul_zero]
  · simp only [gt_iff_lt, h, ↓reduceIte, abs_of_pos h, sign_pos h, SignType.coe_one, mul_one]

/-- The clamp of `v` into `[ε, 1 - ε]`, written with the two comparisons of the probability
losses, is `min (max v ε) (1 - ε)`. -/
theorem clamp_ite_eq_min_max (ε v : ℝ) :
    (if (if v > ε then v else ε) < 1 - ε then (if v > ε then v else ε) else 1 - ε) =
      Min.min (Max.max v ε) (1 - ε) := by
  have h : (if v > ε then v else ε) = Max.max v ε := by
    split_ifs with h
    · exact (max_eq_left h.le).symm
    · exact (max_eq_right (not_lt.mp h)).symm
  rw [h]
  split_ifs with h2
  · exact (min_eq_left h2.le).symm
  · exact (min_eq_right (not_lt.mp h2)).symm

/-- Where `1 / 2 ≤ ε`, `b = 0`, or `a` is neither a clamp kink `ε`, `1 - ε` nor, for `ε ≤ 0`,
zero, the entry `b * log (min (max y ε) (1 - ε))` has derivative `b / a` at `a` strictly inside
`(ε, 1 - ε)` and zero elsewhere. -/
theorem hasDerivAt_mul_log_clamp {ε a b : ℝ}
    (h : 1 / 2 ≤ ε ∨ b = 0 ∨ (a ≠ ε ∧ a ≠ 1 - ε ∧ (ε ≤ 0 → a ≠ 0))) :
    HasDerivAt (fun y => b * Real.log (Min.min (Max.max y ε) (1 - ε)))
      (if ε < a then (if a < 1 - ε then b / a else 0) else 0) a := by
  rcases h with hε | rfl | ⟨h1, h2, h0⟩
  · have hc : (fun y : ℝ => b * Real.log (Min.min (Max.max y ε) (1 - ε))) =
        fun _ => b * Real.log (1 - ε) := by
      funext y
      rw [min_eq_right ((by linarith : 1 - ε ≤ ε).trans (le_max_right y ε))]
    have hd : (if ε < a then (if a < 1 - ε then b / a else 0) else 0) = 0 := by
      split_ifs with h1 h2
      · exfalso
        linarith
      · rfl
      · rfl
    rw [hc, hd]
    exact hasDerivAt_const a _
  · simp only [zero_mul, zero_div, ite_self]
    exact hasDerivAt_const a 0
  · rcases lt_or_gt_of_ne h1 with ha | ha
    · simp only [not_lt.mpr ha.le, ↓reduceIte]
      refine (hasDerivAt_const a (b * Real.log (Min.min ε (1 - ε)))).congr_of_eventuallyEq ?_
      filter_upwards [Iio_mem_nhds ha] with y hy
      rw [max_eq_right (Set.mem_Iio.mp hy).le]
    · rcases lt_or_gt_of_ne h2 with hb | hb
      · have ha0 : a ≠ 0 := by
          rcases le_or_gt ε 0 with hε | hε
          · exact h0 hε
          · exact (hε.trans ha).ne'
        simp only [ha, hb, ↓reduceIte]
        rw [div_eq_mul_inv]
        refine ((Real.hasDerivAt_log ha0).const_mul b).congr_of_eventuallyEq ?_
        filter_upwards [Ioo_mem_nhds ha hb] with y hy
        rw [max_eq_left (Set.mem_Ioo.mp hy).1.le, min_eq_left (Set.mem_Ioo.mp hy).2.le]
      · simp only [ha, not_lt.mpr hb.le, ↓reduceIte]
        refine (hasDerivAt_const a (b * Real.log (1 - ε))).congr_of_eventuallyEq ?_
        filter_upwards [Ioi_mem_nhds hb] with y hy
        rw [min_eq_right ((Set.mem_Ioi.mp hy).le.trans (le_max_left y ε))]

/-- Away from `0` and `1`, the binary cross-entropy term `-(b log y + (1 - b) log (1 - y))` has
derivative `(a - b) / (a (1 - a))` at `a`. -/
theorem hasDerivAt_bce_log {a b : ℝ} (ha : a ≠ 0) (ha1 : a ≠ 1) :
    HasDerivAt (fun y => -(b * Real.log y + (1 - b) * Real.log (1 - y)))
      ((a - b) / (a * (1 - a))) a := by
  have h1 := (Real.hasDerivAt_log ha).const_mul b
  have h2 := (((hasDerivAt_id' a).const_sub 1).log (sub_ne_zero.mpr ha1.symm)).const_mul (1 - b)
  convert (h1.add h2).neg using 1
  field_simp [ha, sub_ne_zero.mpr ha1.symm]
  ring

/-- Where `1 / 2 ≤ ε`, or `b` equals `a` whenever `a` is a clamp kink `ε` or `1 - ε` and `a` is
off `0` and `1` for `ε ≤ 0`, the binary cross-entropy term at the clamped probability has
derivative `(a - b) / (a (1 - a))` at `a` strictly inside `(ε, 1 - ε)` and zero elsewhere. -/
theorem hasDerivAt_bce_clamp {ε a b : ℝ}
    (h : 1 / 2 ≤ ε ∨ ((a = ε → b = ε) ∧ (a = 1 - ε → b = 1 - ε) ∧ (ε ≤ 0 → a ≠ 0 ∧ a ≠ 1))) :
    HasDerivAt (fun y => -(b * Real.log (Min.min (Max.max y ε) (1 - ε)) +
        (1 - b) * Real.log (1 - Min.min (Max.max y ε) (1 - ε))))
      (if ε < a then (if a < 1 - ε then (a - b) / (a * (1 - a)) else 0) else 0) a := by
  rcases le_or_gt (1 / 2) ε with hε2 | hε2
  · have hc : ∀ y : ℝ, Min.min (Max.max y ε) (1 - ε) = 1 - ε := fun y =>
      min_eq_right ((by linarith : 1 - ε ≤ ε).trans (le_max_right y ε))
    have hd : (if ε < a then (if a < 1 - ε then (a - b) / (a * (1 - a)) else 0) else 0) = 0 := by
      split_ifs with h1 h2
      · exfalso
        linarith
      · rfl
      · rfl
    simp only [hc, hd]
    exact hasDerivAt_const a _
  obtain ⟨hl, hu, h0⟩ := h.resolve_left (not_le.mpr hε2)
  have hεu : ε < 1 - ε := by linarith
  have h01 : ε ≤ a → a ≤ 1 - ε → a ≠ 0 ∧ a ≠ 1 := fun h1 h2 => by
    rcases le_or_gt ε 0 with hε | hε
    · exact h0 hε
    · exact ⟨(hε.trans_le h1).ne', (h2.trans_lt (by linarith)).ne⟩
  rcases lt_trichotomy a ε with ha | ha | ha
  · simp only [not_lt.mpr ha.le, ↓reduceIte]
    refine (hasDerivAt_const a (-(b * Real.log (Min.min ε (1 - ε)) +
      (1 - b) * Real.log (1 - Min.min ε (1 - ε))))).congr_of_eventuallyEq ?_
    filter_upwards [Iio_mem_nhds ha] with y hy
    rw [max_eq_right (Set.mem_Iio.mp hy).le]
  · have h' := h01 ha.ge (ha.le.trans hεu.le)
    rw [ha] at h'
    rw [hl ha, ha]
    simp only [lt_irrefl, ↓reduceIte]
    refine hasDerivAt_of_eqOn_Ioc_Ico (b := ε - 1) (c := 1 - ε)
      (g₁ := fun _ => -(ε * Real.log ε + (1 - ε) * Real.log (1 - ε)))
      (g₂ := fun y => -(ε * Real.log y + (1 - ε) * Real.log (1 - y))) (by linarith) hεu ?_ ?_
      (hasDerivAt_const ε _) ?_
    · intro y hy
      simp only [max_eq_right hy.2, min_eq_left hεu.le]
    · intro y hy
      simp only [max_eq_left hy.1, min_eq_left hy.2.le]
    · have h2 := hasDerivAt_bce_log (b := ε) h'.1 h'.2
      simpa only [sub_self, zero_div] using h2
  · rcases lt_trichotomy a (1 - ε) with hb | hb | hb
    · have h' := h01 ha.le hb.le
      simp only [ha, hb, ↓reduceIte]
      refine (hasDerivAt_bce_log (b := b) h'.1 h'.2).congr_of_eventuallyEq ?_
      filter_upwards [Ioo_mem_nhds ha hb] with y hy
      rw [max_eq_left (Set.mem_Ioo.mp hy).1.le, min_eq_left (Set.mem_Ioo.mp hy).2.le]
    · have h' := h01 ha.le hb.le
      rw [hb] at h'
      rw [hu hb, hb]
      simp only [hεu, lt_irrefl, ↓reduceIte]
      refine hasDerivAt_of_eqOn_Ioc_Ico (b := ε) (c := 2 - ε)
        (g₁ := fun y => -((1 - ε) * Real.log y + (1 - (1 - ε)) * Real.log (1 - y)))
        (g₂ := fun _ => -((1 - ε) * Real.log (1 - ε) + (1 - (1 - ε)) * Real.log (1 - (1 - ε))))
        hεu (by linarith) ?_ ?_ ?_ (hasDerivAt_const _ _)
      · intro y hy
        simp only [max_eq_left hy.1.le, min_eq_left hy.2]
      · intro y hy
        simp only [min_eq_right (hy.1.trans (le_max_left y ε))]
      · have h2 := hasDerivAt_bce_log (a := 1 - ε) (b := 1 - ε) h'.1 h'.2
        simpa only [sub_self, zero_div] using h2
    · simp only [ha, not_lt.mpr hb.le, ↓reduceIte]
      refine (hasDerivAt_const a (-(b * Real.log (1 - ε) +
        (1 - b) * Real.log (1 - (1 - ε))))).congr_of_eventuallyEq ?_
      filter_upwards [Ioi_mem_nhds hb] with y hy
      rw [min_eq_right ((Set.mem_Ioi.mp hy).le.trans (le_max_left y ε))]

/-! ## Calculus of the clipped cosine similarity -/

section Cosine

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Away from the origin, the norm of a real inner product space has derivative
`‖x‖⁻¹ • innerSL ℝ x` at `x`. -/
theorem hasFDerivAt_norm_of_ne_zero {x : E} (hx : x ≠ 0) :
    HasFDerivAt (fun y : E => ‖y‖) (‖x‖⁻¹ • innerSL ℝ x) x := by
  have hx' : ‖x‖ ≠ 0 := norm_ne_zero_iff.mpr hx
  have h := (hasStrictFDerivAt_norm_sq x).hasFDerivAt.sqrt (pow_ne_zero 2 hx')
  simp only [Real.sqrt_sq (norm_nonneg _)] at h
  refine h.congr_fderiv (ContinuousLinearMap.ext fun u => ?_)
  simp only [_root_.smul_apply, smul_eq_mul, nsmul_eq_mul, Nat.cast_ofNat, innerSL_apply_apply]
  field_simp

/-- The map `y ↦ ⟪y, t⟫` has derivative `innerSL ℝ t` at every point. -/
theorem hasFDerivAt_inner_const_right (t x : E) :
    HasFDerivAt (fun y : E => ⟪y, t⟫) (innerSL ℝ t) x :=
  (innerSL ℝ t).hasFDerivAt.congr_of_eventuallyEq (Filter.Eventually.of_forall fun y => by
    rw [innerSL_apply_apply]
    exact real_inner_comm t y)

/-- For a constant `c`, the map `y ↦ 1 - ⟪y, t⟫ / c` has derivative `innerSL ℝ (-c⁻¹ • t)` at
every point. -/
theorem hasFDerivAt_one_sub_inner_div_const (t x : E) (c : ℝ) :
    HasFDerivAt (fun y : E => 1 - ⟪y, t⟫ / c) (innerSL ℝ (-c⁻¹ • t)) x := by
  have h := ((hasFDerivAt_inner_const_right t x).mul_const c⁻¹).const_sub 1
  convert h using 1
  · funext y
    rw [div_eq_mul_inv]
  · refine ContinuousLinearMap.ext fun u => ?_
    simp only [innerSL_apply_apply, real_inner_smul_left, _root_.neg_apply,
      _root_.smul_apply, smul_eq_mul]
    ring

/-- Away from the origin, the map `y ↦ 1 - ⟪y, t⟫ / (‖y‖ * T)` has derivative
`innerSL ℝ (c • x - (‖x‖ * T)⁻¹ • t)` at `x`, where `c = ⟪x, t⟫ / (‖x‖ ^ 3 * T)`. -/
theorem hasFDerivAt_one_sub_inner_div_norm_mul (t : E) {T : ℝ} {x : E} (hx : x ≠ 0) :
    HasFDerivAt (fun y : E => 1 - ⟪y, t⟫ / (‖y‖ * T))
      (innerSL ℝ ((⟪x, t⟫ / (‖x‖ ^ 3 * T)) • x - (‖x‖ * T)⁻¹ • t)) x := by
  rcases eq_or_ne T 0 with rfl | hT
  · simp only [mul_zero, div_zero, inv_zero, zero_smul, sub_zero, map_zero]
    exact hasFDerivAt_const 1 x
  have hx' : ‖x‖ ≠ 0 := norm_ne_zero_iff.mpr hx
  have hd := (hasDerivAt_inv (mul_ne_zero hx' hT)).comp_hasFDerivAt x
    ((hasFDerivAt_norm_of_ne_zero hx).mul_const T)
  have h := ((hasFDerivAt_inner_const_right t x).mul hd).const_sub 1
  convert h using 1
  · funext y
    simp only [Pi.mul_apply, Function.comp_apply, div_eq_mul_inv]
  · refine ContinuousLinearMap.ext fun u => ?_
    simp only [innerSL_apply_apply, inner_sub_left, real_inner_smul_left,
      _root_.neg_apply, _root_.add_apply, _root_.smul_apply, smul_eq_mul,
      Function.comp_apply]
    field_simp
    ring

/-- If `f` agrees with `g₁` on `A` and with `g₂` on `B`, the sets `A` and `B` cover the space and
both contain `x`, and `g₁` and `g₂` have the same derivative `f'` at `x`, then `f` has derivative
`f'` at `x`. -/
theorem hasFDerivAt_of_eqOn_of_union_eq_univ {G F : Type*} [NormedAddCommGroup G]
    [NormedSpace ℝ G] [NormedAddCommGroup F] [NormedSpace ℝ F] {f g₁ g₂ : G → F}
    {f' : G →L[ℝ] F} {A B : Set G} {x : G} (hAB : A ∪ B = Set.univ) (hxA : x ∈ A)
    (hxB : x ∈ B) (h₁ : Set.EqOn f g₁ A) (h₂ : Set.EqOn f g₂ B) (hg₁ : HasFDerivAt g₁ f' x)
    (hg₂ : HasFDerivAt g₂ f' x) : HasFDerivAt f f' x := by
  have h := (hg₁.hasFDerivWithinAt.congr h₁ (h₁ hxA)).union
    (hg₂.hasFDerivWithinAt.congr h₂ (h₂ hxB))
  rwa [hAB, hasFDerivWithinAt_univ] at h

/-- Where `0 < ε`, `x ≠ 0` or `t = 0`, and where `‖x‖ = ε` forces `⟪x, t⟫ = 0`, the clipped
cosine loss `y ↦ 1 - ⟪y, t⟫ / (max ‖y‖ ε * max ‖t‖ ε)` has derivative `innerSL ℝ G` at `x`. Here
`G = c • x - (max ‖x‖ ε * max ‖t‖ ε)⁻¹ • t`, where `c = ⟪x, t⟫ / (max ‖x‖ ε ^ 3 * max ‖t‖ ε)` for
`ε < ‖x‖` and `c = 0` otherwise. -/
theorem hasFDerivAt_cosine_clipped {ε : ℝ} (t x : E) (hε : 0 < ε ∨ x ≠ 0 ∨ t = 0)
    (hx : ‖x‖ = ε → ⟪x, t⟫ = 0) :
    HasFDerivAt (fun y : E => 1 - ⟪y, t⟫ / (max ‖y‖ ε * max ‖t‖ ε))
      (innerSL ℝ ((if ε < ‖x‖ then ⟪x, t⟫ / (max ‖x‖ ε ^ 3 * max ‖t‖ ε) else 0) • x -
        (max ‖x‖ ε * max ‖t‖ ε)⁻¹ • t)) x := by
  rcases eq_or_ne t 0 with rfl | ht
  · simp only [inner_zero_right, zero_div, sub_zero, ite_self, zero_smul, smul_zero, map_zero]
    exact hasFDerivAt_const 1 x
  have hx0 : ε ≤ ‖x‖ → x ≠ 0 := fun h => by
    rcases hε with hε | hx0 | ht0
    · exact norm_pos_iff.mp (hε.trans_le h)
    · exact hx0
    · exact absurd ht0 ht
  rcases lt_trichotomy ε ‖x‖ with h | h | h
  · simp only [h, ↓reduceIte, max_eq_left h.le]
    refine (hasFDerivAt_one_sub_inner_div_norm_mul t (hx0 h.le)).congr_of_eventuallyEq ?_
    filter_upwards [(isOpen_lt continuous_const continuous_norm).mem_nhds h] with y hy
    rw [max_eq_left (show ε < ‖y‖ from hy).le]
  · have hxt : ⟪x, t⟫ = 0 := hx h.symm
    have hG : (if ε < ‖x‖ then ⟪x, t⟫ / (max ‖x‖ ε ^ 3 * max ‖t‖ ε) else 0) • x -
        (max ‖x‖ ε * max ‖t‖ ε)⁻¹ • t = -(ε * max ‖t‖ ε)⁻¹ • t := by
      rw [← h]
      simp only [lt_irrefl, ↓reduceIte, zero_smul, zero_sub, max_self, neg_smul]
    rw [hG]
    refine hasFDerivAt_of_eqOn_of_union_eq_univ (A := {y | ε ≤ ‖y‖}) (B := {y | ‖y‖ ≤ ε})
      (g₁ := fun y => 1 - ⟪y, t⟫ / (‖y‖ * max ‖t‖ ε))
      (g₂ := fun y => 1 - ⟪y, t⟫ / (ε * max ‖t‖ ε))
      (Set.eq_univ_of_forall fun y => le_total ε ‖y‖) h.le h.ge
      (fun y hy => by rw [max_eq_left (show ε ≤ ‖y‖ from hy)])
      (fun y hy => by rw [max_eq_right (show ‖y‖ ≤ ε from hy)]) ?_
      (hasFDerivAt_one_sub_inner_div_const t x _)
    refine (hasFDerivAt_one_sub_inner_div_norm_mul t (hx0 h.le)).congr_fderiv ?_
    rw [hxt, zero_div, zero_smul, zero_sub, ← h, neg_smul]
  · simp only [not_lt.mpr h.le, ↓reduceIte, max_eq_right h.le, zero_smul, zero_sub]
    refine ((hasFDerivAt_one_sub_inner_div_const t x (ε * max ‖t‖ ε)).congr_fderiv
      (by rw [neg_smul])).congr_of_eventuallyEq ?_
    filter_upwards [(isOpen_lt continuous_norm continuous_const).mem_nhds h] with y hy
    rw [max_eq_right (show ‖y‖ < ε from hy).le]

end Cosine

/-! ## Softmax identities -/

/-- Entry `i` of the analytic softmax of `v` is `Real.exp (v i)` divided by the sum of the
exponentials of the entries of `v`. -/
theorem Vec.softmax_apply_eq_exp_div {n : Nat} (v : Vec n) (i : Fin n) :
    Vec.softmax v i = Real.exp (v i) / ∑ k, Real.exp (v k) := by
  cases n with
  | zero => exact i.elim0
  | succ n => rfl

/-- The entries of the analytic softmax of a nonempty vector sum to one. -/
theorem Vec.sum_softmax_apply {n : Nat} (v : Vec (n + 1)) : ∑ i, Vec.softmax v i = 1 := by
  simp only [Vec.softmax_apply_eq_exp_div]
  rw [← Finset.sum_div]
  exact div_self (Vec.sumExp_ne_zero v)

/-- The log-softmax pullback at `v` of the softmax of `w` is the softmax of `w` minus the softmax
of `v`. -/
theorem Vec.logSoftmaxVJP_softmax {n : Nat} (v w : Vec n) (j : Fin n) :
    logSoftmaxVJP v (Vec.softmax w) j = Vec.softmax w j - Vec.softmax v j := by
  cases n with
  | zero => exact j.elim0
  | succ n =>
      simp only [logSoftmaxVJP, PiLp.toLp_apply, Vec.sum_softmax_apply, mul_one]

/-- Let `P` and `Q` be the softmaxes of the logits `T` and `X` of one fiber, `S0` the `P`-mean of
`X - T` and `L` the log-normalizer of the centered difference `X - T - S0`. Then the entry at `j`
of the two-branch difference that the distillation backward stores, minus `Q j` times the sum of
its entries, is `Q j - P j`. -/
theorem distill_fiber_eq_sub {m : Nat} (X T P Q : Fin m → ℝ) (S0 L : ℝ)
    (hP : ∀ i, P i = Real.exp (T i) / ∑ k, Real.exp (T k))
    (hQ : ∀ i, Q i = Real.exp (X i) / ∑ k, Real.exp (X k))
    (hS0 : S0 = ∑ i, P i * (X i - T i))
    (hL : L = Real.log (1 + ∑ i, P i * (Real.exp (X i - T i - S0) - 1))) (j : Fin m) :
    (if |X j - T j - S0 - L| < 1 then P j * (Real.exp (X j - T j - S0 - L) - 1) else 0) +
        (if |X j - T j - S0 - L| < 1 then 0 else Q j - P j) -
      Q j * ∑ i, ((if |X i - T i - S0 - L| < 1 then P i * (Real.exp (X i - T i - S0 - L) - 1)
        else 0) + (if |X i - T i - S0 - L| < 1 then 0 else Q i - P i)) = Q j - P j := by
  have hm : (Finset.univ : Finset (Fin m)).Nonempty := ⟨j, Finset.mem_univ j⟩
  have hA : 0 < ∑ k, Real.exp (T k) := Finset.sum_pos (fun k _ => Real.exp_pos _) hm
  have hB : 0 < ∑ k, Real.exp (X k) := Finset.sum_pos (fun k _ => Real.exp_pos _) hm
  have hPsum : ∑ i, P i = 1 := by
    simp only [hP]
    rw [← Finset.sum_div, div_self hA.ne']
  have hQsum : ∑ i, Q i = 1 := by
    simp only [hQ]
    rw [← Finset.sum_div, div_self hB.ne']
  have hw : ∀ i, P i * Real.exp (X i - T i - S0) =
      Real.exp (X i) * (Real.exp (-S0) / ∑ k, Real.exp (T k)) := by
    intro i
    rw [hP, Real.exp_sub, Real.exp_sub, Real.exp_neg]
    field_simp
  have hZ : 1 + ∑ i, P i * (Real.exp (X i - T i - S0) - 1) =
      (∑ k, Real.exp (X k)) * (Real.exp (-S0) / ∑ k, Real.exp (T k)) := by
    simp only [mul_sub, mul_one, Finset.sum_sub_distrib, hPsum, hw, ← Finset.sum_mul]
    ring
  have hZpos : 0 < (∑ k, Real.exp (X k)) * (Real.exp (-S0) / ∑ k, Real.exp (T k)) := by
    positivity
  have hratio : ∀ i, P i * Real.exp (X i - T i - S0 - L) = Q i := by
    intro i
    rw [hL, hZ, Real.exp_sub, Real.exp_log hZpos, ← mul_div_assoc, hw, hQ]
    field_simp
  have hU : ∀ i, ((if |X i - T i - S0 - L| < 1 then P i * (Real.exp (X i - T i - S0 - L) - 1)
      else 0) + (if |X i - T i - S0 - L| < 1 then 0 else Q i - P i)) = Q i - P i := by
    intro i
    split_ifs
    · rw [mul_sub, mul_one, hratio]
      ring
    · ring
  simp only [hU]
  rw [Finset.sum_sub_distrib, hQsum, hPsum]
  ring

namespace Tensor

variable {s : Shape}

/-! ## Real semantics of the smooth losses -/

/-- Over the reals, the mean squared error is the mean of the squared differences. -/
@[op_deriv_simps] theorem meanSquaredError_real (p t : Tensor ℝ s) :
    meanSquaredError p t = (meanDenominator s : ℝ)⁻¹ • Tensor.sum ((p - t) * (p - t)) := by
  rw [meanSquaredError_of_isFinite p t (Context.isFinite_real _), Context.divNat_real,
    smul_eq_mul, div_eq_inv_mul]

/-- Over the reals, the stored gradient of the mean squared error is `2 / n` times `p - t`. -/
@[op_deriv_simps] theorem meanSquaredErrorBackward_real (p t : Tensor ℝ s) :
    meanSquaredErrorBackward p t = (2 / (meanDenominator s : ℝ)) • (p - t) :=
  Rep.ext fun c => by
    simp only [meanSquaredErrorBackward, Tensor.map_apply, Context.isFinite_real,
      ↓reduceIte, Context.divNat_real, Rep.smul_apply, smul_eq_mul]
    ring

/-- Over the reals, the Poisson loss is the mean of `exp p - t * p`. -/
@[op_deriv_simps] theorem poissonNLLLoss_real (p t : Tensor ℝ s) :
    poissonNLLLoss p t =
      (meanDenominator s : ℝ)⁻¹ • Tensor.sum (Tensor.map Real.exp p - t * p) := by
  rw [poissonNLLLoss, Context.divNat_real, smul_eq_mul, div_eq_inv_mul]
  rfl

/-- Over the reals, the stored gradient of the Poisson loss is `(exp p - t) / n`. -/
@[op_deriv_simps] theorem poissonNLLLossBackward_real (p t : Tensor ℝ s) :
    poissonNLLLossBackward p t = (meanDenominator s : ℝ)⁻¹ • (Tensor.map Real.exp p - t) :=
  Rep.ext fun c => by
    simp only [poissonNLLLossBackward, Tensor.map_apply, Context.divNat_real,
      Rep.smul_apply, smul_eq_mul]
    rw [div_eq_inv_mul]
    rfl

/-- Over the reals, the log-cosh loss is the mean of `Real.log (Real.cosh (p - t))`. -/
@[op_deriv_simps] theorem logCoshLoss_real (p t : Tensor ℝ s) :
    logCoshLoss p t = (meanDenominator s : ℝ)⁻¹ •
      Tensor.sum (Tensor.map (fun d => Real.log (Real.cosh d)) (p - t)) := by
  rw [logCoshLoss, Context.divNat_real, smul_eq_mul, div_eq_inv_mul]
  congr 3
  funext d
  have hd : (MathFunctions.abs d == d || MathFunctions.abs d == -d) = true := by
    change (|d| == d || |d| == -d) = true
    rcases abs_choice d with h | h <;> simp [h]
  simp only [hd, ↓reduceIte]
  exact logCosh_branch_real d

/-- Over the reals, the stored gradient of the log-cosh loss is `tanh (p - t) / n`. -/
@[op_deriv_simps] theorem logCoshLossBackward_real (p t : Tensor ℝ s) :
    logCoshLossBackward p t = (meanDenominator s : ℝ)⁻¹ • Tensor.map Real.tanh (p - t) :=
  Rep.ext fun c => by
    simp only [logCoshLossBackward, Tensor.map_apply, Context.divNat_real,
      Rep.smul_apply, smul_eq_mul]
    rw [div_eq_inv_mul]
    rfl

/-- The fiber of `F` along `dim` through `c`, read at the index of `c` along `dim`, is the entry
of `F` at `c`. -/
theorem getScalar_fiber_self {α : Type} [Storage α] (F : Tensor α s) (dim : Fin s.rank)
    (c : s.Coord) : (F.fiber dim c).getScalar (Coord.axisEquiv dim (shape := s) c).1 = F c := by
  rw [getScalar_eq_apply, fiber_apply]
  simp only [Prod.mk.eta, Equiv.symm_apply_apply]

end Tensor

/-- Over the reals, the weighted cross-entropy terms along `dim` are the target times the
log-softmax of the logits along `dim`. -/
@[op_deriv_simps] theorem CrossEntropy.terms_real {s : Shape} (dim : Fin s.rank)
    (x t : Tensor ℝ s) : CrossEntropy.terms dim x t = t * Tensor.logSoftmax x dim :=
  Tensor.Rep.ext fun c => by
    rw [CrossEntropy.terms_apply, vectorTerms_real, ← Op.fiber_logSoftmax, Tensor.mul_def,
      Tensor.getScalar_zipWith, Tensor.getScalar_fiber_self, Tensor.getScalar_fiber_self,
      Tensor.mul_apply]

namespace Tensor

variable {s : Shape}

/-- Over the reals, the logits cross-entropy along `dim` is minus the sum of the target times the
log-softmax, divided by the number of fibers when `mean` holds. -/
@[op_deriv_simps] theorem crossEntropy_logits_real (x t : Tensor ℝ s) (dim : Fin s.rank)
    (mean : Bool) :
    crossEntropy .logits x t dim mean =
      (if mean then -(axisMeanDenom s dim : ℝ)⁻¹ else -1) •
        Tensor.sum (t * Tensor.logSoftmax x dim) := by
  cases mean <;>
    simp only [crossEntropy, CrossEntropy.terms_real, meanOverAxisSlices, Context.divNat_real,
      Bool.false_eq_true, ↓reduceIte, smul_eq_mul] <;> ring

/-- Over the reals, the distillation loss along `dim` is minus the sum of the teacher softmax
times the student log-softmax, divided by the number of fibers. The zero-weight shortcut of the
forward gives the same value. -/
@[op_deriv_simps] theorem distillLoss_real (dim : Fin s.rank) (x t : Tensor ℝ s) :
    distillLoss dim x t = -(axisMeanDenom s dim : ℝ)⁻¹ •
      Tensor.sum (Tensor.softmax t dim * Tensor.logSoftmax x dim) := by
  have h : zipWith (fun w lp => if w == 0 then 0 else w * lp) (Tensor.softmax t dim)
      (Tensor.logSoftmax x dim) = Tensor.softmax t dim * Tensor.logSoftmax x dim :=
    Rep.ext fun c => by
      simp only [zipWith_apply, Tensor.mul_apply]
      split_ifs with hw
      · rw [beq_iff_eq.mp hw, zero_mul]
      · rfl
  simp only [distillLoss, h, meanOverAxisSlices, Context.divNat_real, smul_eq_mul]
  ring

/-- Adding the entries of `Tensor.ofFn g` to zero in order gives the sum of the values of `g`. -/
theorem foldl_add_ofFn {n : Nat} (g : Fin n → ℝ) :
    Tensor.foldl (· + ·) 0 (Tensor.ofFn g) = ∑ i, g i := by
  have h := sum_eq_foldl (Tensor.ofFn g)
  rw [sum_vec] at h
  simp only [getScalar_ofFn] at h
  exact h.symm

/-! ## Real semantics of the losses with kinks -/

/-- Over the reals, the L1 loss is the mean of `|p - t|`. -/
@[op_deriv_simps] theorem l1Loss_real (p t : Tensor ℝ s) :
    l1Loss p t = (meanDenominator s : ℝ)⁻¹ • Tensor.sum (Tensor.map (fun d => |d|) (p - t)) := by
  rw [l1Loss, Context.divNat_real, smul_eq_mul, div_eq_inv_mul]
  rfl

/-- Over the reals, the stored gradient of the L1 loss is the sign of `p - t` divided by `n`,
zero where `p` and `t` agree. -/
@[op_deriv_simps] theorem l1LossBackward_real (p t : Tensor ℝ s) :
    l1LossBackward p t =
      Tensor.map (fun d => (meanDenominator s : ℝ)⁻¹ * (SignType.sign d : ℝ)) (p - t) :=
  Rep.ext fun c => by
    simp only [l1LossBackward, Tensor.map_apply, Context.divNat_real]
    split_ifs with h1 h2
    · rw [sign_pos h1, SignType.coe_one, mul_one, one_div]
    · rw [sign_neg h2, SignType.coe_neg_one, mul_neg_one, neg_div, one_div]
    · rw [le_antisymm (not_lt.mp h1) (not_lt.mp h2), sign_zero, SignType.coe_zero, mul_zero,
        zero_div]

/-- Over the reals, the hinge loss is the mean of `max (1 - p * t) 0`. -/
@[op_deriv_simps] theorem hingeLoss_real (p t : Tensor ℝ s) :
    hingeLoss p t =
      (meanDenominator s : ℝ)⁻¹ • Tensor.sum (Tensor.map (fun m => max (1 - m) 0) (p * t)) := by
  rw [hingeLoss, Context.divNat_real, smul_eq_mul, div_eq_inv_mul]
  congr 3
  funext m
  simp only [Context.maximum_real]
  split_ifs with h
  · rw [beq_iff_eq.mp h, max_self]
  · rfl

/-- Over the reals, the stored gradient of the hinge loss is `-t / n` where `p * t < 1` and zero
elsewhere. -/
@[op_deriv_simps] theorem hingeLossBackward_real (p t : Tensor ℝ s) :
    hingeLossBackward p t =
      Tensor.map (fun m => (meanDenominator s : ℝ)⁻¹ * -(if m < 1 then 1 else 0)) (p * t) * t :=
  Rep.ext fun c => by
    simp only [hingeLossBackward, Tensor.map_apply, Context.divNat_real,
      Tensor.mul_apply, Tensor.neg, beq_self_eq_true, ↓reduceIte]
    split_ifs with h1 h2 h2
    · ring
    · exact absurd (by linarith) h2
    · exact absurd (by linarith) h1
    · ring

/-- Over the reals, the Huber loss with threshold `δ` is the sum of the Huber entries of `p - t`,
divided by `n` when `mean` holds. -/
@[op_deriv_simps] theorem huberLoss_real (p t : Tensor ℝ s) (δ : ℝ) (mean : Bool) :
    huberLoss p t δ mean = (if mean then (meanDenominator s : ℝ)⁻¹ else 1) •
      Tensor.sum (Tensor.map (fun e => if |e| < δ then |e| / 2 * |e| else δ * (|e| - δ / 2))
        (p - t)) := by
  have h : zipWith (fun p t => Huber.term p t δ) p t =
      Tensor.map (fun e => if |e| < δ then |e| / 2 * |e| else δ * (|e| - δ / 2)) (p - t) :=
    Rep.ext fun c => by
      simp only [zipWith_apply, Tensor.map_apply, Rep.hSub_apply, Huber.term, sub_self,
        beq_self_eq_true, Bool.true_or, ↓reduceIte, LawfulContext.abs_eq]
  cases mean
  · simp only [huberLoss, h, Bool.false_eq_true, ↓reduceIte, one_smul]
  · simp only [huberLoss, h, Context.divNat_real, ↓reduceIte, smul_eq_mul, div_eq_inv_mul]

/-- Over the reals, the stored gradient of the Huber loss is the Huber slope of `p - t`, divided
by `n` when `mean` holds. -/
@[op_deriv_simps] theorem huberLossBackward_real (p t : Tensor ℝ s) (δ : ℝ) (mean : Bool) :
    huberLossBackward p t δ mean = (if mean then (meanDenominator s : ℝ)⁻¹ else 1) •
      Tensor.map (fun e => if |e| < δ then e else δ * (SignType.sign e : ℝ)) (p - t) := by
  refine Rep.ext fun c => ?_
  cases mean
  · simp only [huberLossBackward, Tensor.map_apply, Rep.smul_apply, smul_eq_mul,
      Bool.false_eq_true, ↓reduceIte, one_mul, huber_slope_real]
  · simp only [huberLossBackward, Tensor.map_apply, Rep.smul_apply, smul_eq_mul,
      ↓reduceIte, Context.divNat_real, huber_slope_real, div_eq_inv_mul]

/-! ## Real semantics of the clamped probability losses -/

/-- Over the reals, the probability cross-entropy along `dim` is minus the sum of the target times
the logarithm of the clamped probabilities, divided by the number of fibers when `mean` holds. -/
@[op_deriv_simps] theorem crossEntropy_probabilities_real (x t : Tensor ℝ s) (dim : Fin s.rank)
    (ε : ℝ) (mean : Bool) :
    crossEntropy (.probabilities ε) x t dim mean =
      (if mean then -(axisMeanDenom s dim : ℝ)⁻¹ else -1) • Tensor.sum (Tensor.zipWith
        (fun a b => b * Real.log (Min.min (Max.max a ε) (1 - ε))) x t) := by
  have h : t * Tensor.log (Tensor.map (fun v : ℝ =>
        if (if v > ε then v else ε) < 1 - ε then (if v > ε then v else ε) else 1 - ε) x) =
      Tensor.zipWith (fun a b => b * Real.log (Min.min (Max.max a ε) (1 - ε))) x t :=
    Rep.ext fun c => by
      simp only [Tensor.mul_apply, Tensor.log, Tensor.map_apply, zipWith_apply,
        clamp_ite_eq_min_max]
      rfl
  rw [crossEntropy_probabilities_eq]
  cases mean <;>
    simp only [h, meanOverAxisSlices, Context.divNat_real, Bool.false_eq_true, ↓reduceIte,
      smul_eq_mul] <;> ring

/-- Over the reals, the stored gradient of the probability cross-entropy is `-t / x` strictly
inside `(ε, 1 - ε)` and zero elsewhere, divided by the number of fibers when `mean` holds. -/
@[op_deriv_simps] theorem crossEntropyBackward_probabilities_real (x t : Tensor ℝ s)
    (dim : Fin s.rank) (ε : ℝ) (mean : Bool) :
    crossEntropyBackward (.probabilities ε) x t dim mean =
      (if mean then -(axisMeanDenom s dim : ℝ)⁻¹ else -1) •
        Tensor.zipWith (fun a b => if ε < a then (if a < 1 - ε then b / a else 0) else 0) x t :=
  Rep.ext fun c => by
    cases mean <;>
      simp only [crossEntropyBackward, Bool.false_eq_true, ↓reduceIte, Tensor.map_apply,
        Context.divNat_real, Rep.smul_apply, smul_eq_mul, zipWith_apply, gt_iff_lt] <;>
      split_ifs <;> ring

/-- Over the reals, the binary cross-entropy is the mean of `-(t log q + (1 - t) log (1 - q))` at
the clamped probability `q = min (max p ε) (1 - ε)`. -/
@[op_deriv_simps] theorem binaryCrossEntropy_real (p t : Tensor ℝ s) (ε : ℝ) :
    binaryCrossEntropy p t ε = (meanDenominator s : ℝ)⁻¹ • Tensor.sum (Tensor.zipWith
      (fun a b => -(b * Real.log (Min.min (Max.max a ε) (1 - ε)) +
        (1 - b) * Real.log (1 - Min.min (Max.max a ε) (1 - ε)))) p t) := by
  have h : zipWith (fun p y => BinaryCrossEntropy.term p y ε) p t = Tensor.zipWith
      (fun a b => -(b * Real.log (Min.min (Max.max a ε) (1 - ε)) +
        (1 - b) * Real.log (1 - Min.min (Max.max a ε) (1 - ε)))) p t :=
    Rep.ext fun c => by
      have hc := BinaryCrossEntropy.term_eq (p c) (t c) ε
      simp only at hc
      simp only [zipWith_apply, hc, clamp_ite_eq_min_max]
  simp only [binaryCrossEntropy, h, Context.divNat_real, smul_eq_mul, div_eq_inv_mul]

/-- Over the reals, the stored gradient of the binary cross-entropy is `(p - t) / (p (1 - p))`
over `n` strictly inside `(ε, 1 - ε)` and zero elsewhere. -/
@[op_deriv_simps] theorem binaryCrossEntropyBackward_real (p t : Tensor ℝ s) (ε : ℝ) :
    binaryCrossEntropyBackward p t ε = (meanDenominator s : ℝ)⁻¹ • Tensor.zipWith
      (fun a b => if ε < a then (if a < 1 - ε then (a - b) / (a * (1 - a)) else 0) else 0) p t :=
  Rep.ext fun c => by
    simp only [binaryCrossEntropyBackward, Tensor.map_apply, zipWith_apply,
      BinaryCrossEntropy.termBackward, Context.divNat_real, Rep.smul_apply, smul_eq_mul,
      gt_iff_lt]
    exact div_eq_inv_mul _ _

/-! ## Real semantics of the cosine loss -/

/-- Over the reals, the sum of the entrywise product of two tensors is their inner product. -/
theorem sum_mul_eq_inner (x t : Tensor ℝ s) : Tensor.sum (x * t) = ⟪x, t⟫ :=
  (dot_eq_rep_dot x t).trans (dot_eq_inner x t)

/-- Over the reals, the square root of the sum of squares of a tensor is its norm. -/
theorem sqrt_sum_square_eq_norm (x : Tensor ℝ s) :
    Real.sqrt (Tensor.sum (Tensor.square x)) = ‖x‖ := by
  rw [norm_eq_sqrt_sum, sum_eq_coord_sum]
  simp only [square, Tensor.map_apply, sq]

/-- Over the reals, the absolute value of every entry of a tensor is at most its
`Tensor.maxAbs`. -/
theorem abs_le_maxAbs (x : Tensor ℝ s) (c : s.Coord) : |x c| ≤ Tensor.maxAbs x := by
  simp only [Tensor.maxAbs, beq_self_eq_true, Bool.not_true, Bool.false_eq_true, ↓reduceIte,
    Rep.foldl_eq_fin_foldl, Fin.foldl_eq_foldl_finRange]
  exact List.le_foldl_max_of_mem (List.finRange (Shape.size s))
    (fun i => MathFunctions.abs (Rep.getFlat x i)) (List.mem_finRange (Shape.Coord.linearize c))

end Tensor

/-- Over the reals and for every epsilon, `CosineLoss.normFactors x ε` is the pair of the scale
`S = max (Tensor.maxAbs x) |ε|` and the clipped scaled norm `max ‖x‖ ε / S`. Both components
are zero when `S` is zero. -/
theorem CosineLoss.normFactors_real {s : Shape} (x : Tensor ℝ s) (ε : ℝ) :
    CosineLoss.normFactors x ε =
      (max (Tensor.maxAbs x) |ε|, max ‖x‖ ε / max (Tensor.maxAbs x) |ε|) := by
  let scale := max (Tensor.maxAbs x) |ε|
  have hs : 0 ≤ scale := (abs_nonneg ε).trans (le_max_right _ _)
  have hsum : Tensor.sum (Tensor.map (fun value => (value / scale) * (value / scale)) x) =
      Tensor.sum (Tensor.square x) / scale ^ 2 := by
    rw [Tensor.sum_eq_coord_sum, Tensor.sum_eq_coord_sum, Finset.sum_div]
    apply Finset.sum_congr rfl
    intro i _
    simp only [Tensor.map_apply, Tensor.square]
    ring
  simp only [CosineLoss.normFactors, Context.stopGradient, Context.stopGradient?,
    MathFunctions.sqrt, MathFunctions.abs]
  change (scale, if Real.sqrt (Tensor.sum (Tensor.map
    (fun value => (value / scale) * (value / scale)) x)) > ε / scale then
    Real.sqrt (Tensor.sum (Tensor.map (fun value => (value / scale) * (value / scale)) x))
    else ε / scale) = (scale, max ‖x‖ ε / scale)
  rw [hsum, Real.sqrt_div' _ (sq_nonneg scale), Real.sqrt_sq hs,
    Tensor.sqrt_sum_square_eq_norm]
  rcases hs.eq_or_lt with h0 | hpos
  · simp only [← h0, div_zero, gt_iff_lt, lt_irrefl, ↓reduceIte]
  · simp only [gt_iff_lt, div_lt_div_iff_of_pos_right hpos]
    split_ifs with h
    · rw [max_eq_left h.le]
    · rw [max_eq_right (le_of_not_gt h)]

/-- Over the reals and for every epsilon, dividing `y` entrywise by the scale of
`CosineLoss.normFactors y ε` and then by its clipped scaled norm gives `(max ‖y‖ ε)⁻¹ • y`.
When the scale is zero, every entry of `y` is zero, so both sides are zero. -/
theorem CosineLoss.normalized_real {s : Shape} (y : Tensor ℝ s) (ε : ℝ) :
    Tensor.map (fun v => v / max (Tensor.maxAbs y) |ε| /
      (max ‖y‖ ε / max (Tensor.maxAbs y) |ε|)) y = (max ‖y‖ ε)⁻¹ • y := by
  refine Tensor.Rep.ext fun c => ?_
  simp only [Tensor.map_apply, Tensor.Rep.smul_apply, smul_eq_mul]
  rcases ((abs_nonneg ε).trans (le_max_right (Tensor.maxAbs y) |ε|)).eq_or_lt with h0 | hpos
  · have hc : y c = 0 := abs_nonpos_iff.mp
      (((Tensor.abs_le_maxAbs y c).trans (le_max_left _ _)).trans h0.ge)
    simp only [hc, zero_div, mul_zero]
  · rw [div_div_div_cancel_right₀ hpos.ne', div_eq_inv_mul]

/-- For every epsilon, the stored cosine gradient is
`c • x - (max ‖x‖ ε * max ‖t‖ ε)⁻¹ • t`, where `c = ⟪x, t⟫ / (max ‖x‖ ε ^ 3 * max ‖t‖ ε)` for
`ε < ‖x‖` and `c = 0` otherwise. A positive scale of `CosineLoss.normFactors x ε` cancels, and
a zero scale forces `x = 0` and `ε = 0`, where both sides are zero. -/
theorem CosineLoss.fiberLossDeriv_real {s : Shape} (x t : Tensor ℝ s) (ε : ℝ) :
    CosineLoss.fiberLossDeriv x t ε =
      (if ε < ‖x‖ then ⟪x, t⟫ / (max ‖x‖ ε ^ 3 * max ‖t‖ ε) else 0) • x -
        (max ‖x‖ ε * max ‖t‖ ε)⁻¹ • t := by
  have hS : (0 : ℝ) ≤ max (Tensor.maxAbs x) |ε| := (abs_nonneg ε).trans (le_max_right _ _)
  simp only [CosineLoss.fiberLossDeriv, CosineLoss.normFactors_real x,
    CosineLoss.normFactors_real t, CosineLoss.normalized_real, Tensor.sum_mul_eq_inner,
    real_inner_smul_left, real_inner_smul_right]
  refine Tensor.Rep.ext fun c => ?_
  simp only [Tensor.zipWith_apply, Tensor.Rep.hSub_apply, Tensor.Rep.smul_apply, smul_eq_mul,
    gt_iff_lt]
  rcases hS.eq_or_lt with h0 | hpos
  · have hx0 : ∀ c, x c = 0 := fun c => abs_nonpos_iff.mp
      (((Tensor.abs_le_maxAbs x c).trans (le_max_left _ _)).trans h0.ge)
    have hε : ε = 0 := abs_nonpos_iff.mp ((le_max_right _ _).trans h0.ge)
    have hxn : ‖x‖ = 0 := by
      rw [← Tensor.sqrt_sum_square_eq_norm, Tensor.sum_eq_coord_sum]
      simp only [Tensor.square, Tensor.map_apply, hx0, mul_zero,
        Finset.sum_const_zero, Real.sqrt_zero]
    rw [← h0]
    simp only [div_zero, hx0, hxn, hε, max_self, mul_zero, zero_mul, inv_zero, sub_zero]
  · have hactive : ε / max (Tensor.maxAbs x) |ε| < max ‖x‖ ε / max (Tensor.maxAbs x) |ε| ↔
        ε < ‖x‖ := by
      rw [div_lt_div_iff_of_pos_right hpos, lt_max_iff]
      simp only [lt_irrefl, or_false]
    simp only [hactive]
    rw [div_div, div_mul_cancel₀ _ hpos.ne']
    split_ifs <;> ring

/-- For every epsilon, the cosine loss over every axis, reshaped to a scalar, is the scalar
`1 - ⟪x, t⟫ / (max ‖x‖ ε * max ‖t‖ ε)`. -/
theorem Tensor.cosineLoss_reshape_real {s : Shape} (x t : Tensor ℝ s) (ε : ℝ)
    (h : (s.reduce (List.finRange s.rank) false).size = Shape.scalar.size) :
    Tensor.reshape (Tensor.cosineLoss x t (List.finRange s.rank) false ε) Shape.scalar h =
      Tensor.scalar (1 - ⟪x, t⟫ / (max ‖x‖ ε * max ‖t‖ ε)) :=
  Rep.ext fun c => by
    cases c
    rw [reshape_apply, cosineLoss_eq_one_sub_dot x t false ε, scalar_apply]
    simp only [CosineLoss.normFactors_real, CosineLoss.normalized_real, sum_mul_eq_inner,
      real_inner_smul_left, real_inner_smul_right]
    rw [div_eq_mul_inv, mul_inv]
    ring

/-- Where `0 < ε`, `f x ≠ 0` or `t = 0`, and where `‖f x‖ = ε` forces `⟪f x, t⟫ = 0`, the cosine
loss of `f` against a target `t` over every axis has the derivative `⟪G, ·⟫` after `Df` and the
pullback that applies `bf` to `G` scaled by the cotangent, with
`G = CosineLoss.fiberLossDeriv (f x) t ε`. -/
@[fun_prop]
theorem HasPullbackAt.cosineLoss {s : Shape} {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] {f : E → Tensor ℝ s} {Df : E →L[ℝ] Tensor ℝ s}
    {bf : Tensor ℝ s → E} {x : E} (t : Tensor ℝ s) {ε : ℝ}
    (h : (s.reduce (List.finRange s.rank) false).size = Shape.scalar.size)
    (hε : 0 < ε ∨ f x ≠ 0 ∨ t = 0) (hx : ‖f x‖ = ε → ⟪f x, t⟫ = 0)
    (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt
      (fun y => Tensor.reshape (Tensor.cosineLoss (f y) t (List.finRange s.rank) false ε)
        Shape.scalar h)
      ((Tensor.scalarCLM.comp (innerSL ℝ (CosineLoss.fiberLossDeriv (f x) t ε))).comp Df)
      (fun δ => bf (Tensor.scale (CosineLoss.fiberLossDeriv (f x) t ε) (Tensor.item δ))) x := by
  have hF : (fun y => Tensor.reshape (Tensor.cosineLoss (f y) t (List.finRange s.rank) false ε)
      Shape.scalar h) =
      fun y => Tensor.scalar (1 - ⟪f y, t⟫ / (max ‖f y‖ ε * max ‖t‖ ε)) :=
    funext fun y => Tensor.cosineLoss_reshape_real (f y) t ε h
  have hb : (fun δ : Tensor ℝ .scalar =>
      bf (Tensor.scale (CosineLoss.fiberLossDeriv (f x) t ε) (Tensor.item δ))) =
      fun δ => bf (Tensor.item δ • CosineLoss.fiberLossDeriv (f x) t ε) :=
    funext fun δ => congrArg bf (Tensor.Rep.ext fun c => by
      simp only [Tensor.scale, Tensor.map_apply, Tensor.Rep.smul_apply,
        smul_eq_mul]
      ring)
  have hG : HasPullbackAt (fun z : Tensor ℝ s => 1 - ⟪z, t⟫ / (max ‖z‖ ε * max ‖t‖ ε))
      (innerSL ℝ (CosineLoss.fiberLossDeriv (f x) t ε))
      (fun r => r • CosineLoss.fiberLossDeriv (f x) t ε) (f x) := by
    rw [CosineLoss.fiberLossDeriv_real (f x) t ε]
    exact ⟨hasFDerivAt_cosine_clipped t (f x) hε hx, fun u r => by
      rw [innerSL_apply_apply, Real.inner_apply, real_inner_smul_right, real_inner_comm,
        mul_comm]⟩
  rw [hF, hb, ContinuousLinearMap.comp_assoc]
  exact (hG.comp hf).scalar

/-! ## Log-softmax along an axis -/

namespace Op

variable {s : Shape}

/-- Log-softmax along `dim` has at `x` the fiberwise derivative of its in-tree certificate, with
the stored log-softmax pullback, read at the forward output, as its adjoint. -/
theorem hasPullbackAt_logSoftmax (dim : Fin s.rank) (x : Tensor ℝ s) :
    HasPullbackAt (fun y => Tensor.logSoftmax y dim)
      (fiberwiseCLM (Coord.axisEquiv dim (shape := s)) fun r =>
        logSoftmaxDerivCLM (fiberCLM (Coord.axisEquiv dim (shape := s)) r x))
      (fun δ => Tensor.logSoftmaxBackward (Tensor.logSoftmax x dim) δ dim) x := by
  have h := fun δ => logSoftmax_backward_eq_adjoint dim x δ
  simp only [Op.rule?_logSoftmax, Option.get_some] at h
  exact HasPullbackAt.of_adjoint (h 0).1 fun δ => by rw [(h δ).2, (h δ).1.fderiv]

/-- The stored log-softmax pullback, read at the forward output, commutes with multiplying the
cotangent by a real `k`. -/
@[op_deriv_simps]
theorem logSoftmaxBackward_smul (x : Tensor ℝ s) (dim : Fin s.rank) (k : ℝ)
    (δ : Tensor ℝ s) :
    Tensor.logSoftmaxBackward (Tensor.logSoftmax x dim) (k • δ) dim =
      k • Tensor.logSoftmaxBackward (Tensor.logSoftmax x dim) δ dim := by
  rw [(hasPullbackAt_logSoftmax dim x).eq_adjoint, (hasPullbackAt_logSoftmax dim x).eq_adjoint,
    map_smul]

/-- The stored log-softmax pullback of a cotangent multiplied entrywise by the constant tensor
`Tensor.full s k` is `k` times the pullback of the cotangent. -/
theorem logSoftmaxBackward_full_mul (x : Tensor ℝ s) (dim : Fin s.rank) (k : ℝ)
    (δ : Tensor ℝ s) :
    Tensor.logSoftmaxBackward (Tensor.logSoftmax x dim) (Tensor.full s k * δ) dim =
      k • Tensor.logSoftmaxBackward (Tensor.logSoftmax x dim) δ dim := by
  have h : Tensor.full s k * δ = k • δ := Tensor.Rep.ext fun c => by
    simp only [Tensor.mul_apply, Tensor.full_apply, Tensor.Rep.smul_apply, smul_eq_mul]
  rw [h, logSoftmaxBackward_smul]

/-- The fiber at `r` of the softmax along `dim` is the analytic softmax of the fiber at `r`. -/
theorem fiberCLM_softmax (t : Tensor ℝ s) (dim : Fin s.rank) (r : Shape.Coord (s.eraseIdx dim)) :
    fiberCLM (Coord.axisEquiv dim (shape := s)) r (Tensor.softmax t dim) =
      Vec.softmax (fiberCLM (Coord.axisEquiv dim (shape := s)) r t) := by
  ext j
  simp only [fiberCLM_apply, softmax_apply_vec_softmax, Equiv.apply_symm_apply]

/-- The stored log-softmax pullback at `x`, applied to the softmax of `t` along the same axis, is
the softmax of `t` minus the softmax of `x`. -/
theorem logSoftmaxBackward_softmax (x t : Tensor ℝ s) (dim : Fin s.rank) :
    Tensor.logSoftmaxBackward (Tensor.logSoftmax x dim) (Tensor.softmax t dim) dim =
      Tensor.softmax t dim - Tensor.softmax x dim :=
  Tensor.Rep.ext fun c => by
    rw [logSoftmaxBackward_apply_logSoftmaxVJP, fiberCLM_softmax, Vec.logSoftmaxVJP_softmax,
      Tensor.Rep.hSub_apply, softmax_apply_vec_softmax, softmax_apply_vec_softmax]

end Op

/-- Log-softmax along `dim` after a map `f` with a pullback: the derivative is the fiberwise
log-softmax derivative at `f x` after the derivative of `f`, and the pullback applies the stored
log-softmax pullback at the forward output first. -/
@[fun_prop]
theorem HasPullbackAt.logSoftmax {s : Shape} {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] {f : E → Tensor ℝ s} {Df : E →L[ℝ] Tensor ℝ s}
    {bf : Tensor ℝ s → E} {x : E} (dim : Fin s.rank) (hf : HasPullbackAt f Df bf x) :
    HasPullbackAt (fun y => Tensor.logSoftmax (f y) dim)
      ((Op.fiberwiseCLM (Coord.axisEquiv dim (shape := s)) fun r =>
        logSoftmaxDerivCLM (Op.fiberCLM (Coord.axisEquiv dim (shape := s)) r (f x))).comp Df)
      (fun δ => bf (Tensor.logSoftmaxBackward (Tensor.logSoftmax (f x) dim) δ dim)) x :=
  (Op.hasPullbackAt_logSoftmax dim (f x)).comp hf

namespace Tensor

variable {s : Shape}

/-- Over the reals, the stored logits gradient of the cross-entropy along `dim` is minus the
log-softmax pullback of the target, divided by the number of fibers when `mean` holds. -/
@[op_deriv_simps] theorem crossEntropyBackward_logits_real (x t : Tensor ℝ s) (dim : Fin s.rank)
    (mean : Bool) :
    crossEntropyBackward .logits x t dim mean =
      (if mean then -(axisMeanDenom s dim : ℝ)⁻¹ else -1) •
        logSoftmaxBackward (logSoftmax x dim) t dim := by
  have hneg : Tensor.map (fun v => -v) t = (-1 : ℝ) • t :=
    Rep.ext fun c => by simp only [Tensor.map_apply, Rep.smul_apply, smul_eq_mul]; ring
  cases mean
  · simp only [crossEntropyBackward, CrossEntropy.logProbs_real, hneg, Op.logSoftmaxBackward_smul,
      Bool.false_eq_true, ↓reduceIte]
  · simp only [crossEntropyBackward, CrossEntropy.logProbs_real, hneg, Op.logSoftmaxBackward_smul,
      ↓reduceIte]
    refine Rep.ext fun c => ?_
    simp only [Tensor.map_apply, Context.divNat_real, Rep.smul_apply, smul_eq_mul]
    ring

/-- Over the reals, the stored target gradient of the cross-entropy at seed `k` is `-k` times the
log-softmax of the logits, divided by the number of fibers when `mean` holds. -/
@[op_deriv_simps] theorem crossEntropyTargetBackward_real (x : Tensor ℝ s) (dim : Fin s.rank)
    (k : ℝ) (mean : Bool) :
    crossEntropyTargetBackward x dim k mean =
      ((if mean then -(axisMeanDenom s dim : ℝ)⁻¹ else -1) * k) • logSoftmax x dim := by
  refine Rep.ext fun c => ?_
  cases mean <;>
    simp only [crossEntropyTargetBackward, CrossEntropy.terms_real, Bool.false_eq_true,
      ↓reduceIte, Tensor.map_apply, Context.divNat_real, Rep.smul_apply, smul_eq_mul,
      Tensor.mul_apply, Tensor.full_apply, Tensor.neg] <;> ring

/-- Over the reals, the stored student gradient of the distillation loss along `dim` is the
student softmax minus the teacher softmax, divided by the number of fibers, in both of its
numerical branches. -/
@[op_deriv_simps] theorem distillLossBackward_real (dim : Fin s.rank) (x t : Tensor ℝ s) :
    distillLossBackward dim x t =
      (axisMeanDenom s dim : ℝ)⁻¹ • (Tensor.softmax x dim - Tensor.softmax t dim) := by
  refine Rep.ext fun c => ?_
  obtain ⟨⟨j, r⟩, rfl⟩ := (Coord.axisEquiv dim (shape := s)).symm.surjective c
  simp only [distillLossBackward, Tensor.map_apply, Rep.hSub_apply, Tensor.add_apply,
    Tensor.mul_apply, zipWith_apply, broadcast_eraseIdx_apply,
    castShape_reduce_sum_singleton_apply, Op.softmax_apply_vec_softmax, Equiv.apply_symm_apply,
    Context.divNat_real, Rep.smul_apply, smul_eq_mul, LawfulContext.abs_eq,
    Context.log1p_eq_log_one_add, Context.expm1_eq_exp_sub_one, foldl_add_ofFn]
  rw [div_eq_inv_mul]
  congr 1
  exact distill_fiber_eq_sub (fun i => x ((Coord.axisEquiv dim (shape := s)).symm (i, r)))
    (fun i => t ((Coord.axisEquiv dim (shape := s)).symm (i, r)))
    (fun i => Vec.softmax (Op.fiberCLM (Coord.axisEquiv dim (shape := s)) r t) i)
    (fun i => Vec.softmax (Op.fiberCLM (Coord.axisEquiv dim (shape := s)) r x) i) _ _
    (fun i => by simp only [Vec.softmax_apply_eq_exp_div, Op.fiberCLM_apply])
    (fun i => by simp only [Vec.softmax_apply_eq_exp_div, Op.fiberCLM_apply]) rfl rfl j

end Tensor

namespace Op

variable {s : Shape}

/-! ## Smooth losses -/

/-- At every pair of inputs, the mean squared error is differentiable jointly in the prediction
and the target, and the stored backward, `(2 / n) (a - b)` and its negation scaled by the
cotangent, is the adjoint of its derivative. -/
@[op_cert TorchLean.Op.mseLoss derivative 0 1]
theorem mseLoss_backward_eq_adjoint (a b : Tensor ℝ s) (δ : Tensor ℝ .scalar) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
        (((Op.mseLoss (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
      (WithLp.toLp 2 (a, b)) ∧
    WithLp.toLp 2
        ((((Op.mseLoss (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).1,
        (((Op.mseLoss (s := s)).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).2.1) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
        (((Op.mseLoss (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
        (WithLp.toLp 2 (a, b))).adjoint δ := by
  op_deriv [Tensor.scale, Tensor.neg]

/-- At every prediction, the Poisson loss is differentiable in the prediction, and the stored
backward, `(exp x - t) / n` scaled by the cotangent, is the adjoint of its derivative. -/
@[op_cert TorchLean.Op.poissonLoss derivative 0]
theorem poissonLoss_backward_eq_adjoint (x t : Tensor ℝ s) (δ : Tensor ℝ .scalar) :
    DifferentiableAt ℝ (fun y =>
        (((Op.poissonLoss (s := s)).rule? (α := ℝ) |>.get rfl).forward (y, t, ())).1) x ∧
    (((Op.poissonLoss (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, t, ()) (δ, ())).1 =
      (fderiv ℝ (fun y =>
        (((Op.poissonLoss (s := s)).rule? (α := ℝ) |>.get rfl).forward (y, t, ())).1)
        x).adjoint δ := by
  op_deriv [Tensor.scale]

/-- At every prediction, the log-cosh loss is differentiable in the prediction, and the stored
backward, `tanh (x - t) / n` scaled by the cotangent, is the adjoint of its derivative. -/
@[op_cert TorchLean.Op.logCoshLoss derivative 0]
theorem logCoshLoss_backward_eq_adjoint (x t : Tensor ℝ s) (δ : Tensor ℝ .scalar) :
    DifferentiableAt ℝ (fun y =>
        (((Op.logCoshLoss (s := s)).rule? (α := ℝ) |>.get rfl).forward (y, t, ())).1) x ∧
    (((Op.logCoshLoss (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, t, ()) (δ, ())).1 =
      (fderiv ℝ (fun y =>
        (((Op.logCoshLoss (s := s)).rule? (α := ℝ) |>.get rfl).forward (y, t, ())).1)
        x).adjoint δ := by
  have h := fun c => hasDerivAt_log_cosh ((x - t) c)
  op_deriv [Tensor.scale]

/-- At every pair of logits and targets, every axis and both values of `mean`, the logits
cross-entropy is differentiable jointly in the logits and the target, and the stored backward of
both inputs is the adjoint of its derivative. -/
@[op_cert TorchLean.Op.crossEntropy derivative 0 1]
theorem crossEntropy_backward_eq_adjoint (dim : Fin s.rank) (mean : Bool) (a b : Tensor ℝ s)
    (δ : Tensor ℝ .scalar) :
    DifferentiableAt ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
        (((Op.crossEntropy dim mean).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
      (WithLp.toLp 2 (a, b)) ∧
    WithLp.toLp 2
        ((((Op.crossEntropy dim mean).rule? (α := ℝ) |>.get rfl).backward (a, b, ()) (δ, ())).1,
        (((Op.crossEntropy dim mean).rule? (α := ℝ) |>.get rfl).backward (a, b, ())
          (δ, ())).2.1) =
      (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
        (((Op.crossEntropy dim mean).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
        (WithLp.toLp 2 (a, b))).adjoint δ := by
  op_deriv [Tensor.scale, logSoftmaxBackward_full_mul]

/-- At every pair of student and teacher logits and every axis, the distillation loss is
differentiable in the student logits, and the stored backward, the student softmax minus the
teacher softmax over the number of fibers, scaled by the cotangent, is the adjoint of its
derivative. -/
@[op_cert TorchLean.Op.distill derivative 0]
theorem distill_backward_eq_adjoint (dim : Fin s.rank) (x t : Tensor ℝ s)
    (δ : Tensor ℝ .scalar) :
    DifferentiableAt ℝ (fun y =>
        (((Op.distill dim).rule? (α := ℝ) |>.get rfl).forward (y, t, ())).1) x ∧
    (((Op.distill dim).rule? (α := ℝ) |>.get rfl).backward (x, t, ()) (δ, ())).1 =
      (fderiv ℝ (fun y =>
        (((Op.distill dim).rule? (α := ℝ) |>.get rfl).forward (y, t, ())).1) x).adjoint δ := by
  op_deriv [Tensor.scale, logSoftmaxBackward_full_mul, logSoftmaxBackward_softmax]

/-! ## Mean absolute error -/

/-- Where no entry of the prediction equals the target, the mean absolute error is differentiable
in the prediction, and the stored backward, the sign of `x - t` over `n` scaled by the cotangent,
is the adjoint of its derivative. -/
@[op_cert TorchLean.Op.maeLoss derivative 0]
theorem maeLoss_backward_eq_adjoint (x t : Tensor ℝ s) (δ : Tensor ℝ .scalar)
    (hx : ∀ c, x c ≠ t c) :
    DifferentiableAt ℝ (fun y =>
        (((Op.maeLoss (s := s)).rule? (α := ℝ) |>.get rfl).forward (y, t, ())).1) x ∧
    (((Op.maeLoss (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, t, ()) (δ, ())).1 =
      (fderiv ℝ (fun y =>
        (((Op.maeLoss (s := s)).rule? (α := ℝ) |>.get rfl).forward (y, t, ())).1)
        x).adjoint δ := by
  have h : ∀ c, HasDerivAt (fun d : ℝ => |d|) (SignType.sign ((x - t) c) : ℝ) ((x - t) c) :=
    fun c => hasDerivAt_abs (by rw [Tensor.Rep.hSub_apply]; exact sub_ne_zero.mpr (hx c))
  op_deriv [Tensor.scale]

/-- At every input, the stored backward of the mean absolute error is the cotangent over `n`
times the sign of `x - t`, which is zero at the kinks, the entries where `x` equals `t`. -/
@[op_cert TorchLean.Op.maeLoss convention]
theorem maeLoss_backward_eq (x t : Tensor ℝ s) (δ : Tensor ℝ .scalar) :
    (((Op.maeLoss (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, t, ()) (δ, ())).1 =
      Tensor.map (fun d => Tensor.item δ / Tensor.meanDenominator s * (SignType.sign d : ℝ))
        (x - t) :=
  Tensor.Rep.ext fun c => by
    simp only [Op.rule?_maeLoss, Option.get_some, Tensor.l1LossBackward_real, Tensor.scale,
      Tensor.map_apply]
    ring

/-! ## Hinge loss -/

/-- Where no entry of `x * t` equals one, the hinge loss is differentiable in the prediction, and
the stored backward, `-t / n` on the entries with `x * t < 1` scaled by the cotangent, is the
adjoint of its derivative. -/
@[op_cert TorchLean.Op.hingeLoss derivative 0]
theorem hingeLoss_backward_eq_adjoint (x t : Tensor ℝ s) (δ : Tensor ℝ .scalar)
    (hx : ∀ c, x c * t c ≠ 1) :
    DifferentiableAt ℝ (fun y =>
        (((Op.hingeLoss (s := s)).rule? (α := ℝ) |>.get rfl).forward (y, t, ())).1) x ∧
    (((Op.hingeLoss (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, t, ()) (δ, ())).1 =
      (fderiv ℝ (fun y =>
        (((Op.hingeLoss (s := s)).rule? (α := ℝ) |>.get rfl).forward (y, t, ())).1)
        x).adjoint δ := by
  have h : ∀ c, HasDerivAt (fun m : ℝ => Max.max (1 - m) 0) (-(if (x * t) c < 1 then 1 else 0))
      ((x * t) c) :=
    fun c => hasDerivAt_max_one_sub (by rw [Tensor.mul_apply]; exact hx c)
  op_deriv [Tensor.scale]

/-- At every input, the stored backward of the hinge loss is `-t` times the cotangent over `n` on
the entries with `x * t < 1` and zero on the others, which is zero at the kinks `x * t = 1`. -/
@[op_cert TorchLean.Op.hingeLoss convention]
theorem hingeLoss_backward_eq (x t : Tensor ℝ s) (δ : Tensor ℝ .scalar) :
    (((Op.hingeLoss (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, t, ()) (δ, ())).1 =
      Tensor.zipWith (fun p q => if p * q < 1 then -(Tensor.item δ / Tensor.meanDenominator s) * q
        else 0) x t :=
  Tensor.Rep.ext fun c => by
    simp only [Op.rule?_hingeLoss, Option.get_some, Tensor.hingeLossBackward_real, Tensor.scale,
      Tensor.map_apply, Tensor.mul_apply, Tensor.zipWith_apply]
    split_ifs <;> ring

/-! ## Huber loss -/

/-- Where the threshold is nonnegative or no entry of the prediction equals the target, the Huber
loss is differentiable in the prediction for both values of `mean`, and the stored backward, the
Huber slope of `x - t` scaled by the cotangent and divided by `n` when `mean` holds, is the
adjoint of its derivative. -/
@[op_cert TorchLean.Op.huberLoss derivative 0]
theorem huberLoss_backward_eq_adjoint (mean : Bool) (x t : Tensor ℝ s) (d δ : Tensor ℝ .scalar)
    (hx : 0 ≤ Tensor.item d ∨ ∀ c, x c ≠ t c) :
    DifferentiableAt ℝ (fun y =>
        (((Op.huberLoss (s := s) mean).rule? (α := ℝ) |>.get rfl).forward (y, t, d, ())).1) x ∧
    (((Op.huberLoss (s := s) mean).rule? (α := ℝ) |>.get rfl).backward (x, t, d, ())
        (δ, ())).1 =
      (fderiv ℝ (fun y =>
        (((Op.huberLoss (s := s) mean).rule? (α := ℝ) |>.get rfl).forward (y, t, d, ())).1)
        x).adjoint δ := by
  have h : ∀ c, HasDerivAt (fun e : ℝ => if |e| < Tensor.item d then |e| / 2 * |e|
      else Tensor.item d * (|e| - Tensor.item d / 2))
      (if |(x - t) c| < Tensor.item d then (x - t) c
        else Tensor.item d * (SignType.sign ((x - t) c) : ℝ)) ((x - t) c) :=
    fun c => hasDerivAt_huber (hx.imp_right fun h => by
      rw [Tensor.Rep.hSub_apply]; exact sub_ne_zero.mpr (h c))
  op_deriv [Tensor.scale]

/-- At every input, the stored backward of the Huber loss is the cotangent times the Huber slope of
`x - t`, divided by `n` when `mean` holds, which is zero at the kinks, the entries where `x`
equals `t` under a negative threshold. -/
@[op_cert TorchLean.Op.huberLoss convention]
theorem huberLoss_backward_eq (mean : Bool) (x t : Tensor ℝ s) (d δ : Tensor ℝ .scalar) :
    (((Op.huberLoss (s := s) mean).rule? (α := ℝ) |>.get rfl).backward (x, t, d, ())
        (δ, ())).1 =
      (Tensor.item δ * if mean then (Tensor.meanDenominator s : ℝ)⁻¹ else 1) •
        Tensor.map (fun e => if |e| < Tensor.item d then e
          else Tensor.item d * (SignType.sign e : ℝ)) (x - t) :=
  Tensor.Rep.ext fun c => by
    simp only [Op.rule?_huberLoss, Option.get_some, Tensor.huberLossBackward_real, Tensor.scale,
      Tensor.map_apply, Tensor.Rep.smul_apply, smul_eq_mul]
    ring

/-! ## Clamped probability losses -/

/-- With `ε = item d`, where `1 / 2 ≤ ε` or every entry has a zero target or a probability off the
clamp kinks `ε`, `1 - ε` and, for `ε ≤ 0`, off zero, the probability cross-entropy is
differentiable in the probabilities, and the stored backward is the adjoint of its derivative. -/
@[op_cert TorchLean.Op.crossEntropyProbabilities derivative 0]
theorem crossEntropyProbabilities_backward_eq_adjoint (dim : Fin s.rank) (x t : Tensor ℝ s)
    (d δ : Tensor ℝ .scalar)
    (hx : 1 / 2 ≤ Tensor.item d ∨ ∀ c, t c = 0 ∨
      (x c ≠ Tensor.item d ∧ x c ≠ 1 - Tensor.item d ∧ (Tensor.item d ≤ 0 → x c ≠ 0))) :
    DifferentiableAt ℝ (fun y =>
        (((Op.crossEntropyProbabilities dim).rule? (α := ℝ) |>.get rfl).forward
          (y, t, d, ())).1) x ∧
    (((Op.crossEntropyProbabilities dim).rule? (α := ℝ) |>.get rfl).backward (x, t, d, ())
        (δ, ())).1 =
      (fderiv ℝ (fun y =>
        (((Op.crossEntropyProbabilities dim).rule? (α := ℝ) |>.get rfl).forward
          (y, t, d, ())).1) x).adjoint δ := by
  have h : ∀ c, HasDerivAt
      (fun a => t c * Real.log (Min.min (Max.max a (Tensor.item d)) (1 - Tensor.item d)))
      (if Tensor.item d < x c then (if x c < 1 - Tensor.item d then t c / x c else 0) else 0)
      (x c) :=
    fun c => hasDerivAt_mul_log_clamp (hx.imp_right fun h => h c)
  op_deriv [Tensor.scale]

/-- At every input, with `ε = item d`, the stored backward of the probability cross-entropy is
`-t / x` times the cotangent over the number of fibers strictly inside `(ε, 1 - ε)` and zero
elsewhere, which is zero at the clamp kinks `x = ε` and `x = 1 - ε`. -/
@[op_cert TorchLean.Op.crossEntropyProbabilities convention]
theorem crossEntropyProbabilities_backward_eq (dim : Fin s.rank) (x t : Tensor ℝ s)
    (d δ : Tensor ℝ .scalar) :
    (((Op.crossEntropyProbabilities dim).rule? (α := ℝ) |>.get rfl).backward (x, t, d, ())
        (δ, ())).1 =
      Tensor.zipWith (fun a b => if Tensor.item d < a then (if a < 1 - Tensor.item d then
        -(Tensor.item δ / Tensor.axisMeanDenom s dim) * (b / a) else 0) else 0) x t :=
  Tensor.Rep.ext fun c => by
    simp only [Op.rule?_crossEntropyProbabilities, Option.get_some,
      Tensor.crossEntropyBackward_probabilities_real, ↓reduceIte, Tensor.scale,
      Tensor.map_apply, Tensor.Rep.smul_apply, smul_eq_mul, Tensor.zipWith_apply]
    split_ifs <;> ring

/-- With `ε = item d`, where `1 / 2 ≤ ε` or every entry has `t = x` when `x` is a clamp kink `ε`
or `1 - ε` and has `x` off `0` and `1` when `ε ≤ 0`, the binary cross-entropy is differentiable
in the prediction, and the stored backward is the adjoint of its derivative. The set is exact: at
an excluded entry the forward has no derivative, or `ε < 0` and `x = t` is `0` or `1`, where the
entry term has slope `1` or `-1` but the stored slope `(x - t) / (x (1 - x))` is `0 / 0 = 0`. -/
@[op_cert TorchLean.Op.binaryCrossEntropy derivative 0]
theorem binaryCrossEntropy_backward_eq_adjoint (x t : Tensor ℝ s) (d δ : Tensor ℝ .scalar)
    (hx : 1 / 2 ≤ Tensor.item d ∨ ∀ c, (x c = Tensor.item d → t c = Tensor.item d) ∧
      (x c = 1 - Tensor.item d → t c = 1 - Tensor.item d) ∧
      (Tensor.item d ≤ 0 → x c ≠ 0 ∧ x c ≠ 1)) :
    DifferentiableAt ℝ (fun y =>
        (((Op.binaryCrossEntropy (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, t, d, ())).1) x ∧
    (((Op.binaryCrossEntropy (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, t, d, ())
        (δ, ())).1 =
      (fderiv ℝ (fun y =>
        (((Op.binaryCrossEntropy (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, t, d, ())).1) x).adjoint δ := by
  have h : ∀ c, HasDerivAt
      (fun a => -(t c * Real.log (Min.min (Max.max a (Tensor.item d)) (1 - Tensor.item d)) +
        (1 - t c) * Real.log (1 - Min.min (Max.max a (Tensor.item d)) (1 - Tensor.item d))))
      (if Tensor.item d < x c then
        (if x c < 1 - Tensor.item d then (x c - t c) / (x c * (1 - x c)) else 0) else 0)
      (x c) :=
    fun c => hasDerivAt_bce_clamp (hx.imp_right fun h => h c)
  op_deriv [Tensor.scale]

/-- At every input, with `ε = item d`, the stored backward of the binary cross-entropy is
`(x - t) / (x (1 - x))` times the cotangent over `n` strictly inside `(ε, 1 - ε)` and zero
elsewhere, which is zero at the clamp kinks `x = ε` and `x = 1 - ε`. -/
@[op_cert TorchLean.Op.binaryCrossEntropy convention]
theorem binaryCrossEntropy_backward_eq (x t : Tensor ℝ s) (d δ : Tensor ℝ .scalar) :
    (((Op.binaryCrossEntropy (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, t, d, ())
        (δ, ())).1 =
      Tensor.zipWith (fun a b => if Tensor.item d < a then (if a < 1 - Tensor.item d then
        Tensor.item δ / Tensor.meanDenominator s * ((a - b) / (a * (1 - a))) else 0) else 0) x t :=
  Tensor.Rep.ext fun c => by
    simp only [Op.rule?_binaryCrossEntropy, Option.get_some, Tensor.binaryCrossEntropyBackward_real,
      Tensor.scale, Tensor.map_apply, Tensor.Rep.smul_apply, smul_eq_mul,
      Tensor.zipWith_apply]
    split_ifs <;> ring

/-! ## Cosine loss -/

/-- With `ε = item d`, where `0 < ε`, `x ≠ 0` or `t = 0`, and where `‖x‖ = ε` forces
`⟪x, t⟫ = 0`, the cosine loss is differentiable in the prediction, and the stored backward is the
adjoint of its derivative. The set is exact: for `ε ≤ 0` and `t ≠ 0` the forward is not continuous
at `x = 0`, and at `‖x‖ = ε > 0` with `⟪x, t⟫ ≠ 0` its one-sided radial slopes differ. -/
@[op_cert TorchLean.Op.cosineLoss derivative 0]
theorem cosineLoss_backward_eq_adjoint (x t : Tensor ℝ s) (d δ : Tensor ℝ .scalar)
    (hd : 0 < Tensor.item d ∨ x ≠ 0 ∨ t = 0) (hx : ‖x‖ = Tensor.item d → ⟪x, t⟫ = 0) :
    DifferentiableAt ℝ (fun y =>
        (((Op.cosineLoss (s := s)).rule? (α := ℝ) |>.get rfl).forward (y, t, d, ())).1) x ∧
    (((Op.cosineLoss (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, t, d, ()) (δ, ())).1 =
      (fderiv ℝ (fun y =>
        (((Op.cosineLoss (s := s)).rule? (α := ℝ) |>.get rfl).forward (y, t, d, ())).1)
        x).adjoint δ := by
  op_deriv

/-- At every input, with `ε = item d`, the stored backward of the cosine loss is
`item δ • (c • x - (X * T)⁻¹ • t)` with `X = max ‖x‖ ε` and `T = max ‖t‖ ε`, where
`c = ⟪x, t⟫ / (X ^ 3 * T)` for `ε < ‖x‖` and `c = 0` otherwise. For a positive `ε` and a point
of the kink sphere `‖x‖ = ε`, it is `-(item δ / (ε * T)) • t`, which is `item δ` times the
gradient of the clipped inner branch. -/
@[op_cert TorchLean.Op.cosineLoss convention]
theorem cosineLoss_backward_eq (x t : Tensor ℝ s) (d δ : Tensor ℝ .scalar) :
    (((Op.cosineLoss (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, t, d, ()) (δ, ())).1 =
      Tensor.item δ • ((if Tensor.item d < ‖x‖ then ⟪x, t⟫ / (Max.max ‖x‖ (Tensor.item d) ^ 3 *
        Max.max ‖t‖ (Tensor.item d)) else 0) • x -
        (Max.max ‖x‖ (Tensor.item d) * Max.max ‖t‖ (Tensor.item d))⁻¹ • t) := by
  simp only [Op.rule?_cosineLoss, Option.get_some, CosineLoss.fiberLossDeriv_real]
  refine Tensor.Rep.ext fun c => ?_
  simp only [Tensor.scale, Tensor.map_apply, Tensor.Rep.smul_apply, smul_eq_mul]
  ring

end Op

end TorchLean
