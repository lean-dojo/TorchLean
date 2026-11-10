/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Op.Cert
public import TorchLean.NN.Gradients.Activation
public import TorchLean.NN.FDeriv.TensorCoordinates
public import TorchLean.Tensor.Elementwise.Real
public import TorchLean.Tensor.Euclidean
public import Mathlib.Analysis.Calculus.Deriv.Abs
public import Mathlib.Analysis.InnerProductSpace.ProdL2
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.InverseDeriv

/-!
# Derivative certificates of the pointwise operators

A pointwise operator applies a real function at every entry, `f c` at the entry `c`. Where `f c`
has derivative `d c` at each entry `x c`, the Fréchet derivative of the forward map at `x` is the
slope map `slopeCLM d`, which multiplies each entry by its slope. The slope map is self-adjoint for
the Euclidean inner product of the tensors, so a pullback that multiplies the cotangent by the
same slopes is the adjoint of the derivative (`pointwise_backward_eq_adjoint`). Each certificate
states both facts: the slope map is the derivative at `x`, and the pullback is its adjoint. A
binary operator that applies `f c` to the entries of two tensors has as derivative, on their L2
product, the sum of the slope maps of its two partial slopes, and its pair of pullbacks is the
adjoint (`pointwise₂_backward_eq_adjoint`).

The certificates are registered with `op_cert` at every shape, with a frozen coefficient, bound,
epsilon or replacement value held fixed, on these domains, at every entry:

* Everywhere: `exp`, `exp2`, `expm1`, `sin`, `cos`, `neg`, `square`, `sinh`, `cosh`, `sigmoid`,
  `tanh`, `softplus`, `silu`, the tanh and the exact `gelu`, `asinh`, `atan`, `erf`, `scale` and
  `nanToNum`.
* Away from zero: `log`, `log2`, `log10`, `reciprocal`, `sqrt`, `abs` and `relu`, and `elu` and
  `leakyReLU` for every coefficient, everywhere at coefficient one.
* On other domains: `log1p` away from `-1`, `sqrt .clampNegative` and `rsqrt` for positive
  entries, `acosh` above one, `atanh` on `(-1, 1)`, `asin` and `acos` away from `-1` and `1`,
  `tan` where the cosine is nonzero, `safeInv` where `x + ε` is nonzero, `clamp` away from both
  bounds (everywhere for `hi ≤ lo`, where it is the constant `hi`), `safeLog` where
  `softplus x + ε` is nonzero, and `smoothAbs` where `x * x + ε` is nonzero.
* In both inputs together: `add`, `sub` and `mul` everywhere, and true `div` where every entry of
  the denominator is nonzero.
* In one input, the other fixed: true `div` in its first input for every denominator, zero
  entries included, `safeDiv` in its first input (its divisor is a frozen port), and `max` and
  `min` in either input away from ties.

The kinks, the poles and the edges of the real domains have no certificate here.
`TorchLean.NN.Op.Rounding` certifies `sign` and the rounding operators: the derivative off their
jumps, and the stored zero pullback as a convention. The kinks are `relu` and `abs` at zero, `elu`
and `leakyReLU` at zero for a coefficient other than one, `clamp` at a bound for `lo < hi`, and
`max` and `min` at a tie. The poles are `log`, `log2`, `log10` and `reciprocal` at zero, `log1p` at
`-1`, `tan` where the cosine vanishes, `safeInv` where `x + ε` vanishes, `safeLog` where
`softplus x + ε` vanishes, and `div` in its second input at a zero denominator. The edges of the
real domains and the points beyond them are `sqrt` at zero, `smoothAbs` where `x * x + ε` vanishes,
`sqrt .clampNegative` and `rsqrt` at zero and below, `acosh` at one and below, `atanh` outside
`(-1, 1)`, and `asin` and `acos` at `-1` and `1`. There the real root is the clamped `√(max x 0)`,
and `rsqrt` is Lean's `1 / √x`. Beyond an edge the stored slope need not be the derivative of the
real forward: below one the real `acosh` is `log x`, with derivative `2` at `1/2`, where the slope
is zero.

At every input, the forward-mode rule `jvp?` of `log`, `abs`, `reciprocal`, `sqrt` in both modes
and `div` is adjoint to the pullback, registered as an `algebraic` certificate (`log_jvp_adjoint`,
`abs_jvp_adjoint`, `reciprocal_jvp_adjoint`, `sqrt_jvp_adjoint`, `sqrt_clampNegative_jvp_adjoint`,
`div_jvp_adjoint`). A tape node of these arms takes that rule as its JVP and needs a derivative
certificate only where it claims a derivative.
-/

@[expose] public section

open FloatLib.Numerics (MathFunctions)

namespace TorchLean.Op

open TorchLean.Tensor

variable {s : Shape}

/-! ## The slope map -/

/-- Multiplying each entry by its slope `d c`, as a continuous linear map on real tensors. -/
noncomputable def slopeCLM (d : s.Coord → ℝ) : Tensor ℝ s →L[ℝ] Tensor ℝ s :=
  Tensor.assemble fun c => d c • Tensor.coordinateCLM c

/-- The slope map multiplies each entry by its slope. -/
@[simp] theorem slopeCLM_apply (d : s.Coord → ℝ) (y : Tensor ℝ s) (c : s.Coord) :
    slopeCLM d y c = d c * y c := by
  simp [slopeCLM]

/-- The slope map is self-adjoint. -/
theorem slopeCLM_adjoint (d : s.Coord → ℝ) : (slopeCLM d).adjoint = slopeCLM d := by
  refine ((ContinuousLinearMap.eq_adjoint_iff (slopeCLM d) (slopeCLM d)).mpr ?_).symm
  intro u v
  simp only [Tensor.inner_eq_sum, slopeCLM_apply]
  exact Finset.sum_congr rfl fun c _ => by ring

/-- A map that applies `f c` at each entry `c`, where `f c` has derivative `d c` at `x c`, has the
slope map of `d` as derivative at `x`. -/
theorem pointwise_hasFDerivAt (F : Tensor ℝ s → Tensor ℝ s) (f : s.Coord → ℝ → ℝ)
    (d : s.Coord → ℝ) (x : Tensor ℝ s) (hF : ∀ y c, F y c = f c (y c))
    (hf : ∀ c, HasDerivAt (f c) (d c) (x c)) :
    HasFDerivAt F (slopeCLM d) x := by
  refine Tensor.hasFDerivAt_of_coordinates F _ x fun c => ?_
  have h := (hf c).comp_hasFDerivAt x (Tensor.coordinateCLM c).hasFDerivAt
  simpa only [Function.comp_def, Tensor.coordinateCLM_apply, ← hF] using h

/-- If `F` applies `f c` at each entry `c` and `f c` has derivative `d c` at `x c`, then the slope
map of `d` is the derivative of `F` at `x`, and a cotangent `g` that multiplies `δ` by the same
slopes is the adjoint of that derivative applied to `δ`. -/
theorem pointwise_backward_eq_adjoint (F : Tensor ℝ s → Tensor ℝ s) (f : s.Coord → ℝ → ℝ)
    (d : s.Coord → ℝ) (x δ g : Tensor ℝ s) (hF : ∀ y c, F y c = f c (y c))
    (hf : ∀ c, HasDerivAt (f c) (d c) (x c)) (hg : ∀ c, g c = d c * δ c) :
    HasFDerivAt F (slopeCLM d) x ∧ g = (fderiv ℝ F x).adjoint δ := by
  have h := pointwise_hasFDerivAt F f d x hF hf
  refine ⟨h, ?_⟩
  rw [h.fderiv, slopeCLM_adjoint]
  apply Tensor.Rep.ext
  intro c
  rw [slopeCLM_apply, hg]

/-! ## Two inputs -/

/-- A map that applies `f c` to the entries `c` of two tensors, where `f c` has partial slopes
`d₀ c` and `d₁ c` at `(x c, y c)`, has the sum of the two slope maps on the L2 product as
derivative at `(x, y)`. -/
theorem pointwise₂_hasFDerivAt (F : WithLp 2 (Tensor ℝ s × Tensor ℝ s) → Tensor ℝ s)
    (f : s.Coord → ℝ × ℝ → ℝ) (d₀ d₁ : s.Coord → ℝ) (x y : Tensor ℝ s)
    (hF : ∀ p c, F p c = f c (p.fst c, p.snd c))
    (hf : ∀ c, HasFDerivAt (f c)
      (d₀ c • ContinuousLinearMap.fst ℝ ℝ ℝ + d₁ c • ContinuousLinearMap.snd ℝ ℝ ℝ) (x c, y c)) :
    HasFDerivAt F ((slopeCLM d₀).comp (WithLp.fstL 2 ℝ (Tensor ℝ s) (Tensor ℝ s)) +
      (slopeCLM d₁).comp (WithLp.sndL 2 ℝ (Tensor ℝ s) (Tensor ℝ s))) (WithLp.toLp 2 (x, y)) := by
  have hc : ∀ c, HasFDerivAt (fun p => F p c)
      (d₀ c • (Tensor.coordinateCLM c).comp
          (WithLp.fstL 2 ℝ (Tensor ℝ s) (Tensor ℝ s)) +
        d₁ c • (Tensor.coordinateCLM c).comp
          (WithLp.sndL 2 ℝ (Tensor ℝ s) (Tensor ℝ s))) (WithLp.toLp 2 (x, y)) := by
    intro c
    let L := ((Tensor.coordinateCLM c).comp
        (WithLp.fstL 2 ℝ (Tensor ℝ s) (Tensor ℝ s))).prod
      ((Tensor.coordinateCLM c).comp (WithLp.sndL 2 ℝ (Tensor ℝ s) (Tensor ℝ s)))
    have hp : HasFDerivAt (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) => (p.fst c, p.snd c)) L
        (WithLp.toLp 2 (x, y)) :=
      L.hasFDerivAt
    have h := (hf c).comp (WithLp.toLp 2 (x, y)) hp
    simp only [Function.comp_def, ← hF] at h
    refine h.congr_fderiv (ContinuousLinearMap.ext fun p => ?_)
    simp only [ContinuousLinearMap.comp_apply, _root_.add_apply, _root_.smul_apply,
      ContinuousLinearMap.prod_apply, ContinuousLinearMap.coe_fst', ContinuousLinearMap.coe_snd',
      smul_eq_mul, L]
  refine (Tensor.hasFDerivAt_of_coordinates F _ _ hc).congr_fderiv ?_
  refine ContinuousLinearMap.ext fun p => Tensor.Rep.ext fun c => ?_
  simp only [Tensor.assemble_apply, _root_.add_apply, _root_.smul_apply,
    ContinuousLinearMap.comp_apply,
    Tensor.coordinateCLM_apply, smul_eq_mul, Tensor.add_apply, slopeCLM_apply]

/-- If `F` applies `f c` to the entries `c` of two tensors and `f c` has partial slopes `d₀ c` and
`d₁ c` at `(x c, y c)`, then the sum of the two slope maps on the L2 product is the derivative of
`F` at `(x, y)`, and cotangents `g₀` and `g₁` that multiply `δ` by the slopes are the adjoint of
that derivative applied to `δ`. -/
theorem pointwise₂_backward_eq_adjoint (F : WithLp 2 (Tensor ℝ s × Tensor ℝ s) → Tensor ℝ s)
    (f : s.Coord → ℝ × ℝ → ℝ) (d₀ d₁ : s.Coord → ℝ) (x y δ g₀ g₁ : Tensor ℝ s)
    (hF : ∀ p c, F p c = f c (p.fst c, p.snd c))
    (hf : ∀ c, HasFDerivAt (f c)
      (d₀ c • ContinuousLinearMap.fst ℝ ℝ ℝ + d₁ c • ContinuousLinearMap.snd ℝ ℝ ℝ) (x c, y c))
    (hg₀ : ∀ c, g₀ c = d₀ c * δ c) (hg₁ : ∀ c, g₁ c = d₁ c * δ c) :
    HasFDerivAt F ((slopeCLM d₀).comp (WithLp.fstL 2 ℝ (Tensor ℝ s) (Tensor ℝ s)) +
        (slopeCLM d₁).comp (WithLp.sndL 2 ℝ (Tensor ℝ s) (Tensor ℝ s))) (WithLp.toLp 2 (x, y)) ∧
      WithLp.toLp 2 (g₀, g₁) = (fderiv ℝ F (WithLp.toLp 2 (x, y))).adjoint δ := by
  have h := pointwise₂_hasFDerivAt F f d₀ d₁ x y hF hf
  refine ⟨h, ?_⟩
  have h₀ : g₀ = (slopeCLM d₀).adjoint δ := by
    rw [slopeCLM_adjoint]
    exact Tensor.Rep.ext fun c => by rw [slopeCLM_apply, hg₀]
  have h₁ : g₁ = (slopeCLM d₁).adjoint δ := by
    rw [slopeCLM_adjoint]
    exact Tensor.Rep.ext fun c => by rw [slopeCLM_apply, hg₁]
  rw [h.fderiv, h₀, h₁]
  apply ext_inner_left ℝ
  intro q
  rw [ContinuousLinearMap.adjoint_inner_right, WithLp.prod_inner_apply, _root_.add_apply,
    inner_add_left]
  simp only [ContinuousLinearMap.comp_apply, ContinuousLinearMap.adjoint_inner_right,
    WithLp.fstL_apply, WithLp.sndL_apply]
  rfl

/-! ## Scalar derivatives -/

/-- On `(-1, 1)` the inverse hyperbolic tangent has derivative `(1 - x ^ 2)⁻¹`. -/
theorem hasDerivAt_artanh {x : ℝ} (hx : |x| < 1) : HasDerivAt Real.artanh (1 - x ^ 2)⁻¹ x := by
  obtain ⟨h₁, h₂⟩ := abs_lt.mp hx
  have hp : (1 : ℝ) + x ≠ 0 := by linarith
  have hm : (1 : ℝ) - x ≠ 0 := by linarith
  have h : HasDerivAt (fun y => 1 / 2 * (Real.log (1 + y) - Real.log (1 - y)))
      (1 / 2 * (1 / (1 + x) - -1 / (1 - x))) x :=
    ((((hasDerivAt_id' x).const_add 1).log hp).sub
      (((hasDerivAt_id' x).const_sub 1).log hm)).const_mul (1 / 2)
  refine (h.congr_deriv ?_).congr_of_eventuallyEq ?_
  · rw [show (1 : ℝ) - x ^ 2 = (1 + x) * (1 - x) by ring]
    field_simp
    ring
  · filter_upwards [Ioo_mem_nhds h₁ h₂] with y hy
    have hp' : (1 : ℝ) + y ≠ 0 := by linarith [hy.1]
    have hm' : (1 : ℝ) - y ≠ 0 := by linarith [hy.2]
    rw [Real.artanh_eq_half_log ⟨hy.1.le, hy.2.le⟩, Real.log_div hp' hm']

/-- The exact GELU `z / 2 * (1 + erf (z / √2))` has derivative `Φ x + x φ x`, with `Φ` the standard
normal distribution function and `φ` its density. -/
theorem hasDerivAt_geluErf (x : ℝ) :
    HasDerivAt (fun z => z / 2 * (1 + TorchLean.erf (z / √2)))
      ((1 + TorchLean.erf (x / √2)) / 2 + x * Real.exp (-x ^ 2 / 2) / √(2 * Real.pi)) x := by
  have he := (hasDerivAt_erf (x / √2)).comp x ((hasDerivAt_id' x).div_const √2)
  have h : HasDerivAt (fun z => z / 2 * (1 + TorchLean.erf (z / √2)))
      (1 / 2 * (1 + TorchLean.erf (x / √2)) +
        x / 2 * (2 / √Real.pi * Real.exp (-(x / √2) ^ 2) * (1 / √2))) x :=
    ((hasDerivAt_id' x).div_const 2).mul (he.const_add 1)
  refine h.congr_deriv ?_
  have h2 : (√2 : ℝ) ≠ 0 := by positivity
  have hpi : √Real.pi ≠ 0 := (Real.sqrt_pos.mpr Real.pi_pos).ne'
  have hsq : -(x / √2) ^ 2 = -x ^ 2 / 2 := by
    rw [div_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)]
    ring
  rw [hsq, Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 2) Real.pi]
  field_simp

/-- Away from both bounds, `z ↦ min hi (max lo z)` has derivative one strictly between the
bounds and zero elsewhere. -/
theorem hasDerivAt_clamp {lo hi x : ℝ} (hlo : x ≠ lo) (hhi : x ≠ hi) :
    HasDerivAt (fun z => Min.min hi (Max.max lo z)) (if lo < x ∧ x < hi then 1 else 0) x := by
  by_cases h : lo < x ∧ x < hi
  · rw [ite_eq_left h]
    refine (hasDerivAt_id' x).congr_of_eventuallyEq ?_
    filter_upwards [Ioo_mem_nhds h.1 h.2] with z hz
    rw [max_eq_right (le_of_lt hz.1), min_eq_right (le_of_lt hz.2)]
  · rw [ite_eq_right h]
    rcases hlo.lt_or_gt with h₁ | h₁
    · refine (hasDerivAt_const x (Min.min hi lo)).congr_of_eventuallyEq ?_
      filter_upwards [Iio_mem_nhds h₁] with z hz
      rw [max_eq_left (le_of_lt hz)]
    · have h₂ : hi < x := lt_of_le_of_ne (not_lt.mp fun h' => h ⟨h₁, h'⟩) hhi.symm
      refine (hasDerivAt_const x hi).congr_of_eventuallyEq ?_
      filter_upwards [Ioi_mem_nhds h₂] with z hz
      exact min_eq_left (le_trans (le_of_lt hz) (le_max_right lo z))

/-- Away from ties, `z ↦ max a z` has derivative one above `a` and zero below. -/
theorem hasDerivAt_const_max_of_ne {a y : ℝ} (h : a ≠ y) :
    HasDerivAt (fun z => Max.max a z) (if a < y then 1 else 0) y := by
  rcases h.lt_or_gt with h | h
  · rw [ite_eq_left h]
    refine (hasDerivAt_id' y).congr_of_eventuallyEq ?_
    filter_upwards [Ioi_mem_nhds h] with z hz
    exact max_eq_right (le_of_lt hz)
  · rw [ite_eq_right (not_lt.mpr h.le)]
    refine (hasDerivAt_const y a).congr_of_eventuallyEq ?_
    filter_upwards [Iio_mem_nhds h] with z hz
    exact max_eq_left (le_of_lt hz)

/-- Away from ties, `z ↦ min a z` has derivative one below `a` and zero above. -/
theorem hasDerivAt_const_min_of_ne {a y : ℝ} (h : a ≠ y) :
    HasDerivAt (fun z => Min.min a z) (if y < a then 1 else 0) y := by
  rcases h.lt_or_gt with h | h
  · rw [ite_eq_right (not_lt.mpr h.le)]
    refine (hasDerivAt_const y a).congr_of_eventuallyEq ?_
    filter_upwards [Ioi_mem_nhds h] with z hz
    exact min_eq_left (le_of_lt hz)
  · rw [ite_eq_left h]
    refine (hasDerivAt_id' y).congr_of_eventuallyEq ?_
    filter_upwards [Iio_mem_nhds h] with z hz
    exact min_eq_right (le_of_lt hz)

/-- Away from ties, `z ↦ max z a` has derivative one above `a` and zero below. -/
theorem hasDerivAt_max_const_of_ne {a y : ℝ} (h : a ≠ y) :
    HasDerivAt (fun z => Max.max z a) (if a < y then 1 else 0) y :=
  (hasDerivAt_const_max_of_ne h).congr_of_eventuallyEq
    (Filter.Eventually.of_forall fun z => max_comm z a)

/-- Away from ties, `z ↦ min z a` has derivative one below `a` and zero above. -/
theorem hasDerivAt_min_const_of_ne {a y : ℝ} (h : a ≠ y) :
    HasDerivAt (fun z => Min.min z a) (if y < a then 1 else 0) y :=
  (hasDerivAt_const_min_of_ne h).congr_of_eventuallyEq
    (Filter.Eventually.of_forall fun z => min_comm z a)

/-! ## Operators without a domain condition -/

/-- `exp` has its slope map as derivative, and its pullback is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.exp derivative]
theorem exp_backward_eq_adjoint (x δ : Tensor ℝ s) :
    HasFDerivAt (fun y => (((Op.exp (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => Real.exp (x c)) x ∧
      (((Op.exp (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.exp (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ => Real.exp) (fun c => Real.exp (x c)) x δ _
    (fun y c => by
      show (Tensor.exp y) c = _; simp only [Tensor.exp, Tensor.map_apply]; rfl)
    (fun c => Real.hasDerivAt_exp (x c))
    (fun c => by
      show (Tensor.zipWith Context.expMul x δ) c = _
      simp only [Tensor.zipWith_apply]; rfl)

/-- `sin` has its slope map as derivative, and its pullback is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.sin derivative]
theorem sin_backward_eq_adjoint (x δ : Tensor ℝ s) :
    HasFDerivAt (fun y => (((Op.sin (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => Real.cos (x c)) x ∧
      (((Op.sin (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.sin (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ => Real.sin) (fun c => Real.cos (x c)) x δ _
    (fun y c => by
      show (Tensor.sin y) c = _; simp only [Tensor.sin, Tensor.map_apply]; rfl)
    (fun c => Real.hasDerivAt_sin (x c))
    (fun c => by
      show (Tensor.cos x * δ) c = _
      simp only [Tensor.cos, Tensor.mul_def, Tensor.zipWith_apply,
        Tensor.map_apply]; rfl)

/-- `cos` has its slope map as derivative, and its pullback is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.cos derivative]
theorem cos_backward_eq_adjoint (x δ : Tensor ℝ s) :
    HasFDerivAt (fun y => (((Op.cos (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => -Real.sin (x c)) x ∧
      (((Op.cos (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.cos (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ => Real.cos) (fun c => -Real.sin (x c)) x δ _
    (fun y c => by
      show (Tensor.cos y) c = _; simp only [Tensor.cos, Tensor.map_apply]; rfl)
    (fun c => Real.hasDerivAt_cos (x c))
    (fun c => by
      show (Tensor.neg (Tensor.sin x) * δ) c = _
      simp only [Tensor.neg, Tensor.sin, Tensor.mul_def, Tensor.zipWith_apply,
        Tensor.map_apply]; rfl)

/-- `neg` has its slope map as derivative, and its pullback is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.neg derivative]
theorem neg_backward_eq_adjoint (x δ : Tensor ℝ s) :
    HasFDerivAt (fun y => (((Op.neg (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun _ => -1) x ∧
      (((Op.neg (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.neg (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ => Neg.neg) (fun _ => -1) x δ _
    (fun y c => by
      show (Tensor.neg y) c = _; simp only [Tensor.neg, Tensor.map_apply])
    (fun c => hasDerivAt_neg (x c))
    (fun c => by
      show (Tensor.neg δ) c = _
      simp only [Tensor.neg, Tensor.map_apply]; ring)

/-- `square` has its slope map as derivative, and its pullback is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.square derivative]
theorem square_backward_eq_adjoint (x δ : Tensor ℝ s) :
    HasFDerivAt (fun y => (((Op.square (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => 2 * x c) x ∧
      (((Op.square (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.square (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ y => y * y) (fun c => 2 * x c) x δ _
    (fun y c => by
      show (Tensor.square y) c = _
      simp only [Tensor.square, Tensor.map_apply])
    (fun c => Activation.hasDerivAt_mul_self (x c))
    (fun c => by
      show ((Tensor.full s (2 : ℝ) * x) * δ) c = _
      simp only [Tensor.mul_def, Tensor.zipWith_apply, Tensor.full_apply])

/-- `sinh` has its slope map as derivative, and its pullback is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.sinh derivative]
theorem sinh_backward_eq_adjoint (x δ : Tensor ℝ s) :
    HasFDerivAt (fun y => (((Op.sinh (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => MathFunctions.cosh (x c)) x ∧
      (((Op.sinh (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.sinh (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ => MathFunctions.sinh)
    (fun c => MathFunctions.cosh (x c)) x δ _
    (fun y c => by
      show (Tensor.sinh y) c = _; simp only [Tensor.sinh, Tensor.map_apply])
    (fun c => Real.hasDerivAt_sinh (x c))
    (fun c => by
      show (Tensor.map MathFunctions.cosh x * δ) c = _
      simp only [Tensor.mul_def, Tensor.zipWith_apply, Tensor.map_apply])

/-- `cosh` has its slope map as derivative, and its pullback is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.cosh derivative]
theorem cosh_backward_eq_adjoint (x δ : Tensor ℝ s) :
    HasFDerivAt (fun y => (((Op.cosh (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => MathFunctions.sinh (x c)) x ∧
      (((Op.cosh (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.cosh (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ => MathFunctions.cosh)
    (fun c => MathFunctions.sinh (x c)) x δ _
    (fun y c => by
      show (Tensor.cosh y) c = _; simp only [Tensor.cosh, Tensor.map_apply])
    (fun c => Real.hasDerivAt_cosh (x c))
    (fun c => by
      show (Tensor.map MathFunctions.sinh x * δ) c = _
      simp only [Tensor.mul_def, Tensor.zipWith_apply, Tensor.map_apply])

/-- `sigmoid` has its slope map as derivative, and its pullback is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.sigmoid derivative]
theorem sigmoid_backward_eq_adjoint (x δ : Tensor ℝ s) :
    HasFDerivAt (fun y => (((Op.sigmoid (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => TorchLean.Numeric.sigmoidDeriv (x c)) x ∧
      (((Op.sigmoid (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.sigmoid (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ => Context.sigmoid)
    (fun c => TorchLean.Numeric.sigmoidDeriv (x c)) x δ _
    (fun y c => by
      show (Tensor.map Context.sigmoid y) c = _
      simp only [Tensor.map_apply])
    (fun c => Context.hasDerivAt_sigmoid (x c))
    (fun c => by
      show (Tensor.sigmoidBackward x δ) c = _
      simp only [Tensor.sigmoidBackward, Tensor.zipWith_apply]
      rfl)

/-- `tanh` has its slope map as derivative, and its pullback is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.tanh derivative]
theorem tanh_backward_eq_adjoint (x δ : Tensor ℝ s) :
    HasFDerivAt (fun y => (((Op.tanh (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => TorchLean.Numeric.tanhDeriv (x c)) x ∧
      (((Op.tanh (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.tanh (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ => MathFunctions.tanh)
    (fun c => TorchLean.Numeric.tanhDeriv (x c)) x δ _
    (fun y c => by
      show (Tensor.map MathFunctions.tanh y) c = _
      simp only [Tensor.map_apply])
    (fun c => Activation.hasDerivAt_tanh (x c))
    (fun c => by
      show (Tensor.tanhBackward x δ) c = _
      simp only [Tensor.tanhBackward, Tensor.zipWith_apply]
      rfl)

/-- `softplus` has its slope map as derivative, and its pullback is the adjoint of the derivative.
-/
@[op_cert TorchLean.Op.softplus derivative]
theorem softplus_backward_eq_adjoint (x δ : Tensor ℝ s) :
    HasFDerivAt (fun y => (((Op.softplus (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => Context.sigmoid (x c)) x ∧
      (((Op.softplus (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.softplus (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ => Activation.softplus)
    (fun c => Context.sigmoid (x c)) x δ _
    (fun y c => by
      show (Tensor.map Activation.softplus y) c = _
      simp only [Tensor.map_apply])
    (fun c => Activation.hasDerivAt_softplus (x c))
    (fun c => by
      show (Tensor.map Context.sigmoid x * δ) c = _
      simp only [Tensor.mul_def, Tensor.zipWith_apply, Tensor.map_apply])

/-- `silu` has its slope map as derivative, and its pullback is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.silu derivative]
theorem silu_backward_eq_adjoint (x δ : Tensor ℝ s) :
    HasFDerivAt (fun y => (((Op.silu (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => Activation.siluDeriv (x c)) x ∧
      (((Op.silu (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.silu (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ => Activation.silu)
    (fun c => Activation.siluDeriv (x c)) x δ _
    (fun y c => by
      show (Tensor.map Activation.silu y) c = _
      simp only [Tensor.map_apply])
    (fun c => Activation.hasDerivAt_silu (x c))
    (fun c => by
      show (Tensor.map Activation.siluDeriv x * δ) c = _
      simp only [Tensor.mul_def, Tensor.zipWith_apply, Tensor.map_apply])

/-- The tanh approximation of `gelu` has its slope map as derivative, and its pullback is the
adjoint of the derivative. -/
@[op_cert TorchLean.Op.gelu derivative]
theorem gelu_tanh_backward_eq_adjoint (x δ : Tensor ℝ s) :
    HasFDerivAt (fun y => (((Op.gelu (s := s) .tanh).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => Activation.geluTanhDeriv (x c)) x ∧
      (((Op.gelu (s := s) .tanh).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.gelu (s := s) .tanh).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ => Activation.geluTanh)
    (fun c => Activation.geluTanhDeriv (x c)) x δ _
    (fun y c => by
      show (Tensor.map Activation.geluTanh y) c = _
      simp only [Tensor.map_apply])
    (fun c => Activation.hasDerivAt_geluTanh (x c))
    (fun c => by
      show (Tensor.map Activation.geluTanhDeriv x * δ) c = _
      simp only [Tensor.mul_def, Tensor.zipWith_apply, Tensor.map_apply])

/-- The exact `gelu` has its slope map as derivative, and its pullback is the adjoint of the
derivative. -/
@[op_cert TorchLean.Op.gelu derivative]
theorem gelu_backward_eq_adjoint (x δ : Tensor ℝ s) :
    HasFDerivAt (fun y => (((Op.gelu (s := s) .none).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c =>
          (1 + TorchLean.erf (x c / √2)) / 2 + x c * Real.exp (-x c ^ 2 / 2) / √(2 * Real.pi)) x ∧
      (((Op.gelu (s := s) .none).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.gelu (s := s) .none).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ := by
  have hhalf : ((1 / 2 : ℚ) : ℝ) = 1 / 2 := by norm_num
  have hroot : √(1 / 2 : ℝ) = (√2)⁻¹ := by rw [one_div, Real.sqrt_inv]
  refine pointwise_backward_eq_adjoint _ (fun _ z => z / 2 * (1 + TorchLean.erf (z / √2)))
    (fun c => (1 + TorchLean.erf (x c / √2)) / 2 + x c * Real.exp (-x c ^ 2 / 2) / √(2 * Real.pi))
    x δ _ (fun y c => ?_) (fun c => hasDerivAt_geluErf (x c)) (fun c => ?_)
  · show (Tensor.gelu y) c = _
    simp only [Tensor.gelu, Tensor.map_apply, Context.geluErf,
      Context.isInf_real, Bool.false_eq_true, ↓reduceIte]
    show y c * ((1 / 2 : ℚ) : ℝ) * (1 + TorchLean.erf (y c * √((1 / 2 : ℚ) : ℝ))) =
      y c / 2 * (1 + TorchLean.erf (y c / √2))
    rw [hhalf, hroot, ← div_eq_mul_inv]
    ring
  · show (Tensor.geluBackward x δ) c = _
    simp only [Tensor.geluBackward, Tensor.zipWith_apply, Context.isInf_real, Bool.false_eq_true,
      ↓reduceIte]
    show δ c * (((1 / 2 : ℚ) : ℝ) * (1 + TorchLean.erf (x c * √((1 / 2 : ℚ) : ℝ))) +
      x c * ((1 + 1) / √Real.pi * √((1 / 2 : ℚ) : ℝ) * ((1 / 2 : ℚ) : ℝ) *
        Real.exp (x c * x c * -((1 / 2 : ℚ) : ℝ)))) = _
    have h2 : (√2 : ℝ) ≠ 0 := by positivity
    have hpi : √Real.pi ≠ 0 := (Real.sqrt_pos.mpr Real.pi_pos).ne'
    rw [hhalf, show x c * √(1 / 2 : ℝ) = x c / √2 by rw [hroot, div_eq_mul_inv], hroot,
      show x c * x c * -(1 / 2 : ℝ) = -x c ^ 2 / 2 by ring,
      Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 2) Real.pi]
    field_simp
    ring

/-- `asinh` has its slope map as derivative, and its pullback is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.asinh derivative]
theorem asinh_backward_eq_adjoint (x δ : Tensor ℝ s) :
    HasFDerivAt (fun y => (((Op.asinh (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => (√(1 + x c ^ 2))⁻¹) x ∧
      (((Op.asinh (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.asinh (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ => Real.arsinh) (fun c => (√(1 + x c ^ 2))⁻¹) x δ _
    (fun y c => by
      show (Tensor.asinh y) c = _
      simp only [Tensor.asinh, Tensor.map_apply]; rfl)
    (fun c => (Tensor.asinhBackward_real x δ c).1)
    (fun c => (Tensor.asinhBackward_real x δ c).2)

/-- `erf` has its slope map as derivative, and its pullback is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.erf derivative]
theorem erf_backward_eq_adjoint (x δ : Tensor ℝ s) :
    HasFDerivAt (fun y => (((Op.erf (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => 2 / √Real.pi * Real.exp (-x c ^ 2)) x ∧
      (((Op.erf (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.erf (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ => TorchLean.erf)
    (fun c => 2 / √Real.pi * Real.exp (-x c ^ 2)) x δ _
    (fun y c => by
      show (Tensor.erf y) c = _
      simp only [Tensor.erf, Tensor.map_apply]; rfl)
    (fun c => (Tensor.erfBackward_real x δ c).1)
    (fun c => (Tensor.erfBackward_real x δ c).2)

/-- `exp2` has its slope map as derivative, and its pullback is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.exp2 derivative]
theorem exp2_backward_eq_adjoint (x δ : Tensor ℝ s) :
    HasFDerivAt (fun y => (((Op.exp2 (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => (2 : ℝ) ^ x c * Real.log 2) x ∧
      (((Op.exp2 (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.exp2 (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ z => (2 : ℝ) ^ z)
    (fun c => (2 : ℝ) ^ x c * Real.log 2) x δ _
    (fun y c => by show (Tensor.exp2 y) c = _; exact Tensor.exp2_apply y c)
    (fun c => (Real.hasStrictDerivAt_const_rpow (by norm_num) (x c)).hasDerivAt)
    (fun c => by
      show (Tensor.exp2Backward x δ) c = _
      simp only [Tensor.exp2Backward, Tensor.zipWith_apply]
      show δ c * (2 : ℝ) ^ x c * Real.log (1 + 1) = _
      rw [one_add_one_eq_two]
      ring)

/-- `expm1` has its slope map as derivative, and its pullback is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.expm1 derivative]
theorem expm1_backward_eq_adjoint (x δ : Tensor ℝ s) :
    HasFDerivAt (fun y => (((Op.expm1 (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => Real.exp (x c)) x ∧
      (((Op.expm1 (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.expm1 (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ z => Real.exp z - 1) (fun c => Real.exp (x c)) x δ _
    (fun y c => by
      show (Tensor.expm1 y) c = _
      rw [Tensor.expm1_apply]
      exact Context.expm1_eq_exp_sub_one (y c))
    (fun c => (Real.hasDerivAt_exp (x c)).sub_const 1)
    (fun c => by
      show (Tensor.zipWith Context.expMul x δ) c = _
      simp only [Tensor.zipWith_apply]; rfl)

/-- `atan` has its slope map as derivative, and its pullback is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.atan derivative]
theorem atan_backward_eq_adjoint (x δ : Tensor ℝ s) :
    HasFDerivAt (fun y => (((Op.atan (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => (1 + x c ^ 2)⁻¹) x ∧
      (((Op.atan (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.atan (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ => Real.arctan) (fun c => (1 + x c ^ 2)⁻¹) x δ _
    (fun y c => by show (Tensor.atan y) c = _; exact Tensor.atan_apply y c)
    (fun c => Real.hasDerivAt_arctan' (x c))
    (fun c => by
      show (Tensor.atanBackward x δ) c = _
      simp only [Tensor.atanBackward, Tensor.zipWith_apply]
      show (if x c * x c + 1 - (x c * x c + 1) == 0 then δ c / (x c * x c + 1)
        else δ c / x c / x c) = _
      rw [ite_eq_left (by simp)]
      ring)

/-- With its coefficient fixed, `scale` has its slope map as derivative, and its pullback is the
adjoint of the derivative. -/
@[op_cert TorchLean.Op.scale derivative 0]
theorem scale_backward_eq_adjoint (x δ : Tensor ℝ s) (coefficient : Tensor ℝ []) :
    HasFDerivAt (fun y => (((Op.scale (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, coefficient, ())).1)
        (slopeCLM fun _ => coefficient.item) x ∧
      (((Op.scale (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, coefficient, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.scale (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, coefficient, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ y => y * coefficient.item) (fun _ => coefficient.item)
    x δ _
    (fun y c => by
      show (Tensor.scale y coefficient.item) c = _
      simp only [Tensor.scale, Tensor.map_apply])
    (fun c => hasDerivAt_mul_const coefficient.item)
    (fun c => by
      show (Tensor.scale δ coefficient.item) c = _
      simp only [Tensor.scale, Tensor.map_apply]; ring)

/-- With its replacement values fixed, `nanToNum` has its slope map as derivative, and its pullback
is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.nanToNum derivative 0]
theorem nanToNum_backward_eq_adjoint (x δ : Tensor ℝ s) (nan posinf neginf : Tensor ℝ []) :
    HasFDerivAt (fun y => (((Op.nanToNum (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, nan, posinf, neginf, ())).1)
        (slopeCLM fun _ => 1) x ∧
      (((Op.nanToNum (s := s)).rule? (α := ℝ) |>.get rfl).backward
          (x, nan, posinf, neginf, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.nanToNum (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, nan, posinf, neginf, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ z => z) (fun _ => 1) x δ _
    (fun y c => by
      show (Tensor.nanToNum y nan.item posinf.item neginf.item) c = _
      rw [Tensor.nanToNum_apply, Context.nanToNum_real])
    (fun c => hasDerivAt_id' (x c))
    (fun c => by
      show (Tensor.nanToNumBackward x δ) c = _
      simp [Tensor.nanToNumBackward])

/-! ## Operators on a domain -/

/-- Away from zero, `log` has its slope map as derivative, and its pullback is the adjoint of the
derivative. -/
@[op_cert TorchLean.Op.log derivative]
theorem log_backward_eq_adjoint (x δ : Tensor ℝ s) (hx : ∀ c, x c ≠ 0) :
    HasFDerivAt (fun y => (((Op.log (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => (x c)⁻¹) x ∧
      (((Op.log (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.log (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ => Real.log) (fun c => (x c)⁻¹) x δ _
    (fun y c => by
      show (Tensor.log y) c = _; simp only [Tensor.log, Tensor.map_apply]; rfl)
    (fun c => Real.hasDerivAt_log (hx c))
    (fun c => by
      show (δ / x) c = _
      simp only [Tensor.div_def, Tensor.zipWith_apply]; rw [div_eq_inv_mul])

/-- Away from zero, `log2` has its slope map as derivative, and its pullback is the adjoint of the
derivative. -/
@[op_cert TorchLean.Op.log2 derivative]
theorem log2_backward_eq_adjoint (x δ : Tensor ℝ s) (hx : ∀ c, x c ≠ 0) :
    HasFDerivAt (fun y => (((Op.log2 (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => (x c * Real.log 2)⁻¹) x ∧
      (((Op.log2 (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.log2 (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ => Real.logb 2) (fun c => (x c * Real.log 2)⁻¹) x δ _
    (fun y c => by show (Tensor.log2 y) c = _; exact Tensor.log2_apply y c)
    (fun c => ((Real.hasDerivAt_log (hx c)).div_const (Real.log 2)).congr_deriv (by ring))
    (fun c => by
      show (Tensor.log2Backward x δ) c = _
      simp only [Tensor.log2Backward, Tensor.zipWith_apply]
      show δ c / (x c * Real.log (1 + 1)) = _
      rw [one_add_one_eq_two, div_eq_inv_mul])

/-- Away from zero, `log10` has its slope map as derivative, and its pullback is the adjoint of the
derivative. -/
@[op_cert TorchLean.Op.log10 derivative]
theorem log10_backward_eq_adjoint (x δ : Tensor ℝ s) (hx : ∀ c, x c ≠ 0) :
    HasFDerivAt (fun y => (((Op.log10 (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => (x c * Real.log 10)⁻¹) x ∧
      (((Op.log10 (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.log10 (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ => Real.logb 10) (fun c => (x c * Real.log 10)⁻¹)
    x δ _
    (fun y c => by show (Tensor.log10 y) c = _; exact Tensor.log10_apply y c)
    (fun c => ((Real.hasDerivAt_log (hx c)).div_const (Real.log 10)).congr_deriv (by ring))
    (fun c => by
      show (Tensor.log10Backward x δ) c = _
      simp only [Tensor.log10Backward, Tensor.zipWith_apply]
      show δ c / (x c * Real.log ((10 : ℕ) : ℝ)) = _
      rw [Nat.cast_ofNat, div_eq_inv_mul])

/-- Away from `-1`, `log1p` has its slope map as derivative, and its pullback is the adjoint of the
derivative. -/
@[op_cert TorchLean.Op.log1p derivative]
theorem log1p_backward_eq_adjoint (x δ : Tensor ℝ s) (hx : ∀ c, x c ≠ -1) :
    HasFDerivAt (fun y => (((Op.log1p (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => (1 + x c)⁻¹) x ∧
      (((Op.log1p (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.log1p (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ z => Real.log (1 + z)) (fun c => (1 + x c)⁻¹) x δ _
    (fun y c => by
      show (Tensor.log1p y) c = _
      rw [Tensor.log1p_apply]
      exact Context.log1p_eq_log_one_add (y c))
    (fun c => (((hasDerivAt_id' (x c)).const_add 1).log
      (fun h => hx c (by linarith))).congr_deriv (one_div _))
    (fun c => by
      show (Tensor.log1pBackward x δ) c = _
      simp only [Tensor.log1pBackward, Tensor.zipWith_apply]
      show δ c / (1 + x c) = _
      rw [div_eq_inv_mul])

/-- Away from zero, `reciprocal` has its slope map as derivative, and its pullback is the adjoint of
the derivative. -/
@[op_cert TorchLean.Op.reciprocal derivative]
theorem reciprocal_backward_eq_adjoint (x δ : Tensor ℝ s) (hx : ∀ c, x c ≠ 0) :
    HasFDerivAt (fun y => (((Op.reciprocal (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => -((x c) ^ 2)⁻¹) x ∧
      (((Op.reciprocal (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.reciprocal (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ => (·⁻¹)) (fun c => -((x c) ^ 2)⁻¹) x δ _
    (fun y c => by
      show (Tensor.reciprocal y) c = _
      simp only [Tensor.reciprocal, Tensor.map_apply]
      exact one_div _)
    (fun c => hasDerivAt_inv (hx c))
    (fun c => by
      show ((Tensor.neg δ / x) / x) c = _
      simp only [Tensor.div_def, Tensor.zipWith_apply, Tensor.neg,
        Tensor.map_apply]
      field_simp)

/-- For positive entries, `rsqrt` has its slope map as derivative, and its pullback is the adjoint
of the derivative. -/
@[op_cert TorchLean.Op.rsqrt derivative]
theorem rsqrt_backward_eq_adjoint (x δ : Tensor ℝ s) (hx : ∀ c, 0 < x c) :
    HasFDerivAt (fun y => (((Op.rsqrt (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => -(1 / 2) * (√(x c))⁻¹ ^ 3) x ∧
      (((Op.rsqrt (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.rsqrt (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ z => (√z)⁻¹) (fun c => -(1 / 2) * (√(x c))⁻¹ ^ 3)
    x δ _
    (fun y c => by
      show (Tensor.rsqrt y) c = _
      rw [Tensor.rsqrt_apply]
      exact one_div _)
    (fun c => by
      have h : HasDerivAt (fun z => (√z)⁻¹) (-(1 / (2 * √(x c))) / √(x c) ^ 2) (x c) :=
        (Real.hasDerivAt_sqrt (hx c).ne').inv (Real.sqrt_pos.mpr (hx c)).ne'
      exact h.congr_deriv (by ring))
    (fun c => by
      show (Tensor.rsqrtBackward x δ) c = _
      simp only [Tensor.rsqrtBackward, Tensor.zipWith_apply]
      show -((1 / 2 : ℚ) : ℝ) * δ c * (1 / √(x c) * (1 / √(x c)) * (1 / √(x c))) = _
      push_cast
      ring)

/-- Away from zero, `sqrt` has its slope map as derivative (the slope is zero below zero, where the
real root vanishes), and its pullback is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.sqrt derivative]
theorem sqrt_backward_eq_adjoint (x δ : Tensor ℝ s) (hx : ∀ c, x c ≠ 0) :
    HasFDerivAt (fun y => (((Op.sqrt (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => 1 / (2 * √(x c))) x ∧
      (((Op.sqrt (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.sqrt (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ z => √z) (fun c => 1 / (2 * √(x c))) x δ _
    (fun y c => by
      show (Tensor.sqrt y) c = _
      exact Tensor.sqrt_apply y c)
    (fun c => by exact Real.hasDerivAt_sqrt (hx c))
    (fun c => by
      show (Tensor.zipWith _ x δ) c = _
      rw [Tensor.zipWith_apply]
      show δ c / (((2 : ℕ) : ℝ) * |√(x c)|) = 1 / (2 * √(x c)) * δ c
      rw [abs_of_nonneg (Real.sqrt_nonneg _), Nat.cast_ofNat]
      ring)

/-- For positive entries, `sqrt .clampNegative` has its slope map as derivative, and its pullback
is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.sqrt derivative]
theorem sqrt_clampNegative_backward_eq_adjoint (x δ : Tensor ℝ s) (hx : ∀ c, 0 < x c) :
    HasFDerivAt (fun y => (((Op.sqrt (s := s) .clampNegative).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => 1 / (2 * √(x c))) x ∧
      (((Op.sqrt (s := s) .clampNegative).rule? (α := ℝ) |>.get rfl).backward (x, ())
          (δ, ())).1 =
        (fderiv ℝ (fun y =>
          (((Op.sqrt (s := s) .clampNegative).rule? (α := ℝ) |>.get rfl).forward (y, ())).1)
            x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ z => √(Max.max z 0)) (fun c => 1 / (2 * √(x c))) x δ _
    (fun y c => by
      show (Tensor.sqrt y .clampNegative) c = _
      exact Tensor.sqrt_clampNegative_apply y c)
    (fun c => by
      refine (Real.hasDerivAt_sqrt (hx c).ne').congr_of_eventuallyEq ?_
      filter_upwards [lt_mem_nhds (hx c)] with z hz
      rw [max_eq_left (le_of_lt hz)])
    (fun c => by
      show (Tensor.zipWith _ x δ) c = _
      rw [Tensor.zipWith_apply]
      show (if x c > 0 then _ else _) = _
      rw [ite_eq_left (hx c)]
      show δ c / (((2 : ℕ) : ℝ) * √(x c)) = _
      rw [Nat.cast_ofNat]
      ring)

/-- Above one, `acosh` has its slope map as derivative, and its pullback is the adjoint of the
derivative. -/
@[op_cert TorchLean.Op.acosh derivative]
theorem acosh_backward_eq_adjoint (x δ : Tensor ℝ s) (hx : ∀ c, 1 < x c) :
    HasFDerivAt (fun y => (((Op.acosh (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => (√(x c ^ 2 - 1))⁻¹) x ∧
      (((Op.acosh (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.acosh (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ => Real.arcosh) (fun c => (√(x c ^ 2 - 1))⁻¹) x δ _
    (fun y c => by
      show (Tensor.acosh y) c = _
      simp only [Tensor.acosh, Tensor.map_apply]; rfl)
    (fun c => (Tensor.acoshBackward_real x δ c (hx c)).1)
    (fun c => (Tensor.acoshBackward_real x δ c (hx c)).2)

/-- On `(-1, 1)`, `atanh` has its slope map as derivative, and its pullback is the adjoint of the
derivative. -/
@[op_cert TorchLean.Op.atanh derivative]
theorem atanh_backward_eq_adjoint (x δ : Tensor ℝ s) (hx : ∀ c, |x c| < 1) :
    HasFDerivAt (fun y => (((Op.atanh (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => (1 - x c ^ 2)⁻¹) x ∧
      (((Op.atanh (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.atanh (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ => Real.artanh) (fun c => (1 - x c ^ 2)⁻¹) x δ _
    (fun y c => by show (Tensor.atanh y) c = _; exact Tensor.atanh_apply y c)
    (fun c => hasDerivAt_artanh (hx c))
    (fun c => by
      show (Tensor.atanhBackward x δ) c = _
      simp only [Tensor.atanhBackward, Tensor.zipWith_apply]
      show δ c / ((1 - x c) * (1 + x c)) = _
      rw [div_eq_inv_mul, show (1 - x c) * (1 + x c) = 1 - x c ^ 2 by ring])

/-- Away from `-1` and `1`, `asin` has its slope map as derivative, and its pullback is the adjoint
of the derivative. Outside `[-1, 1]` the real `arcsin` is constant and the slope is zero. -/
@[op_cert TorchLean.Op.asin derivative]
theorem asin_backward_eq_adjoint (x δ : Tensor ℝ s) (hx : ∀ c, x c ≠ -1 ∧ x c ≠ 1) :
    HasFDerivAt (fun y => (((Op.asin (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => 1 / √(1 - x c ^ 2)) x ∧
      (((Op.asin (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.asin (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ => Real.arcsin) (fun c => 1 / √(1 - x c ^ 2)) x δ _
    (fun y c => by show (Tensor.asin y) c = _; exact Tensor.asin_apply y c)
    (fun c => Real.hasDerivAt_arcsin (hx c).1 (hx c).2)
    (fun c => by
      show (Tensor.asinBackward x δ) c = _
      simp only [Tensor.asinBackward, Tensor.zipWith_apply]
      show δ c / √((1 - x c) * (1 + x c)) = _
      rw [show (1 - x c) * (1 + x c) = 1 - x c ^ 2 by ring]
      ring)

/-- Away from `-1` and `1`, `acos` has its slope map as derivative, and its pullback is the adjoint
of the derivative. Outside `[-1, 1]` the real `arccos` is constant and the slope is zero. -/
@[op_cert TorchLean.Op.acos derivative]
theorem acos_backward_eq_adjoint (x δ : Tensor ℝ s) (hx : ∀ c, x c ≠ -1 ∧ x c ≠ 1) :
    HasFDerivAt (fun y => (((Op.acos (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => -(1 / √(1 - x c ^ 2))) x ∧
      (((Op.acos (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.acos (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ => Real.arccos) (fun c => -(1 / √(1 - x c ^ 2))) x δ _
    (fun y c => by show (Tensor.acos y) c = _; exact Tensor.acos_apply y c)
    (fun c => Real.hasDerivAt_arccos (hx c).1 (hx c).2)
    (fun c => by
      show (Tensor.acosBackward x δ) c = _
      simp only [Tensor.acosBackward, Tensor.zipWith_apply]
      show -(δ c / √((1 - x c) * (1 + x c))) = _
      rw [show (1 - x c) * (1 + x c) = 1 - x c ^ 2 by ring]
      ring)

/-- Where the cosine is nonzero, `tan` has its slope map as derivative, and its pullback is the
adjoint of the derivative. -/
@[op_cert TorchLean.Op.tan derivative]
theorem tan_backward_eq_adjoint (x δ : Tensor ℝ s) (hx : ∀ c, Real.cos (x c) ≠ 0) :
    HasFDerivAt (fun y => (((Op.tan (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => 1 / Real.cos (x c) ^ 2) x ∧
      (((Op.tan (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.tan (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ => Real.tan) (fun c => 1 / Real.cos (x c) ^ 2) x δ _
    (fun y c => by
      show (Tensor.tan y) c = _
      simp only [Tensor.tan, Tensor.map_apply]; rfl)
    (fun c => (Tensor.tanBackward_real x δ c (hx c)).1)
    (fun c => (Tensor.tanBackward_real x δ c (hx c)).2)

/-- Away from zero, `abs` has its slope map as derivative, and its pullback is the adjoint of the
derivative. -/
@[op_cert TorchLean.Op.abs derivative]
theorem abs_backward_eq_adjoint (x δ : Tensor ℝ s) (hx : ∀ c, x c ≠ 0) :
    HasFDerivAt (fun y => (((Op.abs (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => Real.sign (x c)) x ∧
      (((Op.abs (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.abs (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ z => |z|) (fun c => Real.sign (x c)) x δ _
    (fun y c => by
      show (Tensor.abs y) c = _
      simp only [Tensor.abs, Tensor.map_apply]; rfl)
    (fun c => by
      show HasDerivAt (fun z => |z|) (Real.sign (x c)) (x c)
      rcases (hx c).lt_or_gt with h | h
      · rw [Real.sign_of_neg h]; exact hasDerivAt_abs_neg h
      · rw [Real.sign_of_pos h]; exact hasDerivAt_abs_pos h)
    (fun c => by
      show (Tensor.sign x * δ) c = _
      simp only [Tensor.mul_def, Tensor.zipWith_apply, Tensor.sign_apply, Context.sign_real])

/-- Away from zero, `relu` has its slope map as derivative, and its pullback is the adjoint of the
derivative. -/
@[op_cert TorchLean.Op.relu derivative]
theorem relu_backward_eq_adjoint (x δ : Tensor ℝ s) (hx : ∀ c, x c ≠ 0) :
    HasFDerivAt (fun y => (((Op.relu (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => Activation.reluDeriv (x c)) x ∧
      (((Op.relu (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.relu (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ => Activation.relu)
    (fun c => Activation.reluDeriv (x c)) x δ _
    (fun y c => by
      show (Tensor.map Activation.relu y) c = _
      simp only [Tensor.map_apply])
    (fun c => Activation.hasDerivAt_relu (x c) (hx c))
    (fun c => by
      show (Tensor.map Activation.reluDeriv x * δ) c = _
      simp only [Tensor.mul_def, Tensor.zipWith_apply, Tensor.map_apply])

/-- Where the coefficient is one or no entry is zero, `leakyReLU` has its slope map as derivative,
and its pullback is the adjoint of the derivative. This is the exact set: at a zero entry the
one-sided slopes are the coefficient and one. -/
@[op_cert Op.leakyReLU derivative 0]
theorem leakyReLU_backward_eq_adjoint (x δ : Tensor ℝ s) (slope : Tensor ℝ [])
    (h : slope.item = 1 ∨ ∀ c, x c ≠ 0) :
    HasFDerivAt (fun y => (((Op.leakyReLU (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, slope, ())).1)
        (slopeCLM fun c => Activation.leakyReLUDeriv (x c) slope.item) x ∧
      (((Op.leakyReLU (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, slope, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.leakyReLU (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, slope, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ y => Activation.leakyReLU y slope.item)
    (fun c => Activation.leakyReLUDeriv (x c) slope.item) x δ _
    (fun y c => by
      show (Tensor.map (fun z => Activation.leakyReLU z slope.item) y) c = _
      simp only [Tensor.map_apply])
    (fun c => h.elim (fun h1 => by rw [h1]; exact Activation.hasDerivAt_leakyReLU_one (x c))
      fun hx => Activation.hasDerivAt_leakyReLU slope.item (hx c))
    (fun c => by
      show (Tensor.map (fun z => Activation.leakyReLUDeriv z slope.item) x * δ) c = _
      simp only [Tensor.mul_def, Tensor.zipWith_apply, Tensor.map_apply])

/-- Where the coefficient is one or no entry is zero, `elu` has its slope map as derivative, the
coefficient fixed, and its pullback is the adjoint of the derivative. This is the exact set: at a
zero entry the one-sided slopes are the coefficient and one. -/
@[op_cert TorchLean.Op.elu derivative 0]
theorem elu_backward_eq_adjoint (x δ : Tensor ℝ s) (alpha : Tensor ℝ [])
    (h : alpha.item = 1 ∨ ∀ c, x c ≠ 0) :
    HasFDerivAt (fun y => (((Op.elu (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, alpha, ())).1)
        (slopeCLM fun c => Activation.eluDeriv (x c) alpha.item) x ∧
      (((Op.elu (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, alpha, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.elu (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, alpha, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ y => Activation.elu y alpha.item)
    (fun c => Activation.eluDeriv (x c) alpha.item) x δ _
    (fun y c => by
      show (Tensor.map (fun z => Activation.elu z alpha.item) y) c = _
      simp only [Tensor.map_apply])
    (fun c => h.elim (fun h1 => by rw [h1]; exact Activation.hasDerivAt_elu_one (x c))
      fun hx => Activation.hasDerivAt_elu (x c) alpha.item (hx c))
    (fun c => by
      show (Tensor.map (fun z => Activation.eluDeriv z alpha.item) x * δ) c = _
      simp only [Tensor.mul_def, Tensor.zipWith_apply, Tensor.map_apply])

/-- Where no entry of `softplus x + ε` is zero, `safeLog` has its slope map as derivative, epsilon
fixed, and its pullback is the adjoint of the derivative. This is the exact set, since at a zero
of `softplus x + ε` the logarithm is not continuous. -/
@[op_cert TorchLean.Op.safeLog derivative 0]
theorem safeLog_backward_eq_adjoint (x δ : Tensor ℝ s) (epsilon : Tensor ℝ [])
    (h : ∀ c, Activation.softplus (x c) + epsilon.item ≠ 0) :
    HasFDerivAt (fun y => (((Op.safeLog (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, epsilon, ())).1)
        (slopeCLM fun c => Activation.safeLogDeriv (x c) epsilon.item) x ∧
      (((Op.safeLog (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, epsilon, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.safeLog (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, epsilon, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ y => Activation.safeLog y epsilon.item)
    (fun c => Activation.safeLogDeriv (x c) epsilon.item) x δ _
    (fun y c => by
      show (Tensor.map (fun z => Activation.safeLog z epsilon.item) y) c = _
      simp only [Tensor.map_apply])
    (fun c => Activation.hasDerivAt_safeLog_of_ne_zero (x c) epsilon.item (h c))
    (fun c => by
      show (Tensor.map (fun z => Activation.safeLogDeriv z epsilon.item) x * δ) c = _
      simp only [Tensor.mul_def, Tensor.zipWith_apply, Tensor.map_apply])

/-- Where no entry of `x * x + ε` is zero, `smoothAbs` has its slope map as derivative, epsilon
fixed, and its pullback is the adjoint of the derivative. This is the exact set, since at a zero
of `x * x + ε` the real forward `√(x * x + ε)` has no derivative. -/
@[op_cert TorchLean.Op.smoothAbs derivative 0]
theorem smoothAbs_backward_eq_adjoint (x δ : Tensor ℝ s) (epsilon : Tensor ℝ [])
    (h : ∀ c, x c * x c + epsilon.item ≠ 0) :
    HasFDerivAt (fun y => (((Op.smoothAbs (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, epsilon, ())).1)
        (slopeCLM fun c => Activation.smoothAbsDeriv (x c) epsilon.item) x ∧
      (((Op.smoothAbs (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, epsilon, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.smoothAbs (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, epsilon, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ y => Activation.smoothAbs y epsilon.item)
    (fun c => Activation.smoothAbsDeriv (x c) epsilon.item) x δ _
    (fun y c => by
      show (Tensor.map (fun z => Activation.smoothAbs z epsilon.item) y) c = _
      simp only [Tensor.map_apply])
    (fun c => Activation.hasDerivAt_smoothAbs_of_ne_zero (x c) epsilon.item (h c))
    (fun c => by
      show (Tensor.map (fun z => Activation.smoothAbsDeriv z epsilon.item) x * δ) c = _
      simp only [Tensor.mul_def, Tensor.zipWith_apply, Tensor.map_apply])

/-- Where `x + ε` is nonzero, with `ε` the default epsilon, `safeInv` has its slope map as
derivative, and its pullback is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.safeInv derivative]
theorem safeInv_backward_eq_adjoint (x δ : Tensor ℝ s)
    (hx : ∀ c, x c + Context.defaultEpsilon ≠ 0) :
    HasFDerivAt (fun y => (((Op.safeInv (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1)
        (slopeCLM fun c => -((x c + Context.defaultEpsilon) ^ 2)⁻¹) x ∧
      (((Op.safeInv (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.safeInv (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ z => 1 / (z + Context.defaultEpsilon))
    (fun c => -((x c + Context.defaultEpsilon) ^ 2)⁻¹) x δ _
    (fun y c => by
      show (Tensor.safeDiv (Tensor.full s (1 : ℝ)) y) c = _
      simp only [Tensor.safeDiv, Tensor.zipWith_apply, Tensor.full_apply])
    (fun c => by
      have h : HasDerivAt (fun z => (z + Context.defaultEpsilon)⁻¹)
          (-1 / (x c + Context.defaultEpsilon) ^ 2) (x c) :=
        ((hasDerivAt_id' (x c)).add_const Context.defaultEpsilon).inv (hx c)
      exact (h.congr_deriv (by ring)).congr_of_eventuallyEq
        (Filter.Eventually.of_forall fun z => one_div _))
    (fun c => by
      show ((Tensor.full s (-(1 : ℝ)) * Tensor.square (Tensor.map
        (fun y => (1 : ℝ) / (y + Context.defaultEpsilon)) x)) * δ) c = _
      simp only [Tensor.mul_def, Tensor.zipWith_apply, Tensor.full_apply, Tensor.square,
        Tensor.map_apply]
      rw [← inv_pow]
      ring)

/-- Away from both bounds, or at every input for bounds with `hi ≤ lo` where the forward is the
constant `hi`, `clamp` has its slope map as derivative, the bounds fixed, and its pullback is the
adjoint of the derivative. This is the exact set: for `lo < hi` the one-sided slopes at either
bound are zero and one. -/
@[op_cert TorchLean.Op.clamp derivative 0]
theorem clamp_backward_eq_adjoint (x δ : Tensor ℝ s) (lo hi : Tensor ℝ [])
    (h : hi.item ≤ lo.item ∨ ∀ c, x c ≠ lo.item ∧ x c ≠ hi.item) :
    HasFDerivAt (fun y => (((Op.clamp (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, lo, hi, ())).1)
        (slopeCLM fun c => if lo.item < x c ∧ x c < hi.item then 1 else 0) x ∧
      (((Op.clamp (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, lo, hi, ()) (δ, ())).1 =
        (fderiv ℝ (fun y => (((Op.clamp (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (y, lo, hi, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun _ z => Min.min hi.item (Max.max lo.item z))
    (fun c => if lo.item < x c ∧ x c < hi.item then 1 else 0) x δ _
    (fun y c => by
      show (Tensor.clamp y (Tensor.full s lo.item) (Tensor.full s hi.item) .interior) c = _
      have h : Context.clamp (y c) lo.item hi.item = Min.min hi.item (Max.max lo.item (y c)) := by
        rw [Context.clamp_real, max_comm, min_comm]
      simp only [Tensor.clamp_interior_apply, Tensor.full_apply, h]
      split_ifs <;> rfl)
    (fun c => h.elim
      (fun hle => by
        rw [ite_eq_right fun h' => (h'.1.trans h'.2).not_ge hle]
        exact (hasDerivAt_const (x c) hi.item).congr_of_eventuallyEq
          (Filter.Eventually.of_forall fun z => min_eq_left (hle.trans (le_max_left _ z))))
      fun hx => hasDerivAt_clamp (hx c).1 (hx c).2)
    (fun c => by
      show (Tensor.clampDerivative x (Tensor.full s lo.item) (Tensor.full s hi.item) * δ) c = _
      simp only [Tensor.mul_def, Tensor.zipWith_apply, Tensor.clampDerivative_apply,
        Tensor.full_apply])

/-! ## Binary operators -/

/-- `add` has the sum of the slope maps of its two inputs as derivative on their L2 product, and its
pair of pullbacks is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.add derivative]
theorem add_prod_backward_eq_adjoint (x y δ : Tensor ℝ s) :
    HasFDerivAt (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
          (((Op.add (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
        ((slopeCLM fun _ => 1).comp (WithLp.fstL 2 ℝ (Tensor ℝ s) (Tensor ℝ s)) +
          (slopeCLM fun _ => 1).comp (WithLp.sndL 2 ℝ (Tensor ℝ s) (Tensor ℝ s)))
        (WithLp.toLp 2 (x, y)) ∧
      WithLp.toLp 2
          ((((Op.add (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).1,
            (((Op.add (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).2.1) =
        (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
          (((Op.add (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
          (WithLp.toLp 2 (x, y))).adjoint δ :=
  pointwise₂_backward_eq_adjoint _ (fun _ q => q.1 + q.2) (fun _ => 1) (fun _ => 1) x y δ _ _
    (fun p c => by
      show (p.fst + p.snd) c = _
      simp only [Tensor.add_def, Tensor.zipWith_apply])
    (fun _ => (hasFDerivAt_fst.add hasFDerivAt_snd).congr_fderiv (by simp only [one_smul]))
    (fun c => (one_mul (δ c)).symm)
    (fun c => (one_mul (δ c)).symm)

/-- `sub` has the sum of the slope maps of its two inputs as derivative on their L2 product, and its
pair of pullbacks is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.sub derivative]
theorem sub_prod_backward_eq_adjoint (x y δ : Tensor ℝ s) :
    HasFDerivAt (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
          (((Op.sub (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
        ((slopeCLM fun _ => 1).comp (WithLp.fstL 2 ℝ (Tensor ℝ s) (Tensor ℝ s)) +
          (slopeCLM fun _ => -1).comp (WithLp.sndL 2 ℝ (Tensor ℝ s) (Tensor ℝ s)))
        (WithLp.toLp 2 (x, y)) ∧
      WithLp.toLp 2
          ((((Op.sub (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).1,
            (((Op.sub (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).2.1) =
        (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
          (((Op.sub (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
          (WithLp.toLp 2 (x, y))).adjoint δ :=
  pointwise₂_backward_eq_adjoint _ (fun _ q => q.1 - q.2) (fun _ => 1) (fun _ => -1) x y δ _ _
    (fun p c => by
      show (p.fst - p.snd) c = _
      simp only [Tensor.sub_def, Tensor.zipWith_apply])
    (fun _ => (hasFDerivAt_fst.sub hasFDerivAt_snd).congr_fderiv
      (ContinuousLinearMap.ext fun q => by
        simp only [_root_.sub_apply, _root_.add_apply, _root_.smul_apply,
          ContinuousLinearMap.coe_fst', ContinuousLinearMap.coe_snd', smul_eq_mul]
        ring))
    (fun c => (one_mul (δ c)).symm)
    (fun c => by
      show (Tensor.neg δ) c = _
      simp only [Tensor.neg, Tensor.map_apply]
      ring)

/-- `mul` has the sum of the slope maps of its two inputs, each the other input, as derivative on
their L2 product, and its pair of pullbacks is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.mul derivative]
theorem mul_prod_backward_eq_adjoint (x y δ : Tensor ℝ s) :
    HasFDerivAt (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
          (((Op.mul (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
        ((slopeCLM fun c => y c).comp (WithLp.fstL 2 ℝ (Tensor ℝ s) (Tensor ℝ s)) +
          (slopeCLM fun c => x c).comp (WithLp.sndL 2 ℝ (Tensor ℝ s) (Tensor ℝ s)))
        (WithLp.toLp 2 (x, y)) ∧
      WithLp.toLp 2
          ((((Op.mul (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).1,
            (((Op.mul (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).2.1) =
        (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
          (((Op.mul (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
          (WithLp.toLp 2 (x, y))).adjoint δ :=
  pointwise₂_backward_eq_adjoint _ (fun _ q => q.1 * q.2) (fun c => y c) (fun c => x c) x y δ _ _
    (fun p c => by
      show (p.fst * p.snd) c = _
      simp only [Tensor.mul_def, Tensor.zipWith_apply])
    (fun _ => (hasFDerivAt_fst.mul hasFDerivAt_snd).congr_fderiv (add_comm _ _))
    (fun c => by
      show (δ * y) c = _
      simp only [Tensor.mul_def, Tensor.zipWith_apply]
      ring)
    (fun c => by
      show (δ * x) c = _
      simp only [Tensor.mul_def, Tensor.zipWith_apply]
      ring)

/-- Where every entry of the denominator is nonzero, true division (`rounding := .none`) has the sum
of the slope maps of its two inputs as derivative on their L2 product, and its pair of pullbacks
is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.div derivative]
theorem div_prod_backward_eq_adjoint (x y δ : Tensor ℝ s) (hy : ∀ c, y c ≠ 0) :
    HasFDerivAt (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
          (((Op.div (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
        ((slopeCLM fun c => (y c)⁻¹).comp (WithLp.fstL 2 ℝ (Tensor ℝ s) (Tensor ℝ s)) +
          (slopeCLM fun c => -(x c / y c ^ 2)).comp
            (WithLp.sndL 2 ℝ (Tensor ℝ s) (Tensor ℝ s)))
        (WithLp.toLp 2 (x, y)) ∧
      WithLp.toLp 2
          ((((Op.div (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).1,
            (((Op.div (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).2.1) =
        (fderiv ℝ (fun p : WithLp 2 (Tensor ℝ s × Tensor ℝ s) =>
          (((Op.div (s := s)).rule? (α := ℝ) |>.get rfl).forward (p.fst, p.snd, ())).1)
          (WithLp.toLp 2 (x, y))).adjoint δ :=
  pointwise₂_backward_eq_adjoint _ (fun _ q => q.1 / q.2) (fun c => (y c)⁻¹)
    (fun c => -(x c / y c ^ 2)) x y δ _ _
    (fun p c => by
      show (p.fst / p.snd) c = _
      simp only [Tensor.div_def, Tensor.zipWith_apply])
    (fun c => by
      have h := (hasFDerivAt_fst (𝕜 := ℝ) (p := (x c, y c))).mul
        ((hasDerivAt_inv (hy c)).comp_hasFDerivAt (x c, y c) (hasFDerivAt_snd (𝕜 := ℝ)))
      refine (h.congr_fderiv (ContinuousLinearMap.ext fun q => ?_)).congr_of_eventuallyEq
        (Filter.Eventually.of_forall fun q => div_eq_mul_inv _ _)
      simp only [_root_.add_apply, _root_.smul_apply, ContinuousLinearMap.coe_fst',
        ContinuousLinearMap.coe_snd', smul_eq_mul, Function.comp_apply]
      ring)
    (fun c => by
      show (δ / y) c = _
      simp only [Tensor.div_def, Tensor.zipWith_apply]
      ring)
    (fun c => by
      show (Tensor.neg (δ * ((x / y) / y))) c = _
      simp only [Tensor.neg, Tensor.map_apply, Tensor.mul_def, Tensor.div_def,
        Tensor.zipWith_apply]
      ring)

/-- With its second input fixed, true division (`rounding := .none`) has its slope map as
derivative in the first input, and its pullback to the first input is the adjoint of the
derivative, for every real denominator, zero entries included (real division by zero is zero). The
rounded divisions are certified by `div_rounding_backward_eq_adjoint`. -/
@[op_cert TorchLean.Op.div derivative 0]
theorem div_backward_eq_adjoint (x y δ : Tensor ℝ s) :
    HasFDerivAt (fun z => (((Op.div (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (z, y, ())).1)
        (slopeCLM fun c => (y c)⁻¹) x ∧
      (((Op.div (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).1 =
        (fderiv ℝ (fun z => (((Op.div (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (z, y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun c z => z / y c) (fun c => (y c)⁻¹) x δ _
    (fun z c => by
      show (z / y) c = _
      simp only [Tensor.div_def, Tensor.zipWith_apply])
    (fun c => ((hasDerivAt_id' (x c)).div_const (y c)).congr_deriv (one_div _))
    (fun c => by
      show (δ / y) c = _
      simp only [Tensor.div_def, Tensor.zipWith_apply]
      ring)

/-- With `ε` the default epsilon, the shifted division `x / (y + ε)` has the slope map of
`(y + ε)⁻¹` as derivative in its first input, and its pullback is the adjoint of the derivative.
The divisor `y` is a frozen port and has no pullback. -/
@[op_cert TorchLean.Op.safeDiv derivative 0]
theorem safeDiv_backward_eq_adjoint (x y δ : Tensor ℝ s) :
    HasFDerivAt (fun z => (((Op.safeDiv (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (z, y, ())).1)
        (slopeCLM fun c => (y c + Context.defaultEpsilon)⁻¹) x ∧
      (((Op.safeDiv (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).1 =
        (fderiv ℝ (fun z => (((Op.safeDiv (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (z, y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun c z => z / (y c + Context.defaultEpsilon))
    (fun c => (y c + Context.defaultEpsilon)⁻¹) x δ _
    (fun z c => by
      show (Tensor.safeDiv z y) c = _
      simp only [Tensor.safeDiv, Tensor.zipWith_apply])
    (fun c => ((hasDerivAt_id' (x c)).div_const _).congr_deriv (one_div _))
    (fun c => by
      show (Tensor.safeDiv δ y) c = _
      simp only [Tensor.safeDiv, Tensor.zipWith_apply]
      ring)

/-- At each entry the pullback of `min` to its first input multiplies the cotangent by one where
the first input is smaller, zero where it is larger and one half at a tie. -/
theorem min_backward_apply (x y δ : Tensor ℝ s) (c : s.Coord) :
    (((Op.min (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).1 c =
      (if x c < y c then 1 else if y c < x c then 0 else 1 / 2) * δ c := by
  have hb : (((Op.min (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).1 =
      Tensor.zipWith3 (fun a b seed => if b > a then seed else if a > b then (0 : ℝ)
        else (Rat.cast (1 / 2 : Rat) : ℝ) * seed) x y δ := rfl
  rw [hb]
  simp [ite_mul]

/-- At each entry the pullback of `max` to its first input multiplies the cotangent by one where
the first input is larger, zero where it is smaller and one half at a tie. -/
theorem max_backward_apply (x y δ : Tensor ℝ s) (c : s.Coord) :
    (((Op.max (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).1 c =
      (if y c < x c then 1 else if x c < y c then 0 else 1 / 2) * δ c := by
  have hb : (((Op.max (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).1 =
      Tensor.zipWith3 (fun a b seed => if a > b then seed else if b > a then (0 : ℝ)
        else (Rat.cast (1 / 2 : Rat) : ℝ) * seed) x y δ := rfl
  rw [hb]
  simp [ite_mul]

/-- At a tie the pullback of `min` to its first input keeps half the cotangent. -/
theorem min_backward_apply_of_eq (x y δ : Tensor ℝ s) (c : s.Coord) (h : x c = y c) :
    (((Op.min (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).1 c =
      1 / 2 * δ c := by
  rw [min_backward_apply]
  simp [h]

/-- At a tie the pullback of `max` to its first input keeps half the cotangent. -/
theorem max_backward_apply_of_eq (x y δ : Tensor ℝ s) (c : s.Coord) (h : x c = y c) :
    (((Op.max (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).1 c =
      1 / 2 * δ c := by
  rw [max_backward_apply]
  simp [h]

/-- With its second input fixed, away from ties, `min` has its slope map as derivative in the first
input, and its pullback to the first input is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.min derivative 0]
theorem min_backward_eq_adjoint (x y δ : Tensor ℝ s) (hxy : ∀ c, x c ≠ y c) :
    HasFDerivAt (fun z => (((Op.min (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (z, y, ())).1)
        (slopeCLM fun c => if x c < y c then 1 else 0) x ∧
      (((Op.min (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).1 =
        (fderiv ℝ (fun z => (((Op.min (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (z, y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun c z => Min.min z (y c))
    (fun c => if x c < y c then 1 else 0) x δ _
    (fun z c => by
      show (Tensor.minimum z y) c = _
      simp only [Tensor.minimum_apply, Context.minimum_real])
    (fun c => (hasDerivAt_const_min_of_ne (hxy c).symm).congr_of_eventuallyEq
      (Filter.Eventually.of_forall fun z => min_comm z (y c)))
    (fun c => by
      rw [min_backward_apply]
      rcases (hxy c).lt_or_gt with h | h
      · rw [ite_eq_left h, ite_eq_left h]
      · rw [ite_eq_right (not_lt.mpr h.le), ite_eq_left h, ite_eq_right (not_lt.mpr h.le)])

/-- With its second input fixed, away from ties, `max` has its slope map as derivative in the first
input, and its pullback to the first input is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.max derivative 0]
theorem max_backward_eq_adjoint (x y δ : Tensor ℝ s) (hxy : ∀ c, x c ≠ y c) :
    HasFDerivAt (fun z => (((Op.max (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (z, y, ())).1)
        (slopeCLM fun c => if y c < x c then 1 else 0) x ∧
      (((Op.max (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).1 =
        (fderiv ℝ (fun z => (((Op.max (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (z, y, ())).1) x).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun c z => Max.max z (y c))
    (fun c => if y c < x c then 1 else 0) x δ _
    (fun z c => by
      show (Tensor.maximum z y) c = _
      simp only [Tensor.maximum_apply, Context.maximum_real])
    (fun c => (hasDerivAt_const_max_of_ne (hxy c).symm).congr_of_eventuallyEq
      (Filter.Eventually.of_forall fun z => max_comm z (y c)))
    (fun c => by
      rw [max_backward_apply]
      rcases (hxy c).lt_or_gt with h | h
      · rw [ite_eq_right (not_lt.mpr h.le), ite_eq_left h, ite_eq_right (not_lt.mpr h.le)]
      · rw [ite_eq_left h, ite_eq_left h])

/-- With its first input fixed, away from ties, `max` has its slope map as derivative in the second
input, and its pullback to the second input is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.max derivative 1]
theorem max_right_backward_eq_adjoint (x y δ : Tensor ℝ s) (hxy : ∀ c, x c ≠ y c) :
    HasFDerivAt (fun z => (((Op.max (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (x, z, ())).1)
        (slopeCLM fun c => if x c < y c then 1 else 0) y ∧
      (((Op.max (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).2.1 =
        (fderiv ℝ (fun z => (((Op.max (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (x, z, ())).1) y).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun c z => Max.max (x c) z)
    (fun c => if x c < y c then 1 else 0) y δ _
    (fun z c => by
      show (Tensor.maximum x z) c = _
      simp only [Tensor.maximum_apply, Context.maximum_real])
    (fun c => hasDerivAt_const_max_of_ne (hxy c))
    (fun c => by
      show (Tensor.zipWith3 _ y x δ : Tensor ℝ s) c = _
      simp only [Tensor.zipWith3_apply]
      rcases (hxy c).lt_or_gt with h | h
      · rw [ite_eq_left h, ite_eq_left h, one_mul]
      · rw [ite_eq_right (not_lt.mpr h.le), ite_eq_left h, ite_eq_right (not_lt.mpr h.le),
          zero_mul])

/-- With its first input fixed, away from ties, `min` has its slope map as derivative in the second
input, and its pullback to the second input is the adjoint of the derivative. -/
@[op_cert TorchLean.Op.min derivative 1]
theorem min_right_backward_eq_adjoint (x y δ : Tensor ℝ s) (hxy : ∀ c, x c ≠ y c) :
    HasFDerivAt (fun z => (((Op.min (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (x, z, ())).1)
        (slopeCLM fun c => if y c < x c then 1 else 0) y ∧
      (((Op.min (s := s)).rule? (α := ℝ) |>.get rfl).backward (x, y, ()) (δ, ())).2.1 =
        (fderiv ℝ (fun z => (((Op.min (s := s)).rule? (α := ℝ) |>.get rfl).forward
          (x, z, ())).1) y).adjoint δ :=
  pointwise_backward_eq_adjoint _ (fun c z => Min.min (x c) z)
    (fun c => if y c < x c then 1 else 0) y δ _
    (fun z c => by
      show (Tensor.minimum x z) c = _
      simp only [Tensor.minimum_apply, Context.minimum_real])
    (fun c => hasDerivAt_const_min_of_ne (hxy c))
    (fun c => by
      show (Tensor.zipWith3 _ y x δ : Tensor ℝ s) c = _
      simp only [Tensor.zipWith3_apply]
      rcases (hxy c).lt_or_gt with h | h
      · rw [ite_eq_right (not_lt.mpr h.le), ite_eq_left h, ite_eq_right (not_lt.mpr h.le),
          zero_mul]
      · rw [ite_eq_left h, ite_eq_left h, one_mul])

/-! ## Forward-mode rules -/

section ForwardMode

open Rep (dot)

/-- The forward-mode rule of `log` is adjoint to its pullback, at every real input. -/
@[op_cert TorchLean.Op.log algebraic]
theorem log_jvp_adjoint (x dx δ : Tensor ℝ s) :
    let rule := (Op.log (s := s)).rule? (α := ℝ) |>.get rfl
    Rep.dot ((rule.jvp?.get rfl) (x, ()) (dx, ())).1 δ =
      Rep.dot dx (rule.backward (x, ()) (δ, ())).1 := by
  intro rule
  show Rep.dot (dx / x) δ = Rep.dot dx (δ / x)
  simp only [Rep.dot, Tensor.div_def, Tensor.zipWith_apply]
  exact Finset.sum_congr rfl fun c _ => by ring

/-- The forward-mode rule of `abs` is adjoint to its pullback, at every real input. -/
@[op_cert TorchLean.Op.abs algebraic]
theorem abs_jvp_adjoint (x dx δ : Tensor ℝ s) :
    let rule := (Op.abs (s := s)).rule? (α := ℝ) |>.get rfl
    Rep.dot ((rule.jvp?.get rfl) (x, ()) (dx, ())).1 δ =
      Rep.dot dx (rule.backward (x, ()) (δ, ())).1 := by
  intro rule
  show Rep.dot (Tensor.sign x * dx) δ = Rep.dot dx (Tensor.sign x * δ)
  simp only [Rep.dot, Tensor.mul_def, Tensor.zipWith_apply]
  exact Finset.sum_congr rfl fun c _ => by ring

/-- The forward-mode rule of `reciprocal` is adjoint to its pullback, at every real input. -/
@[op_cert TorchLean.Op.reciprocal algebraic]
theorem reciprocal_jvp_adjoint (x dx δ : Tensor ℝ s) :
    let rule := (Op.reciprocal (s := s)).rule? (α := ℝ) |>.get rfl
    Rep.dot ((rule.jvp?.get rfl) (x, ()) (dx, ())).1 δ =
      Rep.dot dx (rule.backward (x, ()) (δ, ())).1 := by
  intro rule
  show Rep.dot ((Tensor.neg dx / x) / x) δ = Rep.dot dx ((Tensor.neg δ / x) / x)
  simp only [Rep.dot, Tensor.div_def, Tensor.zipWith_apply, Tensor.neg,
    Tensor.map_apply]
  exact Finset.sum_congr rfl fun c _ => by ring

/-- The forward-mode rule of `sqrt` is adjoint to its pullback, at every real input. -/
@[op_cert TorchLean.Op.sqrt algebraic]
theorem sqrt_jvp_adjoint (x dx δ : Tensor ℝ s) :
    let rule := (Op.sqrt (s := s)).rule? (α := ℝ) |>.get rfl
    Rep.dot ((rule.jvp?.get rfl) (x, ()) (dx, ())).1 δ =
      Rep.dot dx (rule.backward (x, ()) (δ, ())).1 := by
  intro rule
  show Rep.dot (Tensor.zipWith _ x dx) δ = Rep.dot dx (Tensor.zipWith _ x δ)
  simp only [Rep.dot, Tensor.zipWith_apply]
  refine Finset.sum_congr rfl fun c _ => ?_
  show dx c / (((2 : ℕ) : ℝ) * |√(x c)|) * δ c = dx c * (δ c / (((2 : ℕ) : ℝ) * |√(x c)|))
  ring

/-- The forward-mode rule of `sqrt .clampNegative` is adjoint to its pullback, at every real
input. -/
@[op_cert TorchLean.Op.sqrt algebraic]
theorem sqrt_clampNegative_jvp_adjoint (x dx δ : Tensor ℝ s) :
    let rule := (Op.sqrt (s := s) .clampNegative).rule? (α := ℝ) |>.get rfl
    Rep.dot ((rule.jvp?.get rfl) (x, ()) (dx, ())).1 δ =
      Rep.dot dx (rule.backward (x, ()) (δ, ())).1 := by
  intro rule
  show Rep.dot (Tensor.zipWith _ x dx) δ = Rep.dot dx (Tensor.zipWith _ x δ)
  simp only [Rep.dot, Tensor.zipWith_apply]
  refine Finset.sum_congr rfl fun c _ => ?_
  show (if x c > 0 then dx c / (((2 : ℕ) : ℝ) * √(x c)) else 0 * dx c) * δ c =
    dx c * (if x c > 0 then δ c / (((2 : ℕ) : ℝ) * √(x c)) else 0 * δ c)
  split_ifs <;> ring

/-- The forward-mode rule of true division is adjoint to its pair of pullbacks, at every pair of
real inputs. -/
@[op_cert TorchLean.Op.div algebraic]
theorem div_jvp_adjoint (x y dx dy δ : Tensor ℝ s) :
    let rule := (Op.div (s := s)).rule? (α := ℝ) |>.get rfl
    Rep.dot ((rule.jvp?.get rfl) (x, y, ()) (dx, dy, ())).1 δ =
      Rep.dot dx (rule.backward (x, y, ()) (δ, ())).1 +
        Rep.dot dy (rule.backward (x, y, ()) (δ, ())).2.1 := by
  intro rule
  show Rep.dot (dx / y - dy * ((x / y) / y)) δ =
    Rep.dot dx (δ / y) + Rep.dot dy (Tensor.neg (δ * ((x / y) / y)))
  simp only [Rep.dot, Tensor.div_def, Tensor.mul_def, Tensor.sub_def, Tensor.zipWith_apply,
    Tensor.neg, Tensor.map_apply, ← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl fun c _ => by ring

end ForwardMode

end TorchLean.Op
