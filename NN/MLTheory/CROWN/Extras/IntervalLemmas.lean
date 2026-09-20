/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import Mathlib.Analysis.SpecialFunctions.Sigmoid
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
public import FloatLib.Floats.Interval.RealBounds
public import NN.MLTheory.CROWN.BoundOps.Lawful

/-!
# Interval arithmetic lemmas (ℝ)

This file provides foundational lemmas for interval arithmetic operations
used in CROWN/LiRPA bound propagation soundness proofs.

It is primarily intended as a small toolbox for proof scripts and examples; it lives under
`NN/MLTheory/CROWN/Extras/` to keep the main CROWN modules focused on the bound propagation API.

-/

@[expose] public section

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq


namespace NN.MLTheory.CROWN.IntervalLemmas

/-! ### Monotone nonlinear functions -/

/-- The real sigmoid written in the form used by `NonlinearBoundOps`. -/
noncomputable def realSigmoid (x : ℝ) : ℝ :=
  1 / (1 + Real.exp (-x))

/-- The real sigmoid is monotone. -/
theorem monotone_realSigmoid : Monotone realSigmoid := by
  intro a b hab
  simpa [realSigmoid, Real.sigmoid, div_eq_mul_inv] using Real.sigmoid_monotone hab

/-- Derivative of real hyperbolic tangent. -/
theorem hasDerivAt_real_tanh (x : ℝ) :
    HasDerivAt Real.tanh (1 / (Real.cosh x) ^ 2) x := by
  have hdiv :
      HasDerivAt (Real.sinh * Real.cosh⁻¹)
        ((Real.cosh x * Real.cosh x - Real.sinh x * Real.sinh x) /
          (Real.cosh x) ^ 2) x := by
    simpa [div_eq_mul_inv] using
      (Real.hasDerivAt_sinh x).div (Real.hasDerivAt_cosh x) (Real.cosh_pos x).ne'
  have htanh :
      HasDerivAt Real.tanh
        ((Real.cosh x * Real.cosh x - Real.sinh x * Real.sinh x) /
          (Real.cosh x) ^ 2) x := by
    convert hdiv using 1
    funext y
    simp [Real.tanh_eq_sinh_div_cosh, div_eq_mul_inv]
  have hIdentity : Real.cosh x * Real.cosh x - Real.sinh x * Real.sinh x = 1 := by
    simpa [pow_two, mul_assoc, mul_left_comm, mul_comm] using Real.cosh_sq_sub_sinh_sq x
  simpa [hIdentity, div_eq_mul_inv, one_div, pow_two, mul_assoc, mul_left_comm, mul_comm]
    using htanh

/-- The real hyperbolic tangent is strictly monotone. -/
theorem strictMono_real_tanh : StrictMono Real.tanh := by
  refine strictMono_of_deriv_pos fun x ↦ ?_
  rw [(hasDerivAt_real_tanh x).deriv]
  exact one_div_pos.mpr (sq_pos_of_pos (Real.cosh_pos x))

/-- The real hyperbolic tangent is monotone. -/
theorem monotone_real_tanh : Monotone Real.tanh :=
  strictMono_real_tanh.monotone

/-! ### Basic Interval Membership -/

/-- A value is in an interval [lo, hi] -/
def inInterval (x lo hi : ℝ) : Prop := lo ≤ x ∧ x ≤ hi

/-- Every value lies in the degenerate interval `[x, x]`. -/
theorem inInterval_refl (x : ℝ) : inInterval x x x :=
  ⟨le_refl x, le_refl x⟩

/-- Introduction rule: two inequalities give membership. Having this as a named lemma keeps the
downstream proofs from unfolding `inInterval` just to build an `And`. -/
theorem inInterval_of_bounds {x lo hi : ℝ}
    (hlo : lo ≤ x) (hhi : x ≤ hi) : inInterval x lo hi := ⟨hlo, hhi⟩

/-! ### Addition Interval Soundness -/

/-- If $x\in[a,b]$ and $y\in[c,d]$, then $x+y\in[a+c,b+d]$. -/
theorem interval_add_sound {x y a b c d : ℝ}
    (hx : inInterval x a b) (hy : inInterval y c d) :
    inInterval (x + y) (a + c) (b + d) := by
  constructor
  · exact add_le_add hx.1 hy.1
  · exact add_le_add hx.2 hy.2

/-! ### Subtraction Interval Soundness -/

/-- If $x\in[a,b]$ and $y\in[c,d]$, then $x-y\in[a-d,b-c]$. -/
theorem interval_sub_sound {x y a b c d : ℝ}
    (hx : inInterval x a b) (hy : inInterval y c d) :
    inInterval (x - y) (a - d) (b - c) := by
  constructor
  · exact sub_le_sub hx.1 hy.2
  · exact sub_le_sub hx.2 hy.1

/-! ### ReLU Interval Soundness -/

/-- Real-valued ReLU used by the interval soundness lemmas. -/
def relu (x : ℝ) : ℝ := max 0 x

/-- Monotonicity of ReLU on real inputs. -/
theorem relu_monotone {x y : ℝ} (h : x ≤ y) : relu x ≤ relu y := by
  unfold relu
  exact max_le_max_left 0 h

/-- ReLU outputs are always nonnegative. -/
theorem relu_nonneg (x : ℝ) : 0 ≤ relu x := by
  unfold relu
  exact le_max_left 0 x

/-- On the nonpositive branch, ReLU evaluates to zero. -/
theorem relu_of_nonpos {x : ℝ} (h : x ≤ 0) : relu x = 0 := by
  unfold relu
  exact max_eq_left h

/-- On the nonnegative branch, ReLU is the identity function. -/
theorem relu_of_nonneg {x : ℝ} (h : 0 ≤ x) : relu x = x := by
  unfold relu
  exact max_eq_right h

/-- ReLU maps an input interval `[l,u]` into `[max 0 l, max 0 u]`. -/
theorem interval_relu_sound {x l u : ℝ} (h : inInterval x l u) :
    inInterval (relu x) (max 0 l) (max 0 u) := by
  unfold inInterval relu
  constructor
  · exact max_le_max_left 0 h.1
  · exact max_le_max_left 0 h.2

/-! ### Square Interval Soundness -/

/-- Squaring function used by interval propagation lemmas. -/
def square (x : ℝ) : ℝ := x * x

/-- Squares over the reals are nonnegative. -/
theorem square_nonneg (x : ℝ) : 0 ≤ square x := mul_self_nonneg x

/-- If $0\le a\le b$, then $a^2\le b^2$. -/
theorem square_le_square_of_nonneg {a b : ℝ} (ha : 0 ≤ a) (hab : a ≤ b) :
    square a ≤ square b := mul_self_le_mul_self ha hab

/-- Lower endpoint for the range of `x ↦ x^2` over an interval. -/
noncomputable def intervalSquareMin (l u : ℝ) : ℝ :=
  if l < 0 then
    if 0 < u then 0
    else min (l * l) (u * u)
  else
    l * l

/-- Maximum square in an interval -/
noncomputable def intervalSquareMax (l u : ℝ) : ℝ := max (l * l) (u * u)

/-- Square interval soundness: if $x\in[l,u]$, then
$x^2\in[\mathrm{minSq},\mathrm{maxSq}]$. -/
theorem interval_square_sound {x l u : ℝ} (h : inInterval x l u) :
    inInterval (square x) (intervalSquareMin l u) (intervalSquareMax l u) := by
  unfold square intervalSquareMin intervalSquareMax inInterval
  constructor
  · split_ifs with hl_neg hu_pos
    · exact mul_self_nonneg x
    · have hx_nonpos : x ≤ 0 := le_trans h.2 (le_of_not_gt hu_pos)
      have hu_neg : u ≤ 0 := le_of_not_gt hu_pos
      have hu_sq_le : u * u ≤ x * x := by
        have h1 : 0 ≤ -u := neg_nonneg.mpr hu_neg
        have h2 : -u ≤ -x := neg_le_neg h.2
        calc u * u = (-u) * (-u) := by ring
             _ ≤ (-x) * (-x) := mul_self_le_mul_self h1 h2
             _ = x * x := by ring
      exact min_le_of_right_le hu_sq_le
    · have hl_nonneg : 0 ≤ l := le_of_not_gt hl_neg
      exact mul_self_le_mul_self hl_nonneg h.1
  · by_cases hx_nonneg : 0 ≤ x
    · have hu_nonneg : 0 ≤ u := le_trans hx_nonneg h.2
      exact le_max_of_le_right (mul_self_le_mul_self hx_nonneg h.2)
    · push Not at hx_nonneg
      have hx_nonpos : x ≤ 0 := le_of_lt hx_nonneg
      have habs : -x ≤ -l := neg_le_neg h.1
      have hx_sq_le_l_sq : x * x ≤ l * l := by
        have h1 : 0 ≤ -x := neg_nonneg.mpr hx_nonpos
        calc x * x = (-x) * (-x) := by ring
             _ ≤ (-l) * (-l) := mul_self_le_mul_self h1 habs
             _ = l * l := by ring
      exact le_max_of_le_left hx_sq_le_l_sq

/-! ### Negation Interval -/

/-- Negation flips and swaps the interval: $-[a,b]=[-b,-a]$. -/
theorem interval_neg_sound {x a b : ℝ} (h : inInterval x a b) :
    inInterval (-x) (-b) (-a) := by
  constructor
  · exact neg_le_neg h.2
  · exact neg_le_neg h.1

/-! ### Absolute Value Interval -/

/-- If $x\in[l,u]$, then $|x|$ is bounded. -/
theorem interval_abs_sound {x l u : ℝ} (h : inInterval x l u) :
    0 ≤ |x| ∧ |x| ≤ max |l| |u| := by
  constructor
  · exact abs_nonneg x
  · by_cases hx : 0 ≤ x
    · have : |x| = x := abs_of_nonneg hx
      rw [this]
      calc x ≤ u := h.2
           _ ≤ |u| := le_abs_self u
           _ ≤ max |l| |u| := le_max_right _ _
    · push Not at hx
      have : |x| = -x := abs_of_neg hx
      rw [this]
      calc -x ≤ -l := neg_le_neg h.1
           _ ≤ |l| := neg_le_abs l
           _ ≤ max |l| |u| := le_max_left _ _

/-! ### Directed endpoint arithmetic -/

section Directed

variable {α : Type} [TorchLean.Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

/-- Exact real meaning of an endpoint supplied by its lawful directed-arithmetic instance. -/
abbrev semanticValue (x : α) : ℝ := LawfulBoundOps.toReal x

/-- `BoundOps.min2` selects the smaller endpoint in the mathematical interpretation. -/
theorem value_min2 (a b : α) :
    semanticValue (BoundOps.min2 a b) = min (semanticValue a) (semanticValue b) := by
  by_cases h : a > b
  · have hv : semanticValue b ≤ semanticValue a :=
      le_of_lt ((LawfulBoundOps.lt_iff b a).mp h)
    simp [BoundOps.min2, h, min_eq_right hv]
  · have hv : semanticValue a ≤ semanticValue b := by
      apply le_of_not_gt
      intro hba
      exact h ((LawfulBoundOps.lt_iff b a).mpr hba)
    simp [BoundOps.min2, h, min_eq_left hv]

/-- `BoundOps.max2` selects the larger endpoint in the mathematical interpretation. -/
theorem value_max2 (a b : α) :
    semanticValue (BoundOps.max2 a b) = max (semanticValue a) (semanticValue b) := by
  by_cases h : a > b
  · have hv : semanticValue b ≤ semanticValue a :=
      le_of_lt ((LawfulBoundOps.lt_iff b a).mp h)
    simp [BoundOps.max2, h, max_eq_left hv]
  · have hv : semanticValue a ≤ semanticValue b := by
      apply le_of_not_gt
      intro hba
      exact h ((LawfulBoundOps.lt_iff b a).mpr hba)
    simp [BoundOps.max2, h, max_eq_right hv]

/-- Directed endpoint addition encloses exact real addition. -/
theorem directed_add_sound {x y a b c d : α}
    (hx : inInterval (semanticValue x) (semanticValue a) (semanticValue b))
    (hy : inInterval (semanticValue y) (semanticValue c) (semanticValue d)) :
    inInterval (semanticValue x + semanticValue y)
      (semanticValue (BoundOps.addDown a c)) (semanticValue (BoundOps.addUp b d)) := by
  constructor
  · exact (LawfulBoundOps.addDown_le a c).trans (add_le_add hx.1 hy.1)
  · exact (add_le_add hx.2 hy.2).trans (LawfulBoundOps.le_addUp b d)

/-- Directed endpoint subtraction encloses exact real subtraction. -/
theorem directed_sub_sound {x y a b c d : α}
    (hx : inInterval (semanticValue x) (semanticValue a) (semanticValue b))
    (hy : inInterval (semanticValue y) (semanticValue c) (semanticValue d)) :
    inInterval (semanticValue x - semanticValue y)
      (semanticValue (BoundOps.subDown a d)) (semanticValue (BoundOps.subUp b c)) := by
  constructor
  · exact (LawfulBoundOps.subDown_le a d).trans (sub_le_sub hx.1 hy.2)
  · exact (sub_le_sub hx.2 hy.1).trans (LawfulBoundOps.le_subUp b c)

/--
The outward-rounded four-corner rule encloses exact real multiplication.

This is the scalar fact needed by rounded IBP and backward CROWN. It depends only on the declared
endpoint interpretation and directed-operation laws; it does not assume that rounded scalars form a
ring or that reassociation is exact.
-/
theorem directed_mul_sound {x y a b c d : α}
    (hx : inInterval (semanticValue x) (semanticValue a) (semanticValue b))
    (hy : inInterval (semanticValue y) (semanticValue c) (semanticValue d)) :
    inInterval (semanticValue x * semanticValue y)
      (semanticValue (BoundOps.min2
        (BoundOps.min2 (BoundOps.mulDown a c) (BoundOps.mulDown a d))
        (BoundOps.min2 (BoundOps.mulDown b c) (BoundOps.mulDown b d))))
      (semanticValue (BoundOps.max2
        (BoundOps.max2 (BoundOps.mulUp a c) (BoundOps.mulUp a d))
        (BoundOps.max2 (BoundOps.mulUp b c) (BoundOps.mulUp b d)))) := by
  have hExact := FloatLib.Floats.Interval.mul_bounds_Icc
    (semanticValue a) (semanticValue b) (semanticValue c) (semanticValue d)
    (semanticValue x) (semanticValue y) hx hy
  constructor
  · rw [value_min2, value_min2, value_min2]
    calc
      min
          (min (semanticValue (BoundOps.mulDown a c))
            (semanticValue (BoundOps.mulDown a d)))
          (min (semanticValue (BoundOps.mulDown b c))
            (semanticValue (BoundOps.mulDown b d))) ≤
          min
            (min (semanticValue a * semanticValue c) (semanticValue a * semanticValue d))
            (min (semanticValue b * semanticValue c) (semanticValue b * semanticValue d)) :=
        min_le_min
          (min_le_min (LawfulBoundOps.mulDown_le a c) (LawfulBoundOps.mulDown_le a d))
          (min_le_min (LawfulBoundOps.mulDown_le b c) (LawfulBoundOps.mulDown_le b d))
      _ ≤ semanticValue x * semanticValue y := by
        simpa [FloatLib.Floats.Interval.minOfFour] using hExact.1
  · rw [value_max2, value_max2, value_max2]
    refine (show semanticValue x * semanticValue y ≤
      max (max (semanticValue a * semanticValue c) (semanticValue a * semanticValue d))
        (max (semanticValue b * semanticValue c) (semanticValue b * semanticValue d)) by
        simpa [FloatLib.Floats.Interval.maxOfFour] using hExact.2).trans
      (max_le_max (max_le_max ?_ ?_) (max_le_max ?_ ?_))
    · exact LawfulBoundOps.le_mulUp a c
    · exact LawfulBoundOps.le_mulUp a d
    · exact LawfulBoundOps.le_mulUp b c
    · exact LawfulBoundOps.le_mulUp b d

end Directed

/-! ### Nonlinear transfer laws over the reals -/

private theorem unaryEnclosure_of_monotone (f : ℝ → ℝ) (hf : Monotone f) :
    UnaryEnclosure (α := ℝ) f (fun lo hi ↦ some (f lo, f hi)) := by
  intro lo hi outLo outHi x hout hxLo hxHi
  have hpair : outLo = f lo ∧ outHi = f hi := by
    simpa using Option.some.inj hout.symm
  rcases hpair with ⟨rfl, rfl⟩
  exact ⟨hf hxLo, hf hxHi⟩

private theorem unaryEnclosure_of_unit_range (f : ℝ → ℝ)
    (hf : ∀ x, -1 ≤ f x ∧ f x ≤ 1) :
    UnaryEnclosure (α := ℝ) f (fun _ _ ↦ some (-1, 1)) := by
  intro lo hi outLo outHi x hout _ _
  have hpair : outLo = -1 ∧ outHi = 1 := by
    simpa using Option.some.inj hout.symm
  rcases hpair with ⟨rfl, rfl⟩
  exact hf x

/-- Exact real nonlinear transfers satisfy their mathematical interval contracts. -/
noncomputable instance instLawfulNonlinearBoundOpsReal : LawfulNonlinearBoundOps ℝ where
  divBounds_enclosure := by
    intro aLo aHi bLo bHi outLo outHi x y hout hxLo hxHi hyLo hyHi
    change
      (if bLo > 0 || 0 > bHi then
        some
          (min (min (aLo / bLo) (aLo / bHi)) (min (aHi / bLo) (aHi / bHi)),
            max (max (aLo / bLo) (aLo / bHi)) (max (aHi / bLo) (aHi / bHi)))
      else none) = some (outLo, outHi) at hout
    split at hout
    next hAvoidsZero =>
      have hpair :
          outLo = min (min (aLo / bLo) (aLo / bHi)) (min (aHi / bLo) (aHi / bHi)) ∧
          outHi = max (max (aLo / bLo) (aLo / bHi)) (max (aHi / bLo) (aHi / bHi)) := by
        simpa using Option.some.inj hout.symm
      rcases hpair with ⟨rfl, rfl⟩
      have hside : bHi < 0 ∨ 0 < bLo := by
        have hz : 0 < bLo ∨ bHi < 0 := by
          simpa using hAvoidsZero
        exact hz.elim Or.inr Or.inl
      have hExact := FloatLib.Floats.Interval.div_bounds_Icc
        aLo aHi bLo bHi x y ⟨hxLo, hxHi⟩ ⟨hyLo, hyHi⟩ hside
      change
        min (min (aLo / bLo) (aLo / bHi)) (min (aHi / bLo) (aHi / bHi)) ≤ x / y ∧
          x / y ≤ max (max (aLo / bLo) (aLo / bHi)) (max (aHi / bLo) (aHi / bHi))
      simpa only [Set.mem_Icc, FloatLib.Floats.Interval.minOfFour,
        FloatLib.Floats.Interval.maxOfFour] using hExact
    next hIncludesZero => simp at hout
  expBounds_enclosure := by
    change UnaryEnclosure (α := ℝ) Real.exp (fun lo hi ↦ some (Real.exp lo, Real.exp hi))
    exact unaryEnclosure_of_monotone Real.exp Real.exp_monotone
  logBounds_enclosure := by
    intro lo hi outLo outHi x hout hxLo hxHi
    change (if lo > 0 then some (Real.log lo, Real.log hi) else none) =
      some (outLo, outHi) at hout
    split at hout
    next hlo =>
      have hpair : outLo = Real.log lo ∧ outHi = Real.log hi := by
        simpa using Option.some.inj hout.symm
      rcases hpair with ⟨rfl, rfl⟩
      exact ⟨Real.log_le_log hlo hxLo, Real.log_le_log (hlo.trans_le hxLo) hxHi⟩
    next hnlo => simp at hout
  sqrtBounds_enclosure := by
    intro lo hi outLo outHi x hout hxLo hxHi
    change (if hi < 0 then none else some (Real.sqrt (max lo 0), Real.sqrt hi)) =
      some (outLo, outHi) at hout
    split at hout
    next hhi => simp at hout
    next hnhi =>
      have hpair : outLo = Real.sqrt (max lo 0) ∧ outHi = Real.sqrt hi := by
        simpa using Option.some.inj hout.symm
      rcases hpair with ⟨rfl, rfl⟩
      have hLower : Real.sqrt (max lo 0) = Real.sqrt lo := by
        by_cases hlo : lo ≤ 0
        · simp [max_eq_right hlo, Real.sqrt_eq_zero_of_nonpos hlo]
        · simp [max_eq_left (le_of_not_ge hlo)]
      rw [hLower]
      exact ⟨Real.sqrt_le_sqrt hxLo, Real.sqrt_le_sqrt hxHi⟩
  sigmoidBounds_enclosure := by
    change UnaryEnclosure (α := ℝ) realSigmoid
      (fun lo hi ↦ some (realSigmoid lo, realSigmoid hi))
    exact unaryEnclosure_of_monotone realSigmoid monotone_realSigmoid
  tanhBounds_enclosure := by
    change UnaryEnclosure (α := ℝ) Real.tanh (fun lo hi ↦ some (Real.tanh lo, Real.tanh hi))
    exact unaryEnclosure_of_monotone Real.tanh monotone_real_tanh
  sinBounds_enclosure := by
    change UnaryEnclosure (α := ℝ) Real.sin (fun _ _ ↦ some (-1, 1))
    exact unaryEnclosure_of_unit_range Real.sin fun x ↦ ⟨Real.neg_one_le_sin x, Real.sin_le_one x⟩
  cosBounds_enclosure := by
    change UnaryEnclosure (α := ℝ) Real.cos (fun _ _ ↦ some (-1, 1))
    exact unaryEnclosure_of_unit_range Real.cos fun x ↦ ⟨Real.neg_one_le_cos x, Real.cos_le_one x⟩
  layerNormAbsBound_sound := by
    intro n radius hout
    change some (Real.sqrt n) = some radius at hout
    change Real.sqrt n ≤ radius
    exact le_of_eq (Option.some.inj hout)
  coupledDerivatives_exact := by
    intro _
    rfl

end NN.MLTheory.CROWN.IntervalLemmas
