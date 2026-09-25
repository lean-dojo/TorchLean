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
public import NN.MLTheory.CROWN.Graph.Engine.Base

/-!
# Interval arithmetic lemmas

Scalar facts shared by the CROWN soundness proofs: monotonicity of the real sigmoid and `tanh`,
the real reading of `BoundOps.min2`/`BoundOps.max2`, the enclosure law for the four-corner product
`Graph.intervalMul`, and the exact real instance of `LawfulNonlinearBoundOps`.
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

/-! ### Directed endpoint arithmetic -/

section Directed

variable {α : Type} [TorchLean.Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/-- `BoundOps.min2` selects the smaller endpoint in the mathematical interpretation. -/
theorem value_min2 (a b : α) :
    value (BoundOps.min2 a b) = min (value a) (value b) := by
  by_cases h : b < a
  · have hreal := (LawfulBoundOps.lt_iff b a).mp h
    simp [BoundOps.min2, h, min_eq_right hreal.le]
  · have hreal : value a ≤ value b :=
      le_of_not_gt fun hlt => h ((LawfulBoundOps.lt_iff b a).mpr hlt)
    simp [BoundOps.min2, h, min_eq_left hreal]

/-- `BoundOps.max2` selects the larger endpoint in the mathematical interpretation. -/
theorem value_max2 (a b : α) :
    value (BoundOps.max2 a b) = max (value a) (value b) := by
  by_cases h : b < a
  · have hreal := (LawfulBoundOps.lt_iff b a).mp h
    simp [BoundOps.max2, h, max_eq_left hreal.le]
  · have hreal : value a ≤ value b :=
      le_of_not_gt fun hlt => h ((LawfulBoundOps.lt_iff b a).mpr hlt)
    simp [BoundOps.max2, h, max_eq_right hreal]

/--
The outward-rounded four-corner product `Graph.intervalMul` encloses every product of enclosed
reals.

It depends only on the endpoint interpretation and the directed-operation laws; it does not assume
that rounded scalars form a ring or that reassociation is exact. The real four-corner bound is
FloatLib's `mul_bounds_Icc`.
-/
theorem intervalMul_encloses {al au bl bu : α} {x y : ℝ}
    (hal : value al ≤ x) (hau : x ≤ value au)
    (hbl : value bl ≤ y) (hbu : y ≤ value bu) :
    value (Graph.intervalMul al au bl bu).1 ≤ x * y ∧
      x * y ≤ value (Graph.intervalMul al au bl bu).2 := by
  have h := FloatLib.Floats.Interval.mul_bounds_Icc _ _ _ _ x y ⟨hal, hau⟩ ⟨hbl, hbu⟩
  simp only [Graph.intervalMul, value_min2, value_max2]
  exact ⟨(min_le_min
      (min_le_min (LawfulBoundOps.mulDown_le al bl) (LawfulBoundOps.mulDown_le al bu))
      (min_le_min (LawfulBoundOps.mulDown_le au bl) (LawfulBoundOps.mulDown_le au bu))).trans h.1,
    h.2.trans (max_le_max
      (max_le_max (LawfulBoundOps.le_mulUp al bl) (LawfulBoundOps.le_mulUp al bu))
      (max_le_max (LawfulBoundOps.le_mulUp au bl) (LawfulBoundOps.le_mulUp au bu)))⟩

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
