/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.BoundOps
public import NN.Spec.Core.FloatInstances
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Interval.Outward
public import FloatLib.Numerics.Enclosure.Interval.ElementaryProof

/-!
# `BoundOps` instance for `ExecFloat.Binary 8 23`

This instance plugs FloatLib's configured binary32 directed-rounding primitives into the
IBP/CROWN endpoint propagation code.

With this, IBP code written in terms of `BoundOps` can use `α := ExecFloat.Binary 8 23` to get
float32-grid, outward-rounded interval propagation (subject to the usual finiteness preconditions).
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)
open FloatLib.Numerics (Interval)

namespace NN.MLTheory.CROWN

/-- `BoundOps` for `ExecFloat.Binary 8 23`, using the executable directed-rounding endpoint
primitives. -/
instance (priority := 1000) : BoundOps (ExecFloat.Binary 8 23) where
  addDown := Binary.addWithRounding (rounding := .towardNegativeInfinity)
  addUp := Binary.addWithRounding (rounding := .towardPositiveInfinity)
  subDown := Binary.subWithRounding (rounding := .towardNegativeInfinity)
  subUp := Binary.subWithRounding (rounding := .towardPositiveInfinity)
  mulDown := Binary.mulWithRounding (rounding := .towardNegativeInfinity)
  mulUp := Binary.mulWithRounding (rounding := .towardPositiveInfinity)

namespace IEEE32ExecBounds

/--
Decode a finite ordered interval, compute a rational enclosure of its entire real image, and
round the enclosure outward. Exceptional inputs and nonfinite rounded endpoints return `none`.
-/
def unaryBounds? (f : Interval ℚ → Option (Interval ℚ)) (lo hi : Binary 8 23) :
    Option (Binary 8 23 × Binary 8 23) := do
  let I ← (⟨lo, hi⟩ : Interval (Binary 8 23)).decode? Binary.toRat?
  if I.lo ≤ I.hi then
    let J ← f I
    let K ← Interval.encloseInterval? Binary.intervalRounding J
    pure (K.lo, K.hi)
  else
    none

/--
Keep exponential argument reduction bounded on ordered input intervals. An upper endpoint above
128 already makes a finite binary32 enclosure impossible, so return `none`. For inputs below
-128, use zero as a lower bound and the enclosure of `exp (-128)` as an upper bound. Intervals
crossing -128 retain the exponential upper bound at their original upper endpoint.

The clamp avoids repeatedly squaring enormous exact rationals for extreme finite inputs.
-/
def expRationalBounds? (I : Interval ℚ) : Option (Interval ℚ) :=
  if I.hi ≤ 128 then
    let J := Interval.expBounds ⟨max I.lo (-128), max I.hi (-128)⟩ 16
    some ⟨if I.lo < -128 then 0 else J.lo, J.hi⟩
  else
    none

/--
Preserve the nonnegative part of intervals crossing zero. The dyadic grid has enough precision
to resolve binary32 square roots even at the smallest positive input.
-/
def sqrtRationalBounds? (I : Interval ℚ) : Option (Interval ℚ) :=
  if 0 ≤ I.hi then Interval.sqrtBounds? ⟨max 0 I.lo, I.hi⟩ 100 else none

end IEEE32ExecBounds

open IEEE32ExecBounds

/--
Nonlinear enclosures computed by FloatLib's whole-interval rational kernels and checked directed
binary32 rounding. Successful results have finite endpoints. Reversed intervals, NaN, infinity,
division through zero, invalid logarithm/square-root domains, and overflow return `none`.

The containment theorems below interpret endpoints by exact finite decoding. They do not assert
a global lawful arithmetic instance for the IEEE carrier, which also contains exceptional values.
-/
instance (priority := 1000) : NonlinearBoundOps (ExecFloat.Binary 8 23) where
  divBounds aLo aHi bLo bHi :=
    if (Interval.ofBounds? Binary.toRat? aLo aHi).isSome &&
        (Interval.ofBounds? Binary.toRat? bLo bHi).isSome then
      (Interval.div? Binary.intervalRounding ⟨aLo, aHi⟩ ⟨bLo, bHi⟩).map
        fun I => (I.lo, I.hi)
    else
      none
  expBounds := unaryBounds? expRationalBounds?
  logBounds := unaryBounds? (fun I => Interval.logBounds? I 16)
  sqrtBounds := unaryBounds? sqrtRationalBounds?
  sigmoidBounds := unaryBounds? (fun _ => some ⟨0, 1⟩)
  tanhBounds := unaryBounds? (fun _ => some ⟨-1, 1⟩)
  sinBounds := unaryBounds? (fun I => some (Interval.sinBounds I 16))
  cosBounds := unaryBounds? (fun I => some (Interval.cosBounds I 16))
  layerNormAbsBound := fun _ => none
  supportsIdealCoupledDerivatives := false

private theorem containsReal_unaryBounds? {f : Interval ℚ → Option (Interval ℚ)}
    {g : ℝ → ℝ}
    (hf : ∀ {I J : Interval ℚ}, f I = some J → ∀ {x : ℝ},
      I.ContainsReal some x → J.ContainsReal some (g x))
    {lo hi outLo outHi : Binary 8 23}
    (h : unaryBounds? f lo hi = some (outLo, outHi)) {x : ℝ}
    (hx : (⟨lo, hi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? x) :
    (⟨outLo, outHi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? (g x) := by
  cases hI : (⟨lo, hi⟩ : Interval (Binary 8 23)).decode? Binary.toRat? with
  | none => simp [unaryBounds?, hI] at h
  | some I =>
    by_cases horder : I.lo ≤ I.hi
    · cases hJ : f I with
      | none => simp [unaryBounds?, hI, horder, hJ] at h
      | some J =>
        cases hK : (Interval.encloseInterval? Binary.intervalRounding J :
            Option (Interval (Binary 8 23))) with
        | none => simp [unaryBounds?, hI, horder, hJ, hK] at h
        | some K =>
          obtain ⟨rfl, rfl⟩ : K.lo = outLo ∧ K.hi = outHi := by
            simpa [unaryBounds?, hI, horder, hJ, hK] using h
          apply Interval.containsReal_encloseInterval? Binary.intervalRounding hK
          exact (Interval.containsReal_some_iff _ _).1
            (hf hJ ((Interval.containsReal_some_iff _ _).2
              ((Interval.containsReal_iff_of_decode? hI x).1 hx)))
    · simp [unaryBounds?, hI, horder] at h

private theorem containsReal_expRationalBounds? {I J : Interval ℚ}
    (h : expRationalBounds? I = some J) {x : ℝ} (hx : I.ContainsReal some x) :
    J.ContainsReal some (Real.exp x) := by
  unfold expRationalBounds? at h
  split at h
  · cases h
    have hx' : (⟨max I.lo (-128), max I.hi (-128)⟩ : Interval ℚ).ContainsReal
        some (max x (-128)) := by
      obtain ⟨hl, hu⟩ := (Interval.containsReal_some_iff _ _).1 hx
      apply (Interval.containsReal_some_iff _ _).2
      simpa using And.intro (max_le_max hl (le_refl (-128 : ℝ)))
        (max_le_max hu (le_refl (-128 : ℝ)))
    obtain ⟨hl, hu⟩ := (Interval.containsReal_some_iff _ _).1
      (Interval.containsReal_expBounds _ 16 hx')
    apply (Interval.containsReal_some_iff _ _).2
    refine ⟨?_, (Real.exp_le_exp.mpr (le_max_left _ _)).trans hu⟩
    dsimp only
    split_ifs with hlo
    · simpa using (Real.exp_pos x).le
    · have hlow : (-128 : ℝ) ≤ (I.lo : ℝ) := by exact_mod_cast le_of_not_gt hlo
      have hlowx := hlow.trans ((Interval.containsReal_some_iff _ _).1 hx).1
      simpa [max_eq_left hlowx] using hl
  · contradiction

private theorem containsReal_sqrtRationalBounds? {I J : Interval ℚ}
    (h : sqrtRationalBounds? I = some J) {x : ℝ} (hx : I.ContainsReal some x) :
    J.ContainsReal some (Real.sqrt x) := by
  unfold sqrtRationalBounds? at h
  split at h
  · rename_i hnonneg
    have hx' : (⟨max 0 I.lo, I.hi⟩ : Interval ℚ).ContainsReal some (max 0 x) := by
      obtain ⟨hl, hu⟩ := (Interval.containsReal_some_iff _ _).1 hx
      apply (Interval.containsReal_some_iff _ _).2
      have hhi : (0 : ℝ) ≤ (I.hi : ℝ) := by exact_mod_cast hnonneg
      simpa using And.intro (max_le_max (le_refl (0 : ℝ)) hl) (max_le hhi hu)
    have hb := Interval.containsReal_sqrtBounds? h hx'
    by_cases hx0 : 0 ≤ x
    · simpa [max_eq_right hx0] using hb
    · have hx0' : x ≤ 0 := le_of_not_ge hx0
      simpa [max_eq_left hx0', Real.sqrt_eq_zero_of_nonpos hx0'] using hb
  · contradiction

namespace IEEE32ExecBounds

/-- The actual division backend encloses every quotient of real interval members. -/
theorem divBounds_containsReal {aLo aHi bLo bHi outLo outHi : Binary 8 23}
    (h : NonlinearBoundOps.divBounds aLo aHi bLo bHi = some (outLo, outHi))
    {x y : ℝ}
    (hx : (⟨aLo, aHi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? x)
    (hy : (⟨bLo, bHi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? y) :
    (⟨outLo, outHi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? (x / y) := by
  change (if _ then _ else none) = some (outLo, outHi) at h
  split at h
  · cases hK : Interval.div? Binary.intervalRounding
        (⟨aLo, aHi⟩ : Interval (Binary 8 23)) ⟨bLo, bHi⟩ with
    | none => simp [hK] at h
    | some K =>
      obtain ⟨rfl, rfl⟩ : K.lo = outLo ∧ K.hi = outHi := by simpa [hK] using h
      exact Interval.containsReal_div? Binary.intervalRounding hK hx hy
  · contradiction

/-- The actual exponential backend encloses the entire real image, including irrational values. -/
theorem expBounds_containsReal {lo hi outLo outHi : Binary 8 23}
    (h : NonlinearBoundOps.expBounds lo hi = some (outLo, outHi)) {x : ℝ}
    (hx : (⟨lo, hi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? x) :
    (⟨outLo, outHi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? (Real.exp x) :=
  containsReal_unaryBounds?
    (fun {_I _J} h {_x} hx => containsReal_expRationalBounds? h hx) h hx

/-- The actual logarithm backend encloses every real member of a positive interval. -/
theorem logBounds_containsReal {lo hi outLo outHi : Binary 8 23}
    (h : NonlinearBoundOps.logBounds lo hi = some (outLo, outHi)) {x : ℝ}
    (hx : (⟨lo, hi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? x) :
    (⟨outLo, outHi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? (Real.log x) :=
  containsReal_unaryBounds?
    (fun {_I _J} h {_x} hx => Interval.containsReal_logBounds? h hx) h hx

/-- Square-root containment preserves the backend's clamping convention across zero. -/
theorem sqrtBounds_containsReal {lo hi outLo outHi : Binary 8 23}
    (h : NonlinearBoundOps.sqrtBounds lo hi = some (outLo, outHi)) {x : ℝ}
    (hx : (⟨lo, hi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? x) :
    (⟨outLo, outHi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? (Real.sqrt x) :=
  containsReal_unaryBounds?
    (fun {_I _J} h {_x} hx => containsReal_sqrtRationalBounds? h hx) h hx

/-- The finite sigmoid fallback encloses the real logistic function. -/
theorem sigmoidBounds_containsReal {lo hi outLo outHi : Binary 8 23}
    (h : NonlinearBoundOps.sigmoidBounds lo hi = some (outLo, outHi)) {x : ℝ}
    (hx : (⟨lo, hi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? x) :
    (⟨outLo, outHi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat?
      (1 / (1 + Real.exp (-x))) := by
  apply containsReal_unaryBounds? (f := fun _ => some ⟨0, 1⟩)
    (g := fun x => 1 / (1 + Real.exp (-x))) ?_ h hx
  intro I J hJ x _
  cases hJ
  apply (Interval.containsReal_some_iff _ _).2
  have hpos : 0 < 1 + Real.exp (-x) := by positivity
  have hu : 1 / (1 + Real.exp (-x)) ≤ (1 : ℝ) := by
    apply (div_le_iff₀ hpos).2
    simpa using (le_add_of_nonneg_right (Real.exp_pos (-x)).le :
      (1 : ℝ) ≤ 1 + Real.exp (-x))
  simpa using And.intro (one_div_pos.mpr hpos).le hu

/-- The finite tanh fallback encloses every real hyperbolic tangent. -/
theorem tanhBounds_containsReal {lo hi outLo outHi : Binary 8 23}
    (h : NonlinearBoundOps.tanhBounds lo hi = some (outLo, outHi)) {x : ℝ}
    (hx : (⟨lo, hi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? x) :
    (⟨outLo, outHi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? (Real.tanh x) := by
  apply containsReal_unaryBounds? (f := fun _ => some ⟨-1, 1⟩) ?_ h hx
  intro I J hJ x _
  cases hJ
  apply (Interval.containsReal_some_iff _ _).2
  simpa using And.intro (Real.neg_one_lt_tanh x).le (Real.tanh_lt_one x).le

/-- Sine containment includes extrema in the interior of the input interval. -/
theorem sinBounds_containsReal {lo hi outLo outHi : Binary 8 23}
    (h : NonlinearBoundOps.sinBounds lo hi = some (outLo, outHi)) {x : ℝ}
    (hx : (⟨lo, hi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? x) :
    (⟨outLo, outHi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? (Real.sin x) := by
  apply containsReal_unaryBounds? (f := fun I => some (Interval.sinBounds I 16)) ?_ h hx
  intro I J hJ x hx
  cases hJ
  exact Interval.containsReal_sinBounds I 16 hx

/-- Cosine containment uses the entire interval rather than just endpoint values. -/
theorem cosBounds_containsReal {lo hi outLo outHi : Binary 8 23}
    (h : NonlinearBoundOps.cosBounds lo hi = some (outLo, outHi)) {x : ℝ}
    (hx : (⟨lo, hi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? x) :
    (⟨outLo, outHi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? (Real.cos x) := by
  apply containsReal_unaryBounds? (f := fun I => some (Interval.cosBounds I 16)) ?_ h hx
  intro I J hJ x hx
  cases hJ
  exact Interval.containsReal_cosBounds I 16 hx

end IEEE32ExecBounds

end NN.MLTheory.CROWN
