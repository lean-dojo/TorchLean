/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.SpecialFunctions.Sigmoid
public import TorchLean.Verify.CROWN.BoundOps
public import TorchLean.Tensor.ConfiguredBinary
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Interval.Outward
public import FloatLib.Numerics.Enclosure.Interval.ElementaryProof

/-!
# Bound-propagation instances for `ExecFloat.Binary 8 23`

`BoundOps.configured` builds the directed endpoint operations of a configured IEEE binary format
from FloatLib's arithmetic rounded down and up. Binary32 registers it as its `BoundOps` instance, so
IBP and CROWN code written against `BoundOps` runs on `α := ExecFloat.Binary 8 23` with outward
rounding on the float32 grid, as long as the endpoints stay finite.

The `NonlinearBoundOps` instance computes a rational enclosure of each transfer and rounds it
outward. The theorems in `CROWN.Binary32`, from `divBounds_sound` to `cosBounds_sound`, prove
that each successful result contains the real image of its input interval.

Neither instance follows a paper. We built both by hand on FloatLib's directed arithmetic and its
rational enclosures.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)
open FloatLib.Numerics (Interval)

namespace TorchLean.CROWN

open FloatLib.Floats.Formats.BinaryInterchange in
/-- `BoundOps` for a configured IEEE binary format, from FloatLib's addition, subtraction and
multiplication rounded down and up. It is not an instance: a format without infinities gives its
native overflow value on a directed overflow, which can lie inside the true bound. -/
@[instance_reducible] def BoundOps.configured {format : FloatFormat}
    {plan : Configured.StoragePlan format} {code : Type}
    [ExecFloat.ModelCodec plan (Model format) code] (_hformat : format.isIEEE = true) :
    BoundOps (ExecFloat (Configured.Family format code plan)) where
  addDown := Binary.add (mode := .down)
  addUp := Binary.add (mode := .up)
  subDown := Binary.sub (mode := .down)
  subUp := Binary.sub (mode := .up)
  mulDown := Binary.mul (mode := .down)
  mulUp := Binary.mul (mode := .up)

/-- `BoundOps` for `ExecFloat.Binary 8 23`, `BoundOps.configured` at IEEE binary32. -/
instance (priority := 1000) instBoundOpsBinary32 :
    BoundOps (ExecFloat.Binary 8 23) := BoundOps.configured rfl

namespace Binary32

/--
Decode both endpoints to rationals, enclose the image with the rational kernel `f`, and round the
result outward to binary32. The result is `none` when an endpoint is exceptional, the interval is
reversed, `f` returns `none`, or a rounded endpoint is not finite.
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
Rational enclosure of `exp` with the argument clamped at -128, so that argument reduction stays
bounded. An upper endpoint above 128 already rules out a finite binary32 enclosure, and the result
is `none`. A lower endpoint below -128 gets the lower bound zero. The upper bound is the enclosure
of `exp` at the upper endpoint, or of `exp (-128)` when the whole interval lies below -128.

The clamp avoids repeatedly squaring enormous exact rationals for extreme finite inputs.
-/
def expRationalBounds? (I : Interval ℚ) : Option (Interval ℚ) :=
  if I.hi ≤ 128 then
    let J := Interval.expBounds ⟨max I.lo (-128), max I.hi (-128)⟩ 16
    some ⟨if I.lo < -128 then 0 else J.lo, J.hi⟩
  else
    none

/--
Rational enclosure of `sqrt` on the nonnegative part of an interval: an interval that crosses zero
has its lower endpoint clamped to zero, and an interval below zero gives `none`. The dyadic grid
has enough precision to resolve binary32 square roots even at the smallest positive input.
-/
def sqrtRationalBounds? (I : Interval ℚ) : Option (Interval ℚ) :=
  if 0 ≤ I.hi then Interval.sqrtBounds? ⟨max 0 I.lo, I.hi⟩ 100 else none

end Binary32

open Binary32

/--
Nonlinear enclosures from FloatLib's whole-interval rational kernels, with checked directed
binary32 rounding. Sigmoid and tanh get the fixed ranges `[0, 1]` and `[-1, 1]`, and layer
normalization returns `none`. Successful results have finite endpoints. Reversed intervals, NaN,
infinity, division through zero, invalid logarithm and square-root domains, and overflow return
`none`.

The containment theorems below read endpoints by exact finite decoding. The IEEE carrier also
holds exceptional values, so it gets no global lawful instance.
-/
instance (priority := 1000) instNonlinearBoundOpsBinary32 :
    NonlinearBoundOps (ExecFloat.Binary 8 23) where
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

private theorem Binary32.unaryBounds?_sound {f : Interval ℚ → Option (Interval ℚ)}
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

private theorem Binary32.expRationalBounds?_sound {I J : Interval ℚ}
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

private theorem Binary32.sqrtRationalBounds?_sound {I J : Interval ℚ}
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

namespace Binary32

/-- If `divBounds` returns an interval, it contains `x / y` for every real `x` in the first input
interval and every real `y` in the second. -/
theorem divBounds_sound {aLo aHi bLo bHi outLo outHi : Binary 8 23}
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

/-- A successful `expBounds` result contains `Real.exp x` for every real `x` in the input interval,
so the whole real image is covered, irrational values included. -/
theorem expBounds_sound {lo hi outLo outHi : Binary 8 23}
    (h : NonlinearBoundOps.expBounds lo hi = some (outLo, outHi)) {x : ℝ}
    (hx : (⟨lo, hi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? x) :
    (⟨outLo, outHi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? (Real.exp x) :=
  unaryBounds?_sound
    (fun {_I _J} h {_x} hx => expRationalBounds?_sound h hx) h hx

/-- When `logBounds` succeeds, which needs a positive lower endpoint, its output contains
`Real.log x` for every real `x` in the input interval. -/
theorem logBounds_sound {lo hi outLo outHi : Binary 8 23}
    (h : NonlinearBoundOps.logBounds lo hi = some (outLo, outHi)) {x : ℝ}
    (hx : (⟨lo, hi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? x) :
    (⟨outLo, outHi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? (Real.log x) :=
  unaryBounds?_sound
    (fun {_I _J} h {_x} hx => Interval.containsReal_logBounds? h hx) h hx

/-- If `sqrtBounds` succeeds, its output contains `Real.sqrt x` for every real `x` in the input
interval. `Real.sqrt` is zero on negative inputs, which matches the clamp of the lower endpoint to
zero. -/
theorem sqrtBounds_sound {lo hi outLo outHi : Binary 8 23}
    (h : NonlinearBoundOps.sqrtBounds lo hi = some (outLo, outHi)) {x : ℝ}
    (hx : (⟨lo, hi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? x) :
    (⟨outLo, outHi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? (Real.sqrt x) :=
  unaryBounds?_sound
    (fun {_I _J} h {_x} hx => sqrtRationalBounds?_sound h hx) h hx

/-- A successful `sigmoidBounds` result contains the logistic value `Real.sigmoid x` for every
real `x` in the input interval. The transfer returns the fixed range `[0, 1]`. -/
theorem sigmoidBounds_sound {lo hi outLo outHi : Binary 8 23}
    (h : NonlinearBoundOps.sigmoidBounds lo hi = some (outLo, outHi)) {x : ℝ}
    (hx : (⟨lo, hi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? x) :
    (⟨outLo, outHi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? (Real.sigmoid x) := by
  apply unaryBounds?_sound (f := fun _ => some ⟨0, 1⟩) (g := Real.sigmoid) ?_ h hx
  intro I J hJ x _
  cases hJ
  apply (Interval.containsReal_some_iff _ _).2
  simpa using And.intro (Real.sigmoid_nonneg x) (Real.sigmoid_le_one x)

/-- When `tanhBounds` succeeds, its output contains `Real.tanh x` for every real `x` in the input
interval. The transfer returns the fixed range `[-1, 1]`. -/
theorem tanhBounds_sound {lo hi outLo outHi : Binary 8 23}
    (h : NonlinearBoundOps.tanhBounds lo hi = some (outLo, outHi)) {x : ℝ}
    (hx : (⟨lo, hi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? x) :
    (⟨outLo, outHi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? (Real.tanh x) := by
  apply unaryBounds?_sound (f := fun _ => some ⟨-1, 1⟩) ?_ h hx
  intro I J hJ x _
  cases hJ
  apply (Interval.containsReal_some_iff _ _).2
  simpa using And.intro (Real.neg_one_lt_tanh x).le (Real.tanh_lt_one x).le

/-- If `sinBounds` succeeds, its output contains `Real.sin x` for every real `x` in the input
interval, so a maximum or minimum of the sine inside the interval is covered too. -/
theorem sinBounds_sound {lo hi outLo outHi : Binary 8 23}
    (h : NonlinearBoundOps.sinBounds lo hi = some (outLo, outHi)) {x : ℝ}
    (hx : (⟨lo, hi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? x) :
    (⟨outLo, outHi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? (Real.sin x) := by
  apply unaryBounds?_sound (f := fun I => some (Interval.sinBounds I 16)) ?_ h hx
  intro I J hJ x hx
  cases hJ
  exact Interval.containsReal_sinBounds I 16 hx

/-- A successful `cosBounds` result contains `Real.cos x` for every real `x` in the input
interval. -/
theorem cosBounds_sound {lo hi outLo outHi : Binary 8 23}
    (h : NonlinearBoundOps.cosBounds lo hi = some (outLo, outHi)) {x : ℝ}
    (hx : (⟨lo, hi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? x) :
    (⟨outLo, outHi⟩ : Interval (Binary 8 23)).ContainsReal Binary.toRat? (Real.cos x) := by
  apply unaryBounds?_sound (f := fun I => some (Interval.cosBounds I 16)) ?_ h hx
  intro I J hJ x hx
  cases hJ
  exact Interval.containsReal_cosBounds I 16 hx

end Binary32

end TorchLean.CROWN
