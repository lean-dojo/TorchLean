/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/
module

public import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
public import FloatLib.Floats.Interval.RealBounds
public import FloatLib.Floats.Interval.Rounders
public import TorchLean.Verify.CROWN.BoundOps.Lawful
public import TorchLean.Verify.CROWN.FlatBox.Transfer
public import TorchLean.Tensor.NF

/-!
# Interval arithmetic lemmas

Scalar facts shared by the CROWN soundness proofs: monotonicity of `tanh`, the enclosure law for
the four-corner product `Graph.intervalMul`, and the exact real instance of
`LawfulNonlinearBoundOps`.

`BoundOps.outward` and its lawful and nonlinear companions build the instances on rounded reals
`NF β fexp rnd` from the real ones. Each endpoint is the exact real result, rounded down (lower) or
up (upper) by a `FloatLib.Floats.Interval.Rounder`, so the real laws carry over when `1` is on the
grid.
-/

@[expose] public section

open FloatLib FloatLib.Numerics FloatLib.Floats.Formats
open Flocq


namespace TorchLean.CROWN

/-! ### Monotone nonlinear functions -/

/-- The derivative of `Real.tanh` at `x` is `1 / (Real.cosh x) ^ 2`. -/
theorem hasDerivAt_tanh (x : ℝ) :
    HasDerivAt Real.tanh (1 / (Real.cosh x) ^ 2) x := by
  convert (Real.hasDerivAt_sinh x).div (Real.hasDerivAt_cosh x) (Real.cosh_pos x).ne'
    using 1
  · funext y
    exact Real.tanh_eq_sinh_div_cosh y
  · simp only [← pow_two, Real.cosh_sq_sub_sinh_sq]

/-- `Real.tanh` is strictly monotone. -/
theorem tanh_strictMono : StrictMono Real.tanh := by
  refine strictMono_of_deriv_pos fun x ↦ ?_
  rw [(hasDerivAt_tanh x).deriv]
  exact one_div_pos.mpr (sq_pos_of_pos (Real.cosh_pos x))

/-- `Real.tanh` is monotone. -/
theorem tanh_monotone : Monotone Real.tanh :=
  tanh_strictMono.monotone

/-- A scalar square lies below the larger square of any enclosing interval's endpoints. -/
theorem square_le_max {lo x hi : ℝ} (hlo : lo ≤ x) (hhi : x ≤ hi) :
    x * x ≤ max (lo * lo) (hi * hi) := by
  by_cases hx : 0 ≤ x
  · exact (mul_self_le_mul_self hx hhi).trans (le_max_right _ _)
  · have hx' : 0 ≤ -x := neg_nonneg.mpr (le_of_not_ge hx)
    have h := mul_self_le_mul_self hx' (neg_le_neg hlo)
    have hsquare : x * x ≤ lo * lo := by simpa only [neg_mul_neg] using h
    exact hsquare.trans (le_max_left _ _)

/-! ### Directed endpoint arithmetic -/

section Directed

variable {α : Type} [Storage α] [Context α] [BoundOps α] [LawfulBoundOps α]

local notation "value" => LawfulBoundOps.toReal (α := α)

/--
For a lawful scalar, if `x` lies between `al` and `au` and `y` between `bl` and `bu` (as reals),
then the outward-rounded four-corner product `Graph.intervalMul al au bl bu` encloses `x * y`.

The proof needs only the endpoint interpretation and the directed-operation laws, with no ring
structure on the rounded scalars and no exact reassociation. The real four-corner bound is
FloatLib's `mul_bounds_Icc`.
-/
theorem Graph.intervalMul_encloses {al au bl bu : α} {x y : ℝ}
    (hal : value al ≤ x) (hau : x ≤ value au)
    (hbl : value bl ≤ y) (hbu : y ≤ value bu) :
    value (Graph.intervalMul al au bl bu).1 ≤ x * y ∧
      x * y ≤ value (Graph.intervalMul al au bl bu).2 := by
  have h := FloatLib.Floats.Interval.mul_bounds_Icc _ _ _ _ x y ⟨hal, hau⟩ ⟨hbl, hbu⟩
  simp only [Graph.intervalMul, LawfulBoundOps.toReal_minimum, LawfulBoundOps.toReal_maximum]
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
    change UnaryEnclosure (α := ℝ) Real.sigmoid
      (fun lo hi ↦ some (Real.sigmoid lo, Real.sigmoid hi))
    exact unaryEnclosure_of_monotone Real.sigmoid Real.sigmoid_monotone
  tanhBounds_enclosure := by
    change UnaryEnclosure (α := ℝ) Real.tanh (fun lo hi ↦ some (Real.tanh lo, Real.tanh hi))
    exact unaryEnclosure_of_monotone Real.tanh tanh_monotone
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

end TorchLean.CROWN

namespace TorchLean.CROWN

/-! ### Outward rounding -/

open FloatLib.Floats.Interval (Rounder)

variable {β : Radix} {fexp : ℤ → ℤ} {rnd : ℝ → ℤ} [ValidExp fexp] [ValidRnd rnd]

/-- Endpoint arithmetic on rounded reals: the exact real result, with the lower endpoint rounded
down and the upper endpoint rounded up by `R`. -/
@[instance_reducible] noncomputable def BoundOps.outward (R : Rounder) :
    BoundOps (NF β fexp rnd) where
  addDown a b := ⟨R.down (BoundOps.addDown a.val b.val)⟩
  addUp a b := ⟨R.up (BoundOps.addUp a.val b.val)⟩
  subDown a b := ⟨R.down (BoundOps.subDown a.val b.val)⟩
  subUp a b := ⟨R.up (BoundOps.subUp a.val b.val)⟩
  mulDown a b := ⟨R.down (BoundOps.mulDown a.val b.val)⟩
  mulUp a b := ⟨R.up (BoundOps.mulUp a.val b.val)⟩
  exactReassociation := false

/-- Outward-rounded endpoints enclose exact real arithmetic. The hypothesis puts `1` on the grid,
so that the endpoint `1` means `1`. -/
@[instance_reducible] noncomputable def LawfulBoundOps.outward (R : Rounder)
    (hone : genericFormat β fexp 1) :
    @LawfulBoundOps (NF β fexp rnd) _ _ (BoundOps.outward R) :=
  letI := BoundOps.outward (β := β) (fexp := fexp) (rnd := rnd) R
  { toReal := NF.val
    lt_iff _ _ := Iff.rfl
    toReal_zero := round_preserves_generic _ _ generic_format_zero
    toReal_one := round_preserves_generic _ _ hone
    toReal_max _ _ := Monotone.map_max fun _ _ h => h
    toReal_min _ _ := Monotone.map_min fun _ _ h => h
    toReal_maximum a b := by
      rw [NF.maximum_eq_max]
      exact Monotone.map_max fun _ _ h => h
    toReal_minimum a b := by
      rw [NF.minimum_eq_min]
      exact Monotone.map_min fun _ _ h => h
    toReal_eq_of_beq h := by rw [beq_iff_eq.mp h]
    addDown_le a b := (R.down_le _).trans (addDown_le (α := ℝ) a.val b.val)
    le_addUp a b := (le_addUp (α := ℝ) a.val b.val).trans (R.le_up _)
    subDown_le a b := (R.down_le _).trans (subDown_le (α := ℝ) a.val b.val)
    le_subUp a b := (le_subUp (α := ℝ) a.val b.val).trans (R.le_up _)
    mulDown_le a b := (R.down_le _).trans (mulDown_le (α := ℝ) a.val b.val)
    le_mulUp a b := (le_mulUp (α := ℝ) a.val b.val).trans (R.le_up _) }

/-- Nonlinear transfers on rounded reals: the exact real transfer, with the lower endpoint rounded
down and the upper endpoint rounded up by `R`. -/
@[instance_reducible] noncomputable def NonlinearBoundOps.outward (R : Rounder) :
    NonlinearBoundOps (NF β fexp rnd) where
  divBounds aLo aHi bLo bHi := (NonlinearBoundOps.divBounds aLo.val aHi.val bLo.val bHi.val).map
    fun p => (⟨R.down p.1⟩, ⟨R.up p.2⟩)
  expBounds lo hi := (NonlinearBoundOps.expBounds lo.val hi.val).map
    fun p => (⟨R.down p.1⟩, ⟨R.up p.2⟩)
  logBounds lo hi := (NonlinearBoundOps.logBounds lo.val hi.val).map
    fun p => (⟨R.down p.1⟩, ⟨R.up p.2⟩)
  sqrtBounds lo hi := (NonlinearBoundOps.sqrtBounds lo.val hi.val).map
    fun p => (⟨R.down p.1⟩, ⟨R.up p.2⟩)
  sigmoidBounds lo hi := (NonlinearBoundOps.sigmoidBounds lo.val hi.val).map
    fun p => (⟨R.down p.1⟩, ⟨R.up p.2⟩)
  tanhBounds lo hi := (NonlinearBoundOps.tanhBounds lo.val hi.val).map
    fun p => (⟨R.down p.1⟩, ⟨R.up p.2⟩)
  sinBounds lo hi := (NonlinearBoundOps.sinBounds lo.val hi.val).map
    fun p => (⟨R.down p.1⟩, ⟨R.up p.2⟩)
  cosBounds lo hi := (NonlinearBoundOps.cosBounds lo.val hi.val).map
    fun p => (⟨R.down p.1⟩, ⟨R.up p.2⟩)
  layerNormAbsBound n := (NonlinearBoundOps.layerNormAbsBound (α := ℝ) n).map fun r => ⟨R.up r⟩

private theorem unaryEnclosure_outward (R : Rounder) (hone : genericFormat β fexp 1)
    {f : ℝ → ℝ} {transfer : ℝ → ℝ → Option (ℝ × ℝ)} (h : UnaryEnclosure f transfer) :
    @UnaryEnclosure (NF β fexp rnd) _ _ (BoundOps.outward R) (LawfulBoundOps.outward R hone) f
      fun lo hi => (transfer lo.val hi.val).map fun p => (⟨R.down p.1⟩, ⟨R.up p.2⟩) := by
  intro lo hi outLo outHi x hout hlo hhi
  obtain ⟨⟨pLo, pHi⟩, hp, hpair⟩ := Option.map_eq_some_iff.mp hout
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj hpair
  have hx := h hp hlo hhi
  exact ⟨(R.down_le pLo).trans hx.1, hx.2.trans (R.le_up pHi)⟩

private theorem binaryEnclosure_outward (R : Rounder) (hone : genericFormat β fexp 1)
    {f : ℝ → ℝ → ℝ} {transfer : ℝ → ℝ → ℝ → ℝ → Option (ℝ × ℝ)}
    (h : BinaryEnclosure f transfer) :
    @BinaryEnclosure (NF β fexp rnd) _ _ (BoundOps.outward R) (LawfulBoundOps.outward R hone) f
      fun aLo aHi bLo bHi => (transfer aLo.val aHi.val bLo.val bHi.val).map fun p =>
        (⟨R.down p.1⟩, ⟨R.up p.2⟩) := by
  intro aLo aHi bLo bHi outLo outHi x y hout hxLo hxHi hyLo hyHi
  obtain ⟨⟨pLo, pHi⟩, hp, hpair⟩ := Option.map_eq_some_iff.mp hout
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj hpair
  have hxy := h hp hxLo hxHi hyLo hyHi
  exact ⟨(R.down_le pLo).trans hxy.1, hxy.2.trans (R.le_up pHi)⟩

/-- If `1` is on the grid, the outward-rounded transfers satisfy `LawfulNonlinearBoundOps`:
rounding lower endpoints down and upper endpoints up keeps every exact real enclosure an
enclosure. -/
theorem LawfulNonlinearBoundOps.outward (R : Rounder) (hone : genericFormat β fexp 1) :
    @LawfulNonlinearBoundOps (NF β fexp rnd) _ _ (BoundOps.outward R)
      (LawfulBoundOps.outward R hone) (NonlinearBoundOps.outward R) :=
  letI := BoundOps.outward (β := β) (fexp := fexp) (rnd := rnd) R
  letI := LawfulBoundOps.outward (rnd := rnd) R hone
  letI := NonlinearBoundOps.outward (β := β) (fexp := fexp) (rnd := rnd) R
  { divBounds_enclosure := binaryEnclosure_outward R hone (divBounds_enclosure (α := ℝ))
    expBounds_enclosure := unaryEnclosure_outward R hone (expBounds_enclosure (α := ℝ))
    logBounds_enclosure := unaryEnclosure_outward R hone (logBounds_enclosure (α := ℝ))
    sqrtBounds_enclosure := unaryEnclosure_outward R hone (sqrtBounds_enclosure (α := ℝ))
    sigmoidBounds_enclosure := unaryEnclosure_outward R hone (sigmoidBounds_enclosure (α := ℝ))
    tanhBounds_enclosure := unaryEnclosure_outward R hone (tanhBounds_enclosure (α := ℝ))
    sinBounds_enclosure := unaryEnclosure_outward R hone (sinBounds_enclosure (α := ℝ))
    cosBounds_enclosure := unaryEnclosure_outward R hone (cosBounds_enclosure (α := ℝ))
    layerNormAbsBound_sound hout := by
      obtain ⟨r, hr, rfl⟩ := Option.map_eq_some_iff.mp hout
      exact (layerNormAbsBound_sound (α := ℝ) hr).trans (R.le_up r) }

/-- Rounding the positive rational stabilizer onto a grid keeps it nonnegative. -/
theorem _root_.TorchLean.normalizationEpsilon_nonneg :
    0 ≤ (normalizationEpsilon : NF β fexp rnd).val :=
  round_nonneg rnd (by norm_num)

end TorchLean.CROWN
