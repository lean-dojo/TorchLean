/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Proofs.GraphCertSoundness.IntervalLemmas
public import NN.Proofs.Gradients.Activation

/-!
# Softplus and safeLog bounds

The executable transfers use elementary bounds on softplus, so they can handle large positive
inputs without evaluating a large exponential. SafeLog then adds its scalar epsilon interval and
checks that the whole logarithm argument is positive. This file connects those transfers to the
real specifications, including the fact that epsilon may vary inside its own interval.
-/

@[expose] public section

namespace NN.MLTheory.CROWN.Graph.CertSoundness

open Spec TorchLean
open TorchLean.Tensor

noncomputable section

/-- The logarithmic tail in the stable softplus branch lies between zero and one. -/
theorem softplus_tail_bounds_real {z : ℝ} (hz : z ≤ 0) :
    0 ≤ Real.log (1 + Real.exp z) ∧ Real.log (1 + Real.exp z) ≤ 1 := by
  have hexp : Real.exp z ≤ 1 := Real.exp_le_one_iff.mpr hz
  have hpositive : 0 < 1 + Real.exp z := by positivity
  refine ⟨Real.log_nonneg (by linarith [Real.exp_pos z]), ?_⟩
  have hlog := Real.log_le_sub_one_of_pos hpositive
  linarith

/-- Softplus differs from `max x 0` by a nonnegative term of size at most one.

The proof follows the same sign branch as the executable specification. In each branch the
exponential argument is nonpositive, which is why the resulting interval rule needs no large
exponential at either endpoint. -/
theorem softplus_envelope_real (x : ℝ) :
    max x 0 ≤ Activation.Math.softplusSpec x ∧
      Activation.Math.softplusSpec x ≤ max x 0 + 1 := by
  simp only [Activation.Math.softplusSpec, MathFunctions.log, MathFunctions.exp]
  by_cases hx : x > 0
  · simp only [ite_eq_left hx, max_eq_left (le_of_lt hx)]
    have htail := softplus_tail_bounds_real (neg_nonpos.mpr (le_of_lt hx))
    constructor <;> linarith
  · have hnonpos : x ≤ 0 := le_of_not_gt hx
    simpa only [ite_eq_right hx, max_eq_right hnonpos, zero_add] using
      softplus_tail_bounds_real hnonpos

/-- Every successful real softplus transfer encloses the source operation. -/
theorem softplusBounds_sound_real {lo hi outLo outHi x : ℝ}
    (hB : NonlinearBoundOps.softplusBounds lo hi = some (outLo, outHi))
    (hxlo : lo ≤ x) (hxhi : x ≤ hi) :
    outLo ≤ Activation.Math.softplusSpec x ∧
      Activation.Math.softplusSpec x ≤ outHi := by
  simp only [NonlinearBoundOps.softplusBounds, max2_eq_max, BoundOps.addUp,
    Option.some.injEq, Prod.mk.injEq] at hB
  rcases hB with ⟨rfl, rfl⟩
  have hsoft := softplus_envelope_real x
  exact ⟨(max_le_max hxlo le_rfl).trans hsoft.1,
    hsoft.2.trans (add_le_add (max_le_max hxhi le_rfl) le_rfl)⟩

/-- The reciprocal fallback for logarithm is valid on the entire positive real axis.

It is deliberately coarse near zero. Its purpose is to give directed-arithmetic backends a usable
enclosure even when they do not supply a correctly rounded logarithm transfer. -/
theorem log_reciprocal_bounds_real {x : ℝ} (hx : 0 < x) :
    1 - 1 / x ≤ Real.log x ∧ Real.log x ≤ x - 1 := by
  simpa only [one_div] using
    And.intro (Real.one_sub_inv_le_log_of_pos hx) (Real.log_le_sub_one_of_pos hx)

/-- SafeLog encloses both its input and its scalar epsilon, without replacing epsilon by a default.

Successful transfer establishes positivity of the lower logarithm argument. The proof uses that
check to apply monotonicity of the real logarithm; no positivity premise is added to the caller. -/
theorem safeLogBounds_sound_real {lo hi epsilonLo epsilonHi outLo outHi x epsilon : ℝ}
    (hB : NonlinearBoundOps.safeLogBounds lo hi epsilonLo epsilonHi = some (outLo, outHi))
    (hxlo : lo ≤ x) (hxhi : x ≤ hi)
    (helo : epsilonLo ≤ epsilon) (hehi : epsilon ≤ epsilonHi) :
    outLo ≤ Activation.Math.safeLogSpec x epsilon ∧
      Activation.Math.safeLogSpec x epsilon ≤ outHi := by
  simp only [NonlinearBoundOps.safeLogBounds, NonlinearBoundOps.softplusBounds,
    max2_eq_max, BoundOps.addDown, BoundOps.addUp] at hB
  dsimp only [Bind.bind, Option.bind] at hB
  split at hB
  next hpos =>
    simp only [NonlinearBoundOps.logBounds, ite_eq_left hpos] at hB
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hB)
    have hsoft := softplus_envelope_real x
    have hlo : max lo 0 + epsilonLo ≤ Activation.Math.softplusSpec x + epsilon :=
      add_le_add ((max_le_max hxlo le_rfl).trans hsoft.1) helo
    have hhi : Activation.Math.softplusSpec x + epsilon ≤ max hi 0 + 1 + epsilonHi :=
      add_le_add (hsoft.2.trans (add_le_add (max_le_max hxhi le_rfl) le_rfl)) hehi
    exact ⟨Real.log_le_log hpos hlo, Real.log_le_log (hpos.trans_le hlo) hhi⟩
  next =>
    cases hB

/-- A scalar enclosure lifts coordinatewise to the flat tensor representation.

The transfer may fail on any coordinate. If it returns a box, `traverseFin` gives the corresponding
successful scalar transfer at every index, so no unchecked default endpoints enter the proof. -/
theorem enclosesBox_boxUnaryEnclosure {f : ℝ → ℝ}
    {enclose : ℝ → ℝ → Option (ℝ × ℝ)}
    (hEnclose : ∀ {lo hi outLo outHi x : ℝ},
      enclose lo hi = some (outLo, outHi) → lo ≤ x → x ≤ hi →
        outLo ≤ f x ∧ f x ≤ outHi)
    {B1 B : FlatBox ℝ} {v1 : Val}
    (h1 : EnclosesBox B1 v1) (hB : boxUnaryEnclosure? enclose B1 = some B) :
    EnclosesBox B ⟨v1.n, Tensor.mapSpec f v1.v⟩ := by
  obtain ⟨hDim, hx⟩ := h1
  obtain ⟨n, lo, hi⟩ := B1
  obtain ⟨m, x⟩ := v1
  simp only at hDim
  subst hDim
  simp only [castDimScalar_self] at hx
  unfold boxUnaryEnclosure? at hB
  cases hb : Internal.traverseFin (fun i => enclose (lo.getScalar i) (hi.getScalar i)) with
  | none =>
      rw [hb] at hB
      cases hB
  | some bounds =>
      rw [hb] at hB
      obtain rfl := Option.some.inj hB
      refine ⟨rfl, ?_⟩
      intro i
      have hiB := Internal.traverseFin_eq_some_iff.mp hb i
      simpa using hEnclose hiB (hx i).1 (hx i).2

/-- The executable softplus box transfer encloses the real tensor specification. -/
theorem enclosesBox_boxSoftplus {B1 B : FlatBox ℝ} {v1 : Val}
    (h1 : EnclosesBox B1 v1) (hB : boxSoftplus? B1 = some B) :
    EnclosesBox B ⟨v1.n, Activation.softplusSpec v1.v⟩ :=
  enclosesBox_boxUnaryEnclosure softplusBounds_sound_real h1 hB

/-- The executable safeLog transfer encloses a tensor and a shared scalar epsilon.

The epsilon value is read from its one-element flat tensor after aligning the parent box's
dimension with that value. Thus the same epsilon interval is used for every input coordinate. -/
theorem enclosesBox_boxSafeLog {B1 epsilonBox B : FlatBox ℝ} {v1 : Val}
    {epsilon : Tensor ℝ [1]}
    (h1 : EnclosesBox B1 v1) (he : EnclosesBox epsilonBox ⟨1, epsilon⟩)
    (hB : boxSafeLog? B1 epsilonBox = some B) :
    EnclosesBox B ⟨v1.n, Activation.safeLogSpec v1.v (epsilon.getScalar 0)⟩ := by
  obtain ⟨heDim, heBounds⟩ := he
  obtain ⟨ne, elo, ehi⟩ := epsilonBox
  simp only at heDim
  subst ne
  simp only [castDimScalar_self] at heBounds
  simp only [boxSafeLog?, ↓reduceDIte] at hB
  apply enclosesBox_boxUnaryEnclosure (h1 := h1) (hB := hB)
  intro lo hi outLo outHi x htransfer hxlo hxhi
  exact safeLogBounds_sound_real htransfer hxlo hxhi (heBounds 0).1 (heBounds 0).2

end

end NN.MLTheory.CROWN.Graph.CertSoundness
