/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# Bracketing a Real Value

Effective rounding algorithms refine an interval until its endpoints are adjacent representable
values.  A location records whether the input is the lower endpoint or, for an interior point,
whether it lies below, at, or above the interval midpoint.  This is the Lean counterpart of
Flocq's `Calc.Bracket` location semantics.
-/

@[expose] public section

namespace TorchLean.Floats

/-- Position of a real value relative to the lower endpoint and midpoint of a bracket. -/
inductive NeuralLocation where
  | exact
  | inexact (midpointOrder : Ordering)
  deriving DecidableEq, Repr

/-- Compute the location of `x` in a bracket whose lower endpoint is `d`. -/
noncomputable def neuralInbetweenLocation (d u x : ℝ) : NeuralLocation :=
  if d < x then .inexact (cmp x ((d + u) / 2)) else .exact

/-- Semantic meaning of a bracket location. -/
inductive NeuralInbetween (d u x : ℝ) : NeuralLocation → Prop where
  | exact (hx : x = d) : NeuralInbetween d u x .exact
  | inexact (l : Ordering) (hx : d < x ∧ x < u)
      (hmid : cmp x ((d + u) / 2) = l) : NeuralInbetween d u x (.inexact l)

/-- The computed location satisfies the bracket semantics. -/
theorem neuralInbetweenLocation_spec {d u x : ℝ} (_hdu : d < u) (hx : d ≤ x ∧ x < u) :
    NeuralInbetween d u x (neuralInbetweenLocation d u x) := by
  by_cases hdx : d < x
  · rw [neuralInbetweenLocation, if_pos hdx]
    exact .inexact _ ⟨hdx, hx.2⟩ rfl
  · have hxd : x = d := le_antisymm (le_of_not_gt hdx) hx.1
    rw [neuralInbetweenLocation, if_neg hdx]
    exact .exact hxd

/-- A value has at most one semantic location in a fixed bracket. -/
theorem neuralInbetween_unique {d u x : ℝ} {l l' : NeuralLocation}
    (hl : NeuralInbetween d u x l) (hl' : NeuralInbetween d u x l') : l = l' := by
  cases hl with
  | exact hx =>
      cases hl' with
      | exact => rfl
      | inexact _ hx' _ => exact (not_lt_of_ge (le_of_eq hx)) hx'.1 |>.elim
  | inexact order hbetween hmid =>
      cases hl' with
      | exact hx => exact (ne_of_lt hbetween.1) hx.symm |>.elim
      | inexact order' _ hmid' =>
          congr
          exact hmid.symm.trans hmid'

/-- Every semantic location places the input in the half-open bracket `[d, u)`. -/
theorem neuralInbetween_bounds {d u x : ℝ} {l : NeuralLocation} (hdu : d < u)
    (hl : NeuralInbetween d u x l) : d ≤ x ∧ x < u := by
  cases hl with
  | exact hx => simpa [hx] using hdu
  | inexact _ hx _ => exact ⟨hx.1.le, hx.2⟩

/-- An inexact location places the input strictly between the bracket endpoints. -/
theorem neuralInbetween_strict_bounds {d u x : ℝ} {order : Ordering}
    (hl : NeuralInbetween d u x (.inexact order)) : d < x ∧ x < u := by
  cases hl
  assumption

/-- Midpoint comparison is equivalent to comparing distances from the two endpoints. -/
theorem neuralInbetween_distance {d u x : ℝ} {order : Ordering}
    (hl : NeuralInbetween d u x (.inexact order)) :
    cmp (x - d) (u - x) = order := by
  cases hl with
  | inexact _ _ hmid =>
      cases order with
      | lt =>
          rw [cmp_eq_lt_iff] at hmid ⊢
          linarith
      | eq =>
          rw [cmp_eq_eq_iff] at hmid ⊢
          linarith
      | gt =>
          rw [cmp_eq_gt_iff] at hmid ⊢
          linarith

/-- The midpoint order also compares the absolute distances to the endpoints. -/
theorem neuralInbetween_abs_distance {d u x : ℝ} {order : Ordering}
    (hl : NeuralInbetween d u x (.inexact order)) :
    cmp |d - x| |u - x| = order := by
  have hx := neuralInbetween_strict_bounds hl
  rw [abs_of_nonpos (sub_nonpos.mpr hx.1.le), abs_of_nonneg (sub_nonneg.mpr hx.2.le),
    neg_sub]
  exact neuralInbetween_distance hl

/-- Every location is realized by a point in every nondegenerate bracket. -/
theorem neuralInbetween_exists {d u : ℝ} (hdu : d < u) (location : NeuralLocation) :
    ∃ x, NeuralInbetween d u x location := by
  cases location with
  | exact => exact ⟨d, .exact rfl⟩
  | inexact order =>
      cases order with
      | lt =>
          refine ⟨(3 * d + u) / 4, .inexact .lt ?_ ?_⟩
          · constructor <;> linarith
          · rw [cmp_eq_lt_iff]
            linarith
      | eq =>
          refine ⟨(d + u) / 2, .inexact .eq ?_ ?_⟩
          · constructor <;> linarith
          · rw [cmp_eq_eq_iff]
      | gt =>
          refine ⟨(d + 3 * u) / 4, .inexact .gt ?_ ?_⟩
          · constructor <;> linarith
          · rw [cmp_eq_gt_iff]
            linarith

/-- Consecutive points in a positive-step arithmetic progression are strictly ordered. -/
theorem neuralOrderedSteps {start step : ℝ} (hstep : 0 < step) (k : ℤ) :
    start + (k : ℝ) * step < start + (k + 1 : ℤ) * step := by
  norm_num only [Int.cast_add, Int.cast_one]
  linarith

/-- Midpoint of an arithmetic progression interval. -/
theorem neuralStepMidpoint (start step : ℝ) (steps : ℤ) :
    (start + (start + (steps : ℝ) * step)) / 2 =
      start + ((steps : ℝ) / 2) * step := by
  ring

/--
Lift a location in an interior arithmetic-progression cell to the full interval.  The caller
supplies the comparison with the full midpoint; later refinement rules compute that comparison
from the cell index and local location.
-/
theorem neuralInbetween_step_interior {start step x : ℝ} {steps k : ℤ}
    {localLocation : NeuralLocation} {globalOrder : Ordering}
    (hstep : 0 < step) (_hsteps : 1 < steps) (hk : 0 < k ∧ k < steps)
    (hl : NeuralInbetween
      (start + (k : ℝ) * step)
      (start + (k + 1 : ℤ) * step) x localLocation)
    (hmid : cmp x (start + ((steps : ℝ) / 2) * step) = globalOrder) :
    NeuralInbetween start (start + (steps : ℝ) * step) x (.inexact globalOrder) := by
  have hlocalOrder := neuralOrderedSteps (start := start) hstep k
  have hlocal := neuralInbetween_bounds hlocalOrder hl
  have hk0 : (0 : ℝ) < k := by exact_mod_cast hk.1
  have hkn : (k + 1 : ℤ) ≤ steps := Int.add_one_le_iff.mpr hk.2
  have hknR : ((k + 1 : ℤ) : ℝ) ≤ (steps : ℝ) := by exact_mod_cast hkn
  have hmid' : cmp x ((start + (start + (steps : ℝ) * step)) / 2) = globalOrder := by
    rw [neuralStepMidpoint]
    exact hmid
  refine .inexact globalOrder ?_ hmid'
  constructor
  · calc
      start < start + (k : ℝ) * step := by nlinarith
      _ ≤ x := hlocal.1
  · calc
      x < start + ((k + 1 : ℤ) : ℝ) * step := hlocal.2
      _ ≤ start + (steps : ℝ) * step := by
        simpa [add_comm] using
          add_le_add_left (mul_le_mul_of_nonneg_right hknR hstep.le) start

/-- A subinterval strictly below the global midpoint has global location `.lt`. -/
theorem neuralInbetween_step_low {start step x : ℝ} {steps k : ℤ}
    {localLocation : NeuralLocation}
    (hstep : 0 < step) (hsteps : 1 < steps) (hk : 0 < k)
    (hcell : 2 * k + 1 < steps)
    (hl : NeuralInbetween
      (start + (k : ℝ) * step)
      (start + (k + 1 : ℤ) * step) x localLocation) :
    NeuralInbetween start (start + (steps : ℝ) * step) x (.inexact .lt) := by
  have hkSteps : k < steps := by linarith
  apply neuralInbetween_step_interior hstep hsteps ⟨hk, hkSteps⟩ hl
  rw [cmp_eq_lt_iff]
  have hlocal := neuralInbetween_bounds (neuralOrderedSteps (start := start) hstep k) hl
  have hindex : 2 * (k + 1) ≤ steps := by linarith
  have hindexR : (2 * ((k + 1 : ℤ) : ℝ)) ≤ (steps : ℝ) := by
    exact_mod_cast hindex
  have hscaled := mul_le_mul_of_nonneg_right hindexR hstep.le
  have hbeforeMid : (((k + 1 : ℤ) : ℝ) * step) ≤
      ((steps : ℝ) / 2) * step := by
    nlinarith
  exact hlocal.2.trans_le (by
    simpa [add_comm] using add_le_add_left hbeforeMid start)

/-- A subinterval strictly above the global midpoint has global location `.gt`. -/
theorem neuralInbetween_step_high {start step x : ℝ} {steps k : ℤ}
    {localLocation : NeuralLocation}
    (hstep : 0 < step) (hsteps : 1 < steps) (hcell : steps < 2 * k)
    (hk : k < steps)
    (hl : NeuralInbetween
      (start + (k : ℝ) * step)
      (start + (k + 1 : ℤ) * step) x localLocation) :
    NeuralInbetween start (start + (steps : ℝ) * step) x (.inexact .gt) := by
  have hkPos : 0 < k := by linarith
  apply neuralInbetween_step_interior hstep hsteps ⟨hkPos, hk⟩ hl
  rw [cmp_eq_gt_iff]
  have hlocal := neuralInbetween_bounds (neuralOrderedSteps (start := start) hstep k) hl
  have hindexR : (steps : ℝ) < 2 * (k : ℝ) := by exact_mod_cast hcell
  have hscaled := mul_lt_mul_of_pos_right hindexR hstep
  have hafterMid : ((steps : ℝ) / 2) * step < (k : ℝ) * step := by
    nlinarith
  have hafterMidStart :
      start + ((steps : ℝ) / 2) * step < start + (k : ℝ) * step := by
    simpa [add_comm] using add_lt_add_left hafterMid start
  exact hafterMidStart.trans_le hlocal.1

/-- Exactness at the first cell's lower endpoint remains exact in the full interval. -/
theorem neuralInbetween_step_zero_exact {start step x : ℝ} {steps : ℤ}
    (hl : NeuralInbetween start (start + step) x .exact) :
    NeuralInbetween start (start + (steps : ℝ) * step) x .exact := by
  cases hl with
  | exact hx => exact .exact hx

/-- An inexact point in the first cell lies below the full midpoint when there are several cells. -/
theorem neuralInbetween_step_zero_inexact {start step x : ℝ} {steps : ℤ}
    {order : Ordering} (hstep : 0 < step) (hsteps : 1 < steps)
    (hl : NeuralInbetween start (start + step) x (.inexact order)) :
    NeuralInbetween start (start + (steps : ℝ) * step) x (.inexact .lt) := by
  have hlocal := neuralInbetween_strict_bounds hl
  have htwo : (2 : ℤ) ≤ steps := Int.add_one_le_iff.mpr hsteps
  have htwoR : (2 : ℝ) ≤ (steps : ℝ) := by exact_mod_cast htwo
  have hscaled := mul_le_mul_of_nonneg_right htwoR hstep.le
  have hstepMid : step ≤ ((steps : ℝ) / 2) * step := by nlinarith
  have hstepAll : step < (steps : ℝ) * step := by
    have hstepsR : (1 : ℝ) < (steps : ℝ) := by exact_mod_cast hsteps
    nlinarith
  have hstepAllStart : start + step < start + (steps : ℝ) * step := by
    simpa [add_comm] using add_lt_add_left hstepAll start
  refine .inexact .lt ⟨hlocal.1, hlocal.2.trans hstepAllStart⟩ ?_
  rw [cmp_eq_lt_iff, neuralStepMidpoint]
  exact hlocal.2.trans_le (by
    simpa [add_comm] using add_le_add_left hstepMid start)

/-- At the central lower endpoint of an even subdivision, the global location is midpoint exact. -/
theorem neuralInbetween_step_middle_even_exact {start step x : ℝ} {steps k : ℤ}
    (hstep : 0 < step) (hsteps : 1 < steps) (hmiddle : 2 * k = steps)
    (hl : NeuralInbetween
      (start + (k : ℝ) * step)
      (start + (k + 1 : ℤ) * step) x .exact) :
    NeuralInbetween start (start + (steps : ℝ) * step) x (.inexact .eq) := by
  cases hl with
  | exact hx =>
      have hkPos : 0 < k := by linarith
      have hkSteps : k < steps := by linarith
      apply neuralInbetween_step_interior hstep hsteps ⟨hkPos, hkSteps⟩ (.exact hx)
      rw [cmp_eq_eq_iff]
      have hmiddleR : (2 : ℝ) * (k : ℝ) = (steps : ℝ) := by exact_mod_cast hmiddle
      rw [hx]
      nlinarith

/-- An inexact point in the central cell of an even subdivision is above the global midpoint. -/
theorem neuralInbetween_step_middle_even_inexact {start step x : ℝ} {steps k : ℤ}
    {order : Ordering} (hstep : 0 < step) (hsteps : 1 < steps)
    (hmiddle : 2 * k = steps)
    (hl : NeuralInbetween
      (start + (k : ℝ) * step)
      (start + (k + 1 : ℤ) * step) x (.inexact order)) :
    NeuralInbetween start (start + (steps : ℝ) * step) x (.inexact .gt) := by
  have hkPos : 0 < k := by linarith
  have hkSteps : k < steps := by linarith
  apply neuralInbetween_step_interior hstep hsteps ⟨hkPos, hkSteps⟩ hl
  rw [cmp_eq_gt_iff]
  have hlocal := neuralInbetween_strict_bounds hl
  have hmiddleR : (2 : ℝ) * (k : ℝ) = (steps : ℝ) := by exact_mod_cast hmiddle
  nlinarith

/-- At the lower endpoint of the central cell of an odd subdivision, the global location is low. -/
theorem neuralInbetween_step_middle_odd_exact {start step x : ℝ} {steps k : ℤ}
    (hstep : 0 < step) (hsteps : 1 < steps) (hmiddle : 2 * k + 1 = steps)
    (hl : NeuralInbetween
      (start + (k : ℝ) * step)
      (start + (k + 1 : ℤ) * step) x .exact) :
    NeuralInbetween start (start + (steps : ℝ) * step) x (.inexact .lt) := by
  cases hl with
  | exact hx =>
      have hkPos : 0 < k := by linarith
      have hkSteps : k < steps := by linarith
      apply neuralInbetween_step_interior hstep hsteps ⟨hkPos, hkSteps⟩ (.exact hx)
      rw [cmp_eq_lt_iff, hx]
      have hmiddleR : (2 : ℝ) * (k : ℝ) + 1 = (steps : ℝ) := by
        exact_mod_cast hmiddle
      nlinarith

/-- In the central cell of an odd subdivision, local and global midpoint locations agree. -/
theorem neuralInbetween_step_middle_odd_inexact {start step x : ℝ} {steps k : ℤ}
    {order : Ordering} (hstep : 0 < step) (hsteps : 1 < steps)
    (hmiddle : 2 * k + 1 = steps)
    (hl : NeuralInbetween
      (start + (k : ℝ) * step)
      (start + (k + 1 : ℤ) * step) x (.inexact order)) :
    NeuralInbetween start (start + (steps : ℝ) * step) x (.inexact order) := by
  have hkPos : 0 < k := by linarith
  have hkSteps : k < steps := by linarith
  apply neuralInbetween_step_interior hstep hsteps ⟨hkPos, hkSteps⟩ hl
  cases hl with
  | inexact _ _ hlocalMid =>
      have hmiddleR : (2 : ℝ) * (k : ℝ) + 1 = (steps : ℝ) := by
        exact_mod_cast hmiddle
      have hmidpoints :
          ((start + (k : ℝ) * step) +
              (start + ((k + 1 : ℤ) : ℝ) * step)) / 2 =
            start + ((steps : ℝ) / 2) * step := by
        norm_num only [Int.cast_add, Int.cast_one]
        nlinarith
      rw [← hmidpoints]
      exact hlocalMid

/-- Refine a local location when the enclosing subdivision count is even. -/
def neuralRefineLocationEven (steps k : ℤ) (location : NeuralLocation) : NeuralLocation :=
  if k = 0 then
    match location with
    | .exact => .exact
    | .inexact _ => .inexact .lt
  else
    match cmp (2 * k) steps with
    | .lt => .inexact .lt
    | .eq =>
        match location with
        | .exact => .inexact .eq
        | .inexact _ => .inexact .gt
    | .gt => .inexact .gt

/-- Refine a local location when the enclosing subdivision count is odd. -/
def neuralRefineLocationOdd (steps k : ℤ) (location : NeuralLocation) : NeuralLocation :=
  if k = 0 then
    match location with
    | .exact => .exact
    | .inexact _ => .inexact .lt
  else
    match cmp (2 * k + 1) steps with
    | .lt => .inexact .lt
    | .eq =>
        match location with
        | .exact => .inexact .lt
        | .inexact order => .inexact order
    | .gt => .inexact .gt

/-- Correctness of location refinement for an even number of cells. -/
theorem neuralRefineLocationEven_correct {start step x : ℝ} {steps k : ℤ}
    {location : NeuralLocation} (hstep : 0 < step) (hsteps : 1 < steps)
    (heven : Even steps) (hk : 0 ≤ k ∧ k < steps)
    (hl : NeuralInbetween
      (start + (k : ℝ) * step)
      (start + (k + 1 : ℤ) * step) x location) :
    NeuralInbetween start (start + (steps : ℝ) * step) x
      (neuralRefineLocationEven steps k location) := by
  by_cases hkZero : k = 0
  · subst k
    cases location with
    | exact =>
        simpa [neuralRefineLocationEven] using
          (neuralInbetween_step_zero_exact (steps := steps) (by simpa using hl))
    | inexact order =>
        simpa [neuralRefineLocationEven] using
          (neuralInbetween_step_zero_inexact (steps := steps) hstep hsteps (by simpa using hl))
  · have hkPos : 0 < k := lt_of_le_of_ne hk.1 (Ne.symm hkZero)
    obtain ⟨half, hhalf⟩ := even_iff_exists_two_mul.mp heven
    rcases lt_trichotomy (2 * k) steps with hlow | hmiddle | hhigh
    · have hkHalf : k < half := by linarith
      have hkNext : k + 1 ≤ half := Int.add_one_le_iff.mpr hkHalf
      have hcell : 2 * k + 1 < steps := by linarith
      simpa [neuralRefineLocationEven, hkZero, (cmp_eq_lt_iff _ _).2 hlow] using
        neuralInbetween_step_low hstep hsteps hkPos hcell hl
    · cases location with
      | exact =>
          simpa [neuralRefineLocationEven, hkZero, (cmp_eq_eq_iff _ _).2 hmiddle] using
            neuralInbetween_step_middle_even_exact hstep hsteps hmiddle hl
      | inexact order =>
          simpa [neuralRefineLocationEven, hkZero, (cmp_eq_eq_iff _ _).2 hmiddle] using
            neuralInbetween_step_middle_even_inexact hstep hsteps hmiddle hl
    · simpa [neuralRefineLocationEven, hkZero, (cmp_eq_gt_iff _ _).2 hhigh] using
        neuralInbetween_step_high hstep hsteps hhigh hk.2 hl

/-- Correctness of location refinement for an odd number of cells. -/
theorem neuralRefineLocationOdd_correct {start step x : ℝ} {steps k : ℤ}
    {location : NeuralLocation} (hstep : 0 < step) (hsteps : 1 < steps)
    (hodd : Odd steps) (hk : 0 ≤ k ∧ k < steps)
    (hl : NeuralInbetween
      (start + (k : ℝ) * step)
      (start + (k + 1 : ℤ) * step) x location) :
    NeuralInbetween start (start + (steps : ℝ) * step) x
      (neuralRefineLocationOdd steps k location) := by
  by_cases hkZero : k = 0
  · subst k
    cases location with
    | exact =>
        simpa [neuralRefineLocationOdd] using
          (neuralInbetween_step_zero_exact (steps := steps) (by simpa using hl))
    | inexact order =>
        simpa [neuralRefineLocationOdd] using
          (neuralInbetween_step_zero_inexact (steps := steps) hstep hsteps (by simpa using hl))
  · have hkPos : 0 < k := lt_of_le_of_ne hk.1 (Ne.symm hkZero)
    obtain ⟨half, hhalf⟩ := odd_iff_exists_bit1.mp hodd
    rcases lt_trichotomy (2 * k + 1) steps with hlow | hmiddle | hhigh
    · simpa [neuralRefineLocationOdd, hkZero, (cmp_eq_lt_iff _ _).2 hlow] using
        neuralInbetween_step_low hstep hsteps hkPos hlow hl
    · cases location with
      | exact =>
          simpa [neuralRefineLocationOdd, hkZero, (cmp_eq_eq_iff _ _).2 hmiddle] using
            neuralInbetween_step_middle_odd_exact hstep hsteps hmiddle hl
      | inexact order =>
          simpa [neuralRefineLocationOdd, hkZero, (cmp_eq_eq_iff _ _).2 hmiddle] using
            neuralInbetween_step_middle_odd_inexact hstep hsteps hmiddle hl
    · have hhalfK : half < k := by linarith
      have hhalfNext : half + 1 ≤ k := Int.add_one_le_iff.mpr hhalfK
      have hcell : steps < 2 * k := by linarith
      simpa [neuralRefineLocationOdd, hkZero, (cmp_eq_gt_iff _ _).2 hhigh] using
        neuralInbetween_step_high hstep hsteps hcell hk.2 hl

/-- Refine a local location, selecting the even or odd subdivision rule automatically. -/
noncomputable def neuralRefineLocation (steps k : ℤ)
    (location : NeuralLocation) : NeuralLocation :=
  if Even steps then
    neuralRefineLocationEven steps k location
  else
    neuralRefineLocationOdd steps k location

/-- Correctness of automatic local-to-global location refinement. -/
theorem neuralRefineLocation_correct {start step x : ℝ} {steps k : ℤ}
    {location : NeuralLocation} (hstep : 0 < step) (hsteps : 1 < steps)
    (hk : 0 ≤ k ∧ k < steps)
    (hl : NeuralInbetween
      (start + (k : ℝ) * step)
      (start + (k + 1 : ℤ) * step) x location) :
    NeuralInbetween start (start + (steps : ℝ) * step) x
      (neuralRefineLocation steps k location) := by
  by_cases heven : Even steps
  · rw [neuralRefineLocation, if_pos heven]
    exact neuralRefineLocationEven_correct hstep hsteps heven hk hl
  · rw [neuralRefineLocation, if_neg heven]
    exact neuralRefineLocationOdd_correct hstep hsteps
      (Int.not_even_iff_odd.mp heven) hk hl

end TorchLean.Floats
