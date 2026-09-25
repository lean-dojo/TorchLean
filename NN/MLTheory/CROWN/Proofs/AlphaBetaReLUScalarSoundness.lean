/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Cert.AlphaBetaCROWN
public import NN.MLTheory.CROWN.Proofs.AlphaReLULowerBound
public import NN.MLTheory.CROWN.Proofs.ReLUUpperBound

/-!
# Scalar soundness for α/β-ReLU relaxations (over `ℝ`)

This file proves the *operator-level* soundness facts used by α/β-CROWN at ReLU nodes:

* `phaseRelaxUpperScalar` is an upper bound on `relu` for any `x ∈ [l,u]`.
* `phaseRelaxLowerScalar` is a lower bound on `relu` for any `x ∈ [l,u]` (with `0 ≤ α ≤ 1`).

The β-phase cases (`inactive`/`active`) are exact, and the `unstable` case reduces to the
standard CROWN (upper) and α-CROWN (lower) relaxations.
-/

@[expose] public section


namespace NN.MLTheory.CROWN.Proofs

open Spec TorchLean
open TorchLean.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Cert

noncomputable section

private theorem phaseConsistent_inactive_of_some (l u : ℝ)
    (h : phaseConsistentScalar? (α := ℝ) l u ReLUPhase.inactive = some ()) :
    u ≤ 0 := by
  -- `inactive` checks `¬ (0 < u)` via the executable `if u > 0 then none else some ()`.
  unfold phaseConsistentScalar? at h
  by_cases hu : u > 0
  · simp [hu] at h
  · have : ¬ u > 0 := hu
    exact le_of_not_gt this

private theorem phaseConsistent_active_of_some (l u : ℝ)
    (h : phaseConsistentScalar? (α := ℝ) l u ReLUPhase.active = some ()) :
    0 ≤ l := by
  -- `active` checks `¬ (l < 0)` via `if l < 0 then none else some ()`.
  unfold phaseConsistentScalar? at h
  by_cases hl : l < 0
  · simp [hl] at h
  · have : ¬ l < 0 := hl
    exact le_of_not_gt this

/-- The phase-aware upper relaxation is sound for any phase the interval bounds actually admit.

The consistency hypothesis is what makes this true: on a claimed-inactive neuron the relaxation is
the constant zero line, which only dominates `relu` because `u ≤ 0` was checked first. -/
theorem phaseRelaxUpperScalar_sound
    (l u x : ℝ) (hlx : l ≤ x) (hxu : x ≤ u) (ph : ReLUPhase)
    (hcons : phaseConsistentScalar? (α := ℝ) l u ph = some ()) :
    let rp := phaseRelaxUpperScalar (α := ℝ) l u ph
    Activation.Math.reluSpec (α := ℝ) x ≤ rp.slope * x + rp.bias := by
  cases ph with
  | inactive =>
      have hu0 : u ≤ 0 := phaseConsistent_inactive_of_some (l := l) (u := u) hcons
      have hxle : x ≤ 0 := le_trans hxu hu0
      simp [phaseRelaxUpperScalar, Activation.Math.reluSpec_eq_max, max_eq_right hxle]
  | active =>
      have hl0 : 0 ≤ l := phaseConsistent_active_of_some (l := l) (u := u) hcons
      have hxnonneg : 0 ≤ x := le_trans hl0 hlx
      simp [phaseRelaxUpperScalar, Activation.Math.reluSpec_eq_max, max_eq_left hxnonneg]
  | unstable =>
      simpa [phaseRelaxUpperScalar] using
        (relu_relax_scalar_upper_real_runtime (l := l) (u := u) (x := x) hlx hxu)

/-- The phase-aware lower relaxation is sound, for every slope `a ∈ [0, 1]`.

The free `a` is the α of α-CROWN: on an unstable neuron any slope in the unit interval gives a valid
lower line, and the search is allowed to pick whichever one tightens the final bound. -/
theorem phaseRelaxLowerScalar_sound
    (l u a x : ℝ) (hlx : l ≤ x) (hxu : x ≤ u)
    (ha0 : 0 ≤ a) (ha1 : a ≤ 1)
    (ph : ReLUPhase) (hcons : phaseConsistentScalar? (α := ℝ) l u ph = some ()) :
    let rp := phaseRelaxLowerScalar (α := ℝ) l u a ph
    rp.slope * x + rp.bias ≤ Activation.Math.reluSpec (α := ℝ) x := by
  cases ph with
  | inactive =>
      have hu0 : u ≤ 0 := phaseConsistent_inactive_of_some (l := l) (u := u) hcons
      have hxle : x ≤ 0 := le_trans hxu hu0
      -- relaxation is 0; relu is 0 on `x ≤ 0`.
      simp [phaseRelaxLowerScalar, Activation.Math.reluSpec_eq_max, max_eq_right hxle]
  | active =>
      have hl0 : 0 ≤ l := phaseConsistent_active_of_some (l := l) (u := u) hcons
      have hxnonneg : 0 ≤ x := le_trans hl0 hlx
      simp [phaseRelaxLowerScalar, Activation.Math.reluSpec_eq_max, max_eq_left hxnonneg]
  | unstable =>
      simpa [phaseRelaxLowerScalar] using
        (alphaRelaxLowerScalar_sound
          (lower := l) (upper := u) (alpha := a) (input := x) hlx hxu ha0 ha1)

end
end NN.MLTheory.CROWN.Proofs
