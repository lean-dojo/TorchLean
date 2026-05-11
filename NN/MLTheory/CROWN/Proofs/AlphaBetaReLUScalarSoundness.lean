/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.CROWN.Cert.AlphaBetaCROWN
public import NN.MLTheory.CROWN.Cert.AlphaCROWN
public import NN.Spec.Layers.Activation
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

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

open _root_.Spec
open _root_.Spec.Tensor
open NN.MLTheory.CROWN
open NN.MLTheory.CROWN.Cert

noncomputable section

private lemma phaseConsistent_inactive_of_some (l u : ℝ)
    (h : phaseConsistentScalar? (α := ℝ) l u ReLUPhase.inactive = some ()) :
    u ≤ 0 := by
  -- `inactive` checks `¬ (0 < u)` via the executable `if u > 0 then none else some ()`.
  unfold phaseConsistentScalar? at h
  by_cases hu : u > 0
  · simp [hu] at h
  · have : ¬ u > 0 := hu
    exact le_of_not_gt this

private lemma phaseConsistent_active_of_some (l u : ℝ)
    (h : phaseConsistentScalar? (α := ℝ) l u ReLUPhase.active = some ()) :
    0 ≤ l := by
  -- `active` checks `¬ (l < 0)` via `if l < 0 then none else some ()`.
  unfold phaseConsistentScalar? at h
  by_cases hl : l < 0
  · simp [hl] at h
  · have : ¬ l < 0 := hl
    exact le_of_not_gt this

private lemma relu_relax_scalar_upper_real_runtime
    (l u x : ℝ) (hlx : l ≤ x) (hxu : x ≤ u) :
    let rp := NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar (α := ℝ) l u
    Activation.Math.reluSpec (α := ℝ) x ≤ rp.slope * x + rp.bias := by
  -- Same proof structure as the MLP-level lemma, but specialized to the runtime definition.
  unfold NN.MLTheory.CROWN.Runtime.Ops.ReLU.relaxScalar
  by_cases hu : u > 0
  · by_cases hlpos : l > 0
    · have hxpos : 0 < x := lt_of_lt_of_le hlpos hlx
      have hxnonneg : 0 ≤ x := le_of_lt hxpos
      simp [hu, hlpos, Activation.Math.reluSpec, max_eq_left hxnonneg]
    · have hle0 : l ≤ 0 := le_of_not_gt hlpos
      have hden : 0 < (u - l) := by linarith
      simp only [hu, hlpos, if_true, if_false]
      by_cases hxpos : 0 < x
      · have hxnonneg : 0 ≤ x := le_of_lt hxpos
        simp [Activation.Math.reluSpec, max_eq_left hxnonneg]
        -- Standard triangular relaxation inequality.
        have hx_to_goal : x ≤ u / (u - l) * (x - l) := by
          have hrewrite : (u - l) * x - u * (x - l) = l * (u - x) := by ring
          have hxux : 0 ≤ u - x := sub_nonneg.mpr hxu
          have hxmul_le : l * (u - x) ≤ 0 := mul_nonpos_of_nonpos_of_nonneg hle0 hxux
          have hmul_goal : (u - l) * x ≤ u * (x - l) := by
            have : (u - l) * x - u * (x - l) ≤ 0 := by simpa [hrewrite] using hxmul_le
            exact sub_nonpos.mp this
          have hx_to_goal' : x ≤ (u * (x - l)) / (u - l) := by
            have : x * (u - l) ≤ u * (x - l) := by simpa [mul_comm] using hmul_goal
            exact (le_div_iff₀ (G₀ := ℝ) hden).mpr this
          simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using hx_to_goal'
        have h2 : u / (u - l) * (x - l) = u / (u - l) * x + -(u / (u - l)) * l := by ring
        simpa [h2] using hx_to_goal
      · have hxle : x ≤ 0 := le_of_not_gt hxpos
        have h1 : u / (u - l) * x + -(u / (u - l) * l) = u / (u - l) * (x - l) := by ring
        have : 0 ≤ u / (u - l) * (x - l) := by
          apply mul_nonneg
          · have : 0 ≤ u := le_of_lt hu
            exact div_nonneg this (le_of_lt hden)
          · linarith
        simpa [Activation.Math.reluSpec, max_eq_right hxle, h1] using this
  · have hxle : x ≤ 0 := le_trans hxu (le_of_not_gt hu)
    simp [hu, Activation.Math.reluSpec, max_eq_right hxle]

private lemma relu_ge_alpha_mul (a z : ℝ) (ha0 : 0 ≤ a) (ha1 : a ≤ 1) :
    a * z ≤ Activation.Math.reluSpec (α := ℝ) z := by
  by_cases hz : z ≤ 0
  · have : a * z ≤ 0 := mul_nonpos_of_nonneg_of_nonpos ha0 hz
    simpa [Activation.Math.reluSpec, max_eq_right hz] using this
  · have hz' : 0 ≤ z := le_of_not_ge hz
    have : a * z ≤ (1 : ℝ) * z := mul_le_mul_of_nonneg_right ha1 hz'
    simpa [Activation.Math.reluSpec, max_eq_left hz', one_mul] using this

private lemma alphaRelaxLowerScalar_sound
    (l u a x : ℝ) (hlx : l ≤ x) (hxu : x ≤ u) (ha0 : 0 ≤ a) (ha1 : a ≤ 1) :
    let rp := alphaRelaxLowerScalar (α := ℝ) l u a
    rp.slope * x + rp.bias ≤ Activation.Math.reluSpec (α := ℝ) x := by
  unfold alphaRelaxLowerScalar
  by_cases hu : u > 0
  · by_cases hlpos : l > 0
    · have hxpos : 0 < x := lt_of_lt_of_le hlpos hlx
      have hxnonneg : 0 ≤ x := le_of_lt hxpos
      simp [hu, hlpos, Activation.Math.reluSpec, max_eq_left hxnonneg]
    · simp [hu, hlpos, relu_ge_alpha_mul (a := a) (z := x) ha0 ha1]
  · have hxle : x ≤ 0 := le_trans hxu (le_of_not_gt hu)
    simp [hu, Activation.Math.reluSpec, max_eq_right hxle]

theorem phaseRelaxUpperScalar_sound
    (l u x : ℝ) (hlx : l ≤ x) (hxu : x ≤ u) (ph : ReLUPhase)
    (hcons : phaseConsistentScalar? (α := ℝ) l u ph = some ()) :
    let rp := phaseRelaxUpperScalar (α := ℝ) l u ph
    Activation.Math.reluSpec (α := ℝ) x ≤ rp.slope * x + rp.bias := by
  cases ph with
  | inactive =>
      have hu0 : u ≤ 0 := phaseConsistent_inactive_of_some (l := l) (u := u) hcons
      have hxle : x ≤ 0 := le_trans hxu hu0
      simp [phaseRelaxUpperScalar, Activation.Math.reluSpec, max_eq_right hxle]
  | active =>
      have hl0 : 0 ≤ l := phaseConsistent_active_of_some (l := l) (u := u) hcons
      have hxnonneg : 0 ≤ x := le_trans hl0 hlx
      simp [phaseRelaxUpperScalar, Activation.Math.reluSpec, max_eq_left hxnonneg]
  | unstable =>
      simpa [phaseRelaxUpperScalar] using
        (relu_relax_scalar_upper_real_runtime (l := l) (u := u) (x := x) hlx hxu)

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
      simp [phaseRelaxLowerScalar, Activation.Math.reluSpec, max_eq_right hxle]
  | active =>
      have hl0 : 0 ≤ l := phaseConsistent_active_of_some (l := l) (u := u) hcons
      have hxnonneg : 0 ≤ x := le_trans hl0 hlx
      simp [phaseRelaxLowerScalar, Activation.Math.reluSpec, max_eq_left hxnonneg]
  | unstable =>
      simpa [phaseRelaxLowerScalar] using
        (alphaRelaxLowerScalar_sound (l := l) (u := u) (a := a) (x := x) hlx hxu ha0 ha1)

end
end NN.MLTheory.CROWN.Proofs
