/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Models.Hopfield.Update
public import TorchLean.Tensor.List
public import Std.Tactic.BVDecide.Normalize.BitVec
public import Mathlib.LinearAlgebra.Matrix.Symmetric

/-!
# Hopfield energy: single-step dynamics (spec layer)

The key global-dynamics lemma of the Hopfield literature, `energy_updateAt_le`, says that with
symmetric weights and zero diagonal an asynchronous update never increases the energy. It follows
from the exact one-step change `energy_updateAt_delta`, which also shows that the energy stays flat
when the updated unit sits at its threshold and drops strictly when a unit away from it flips.
That one-step change is equation [8] of Hopfield (1982), and his argument runs the same way.

We work over `ℝ`, where the classical energy argument is algebra. The spec's `net` and `energy` are
left folds from zero, and `net_eq_mulVec` and `energy_eq_energyVec` rewrite them with Mathlib's
`Matrix.mulVec` and `dotProduct`, the form the argument uses. Nothing here bounds a run over
`Float`, where the folds round.
-/

@[expose] public section


namespace TorchLean.Hopfield

open scoped BigOperators
open scoped Matrix

variable {n : Nat}

/-- Energy as a function of the activation vector: `-½ xᵀWx + θᵀx`. -/
noncomputable def energyVec (p : Params ℝ n) (x : Fin n → ℝ) : ℝ :=
  (-(1 / (2 : ℝ))) * (x ⬝ᵥ (Matrix.of p.W *ᵥ x)) + p.θ ⬝ᵥ x

/-- Over `ℝ`, the state-level `energy`, a left fold, equals `energyVec` of the activation vector. -/
theorem energy_eq_energyVec (p : Params ℝ n) (s : State n) :
    energy (α := ℝ) p s = energyVec (n := n) p (actVec (α := ℝ) s) := by
  have hsum (f : Fin n → ℝ) :
      (List.finRange n).foldl (fun (acc : ℝ) i => acc + f i) 0 = ∑ i, f i :=
    ((Fin.sum_univ_def f).trans (List.sum_eq_foldl.trans List.foldl_map)).symm
  have hquad (y : Fin n → ℝ) :
      y ⬝ᵥ (Matrix.of p.W *ᵥ y) = ∑ i, ∑ j, p.W i j * y i * y j := by
    simp only [dotProduct, Matrix.mulVec_apply_eq_sum, Matrix.of_apply, Finset.mul_sum]
    refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
    ring
  have hlin (y : Fin n → ℝ) : p.θ ⬝ᵥ y = ∑ i, p.θ i * y i := rfl
  have htwo : (1 : ℝ) + 1 = 2 := one_add_one_eq_two
  simp only [Hopfield.energy, energyVec, hquad, hlin, hsum, htwo]

/-- Over `ℝ`, the fold `net p s u` is entry `u` of `Matrix.of p.W *ᵥ actVec s`, the weight matrix
applied to the activation vector. -/
theorem net_eq_mulVec (p : Params ℝ n) (s : State n) (u : Fin n) :
    net (α := ℝ) p s u = (Matrix.of p.W *ᵥ actVec (α := ℝ) s) u := by
  simp only [Hopfield.net, Matrix.mulVec_apply_eq_sum, Matrix.of_apply]
  exact ((Fin.sum_univ_def _).trans (List.sum_eq_foldl.trans List.foldl_map)).symm

/-- An update of the state becomes a `Function.update` of the activation vector. -/
theorem actVec_updateAt (p : Params ℝ n) (s : State n) (u : Fin n) :
    actVec (α := ℝ) (updateAt (α := ℝ) p s u) =
      Function.update (actVec (α := ℝ) s) u
        (act (α := ℝ) (decide (p.θ u ≤ net (α := ℝ) p s u))) := by
  classical
  funext i
  by_cases h : i = u
  · subst h
    simp [updateAt, Hopfield.actVec, Function.update, Hopfield.act]
  · simp [updateAt, Hopfield.actVec, Function.update, h, Hopfield.act]

/-- Changing coordinate `u` shifts the net input at `u` only through the self-weight `W u u`:
`(W *ᵥ update x0 u xu') u = (W *ᵥ x0) u + W u u * (xu' - x0 u)`. -/
theorem mulVec_update_self (p : Params ℝ n) (x0 : Fin n → ℝ) (u : Fin n) (xu' : ℝ) :
    (Matrix.of p.W *ᵥ Function.update x0 u xu') u
      =
    (Matrix.of p.W *ᵥ x0) u + p.W u u * (xu' - x0 u) := by
  have hx : Function.update x0 u xu' = x0 + Pi.single u (xu' - x0 u) := by
    ext i
    by_cases hi : i = u
    · subst hi
      simp
    · simp [hi]
  have hcol : (Matrix.of p.W *ᵥ Pi.single u (xu' - x0 u)) u = p.W u u * (xu' - x0 u) :=
    dotProduct_single _ _ _
  rw [hx, Matrix.mulVec_add, Pi.add_apply, hcol]

/-- For symmetric weights with zero diagonal, a single-unit update changes the quadratic form by
`2 (x'ᵤ - xᵤ) · netᵤ`.

Symmetry is what merges the row and column contributions into a single factor of two, and the zero
diagonal is what removes the term where the unit acts on itself. Without either hypothesis the
energy can increase and the network need not converge. -/
theorem quad_delta_update (p : Params ℝ n) (hsym : (Matrix.of p.W).IsSymm)
    (hdiag : (Matrix.of p.W).diag = 0)
    (x0 : Fin n → ℝ) (u : Fin n) (xu' : ℝ) :
    Function.update x0 u xu' ⬝ᵥ (Matrix.of p.W *ᵥ Function.update x0 u xu') -
        x0 ⬝ᵥ (Matrix.of p.W *ᵥ x0)
      =
      2 * (xu' - x0 u) * (Matrix.of p.W *ᵥ x0) u := by
  have hx : Function.update x0 u xu' = x0 + Pi.single u (xu' - x0 u) := by
    ext i
    by_cases hi : i = u
    · subst hi
      simp
    · simp [hi]
  have hWuu : p.W u u = 0 := congrFun hdiag u
  have hcol : (Matrix.of p.W *ᵥ Pi.single u (xu' - x0 u)) u = p.W u u * (xu' - x0 u) :=
    dotProduct_single _ _ _
  rw [hx, Matrix.mulVec_add, dotProduct_add, add_dotProduct, add_dotProduct,
    hsym.dotProduct_mulVec_comm (x := x0) (y := Pi.single u (xu' - x0 u)),
    single_dotProduct, single_dotProduct, hcol, hWuu]
  ring

/-- For symmetric weights with zero diagonal, one update at `u` changes the energy by exactly
`-(x'ᵤ - xᵤ)(netᵤ - θᵤ)`.

The sign is forced: the update sets `x'ᵤ` to agree with the sign of `netᵤ - θᵤ`, so the product is
nonnegative and the energy change is nonpositive. The lemmas after it in this file are its
corollaries. -/
theorem energy_updateAt_delta (p : Params ℝ n)
    (hsym : (Matrix.of p.W).IsSymm) (hdiag : (Matrix.of p.W).diag = 0)
    (s : State n) (u : Fin n) :
    let x0 : Fin n → ℝ := actVec (α := ℝ) s
    let xu' : ℝ := act (α := ℝ) (decide (p.θ u ≤ net (α := ℝ) p s u))
    energy (α := ℝ) p (updateAt (α := ℝ) p s u) - energy (α := ℝ) p s
      =
    - (xu' - x0 u) * (net (α := ℝ) p s u - p.θ u) := by
  intro x0 xu'
  have hE1 : energy (α := ℝ) p (updateAt (α := ℝ) p s u) =
      energyVec (n := n) p (Function.update x0 u xu') :=
    (energy_eq_energyVec p _).trans (congrArg (energyVec p) (actVec_updateAt p s u))
  have hE0 : energy (α := ℝ) p s = energyVec (n := n) p x0 := energy_eq_energyVec p s
  have hnet : net (α := ℝ) p s u = (Matrix.of p.W *ᵥ x0) u := net_eq_mulVec p s u
  have hquad := quad_delta_update p hsym hdiag x0 u xu'
  have hlin : p.θ ⬝ᵥ Function.update x0 u xu' - p.θ ⬝ᵥ x0 = p.θ u * (xu' - x0 u) := by
    rw [Pi.update_eq_sub_add_single, dotProduct_add, dotProduct_sub, dotProduct_single,
      dotProduct_single]
    ring
  rw [hE1, hE0, hnet]
  simp only [energyVec]
  linear_combination (-(1 / (2 : ℝ))) * hquad + hlin

/-- For symmetric weights with zero diagonal, a single asynchronous update never increases the
energy. -/
theorem energy_updateAt_le (p : Params ℝ n)
    (hsym : (Matrix.of p.W).IsSymm) (hdiag : (Matrix.of p.W).diag = 0)
    (s : State n) (u : Fin n) :
    energy (α := ℝ) p (updateAt (α := ℝ) p s u) ≤ energy (α := ℝ) p s := by
  classical
  apply sub_nonpos.mp
  rw [energy_updateAt_delta p hsym hdiag s u]
  by_cases hθ : p.θ u ≤ net (α := ℝ) p s u
  · apply mul_nonpos_of_nonpos_of_nonneg ?_ (sub_nonneg.mpr hθ)
    cases hs : s u <;> simp [Hopfield.actVec, Hopfield.act, hθ, hs]
  · apply mul_nonpos_of_nonneg_of_nonpos ?_ (le_of_lt (sub_neg.mpr (lt_of_not_ge hθ)))
    cases hs : s u <;> simp [Hopfield.actVec, Hopfield.act, hθ, hs]

/-- For symmetric weights with zero diagonal, an update at a unit whose net input equals its
threshold leaves the energy unchanged, the flat case that the active-unit counter handles. -/
theorem energy_updateAt_eq_of_net_eq (p : Params ℝ n)
    (hsym : (Matrix.of p.W).IsSymm) (hdiag : (Matrix.of p.W).diag = 0)
    (s : State n) (u : Fin n) (hnet : net (α := ℝ) p s u = p.θ u) :
    energy (α := ℝ) p (updateAt (α := ℝ) p s u) = energy (α := ℝ) p s := by
  classical
  -- The delta formula vanishes because `net - θ = 0`.
  have hΔ :=
    energy_updateAt_delta (n := n) p hsym hdiag s u
  -- `simpa` unfolds the `let`s in `hΔ` and rewrites `net` to `θ` with `hnet`.
  have : energy (α := ℝ) p (updateAt (α := ℝ) p s u) - energy (α := ℝ) p s = 0 := by
    simpa [hnet, sub_self] using hΔ
  linarith

/-- For symmetric weights with zero diagonal, an update that changes the state at a unit whose net
input differs from its threshold strictly lowers the energy. -/
theorem energy_updateAt_lt_of_net_ne (p : Params ℝ n)
    (hsym : (Matrix.of p.W).IsSymm) (hdiag : (Matrix.of p.W).diag = 0)
    (s : State n) (u : Fin n)
    (hchange : updateAt (α := ℝ) p s u ≠ s)
    (hne : net (α := ℝ) p s u ≠ p.θ u) :
    energy (α := ℝ) p (updateAt (α := ℝ) p s u) < energy (α := ℝ) p s := by
  classical
  -- Reduce to showing the delta is negative.
  have hΔ :=
    energy_updateAt_delta (n := n) p hsym hdiag s u
  -- Case split on the update decision.
  by_cases hθ : p.θ u ≤ net (α := ℝ) p s u
  · -- The update sets `u` to `true`, so a change means `s u = false`.
    have hdec : decide (p.θ u ≤ net (α := ℝ) p s u) = true := by
      simp [hθ]
    have hsuf : s u = false := by
      simpa [hdec] using apply_eq_not_of_updateAt_ne p s u hchange
    have hθlt : p.θ u < net (α := ℝ) p s u := by
      exact lt_of_le_of_ne hθ (by simpa [eq_comm] using hne)
    have hpos : 0 < net (α := ℝ) p s u - p.θ u := by linarith
    have hΔ' :
        energy (α := ℝ) p (updateAt (α := ℝ) p s u) - energy (α := ℝ) p s
          =
        ((-1 : ℝ) + (-1 : ℝ)) * (net (α := ℝ) p s u - p.θ u) := by
      -- Under `hdec` and `hsuf`, `xu' - x0 u = 2`, so `ΔE = -2 * (net-θ)`.
      simpa [hsuf, hdec, Hopfield.actVec, Hopfield.act] using hΔ
    have : energy (α := ℝ) p (updateAt (α := ℝ) p s u) - energy (α := ℝ) p s < 0 := by
      rw [hΔ']
      nlinarith [hpos]
    linarith
  · -- The update sets `u` to `false`, so a change means `s u = true`.
    have hdec : decide (p.θ u ≤ net (α := ℝ) p s u) = false := by
      -- `decide p = false` when `¬ p`.
      simpa [decide_eq_false_iff_not] using (show ¬ p.θ u ≤ net (α := ℝ) p s u from hθ)
    have hsut : s u = true := by
      simpa [hdec] using apply_eq_not_of_updateAt_ne p s u hchange
    have hlt : net (α := ℝ) p s u < p.θ u := lt_of_not_ge hθ
    have hΔ' :
        energy (α := ℝ) p (updateAt (α := ℝ) p s u) - energy (α := ℝ) p s
          =
        ((1 : ℝ) + (1 : ℝ)) * (net (α := ℝ) p s u - p.θ u) := by
      -- Under `hdec` and `hsut`, `xu' - x0 u = -2`, so `ΔE = 2 * (net-θ)`.
      simpa [hsut, hdec, Hopfield.actVec, Hopfield.act] using hΔ
    have : energy (α := ℝ) p (updateAt (α := ℝ) p s u) - energy (α := ℝ) p s < 0 := by
      rw [hΔ']
      have : net (α := ℝ) p s u - p.θ u < 0 := by linarith
      nlinarith [this]
    linarith

end TorchLean.Hopfield
