/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.BigOperators.Group.Finset.Piecewise
public import NN.MLTheory.Proofs.Hopfield.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Hopfield energy: single-step dynamics (spec layer)

This file proves the key “global dynamics” lemma from the Hopfield literature:

> Under symmetric, zero-diagonal weights, the Hopfield energy is non-increasing under an
> asynchronous update.

We work over `ℝ`, where the classical energy argument is algebraic. The executable Hopfield
implementation uses `IEEE32Exec`; floating-point executions are connected to this theorem only
through explicit runtime/rounding bridge statements, not by silently reusing real arithmetic laws.
-/

@[expose] public section


namespace NN.MLTheory.Proofs.Hopfield

open scoped BigOperators
open Spec

open Spec.Hopfield

variable {n : Nat}

/-- Symmetry condition on Hopfield weights: `W i j = W j i`. -/
def SymmetricW (p : Params ℝ n) : Prop := ∀ i j, p.W i j = p.W j i
/-- Zero-diagonal condition on Hopfield weights: `W i i = 0`. -/
def DiagonalZero (p : Params ℝ n) : Prop := ∀ i, p.W i i = 0

noncomputable def x (s : State n) : Fin n → ℝ :=
  actVec (α := ℝ) s

noncomputable def U : Finset (Fin n) := Finset.univ

noncomputable def netx (p : Params ℝ n) (x : Fin n → ℝ) (u : Fin n) : ℝ :=
  ∑ j ∈ (U (n := n)), p.W u j * x j

noncomputable def quad (p : Params ℝ n) (x : Fin n → ℝ) : ℝ :=
  ∑ i ∈ (U (n := n)), ∑ j ∈ (U (n := n)), p.W i j * x i * x j

noncomputable def energyU (p : Params ℝ n) (x : Fin n → ℝ) : ℝ :=
  (-(1 / (2 : ℝ))) * quad (n := n) p x + ∑ i ∈ (U (n := n)), p.θ i * x i

lemma energy_eq_energyU (p : Params ℝ n) (s : State n) :
    energy (α := ℝ) p s = energyU (n := n) p (x (n := n) s) := by
  classical
  -- `energy` uses `∑ i : Fin n`, which is definally over `Finset.univ`.
  simp [Spec.Hopfield.energy, energyU, quad, x, U, Spec.Hopfield.actVec, mul_left_comm, mul_comm,
    add_comm]

lemma net_eq_netx (p : Params ℝ n) (s : State n) (u : Fin n) :
    net (α := ℝ) p s u = netx (n := n) p (x (n := n) s) u := by
  simp [Spec.Hopfield.net, Spec.Hopfield.mulVec, netx, x, U, Spec.Hopfield.actVec]

lemma x_updateAt_eq_update (p : Params ℝ n) (s : State n) (u : Fin n) :
    x (n := n) (updateAt (α := ℝ) p s u) =
      Function.update (x (n := n) s) u (act (α := ℝ) (decide (p.θ u ≤ net (α := ℝ) p s u))) := by
  classical
  funext i
  by_cases h : i = u
  · subst h
    simp [x, updateAt, Spec.Hopfield.actVec, Function.update, Spec.Hopfield.act]
  · simp [x, updateAt, Spec.Hopfield.actVec, Function.update, h, Spec.Hopfield.act]

lemma netx_update_eq (p : Params ℝ n) (x0 : Fin n → ℝ) (u : Fin n) (xu' : ℝ) :
    netx (n := n) p (Function.update x0 u xu') u
      =
    netx (n := n) p x0 u + p.W u u * (xu' - x0 u) := by
  classical
  -- Only the `j=u` term changes.
  have hu : u ∈ (U (n := n)) := by simp [U]
  -- Decompose `∑ j ∈ U, W u j * x' j` using `sum_update_of_mem`.
  let f : Fin n → ℝ := fun j => p.W u j * x0 j
  have hs1 :
      (∑ j ∈ (U (n := n)), p.W u j * (Function.update x0 u xu' j))
        =
      (p.W u u * xu') + ∑ j ∈ (U (n := n) \ {u}), p.W u j * x0 j := by
    have hx :
        (fun j => p.W u j * (Function.update x0 u xu' j)) =
          Function.update f u (p.W u u * xu') := by
      funext j
      by_cases hj : j = u
      · subst hj
        simp [f]
      · simp [f, Function.update, hj]
    -- Apply `sum_update_of_mem` to `f` and rewrite the LHS using `hx`.
    simpa [hx, U] using
      (Finset.sum_update_of_mem (s := (U (n := n))) (i := u) hu f (p.W u u * xu'))
  have hs0 :
      (∑ j ∈ (U (n := n)), p.W u j * x0 j)
        =
      (p.W u u * x0 u) + ∑ j ∈ (U (n := n) \ {u}), p.W u j * x0 j := by
    rw [Finset.sum_eq_add_sum_diff_singleton_of_mem hu]
  -- Subtract and simplify.
  calc
    netx (n := n) p (Function.update x0 u xu') u
        =
      (p.W u u * xu') + ∑ j ∈ (U (n := n) \ {u}), p.W u j * x0 j := by
        simp [netx, hs1]
    _ =
      ((p.W u u * x0 u) + ∑ j ∈ (U (n := n) \ {u}), p.W u j * x0 j) + p.W u u * (xu' - x0 u) := by
        ring
    _ = netx (n := n) p x0 u + p.W u u * (xu' - x0 u) := by
        simp [netx, hs0]

lemma quad_inner_delta_ne (p : Params ℝ n) {u i : Fin n} (hi : i ≠ u)
    (x0 : Fin n → ℝ) (xu' : ℝ) :
    (∑ j ∈ (U (n := n)), p.W i j * x0 i * (Function.update x0 u xu' j))
      -
    (∑ j ∈ (U (n := n)), p.W i j * x0 i * x0 j)
    =
    p.W i u * x0 i * (xu' - x0 u) := by
  classical
  have hu : u ∈ (U (n := n)) := by simp [U]
  let f : Fin n → ℝ := fun j => p.W i j * x0 i * x0 j
  have hs1 :
      (∑ j ∈ (U (n := n)), p.W i j * x0 i * (Function.update x0 u xu' j))
        =
      (p.W i u * x0 i * xu') + ∑ j ∈ (U (n := n) \ {u}), p.W i j * x0 i * x0 j := by
    have := (Finset.sum_update_of_mem (s := (U (n := n))) (i := u) hu f (p.W i u * x0 i * xu'))
    -- Match the update: only `x0 j` changes at `u`.
    -- For `j=u` we pick the updated value `xu'`; for `j≠u` it's the old `x0 j`.
    have hx :
        (fun j => p.W i j * x0 i * (Function.update x0 u xu' j))
          =
        (fun j => Function.update (fun j => p.W i j * x0 i * x0 j) u (p.W i u * x0 i * xu') j) := by
      funext j
      by_cases hj : j = u
      · subst hj; simp
      · simp [Function.update, hj]
    simpa [hx, U, f] using this
  have hs0 :
      (∑ j ∈ (U (n := n)), p.W i j * x0 i * x0 j)
        =
      (p.W i u * x0 i * x0 u) + ∑ j ∈ (U (n := n) \ {u}), p.W i j * x0 i * x0 j := by
    rw [Finset.sum_eq_add_sum_diff_singleton_of_mem hu]
  calc
    (∑ j ∈ (U (n := n)), p.W i j * x0 i * (Function.update x0 u xu' j))
        -
      (∑ j ∈ (U (n := n)), p.W i j * x0 i * x0 j)
        =
      ((p.W i u * x0 i * xu') + ∑ j ∈ (U (n := n) \ {u}), p.W i j * x0 i * x0 j)
        -
      ((p.W i u * x0 i * x0 u) + ∑ j ∈ (U (n := n) \ {u}), p.W i j * x0 i * x0 j) := by
        simp [hs1, hs0]
    _ = p.W i u * x0 i * (xu' - x0 u) := by
        ring

lemma quad_delta_update (p : Params ℝ n) (hsym : SymmetricW (n := n) p) (hdiag : DiagonalZero (n :=
  n) p)
    (x0 : Fin n → ℝ) (u : Fin n) (xu' : ℝ) :
    quad (n := n) p (Function.update x0 u xu') - quad (n := n) p x0
      =
      2 * (xu' - x0 u) * netx (n := n) p x0 u := by
  classical
  have hu : u ∈ (U (n := n)) := by simp [U]
  let x1 : Fin n → ℝ := Function.update x0 u xu'
  have hx1u : x1 u = xu' := by simp [x1]
  have hx0u : x0 u = x0 u := rfl
  -- Split `quad` into the `i=u` row and the rest.
  have hsplit1 :
      quad (n := n) p x1
        =
      (∑ j ∈ (U (n := n)), p.W u j * x1 u * x1 j)
        +
      ∑ i ∈ (U (n := n) \ {u}), ∑ j ∈ (U (n := n)), p.W i j * x1 i * x1 j := by
    -- `sum_eq_add_sum_diff_singleton` with `s = U`.
    simp [quad]
    rw [Finset.sum_eq_add_sum_diff_singleton_of_mem hu]
  have hsplit0 :
      quad (n := n) p x0
        =
      (∑ j ∈ (U (n := n)), p.W u j * x0 u * x0 j)
        +
      ∑ i ∈ (U (n := n) \ {u}), ∑ j ∈ (U (n := n)), p.W i j * x0 i * x0 j := by
    simp [quad]
    rw [Finset.sum_eq_add_sum_diff_singleton_of_mem hu]
  -- For `i ≠ u`, `x1 i = x0 i`, and only the inner sum's `j=u` term changes.
  have hrest :
      (∑ i ∈ (U (n := n) \ {u}), ∑ j ∈ (U (n := n)), p.W i j * x1 i * x1 j)
        -
      (∑ i ∈ (U (n := n) \ {u}), ∑ j ∈ (U (n := n)), p.W i j * x0 i * x0 j)
        =
      (∑ i ∈ (U (n := n) \ {u}), p.W i u * x0 i) * (xu' - x0 u) := by
    -- Compute termwise via `quad_inner_delta_ne`.
    have :
        (∑ i ∈ (U (n := n) \ {u}),
          ((∑ j ∈ (U (n := n)), p.W i j * x0 i * x1 j) - (∑ j ∈ (U (n := n)), p.W i j * x0 i * x0
            j)))
          =
        (∑ i ∈ (U (n := n) \ {u}), p.W i u * x0 i * (xu' - x0 u)) := by
      refine Finset.sum_congr rfl ?_
      intro i hi
      have hiu : i ≠ u := by
        have : i ∉ ({u} : Finset (Fin n)) := (Finset.mem_sdiff.1 hi).2
        simpa [Finset.mem_singleton] using this
      -- Replace `x1` by `update`.
      have hxj : (fun j => x1 j) = Function.update x0 u xu' := by rfl
      -- Use the delta lemma.
      simpa [x1, Function.update] using (quad_inner_delta_ne (n := n) (p := p) (u := u) (i := i) hiu
        x0 xu')
    -- Replace `x1 i` by `x0 i` (since `i≠u`) and factor the constant term.
    -- Also note `x1 i = x0 i` on `U \ {u}`.
    have hxi : ∀ i ∈ (U (n := n) \ {u}), x1 i = x0 i := by
      intro i hi
      have hiu : i ≠ u := by
        have : i ∉ ({u} : Finset (Fin n)) := (Finset.mem_sdiff.1 hi).2
        simpa [Finset.mem_singleton] using this
      simp [x1, hiu]
    -- Use `sum_sub_distrib`, then rewrite `x1 i = x0 i` for `i ≠ u`.
    have hsub :
        (∑ i ∈ (U (n := n) \ {u}), ∑ j ∈ (U (n := n)), p.W i j * x1 i * x1 j)
          -
        (∑ i ∈ (U (n := n) \ {u}), ∑ j ∈ (U (n := n)), p.W i j * x0 i * x0 j)
          =
        ∑ i ∈ (U (n := n) \ {u}),
          ((∑ j ∈ (U (n := n)), p.W i j * x1 i * x1 j) - (∑ j ∈ (U (n := n)), p.W i j * x0 i * x0
            j)) := by
      -- `sum_sub_distrib` is stated as `∑ i∈s, (a i - b i) = (∑ i∈s, a i) - (∑ i∈s, b i)`.
      -- We rewrite it in the direction we need.
      exact (Eq.symm (Finset.sum_sub_distrib (s := (U (n := n) \ {u}))
        (f := fun i => ∑ j ∈ (U (n := n)), p.W i j * x1 i * x1 j)
        (g := fun i => ∑ j ∈ (U (n := n)), p.W i j * x0 i * x0 j)))
    calc
      (∑ i ∈ (U (n := n) \ {u}), ∑ j ∈ (U (n := n)), p.W i j * x1 i * x1 j)
          -
        (∑ i ∈ (U (n := n) \ {u}), ∑ j ∈ (U (n := n)), p.W i j * x0 i * x0 j)
          =
        ∑ i ∈ (U (n := n) \ {u}),
          ((∑ j ∈ (U (n := n)), p.W i j * x1 i * x1 j) - (∑ j ∈ (U (n := n)), p.W i j * x0 i * x0
            j)) := hsub
      _ = (∑ i ∈ (U (n := n) \ {u}), p.W i u * x0 i * (xu' - x0 u)) := by
        refine Finset.sum_congr rfl ?_
        intro i hi
        have hix : x1 i = x0 i := hxi i hi
        -- Replace `x1 i` with `x0 i` and apply the inner delta lemma.
        simpa [hix, x1] using (quad_inner_delta_ne (n := n) (p := p) (u := u) (i := i)
          (by
            have : i ∉ ({u} : Finset (Fin n)) := (Finset.mem_sdiff.1 hi).2
            simpa [Finset.mem_singleton] using this)
          x0 xu')
      _ = (∑ i ∈ (U (n := n) \ {u}), p.W i u * x0 i) * (xu' - x0 u) := by
        simp [Finset.sum_mul]
  -- Row `u`: use diagonal-zero to show `netx` doesn't change when updating `x0 u`.
  have huRow :
      (∑ j ∈ (U (n := n)), p.W u j * x1 u * x1 j)
        -
      (∑ j ∈ (U (n := n)), p.W u j * x0 u * x0 j)
        =
      (xu' - x0 u) * netx (n := n) p x0 u := by
    -- Factor out `x1 u` / `x0 u`.
    have hfac1 :
        (∑ j ∈ (U (n := n)), p.W u j * x1 u * x1 j) =
          x1 u * (∑ j ∈ (U (n := n)), p.W u j * x1 j) := by
      simp [Finset.mul_sum, mul_left_comm, mul_comm]
    have hfac0 :
        (∑ j ∈ (U (n := n)), p.W u j * x0 u * x0 j) =
          x0 u * (∑ j ∈ (U (n := n)), p.W u j * x0 j) := by
      simp [Finset.mul_sum, mul_left_comm, mul_comm]
    have hnet : (∑ j ∈ (U (n := n)), p.W u j * x1 j) = (∑ j ∈ (U (n := n)), p.W u j * x0 j) := by
      -- net changes only by `W u u * (xu' - x0 u)`, and `W u u = 0`.
      have h := netx_update_eq (n := n) (p := p) (x0 := x0) (u := u) (xu' := xu')
      have hdiag' : p.W u u = 0 := hdiag u
      -- `netx x1 u = netx x0 u + Wuu*(...)`, but `netx x1 u` is exactly this sum.
      simpa [netx, x1, hdiag'] using h
    -- Finish.
    calc
      (∑ j ∈ (U (n := n)), p.W u j * x1 u * x1 j)
          -
        (∑ j ∈ (U (n := n)), p.W u j * x0 u * x0 j)
          =
        x1 u * (∑ j ∈ (U (n := n)), p.W u j * x1 j) - x0 u * (∑ j ∈ (U (n := n)), p.W u j * x0 j) :=
          by
          simp [hfac1, hfac0]
      _ = xu' * (∑ j ∈ (U (n := n)), p.W u j * x0 j) - x0 u * (∑ j ∈ (U (n := n)), p.W u j * x0 j)
        := by
          simp [hx1u, hnet]
      _ = (xu' - x0 u) * (∑ j ∈ (U (n := n)), p.W u j * x0 j) := by ring
      _ = (xu' - x0 u) * netx (n := n) p x0 u := by simp [netx]
  -- Symmetry: relate `∑_{i≠u} W i u x_i` to `netx`.
  have hsymRest :
      (∑ i ∈ (U (n := n) \ {u}), p.W i u * x0 i) = netx (n := n) p x0 u := by
    -- Expand `netx` into the `i=u` term plus the remaining sum.
    have hu' : u ∈ (U (n := n)) := by simp [U]
    have hs :
        netx (n := n) p x0 u =
          (p.W u u * x0 u) + ∑ i ∈ (U (n := n) \ {u}), p.W u i * x0 i := by
      simp [netx]
      rw [Finset.sum_eq_add_sum_diff_singleton_of_mem hu']
    have hdiag' : p.W u u = 0 := hdiag u
    have hswap :
        (∑ i ∈ (U (n := n) \ {u}), p.W i u * x0 i) =
          ∑ i ∈ (U (n := n) \ {u}), p.W u i * x0 i := by
      refine Finset.sum_congr rfl ?_
      intro i _hi
      simp [hsym i u]
    -- The remaining sum equals `netx` because `W u u = 0`.
    calc
      (∑ i ∈ (U (n := n) \ {u}), p.W i u * x0 i)
          =
        ∑ i ∈ (U (n := n) \ {u}), p.W u i * x0 i := hswap
      _ = netx (n := n) p x0 u := by
        -- From `hs` and `Wuu=0`, `netx = tail`.
        have htail :
            netx (n := n) p x0 u = ∑ i ∈ (U (n := n) \ {u}), p.W u i * x0 i := by
          simp [hs, hdiag']
        simpa using htail.symm
  -- Combine `u` row and rest.
  let row1 : ℝ := ∑ j ∈ (U (n := n)), p.W u j * x1 u * x1 j
  let row0 : ℝ := ∑ j ∈ (U (n := n)), p.W u j * x0 u * x0 j
  let rest1 : ℝ := ∑ i ∈ (U (n := n) \ {u}), ∑ j ∈ (U (n := n)), p.W i j * x1 i * x1 j
  let rest0 : ℝ := ∑ i ∈ (U (n := n) \ {u}), ∑ j ∈ (U (n := n)), p.W i j * x0 i * x0 j
  have hdecomp : quad (n := n) p x1 - quad (n := n) p x0 = (row1 - row0) + (rest1 - rest0) := by
    -- Substitute the split forms and regroup.
    simp [row1, row0, rest1, rest0, hsplit1, hsplit0]
    ring
  calc
    quad (n := n) p x1 - quad (n := n) p x0
        = (row1 - row0) + (rest1 - rest0) := hdecomp
    _ = (xu' - x0 u) * netx (n := n) p x0 u + (netx (n := n) p x0 u) * (xu' - x0 u) := by
          -- Rewrite via `huRow` and `hrest`, then use `hsymRest`. Let `ring` handle reassociation.
          simp [row1, row0, rest1, rest0, huRow, hrest, hsymRest]
    _ = 2 * (xu' - x0 u) * netx (n := n) p x0 u := by
          ring

theorem energy_updateAt_le (p : Params ℝ n) (hsym : SymmetricW (n := n) p) (hdiag : DiagonalZero (n
  := n) p)
    (s : State n) (u : Fin n) :
    energy (α := ℝ) p (updateAt (α := ℝ) p s u) ≤ energy (α := ℝ) p s := by
  classical
  -- Let `x0` be the activation vector, and `xu'` the updated activation at `u`.
  let x0 : Fin n → ℝ := x (n := n) s
  let xu' : ℝ := act (α := ℝ) (decide (p.θ u ≤ net (α := ℝ) p s u))
  let x1 : Fin n → ℝ := Function.update x0 u xu'
  -- Rewrite `actVec` after update.
  have hx1 : x (n := n) (updateAt (α := ℝ) p s u) = x1 := by
    simpa [x0, xu', x1] using x_updateAt_eq_update (n := n) p s u
  -- Use the quadratic delta formula.
  have hquad :
      quad (n := n) p x1 - quad (n := n) p x0 =
        2 * (xu' - x0 u) * netx (n := n) p x0 u :=
    quad_delta_update (n := n) p hsym hdiag x0 u xu'
  -- Linear term changes only at `u`.
  have hlin :
      (∑ i ∈ (U (n := n)), p.θ i * x1 i) - (∑ i ∈ (U (n := n)), p.θ i * x0 i) = p.θ u * (xu' - x0 u)
        := by
    have hu : u ∈ (U (n := n)) := by simp [U]
    let f : Fin n → ℝ := fun i => p.θ i * x0 i
    have hs1 :
        (∑ i ∈ (U (n := n)), p.θ i * x1 i)
          =
        (p.θ u * xu') + ∑ i ∈ (U (n := n) \ {u}), p.θ i * x0 i := by
      have := (Finset.sum_update_of_mem (s := (U (n := n))) (i := u) hu f (p.θ u * xu'))
      -- Match `x1` as an update on `x0`.
      have hx : (fun i => p.θ i * x1 i) =
          (fun i => Function.update (fun i => p.θ i * x0 i) u (p.θ u * xu') i) := by
        funext i
        by_cases hi : i = u
        · subst hi; simp [x1]
        · simp [x1, Function.update, hi]
      simpa [hx, U, f] using this
    have hs0 :
        (∑ i ∈ (U (n := n)), p.θ i * x0 i)
          =
        (p.θ u * x0 u) + ∑ i ∈ (U (n := n) \ {u}), p.θ i * x0 i := by
      rw [Finset.sum_eq_add_sum_diff_singleton_of_mem hu]
    calc
      (∑ i ∈ (U (n := n)), p.θ i * x1 i) - (∑ i ∈ (U (n := n)), p.θ i * x0 i)
          =
        ((p.θ u * xu') + ∑ i ∈ (U (n := n) \ {u}), p.θ i * x0 i)
          -
        ((p.θ u * x0 u) + ∑ i ∈ (U (n := n) \ {u}), p.θ i * x0 i) := by
        simp [hs1, hs0]
      _ = p.θ u * (xu' - x0 u) := by ring
  -- Energy is `-(1/2)*quad + linear`.
  -- Show the net change is `-(xu'-xu)*(net-θ) ≤ 0`.
  have hnet : netx (n := n) p x0 u = net (α := ℝ) p s u := by
    symm; simpa [x0] using net_eq_netx (n := n) p s u
  -- Compute `ΔE = E(x1) - E(x0)` and show it is ≤ 0.
  -- Then rewrite `E(updateAt s) = E(x1)` by `hx1`.
  have hΔ :
      (-(1 / (2 : ℝ))) * (quad (n := n) p x1 - quad (n := n) p x0) +
          ((∑ i ∈ (U (n := n)), p.θ i * x1 i) - (∑ i ∈ (U (n := n)), p.θ i * x0 i))
        =
      - (xu' - x0 u) * (net (α := ℝ) p s u - p.θ u) := by
    -- Substitute `hquad`/`hlin` and simplify.
    simp [hquad, hlin, hnet]
    ring
  -- Prove the RHS is ≤ 0 by case split on the update decision.
  have hRhs : - (xu' - x0 u) * (net (α := ℝ) p s u - p.θ u) ≤ 0 := by
    by_cases hθ : p.θ u ≤ net (α := ℝ) p s u
    · have hnetθ : 0 ≤ net (α := ℝ) p s u - p.θ u := by linarith
      cases hs : s u <;> (simp [x0, xu', x, Spec.Hopfield.actVec, Spec.Hopfield.act, hθ, hs] ; try
        linarith)
    · have hnetθ : net (α := ℝ) p s u - p.θ u < 0 := by
        have : net (α := ℝ) p s u < p.θ u := lt_of_not_ge hθ
        linarith
      cases hs : s u <;> (simp [x0, xu', x, Spec.Hopfield.actVec, Spec.Hopfield.act, hθ, hs] ; try
        linarith)
  -- Finish: rewrite energies via `energyU`, then use `hΔ` + `hRhs`.
  let s' : State n := updateAt (α := ℝ) p s u
  have hE0 : energy (α := ℝ) p s = energyU (n := n) p x0 := by
    simpa [x0] using (energy_eq_energyU (n := n) p s)
  have hE1 : energy (α := ℝ) p s' = energyU (n := n) p x1 := by
    -- Use `hx1 : x s' = x1`.
    have hx1' : x (n := n) s' = x1 := by simpa [s'] using hx1
    simpa [hx1'] using (energy_eq_energyU (n := n) p s')
  have hdiff :
      energy (α := ℝ) p s' - energy (α := ℝ) p s
        =
      (-(1 / (2 : ℝ))) * (quad (n := n) p x1 - quad (n := n) p x0) +
        ((∑ i ∈ (U (n := n)), p.θ i * x1 i) - (∑ i ∈ (U (n := n)), p.θ i * x0 i)) := by
    -- Expand the two `energyU` forms and regroup.
    simp [hE0, hE1, energyU, sub_eq_add_neg, add_assoc, add_left_comm, add_comm, mul_add]
  -- Use `hΔ` to rewrite the RHS, then apply `hRhs`.
  have : energy (α := ℝ) p s' - energy (α := ℝ) p s ≤ 0 := by
    -- Replace using `hdiff` then `hΔ`.
    -- `hΔ` was exactly the RHS of `hdiff`.
    have : energy (α := ℝ) p s' - energy (α := ℝ) p s = - (xu' - x0 u) * (net (α := ℝ) p s u - p.θ
      u) := by
      simpa [hdiff] using hΔ
    -- Conclude with `hRhs`.
    simpa [this] using hRhs
  -- `E' - E ≤ 0` implies `E' ≤ E`.
  linarith

theorem energy_updateAt_delta (p : Params ℝ n)
    (hsym : SymmetricW (n := n) p) (hdiag : DiagonalZero (n := n) p)
    (s : State n) (u : Fin n) :
    let x0 : Fin n → ℝ := x (n := n) s
    let xu' : ℝ := act (α := ℝ) (decide (p.θ u ≤ net (α := ℝ) p s u))
    energy (α := ℝ) p (updateAt (α := ℝ) p s u) - energy (α := ℝ) p s
      =
    - (xu' - x0 u) * (net (α := ℝ) p s u - p.θ u) := by
  classical
  intro x0 xu'
  let x1 : Fin n → ℝ := Function.update x0 u xu'
  have hx1 : x (n := n) (updateAt (α := ℝ) p s u) = x1 := by
    simpa [x0, xu', x1] using x_updateAt_eq_update (n := n) p s u
  have hquad :
      quad (n := n) p x1 - quad (n := n) p x0 =
        2 * (xu' - x0 u) * netx (n := n) p x0 u :=
    quad_delta_update (n := n) p hsym hdiag x0 u xu'
  have hlin :
      (∑ i ∈ (U (n := n)), p.θ i * x1 i) - (∑ i ∈ (U (n := n)), p.θ i * x0 i) = p.θ u * (xu' - x0 u)
        := by
    have hu : u ∈ (U (n := n)) := by simp [U]
    let f : Fin n → ℝ := fun i => p.θ i * x0 i
    have hs1 :
        (∑ i ∈ (U (n := n)), p.θ i * x1 i)
          =
        (p.θ u * xu') + ∑ i ∈ (U (n := n) \ {u}), p.θ i * x0 i := by
      have := (Finset.sum_update_of_mem (s := (U (n := n))) (i := u) hu f (p.θ u * xu'))
      have hx : (fun i => p.θ i * x1 i) =
          (fun i => Function.update (fun i => p.θ i * x0 i) u (p.θ u * xu') i) := by
        funext i
        by_cases hi : i = u
        · subst hi; simp [x1]
        · simp [x1, Function.update, hi]
      simpa [hx, U, f] using this
    have hs0 :
        (∑ i ∈ (U (n := n)), p.θ i * x0 i)
          =
        (p.θ u * x0 u) + ∑ i ∈ (U (n := n) \ {u}), p.θ i * x0 i := by
      rw [Finset.sum_eq_add_sum_diff_singleton_of_mem hu]
    calc
      (∑ i ∈ (U (n := n)), p.θ i * x1 i) - (∑ i ∈ (U (n := n)), p.θ i * x0 i)
          =
        ((p.θ u * xu') + ∑ i ∈ (U (n := n) \ {u}), p.θ i * x0 i)
          -
        ((p.θ u * x0 u) + ∑ i ∈ (U (n := n) \ {u}), p.θ i * x0 i) := by
          simp [hs1, hs0]
      _ = p.θ u * (xu' - x0 u) := by ring
  have hnet : netx (n := n) p x0 u = net (α := ℝ) p s u := by
    symm; simpa [x0] using net_eq_netx (n := n) p s u
  -- Convert the energy difference to `quad/lin` differences via `energyU`, then substitute.
  let s' : State n := updateAt (α := ℝ) p s u
  have hE0 : energy (α := ℝ) p s = energyU (n := n) p x0 := by
    simpa [x0] using (energy_eq_energyU (n := n) p s)
  have hE1 : energy (α := ℝ) p s' = energyU (n := n) p x1 := by
    have hx1' : x (n := n) s' = x1 := by simpa [s'] using hx1
    simpa [hx1'] using (energy_eq_energyU (n := n) p s')
  have hdiff :
      energy (α := ℝ) p s' - energy (α := ℝ) p s
        =
      (-(1 / (2 : ℝ))) * (quad (n := n) p x1 - quad (n := n) p x0) +
        ((∑ i ∈ (U (n := n)), p.θ i * x1 i) - (∑ i ∈ (U (n := n)), p.θ i * x0 i)) := by
    simp [hE0, hE1, energyU, sub_eq_add_neg, add_assoc, add_left_comm, add_comm, mul_add]
  -- Substitute the `quad`/linear deltas and simplify.
  calc
    energy (α := ℝ) p s' - energy (α := ℝ) p s
        =
      (-(1 / (2 : ℝ))) * (quad (n := n) p x1 - quad (n := n) p x0) +
        ((∑ i ∈ (U (n := n)), p.θ i * x1 i) - (∑ i ∈ (U (n := n)), p.θ i * x0 i)) := hdiff
    _ = - (xu' - x0 u) * (net (α := ℝ) p s u - p.θ u) := by
        simp [hquad, hlin, hnet]
        ring

theorem energy_updateAt_eq_of_net_eq_theta (p : Params ℝ n)
    (hsym : SymmetricW (n := n) p) (hdiag : DiagonalZero (n := n) p)
    (s : State n) (u : Fin n) (hnet : net (α := ℝ) p s u = p.θ u) :
    energy (α := ℝ) p (updateAt (α := ℝ) p s u) = energy (α := ℝ) p s := by
  classical
  -- Use the explicit delta formula and `net - θ = 0`.
  have hΔ :=
    energy_updateAt_delta (n := n) p hsym hdiag s u
  -- Unfold the `let`s via `simp` by rewriting `net`.
  have : energy (α := ℝ) p (updateAt (α := ℝ) p s u) - energy (α := ℝ) p s = 0 := by
    simpa [hnet, sub_self] using hΔ
  linarith

theorem energy_updateAt_lt_of_change_of_ne (p : Params ℝ n)
    (hsym : SymmetricW (n := n) p) (hdiag : DiagonalZero (n := n) p)
    (s : State n) (u : Fin n)
    (hchange : updateAt (α := ℝ) p s u ≠ s)
    (hne : net (α := ℝ) p s u ≠ p.θ u) :
    energy (α := ℝ) p (updateAt (α := ℝ) p s u) < energy (α := ℝ) p s := by
  classical
  -- Reduce to showing the delta is negative.
  have hΔ :=
    energy_updateAt_delta (n := n) p hsym hdiag s u
  -- Extract that `u` is indeed the changed coordinate.
  have hsu : updateAt (α := ℝ) p s u u ≠ s u := by
    intro hEq
    apply hchange
    funext i
    by_cases hi : i = u
    · subst hi; simpa using hEq
    · simp [updateAt_apply_ne (p := p) (s := s) (u := u) (v := i) hi]
  -- Case split on the update decision.
  by_cases hθ : p.θ u ≤ net (α := ℝ) p s u
  · -- Update sets `u := true`, so changing means it was `false`.
    have hdec : decide (p.θ u ≤ net (α := ℝ) p s u) = true := by
      simp [hθ]
    have hup : updateAt (α := ℝ) p s u u = true := by
      have hu'' :
          updateAt (α := ℝ) p s u u = decide (p.θ u ≤ net (α := ℝ) p s u) := by
        simp [updateAt]
      exact hu''.trans hdec
    have hsuf : s u = false := by
      cases hsu0 : s u <;> try rfl
      -- If `s u = true`, then `updateAt` wouldn't change at `u`.
      exfalso
      exact hsu (by simpa [hsu0] using hup)
    have hθlt : p.θ u < net (α := ℝ) p s u := by
      exact lt_of_le_of_ne hθ (by simpa [eq_comm] using hne)
    have hpos : 0 < net (α := ℝ) p s u - p.θ u := by linarith
    have hΔ' :
        energy (α := ℝ) p (updateAt (α := ℝ) p s u) - energy (α := ℝ) p s
          =
        ((-1 : ℝ) + (-1 : ℝ)) * (net (α := ℝ) p s u - p.θ u) := by
      -- Under `hdec` and `hsuf`, `xu' - x0 u = 2`, so `ΔE = -2 * (net-θ)`.
      simpa [hsuf, hdec, x, Spec.Hopfield.actVec, Spec.Hopfield.act] using hΔ
    have : energy (α := ℝ) p (updateAt (α := ℝ) p s u) - energy (α := ℝ) p s < 0 := by
      rw [hΔ']
      nlinarith [hpos]
    linarith
  · -- Update sets `u := false`, so changing means it was `true`.
    have hdec : decide (p.θ u ≤ net (α := ℝ) p s u) = false := by
      -- `decide p = false` when `¬ p`.
      simpa [decide_eq_false_iff_not] using (show ¬ p.θ u ≤ net (α := ℝ) p s u from hθ)
    have hup : updateAt (α := ℝ) p s u u = false := by
      have hu'' :
          updateAt (α := ℝ) p s u u = decide (p.θ u ≤ net (α := ℝ) p s u) := by
        simp [updateAt]
      exact hu''.trans hdec
    have hsut : s u = true := by
      cases hsu0 : s u <;> try rfl
      -- If `s u = false`, then `updateAt` wouldn't change at `u`.
      exfalso
      exact hsu (by simpa [hsu0] using hup)
    have hlt : net (α := ℝ) p s u < p.θ u := lt_of_not_ge hθ
    have hΔ' :
        energy (α := ℝ) p (updateAt (α := ℝ) p s u) - energy (α := ℝ) p s
          =
        ((1 : ℝ) + (1 : ℝ)) * (net (α := ℝ) p s u - p.θ u) := by
      -- Under `hdec` and `hsut`, `xu' - x0 u = -2`, so `ΔE = 2 * (net-θ)`.
      simpa [hsut, hdec, x, Spec.Hopfield.actVec, Spec.Hopfield.act] using hΔ
    have : energy (α := ℝ) p (updateAt (α := ℝ) p s u) - energy (α := ℝ) p s < 0 := by
      rw [hΔ']
      have : net (α := ℝ) p s u - p.θ u < 0 := by linarith
      nlinarith [this]
    linarith

end NN.MLTheory.Proofs.Hopfield
