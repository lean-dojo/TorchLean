/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import Mathlib.Algebra.Order.Algebra
public import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Hopfield networks (spec-level)

A Hopfield network on `n` units, written as plain mathematical definitions:

- A state is a Boolean vector `Fin n → Bool` (activation `true ↦ +1`, `false ↦ -1`).
- Parameters are a weight matrix `W : Fin n → Fin n → α` and thresholds `θ : Fin n → α`.
- An asynchronous update picks a neuron `u` and updates only that coordinate.

Every sum is a left fold from zero in increasing index over the scalar's own `+` and `*`, so the
definitions need only the operations they use and evaluate over `ℚ`, as in the guide's examples,
and over `Float`. The theorems in `TorchLean/NN/Models/Hopfield/*` are stated over `ℝ`, where
`net_eq_mulVec` and `energy_eq_energyVec` rewrite the folds as `Finset` sums.

Hopfield (1982) is the model here, written by hand with `±1` states where he uses `0` and `1`,
and with the threshold term `Σ_i θ_i x_i` added to his energy [7]. He lets each unit update at
random times. Here the caller fixes the order, through `seqStates` or the sweep `cycleUpdate`.
Rule [1] covers inputs above and below the threshold and leaves equality open, which is the tie
`updateAt` settles. Our `net` and `energy` also keep the diagonal terms that his sums over `j ≠ i`
skip, and the theorems' zero-diagonal hypothesis removes them. With that, the one-step change
`energy_updateAt_delta` is his [8] with `net_u - θ_u` in place of his input sum.

## References

- Hopfield (1982),
  "Neural networks and physical systems with emergent collective computational abilities",
  Proceedings of the National Academy of Sciences 79(8), 2554-2558, equations [1], [7] and [8]:
  https://www.pnas.org/doi/10.1073/pnas.79.8.2554
- Hopfield (1984), "Neurons with graded response have collective computational properties
  like those of two-state neurons", Proceedings of the National Academy of Sciences 81(10),
  3088-3092:
  https://www.pnas.org/doi/10.1073/pnas.81.10.3088
- Ramsauer et al. (2020), "Hopfield Networks is All You Need":
  https://arxiv.org/abs/2008.02217
- Cipollina, Karatarakis and Wiedijk (2025), "Formalized Hopfield Networks and Boltzmann
  Machines", arXiv:2512.07766 (LPAR-26, EPiC Series in Computing):
  https://arxiv.org/abs/2512.07766

## Relationship to the Hopfield formalization case study

The update rule, the energy and the trajectories are defined here, and the results about global
dynamics are proved against them in `TorchLean/NN/Models/Hopfield/*`. With `W` symmetric and zero
on the diagonal, an update never increases the energy, an update that changes the state at equal
energy turns a unit on, and repeated cyclic sweeps reach a fixed point within `2 ^ n` sweeps
(`exists_fixedPoint_le_pow`).

The definitions correspond to those of “Formalized Hopfield Networks and Boltzmann Machines” (2025)
as follows:
- `updateAt` / `seqStates` formalize the asynchronous update dynamics,
- `energy` is the Lyapunov/energy functional used for convergence arguments,
- theorems in `TorchLean.NN.Models.Hopfield.Energy` and
  `TorchLean.NN.Models.Hopfield.Convergence` capture the classic “energy decreases ⇒ convergence”
  theorem pattern for finite state spaces.

## Implementation status

There is no `TorchLean.nn` model builder for Hopfield networks. The definitions are used by the
structural proofs in `TorchLean/NN/Models/Hopfield/*` and the guide's rational-arithmetic
examples.
-/

@[expose] public section


namespace TorchLean
namespace Hopfield

/-! ## Core data -/

/-- The `±1` value of a Boolean activation in the scalar type `α`: `true` gives `1` and `false`
gives `-1`.

We store Hopfield states as `Bool` because many proofs are simpler when the only possible
activations are `+1` and `-1`.
-/
def act {α : Type} [One α] [Neg α] : Bool → α
  | true => 1
  | false => -1

/-- A `true` unit activates to `+1`. -/
@[simp] theorem act_true {α : Type} [One α] [Neg α] : act (α := α) true = 1 := rfl
/-- A `false` unit activates to `-1`, as in the classical bipolar Hopfield model. -/
@[simp] theorem act_false {α : Type} [One α] [Neg α] : act (α := α) false = -1 := rfl

/-- Hopfield state as a Boolean activation vector. -/
abbrev State (n : Nat) : Type := Fin n → Bool

/-- The `±1` activation vector associated to a Boolean state. -/
def actVec {α : Type} [One α] [Neg α] {n : Nat} (s : State n) : Fin n → α :=
  fun i => act (α := α) (s i)

/-- Hopfield parameters:

- `W` is the weight matrix
- `θ` is the per-unit threshold / bias

Classic convergence results assume `W` symmetric with a zero diagonal. The structure accepts any
`W`, and the theorems that need those conditions take them as the hypotheses
`(Matrix.of p.W).IsSymm` and `(Matrix.of p.W).diag = 0`.
-/
structure Params (α : Type) (n : Nat) where
  /-- Weight from unit `j` to unit `i`, indexed as `W i j`. -/
  W : Fin n → Fin n → α
  /-- Activation threshold for each unit. -/
  θ : Fin n → α

/-! ## Hopfield update dynamics -/

/-- Net input to unit `u`: `(W * x)_u`, where `x = actVec s` is the `±1` encoding of the state,
added to zero in increasing `j`. -/
def net {α : Type} [Add α] [Mul α] [Zero α] [One α] [Neg α] {n : Nat}
    (p : Params α n) (s : State n) (u : Fin n) : α :=
  (List.finRange n).foldl (fun acc j => acc + p.W u j * actVec (α := α) s j) 0

/-- Asynchronous update at a single coordinate `u`, by the thresholded sign rule

`s[u] := (θ_u ≤ net_u)`,

which in `±1` terms sets `x_u := +1` if `net_u ≥ θ_u` and `x_u := -1` otherwise.

A tie `net_u = θ_u` goes to `+1`, and the convergence proofs rely on that. With `W` symmetric and
zero on the diagonal, an update that changes the state without lowering the energy is a tie, so it
turns a unit on and raises `pluses`.
-/
def updateAt {α : Type} [Add α] [Mul α] [Zero α] [One α] [Neg α]
    [LE α] [DecidableRel ((· ≤ ·) : α → α → Prop)] {n : Nat}
    (p : Params α n) (s : State n) (u : Fin n) : State n :=
  let x := net (α := α) p s u
  Function.update s u (decide (p.θ u ≤ x))

/-- A state is stable (a fixed point) if updating any single coordinate does nothing. -/
def IsStable {α : Type} [Add α] [Mul α] [Zero α] [One α] [Neg α]
    [LE α] [DecidableRel ((· ≤ ·) : α → α → Prop)] {n : Nat}
    (p : Params α n) (s : State n) : Prop :=
  ∀ u : Fin n, updateAt (α := α) p s u = s

/-- Count how many units are in the `true` (`+1`) state. -/
def pluses {n : Nat} (s : State n) : Nat :=
  (Finset.univ.filter fun i : Fin n => s i = true).card

/-! ## Hopfield energy -/

/-- The classical Hopfield energy for a state `s` (quadratic term + linear threshold term).

With `x = actVec s` the `±1` encoding, the energy is:

`E(s) = -1/2 * Σ_i Σ_j W_ij x_i x_j + Σ_i θ_i x_i`,

with each sum added to zero in increasing index and the half computed as `1 / (1 + 1)`. Over `ℝ`,
with `W` symmetric and zero on the diagonal, `energy_updateAt_le` proves that an asynchronous
update does not increase `E`, the Lyapunov argument for convergence.
-/
def energy {α : Type} [Add α] [Mul α] [Zero α] [One α] [Neg α] [Div α] {n : Nat}
    (p : Params α n) (s : State n) : α :=
  let x := actVec (α := α) s
  let quadratic := (List.finRange n).foldl (fun acc i =>
    acc + (List.finRange n).foldl (fun acc' j => acc' + p.W i j * x i * x j) 0) 0
  let linear := (List.finRange n).foldl (fun acc i => acc + p.θ i * x i) 0
  (-(1 / (1 + 1 : α))) * quadratic + linear

/-- State sequence of an asynchronous update schedule: `seqStates p useq s0 k` is the state after
`k` updates from `s0`, where the update from step `k` to step `k + 1` acts on coordinate `useq k`.

PyTorch analogy: typical NN libraries have no counterpart. The closest picture is a deterministic
update schedule for coordinate descent.
-/
def seqStates {α : Type} [Add α] [Mul α] [Zero α] [One α] [Neg α]
    [LE α] [DecidableRel ((· ≤ ·) : α → α → Prop)] {n : Nat}
    (p : Params α n) (useq : Nat → Fin n) (s0 : State n) : Nat → State n
  | 0 => s0
  | k + 1 => updateAt (α := α) p (seqStates p useq s0 k) (useq k)

end Hopfield
end TorchLean
