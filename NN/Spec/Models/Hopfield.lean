/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Spec.Core.Tensor

/-!
# Hopfield networks (spec-level)

This file gives a Hopfield network model as a set of explicit mathematical definitions.

- A state is a Boolean vector `Fin n → Bool` (activation `true ↦ +1`, `false ↦ -1`).
- Parameters are a weight matrix `W : Fin n → Fin n → α` and thresholds `θ : Fin n → α`.
- An asynchronous update picks a neuron `u` and updates only that coordinate.

Intended use: mathematical scalars (`ℚ`, `ℝ`, etc.) and theorem statements/proofs.
For computation, instantiate `α := ℚ` (or `Rat`) and evaluate `seqStates` / `energy` via `#eval`
in an example file.

Note: we also provide Tensor-shaped wrappers (`StateT`, `ParamsT`) so users can work with
`Tensor _ (.dim n .scalar)` instead of raw `Fin n → _`.

## References

- Hopfield (1982),
  "Neural networks and physical systems with emergent collective computational abilities":
  https://www.pnas.org/doi/10.1073/pnas.79.8.2554
("Neurons with graded response have collective computational properties " ++
  "like those of two-state neurons"):
  https://www.pnas.org/doi/10.1073/pnas.81.10.3088
- Ramsauer et al. (2020), "Hopfield Networks is All You Need":
  https://arxiv.org/abs/2008.02217

## Relationship to the Hopfield formalization case study

This file provides the *definitions* (update rule, energy, trajectories). The global-dynamics
results (energy monotonicity, fixed-point convergence bounds, tie-handling lemmas, etc.) are proved
in `NN/MLTheory/Proofs/Hopfield/*` against these definitions.

If you’re reading this alongside “Formalized Hopfield Networks and Boltzmann Machines” (2025),
the correspondence is:
- `updateAt` / `seqStates` formalize the asynchronous update dynamics,
- `energy` is the Lyapunov/energy functional used for convergence arguments,
- theorems in `NN.MLTheory.Proofs.Hopfield.Energy` and `NN.MLTheory.Proofs.Hopfield.Convergence`
  capture the classic “energy decreases ⇒ convergence” theorem pattern for finite state spaces.
-/

@[expose] public section


namespace Spec
namespace Hopfield

open scoped BigOperators

/-! ## Core data -/

/-- Boolean activations interpreted as `±1`.

We intentionally store Hopfield states as `Bool` (instead of `α`) because many proofs are simpler
when the only possible activations are `+1` and `-1`. The embedding into a numeric scalar `α` is
explicit via `act`.
-/
def act {α : Type} [One α] [Neg α] : Bool → α
  | true => 1
  | false => -1

@[simp] lemma act_true {α : Type} [One α] [Neg α] : act (α := α) true = 1 := rfl
@[simp] lemma act_false {α : Type} [One α] [Neg α] : act (α := α) false = -1 := rfl

/-- Hopfield state as a Boolean activation vector. -/
abbrev State (n : Nat) : Type := Fin n → Bool

/-- Hopfield state as a vector tensor `Tensor Bool (.dim n .scalar)`.

This is a convenience wrapper so Hopfield models can interoperate with the rest of TorchLean's
tensor-shaped spec APIs.
-/
abbrev StateT (n : Nat) : Type := Tensor Bool (.dim n .scalar)

/-- Convert a tensor state to the underlying function representation. -/
def StateT.toFun {n : Nat} (s : StateT n) : State n :=
  (Tensor.dimScalarEquiv (α := Bool) n).toFun s

/-- Convert a function state to a tensor state. -/
def StateT.ofFun {n : Nat} (s : State n) : StateT n :=
  (Tensor.dimScalarEquiv (α := Bool) n).invFun s

@[simp] lemma StateT.toFun_ofFun {n : Nat} (s : State n) :
    StateT.toFun (StateT.ofFun (n := n) s) = s := by
  simp [StateT.toFun, StateT.ofFun]

@[simp] lemma StateT.ofFun_toFun {n : Nat} (s : StateT n) :
    StateT.ofFun (n := n) (StateT.toFun s) = s := by
  simp [StateT.toFun, StateT.ofFun]

/-- The `±1` activation vector associated to a Boolean state. -/
def actVec {α : Type} [One α] [Neg α] {n : Nat} (s : State n) : Fin n → α :=
  fun i => act (α := α) (s i)

/-- Dot product on vectors indexed by `Fin n`.

We define this directly as a `Fin`-indexed sum. This keeps the model independent from any concrete
matrix representation.
-/
def dot {α : Type} [AddCommMonoid α] [Mul α] {n : Nat} (x y : Fin n → α) : α :=
  ∑ i : Fin n, x i * y i

/-- Matrix-vector product (as a function, not an array-backed matrix). -/
def mulVec {α : Type} [AddCommMonoid α] [Mul α] {n : Nat} (W : Fin n → Fin n → α) (x : Fin n → α) :
    Fin n → α :=
  fun i => ∑ j : Fin n, W i j * x j

/-- Hopfield parameters:

- `W` is the weight matrix
- `θ` is the per-unit threshold / bias

Classic convergence results typically assume `W` is symmetric and has a zero diagonal. Those
assumptions are *not baked into this structure*; they are stated and proved where needed.
-/
structure Params (α : Type) (n : Nat) where
  /-- W. -/
  W : Fin n → Fin n → α
  θ : Fin n → α

/-- Hopfield parameters as tensor-shaped weights and thresholds. -/
structure ParamsT (α : Type) (n : Nat) where
  /-- W. -/
  W : Tensor α (.dim n (.dim n .scalar))
  θ : Tensor α (.dim n .scalar)

/-- Convert tensor-shaped parameters to the function representation. -/
def ParamsT.toFun {α : Type} {n : Nat} (p : ParamsT α n) : Params α n where
  W := fun i j => Spec.get2 p.W i j
  θ := fun i => Tensor.vecGet p.θ i

/-! ## Hopfield update dynamics -/

/-- Net input to unit `u`: `(W * x)_u`, where `x = actVec s` is the `±1` encoding of the state. -/
def net {α : Type} [AddCommMonoid α] [Mul α] [One α] [Neg α] {n : Nat}
    (p : Params α n) (s : State n) (u : Fin n) : α :=
  mulVec p.W (actVec (α := α) s) u

/-- Tensor-shaped wrapper for `net` (useful for interop with TorchLean tensor APIs). -/
def netT {α : Type} [AddCommMonoid α] [Mul α] [One α] [Neg α] {n : Nat}
    (p : ParamsT α n) (s : StateT n) (u : Fin n) : α :=
  net (α := α) (p := p.toFun) (s := s.toFun) u

/-- Asynchronous update at a single coordinate `u`.

We implement the standard thresholded sign rule:

`s[u] := (θ_u ≤ net_u)`

Interpreting `true ↦ +1` and `false ↦ -1`, this corresponds to:

`x_u := +1` if `net_u ≥ θ_u`, otherwise `x_u := -1`.

Tie-handling (`net_u = θ_u`) matters for formal convergence arguments; we pick the convention
"ties go to `+1`" via `≤`.
-/
def updateAt {α : Type} [AddCommMonoid α] [Mul α] [One α] [Neg α]
    [LE α] [DecidableRel ((· ≤ ·) : α → α → Prop)] {n : Nat}
    (p : Params α n) (s : State n) (u : Fin n) : State n :=
  let x := net (α := α) p s u
  Function.update s u (decide (p.θ u ≤ x))

/-- Tensor-shaped wrapper for `updateAt`. -/
def updateAtT {α : Type} [AddCommMonoid α] [Mul α] [One α] [Neg α]
    [LE α] [DecidableRel ((· ≤ ·) : α → α → Prop)] {n : Nat}
    (p : ParamsT α n) (s : StateT n) (u : Fin n) : StateT n :=
  StateT.ofFun (n := n) (updateAt (α := α) (p := p.toFun) (s := s.toFun) u)

/-- A state is stable (a fixed point) if updating any single coordinate does nothing. -/
def IsStable {α : Type} [AddCommMonoid α] [Mul α] [One α] [Neg α]
    [LE α] [DecidableRel ((· ≤ ·) : α → α → Prop)] {n : Nat}
    (p : Params α n) (s : State n) : Prop :=
  ∀ u : Fin n, updateAt (α := α) p s u = s

/-- Count how many units are in the `true` (`+1`) state. -/
def pluses {n : Nat} (s : State n) : Nat :=
  (Finset.univ.filter fun i : Fin n => s i = true).card

/-! ## Hopfield energy -/

/-- The classical Hopfield energy for a state `s` (quadratic term + linear threshold term).

With `x = actVec s` the `±1` encoding, the energy is:

`E(s) = -1/2 * Σ_i Σ_j W_ij x_i x_j + Σ_i θ_i x_i`.

When `W` is symmetric and has a zero diagonal, asynchronous updates are known to monotonically
decrease (or not increase) `E`, which is the classic Lyapunov-style argument for convergence.
-/
def energy {α : Type} [Field α] {n : Nat}
    (p : Params α n) (s : State n) : α :=
  let x := actVec (α := α) s
  (-(1 / (2 : α))) * (∑ i : Fin n, ∑ j : Fin n, p.W i j * x i * x j) +
    ∑ i : Fin n, p.θ i * x i

/-- Tensor-shaped wrapper for `energy`. -/
def energyT {α : Type} [Field α] {n : Nat}
    (p : ParamsT α n) (s : StateT n) : α :=
  energy (α := α) (p := p.toFun) (s := s.toFun)

/-- State sequence induced by an asynchronous update schedule `useq`.

`useq : Nat → Fin n` picks which coordinate to update at each discrete time step.

PyTorch analogy: there is no direct equivalent in typical NN libraries, but you can think of it as
an explicit, deterministic "update schedule" for coordinate descent.
-/
def seqStates {α : Type} [AddCommMonoid α] [Mul α] [One α] [Neg α]
    [LE α] [DecidableRel ((· ≤ ·) : α → α → Prop)] {n : Nat}
    (p : Params α n) (useq : Nat → Fin n) (s0 : State n) : Nat → State n
  | 0 => s0
  | k + 1 => updateAt (α := α) p (seqStates p useq s0 k) (useq k)

/-- Tensor-shaped wrapper for `seqStates`. -/
def seqStatesT {α : Type} [AddCommMonoid α] [Mul α] [One α] [Neg α]
    [LE α] [DecidableRel ((· ≤ ·) : α → α → Prop)] {n : Nat}
    (p : ParamsT α n) (useq : Nat → Fin n) (s0 : StateT n) : Nat → StateT n
  | 0 => s0
  | k + 1 => updateAtT (α := α) p (seqStatesT p useq s0 k) (useq k)

/-- A cyclic update schedule (0,1,2,...,n-1,0,1,...) for `n > 0`. -/
def cyclicUseq (n : Nat) (hn : 0 < n) : Nat → Fin n :=
  fun k => ⟨k % n, Nat.mod_lt k hn⟩

/-!
## Executable Hopfield (backend-friendly)

The definitions above (`net`, `energy`, …) are written in a math-first style using `∑` over `Fin n`.
That is the right presentation for proofs, but it bakes in algebraic typeclasses like
`AddCommMonoid` and uses `Finset` sums.

When we execute Hopfield over IEEE-like scalars (e.g. `Float`, `IEEE32Exec`), we do *not* want to
pretend those algebraic laws hold exactly: NaNs and rounding make addition non-associative and
non-commutative in general. So for runtime execution we provide a “plain loop” variant that:

- requires only the operations from `[Context α]`,
- uses explicit `List.foldl` iteration over `Fin n`.

This is the same model, just written in a way that stays honest about runtime arithmetic.

References:
- IEEE 754-2019 (why NaNs/rounding break algebraic laws):
  https://doi.org/10.1109/IEEESTD.2019.8766229
- Goldberg (1991), classic floating-point background: https://doi.org/10.1145/103162.103163
-/

namespace Exec

open Tensor

variable {α : Type} [Context α] [DecidableRel ((· ≤ ·) : α → α → Prop)]

@[inline] def act : Bool → α := Hopfield.act (α := α)

@[inline] def theta {n : Nat} (p : ParamsT α n) (i : Fin n) : α :=
  Tensor.vecGet p.θ i

@[inline] def weight {n : Nat} (p : ParamsT α n) (i j : Fin n) : α :=
  Spec.get2 p.W i j

/--
Net input to unit `u` computed by explicit iteration.

This matches `Hopfield.netT`, but avoids `Finset` sums so it can execute over IEEE-like scalars.
-/
def net {n : Nat} (p : ParamsT α n) (s : StateT n) (u : Fin n) : α :=
  let sf := Hopfield.StateT.toFun s
  (List.finRange n).foldl (fun acc j => acc + weight p u j * act (sf j)) 0

/--
Asynchronous update of a single coordinate `u`, using the same “ties go to +1” convention as the
spec definition (`θ_u ≤ net_u`).
-/
def updateAt {n : Nat} (p : ParamsT α n) (s : StateT n) (u : Fin n) : StateT n :=
  let sf := Hopfield.StateT.toFun s
  let x := net p s u
  let b := decide (theta p u ≤ x)
  Hopfield.StateT.ofFun (n := n) (Function.update sf u b)

/-- State sequence induced by an update schedule `useq : Nat → Fin n` (loop-based). -/
def seqStates {n : Nat} (p : ParamsT α n) (useq : Nat → Fin n) (s0 : StateT n) : Nat → StateT n
  | 0 => s0
  | k + 1 => Exec.updateAt p (seqStates p useq s0 k) (useq k)

/--
Hopfield energy computed by explicit iteration.

This matches the standard formula:
`E(s) = -1/2 * Σ_i Σ_j W_ij x_i x_j + Σ_i θ_i x_i`,
with `x_i ∈ {+1,-1}` obtained from the Boolean state.
-/
def energy {n : Nat} (p : ParamsT α n) (s : StateT n) : α :=
  let sf := Hopfield.StateT.toFun s
  let half : α := ((1 : Nat) : α) / ((2 : Nat) : α)
  let q :=
    (List.finRange n).foldl (fun acc i =>
      acc + (List.finRange n).foldl (fun acc2 j =>
        acc2 + weight p i j * act (sf i) * act (sf j)
      ) 0
    ) 0
  let lin :=
    (List.finRange n).foldl (fun acc i => acc + theta p i * act (sf i)) 0
  (-half) * q + lin

end Exec

end Hopfield
end Spec
