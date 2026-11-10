/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Tensor.Arithmetic

/-!
# System

Discrete-time dynamical systems on spec tensors.

A `System s` is a transition `step : Tensor ℝ s → Tensor ℝ s`. Its `n`-step map is
Mathlib's iterate `sys.step^[n]`, and its fixed points are `Function.IsFixedPt sys.step`. A
`DrivenSystem s u` also reads an input of shape `u` at each step. `DrivenSystem.iterate` and
`trajectory` run it on an input stream. `DrivenSystem.freeze` fixes the input, and
`iterate_const` proves that constant-input iteration is the iterate of the
frozen system.

The predicates `System.IsLyapunovStable` and `System.IsAsymptoticallyStable` take the size
function as a parameter, measure the distance of two states as the size of their difference and
assume no norm laws.

The two stability predicates are the usual ε-δ definitions in discrete time, written by hand.
Neither asks for `sys.step equilibrium = equilibrium`. Under a true norm it follows from
`IsLyapunovStable` at `x₀ = equilibrium`, whose first iterate then lies within every `ε` of the
equilibrium. No theorem uses the stability predicates. Contraction of a system is stated on its
step with `Robustness.IsContractive`, as in `ImageDDIM.stepSystem_contracts_of_step_contracts` and
`pfODEEulerSystem_contracts`.

Cipollina, Karatarakis and Wiedijk (2025) formalize Hopfield networks and Boltzmann machines in
Lean 4. TorchLean proves the deterministic asynchronous Hopfield facts in `TorchLean.Hopfield`, not
the Boltzmann, Hebbian, ergodicity or Perron-Frobenius parts.

References:
- Matteo Cipollina, Michail Karatarakis, and Freek Wiedijk (2025), "Formalized Hopfield Networks
  and Boltzmann Machines", arXiv:2512.07766.
-/

@[expose] public section


namespace TorchLean.Dynamics

/-- Discrete-time dynamical system on spec tensors. -/
structure System (s : Shape) where
  /-- One autonomous transition of the system state. -/
  step : Tensor ℝ s → Tensor ℝ s

/-- Driven system with exogenous input. -/
structure DrivenSystem (s u : Shape) where
  /-- One transition of the system state, parameterized by the current external input. -/
  step : Tensor ℝ s → Tensor ℝ u → Tensor ℝ s

/-- Iteration semantics for a driven system with an input stream. -/
def DrivenSystem.iterate {s u : Shape} (sys : DrivenSystem s u) (inputSeq : Nat → Tensor ℝ u) :
    Nat → Tensor ℝ s → Tensor ℝ s
  | 0, x => x
  | n + 1, x => sys.step (iterate sys inputSeq n x) (inputSeq n)

/-- The trajectory (orbit) of a driven system under a fixed input stream. -/
def DrivenSystem.trajectory {s u : Shape} (sys : DrivenSystem s u)
    (inputSeq : Nat → Tensor ℝ u) (x0 : Tensor ℝ s) : Nat → Tensor ℝ s :=
  fun t => iterate sys inputSeq t x0

/-- Zero driven steps leave the initial state unchanged. -/
@[simp] theorem DrivenSystem.iterate_zero {s u : Shape} (sys : DrivenSystem s u)
    (inputSeq : Nat → Tensor ℝ u) (x : Tensor ℝ s) :
    iterate sys inputSeq 0 x = x := by
  rfl

/-- The first point of a driven trajectory is its initial condition. -/
@[simp] theorem DrivenSystem.trajectory_zero {s u : Shape} (sys : DrivenSystem s u)
    (inputSeq : Nat → Tensor ℝ u) (x0 : Tensor ℝ s) :
    trajectory sys inputSeq x0 0 = x0 := by
  rfl

/-- A driven trajectory advances by applying the next input to the previous driven state. -/
@[simp] theorem DrivenSystem.iterate_succ {s u : Shape} (sys : DrivenSystem s u)
    (inputSeq : Nat → Tensor ℝ u) (n : Nat) (x : Tensor ℝ s) :
    iterate sys inputSeq (n + 1) x =
      sys.step (iterate sys inputSeq n x) (inputSeq n) := by
  rfl

/-- Successive driven trajectory points follow the driven transition and input stream. -/
@[simp] theorem DrivenSystem.trajectory_succ {s u : Shape} (sys : DrivenSystem s u)
    (inputSeq : Nat → Tensor ℝ u) (x0 : Tensor ℝ s) (n : Nat) :
    trajectory sys inputSeq x0 (n + 1) =
      sys.step (trajectory sys inputSeq x0 n) (inputSeq n) := by
  rfl

/-- Freeze a driven system at a fixed input, turning it into an autonomous system. -/
def DrivenSystem.freeze {s u : Shape} (sys : DrivenSystem s u)
    (input : Tensor ℝ u) : System s where
  step := fun x => sys.step x input

/-- A frozen driven system takes the same step as the original system at the frozen input. -/
@[simp] theorem DrivenSystem.freeze_step {s u : Shape} (sys : DrivenSystem s u)
    (input : Tensor ℝ u) (x : Tensor ℝ s) :
    (sys.freeze input).step x = sys.step x input := by
  rfl

/-- Constant-input driven iteration agrees with iteration of the frozen autonomous system. -/
theorem DrivenSystem.iterate_const {s u : Shape} (sys : DrivenSystem s u)
    (input : Tensor ℝ u) (n : Nat) (x : Tensor ℝ s) :
    iterate sys (fun _ => input) n x = (sys.freeze input).step^[n] x := by
  induction n generalizing x with
  | zero =>
      rfl
  | succ n ih =>
      exact (congrArg (sys.freeze input).step (ih x)).trans
        (Function.iterate_succ_apply' (sys.freeze input).step n x).symm

/-- Constant-input driven trajectories agree with trajectories of the frozen autonomous system. -/
theorem DrivenSystem.trajectory_const {s u : Shape} (sys : DrivenSystem s u)
    (input : Tensor ℝ u) (x0 : Tensor ℝ s) (n : Nat) :
    trajectory sys (fun _ => input) x0 n = (sys.freeze input).step^[n] x0 :=
  iterate_const sys input n x0

/--
Lyapunov stability of an equilibrium: for every `ε > 0` there is a `δ > 0` such that every
trajectory starting within `δ` of the equilibrium stays within `ε` of it at every step.
-/
def System.IsLyapunovStable {s : Shape}
    (sys : System s)
    (norm : ∀ {s : Shape}, Tensor ℝ s → ℝ)
    (equilibrium : Tensor ℝ s) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x₀,
    norm (equilibrium - x₀) < δ →
    ∀ n, norm (equilibrium - sys.step^[n] x₀) < ε

/-- Lyapunov stability together with local convergence to the equilibrium. -/
def System.IsAsymptoticallyStable {s : Shape}
    (sys : System s)
    (norm : ∀ {s : Shape}, Tensor ℝ s → ℝ)
    (equilibrium : Tensor ℝ s) : Prop :=
  IsLyapunovStable sys norm equilibrium ∧
  ∃ δ > 0, ∀ x₀,
    norm (equilibrium - x₀) < δ →
    ∀ ε > 0, ∃ N, ∀ n ≥ N, norm (equilibrium - sys.step^[n] x₀) < ε

end TorchLean.Dynamics
