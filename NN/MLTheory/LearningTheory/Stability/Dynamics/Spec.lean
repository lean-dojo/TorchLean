/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.LearningTheory.Robustness.Spec

/-!
# `NN.MLTheory.Stability.Spec`

Scalar-polymorphic stability definitions for discrete-time dynamical systems
`x_{t+1} = f x_t` over shape-indexed tensors.
-/

@[expose] public section

open Spec

namespace NN.MLTheory.Stability.Spec

/-!
# Stability specifications (discrete-time dynamical systems)

This module defines standard stability notions for iterated maps `f : Tensor α s → Tensor α s`,
phrased over TorchLean's shape-indexed tensors:
- Lyapunov stability,
- asymptotic/exponential stability,
- global stability, and
- input-to-state stability (ISS).

The definitions are polymorphic in the scalar type `α` via `[Context α]`; for noncomputable
quantities (e.g. the supremum defining a stability margin on `ℝ`), we expose the notion via a type
class `StabilityMarginComputable`. TorchLean installs the real supremum instance globally and keeps
the conservative `0` lower-bound instance behind an explicit opt-in scope for examples and tests.

## References

These are standard definitions in control theory / dynamical systems. Useful entry points include:

- H. K. Khalil, *Nonlinear Systems* (Lyapunov stability, exponential stability, ISS).
- E. D. Sontag, *Input-to-State Stability: Basic Concepts and Results* (ISS).
-/

open NN.MLTheory.Robustness.Spec

/-- Iterate `f` for `n` steps: `iterate f n x = f^[n] x`. -/
abbrev iterate {α : Type} {s : Shape} (f : Tensor α s → Tensor α s) (n : Nat) (x : Tensor α s) :
  Tensor α s :=
  Nat.iterate f n x

/--
A computable notion of stability margin, matching the "largest invariant ball" intuition:
the largest `r ≥ 0` such that the closed ball `{x | dist(eq, x) ≤ r}` is forward-invariant.

TorchLean only installs a real supremum-based instance globally for `ℝ`. Other scalar backends can
opt into the conservative lower-bound instance below explicitly; this avoids silently reporting `0` as a
semantic stability margin for arbitrary scalar types.
-/
class StabilityMarginComputable (α : Type) [Context α] where
  compute_stability_margin :
      ∀ {s : Shape},
        (Tensor α s → Tensor α s) →
        (∀ {s : Shape}, Tensor α s → α) →
        Tensor α s →
        α

namespace StabilityMarginComputable

/-!
Named opt-in scope for the conservative stability-margin lower bound.

Use `open scoped NN.MLTheory.Stability.ConservativeMargin` only in examples/tests that deliberately want
a total fallback. Production theorem statements should either use the `ℝ` instance or require an
explicit `StabilityMarginComputable α` hypothesis.
-/
namespace ConservativeMargin

scoped instance (α : Type) [Context α] : StabilityMarginComputable α where
  compute_stability_margin := fun _ _ _ => 0

end ConservativeMargin
end StabilityMarginComputable

noncomputable instance : StabilityMarginComputable ℝ where
  compute_stability_margin := fun {s} f norm equilibrium =>
    sSup {r : ℝ | 0 ≤ r ∧ ∀ x₀ : Tensor ℝ s,
      tensorDistance norm equilibrium x₀ ≤ r →
        ∀ n : Nat, tensorDistance norm equilibrium (iterate f n x₀) ≤ r}

variable {α : Type} [Context α]

/--
Lyapunov stability of `equilibrium` for the discrete-time system `x_{t+1} = f x_t`.

This is the usual `ε`/`δ` definition using the distance induced by `norm`.
-/
def isLyapunovStable {s : Shape}
    (f : Tensor α s → Tensor α s)
    (norm : ∀ {s : Shape}, Tensor α s → α)
    (equilibrium : Tensor α s) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x₀ : Tensor α s,
    tensorDistance norm equilibrium x₀ < δ →
    ∀ n : Nat, tensorDistance norm equilibrium (iterate f n x₀) < ε

/--
Asymptotic stability: Lyapunov stability plus convergence to `equilibrium` for nearby initial
conditions.
-/
def isAsymptoticallyStable {s : Shape}
    (f : Tensor α s → Tensor α s)
    (norm : ∀ {s : Shape}, Tensor α s → α)
    (equilibrium : Tensor α s) : Prop :=
  isLyapunovStable f norm equilibrium ∧
  ∃ δ > 0, ∀ x₀ : Tensor α s,
    tensorDistance norm equilibrium x₀ < δ →
    ∀ ε > 0, ∃ N : Nat, ∀ n ≥ N,
      tensorDistance norm equilibrium (iterate f n x₀) < ε

/--
Exponential stability with decay parameters `decay_rate` and `M`.

This is a quantitative strengthening of asymptotic stability.
-/
def isExponentiallyStable {s : Shape}
    (f : Tensor α s → Tensor α s)
    (norm : ∀ {s : Shape}, Tensor α s → α)
    (equilibrium : Tensor α s)
    (decay_rate : α) (M : α) : Prop :=
  decay_rate > 0 ∧ M > 0 ∧
  ∃ δ > 0, ∀ x₀ : Tensor α s,
    tensorDistance norm equilibrium x₀ < δ →
    ∀ n : Nat, tensorDistance norm equilibrium (iterate f n x₀) ≤
      M * (MathFunctions.exp (-decay_rate * n)) * tensorDistance norm equilibrium x₀

/--
Global stability: every initial condition converges to `equilibrium`.

This is stated as convergence in the distance induced by `norm`.
-/
def isGloballyStable {s : Shape}
    (f : Tensor α s → Tensor α s)
    (norm : ∀ {s : Shape}, Tensor α s → α)
    (equilibrium : Tensor α s) : Prop :=
  ∀ x₀ : Tensor α s, ∀ ε > 0, ∃ N : Nat, ∀ n ≥ N,
    tensorDistance norm equilibrium (iterate f n x₀) < ε

/--
Input-to-state stability (ISS) for an input-driven system `x_{t+1} = f x_t u_t`.

This is the standard bound `‖x_t‖ ≤ β(‖x_0‖, t) + γ( sup_{k ≤ t} ‖u_k‖ )` packaged as a `Prop`.
-/
def isInputToStateStable {s₁ s₂ : Shape}
    (f : Tensor α s₁ → Tensor α s₂ → Tensor α s₁)
    (norm : ∀ {s : Shape}, Tensor α s → α)
    (β : α → α → α) (γ : α → α) : Prop :=
  ∀ x₀ : Tensor α s₁, ∀ input_seq : Nat → Tensor α s₂,
    ∀ t : Nat, norm (iterateWithInput f input_seq t x₀) ≤
      β (norm x₀) (↑t) + γ (sup_norm_over_time input_seq t)
where
  iterateWithInput (f : Tensor α s₁ → Tensor α s₂ → Tensor α s₁)
      (input_seq : Nat → Tensor α s₂) : Nat → Tensor α s₁ → Tensor α s₁
    | 0, x => x
    | n + 1, x => f (iterateWithInput f input_seq n x) (input_seq n)
  sup_norm_over_time (input_seq : Nat → Tensor α s₂) (t : Nat) : α :=
    (List.finRange t).foldl (fun acc i => max acc (norm (input_seq i))) 0

/--
Bounded-input bounded-output (BIBO) stability with respect to `norm₁` and `norm₂`.
-/
def isBiboStable {s₁ s₂ : Shape}
    (f : Tensor α s₁ → Tensor α s₂)
    (norm₁ : ∀ {s : Shape}, Tensor α s → α)
    (norm₂ : ∀ {s : Shape}, Tensor α s → α)
    (bound : α) : Prop :=
  ∀ x : Tensor α s₁, norm₁ x ≤ bound → norm₂ (f x) ≤ bound

/--
Incremental stability: distances between trajectories contract by `contraction_factor`.

This is a discrete-time contraction condition phrased using `tensor_distance`.
-/
def isIncrementallyStable {s : Shape}
    (f : Tensor α s → Tensor α s)
    (norm : ∀ {s : Shape}, Tensor α s → α)
    (contraction_factor : α) : Prop :=
  contraction_factor < 1 ∧
  ∀ x₁ x₂ : Tensor α s,
    tensorDistance norm (f x₁) (f x₂) ≤ contraction_factor * tensorDistance norm x₁ x₂

/--
Stability margin: the largest forward-invariant ball radius around `equilibrium` (when computable).
-/
def stabilityMargin {s : Shape} [StabilityMarginComputable α]
    (f : Tensor α s → Tensor α s)
    (norm : ∀ {s : Shape}, Tensor α s → α)
    (equilibrium : Tensor α s) : α :=
  StabilityMarginComputable.compute_stability_margin (α := α) f norm equilibrium

/--
Finite-time stability: trajectories reach `equilibrium` exactly within a fixed step budget.
-/
def isFiniteTimeStable {s : Shape}
    (f : Tensor α s → Tensor α s)
    (_norm : ∀ {s : Shape}, Tensor α s → α)
    (equilibrium : Tensor α s)
    (settling_time_steps : Nat) : Prop :=
  ∀ x₀ : Tensor α s, ∃ T ≤ settling_time_steps, ∀ t ≥ T,
    iterate f t x₀ = equilibrium

/--
Practical stability: trajectories eventually enter and remain in a fixed `ultimate_bound` ball.
-/
def isPracticallyStable {s : Shape}
    (f : Tensor α s → Tensor α s)
    (norm : ∀ {s : Shape}, Tensor α s → α)
    (equilibrium : Tensor α s)
    (ultimate_bound : α) : Prop :=
  ∀ x₀ : Tensor α s, ∃ T : Nat, ∀ t ≥ T,
    tensorDistance norm equilibrium (iterate f t x₀) ≤ ultimate_bound

/--
One-step monotonicity of a training loss under an update rule.

This is the “training stability” predicate used as a spec for decreasing-loss update rules.
-/
def isTrainingStable {s : Shape}
    (update_rule : Tensor α s → Tensor α s)
    (loss : Tensor α s → α)
    (parameters : Tensor α s) : Prop :=
  loss (update_rule parameters) ≤ loss parameters

/--
Generalization stability of a learning algorithm: small dataset changes produce small prediction
changes.

This is a generic stability-style specification; concrete instances typically choose a specific
dataset metric and output norm.
-/
def isGeneralizationStable {s₁ s₂ : Shape}
    (training_algorithm : List (Tensor α s₁ × Tensor α s₂) → (Tensor α s₁ → Tensor α s₂))
    (norm₁ : ∀ {s : Shape}, Tensor α s → α)
    (norm₂ : ∀ {s : Shape}, Tensor α s → α)
    (stability_constant : α) : Prop :=
  ∀ dataset₁ dataset₂ : List (Tensor α s₁ × Tensor α s₂),
    let model₁ := training_algorithm dataset₁
    let model₂ := training_algorithm dataset₂
    ∀ x : Tensor α s₁,
      tensorDistance norm₂ (model₁ x) (model₂ x) ≤
      stability_constant * dataset_distance dataset₁ dataset₂
where
  dataset_distance (d₁ d₂ : List (Tensor α s₁ × Tensor α s₂)) : α :=
    let sample_distance : (Tensor α s₁ × Tensor α s₂) → (Tensor α s₁ × Tensor α s₂) → α :=
      fun p q => tensorDistance norm₁ p.1 q.1 + tensorDistance norm₂ p.2 q.2
    let aligned_distance :=
      (d₁.zip d₂).foldl (fun acc pq => acc + sample_distance pq.1 pq.2) 0
    let len_penalty := MathFunctions.abs ((d₁.length : α) - (d₂.length : α))
    aligned_distance + len_penalty

end NN.MLTheory.Stability.Spec
