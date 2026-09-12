/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.MLTheory.LearningTheory.Robustness.Spec
public import NN.Spec.Core.Context.Real

/-!
# `NN.MLTheory.Stability.Spec`

Scalar-polymorphic stability definitions for discrete-time dynamical systems
$x_{t+1}=f(x_t)$ over shape-indexed tensors.
-/

@[expose] public section

open Spec TorchLean

namespace NN.MLTheory.Stability.Spec

/-!
# Stability specifications (discrete-time dynamical systems)

This module defines standard stability notions for iterated maps `f : Tensor α s → Tensor α s`,
phrased over TorchLean's shape-indexed tensors:
- Lyapunov stability,
- asymptotic/exponential stability,
- global stability, and
- input-to-state stability (ISS).

The definitions are polymorphic in the scalar type `α` via `[TorchLean.Storage α] [Context α]`; for
noncomputable quantities (e.g. the supremum defining a stability margin on `ℝ`), we expose the
notion via a type class `StabilityMarginComputable`. TorchLean installs the real supremum instance
globally and keeps the conservative `0` lower-bound instance behind an explicit opt-in scope for
examples and tests.

## References

These are standard definitions in control theory / dynamical systems. Useful entry points include:

- H. K. Khalil, *Nonlinear Systems* (Lyapunov stability, exponential stability, ISS).
- E. D. Sontag, *Input-to-State Stability: Basic Concepts and Results* (ISS).
-/

open NN.MLTheory.Robustness.Spec

/-- Iterate `f` for $n$ steps: $\operatorname{iterate}(f,n,x)=f^{[n]}(x)$. -/
abbrev iterate {α : Type} [TorchLean.Storage α] {s : Shape}
    (f : Tensor α s → Tensor α s) (n : Nat) (x : Tensor α s) : Tensor α s :=
  Nat.iterate f n x

/--
An interface for a scalar stability-margin calculation. The `ℝ` instance takes the supremum of
nonnegative forward-invariant closed-ball radii. It is not necessarily an attained largest radius;
the real supremum requires a nonempty, bounded-above radius set for its usual interpretation.
The class itself contains no correctness law for other instances.

TorchLean only installs a real supremum-based instance globally for `ℝ`. Other scalar backends can
opt into the conservative lower-bound instance below explicitly; this avoids silently reporting `0`
as a semantic stability margin for arbitrary scalar types.
-/
class StabilityMarginComputable (α : Type) [TorchLean.Storage α] [Context α] where
  computeStabilityMargin :
      ∀ {s : Shape},
        (Tensor α s → Tensor α s) →
        (∀ {s : Shape}, Tensor α s → α) →
        Tensor α s →
        α

namespace StabilityMarginComputable

/-!
Named opt-in scope for the conservative stability-margin lower bound.

Use
`open scoped NN.MLTheory.Stability.Spec.StabilityMarginComputable.ConservativeMargin`
only in examples/tests that deliberately want a total fallback. Production theorem statements
should either use the `ℝ` instance or require an explicit `StabilityMarginComputable α` hypothesis.
-/
namespace ConservativeMargin

scoped instance (α : Type) [TorchLean.Storage α] [Context α] : StabilityMarginComputable α where
  computeStabilityMargin := fun _ _ _ => 0

end ConservativeMargin
end StabilityMarginComputable

noncomputable instance : StabilityMarginComputable ℝ where
  computeStabilityMargin := fun {s} f norm equilibrium =>
    sSup {r : ℝ | 0 ≤ r ∧ ∀ x₀ : Tensor ℝ s,
      tensorDistance norm equilibrium x₀ ≤ r →
        ∀ n : Nat, tensorDistance norm equilibrium (iterate f n x₀) ≤ r}

variable {α : Type} [TorchLean.Storage α] [Context α]

/--
Lyapunov stability of `equilibrium` for the discrete-time system $x_{t+1}=f(x_t)$.

This is the usual $\varepsilon$/$\delta$ definition using the distance induced by `norm`.
-/
def IsLyapunovStable {s : Shape}
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
def IsAsymptoticallyStable {s : Shape}
    (f : Tensor α s → Tensor α s)
    (norm : ∀ {s : Shape}, Tensor α s → α)
    (equilibrium : Tensor α s) : Prop :=
  IsLyapunovStable f norm equilibrium ∧
  ∃ δ > 0, ∀ x₀ : Tensor α s,
    tensorDistance norm equilibrium x₀ < δ →
    ∀ ε > 0, ∃ N : Nat, ∀ n ≥ N,
      tensorDistance norm equilibrium (iterate f n x₀) < ε

/--
Exponential stability with decay parameters `decayRate` and `M`.

Over real scalars with the usual norm and exponential laws, this expresses quantitative decay.
The generic predicate alone does not supply those laws.
-/
def IsExponentiallyStable {s : Shape}
    (f : Tensor α s → Tensor α s)
    (norm : ∀ {s : Shape}, Tensor α s → α)
    (equilibrium : Tensor α s)
    (decayRate : α) (M : α) : Prop :=
  decayRate > 0 ∧ M > 0 ∧
  ∃ δ > 0, ∀ x₀ : Tensor α s,
    tensorDistance norm equilibrium x₀ < δ →
    ∀ n : Nat, tensorDistance norm equilibrium (iterate f n x₀) ≤
      M * (MathFunctions.exp (-decayRate * n)) * tensorDistance norm equilibrium x₀

/--
Global attraction: every initial condition converges to `equilibrium`.

The established name `IsGloballyStable` denotes convergence only; it does not also require the
Lyapunov-stability predicate.

This is stated as convergence in the distance induced by `norm`.
-/
def IsGloballyStable {s : Shape}
    (f : Tensor α s → Tensor α s)
    (norm : ∀ {s : Shape}, Tensor α s → α)
    (equilibrium : Tensor α s) : Prop :=
  ∀ x₀ : Tensor α s, ∀ ε > 0, ∃ N : Nat, ∀ n ≥ N,
    tensorDistance norm equilibrium (iterate f n x₀) < ε

/--
Input-to-state stability (ISS) for an input-driven system $x_{t+1}=f(x_t,u_t)$.

This is the standard bound

$$
\lVert x_t\rVert
\leq \beta(\lVert x_0\rVert,t)
  +\gamma\!\left(\max_{k<t}\lVert u_k\rVert\right)
$$

packaged as a `Prop`. The maximum includes an initial zero and inputs at indices `k < t`,
which are the inputs used to reach state `t`. This predicate does not require the usual class-KL
conditions on `β` or class-K conditions on `γ`; those must be supplied separately for standard ISS.
-/
def IsInputToStateStable {s₁ s₂ : Shape}
    (f : Tensor α s₁ → Tensor α s₂ → Tensor α s₁)
    (norm : ∀ {s : Shape}, Tensor α s → α)
    (β : α → α → α) (γ : α → α) : Prop :=
  ∀ x₀ : Tensor α s₁, ∀ inputSeq : Nat → Tensor α s₂,
    ∀ t : Nat, norm (iterateWithInput f inputSeq t x₀) ≤
      β (norm x₀) (↑t) + γ (supNormOverTime inputSeq t)
where
  iterateWithInput (f : Tensor α s₁ → Tensor α s₂ → Tensor α s₁)
      (inputSeq : Nat → Tensor α s₂) : Nat → Tensor α s₁ → Tensor α s₁
    | 0, x => x
    | n + 1, x => f (iterateWithInput f inputSeq n x) (inputSeq n)
  supNormOverTime (inputSeq : Nat → Tensor α s₂) (t : Nat) : α :=
    (List.finRange t).foldl (fun acc i => max acc (norm (inputSeq i))) 0

/--
A fixed-bound input/output implication: inputs of norm at most `bound` have outputs of norm
at most the same `bound`. This is not the general BIBO quantification over input and output bounds.
-/
def IsBiboStable {s₁ s₂ : Shape}
    (f : Tensor α s₁ → Tensor α s₂)
    (norm₁ : ∀ {s : Shape}, Tensor α s → α)
    (norm₂ : ∀ {s : Shape}, Tensor α s → α)
    (bound : α) : Prop :=
  ∀ x : Tensor α s₁, norm₁ x ≤ bound → norm₂ (f x) ≤ bound

/--
Incremental stability: distances between trajectories contract by `contractionFactor`.

This is a discrete-time contraction condition phrased using `tensorDistance`.
-/
def IsIncrementallyStable {s : Shape}
    (f : Tensor α s → Tensor α s)
    (norm : ∀ {s : Shape}, Tensor α s → α)
    (contractionFactor : α) : Prop :=
  contractionFactor < 1 ∧
  ∀ x₁ x₂ : Tensor α s,
    tensorDistance norm (f x₁) (f x₂) ≤ contractionFactor * tensorDistance norm x₁ x₂

/--
Return the configured stability-margin value. The real instance uses a supremum, which need not
be attained and requires boundedness for its usual interpretation; custom instances have no
correctness law in this interface.
-/
def stabilityMargin {s : Shape} [StabilityMarginComputable α]
    (f : Tensor α s → Tensor α s)
    (norm : ∀ {s : Shape}, Tensor α s → α)
    (equilibrium : Tensor α s) : α :=
  StabilityMarginComputable.computeStabilityMargin (α := α) f norm equilibrium

/--
Finite-time stability: trajectories reach `equilibrium` exactly within a fixed step budget.
-/
def IsFiniteTimeStable {s : Shape}
    (f : Tensor α s → Tensor α s)
    (_norm : ∀ {s : Shape}, Tensor α s → α)
    (equilibrium : Tensor α s)
    (settlingTimeSteps : Nat) : Prop :=
  ∀ x₀ : Tensor α s, ∃ T ≤ settlingTimeSteps, ∀ t ≥ T,
    iterate f t x₀ = equilibrium

/--
Practical stability: trajectories eventually enter and remain in a fixed `ultimateBound` ball.
-/
def IsPracticallyStable {s : Shape}
    (f : Tensor α s → Tensor α s)
    (norm : ∀ {s : Shape}, Tensor α s → α)
    (equilibrium : Tensor α s)
    (ultimateBound : α) : Prop :=
  ∀ x₀ : Tensor α s, ∃ T : Nat, ∀ t ≥ T,
    tensorDistance norm equilibrium (iterate f t x₀) ≤ ultimateBound

/--
One-step monotonicity of a training loss under an update rule.

This is the “training stability” predicate used as a spec for decreasing-loss update rules.
-/
def IsTrainingStable {s : Shape}
    (updateRule : Tensor α s → Tensor α s)
    (loss : Tensor α s → α)
    (parameters : Tensor α s) : Prop :=
  loss (updateRule parameters) ≤ loss parameters

/--
Generalization stability of a learning algorithm: small dataset changes produce small prediction
changes.

This is a generic stability-style specification; concrete instances typically choose a specific
dataset metric and output norm.
-/
def IsGeneralizationStable {s₁ s₂ : Shape}
    (trainingAlgorithm : Array (Tensor α s₁ × Tensor α s₂) → (Tensor α s₁ → Tensor α s₂))
    (norm₁ : ∀ {s : Shape}, Tensor α s → α)
    (norm₂ : ∀ {s : Shape}, Tensor α s → α)
    (stabilityConstant : α) : Prop :=
  ∀ dataset₁ dataset₂ : Array (Tensor α s₁ × Tensor α s₂),
    let model₁ := trainingAlgorithm dataset₁
    let model₂ := trainingAlgorithm dataset₂
    ∀ x : Tensor α s₁,
      tensorDistance norm₂ (model₁ x) (model₂ x) ≤
      stabilityConstant * datasetDistance dataset₁ dataset₂
where
  datasetDistance (d₁ d₂ : Array (Tensor α s₁ × Tensor α s₂)) : α :=
    let sampleDistance : (Tensor α s₁ × Tensor α s₂) → (Tensor α s₁ × Tensor α s₂) → α :=
      fun p q => tensorDistance norm₁ p.1 q.1 + tensorDistance norm₂ p.2 q.2
    let alignedDistance :=
      (d₁.zip d₂).foldl (fun acc pq => acc + sampleDistance pq.1 pq.2) 0
    let lenPenalty := MathFunctions.abs ((d₁.size : α) - (d₂.size : α))
    alignedDistance + lenPenalty

end NN.MLTheory.Stability.Spec
