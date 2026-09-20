/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Proofs.RuntimeApprox.Graph.ForwardApprox
public import NN.Runtime.Optim.Optimizers

/-!
# Numerical Contracts for Optimizer Steps

An optimizer proof has two kinds of state: the mathematical recurrence and the rounded runtime
state. `NumericalStepContract` records their relation once. A concrete optimizer supplies its exact
and runtime update equations, a transformer for state/parameter error bounds, and a proof that one
step preserves the relation. `run_approx` then composes that local proof over any finite gradient
stream.

This interface is deliberately independent of SGD, Adam, or a particular scalar backend. It avoids
duplicating an induction theorem for every optimizer and, unlike an equality theorem obtained by
unfolding two identical definitions, states the numerical refinement claim needed by training.

For the distinction between local rounding errors and their propagation through an iterative
algorithm, see N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., 2002.
-/

@[expose] public section

namespace Proofs.RuntimeApprox.Optimizer

open Spec TorchLean

noncomputable section

/-- Per-step optimizer-state and parameter errors computed by a numerical optimizer contract. -/
structure StepError (StateError : Shape → Type) (shape : Shape) where
  /-- Bound object for the optimizer's private state after the step. -/
  optimizerStateError : StateError shape
  /-- Infinity-norm error budget for the parameter tensor after the step. -/
  parameterError : ℝ

/-- A numerical refinement contract for one shape-polymorphic optimizer update.

`StepAssumptions` carries numerical information required only for the current update. It is `Unit`
for unconditional rules such as SGD, while adaptive optimizers use it for denominator margins and
rounded scalar-expression bounds. This lets one finite-run theorem cover both cases.
-/
structure NumericalStepContract (R : Type) (toSpec : R → ℝ) where
  /-- Stable optimizer name used in numerical reports. -/
  name : String
  /-- Mathematical optimizer state. -/
  ExactState : Shape → Type
  /-- Rounded runtime optimizer state. -/
  RuntimeState : Shape → Type
  /-- Error information relating mathematical and runtime state. -/
  StateError : Shape → Type
  /-- Numerical data and domain margins supplied for one update. -/
  StepAssumptions : Shape → Type
  /-- Relation certified between mathematical and runtime state. -/
  stateApprox : {shape : Shape} →
    ExactState shape → RuntimeState shape → StateError shape → Prop
  /-- Conditions under which one step's numerical data is valid. -/
  assumptionsHold : {shape : Shape} →
    ExactState shape → RuntimeState shape → StateError shape →
    Tensor ℝ shape → Tensor R shape → ℝ →
    Tensor ℝ shape → Tensor R shape → ℝ → StepAssumptions shape → Prop
  /-- One exact-real optimizer update. -/
  updateExact : {shape : Shape} →
    ExactState shape → Tensor ℝ shape → Tensor ℝ shape →
      Optim.Step ℝ shape (ExactState shape)
  /-- One rounded runtime optimizer update. -/
  updateRuntime : {shape : Shape} →
    RuntimeState shape → Tensor R shape → Tensor R shape →
      Optim.Step R shape (RuntimeState shape)
  /-- Compute the next state/parameter bounds from current errors and runtime values. -/
  nextError : {shape : Shape} → StateError shape → ℝ → ℝ →
    RuntimeState shape → Tensor R shape → Tensor R shape → StepAssumptions shape →
      StepError StateError shape
  /-- Proof-free scalar components of a state bound for reports and UI consumers. -/
  stateErrorReport : {shape : Shape} → StateError shape → Array (String × ℝ)
  /-- Proof-free scalar components of one step's side data. -/
  assumptionReport : {shape : Shape} → StepAssumptions shape → Array (String × ℝ)
  /-- One-step numerical soundness. -/
  updateApprox : ∀ {shape : Shape}
      (exactState : ExactState shape) (runtimeState : RuntimeState shape)
      (stateError : StateError shape)
      (exactParameters : Tensor ℝ shape) (runtimeParameters : Tensor R shape)
      (parameterError : ℝ)
      (exactGradients : Tensor ℝ shape) (runtimeGradients : Tensor R shape)
      (gradientError : ℝ)
      (assumptions : StepAssumptions shape),
    stateApprox exactState runtimeState stateError →
    approxTensor (α := R) (toSpec := toSpec)
      exactParameters runtimeParameters parameterError →
    approxTensor (α := R) (toSpec := toSpec)
      exactGradients runtimeGradients gradientError →
    assumptionsHold exactState runtimeState stateError
      exactParameters runtimeParameters parameterError
      exactGradients runtimeGradients gradientError assumptions →
      let error := nextError stateError parameterError gradientError
        runtimeState runtimeParameters runtimeGradients assumptions
      stateApprox
          (updateExact exactState exactParameters exactGradients).optimizerState
          (updateRuntime runtimeState runtimeParameters runtimeGradients).optimizerState
          error.optimizerStateError ∧
        approxTensor (α := R) (toSpec := toSpec)
          (updateExact exactState exactParameters exactGradients).parameters
          (updateRuntime runtimeState runtimeParameters runtimeGradients).parameters
          error.parameterError

namespace NumericalStepContract

variable {R : Type} {toSpec : R → ℝ}

/-- Exact, rounded, and error information for one optimizer update. -/
structure StepInput (contract : NumericalStepContract R toSpec) (shape : Shape) where
  /-- Exact-real gradient. -/
  exactGradient : Tensor ℝ shape
  /-- Rounded runtime gradient. -/
  runtimeGradient : Tensor R shape
  /-- Infinity-norm error relating the exact and runtime gradients. -/
  gradientError : ℝ
  /-- Optimizer-specific side data and domain margins. -/
  assumptions : contract.StepAssumptions shape

/-- Execute a finite step stream using the exact-real recurrence. -/
def runExact (contract : NumericalStepContract R toSpec) {shape : Shape}
    (initial : Optim.Step ℝ shape (contract.ExactState shape))
    (steps : Array (StepInput contract shape)) :
    Optim.Step ℝ shape (contract.ExactState shape) :=
  steps.foldl
    (fun current step =>
      contract.updateExact current.optimizerState current.parameters step.exactGradient)
    initial

/-- Execute the same finite step stream using the rounded runtime recurrence. -/
def runRuntime (contract : NumericalStepContract R toSpec) {shape : Shape}
    (initial : Optim.Step R shape (contract.RuntimeState shape))
    (steps : Array (StepInput contract shape)) :
    Optim.Step R shape (contract.RuntimeState shape) :=
  steps.foldl
    (fun current step =>
      contract.updateRuntime current.optimizerState current.parameters step.runtimeGradient)
    initial

/-- Propagate state and parameter errors over a bundled optimizer step stream. -/
def runErrors (contract : NumericalStepContract R toSpec) {shape : Shape}
    (initialError : StepError contract.StateError shape)
    (initialRuntime : Optim.Step R shape (contract.RuntimeState shape))
    (steps : Array (StepInput contract shape)) : StepError contract.StateError shape :=
  match steps.foldl
    (fun (currentError, currentRuntime) step =>
      let nextError := contract.nextError
        currentError.optimizerStateError
        currentError.parameterError
        step.gradientError
        currentRuntime.optimizerState
        currentRuntime.parameters
        step.runtimeGradient
        step.assumptions
      let nextRuntime := contract.updateRuntime
        currentRuntime.optimizerState
        currentRuntime.parameters
        step.runtimeGradient
      (nextError, nextRuntime))
    (initialError, initialRuntime) with
  | (finalError, _) => finalError

/-- Approximation and side-condition evidence for a complete optimizer run.

The indices thread exact state, runtime state, and error bounds through the same recurrence used by
`runExact`, `runRuntime`, and `runErrors`. Adaptive-domain conditions are therefore checked at the
step where they are needed rather than asserted once for an entire run.
-/
inductive StepStreamApprox (contract : NumericalStepContract R toSpec) {shape : Shape} :
    Optim.Step ℝ shape (contract.ExactState shape) →
    Optim.Step R shape (contract.RuntimeState shape) →
    StepError contract.StateError shape →
    Array (StepInput contract shape) → Prop
  | empty {exact runtime error} : StepStreamApprox contract exact runtime error #[]
  | cons {exact runtime error step steps} :
      approxTensor (α := R) (toSpec := toSpec)
        step.exactGradient step.runtimeGradient step.gradientError →
      contract.assumptionsHold
        exact.optimizerState
        runtime.optimizerState
        error.optimizerStateError
        exact.parameters
        runtime.parameters
        error.parameterError
        step.exactGradient
        step.runtimeGradient
        step.gradientError
        step.assumptions →
      StepStreamApprox contract
        (contract.updateExact
          exact.optimizerState exact.parameters step.exactGradient)
        (contract.updateRuntime
          runtime.optimizerState runtime.parameters step.runtimeGradient)
        (contract.nextError
          error.optimizerStateError
          error.parameterError
          step.gradientError
          runtime.optimizerState
          runtime.parameters
          step.runtimeGradient
          step.assumptions)
        steps →
      StepStreamApprox contract exact runtime error
        (#[step] ++ steps)

/-- Final soundness statement associated with one finite optimizer run. -/
def RunApprox (contract : NumericalStepContract R toSpec) {shape : Shape}
    (exact : Optim.Step ℝ shape (contract.ExactState shape))
    (runtime : Optim.Step R shape (contract.RuntimeState shape))
    (error : StepError contract.StateError shape)
    (steps : Array (StepInput contract shape)) : Prop :=
  contract.stateApprox
      (contract.runExact exact steps).optimizerState
      (contract.runRuntime runtime steps).optimizerState
      (contract.runErrors error runtime steps).optimizerStateError ∧
    approxTensor (α := R) (toSpec := toSpec)
      (contract.runExact exact steps).parameters
      (contract.runRuntime runtime steps).parameters
      (contract.runErrors error runtime steps).parameterError

/-- A local optimizer contract composes over any finite validated gradient stream. -/
theorem run_approx (contract : NumericalStepContract R toSpec) {shape : Shape}
    {exact : Optim.Step ℝ shape (contract.ExactState shape)}
    {runtime : Optim.Step R shape (contract.RuntimeState shape)}
    {error : StepError contract.StateError shape}
    {steps : Array (StepInput contract shape)}
    (stepsApprox : StepStreamApprox contract exact runtime error steps) :
    contract.stateApprox
      exact.optimizerState runtime.optimizerState error.optimizerStateError →
    approxTensor (α := R) (toSpec := toSpec)
      exact.parameters runtime.parameters error.parameterError →
    RunApprox contract exact runtime error steps := by
  cases stepsApprox with
  | empty =>
      intro stateApprox parametersApprox
      simpa [RunApprox, runExact, runRuntime, runErrors] using
        And.intro stateApprox parametersApprox
  | @cons exact runtime error step steps gradientApprox assumptionsHold tail =>
      intro stateApprox parametersApprox
      have stepApprox := contract.updateApprox
        exact.optimizerState
        runtime.optimizerState
        error.optimizerStateError
        exact.parameters
        runtime.parameters
        error.parameterError
        step.exactGradient
        step.runtimeGradient
        step.gradientError
        step.assumptions
        stateApprox
        parametersApprox
        gradientApprox
        assumptionsHold
      rcases stepApprox with ⟨nextStateApprox, nextParametersApprox⟩
      simpa [RunApprox, runExact, runRuntime, runErrors] using run_approx contract
        (exact := contract.updateExact
          exact.optimizerState exact.parameters step.exactGradient)
        (runtime := contract.updateRuntime
          runtime.optimizerState runtime.parameters step.runtimeGradient)
        (error := contract.nextError
          error.optimizerStateError
          error.parameterError
          step.gradientError
          runtime.optimizerState
          runtime.parameters
          step.runtimeGradient
          step.assumptions)
        (steps := steps)
        tail nextStateApprox nextParametersApprox
termination_by steps.size
decreasing_by simp

end NumericalStepContract

end
end Proofs.RuntimeApprox.Optimizer
