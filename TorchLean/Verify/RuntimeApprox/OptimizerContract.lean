/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.Verify.RuntimeApprox.Graph.Forward
public import TorchLean.NN.Optimizer.Update

/-!
# Numerical Contracts for Optimizer Steps

An optimizer proof has two kinds of state, the mathematical recurrence and the rounded runtime
state, and `OptimizerContract` records their relation once. A concrete optimizer supplies its exact
and runtime update equations, a transformer for the state and parameter error bounds, and a proof
that one step preserves the relation. `runApprox_of_stepStreamApprox` then composes that local proof
over any finite gradient stream.

The contract is independent of SGD, Adam and the scalar backend, so the induction over a run is
proved once for all optimizers. Its conclusion is the numerical refinement claim that training
needs: after a finite run, the rounded state and parameters are within the propagated error bounds
of the exact-real ones.

For the distinction between local rounding errors and their propagation through an iterative
algorithm, see N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., 2002.
-/

@[expose] public section

namespace TorchLean.OptimizerContract

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
rounded scalar-expression bounds, so one finite-run theorem covers both cases.
-/
structure _root_.TorchLean.OptimizerContract (R : Type) (toSpec : R → ℝ) where
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
      OptimizerStep ℝ shape (ExactState shape)
  /-- One rounded runtime optimizer update. -/
  updateRuntime : {shape : Shape} →
    RuntimeState shape → Tensor R shape → Tensor R shape →
      OptimizerStep R shape (RuntimeState shape)
  /-- Compute the next state/parameter bounds from current errors and runtime values. -/
  nextError : {shape : Shape} → StateError shape → ℝ → ℝ →
    RuntimeState shape → Tensor R shape → Tensor R shape → StepAssumptions shape →
      StepError StateError shape
  /-- Labelled real-valued components of a state bound. Displaying them requires a separate
  executable representation. -/
  stateErrorReport : {shape : Shape} → StateError shape → Array (String × ℝ)
  /-- Labelled real-valued components of one step's side data. -/
  assumptionReport : {shape : Shape} → StepAssumptions shape → Array (String × ℝ)
  /-- One-step soundness: if the state, parameters and gradients are within their bounds and the
  step's assumptions hold, the updated state and parameters are within the bounds that
  `nextError` computes. -/
  update_sound : ∀ {shape : Shape}
      (exactState : ExactState shape) (runtimeState : RuntimeState shape)
      (stateError : StateError shape)
      (exactParameters : Tensor ℝ shape) (runtimeParameters : Tensor R shape)
      (parameterError : ℝ)
      (exactGradients : Tensor ℝ shape) (runtimeGradients : Tensor R shape)
      (gradientError : ℝ)
      (assumptions : StepAssumptions shape),
    stateApprox exactState runtimeState stateError →
    ApproxTensor (α := R) (toSpec := toSpec)
      exactParameters runtimeParameters parameterError →
    ApproxTensor (α := R) (toSpec := toSpec)
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
        ApproxTensor (α := R) (toSpec := toSpec)
          (updateExact exactState exactParameters exactGradients).parameters
          (updateRuntime runtimeState runtimeParameters runtimeGradients).parameters
          error.parameterError

variable {R : Type} {toSpec : R → ℝ}

/-- Exact, rounded, and error information for one optimizer update. -/
structure StepInput (contract : OptimizerContract R toSpec) (shape : Shape) where
  /-- Exact-real gradient. -/
  exactGradient : Tensor ℝ shape
  /-- Rounded runtime gradient. -/
  runtimeGradient : Tensor R shape
  /-- Infinity-norm error relating the exact and runtime gradients. -/
  gradientError : ℝ
  /-- Optimizer-specific side data and domain margins. -/
  assumptions : contract.StepAssumptions shape

/-- Fold a finite step stream using the exact-real recurrence. -/
def runExact (contract : OptimizerContract R toSpec) {shape : Shape}
    (initial : OptimizerStep ℝ shape (contract.ExactState shape))
    (steps : Array (StepInput contract shape)) :
    OptimizerStep ℝ shape (contract.ExactState shape) :=
  steps.foldl
    (fun current step =>
      contract.updateExact current.optimizerState current.parameters step.exactGradient)
    initial

/-- Fold the same finite step stream using the rounded recurrence. -/
def runRuntime (contract : OptimizerContract R toSpec) {shape : Shape}
    (initial : OptimizerStep R shape (contract.RuntimeState shape))
    (steps : Array (StepInput contract shape)) :
    OptimizerStep R shape (contract.RuntimeState shape) :=
  steps.foldl
    (fun current step =>
      contract.updateRuntime current.optimizerState current.parameters step.runtimeGradient)
    initial

/-- Propagate the state and parameter error bounds along a step stream, threading the runtime state
that `nextError` reads. -/
def runErrors (contract : OptimizerContract R toSpec) {shape : Shape}
    (initialError : StepError contract.StateError shape)
    (initialRuntime : OptimizerStep R shape (contract.RuntimeState shape))
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

The indices thread exact state, runtime state and error bounds through the same recurrence as
`runExact`, `runRuntime` and `runErrors`. Each step needs its gradient bound and a proof of
`assumptionsHold` (for an adaptive optimizer, the domain margins) at the states the run reaches.
These conditions are proved, and nothing checks them at run time.
-/
inductive StepStreamApprox (contract : OptimizerContract R toSpec) {shape : Shape} :
    OptimizerStep ℝ shape (contract.ExactState shape) →
    OptimizerStep R shape (contract.RuntimeState shape) →
    StepError contract.StateError shape →
    Array (StepInput contract shape) → Prop
  /-- The empty run. -/
  | empty {exact runtime error} : StepStreamApprox contract exact runtime error #[]
  /-- A step whose gradient bound and assumptions hold at the current states, followed by a run
  from the updated states. -/
  | cons {exact runtime error step steps} :
      ApproxTensor (α := R) (toSpec := toSpec)
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

/-- Soundness at the end of a finite run: the final exact and runtime states satisfy `stateApprox`
at the bound that `runErrors` computes, and the final parameters are within its parameter bound. -/
def RunApprox (contract : OptimizerContract R toSpec) {shape : Shape}
    (exact : OptimizerStep ℝ shape (contract.ExactState shape))
    (runtime : OptimizerStep R shape (contract.RuntimeState shape))
    (error : StepError contract.StateError shape)
    (steps : Array (StepInput contract shape)) : Prop :=
  contract.stateApprox
      (contract.runExact exact steps).optimizerState
      (contract.runRuntime runtime steps).optimizerState
      (contract.runErrors error runtime steps).optimizerStateError ∧
    ApproxTensor (α := R) (toSpec := toSpec)
      (contract.runExact exact steps).parameters
      (contract.runRuntime runtime steps).parameters
      (contract.runErrors error runtime steps).parameterError

/-- A one-step contract composes over a finite run: if the initial states satisfy `stateApprox`
and the initial parameters are within their bound, a `StepStreamApprox` proof gives `RunApprox`. -/
theorem runApprox_of_stepStreamApprox (contract : OptimizerContract R toSpec) {shape : Shape}
    {exact : OptimizerStep ℝ shape (contract.ExactState shape)}
    {runtime : OptimizerStep R shape (contract.RuntimeState shape)}
    {error : StepError contract.StateError shape}
    {steps : Array (StepInput contract shape)}
    (stepsApprox : StepStreamApprox contract exact runtime error steps) :
    contract.stateApprox
      exact.optimizerState runtime.optimizerState error.optimizerStateError →
    ApproxTensor (α := R) (toSpec := toSpec)
      exact.parameters runtime.parameters error.parameterError →
    RunApprox contract exact runtime error steps := by
  cases stepsApprox with
  | empty =>
      intro stateApprox parametersApprox
      simpa [RunApprox, runExact, runRuntime, runErrors] using
        And.intro stateApprox parametersApprox
  | @cons exact runtime error step steps gradientApprox assumptionsHold tail =>
      intro stateApprox parametersApprox
      have stepApprox := contract.update_sound
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
      simpa [RunApprox, runExact, runRuntime, runErrors]
        using runApprox_of_stepStreamApprox contract
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

end
end TorchLean.OptimizerContract
