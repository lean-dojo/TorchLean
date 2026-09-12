/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.Runtime.Optim.Optimizers

/-!
# Optimizer Law Interface

This module gives TorchLean optimizers a small proof layer interface.

Runtime optimizers live in `NN.Runtime.Optim.Optimizers` as executable tensor equations.  The
definitions below package those equations as shape-polymorphic optimizers and provide a common
interface for independent update specifications.

The pattern for adding an optimizer is:

1. define a pure per-tensor `init` and `update` equation;
2. package it as a `TensorOptimizer`;
3. state an independent `StepSpec` when a proof-facing recurrence is needed;
4. prove optimizer-specific algebraic facts as consequences of that generic interface.

TorchLean does not register a second, definitionally identical copy of every runtime update. Such a
copy would add a theorem name without adding an independent claim. Higher-level trainer proofs can
instead quantify over any `TensorOptimizer`, reason about whole gradient streams via `runSteps`,
and introduce a `StepSpec` only when its equations come from a separate mathematical description.
-/

@[expose] public section

namespace Optim

open Spec TorchLean
open TorchLean TorchLean.Tensor

variable {α : Type} [TorchLean.Storage α] [Context α]

/-- A shape-polymorphic per-tensor optimizer. -/
structure TensorOptimizer (α : Type) [TorchLean.Storage α] [Context α] where
  /-- Per-parameter optimizer state for a tensor of shape `s`. -/
  State : Shape → Type
  /-- Initialize optimizer state from the current parameter tensor. -/
  init : {s : Shape} → Tensor α s → State s
  /-- One update from state, parameters, and gradients. -/
  update : {s : Shape} → State s → Tensor α s → Tensor α s →
    Step α s (State s)

namespace TensorOptimizer

section ConcreteOptimizers

variable [DecidableRel ((· > ·) : α → α → Prop)]

/-- Package plain SGD as a `TensorOptimizer`. -/
def sgd (learningRate : α) : TensorOptimizer α :=
  { State := SGD.State α
    init := fun {s} parameters =>
      SGD.init (α := α) (s := s) learningRate parameters
    update := fun {_s} state parameters gradients =>
      SGD.update (α := α) state parameters gradients }

/-- Package momentum SGD as a `TensorOptimizer`. -/
def momentumSGD (learningRate momentum : α) : TensorOptimizer α :=
  { State := MomentumSGD.State α
    init := fun {s} parameters =>
      MomentumSGD.init (α := α) (s := s) learningRate momentum parameters
    update := fun {_s} state parameters gradients =>
      MomentumSGD.update (α := α) state parameters gradients }

/-- Package AdaGrad as a `TensorOptimizer`. -/
def adagrad (learningRate epsilon : α) : TensorOptimizer α :=
  { State := AdaGrad.State α
    init := fun {s} parameters =>
      AdaGrad.init (α := α) (s := s) learningRate epsilon parameters
    update := fun {_s} state parameters gradients =>
      AdaGrad.update (α := α) state parameters gradients }

/-- Package RMSProp as a `TensorOptimizer`. -/
def rmsprop (learningRate decay epsilon : α) : TensorOptimizer α :=
  { State := RMSProp.State α
    init := fun {s} parameters =>
      RMSProp.init (α := α) (s := s) learningRate decay epsilon parameters
    update := fun {_s} state parameters gradients =>
      RMSProp.update (α := α) state parameters gradients }

/-- Package Adam as a `TensorOptimizer`. -/
def adam (learningRate beta1 beta2 epsilon : α) : TensorOptimizer α :=
  { State := Adam.State α
    init := fun {s} parameters =>
      Adam.init (α := α) (s := s) learningRate beta1 beta2 epsilon parameters
    update := fun {_s} state parameters gradients =>
      Adam.update (α := α) state parameters gradients }

/-- Package AdamW as a `TensorOptimizer`. -/
def adamw (learningRate weightDecay beta1 beta2 epsilon : α) : TensorOptimizer α :=
  { State := AdamW.State α
    init := fun {s} parameters =>
      AdamW.init
        (α := α) (s := s) learningRate weightDecay beta1 beta2 epsilon parameters
    update := fun {_s} state parameters gradients =>
      AdamW.update (α := α) state parameters gradients }

/-- Package Adadelta as a `TensorOptimizer`. -/
def adadelta (learningRate rho epsilon : α) : TensorOptimizer α :=
  { State := Adadelta.State α
    init := fun {s} parameters =>
      Adadelta.init (α := α) (s := s) learningRate rho epsilon parameters
    update := fun {_s} state parameters gradients =>
      Adadelta.update (α := α) state parameters gradients }

/-- Package Muon-style orthogonalized momentum as a `TensorOptimizer`. -/
def muon (learningRate momentum : α)
    (orthogonalizer : {s : Shape} → Muon.Orthogonalizer α s :=
      fun {s} => Muon.identityOrthogonalizer (α := α) (s := s)) :
    TensorOptimizer α :=
  { State := Muon.State α
    init := fun {s} parameters =>
      Muon.init
        (α := α) (s := s) learningRate momentum
        (orthogonalizer (s := s)) parameters
    update := fun {_s} state parameters gradients =>
      Muon.update (α := α) state parameters gradients }

end ConcreteOptimizers

/-- Run one optimizer step on its current state and parameters. -/
def step (opt : TensorOptimizer α) {s : Shape}
    (current : Step α s (opt.State s)) (gradients : Tensor α s) :
    Step α s (opt.State s) :=
  opt.update current.optimizerState current.parameters gradients

/-- Run a finite stream of gradients through an optimizer. -/
def runSteps (opt : TensorOptimizer α) {s : Shape}
    (current : Step α s (opt.State s)) (gradients : Array (Tensor α s)) :
    Step α s (opt.State s) :=
  gradients.foldl (fun current gradient => opt.step current gradient) current

/--
Splitting a gradient stream and running the two pieces sequentially gives the same state and
parameters as running the concatenated stream.
-/
theorem runSteps_append (opt : TensorOptimizer α) {s : Shape}
    (current : Step α s (opt.State s)) (left right : Array (Tensor α s)) :
    opt.runSteps current (left ++ right) = opt.runSteps (opt.runSteps current left) right := by
  simp [runSteps]

/-- Optimizer state after a finite gradient stream. -/
def stateAfter (opt : TensorOptimizer α) {s : Shape}
    (current : Step α s (opt.State s)) (gradients : Array (Tensor α s)) :
    opt.State s :=
  (opt.runSteps current gradients).optimizerState

/-- Optimizer parameters after a finite gradient stream. -/
def parametersAfter (opt : TensorOptimizer α) {s : Shape}
    (current : Step α s (opt.State s)) (gradients : Array (Tensor α s)) :
    Tensor α s :=
  (opt.runSteps current gradients).parameters

/-- State projection of `runSteps_append`. -/
theorem stateAfter_append (opt : TensorOptimizer α) {s : Shape}
    (current : Step α s (opt.State s)) (left right : Array (Tensor α s)) :
    opt.stateAfter current (left ++ right) =
      opt.stateAfter (opt.runSteps current left) right := by
  exact congrArg Step.optimizerState (opt.runSteps_append current left right)

/-- Parameter projection of `runSteps_append`. -/
theorem parametersAfter_append (opt : TensorOptimizer α) {s : Shape}
    (current : Step α s (opt.State s)) (left right : Array (Tensor α s)) :
    opt.parametersAfter current (left ++ right) =
      opt.parametersAfter (opt.runSteps current left) right := by
  exact congrArg Step.parameters (opt.runSteps_append current left right)

end TensorOptimizer

/-! ## Generic step specifications -/

/--
Proof-facing specification of one optimizer step.

An optimizer-specific file only has to identify the next-state and next-parameter equations once.
The generic theorems below then lift that one-step fact to whole finite gradient streams.
-/
structure StepSpec (opt : TensorOptimizer α) where
  /-- Spec equation for the next optimizer state. -/
  nextState : {s : Shape} → opt.State s → Tensor α s → Tensor α s → opt.State s
  /-- Spec equation for the next parameter tensor. -/
  nextParameters : {s : Shape} → opt.State s → Tensor α s → Tensor α s → Tensor α s
  /-- The executable optimizer update agrees with the stated step equations. -/
  update_eq : ∀ {s : Shape} (state : opt.State s) (parameters gradients : Tensor α s),
    opt.update state parameters gradients =
      { optimizerState := nextState state parameters gradients
        parameters := nextParameters state parameters gradients }

namespace StepSpec

variable {opt : TensorOptimizer α}

/-- Run one step through the proof layer equations. -/
def step (law : StepSpec opt) {s : Shape}
    (current : Step α s (opt.State s)) (gradients : Tensor α s) :
    Step α s (opt.State s) :=
  { optimizerState :=
      law.nextState current.optimizerState current.parameters gradients
    parameters :=
      law.nextParameters current.optimizerState current.parameters gradients }

/-- Run a finite stream of gradients through the proof layer equations. -/
def runSteps (law : StepSpec opt) {s : Shape}
    (current : Step α s (opt.State s)) (gradients : Array (Tensor α s)) :
    Step α s (opt.State s) :=
  gradients.foldl (fun current gradient => law.step current gradient) current

/-- A registered step spec agrees with the executable optimizer for one step. -/
theorem step_eq_optimizer_step (law : StepSpec opt) {s : Shape}
    (current : Step α s (opt.State s)) (gradients : Tensor α s) :
    law.step current gradients = opt.step current gradients := by
  simp [step, TensorOptimizer.step, law.update_eq]

/--
A registered one-step optimizer spec agrees with the executable optimizer over any finite gradient
stream.  This is the general theorem optimizer-specific registrations feed into.
-/
theorem runSteps_eq_optimizer_runSteps (law : StepSpec opt) {s : Shape}
    (current : Step α s (opt.State s)) (gradients : Array (Tensor α s)) :
    law.runSteps current gradients = opt.runSteps current gradients := by
  have hstep :
      (fun (current : Step α s (opt.State s)) (gradient : Tensor α s) =>
        law.step current gradient) =
      (fun current gradient => opt.step current gradient) := by
    funext current gradient
    exact law.step_eq_optimizer_step current gradient
  unfold runSteps TensorOptimizer.runSteps
  rw [hstep]

/--
The proof layer equations compose over concatenated gradient streams just like the executable
optimizer.
-/
theorem runSteps_append (law : StepSpec opt) {s : Shape}
    (current : Step α s (opt.State s))
    (left right : Array (Tensor α s)) :
    law.runSteps current (left ++ right) = law.runSteps (law.runSteps current left) right := by
  simp [runSteps]

end StepSpec

/-! ## Muon comparison laws -/

variable [DecidableRel ((· > ·) : α → α → Prop)]

namespace Muon

/--
If a Muon backend returns the fresh momentum buffer unchanged on this step, then the parameter
update agrees with momentum SGD for this step.
-/
theorem update_params_eq_momentumSGD_of_apply_eq {s : Shape}
    (state : State α s) (parameters gradients : Tensor α s)
    (happly :
      state.orthogonalizer.apply
        (updateMomentumBuffer state.momentumBuffer state.momentum gradients) =
        updateMomentumBuffer state.momentumBuffer state.momentum gradients) :
    (update state parameters gradients).parameters =
      (MomentumSGD.update
        ({ learningRate := state.learningRate
           momentum := state.momentum
           momentumBuffer := state.momentumBuffer } :
          MomentumSGD.State α s)
        parameters gradients).parameters := by
  simp [update, MomentumSGD.update, happly]

/--
Initialized version of `update_params_eq_momentumSGD_of_apply_eq`.
-/
theorem init_update_params_eq_momentumSGD_of_apply_eq {s : Shape}
    (learningRate momentum : α) (orthogonalizer : Orthogonalizer α s)
    (parameters gradients : Tensor α s)
    (happly :
      orthogonalizer.apply (updateMomentumBuffer (Tensor.full s 0) momentum gradients) =
        updateMomentumBuffer (Tensor.full s 0) momentum gradients) :
    (update
      (init learningRate momentum orthogonalizer parameters)
      parameters gradients).parameters =
    (MomentumSGD.update
      (MomentumSGD.init learningRate momentum parameters)
      parameters gradients).parameters := by
  exact update_params_eq_momentumSGD_of_apply_eq
    (state := init learningRate momentum orthogonalizer parameters)
    (parameters := parameters) (gradients := gradients) happly

end Muon

end Optim
