/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

/-!
# Stateful training loops

`run` collects the outputs of a monadic step function. `Trainer.run` also calls a logger after
each successful step, with its zero-based index, updated state and loss/metric report.
-/

@[expose] public section

namespace Runtime
namespace Autograd
namespace Train

/-- A named scalar metric, such as accuracy, gradient norm or learning rate. -/
structure Metric (a : Type) where
  /-- Metric name (used as the key in logs). -/
  name : String
  /-- Metric value (typically the same scalar type as the loss). -/
  value : a

/-- Loss and optional named metrics from one training or evaluation step. -/
structure Report (a : Type) where
  /-- Scalar objective value for this step or evaluation pass. -/
  loss : a
  /-- Additional named scalar metrics for logging/monitoring. -/
  metrics : Array (Metric a) := #[]

/-- The state and output produced by one stateful step. -/
structure Step (State Output : Type) where
  /-- State to pass to the next step. -/
  nextState : State
  /-- Output produced by this step. -/
  output : Output

/-- The final state and collected outputs from a fixed-length run. -/
structure Trace (State Output : Type) where
  /-- State after every requested step has completed. -/
  finalState : State
  /-- Outputs in execution order. -/
  outputs : Array Output

/-- Run a monadic step function, collecting outputs in execution order.
Zero steps returns the initial state and no outputs without calling `step`. -/
def run {m : Type -> Type} [Monad m] {State Output : Type}
    (steps : Nat) (initialState : State)
    (step : State -> m (Step State Output)) : m (Trace State Output) := by
  let rec go : Nat -> State -> Array Output -> m (Trace State Output)
    | 0, state, outputs => pure { finalState := state, outputs := outputs }
    | n + 1, state, outputs => do
        let result ← step state
        go n result.nextState (outputs.push result.output)
  exact go steps initialState #[]

/-- Initial state, training step and post-update logger.
Use `fun _ _ _ => pure ()` for a logger that does nothing. -/
structure Trainer (m : Type -> Type) (state : Type) (a : Type) where
  /-- Initial training state. -/
  initialState : state
  /-- A single training step: update state and produce a report. -/
  step : state -> m (Step state (Report a))
  /-- Logger called after each successful step with its zero-based index. -/
  logger : Nat -> state -> Report a -> m Unit

namespace Trainer

/-- Run a trainer for `steps` steps, returning the final state and the collected reports. -/
def run {m : Type -> Type} [Monad m] {state a : Type}
    (steps : Nat) (trainer : Trainer m state a) :
    m (Trace state (Report a)) := by
  let rec go : Nat -> Nat -> state -> Array (Report a) ->
      m (Trace state (Report a))
    | 0, _, currentState, reports =>
        pure { finalState := currentState, outputs := reports }
    | n + 1, stepIndex, currentState, reports => do
        let result ← trainer.step currentState
        trainer.logger stepIndex result.nextState result.output
        go n (stepIndex + 1) result.nextState (reports.push result.output)
  exact go steps 0 trainer.initialState #[]

end Trainer

end Train
end Autograd
end Runtime
