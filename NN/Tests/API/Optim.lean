/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Module
public import NN.API.Trainer.Scheduler

/-!
# Optimizer API Tests

Regression checks for optimizer configuration, scalar conversion at the manual module boundary,
and algorithm selection.
-/

@[expose] public section

namespace NN.Tests.API.Optim

open TorchLean

def fail {α : Type} (message : String) : IO α :=
  throw <| IO.userError s!"optimizer API check failed: {message}"

/-- A user-defined scalar can retain the original conversion-only instance contract. -/
structure CustomScalar where
  value : Float

instance : Runtime.FromFloat CustomScalar where
  ofFloat value := ⟨value⟩

/-- Invalid settings must fail before optimizer binding reaches the caller's continuation. -/
def checkManualOptimizer {α : Type}
    [TorchLean.Storage α] [Context α] [Runtime.FromFloat α]
    (label : String) (optimizer : optim.Optimizer) (accepts : Bool)
    (runtime : Runtime.Config := {}) : IO Unit := do
  let entered ← IO.mkRef false
  let succeeded ← try
    Module.Internal.withConfiguredOptimizer (α := α) (stateShapes := [])
        runtime optimizer fun _ => entered.set true
    pure true
  catch _ =>
    pure false
  unless succeeded == accepts && (← entered.get) == accepts do
    fail s!"manual {label} validation or optimizer binding disagreed with expected {accepts}"

def run : IO Unit := do
  let algorithm ←
    match optim.Algorithm.parse "adamw" with
    | .ok algorithm => pure algorithm
    | .error message => fail s!"could not parse adamw: {message}"
  let configured := algorithm.configure 0.001
  unless configured.learningRate == 0.001 && configured.validate.isOk do
    fail "AdamW command defaults produced an invalid optimizer"

  match optim.Algorithm.parse "not-an-optimizer" with
  | .ok _ => fail "accepted an unknown optimizer"
  | .error _ => pure ()

  for optimizer in #[
      optim.sgd { learningRate := 1e300 },
      optim.sgd { learningRate := 0.1, momentum := 0.999999999 },
      optim.adaGrad { learningRate := 0.1, epsilon := 1e-300 },
      optim.rmsProp { learningRate := 0.1, decay := 0.999999999 },
      optim.adam { learningRate := 0.1, beta1 := 0.999999999 },
      optim.adam { learningRate := 0.1, beta2 := 0.999999999 },
      optim.adamW { learningRate := 0.1, weightDecay := 1e300 },
      optim.adaDelta { rho := 0.999999999 }] do
    unless optimizer.validate.isOk && !optimizer.validateFloat32.isOk do
      fail "configuration must be valid in binary64 but invalid after binary32 conversion"
    checkManualOptimizer (α := Float32) "Float32" optimizer false
    checkManualOptimizer (α := TorchLean.Floats.IEEE754.IEEE32Exec) "IEEE32" optimizer false
    checkManualOptimizer (α := Float) "Float" optimizer true
    checkManualOptimizer (α := Float) "Float/CUDA" optimizer false { device := .cuda }
    unless (optimizer.validateFor (α := CustomScalar)).isOk do
      fail "custom scalars without a validation override must retain binary64 input checks"
    if (optimizer.validateFor (α := Runtime.Autograd.Model.Dual Float32)).isOk ||
        (optimizer.validateFor
          (α := TorchLean.Complex TorchLean.Floats.IEEE754.IEEE32Exec)).isOk then
      fail "dual and complex scalars must inherit the component's validation rounding"
  let ordinary := optim.adam { learningRate := 0.001 }
  unless ordinary.validateFloat32.isOk do
    fail "default Adam settings should remain valid in binary32"
  checkManualOptimizer (α := Float32) "Float32" ordinary true
  checkManualOptimizer (α := TorchLean.Floats.IEEE754.IEEE32Exec) "IEEE32" ordinary true
  checkManualOptimizer (α := Float) "Float" ordinary true
  checkManualOptimizer (α := Float) "Float/CUDA" ordinary true { device := .cuda }
  let negative := optim.sgd { learningRate := -1e-300 }
  unless !negative.validateFloat32.isOk do
    fail "binary32 underflow must not hide a negative input learning rate"
  checkManualOptimizer (α := Float32) "Float32" negative false
  checkManualOptimizer (α := TorchLean.Floats.IEEE754.IEEE32Exec) "IEEE32" negative false
  checkManualOptimizer (α := Float) "Float" negative false
  checkManualOptimizer (α := Float) "Float/CUDA" negative false { device := .cuda }

  for schedule in #[
      Trainer.Scheduler.constant 1e300,
      Trainer.Scheduler.step 1e300 2,
      Trainer.Scheduler.exponential 1e300 0.9,
      Trainer.Scheduler.warmupCosine 1e300 0.0 2 4] do
    unless (Trainer.Scheduler.validate schedule).isOk &&
        !(Trainer.Scheduler.validateFloat32 schedule).isOk do
      fail "schedule must reject a rate that overflows binary32"
  unless (Trainer.Scheduler.validateFloat32
      (Trainer.Scheduler.warmupCosine 0.01 0.001 2 4)).isOk do
    fail "ordinary warmup schedule should be valid in binary32"
  let peak := 1e308
  unless Trainer.Scheduler.learningRateAt
      (Trainer.Scheduler.warmupCosine peak 0.0 2 4) 1 == peak do
    fail "the final warmup update must reach a finite peak without intermediate overflow"

end NN.Tests.API.Optim
