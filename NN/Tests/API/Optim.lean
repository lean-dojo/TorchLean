/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Module
public import NN.API.Trainer.Scheduler
public import FloatLib.Floats.Formats.BinaryInterchange.Configured

import Lean.Data.Json.Parser

/-!
# Optimizer API Tests

Regression checks for optimizer configuration and descriptions, scalar conversion at the manual
module boundary, and algorithm selection.
-/

@[expose] public section

open FloatLib.Floats (ExecFloat)
open FloatLib.Floats.ExecFloat (Binary)
open FloatLib.Floats.Formats.BinaryInterchange (Model FloatFormat)

namespace NN.Tests.API.Optim

open TorchLean

def fail {α : Type} (message : String) : IO α :=
  throw <| IO.userError s!"optimizer API check failed: {message}"

/-- Read decimal text or the exact `Float.ofBits` expression emitted for exceptional cases. -/
def parseScalarDescription? (text : String) : Option Float :=
  match Lean.Json.parse text with
  | .ok (.num number) => some number.toFloat
  | _ =>
      match text.splitOn " " with
      | ["(Float.ofBits", word] => do
          let bits ← ((word.dropEnd 1).toString).toNat?
          if text == s!"(Float.ofBits {bits})" && bits < 2 ^ 64 then
            some (Float.ofBits bits.toUInt64)
          else
            none
      | _ => none

/-- Descriptions must preserve bits across binary64 range and precision boundaries. -/
def checkScalarDescriptions : IO Unit := do
  let values : Array Float := #[0.9999999, 1e-8, 0.03, 1.0 / 3.0,
    Float.ofBits 1, Float.ofBits 2,
    Float.ofBits 0x000fffffffffffff, Float.ofBits 0x0010000000000000,
    Float.ofBits 0x0010000000000001, Float.ofBits 0x3fefffffffffffff,
    1.0, Float.ofBits 0x3ff0000000000001, Float.ofBits 0x7fefffffffffffff]
  for value in #[0.0] ++ values ++ values.map (fun value => -value) do
    let text := optim.Optimizer.Internal.formatScalar value
    let some decoded := parseScalarDescription? text
      | fail s!"invalid scalar description: {text}"
    unless decoded.toBits == value.toBits do
      fail s!"scalar description changed bits {value.toBits}: {text}"
  let special : Array (Float × String) := #[
    (Float.ofBits 0x8000000000000000, "(Float.ofBits 9223372036854775808)"),
    (Float.inf, "(Float.ofBits 9218868437227405312)"),
    (-Float.inf, "(Float.ofBits 18442240474082181120)"),
    (Float.nan, "(Float.ofBits 9221120237041090560)")]
  for (value, expected) in special do
    let text := optim.Optimizer.Internal.formatScalar value
    unless text == expected do
      fail s!"special value must use an explicit Lean expression: {expected}"
    let some decoded := parseScalarDescription? text
      | fail s!"invalid special-value description: {text}"
    unless decoded.toBits == value.toBits do
      fail s!"special-value description changed bits: {text}"

/-- Copying a description must not turn a valid Adam coefficient into the invalid value one. -/
def checkOptimizerDescription : IO Unit := do
  let config : optim.Adam.Config := { learningRate := 0.01, beta2 := 0.9999999 }
  let optimizer := optim.adam config
  let expected :=
    "optim.adam { learningRate := 1.0000000000000000e-2, " ++
    "beta1 := 9.0000000000000002e-1, beta2 := 9.9999990000000005e-1, " ++
    "epsilon := 1.0000000000000000e-8 }"
  unless optimizer.describe == expected && toString optimizer == expected &&
      reprStr optimizer == expected do
    fail "describe, ToString, and Repr must retain the Adam coefficients"
  -- These literals are the fields in `expected`, checked by Lean's term elaborator.
  let rebuilt : optim.Adam.Config := {
    learningRate := 1.0000000000000000e-2
    beta1 := 9.0000000000000002e-1
    beta2 := 9.9999990000000005e-1
    epsilon := 1.0000000000000000e-8 }
  for (original, decoded) in #[
      (config.learningRate, rebuilt.learningRate), (config.beta1, rebuilt.beta1),
      (config.beta2, rebuilt.beta2), (config.epsilon, rebuilt.epsilon)] do
    unless original.toBits == decoded.toBits do
      fail "rebuilding the described Adam configuration changed a field's bits"
  unless optimizer.validate.isOk && optimizer.validateFloat32.isOk &&
      (optim.adam rebuilt).validate.isOk && (optim.adam rebuilt).validateFloat32.isOk do
    fail "copying the Adam description must preserve binary64 and binary32 validity"

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
  checkScalarDescriptions
  checkOptimizerDescription
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
    checkManualOptimizer (α := (Binary 8 23)) "IEEE32" optimizer false
    checkManualOptimizer (α := Float) "Float" optimizer true
    checkManualOptimizer (α := Float) "Float/CUDA" optimizer false { device := .cuda }
    unless (optimizer.validateFor (α := CustomScalar)).isOk do
      fail "custom scalars without a validation override must retain binary64 input checks"
    if (optimizer.validateFor (α := Runtime.Autograd.Model.Dual Float32)).isOk ||
        (optimizer.validateFor
          (α := TorchLean.Complex (Binary 8 23))).isOk then
      fail "dual and complex scalars must inherit the component's validation rounding"
  let ordinary := optim.adam { learningRate := 0.001 }
  unless ordinary.validateFloat32.isOk do
    fail "default Adam settings should remain valid in binary32"
  checkManualOptimizer (α := Float32) "Float32" ordinary true
  checkManualOptimizer (α := (Binary 8 23)) "IEEE32" ordinary true
  checkManualOptimizer (α := Float) "Float" ordinary true
  checkManualOptimizer (α := Float) "Float/CUDA" ordinary true { device := .cuda }
  let negative := optim.sgd { learningRate := -1e-300 }
  unless !negative.validateFloat32.isOk do
    fail "binary32 underflow must not hide a negative input learning rate"
  checkManualOptimizer (α := Float32) "Float32" negative false
  checkManualOptimizer (α := (Binary 8 23)) "IEEE32" negative false
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
