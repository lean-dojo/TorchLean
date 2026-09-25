/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import NN.API.Precision
public import NN.API.Trainer.Constructor

/-!
# Typed trainer precision and snapshot regressions

Independent rational expectations detect a binary64 intermediate in input, parameter, loss, or
update paths. Checkpoint comparisons use exact encodings, including after the live session changes.
-/

@[expose] public section

namespace NN.Tests.API.PrecisionTrainer

open TorchLean
open FloatLib.Floats

def expect (label : String) (condition : Bool) : IO Unit := do
  unless condition do
    throw <| IO.userError s!"typed trainer precision: {label}"

def expectFailure {α : Type} (label : String) (action : IO α) : IO Unit := do
  let failed ← try
      let _ ← action
      pure false
    catch _ => pure true
  expect (label ++ " should fail") failed

def trainer : TorchLean.Trainer [1] [1] :=
  Trainer.new (nn.linear 1 1)
    { objective := .mse
      optimizer := optim.sgd { learningRate := 0.25 }
      seed := 17 }

def inputGap : Rat := 1 / (2 ^ 80 : Nat)
def stateGap : Rat := 1 / (2 ^ 100 : Nat)

def sample : Sample.Supervised (ExecFloat.Binary 15 112) [1] [1] :=
  { input := [Rat.cast (1 + inputGap)]
    target := [Rat.cast (1 + inputGap)] }

def initialState : nn.State (ExecFloat.Binary 15 112) (nn.stateShapes trainer.model) :=
  (nn.State.full 1).set ⟨1, by decide⟩ (Tensor.full _ (Rat.cast (1 + stateGap)))

/-- Shape metadata and exact scalar words allow comparison with an opaque result's state layout. -/
def encodeState {α : Type} [Storage α] [Checkpoint.Encoding α] {shapes : List Shape}
    (state : nn.State α shapes) : Lean.Json :=
  Runtime.Autograd.Model.StateIO.stateToJsonBits (nn.State.Internal.toTensorPack state)

def expectValue (label : String) (actual : ExecFloat.Binary 15 112) (expected : Rat) :
    IO Unit :=
  let observed := ExecFloat.Binary.toRat? actual
  expect s!"{label}: expected {expected}, got {reprStr observed}" (observed == some expected)

/-- Check the input and state sentinels before any optimizer mutation. -/
def checkInitialValues {source : TorchLean.Trainer [1] [1]}
    (session : Trainer.Session source (ExecFloat.Binary 15 112)) :
    IO (ExecFloat.Binary 15 112) := do
  expect "explicit initial state retains its encoding"
    (encodeState (← session.state) == encodeState initialState)
  expectValue "input sentinel" sample.input[0] (1 + inputGap)
  expectValue "parameter sentinel" ((initialState.get ⟨1, by decide⟩).getScalar
    ⟨0, by decide⟩) (1 + stateGap)
  expectValue "typed prediction" (← session.predict sample.input)[0]
    (2 + inputGap + stateGap)
  let before ← session.loss sample
  expectValue "typed loss" before (1 + 2 * stateGap)
  let stream := Data.SampleStream.fromArray #[sample, sample]
  expectValue "typed stream mean" (← session.loss stream (batch := true)) (1 + 2 * stateGap)
  pure before

def checkEmptyBatch {source : TorchLean.Trainer [1] [1]}
    (session : Trainer.Session source (ExecFloat.Binary 15 112)) : IO Unit := do
  let empty := Data.SampleStream.fromArray
    (#[] : Array (Sample.Supervised (ExecFloat.Binary 15 112) [1] [1]))
  expectValue "empty typed stream" (← session.loss empty (batch := true)) 0
  expectFailure "empty typed update" <|
    session.step (#[] : Array (Sample.Supervised (ExecFloat.Binary 15 112) [1] [1]))
      (batch := true)
  expect "empty update does not advance steps" ((← session.steps) == 0)
  expect "empty update does not mutate state"
    (encodeState (← session.state) == encodeState initialState)

/-- Independent rational expectations for the two parameter updates. -/
def checkUpdatedState
    (updated : nn.State (ExecFloat.Binary 15 112) (nn.stateShapes trainer.model)) : IO Unit := do
  let weight : Tensor (ExecFloat.Binary 15 112) [1, 1] := updated.get ⟨0, by decide⟩
  let bias : Tensor (ExecFloat.Binary 15 112) [1] := updated.get ⟨1, by decide⟩
  expectValue "typed weight update" (Tensor.get2 weight 0 0) ((1 - inputGap - stateGap) / 2)
  expectValue "typed bias update" bias[0] ((1 + stateGap) / 2)

def checkUpdatedPrediction {source : TorchLean.Trainer [1] [1]}
    (session : Trainer.Session source (ExecFloat.Binary 15 112)) :
    IO (ExecFloat.Binary 15 112) := do
  expectValue "updated prediction" (← session.predict sample.input)[0] 1
  let after ← session.loss sample
  expectValue "updated loss" after (inputGap * inputGap)
  pure after

/-- Reports keep typed losses and name the scalar actually used by the session. -/
def checkReport (result : Trainer.Result [1] [1] (ExecFloat.Binary 15 112))
    (before after : ExecFloat.Binary 15 112) : IO Unit := do
  let report : Trainer.Report (ExecFloat.Binary 15 112) := result.report
  expect "report counts the update" (report.steps == 1)
  expectValue "report before loss" report.loss.before (1 + 2 * stateGap)
  expectValue "report after loss" report.loss.after (inputGap * inputGap)
  let format := Checkpoint.Encoding.format (α := ExecFloat.Binary 15 112)
  expect "report names the selected scalar"
    (report.scalarFormat? == some format && report.runtimeScalar == format)
  expect "summary retains typed losses and scalar identity"
    (report.summary == s!"steps=1 scalar={format} loss={before} -> {after}")

/-- Continuing the live session cannot change a finished result. -/
def checkFrozenUpdate {source : TorchLean.Trainer [1] [1]}
    (session : Trainer.Session source (ExecFloat.Binary 15 112))
    (result : Trainer.Result [1] [1] (ExecFloat.Binary 15 112))
    (frozen : Lean.Json) : IO Unit := do
  expect "result snapshots the updated state" (frozen == encodeState (← session.state))
  expectValue "result predicts with typed state" (← result.predict sample.input)[0] 1
  session.step { sample with target := [0] }
  expect "continued update changes live state" (encodeState (← session.state) != frozen)
  expect "continued update preserves the result" (encodeState (← result.state) == frozen)
  expectValue "continued update preserves result prediction" (← result.predict sample.input)[0] 1

/-- Both save surfaces round-trip exact state, including into a fresh typed session. -/
def checkCheckpointRoundtrip {source : TorchLean.Trainer [1] [1]}
    (session : Trainer.Session source (ExecFloat.Binary 15 112))
    (result : Trainer.Result [1] [1] (ExecFloat.Binary 15 112))
    (frozen : Lean.Json) (path : System.FilePath) : IO Unit := do
  session.save path
  let savedLive ← Checkpoint.State.load (α := ExecFloat.Binary 15 112) source.model path
  expect "session checkpoint retains the current typed state"
    (encodeState savedLive == encodeState (← session.state))
  result.save path
  let checkpoint ← IO.ofExcept (Lean.Json.parse (← IO.FS.readFile path))
  let savedFormat ← IO.ofExcept (checkpoint.getObjValAs? String "format")
  expect "result checkpoint uses the typed format"
    (savedFormat == Checkpoint.Encoding.format (α := ExecFloat.Binary 15 112))
  let loaded ← Checkpoint.State.load (α := ExecFloat.Binary 15 112) source.model path
  expect "result checkpoint retains every state bit" (encodeState loaded == frozen)
  let restored ← source.openTyped (α := ExecFloat.Binary 15 112)
  restored.load path
  expect "fresh typed session loads exact state" (encodeState (← restored.state) == frozen)
  expectValue "fresh typed session reproduces the prediction"
    (← restored.predict sample.input)[0] 1
  expect "load keeps the fresh step count" ((← restored.steps) == 0)

/-- Loading and updating live state also leave the result's state, prediction, and report frozen. -/
def checkFrozenLoad {source : TorchLean.Trainer [1] [1]}
    (session : Trainer.Session source (ExecFloat.Binary 15 112))
    (result : Trainer.Result [1] [1] (ExecFloat.Binary 15 112))
    (frozen : Lean.Json) (path : System.FilePath) : IO Unit := do
  let replacement : nn.State (ExecFloat.Binary 15 112) (nn.stateShapes source.model) :=
    nn.State.full 7
  Checkpoint.State.save source.model replacement path
  session.load path
  expect "load replaces the live state" (encodeState (← session.state) == encodeState replacement)
  expect "load retains the live step count" ((← session.steps) == 2)
  expect "load preserves the result snapshot" (encodeState (← result.state) == frozen)
  expectValue "load preserves result prediction" (← result.predict sample.input)[0] 1
  session.step sample
  expect "update after load preserves the result" (encodeState (← result.state) == frozen)
  expect "result report remains frozen" (result.report.steps == 1)
  expectValue "result report retains its original loss" result.report.loss.after
    (inputGap * inputGap)
  result.save path
  let savedAgain ← Checkpoint.State.load (α := ExecFloat.Binary 15 112) source.model path
  expect "save after update/load still writes the snapshot" (encodeState savedAgain == frozen)

/--
At the supplied input, the residual is `1 + stateGap`. Squaring rounds away `stateGap²`.
One SGD step gives weight `(1 - inputGap - stateGap)/2` and bias `(1 + stateGap)/2`.
The next prediction rounds to exactly one, leaving loss `inputGap²`.
-/
def checkPrecision (execution : Runtime.ExecutionMode) : IO Unit := do
  let source : TorchLean.Trainer [1] [1] :=
    { trainer with runtime := { trainer.runtime with execution } }
  let session ← source.openTyped (α := ExecFloat.Binary 15 112)
    (initialState? := some initialState)
  let before ← checkInitialValues session
  checkEmptyBatch session
  expectValue "step returns the typed pre-update loss"
    (← session.step sample (loss := true)) (1 + 2 * stateGap)
  checkUpdatedState (← session.state)
  let after ← checkUpdatedPrediction session
  let result ← session.finish { before, after }
  checkReport result before after
  let frozen := encodeState (← result.state)
  checkFrozenUpdate session result frozen
  IO.FS.withTempFile fun _ path => do
    checkCheckpointRoundtrip session result frozen path
    checkFrozenLoad session result frozen path
  expectFailure "typed result verification" <| result.verify [1] (radius := 0.1)
  IO.println s!"  typed trainer exact values and snapshots ({reprStr execution}): passed"

/-- A failed load must leave both the live state and the optimizer-step count untouched. -/
def expectLoadRejected
    (session : Trainer.Session trainer (ExecFloat.Binary 15 112))
    (label : String) (path : System.FilePath) : IO Unit := do
  let before := encodeState (← session.state)
  let steps ← session.steps
  expectFailure label (session.load path)
  expect (label ++ " preserves state") (encodeState (← session.state) == before)
  expect (label ++ " preserves steps") ((← session.steps) == steps)

/--
Format identity includes width, bias, and exceptional-value encoding; loads are transactional.
-/
def checkCheckpointRejection : IO Unit := do
  let session ← trainer.openTyped (α := ExecFloat.Binary 15 112)
    (initialState? := some initialState)
  session.step sample
  let replacement : nn.State (ExecFloat.Binary 15 112) (nn.stateShapes trainer.model) :=
    nn.State.zeros
  let body := encodeState replacement
  IO.FS.withTempFile fun _ path => do
    for (label, format) in #[
        ("binary64 format", Checkpoint.Encoding.format (α := Float)),
        ("same-width different bias",
          Checkpoint.Encoding.format (α := ExecFloat.Binary 15 112 (bias := 16382))),
        ("same-width different exceptional encoding",
          Checkpoint.Encoding.format (α := ExecFloat.Binary 15 112 (encoding := .finite)))] do
      IO.FS.writeFile path (Lean.Json.mkObj
        [("format", Lean.Json.str format), ("state", body)]).compress
      expectLoadRejected session label path
    let tensors ← IO.ofExcept body.getArr?
    let malformed := tensors.set! 1 (Lean.Json.mkObj
      [("shape", Lean.toJson ([2] : List Nat)), ("values", Lean.Json.arr #[])])
    IO.FS.writeFile path (Lean.Json.mkObj
      [("format", Lean.Json.str (Checkpoint.Encoding.format (α := ExecFloat.Binary 15 112))),
        ("state", Lean.Json.arr malformed)]).compress
    expectLoadRejected session "malformed second tensor after a valid replacement weight" path

/-- Seeded initialization stays Float sourced, while the chosen session scalar is preserved. -/
def checkSeededDefaults : IO Unit := do
  let defaultState : nn.State Float (nn.stateShapes trainer.model) :=
    nn.initialState trainer.model
  let typedState := nn.initialState trainer.model (α := ExecFloat.Binary 15 112)
  let expected := defaultState.map
    (Tensor.map (Runtime.ofFloat (α := ExecFloat.Binary 15 112)))
  expect "typed seeded initializer converts the documented Float source"
    (encodeState typedState == encodeState expected)
  let typed ← trainer.openTyped (α := ExecFloat.Binary 15 112)
  expect "typed opening without explicit state uses the seeded initializer"
    (encodeState (← typed.state) == encodeState typedState)
  let legacy : Trainer.Session trainer ← trainer.open
  let nativeState := defaultState.map (Tensor.map fun (value : Float) => value.toFloat32.toFloat)
  expect "default opening retains binary32 runtime initialization"
    (encodeState (← legacy.state) == encodeState nativeState)
  let loss : Float ← legacy.loss { input := [1], target := [1] }
  let result : Trainer.Result [1] [1] ← legacy.finish { before := loss, after := loss }
  let report : Trainer.Report := result.report
  expect "legacy report retains its scalar identity"
    (report.scalarFormat?.isNone && report.runtimeScalar == "Float32")
  expect "legacy result retains the host Float boundary"
    (encodeState (← result.state) == encodeState nativeState)
  let onCuda : TorchLean.Trainer [1] [1] :=
    { trainer with runtime := { trainer.runtime with device := .cuda } }
  expectFailure "typed CUDA opening" (onCuda.openTyped (α := ExecFloat.Binary 15 112))

/-- Manual module constructors retain the same exact state override as typed sessions. -/
def checkManualModules : IO Unit := do
  let indexed : nn.IndexedModel [1] [1, 1] (Fin 2) :=
    (nn.build 17 (nn.embedding 2 1)).model [1]
  let indexedState : nn.State (ExecFloat.Binary 15 112) indexed.stateShapes :=
    nn.State.full (Rat.cast (1 + stateGap))
  for execution in [Runtime.ExecutionMode.eager, .typedGraph] do
    let ordinary ← nn.Module.instantiate trainer.model { execution }
      (α := ExecFloat.Binary 15 112) (initialState? := some initialState)
    expect "manual module keeps supplied parameter bits"
      (encodeState (← ordinary.state) == encodeState initialState)
    expectValue "manual module predicts with supplied typed parameters"
      (← ordinary.forward sample.input)[0] (2 + inputGap + stateGap)
    let lookup ← nn.IndexedModule.instantiate indexed { execution }
      (α := ExecFloat.Binary 15 112) (initialState? := some indexedState)
    expect "indexed module keeps supplied parameter bits"
      (encodeState (← lookup.state) == encodeState indexedState)
    let prediction : Tensor (ExecFloat.Binary 15 112) [1, 1] ← lookup.forward [1]
    expectValue "indexed module reads supplied typed parameters"
      (Tensor.get2 prediction 0 0) (1 + stateGap)

/-- Configured conversion preserves the exact binary64 source before destination rounding. -/
def checkConfiguredConversion : IO Unit := do
  expectValue "configured conversion retains the low binary64 significand bit"
    (Runtime.ofFloat (Float.ofBits 0x3ff0000000000001)) (1 + 1 / (2 ^ 52 : Nat))
  expectValue "configured conversion retains binary64 subnormals"
    (Runtime.ofFloat (Float.ofBits 1)) (1 / (2 ^ 1074 : Nat))
  expectValue "configured conversion retains the largest finite binary64 value"
    (Runtime.ofFloat (Float.ofBits 0x7fefffffffffffff))
    (((2 ^ 53 - 1) * 2 ^ 971 : Nat) : Rat)
  let negativeZero : ExecFloat.Binary 15 112 := Runtime.ofFloat (-0.0)
  expect "configured conversion preserves negative zero"
    (ExecFloat.Binary.toNatBits negativeZero == 2 ^ 127)
  let midpoint : ExecFloat.Binary 5 10 := Runtime.ofFloat 1.00048828125
  expect "configured conversion rounds a binary16 midpoint to even"
    (ExecFloat.Binary.toRat? midpoint == some 1)

/-- Scheduler rates must stay finite; a rate rounded to zero remains a valid no-op update. -/
def checkSchedulerConversion : IO Unit := do
  let overflowing : ExecFloat.Binary 5 10 := Runtime.ofFloat 70000
  let underflowing : ExecFloat.Binary 5 10 := Runtime.ofFloat 1e-20
  expect "overflow regression exceeds binary16 finite range"
    (ExecFloat.Binary.toRat? overflowing == none)
  expect "underflow regression rounds to binary16 zero"
    (ExecFloat.Binary.toRat? underflowing == some 0)
  expectFailure "scheduler rate overflowing binary16" <|
    trainer.openTyped (α := ExecFloat.Binary 5 10)
      (scheduler := some (.constant 70000))
  let session ← trainer.openTyped (α := ExecFloat.Binary 5 10)
    (scheduler := some (.constant 1e-20)) (initialState? := some (nn.State.full 1))
  let before := encodeState (← session.state)
  session.step { input := [1], target := [0] }
  expect "scheduler rate rounded to zero leaves parameters unchanged"
    (encodeState (← session.state) == before)
  expect "zero-rate update advances the step count" ((← session.steps) == 1)
  let tinyOverflow : ExecFloat.Binary 3 2 := Runtime.ofFloat 100
  expect "tiny-format regression exceeds its finite range"
    (ExecFloat.Binary.toRat? tinyOverflow == none)
  expectFailure "scheduler rate overflowing a tiny format" <|
    trainer.openTyped (α := ExecFloat.Binary 3 2) (scheduler := some (.constant 100))
  let invalidOptimizer : TorchLean.Trainer [1] [1] :=
    { trainer with runtime :=
        { trainer.runtime with optimizer := optim.sgd { learningRate := 100 } } }
  expectFailure "optimizer rate overflowing a tiny format" <|
    invalidOptimizer.openTyped (α := ExecFloat.Binary 3 2)

def run : IO Unit := do
  checkPrecision .eager
  checkPrecision .typedGraph
  checkCheckpointRejection
  checkSeededDefaults
  checkManualModules
  checkConfiguredConversion
  checkSchedulerConversion
  IO.println "  typed trainer precision, frozen snapshots, and checkpoint rejection: passed"

end NN.Tests.API.PrecisionTrainer
