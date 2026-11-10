/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Session.Apply
public import TorchLean.NN.Session.Optimizer
public import TorchLean.NN.ScalarTrainer.Checkpoint
public import TorchLean.Tensor.Literal

/-!
# Parameter Storage Aliases

Parameters on LibTorch's CUDA backend. A recording snapshots the parameter's value, tied AdamW
parameters follow an independent run whose gradients are summed before the optimizer, a retied
layout with separate histories is rejected without updating storage, and the shared AdamW
checkpoint restores byte for byte. Native binary32 division keeps the eager successive-division
pullback at the extremes.
-/

public section

namespace TorchLean.Test.ParameterAliases

open Backend.Device ParamList Param
open Session

private def check (label : String) (condition : Bool) : IO Unit := do
  unless condition do
    throw <| IO.userError s!"parameter aliases: {label}"

private def rejects (label : String) (action : IO Unit) : IO Unit := do
  let rejected ← try
    action
    pure false
  catch _ => pure true
  check label rejected

private def scalar (value : Float := 2.0) : IO (Param Float []) :=
  Param.create (Tensor.scalar value)

private def bits (p : Param Float []) : IO UInt64 := do
  syncToHost p
  pure (← p.value.get).item.toBits

/-- A recording snapshots the parameter's value across host mutation, repeated reads of one
parameter form one storage group, and resetting one session keeps another session's snapshots. -/
def checkSnapshots (device : Backend.Device) : IO Unit := do
  let p ← scalar 1.0
  let session ← Session.create (α := Float) { device }
  let observer ← Session.create (α := Float) { device }
  let observed ← observer.use p
  let first ← session.use p
  setHostValue p (Tensor.scalar 5.0)
  let second ← session.use p
  let groups ← session.parameterGroups
  check "two reads form one storage group"
    (groups.map (·.leaves) == #[#[first.id, second.id]])
  check "first recording remains a snapshot" ((← session.getValue first).item == 1.0)
  check "second recording observes host mutation" ((← session.getValue second).item == 5.0)
  session.resetTape
  check "reset preserves other session snapshot" ((← observer.getValue observed).item == 1.0)
  observer.resetTape

private def triple (p q : Param Float []) : ParamList Float [[], [], []] :=
  .cons p (.cons q (.cons p .nil))

private def cudaGradients (a b c : Float) : IO LibTorch.Tape.SparseGradMap := do
  let first ← TensorTransfer.toSomeBuffer (Tensor.scalar a)
  let second ← TensorTransfer.toSomeBuffer (Tensor.scalar b)
  let third ← TensorTransfer.toSomeBuffer (Tensor.scalar c)
  pure <| (Std.HashMap.emptyWithCapacity : LibTorch.Tape.SparseGradMap)
    |>.insert 0 first |>.insert 1 second |>.insert 2 third

private def recordTriple (session : Session Float) (p q : Param Float []) : IO Unit := do
  session.resetTape
  discard <| session.use p
  discard <| session.use q
  discard <| session.use p

private def firstMomentBytes (state : CUDAAdamState) (index : Nat) : IO ByteArray := do
  let some entry := state.get? index
    | throw <| IO.userError "parameter aliases: missing CUDA Adam state"
  LibTorch.Buffer.toBytesIO entry.m

/-- After an untied CUDA AdamW step, a step that ties the two slots is rejected without updating
storage, since their optimizer histories are separate. -/
def checkCUDARetie : IO Unit := do
  let p ← scalar
  let q ← scalar
  let session ← Session.create (α := Float) { device := cuda }
  let config ← IO.mkRef (none : Option CUDAAdamConfig)
  let state ← IO.mkRef (Std.HashMap.emptyWithCapacity : CUDAAdamState)
  try
    discard <| session.use p
    discard <| session.use q
    let first ← TensorTransfer.toSomeBuffer (Tensor.scalar (1.0 : Float))
    let second ← TensorTransfer.toSomeBuffer (Tensor.scalar (3.0 : Float))
    let gradients := (Std.HashMap.emptyWithCapacity : LibTorch.Tape.SparseGradMap)
      |>.insert 0 first |>.insert 1 second
    withSparseGrads gradients fun gradients =>
      session.adamWStepCUDA config state 0.1 0.05 0.8 0.9 1e-8 gradients
    let before ← bits p
    let beforeState ← state.get
    let beforeFirst ← firstMomentBytes beforeState 0
    let beforeSecond ← firstMomentBytes beforeState 1
    session.resetTape
    discard <| session.use p
    discard <| session.use p
    let first ← TensorTransfer.toSomeBuffer (Tensor.scalar (1.0 : Float))
    let second ← TensorTransfer.toSomeBuffer (Tensor.scalar (3.0 : Float))
    let gradients := (Std.HashMap.emptyWithCapacity : LibTorch.Tape.SparseGradMap)
      |>.insert 0 first |>.insert 1 second
    withSparseGrads gradients fun gradients =>
      rejects "retied CUDA layout with separate histories"
        (session.adamWStepCUDA config state 0.1 0.05 0.8 0.9 1e-8 gradients)
    check "rejected retie preserves parameter" ((← bits p) == before)
    let afterState ← state.get
    check "rejected retie preserves state count and steps"
      (afterState.size == 2 && afterState.toList.all (fun entry => entry.2.t == 1))
    check "rejected retie preserves first moment"
      ((← firstMomentBytes afterState 0) == beforeFirst)
    check "rejected retie preserves second moment"
      ((← firstMomentBytes afterState 1) == beforeSecond)
  finally
    session.resetTape
    releaseAdam (← state.get)

private def cudaStep (session : Session Float)
    (config : IO.Ref (Option CUDAAdamConfig)) (state : IO.Ref CUDAAdamState) : IO Unit := do
  let gradients ← cudaGradients 1.0 5.0 3.0
  withSparseGrads gradients fun gradients =>
    session.adamWStepCUDA config state 0.1 0.05 0.8 0.9 1e-8 gradients

private def sameCUDAState (left right : CUDAAdamState) : IO Bool := do
  unless left.size == right.size do return false
  for (id, entry) in left.toList do
    let some other := right.get? id | return false
    unless entry.t == other.t do return false
    unless entry.firstPower.toBits == other.firstPower.toBits &&
        entry.secondPower.toBits == other.secondPower.toBits do return false
    unless (← LibTorch.Buffer.toBytesIO entry.m) ==
        (← LibTorch.Buffer.toBytesIO other.m) do return false
    unless (← LibTorch.Buffer.toBytesIO entry.v) ==
        (← LibTorch.Buffer.toBytesIO other.v) do return false
  pure true

/-- Tied CUDA AdamW follows an independent run with gradients summed before the optimizer. -/
def checkCUDAOptimizer : IO Unit := do
  let p ← scalar
  let q ← scalar
  let referenceP ← scalar
  let referenceQ ← scalar
  let session ← Session.create (α := Float) { device := cuda }
  let reference ← Session.create (α := Float) { device := cuda }
  let config ← IO.mkRef (none : Option CUDAAdamConfig)
  let state ← IO.mkRef (Std.HashMap.emptyWithCapacity : CUDAAdamState)
  let referenceConfig ← IO.mkRef (none : Option CUDAAdamConfig)
  let referenceState ← IO.mkRef (Std.HashMap.emptyWithCapacity : CUDAAdamState)
  try
    for step in [:6] do
      recordTriple session p q
      reference.resetTape
      discard <| reference.use referenceP
      discard <| reference.use referenceQ
      let a := (step + 1).toFloat
      let b := (step + 3).toFloat
      let c := (step + 5).toFloat
      let gradients ← cudaGradients a b c
      withSparseGrads gradients fun gradients =>
        session.adamWStepCUDA config state 0.1 0.05 0.8 0.9 1e-8 gradients
      let first ← TensorTransfer.toSomeBuffer (Tensor.scalar (a + c))
      let second ← TensorTransfer.toSomeBuffer (Tensor.scalar b)
      let referenceGradients := (Std.HashMap.emptyWithCapacity : LibTorch.Tape.SparseGradMap)
        |>.insert 0 first |>.insert 1 second
      withSparseGrads referenceGradients fun gradients =>
        reference.adamWStepCUDA referenceConfig referenceState
          0.1 0.05 0.8 0.9 1e-8 gradients
      check "CUDA tied AdamW trajectory" ((← bits p) == (← bits referenceP))
      check "CUDA independent AdamW trajectory" ((← bits q) == (← bits referenceQ))
      check "CUDA tied AdamW histories"
        (← sameCUDAState (← state.get) (← referenceState.get))
  finally
    session.resetTape
    reference.resetTape
    releaseAdam (← state.get)
    releaseAdam (← referenceState.get)

/--
Restore tied AdamW state byte-for-byte, reject retied destinations, then match the next update.
-/
def checkCUDACheckpoint : IO Unit := IO.FS.withTempDir fun directory => do
  let p ← scalar
  let q ← scalar
  let session ← Session.create (α := Float) { device := cuda }
  let config ← IO.mkRef (none : Option CUDAAdamConfig)
  let state ← IO.mkRef (Std.HashMap.emptyWithCapacity : CUDAAdamState)
  let restoredConfig ← IO.mkRef (none : Option CUDAAdamConfig)
  let restoredState ← IO.mkRef (Std.HashMap.emptyWithCapacity : CUDAAdamState)
  let resumed ← Session.create (α := Float) { device := cuda }
  try
    for _ in [:4] do
      recordTriple session p q
      cudaStep session config state
    let currentState ← state.get
    check "CUDA tied parameters have one history per storage"
      (currentState.size == 2 && currentState.contains 0 && currentState.contains 1 &&
        !currentState.contains 2 && currentState.toList.all (fun entry => entry.2.t == 4))
    let schema ← checkpointSchema (triple p q)
    let saved := directory / "adamw.bin"
    saveAdam saved schema config state
    let pValue ← bits p
    let qValue ← bits q
    let restoredP ← scalar (Float.ofBits pValue)
    let restoredQ ← scalar (Float.ofBits qValue)
    let restoredSchema ← checkpointSchema (triple restoredP restoredQ)
    loadAdam saved restoredSchema restoredConfig restoredState (dtype := .float64)
    check "restored moment bytes" (← sameCUDAState (← state.get) (← restoredState.get))
    let wrong : ParamList Float [[], [], []] :=
      .cons restoredP (.cons restoredQ (.cons restoredQ .nil))
    let wrongSchema ← checkpointSchema wrong
    rejects "retied checkpoint destination"
      (loadAdam saved wrongSchema restoredConfig restoredState (dtype := .float64))
    check "rejected restore preserves existing histories"
      (← sameCUDAState (← state.get) (← restoredState.get))
    recordTriple session p q
    recordTriple resumed restoredP restoredQ
    cudaStep session config state
    cudaStep resumed restoredConfig restoredState
    check "checkpoint next tied update" ((← bits p) == (← bits restoredP))
    check "checkpoint next independent update" ((← bits q) == (← bits restoredQ))
    check "checkpoint next moment bytes" (← sameCUDAState (← state.get) (← restoredState.get))
    let savedNext := directory / "next.bin"
    let restoredNext := directory / "restored-next.bin"
    saveAdam savedNext schema config state
    saveAdam restoredNext restoredSchema restoredConfig restoredState
    check "next checkpoint bytes"
      ((← IO.FS.readBinFile savedNext) == (← IO.FS.readBinFile restoredNext))
    let stochastic := directory / "stochastic.bin"
    let counter ← IO.mkRef 99
    saveAdam stochastic schema config state (some 7)
    loadAdam stochastic restoredSchema restoredConfig restoredState (some counter) .float64
    check "checkpoint restores the next random invocation" ((← counter.get) == 7)
    rejects "random checkpoint without a destination counter"
      (loadAdam stochastic restoredSchema restoredConfig restoredState (dtype := .float64))
    rejects "random checkpoint with a retied destination"
      (loadAdam stochastic wrongSchema restoredConfig restoredState (some counter) .float64)
    let truncated := directory / "truncated.bin"
    let bytes ← IO.FS.readBinFile stochastic
    IO.FS.writeBinFile truncated (bytes.extract 0 (bytes.size - 1))
    rejects "truncated random checkpoint"
      (loadAdam truncated restoredSchema restoredConfig restoredState
        (some counter) .float64)
    check "rejected checkpoint preserves random counter" ((← counter.get) == 7)
    check "rejected random checkpoint preserves moments"
      (← sameCUDAState (← state.get) (← restoredState.get))
  finally
    session.resetTape
    resumed.resetTape
    releaseAdam (← state.get)
    releaseAdam (← restoredState.get)

/-- Native division retains the eager successive-division VJP at binary32 extremes. -/
def checkCUDADivisionSchedule : IO Unit := do
  for (a, b) in [(1.0e30, 1.0e20), (1.0e-30, 1.0e-25)] do
    let session ← Session.create (α := Float32) (options := { device := cuda })
    let x ← session.input (Tensor.scalar a.toFloat32) (requiresGrad := true)
    let y ← session.input (Tensor.scalar b.toFloat32) (requiresGrad := true)
    let quotient ← session.apply "div" (.div (s := [])) (x, y, ())
    let gradients ← session.backwardDenseAll quotient (Tensor.scalar 1)
    let db ← Session.grad gradients y
    let expected := (-((a.toFloat32 / b.toFloat32) / b.toFloat32)).toFloat
    check "native denominator VJP stays finite and follows successive division"
      (expected.isFinite && expected < 0 && db.item.toFloat.toBits == expected.toBits)
    session.resetTape

/-- Require the native runtime, then run `checkCUDADivisionSchedule`, `checkSnapshots .cuda`,
`checkCUDARetie`, `checkCUDAOptimizer` and `checkCUDACheckpoint`. -/
def runCUDA : IO Unit := do
  LibTorch.Buffer.requireNativeRuntime
  checkCUDADivisionSchedule
  checkSnapshots .cuda
  checkCUDARetie
  checkCUDAOptimizer
  checkCUDACheckpoint
  LibTorch.Buffer.collectGarbage

end TorchLean.Test.ParameterAliases
