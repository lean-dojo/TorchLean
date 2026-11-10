/-
Copyright (c) 2026 TorchLean
Released under MIT license as described in the file LICENSE.
Authors: TorchLean Team
-/

module

public import TorchLean.NN.Session.Optimizer
public import TorchLean.NN.Session.Apply

/-!
When the native runtime is available, `run` checks on a CUDA session that the backward entry
points refuse foreign, unowned and stale output references without recording, cache changes or
native allocations. It also checks that a valid backward returns its gradient and that releasing
the cached CUDA value of a parameter clears it.
-/

@[expose] public section

open TorchLean.Tensor
open TorchLean.LibTorch
open TorchLean (Session)

namespace TorchLean.Test.SessionRefIdentity

/-- Observe recording and cache counters without evaluating or changing the session. -/
def recordingState (state : Session Float) : IO (Array Nat) := do
  pure #[(← state.tape.get).size,
    ← state.rngCounter.get, ← state.refGeneration.get,
    (← state.selectedBackends.get).size, (← state.paramsByLeaf.get).size]

/-- Rejected references fail before recording, cache changes, or native allocations. -/
def expectReferenceFailure {α : Type} (session : Session Float)
    (label expected : String) (action : IO α) : IO Unit := do
  let before ← recordingState session
  let memoryBefore ← LibTorch.Buffer.Memory.read
  let error? ← try
    let _ ← action
    pure none
  catch error =>
    pure (some error.toString)
  unless error? == some expected do
    throw <| IO.userError s!"{label}: expected {expected}, got {error?}"
  unless (← recordingState session) == before do
    throw <| IO.userError s!"{label}: rejection changed recording or cache state"
  let memoryAfter ← LibTorch.Buffer.Memory.read
  unless memoryAfter.allocCount == memoryBefore.allocCount &&
      memoryAfter.wrapperAllocCount == memoryBefore.wrapperAllocCount do
    throw <| IO.userError s!"{label}: rejection allocated native buffers"

/-- Reused output ids must not bypass ownership checks, even before a CUDA device gate. -/
def checkBackwardReferences (device : Backend.Device) : IO Unit := do
  let session ← Session.create (α := Float) { device }
  let foreignSession ← Session.create (α := Float) { device }
  try
    let input ← session.input (Tensor.scalar 2.0) (requiresGrad := true)
    let output ← session.apply "mul" .mul (input, input, ())
    let foreignInput ← foreignSession.input (Tensor.scalar 4.0) (requiresGrad := true)
    let foreignOutput ← foreignSession.apply "mul" .mul (foreignInput, foreignInput, ())
    let unowned : TensorRef Float Shape.scalar := { id := output.id }
    let actions : Array (String × (TensorRef Float Shape.scalar → IO Unit)) := #[
      ("backwardDenseAll", fun ref =>
        discard <| session.backwardDenseAll ref (Tensor.scalar 1.0)),
      ("backwardDenseAllCuda", fun ref =>
        discard <| session.backwardDenseAllCUDA ref (Tensor.scalar 1.0)),
      ("backwardScalarParamGradsCuda", fun ref =>
        discard <| session.paramGradsCUDA ref)]
    for (name, action) in actions do
      expectReferenceFailure session s!"{device.cliName} {name}: foreign output"
        "torch: tensor reference belongs to a different session" (action foreignOutput)
      expectReferenceFailure session s!"{device.cliName} {name}: unowned output"
        "torch: tensor reference has no session owner" (action unowned)

    session.resetTape
    let parameter ← Param.create (Tensor.scalar 3.0)
    let current ← session.use parameter
    let square ← session.apply "mul" .mul (current, current, ())
    unless square.id == output.id do
      throw <| IO.userError "backward reference check did not reuse the output id"
    for (name, action) in actions do
      expectReferenceFailure session s!"{device.cliName} {name}: stale output"
        ("torch: stale tensor reference from generation 0; current generation is 1 " ++
          "(resetTape invalidates recorded references)") (action output)
    let gradients ← session.backwardDenseAll square (Tensor.scalar 1.0)
    let gradient ← Session.grad gradients current
    unless gradient.item == 6.0 do
      throw <| IO.userError "valid square backward failed after rejected references"
    if device == .cuda then
      let gradients ← session.paramGradsCUDA square
      try
        let some gradient := gradients.get? current.id
          | throw <| IO.userError "valid sparse square backward omitted its parameter gradient"
        let values ← LibTorch.Buffer.toFloatArrayIO gradient.buf
        unless values.size == 1 && values.get! 0 == 6.0 do
          throw <| IO.userError "valid sparse square backward returned the wrong gradient"
      finally
        LibTorch.Tape.releaseSparseGrads gradients
  finally
    session.resetTape
    foreignSession.resetTape

/-- `SomeParam.releaseCachedCUDAValue` clears the cached CUDA value of a parameter, and a second
release does not fail. -/
def checkCUDACacheClear : IO Unit := do
  let value ← IO.mkRef (Tensor.scalar 1.0)
  let buffer ← LibTorch.Buffer.fullIO 1 1.0
  let cudaValue ← IO.mkRef (some { s := Shape.scalar, buf := buffer })
  let hostCurrent ← IO.mkRef true
  let param : Param Float Shape.scalar :=
    { value, cudaValue, hostSynced := hostCurrent, requiresGrad := true }
  SomeParam.releaseCachedCUDAValue param
  match ← cudaValue.get with
  | none => pure ()
  | some _ => throw <| IO.userError "CUDA cache release did not clear cudaValue"
  SomeParam.releaseCachedCUDAValue param

/-- Run `checkBackwardReferences .cuda` and `checkCUDACacheClear` when the native runtime is
available. -/
def run : IO Unit := do
  if (← LibTorch.Buffer.RuntimeStatus.query) == .nativeAvailable then
    checkBackwardReferences .cuda
    checkCUDACacheClear

end TorchLean.Test.SessionRefIdentity
